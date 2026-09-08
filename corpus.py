"""
Per-assignment corpus store.

Each assignment owns one SQLite-backed metadata store (shared across assignments)
and one on-disk FAISS index. A submission is scored against the prior corpus
(its own chunks excluded) and then appended.

LOCAL_ONLY mode of llm_providers.py keeps the LLM on-host; this module keeps
the corpus on-host too. No data leaves the disk it is written on.
"""

import os
import sqlite3
import logging
import threading
from contextlib import contextmanager
from typing import Iterable, List, Dict, Optional, Tuple

import numpy as np
import faiss

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS assignments (
    assignment_id TEXT PRIMARY KEY,
    name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_count INTEGER DEFAULT 0,
    embedding_model TEXT
);

CREATE TABLE IF NOT EXISTS submissions (
    submission_id TEXT PRIMARY KEY,
    assignment_id TEXT NOT NULL,
    student_id TEXT,
    filename TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_count INTEGER,
    avg_novelty REAL,
    FOREIGN KEY (assignment_id) REFERENCES assignments(assignment_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id TEXT NOT NULL,
    assignment_id TEXT NOT NULL,
    chunk_index INTEGER,
    text TEXT,
    prompt TEXT,
    novelty_score REAL,
    embedding_row INTEGER NOT NULL,
    FOREIGN KEY (submission_id) REFERENCES submissions(submission_id),
    FOREIGN KEY (assignment_id) REFERENCES assignments(assignment_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_assignment ON chunks(assignment_id);
CREATE INDEX IF NOT EXISTS idx_chunks_submission ON chunks(submission_id);
CREATE INDEX IF NOT EXISTS idx_submissions_assignment ON submissions(assignment_id);
"""


class EmbeddingModelMismatch(RuntimeError):
    """The corpus on disk was built with a different embedding model.

    Vectors from two models are not comparable, and usually not even the same
    dimension. Rebuild the partition (`rebuild_index`) with the current model,
    point CORPUS_DIR at a fresh directory, or switch EMBEDDING_MODEL back.
    """


class CorpusStore:
    """SQLite + FAISS corpus, one index per assignment.

    `embedding_model` is the name of the sentence-embedding model that
    produced the vectors. It is recorded per assignment the first time
    vectors are added and enforced afterwards, so a change of EMBEDDING_MODEL
    cannot silently mix incomparable vectors in one index.
    """

    def __init__(self, corpus_dir: str, embedding_dim: int,
                 embedding_model: Optional[str] = None):
        self.corpus_dir = corpus_dir
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        os.makedirs(corpus_dir, exist_ok=True)
        self.db_path = os.path.join(corpus_dir, 'corpus.db')
        self._index_cache: Dict[str, faiss.Index] = {}
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # Migration for corpora created before embedding_model was tracked.
            cols = {r['name'] for r in conn.execute("PRAGMA table_info(assignments)")}
            if 'embedding_model' not in cols:
                conn.execute("ALTER TABLE assignments ADD COLUMN embedding_model TEXT")

    def _check_embedding_model(self, conn, assignment_id: str):
        """Record the embedding model on first use; refuse a different one later."""
        if not self.embedding_model:
            return
        row = conn.execute(
            "SELECT embedding_model FROM assignments WHERE assignment_id = ?",
            (assignment_id,),
        ).fetchone()
        stored = row['embedding_model'] if row else None
        if stored is None:
            conn.execute(
                "UPDATE assignments SET embedding_model = ? WHERE assignment_id = ?",
                (self.embedding_model, assignment_id),
            )
        elif stored != self.embedding_model:
            raise EmbeddingModelMismatch(
                f"Assignment {assignment_id!r} was embedded with {stored!r} but the "
                f"current embedding model is {self.embedding_model!r}. Rebuild the "
                "index with the new model or use a different CORPUS_DIR."
            )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _safe_dirname(assignment_id: str) -> str:
        """Map an assignment_id to a filesystem-safe directory name.

        SQLite keeps the original id; only the on-disk FAISS folder is sanitised.
        Windows in particular rejects `:` and a few other chars in path segments.
        """
        return ''.join(c if c.isalnum() or c in '._-' else '_' for c in assignment_id)

    def _index_path(self, assignment_id: str) -> str:
        return os.path.join(self.corpus_dir, self._safe_dirname(assignment_id), 'index.faiss')

    def _load_index(self, assignment_id: str) -> faiss.Index:
        """Load (or create) the FAISS index for an assignment."""
        if assignment_id in self._index_cache:
            return self._index_cache[assignment_id]
        path = self._index_path(assignment_id)
        if os.path.exists(path):
            index = faiss.read_index(path)
            if index.d != self.embedding_dim:
                raise EmbeddingModelMismatch(
                    f"FAISS index for {assignment_id!r} has dimension {index.d} but the "
                    f"current embedding model produces {self.embedding_dim}. The corpus was "
                    "built with a different EMBEDDING_MODEL; rebuild it or use a fresh "
                    "CORPUS_DIR."
                )
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            index = faiss.IndexFlatL2(self.embedding_dim)
        self._index_cache[assignment_id] = index
        return index

    def _persist_index(self, assignment_id: str):
        index = self._index_cache.get(assignment_id)
        if index is None:
            return
        path = self._index_path(assignment_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(index, path)

    def ensure_assignment(self, assignment_id: str, name: Optional[str] = None):
        """Idempotently register an assignment."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO assignments(assignment_id, name) VALUES (?, ?)",
                (assignment_id, name),
            )
            if name is not None:
                conn.execute(
                    "UPDATE assignments SET name = ? WHERE assignment_id = ? AND (name IS NULL OR name = '')",
                    (name, assignment_id),
                )

    def assignment_exists(self, assignment_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM assignments WHERE assignment_id = ?",
                (assignment_id,),
            ).fetchone()
            return row is not None

    def score_against_corpus(
        self,
        assignment_id: str,
        query_embeddings: np.ndarray,
        exclude_submission_id: Optional[str] = None,
        k: int = 5,
        distance_scale: float = 2.0,
    ) -> List[float]:
        """
        Score each query embedding against the assignment corpus.

        Returns a novelty score per query (0-1, higher = more novel). When the
        corpus is empty (no prior submissions) every query scores 1.0.

        exclude_submission_id: when re-scoring a submission already in the
        corpus, drop its own chunks from the comparison.
        """
        with self._lock:
            index = self._load_index(assignment_id)
            if index.ntotal == 0:
                return [1.0] * len(query_embeddings)

            excluded_rows = set()
            if exclude_submission_id is not None:
                with self._connect() as conn:
                    rows = conn.execute(
                        "SELECT embedding_row FROM chunks "
                        "WHERE assignment_id = ? AND submission_id = ?",
                        (assignment_id, exclude_submission_id),
                    ).fetchall()
                    excluded_rows = {r['embedding_row'] for r in rows}

            # Search a few extra neighbours so we can drop excluded rows and
            # still have k real hits.
            search_k = min(index.ntotal, k + len(excluded_rows))
            queries = np.asarray(query_embeddings, dtype='float32')
            if queries.ndim == 1:
                queries = queries.reshape(1, -1)
            distances, indices = index.search(queries, search_k)

            scores = []
            for dist_row, idx_row in zip(distances, indices):
                kept = [d for d, i in zip(dist_row, idx_row) if i not in excluded_rows]
                kept = kept[:k]
                if not kept:
                    scores.append(1.0)
                    continue
                avg = float(np.mean(kept))
                novelty = 1.0 - float(np.exp(-avg / distance_scale))
                scores.append(min(1.0, max(0.0, novelty)))
            return scores

    def add_submission(
        self,
        assignment_id: str,
        submission_id: str,
        student_id: Optional[str],
        filename: Optional[str],
        chunks: List[Dict],
        embeddings: np.ndarray,
        novelty_scores: List[float],
    ) -> Dict:
        """
        Append a scored submission to the corpus. Caller must have already
        computed novelty_scores via score_against_corpus.

        chunks: list of dicts with at least 'text' and 'prompt' keys.
        embeddings: numpy array shape (len(chunks), embedding_dim).
        """
        if len(chunks) != len(embeddings) or len(chunks) != len(novelty_scores):
            raise ValueError("chunks, embeddings and novelty_scores length mismatch")
        embeddings = np.asarray(embeddings, dtype='float32')
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embeddings must have shape (n, {self.embedding_dim}); got {embeddings.shape}"
            )

        with self._lock:
            self.ensure_assignment(assignment_id)
            with self._connect() as conn:
                self._check_embedding_model(conn, assignment_id)
            index = self._load_index(assignment_id)
            start_row = index.ntotal
            index.add(np.asarray(embeddings, dtype='float32'))

            avg_novelty = float(np.mean(novelty_scores)) if novelty_scores else 0.0

            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO submissions"
                    "(submission_id, assignment_id, student_id, filename, chunk_count, avg_novelty)"
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (submission_id, assignment_id, student_id, filename, len(chunks), avg_novelty),
                )
                # If this submission_id existed before, wipe its old chunk rows.
                conn.execute(
                    "DELETE FROM chunks WHERE submission_id = ?",
                    (submission_id,),
                )
                for i, (chunk, score) in enumerate(zip(chunks, novelty_scores)):
                    conn.execute(
                        "INSERT INTO chunks"
                        "(submission_id, assignment_id, chunk_index, text, prompt, novelty_score, embedding_row)"
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            submission_id, assignment_id, i,
                            chunk.get('text', ''), chunk.get('prompt', ''),
                            float(score), start_row + i,
                        ),
                    )
                conn.execute(
                    "UPDATE assignments SET chunk_count = chunk_count + ? WHERE assignment_id = ?",
                    (len(chunks), assignment_id),
                )

            self._persist_index(assignment_id)
            return {
                'submission_id': submission_id,
                'assignment_id': assignment_id,
                'chunk_count': len(chunks),
                'avg_novelty': avg_novelty,
            }

    def get_assignment(self, assignment_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT assignment_id, name, created_at, chunk_count, embedding_model "
                "FROM assignments WHERE assignment_id = ?",
                (assignment_id,),
            ).fetchone()
            if row is None:
                return None
            sub_count = conn.execute(
                "SELECT COUNT(*) AS n FROM submissions WHERE assignment_id = ?",
                (assignment_id,),
            ).fetchone()['n']
            return {
                'assignment_id': row['assignment_id'],
                'name': row['name'],
                'created_at': row['created_at'],
                'chunk_count': row['chunk_count'],
                'submission_count': sub_count,
                'embedding_model': row['embedding_model'],
            }

    def list_submissions(self, assignment_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT submission_id, student_id, filename, submitted_at, "
                "chunk_count, avg_novelty FROM submissions "
                "WHERE assignment_id = ? ORDER BY submitted_at DESC",
                (assignment_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def submission_exists(self, submission_id: str) -> bool:
        """True if a submission with this id is already in the corpus."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM submissions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
            return row is not None

    def get_submission(self, submission_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            sub = conn.execute(
                "SELECT submission_id, assignment_id, student_id, filename, "
                "submitted_at, chunk_count, avg_novelty FROM submissions "
                "WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
            if sub is None:
                return None
            chunks = conn.execute(
                "SELECT chunk_index, text, prompt, novelty_score "
                "FROM chunks WHERE submission_id = ? ORDER BY chunk_index",
                (submission_id,),
            ).fetchall()
            result = dict(sub)
            result['chunks'] = [dict(c) for c in chunks]
            return result

    def rebuild_index(self, assignment_id: str, embedding_model,
                      model_name: Optional[str] = None) -> Dict:
        """
        Rebuild the FAISS index for an assignment from the chunks currently
        in SQLite, dropping any orphan rows left behind by re-uploads.

        Behaviour:
          - For each surviving chunk: re-embed its prompt (cohort mode) or
            its text (read-folder / citation-graph mode where prompt='').
          - Build a fresh IndexFlatL2 in chunk_id order.
          - Update embedding_row in SQLite so scoring still aligns.
          - Record `model_name` (default: this store's embedding_model) as the
            assignment's embedding model, so this is also the migration path
            after changing EMBEDDING_MODEL.
          - Persist the new index to disk.

        Use after heavy resubmission cycles or an embedding-model change.

        Returns counts: rows_before, rows_after, orphans_removed, embeddings_recomputed.
        """
        model_name = model_name or self.embedding_model
        with self._lock:
            cached = self._index_cache.pop(assignment_id, None)
            rows_before = cached.ntotal if cached is not None else (
                faiss.read_index(self._index_path(assignment_id)).ntotal
                if os.path.exists(self._index_path(assignment_id)) else 0
            )

            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT chunk_id, text, prompt FROM chunks "
                    "WHERE assignment_id = ? ORDER BY chunk_id",
                    (assignment_id,),
                ).fetchall()

                texts_to_embed = [
                    (r['prompt'] or '').strip() or r['text']
                    for r in rows
                ]
                if texts_to_embed:
                    embeddings = np.asarray(
                        embedding_model.encode(texts_to_embed, show_progress_bar=False),
                        dtype='float32',
                    )
                    dim = embeddings.shape[1]
                else:
                    embeddings = np.zeros((0, self.embedding_dim), dtype='float32')
                    dim = self.embedding_dim

                fresh = faiss.IndexFlatL2(dim)
                if len(embeddings) > 0:
                    fresh.add(embeddings)

                # Atomically rewrite chunk row assignments under the same lock
                for new_row, r in enumerate(rows):
                    conn.execute(
                        "UPDATE chunks SET embedding_row = ? WHERE chunk_id = ?",
                        (new_row, r['chunk_id']),
                    )
                # Sync chunk_count to match
                conn.execute(
                    "UPDATE assignments SET chunk_count = ? WHERE assignment_id = ?",
                    (len(rows), assignment_id),
                )
                if model_name:
                    conn.execute(
                        "UPDATE assignments SET embedding_model = ? WHERE assignment_id = ?",
                        (model_name, assignment_id),
                    )

            self.embedding_dim = dim
            self._index_cache[assignment_id] = fresh
            self._persist_index(assignment_id)

            return {
                'rows_before': rows_before,
                'rows_after': fresh.ntotal,
                'orphans_removed': max(0, rows_before - fresh.ntotal),
                'embeddings_recomputed': len(rows),
            }

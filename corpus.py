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
    chunk_count INTEGER DEFAULT 0
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


class CorpusStore:
    """SQLite + FAISS corpus, one index per assignment."""

    def __init__(self, corpus_dir: str, embedding_dim: int):
        self.corpus_dir = corpus_dir
        self.embedding_dim = embedding_dim
        os.makedirs(corpus_dir, exist_ok=True)
        self.db_path = os.path.join(corpus_dir, 'corpus.db')
        self._index_cache: Dict[str, faiss.Index] = {}
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _index_path(self, assignment_id: str) -> str:
        return os.path.join(self.corpus_dir, assignment_id, 'index.faiss')

    def _load_index(self, assignment_id: str) -> faiss.Index:
        """Load (or create) the FAISS index for an assignment."""
        if assignment_id in self._index_cache:
            return self._index_cache[assignment_id]
        path = self._index_path(assignment_id)
        if os.path.exists(path):
            index = faiss.read_index(path)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            index = faiss.IndexFlatL2(self.embedding_dim)
        self._index_cache[assignment_id] = index
        return index

    def _persist_index(self, assignment_id: str):
        index = self._index_cache.get(assignment_id)
        if index is None:
            return
        faiss.write_index(index, self._index_path(assignment_id))

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

        with self._lock:
            self.ensure_assignment(assignment_id)
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
                "SELECT assignment_id, name, created_at, chunk_count "
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

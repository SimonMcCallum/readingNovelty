"""
PhD-style personal novelty: ingest a folder of "papers I have read" into a
named corpus, then score a new PDF against it.

Typical use:

  # First time: bulk-ingest everything in the folder
  python read_folder_cli.py \\
      --read-folder ~/papers/read --corpus-id phd_alice --reingest

  # Routine use: drop a new PDF in /inbox and score it
  python read_folder_cli.py \\
      --read-folder ~/papers/read --corpus-id phd_alice \\
      --new ~/papers/inbox/just_arrived.pdf --alpha 0.7 \\
      --output annotated_just_arrived.pdf

Privacy: this CLI runs entirely on the host. LOCAL_ONLY=1 (the default)
keeps the LLM call local. If you set LOCAL_ONLY=0 and configure a cloud
provider, the new PDF's chunks plus their neighbours WILL be sent to that
provider for the predictive-novelty signal (alpha < 1.0). The read corpus
itself never leaves disk regardless.
"""

import argparse
import json
import logging
import os
import sys
from hashlib import sha1
from typing import Dict, List, Optional, Tuple

import numpy as np

from corpus import CorpusStore
from llm_providers import _local_only_enabled
from novelty_detector import NoveltyDetector
from pdf_processor import PDFProcessor


logger = logging.getLogger(__name__)


def _submission_id_for_path(path: str) -> str:
    """Stable id for a read PDF based on its absolute path."""
    abs_path = os.path.abspath(path)
    digest = sha1(abs_path.encode('utf-8')).hexdigest()[:12]
    return f"read:{os.path.basename(path)}:{digest}"


def _iter_pdfs(folder: str):
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith('.pdf'):
                yield os.path.join(root, name)


def ingest_folder(
    folder: str,
    corpus_id: str,
    corpus: CorpusStore,
    detector: NoveltyDetector,
    processor: PDFProcessor,
    force: bool = False,
) -> Dict:
    """Bulk-ingest every PDF in folder. Skips files already in the corpus
    unless force=True."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"--read-folder does not exist: {folder}")

    ingested, skipped, failed = 0, 0, 0
    for pdf_path in _iter_pdfs(folder):
        submission_id = _submission_id_for_path(pdf_path)
        if not force and corpus.submission_exists(submission_id):
            skipped += 1
            continue
        try:
            chunks = processor.extract_and_chunk_text(pdf_path)
            if not chunks:
                logger.warning("No text extracted from %s; skipping", pdf_path)
                failed += 1
                continue
            embeddings, prompts = detector.embed_and_prompt_chunks(chunks)
            for chunk, prompt in zip(chunks, prompts):
                chunk['prompt'] = prompt
            # Read-folder ingestion is unidirectional: we don't score each PDF
            # against its predecessors. The corpus is the *reference*, not a
            # ranked archive. Use a placeholder score of 1.0 for posterity.
            placeholder_scores = [1.0] * len(chunks)
            corpus.add_submission(
                assignment_id=corpus_id,
                submission_id=submission_id,
                student_id=None,
                filename=os.path.basename(pdf_path),
                chunks=chunks,
                embeddings=embeddings,
                novelty_scores=placeholder_scores,
            )
            ingested += 1
            logger.info("Ingested %s (%d chunks)", pdf_path, len(chunks))
        except Exception as e:
            logger.error("Failed to ingest %s: %s", pdf_path, e)
            failed += 1

    return {'ingested': ingested, 'skipped': skipped, 'failed': failed}


def score_new_pdf(
    new_pdf: str,
    corpus_id: str,
    corpus: CorpusStore,
    detector: NoveltyDetector,
    processor: PDFProcessor,
    alpha: float,
    annotated_output: Optional[str] = None,
    predict_workers: int = 4,
) -> Dict:
    """Score a new PDF against the read corpus. Returns a report dict."""
    if not os.path.isfile(new_pdf):
        raise FileNotFoundError(f"--new does not exist: {new_pdf}")

    chunks = processor.extract_and_chunk_text(new_pdf)
    if not chunks:
        raise ValueError(f"No text extracted from {new_pdf}")

    embeddings, prompts = detector.embed_and_prompt_chunks(chunks)
    for chunk, prompt in zip(chunks, prompts):
        chunk['prompt'] = prompt

    corpus_scores = corpus.score_against_corpus(corpus_id, embeddings)

    llm_scores: List[float] = []
    if alpha < 1.0:
        if detector.active_provider.name == 'fallback':
            raise RuntimeError(
                "alpha < 1.0 requires an LLM provider but the active provider "
                "is 'fallback'. Configure OLLAMA_HOST (and start Ollama) or "
                "set LOCAL_ONLY=0 with a cloud provider key."
            )
        llm_scores = detector.analyze_llm_novelty(chunks, max_workers=predict_workers)
        blended = NoveltyDetector.combine_novelty_scores(corpus_scores, llm_scores, alpha)
    else:
        blended = list(corpus_scores)

    if annotated_output:
        processor.create_annotated_pdf(new_pdf, chunks, blended, annotated_output)

    info = corpus.get_assignment(corpus_id) or {}
    return {
        'new_pdf': new_pdf,
        'corpus_id': corpus_id,
        'corpus_size_chunks': info.get('chunk_count', 0),
        'corpus_size_submissions': info.get('submission_count', 0),
        'chunks_scored': len(chunks),
        'alpha': alpha,
        'avg_corpus_novelty': float(np.mean(corpus_scores)) if corpus_scores else 0.0,
        'avg_llm_novelty': float(np.mean(llm_scores)) if llm_scores else None,
        'avg_blended_novelty': float(np.mean(blended)) if blended else 0.0,
        'per_chunk': [
            {
                'chunk_index': i,
                'text_preview': c['text'][:100],
                'corpus': corpus_scores[i],
                'llm': llm_scores[i] if llm_scores else None,
                'blended': blended[i],
            }
            for i, c in enumerate(chunks)
        ],
        'annotated_pdf': annotated_output,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--read-folder', required=True,
                        help='Directory of PDFs to use as the personal corpus.')
    parser.add_argument('--corpus-id', required=True,
                        help='Stable name for this personal corpus partition (e.g. phd_alice).')
    parser.add_argument('--new', help='Path to the new PDF to score. Omit to ingest only.')
    parser.add_argument('--corpus-dir', default=os.getenv('CORPUS_DIR', 'corpus'),
                        help='Where the SQLite + FAISS indices live (default: ./corpus).')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Blend: 1.0 = corpus only, 0.0 = LLM predictive only. Default 1.0.')
    parser.add_argument('--output',
                        help='Path for the annotated PDF (only used with --new).')
    parser.add_argument('--reingest', action='store_true',
                        help='Re-ingest every PDF in --read-folder even if already present.')
    parser.add_argument('--predict-workers', type=int, default=4,
                        help='Concurrent predict_chunk calls when alpha < 1.0. '
                             'Default 4. Use 1 to debug, 2-3 for Gemini free tier, '
                             'higher only if Ollama is configured with OLLAMA_NUM_PARALLEL.')
    parser.add_argument('--report-json',
                        help='Write the per-chunk report as JSON to this path.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s %(name)s: %(message)s',
    )

    if not (0.0 <= args.alpha <= 1.0):
        parser.error("--alpha must be in [0.0, 1.0]")

    if not _local_only_enabled() and args.new:
        logger.warning(
            "LOCAL_ONLY is off. The new PDF's chunks will be sent to the active "
            "cloud provider for the predictive-novelty step (alpha < 1.0)."
        )

    detector = NoveltyDetector()
    processor = PDFProcessor()
    corpus = CorpusStore(args.corpus_dir, embedding_dim=detector.embedding_dim,
                         embedding_model=detector.embedding_model_name)
    corpus.ensure_assignment(args.corpus_id, name=f"Read folder: {args.read_folder}")

    ingest_report = ingest_folder(
        args.read_folder, args.corpus_id, corpus, detector, processor,
        force=args.reingest,
    )
    logger.info(
        "Ingestion: %d new, %d skipped (already ingested), %d failed",
        ingest_report['ingested'], ingest_report['skipped'], ingest_report['failed'],
    )

    if not args.new:
        return 0

    annotated_output = args.output or os.path.join(
        os.path.dirname(args.new) or '.',
        f"annotated_{os.path.basename(args.new)}",
    )
    report = score_new_pdf(
        args.new, args.corpus_id, corpus, detector, processor,
        alpha=args.alpha, annotated_output=annotated_output,
        predict_workers=args.predict_workers,
    )

    logger.info(
        "Scored %s against %d chunks from %d papers (alpha=%.2f)",
        args.new, report['corpus_size_chunks'], report['corpus_size_submissions'],
        args.alpha,
    )
    logger.info(
        "  avg corpus novelty:  %.3f", report['avg_corpus_novelty']
    )
    if report['avg_llm_novelty'] is not None:
        logger.info("  avg LLM novelty:     %.3f", report['avg_llm_novelty'])
    logger.info(
        "  avg blended novelty: %.3f", report['avg_blended_novelty']
    )
    logger.info("Annotated PDF: %s", annotated_output)

    if args.report_json:
        with open(args.report_json, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2)
        logger.info("JSON report: %s", args.report_json)

    return 0


if __name__ == '__main__':
    sys.exit(main())

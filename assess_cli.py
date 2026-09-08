"""
Run novelty assessment on a Canvas assignment from the lecturer's machine.

Pulls every submission to a given (course_id, assignment_id), runs novelty
against the per-assignment corpus, and posts the annotated PDF plus a
novelty summary back as a submission comment on each one.

Designed to run on the same box as Ollama + the corpus store, so the PDF
content never crosses an external boundary except returning to Canvas (where
it originated).

Usage:
  python assess_cli.py --course 12345 --assignment 67890 --internal-id cs101-a1-2026
  # extra knobs:
  #   --base-url https://canvas.example.edu     (default: env CANVAS_BASE_URL)
  #   --token   ...                             (default: env CANVAS_TOKEN)
  #   --skip-already-commented                  (default on; pass --rescore to override)
  #   --dry-run                                 (process + score, but do not post back)
"""

import argparse
import logging
import os
import sys
import tempfile
from typing import Dict, List, Optional

import numpy as np

from canvas_client import CanvasClient, CanvasError
from corpus import CorpusStore
from llm_providers import _local_only_enabled
from novelty_detector import NoveltyDetector
from pdf_processor import PDFProcessor


logger = logging.getLogger(__name__)


NOVELTY_BAND_THRESHOLDS = [
    (0.7, 'high'),
    (0.4, 'medium'),
    (0.2, 'low'),
    (0.0, 'very low'),
]

ASSESSMENT_TAG = '[novelty-bot]'   # marker we leave in posted comments so we
                                   # can detect re-runs and skip by default.


def band_for(score: float) -> str:
    for threshold, label in NOVELTY_BAND_THRESHOLDS:
        if score >= threshold:
            return label
    return 'very low'


def format_summary(score_summary: Dict, corpus_priors: int) -> str:
    avg = score_summary['avg_novelty']
    band = band_for(avg)
    distribution = score_summary['distribution']
    lines = [
        f"{ASSESSMENT_TAG} Novelty assessment vs. cohort",
        f"Average novelty: {avg:.2f} ({band})",
        f"Chunks scored: {score_summary['chunk_count']}",
        f"Corpus size at scoring time: {corpus_priors} chunks from prior submissions",
        "",
        "Per-band counts:",
        f"  high (>= 0.7):     {distribution['high']}",
        f"  medium (0.4-0.7):  {distribution['medium']}",
        f"  low (0.2-0.4):     {distribution['low']}",
        f"  very low (< 0.2):  {distribution['very_low']}",
        "",
        "Higher score = the chunk is dissimilar from the rest of the cohort's submissions.",
        "Lower score = the chunk closely mirrors content already submitted by peers.",
    ]
    return '\n'.join(lines)


def score_distribution(scores: List[float]) -> Dict[str, int]:
    return {
        'high': sum(1 for s in scores if s >= 0.7),
        'medium': sum(1 for s in scores if 0.4 <= s < 0.7),
        'low': sum(1 for s in scores if 0.2 <= s < 0.4),
        'very_low': sum(1 for s in scores if s < 0.2),
    }


def already_assessed(submission: Dict) -> bool:
    """True if a prior run posted an assessment comment on this submission."""
    for comment in submission.get('submission_comments') or []:
        if ASSESSMENT_TAG in (comment.get('comment') or ''):
            return True
    return False


def process_submission(
    submission: Dict,
    course_id: str,
    canvas_assignment_id: str,
    internal_assignment_id: str,
    canvas: CanvasClient,
    corpus: CorpusStore,
    detector: NoveltyDetector,
    processor: PDFProcessor,
    work_dir: str,
    dry_run: bool,
) -> Optional[Dict]:
    user_id = str(submission.get('user_id'))
    submission_id = f"{internal_assignment_id}:{submission.get('id')}"

    attachment = canvas.pdf_attachment(submission)
    if attachment is None:
        logger.info("Submission %s has no PDF attachment; skipping", submission_id)
        return None

    pdf_path = os.path.join(work_dir, f"{submission_id}.pdf")
    canvas.download_attachment(attachment, pdf_path)

    chunks = processor.extract_and_chunk_text(pdf_path)
    if not chunks:
        logger.warning("Submission %s extracted no text; skipping", submission_id)
        return None

    embeddings, prompts = detector.embed_and_prompt_chunks(chunks)
    novelty_scores = corpus.score_against_corpus(
        internal_assignment_id, embeddings, exclude_submission_id=submission_id
    )
    for chunk, prompt in zip(chunks, prompts):
        chunk['prompt'] = prompt

    pre_ingest_priors = corpus.get_assignment(internal_assignment_id)
    corpus_priors = pre_ingest_priors['chunk_count'] if pre_ingest_priors else 0

    record = corpus.add_submission(
        assignment_id=internal_assignment_id,
        submission_id=submission_id,
        student_id=user_id,
        filename=os.path.basename(pdf_path),
        chunks=chunks,
        embeddings=embeddings,
        novelty_scores=novelty_scores,
    )

    annotated_path = os.path.join(work_dir, f"annotated_{submission_id}.pdf")
    processor.create_annotated_pdf(pdf_path, chunks, novelty_scores, annotated_path)

    summary_text = format_summary(
        {
            'avg_novelty': record['avg_novelty'],
            'chunk_count': record['chunk_count'],
            'distribution': score_distribution(novelty_scores),
        },
        corpus_priors,
    )

    if dry_run:
        logger.info("DRY RUN — would post on user %s:\n%s", user_id, summary_text)
        return record

    file_id = canvas.upload_comment_file(
        course_id, canvas_assignment_id, user_id, annotated_path
    )
    canvas.post_submission_comment(
        course_id, canvas_assignment_id, user_id, summary_text, file_ids=[file_id]
    )
    logger.info("Posted assessment for user %s (avg novelty %.2f)",
                user_id, record['avg_novelty'])
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course', required=True, help='Canvas course id')
    parser.add_argument('--assignment', required=True, help='Canvas assignment id')
    parser.add_argument('--internal-id', required=True,
                        help='Stable internal assignment id used for the corpus partition '
                             '(e.g. cs101-a1-2026)')
    parser.add_argument('--base-url', default=os.getenv('CANVAS_BASE_URL'),
                        help='Canvas base URL (or env CANVAS_BASE_URL)')
    parser.add_argument('--token', default=os.getenv('CANVAS_TOKEN'),
                        help='Lecturer access token (or env CANVAS_TOKEN)')
    parser.add_argument('--corpus-dir', default=os.getenv('CORPUS_DIR', 'corpus'))
    parser.add_argument('--rescore', action='store_true',
                        help='Re-process submissions even if a [novelty-bot] comment exists.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Score locally but do not post comments back to Canvas.')
    parser.add_argument('--preflight', action='store_true',
                        help='Validate token + show what would be processed. No scoring, no posting.')
    parser.add_argument('--limit', type=int,
                        help='Stop after this many submissions. Useful for a first real-run test.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s %(name)s: %(message)s',
    )

    if not args.base_url or not args.token:
        parser.error("--base-url and --token (or CANVAS_BASE_URL / CANVAS_TOKEN) are required")

    canvas = CanvasClient(args.base_url, args.token)

    # Preflight: validate token + show what would be processed, then exit.
    if args.preflight:
        return _preflight(canvas, args)

    if not _local_only_enabled():
        logger.warning(
            "LOCAL_ONLY is off. This CLI processes copyrighted student submissions; "
            "re-enable LOCAL_ONLY=1 unless you know that material is licensed for cloud use."
        )

    detector = NoveltyDetector()
    if detector.active_provider.name == 'fallback':
        parser.error(
            "No LLM provider available (active=fallback). Configure OLLAMA_HOST and ensure Ollama is running."
        )

    processor = PDFProcessor()
    corpus = CorpusStore(args.corpus_dir, embedding_dim=detector.embedding_dim,
                         embedding_model=detector.embedding_model_name)
    corpus.ensure_assignment(args.internal_id, name=f"Canvas {args.course}/{args.assignment}")

    try:
        submissions = canvas.list_submissions(args.course, args.assignment)
    except CanvasError as e:
        logger.error("Could not list submissions: %s", e)
        return 1

    with tempfile.TemporaryDirectory(prefix='novelty_') as work_dir:
        processed = 0
        skipped = 0
        for submission in submissions:
            if args.limit is not None and processed >= args.limit:
                logger.info("--limit %d reached; stopping.", args.limit)
                break
            if not submission.get('attachments'):
                continue
            if not args.rescore and already_assessed(submission):
                logger.info("Submission %s already assessed; skipping", submission.get('id'))
                skipped += 1
                continue
            try:
                if process_submission(
                    submission, args.course, args.assignment, args.internal_id,
                    canvas, corpus, detector, processor, work_dir, args.dry_run,
                ):
                    processed += 1
            except Exception as e:
                logger.error("Failed on submission %s: %s",
                             submission.get('id'), e, exc_info=args.verbose)

    logger.info("Done. Processed: %d, skipped: %d", processed, skipped)
    return 0


def _preflight(canvas: CanvasClient, args) -> int:
    """Validate the Canvas token and survey what a real run would process.

    Returns 0 if everything looks good, non-zero if a problem was found.
    """
    print(f"Canvas base URL: {canvas.base_url}")

    try:
        me = canvas.get_self()
    except CanvasError as e:
        print(f"FAIL: token check failed -- {e}")
        return 2
    print(f"Token belongs to: {me.get('name')} (id={me.get('id')}, email={me.get('primary_email')})")

    try:
        assignment = canvas.get_assignment(args.course, args.assignment)
    except CanvasError as e:
        print(f"FAIL: cannot fetch assignment {args.course}/{args.assignment} -- {e}")
        return 3
    print(f"Assignment: '{assignment.get('name')}' (id={assignment.get('id')})")
    if assignment.get('due_at'):
        print(f"  Due: {assignment.get('due_at')}")
    print(f"  Submission types: {assignment.get('submission_types')}")

    try:
        submissions = canvas.list_submissions(args.course, args.assignment)
    except CanvasError as e:
        print(f"FAIL: cannot list submissions -- {e}")
        return 4

    has_attach, pdf_attach, already_done, no_attach = 0, 0, 0, 0
    for sub in submissions:
        if not sub.get('attachments'):
            no_attach += 1
            continue
        has_attach += 1
        if CanvasClient.pdf_attachment(sub):
            pdf_attach += 1
        if already_assessed(sub):
            already_done += 1

    print(f"\nSubmissions found: {len(submissions)}")
    print(f"  with attachments: {has_attach}")
    print(f"  with PDF attachment: {pdf_attach}")
    print(f"  already assessed (will skip unless --rescore): {already_done}")
    print(f"  no attachment (will skip): {no_attach}")
    will_process = pdf_attach - already_done
    if args.rescore:
        will_process = pdf_attach
    if args.limit is not None:
        will_process = min(will_process, args.limit)
    print(f"\nA real run with these flags would process: {will_process} submissions")
    return 0


if __name__ == '__main__':
    sys.exit(main())

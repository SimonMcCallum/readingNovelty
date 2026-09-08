"""
Citation-graph novelty: how much does a paper add beyond what its own
references already said?

Pipeline:
  1. Take a target paper as PDF + DOI (or arXiv id).
  2. Fetch each cited work's abstract via Semantic Scholar.
  3. Optionally filter to references published before --target-year.
  4. Build a corpus partition from the reference abstracts.
  5. Score the paper's chunks against that corpus (text-level embedding;
     no LLM prompt step, because we want raw semantic distance to the
     prior-art summaries, not paraphrase-of-paraphrase distance).
  6. Annotate the PDF + write a JSON report.

This deliberately does NOT use the LLM-predictive novelty path: that
signal measures "what would an LLM write here?" which is a different
question from "what part of this paper isn't already in the references?"

Privacy: the abstract fetch is a public, anonymous API call. The paper
PDF stays on disk. Set LOCAL_ONLY=1 (the default) to keep the embedding
step local too.
"""

import argparse
import json
import logging
import os
import sys
from hashlib import sha1
from typing import Dict, List, Optional

import numpy as np

from citation_extractor import extract_dois
from citation_fetcher import SemanticScholarClient, CitationFetchError
from corpus import CorpusStore
from novelty_detector import NoveltyDetector
from pdf_processor import PDFProcessor


logger = logging.getLogger(__name__)


def _corpus_id_for(paper_doi: str) -> str:
    digest = sha1(paper_doi.lower().encode('utf-8')).hexdigest()[:12]
    return f"citation_{digest}"


def build_citation_corpus(
    paper_doi: str,
    fetcher: SemanticScholarClient,
    corpus: CorpusStore,
    processor: PDFProcessor,
    embedding_model,
    target_year: Optional[int] = None,
    force: bool = False,
) -> Dict:
    """
    Build a corpus partition from the references of paper_doi.

    Each reference's abstract is added as a separate submission with one
    chunk (or split if the abstract is long). Re-runs skip already-present
    references unless force=True.
    """
    corpus_id = _corpus_id_for(paper_doi)
    corpus.ensure_assignment(corpus_id, name=f"Citations of {paper_doi}")

    refs = fetcher.fetch_references(paper_doi, year_max=target_year)

    ingested, skipped_no_abstract, skipped_exists = 0, 0, 0
    for ref in refs:
        paper_id = ref.get('paperId') or ref.get('externalIds', {}).get('DOI')
        if not paper_id:
            continue
        submission_id = f"ref:{paper_id}"
        if not force and corpus.submission_exists(submission_id):
            skipped_exists += 1
            continue
        abstract = (ref.get('abstract') or '').strip()
        if len(abstract) < 50:
            skipped_no_abstract += 1
            continue

        chunks = processor.chunk_text(abstract)
        if not chunks:
            skipped_no_abstract += 1
            continue
        texts = [c['text'] for c in chunks]
        embeddings = np.asarray(embedding_model.encode(texts, show_progress_bar=False))
        for c in chunks:
            c['prompt'] = ''  # not used in citation-graph mode

        corpus.add_submission(
            assignment_id=corpus_id,
            submission_id=submission_id,
            student_id=None,
            filename=ref.get('title') or paper_id,
            chunks=chunks,
            embeddings=embeddings,
            novelty_scores=[1.0] * len(chunks),
        )
        ingested += 1

    info = corpus.get_assignment(corpus_id) or {}
    return {
        'corpus_id': corpus_id,
        'references_total': len(refs),
        'references_with_abstract_ingested': ingested,
        'references_skipped_no_abstract': skipped_no_abstract,
        'references_skipped_already_present': skipped_exists,
        'corpus_size_chunks': info.get('chunk_count', 0),
        'corpus_size_papers': info.get('submission_count', 0),
    }


def score_paper_against_citations(
    pdf_path: str,
    corpus_id: str,
    corpus: CorpusStore,
    processor: PDFProcessor,
    embedding_model,
    annotated_output: Optional[str] = None,
) -> Dict:
    """Score the target paper's chunks against its cited-papers corpus."""
    chunks = processor.extract_and_chunk_text(pdf_path)
    if not chunks:
        raise ValueError(f"No text extracted from {pdf_path}")

    texts = [c['text'] for c in chunks]
    embeddings = np.asarray(embedding_model.encode(texts, show_progress_bar=False))
    scores = corpus.score_against_corpus(corpus_id, embeddings)

    if annotated_output:
        processor.create_annotated_pdf(pdf_path, chunks, scores, annotated_output)

    return {
        'pdf_path': pdf_path,
        'corpus_id': corpus_id,
        'chunks_scored': len(chunks),
        'avg_novelty': float(np.mean(scores)) if scores else 0.0,
        'per_chunk': [
            {
                'chunk_index': i,
                'text_preview': c['text'][:100],
                'novelty': scores[i],
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
    parser.add_argument('--pdf', required=True, help='Path to the target paper PDF.')
    parser.add_argument(
        '--paper-doi',
        help="Target paper's DOI. If omitted, the first DOI found inside the PDF text is used.",
    )
    parser.add_argument(
        '--target-year', type=int,
        help='Only include references published strictly before this year.',
    )
    parser.add_argument(
        '--corpus-dir', default=os.getenv('CORPUS_DIR', 'corpus'),
        help='Where SQLite + FAISS indices live.',
    )
    parser.add_argument(
        '--cache-dir',
        default=os.getenv('S2_CACHE_DIR', os.path.join(os.getenv('CORPUS_DIR', 'corpus'), 's2_cache')),
        help='Directory for Semantic Scholar response cache.',
    )
    parser.add_argument(
        '--rate-limit', type=float, default=1.0,
        help='Seconds between Semantic Scholar requests (lower if you have an API key).',
    )
    parser.add_argument('--output', help='Path for the annotated PDF.')
    parser.add_argument('--report-json', help='Path for the JSON per-chunk report.')
    parser.add_argument(
        '--rebuild-corpus', action='store_true',
        help='Re-fetch references even if they are already in the corpus partition.',
    )
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s %(name)s: %(message)s',
    )

    if not os.path.isfile(args.pdf):
        parser.error(f"--pdf does not exist: {args.pdf}")

    processor = PDFProcessor()

    paper_doi = args.paper_doi
    if not paper_doi:
        text = processor.extract_text_from_pdf(args.pdf)
        dois = extract_dois(text)
        if not dois:
            parser.error(
                "No --paper-doi provided and no DOI found in the PDF text. "
                "Pass --paper-doi explicitly."
            )
        paper_doi = dois[0]
        logger.info("Using DOI from PDF text: %s", paper_doi)

    # NoveltyDetector pulls in the embedding model. We do not use its
    # prompt-generation path in citation-graph mode (LLM provider not needed
    # for this signal), so the active provider doesn't matter here.
    detector = NoveltyDetector()
    corpus = CorpusStore(args.corpus_dir, embedding_dim=detector.embedding_dim,
                         embedding_model=detector.embedding_model_name)
    fetcher = SemanticScholarClient(
        cache_dir=args.cache_dir, rate_limit_seconds=args.rate_limit
    )

    try:
        build_report = build_citation_corpus(
            paper_doi, fetcher, corpus, processor, detector.embedding_model,
            target_year=args.target_year, force=args.rebuild_corpus,
        )
    except CitationFetchError as e:
        logger.error("Could not fetch references: %s", e)
        return 1

    logger.info(
        "Citation corpus: %d references found, %d ingested (with abstracts), "
        "%d skipped (no abstract), %d skipped (already present).",
        build_report['references_total'],
        build_report['references_with_abstract_ingested'],
        build_report['references_skipped_no_abstract'],
        build_report['references_skipped_already_present'],
    )

    if build_report['corpus_size_chunks'] == 0:
        logger.error(
            "Reference corpus is empty (no abstracts available). "
            "Citation-graph novelty cannot be computed for this paper."
        )
        return 2

    annotated_output = args.output or os.path.join(
        os.path.dirname(args.pdf) or '.',
        f"citation_annotated_{os.path.basename(args.pdf)}",
    )
    score_report = score_paper_against_citations(
        args.pdf, build_report['corpus_id'], corpus, processor,
        detector.embedding_model, annotated_output=annotated_output,
    )

    logger.info(
        "Scored %d chunks against %d reference chunks. Avg citation-graph novelty: %.3f",
        score_report['chunks_scored'], build_report['corpus_size_chunks'],
        score_report['avg_novelty'],
    )
    logger.info("Annotated PDF: %s", annotated_output)

    report = {**build_report, 'scoring': score_report, 'paper_doi': paper_doi}
    if args.report_json:
        with open(args.report_json, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2)
        logger.info("JSON report: %s", args.report_json)

    return 0


if __name__ == '__main__':
    sys.exit(main())

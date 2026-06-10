"""Live smoke test: build a real citation corpus from BERT's references and
score two synthesized PDFs against it.

Hits the public Semantic Scholar API. ~30 seconds total."""

import os
import shutil
import tempfile
import time

import fitz

from citation_fetcher import SemanticScholarClient
from citation_novelty_cli import build_citation_corpus, score_paper_against_citations
from corpus import CorpusStore
from novelty_detector import NoveltyDetector
from pdf_processor import PDFProcessor


def build_pdf(path, paragraphs):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 60
    for para in paragraphs:
        s = para
        while s:
            page.insert_text((50, y), s[:80], fontsize=11, fontname="helv")
            s = s[80:]
            y += 16
            if y > 780:
                page = doc.new_page(width=595, height=842)
                y = 60
        y += 14
    doc.save(path)
    doc.close()


def main():
    tmp = tempfile.mkdtemp(prefix="s2_smoke_")
    try:
        time.sleep(3)
        fetcher = SemanticScholarClient(
            cache_dir=os.path.join(tmp, "cache"),
            rate_limit_seconds=1.5,
        )
        detector = NoveltyDetector()
        processor = PDFProcessor()
        corpus = CorpusStore(os.path.join(tmp, "corpus"),
                             embedding_dim=detector.embedding_dim)

        bert_doi = "10.18653/v1/N19-1423"
        print(f"[1/4] Building citation corpus for BERT ({bert_doi}) ...")
        build_report = build_citation_corpus(
            bert_doi, fetcher, corpus, processor,
            detector.embedding_model,
        )
        print("  references_total          :", build_report["references_total"])
        print("  ingested (with abstracts) :", build_report["references_with_abstract_ingested"])
        print("  skipped (no abstract)     :", build_report["references_skipped_no_abstract"])
        print("  corpus chunks             :", build_report["corpus_size_chunks"])
        print("  corpus papers             :", build_report["corpus_size_papers"])

        # Target 1: paraphrases part of BERT's own abstract -> low novelty expected
        similar_text = (
            "We introduce BERT, a new language representation model. BERT stands for "
            "Bidirectional Encoder Representations from Transformers. Unlike recent "
            "language representation models such as ELMo and GPT, BERT is designed to "
            "pre-train deep bidirectional representations from unlabeled text by jointly "
            "conditioning on both left and right context in all layers. As a result, the "
            "pre-trained BERT model can be fine-tuned with just one additional output layer "
            "to create state-of-the-art models for a wide range of tasks. "
        ) * 3
        similar_pdf = os.path.join(tmp, "bert_like.pdf")
        build_pdf(similar_pdf, [similar_text])

        # Target 2: about something unrelated -> high novelty expected
        novel_text = (
            "Sourdough fermentation relies on a symbiotic culture of wild yeast and "
            "lactobacilli sustained by regular feedings of flour and water. The microbial "
            "ecology of an active starter shifts in response to ambient temperature, "
            "hydration ratio and feeding cadence. "
        ) * 5
        novel_pdf = os.path.join(tmp, "sourdough.pdf")
        build_pdf(novel_pdf, [novel_text])

        print(f"\n[2/4] Scoring BERT-like text against BERT's citation corpus ...")
        r1 = score_paper_against_citations(
            similar_pdf, build_report["corpus_id"], corpus, processor,
            detector.embedding_model,
        )
        print(f"  chunks={r1['chunks_scored']} avg_novelty={r1['avg_novelty']:.3f}")

        print(f"\n[3/4] Scoring sourdough text against BERT's citation corpus ...")
        r2 = score_paper_against_citations(
            novel_pdf, build_report["corpus_id"], corpus, processor,
            detector.embedding_model,
        )
        print(f"  chunks={r2['chunks_scored']} avg_novelty={r2['avg_novelty']:.3f}")

        print(f"\n[4/4] Assertion: sourdough novelty > BERT-like novelty")
        ok = r2["avg_novelty"] > r1["avg_novelty"]
        print(f"  result: {'PASS' if ok else 'FAIL'} "
              f"({r2['avg_novelty']:.3f} vs {r1['avg_novelty']:.3f})")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

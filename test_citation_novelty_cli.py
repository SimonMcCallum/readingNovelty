"""
Integration test for citation_novelty_cli: builds a corpus from mocked
reference abstracts, scores a synthesized PDF, and verifies the math.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import fitz
import numpy as np

import citation_novelty_cli
from corpus import CorpusStore
from novelty_detector import NoveltyDetector
from pdf_processor import PDFProcessor


def _build_pdf(path: str, paragraphs):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 60
    for para in paragraphs:
        chunked = para
        while chunked:
            page.insert_text((50, y), chunked[:80], fontsize=11, fontname='helv')
            chunked = chunked[80:]
            y += 16
            if y > 780:
                page = doc.new_page(width=595, height=842)
                y = 60
        y += 14
    doc.save(path)
    doc.close()


class TestBuildCitationCorpus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='cite_test_')
        os.environ['ANTHROPIC_API_KEY'] = ''
        os.environ['OPENAI_API_KEY'] = ''
        os.environ['GEMINI_API_KEY'] = ''
        os.environ['OLLAMA_HOST'] = ''
        os.environ['OLLAMA_REMOTE_URL'] = ''
        self.detector = NoveltyDetector()
        self.processor = PDFProcessor()
        self.corpus = CorpusStore(
            os.path.join(self.tmp, 'corpus'),
            embedding_dim=self.detector.embedding_dim,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ingests_refs_with_abstracts_drops_short_ones(self):
        fetcher = MagicMock()
        fetcher.fetch_references.return_value = [
            {'paperId': 'ref1', 'abstract': 'Quantum mechanics underpins cryptography. ' * 6,
             'year': 2015, 'title': 'A'},
            {'paperId': 'ref2', 'abstract': 'Too short.', 'year': 2015, 'title': 'B'},
            {'paperId': 'ref3', 'abstract': '', 'year': 2015, 'title': 'C'},
            {'paperId': 'ref4', 'abstract': 'Marine biology in deep ocean ecosystems matters. ' * 6,
             'year': 2017, 'title': 'D'},
        ]
        report = citation_novelty_cli.build_citation_corpus(
            '10.x/y', fetcher, self.corpus, self.processor,
            self.detector.embedding_model,
        )
        self.assertEqual(report['references_total'], 4)
        self.assertEqual(report['references_with_abstract_ingested'], 2)
        self.assertEqual(report['references_skipped_no_abstract'], 2)
        info = self.corpus.get_assignment(report['corpus_id'])
        self.assertEqual(info['submission_count'], 2)

    def test_idempotent_by_default(self):
        fetcher = MagicMock()
        fetcher.fetch_references.return_value = [
            {'paperId': 'ref1', 'abstract': 'Quantum entanglement is foundational. ' * 6,
             'year': 2015, 'title': 'A'},
        ]
        first = citation_novelty_cli.build_citation_corpus(
            '10.x/y', fetcher, self.corpus, self.processor, self.detector.embedding_model,
        )
        second = citation_novelty_cli.build_citation_corpus(
            '10.x/y', fetcher, self.corpus, self.processor, self.detector.embedding_model,
        )
        self.assertEqual(first['references_with_abstract_ingested'], 1)
        self.assertEqual(second['references_with_abstract_ingested'], 0)
        self.assertEqual(second['references_skipped_already_present'], 1)

    def test_passes_year_max_to_fetcher(self):
        fetcher = MagicMock()
        fetcher.fetch_references.return_value = []
        citation_novelty_cli.build_citation_corpus(
            '10.x/y', fetcher, self.corpus, self.processor,
            self.detector.embedding_model, target_year=2020,
        )
        fetcher.fetch_references.assert_called_once_with('10.x/y', year_max=2020)

    def test_score_paper_against_citations(self):
        # Seed the corpus with two reference abstracts
        fetcher = MagicMock()
        fetcher.fetch_references.return_value = [
            {'paperId': 'ref1',
             'abstract': 'Quantum entanglement enables superdense coding protocols. ' * 6,
             'year': 2015, 'title': 'A'},
            {'paperId': 'ref2',
             'abstract': 'Marine biology of cephalopod intelligence in deep ocean trenches. ' * 6,
             'year': 2017, 'title': 'B'},
        ]
        build = citation_novelty_cli.build_citation_corpus(
            '10.x/y', fetcher, self.corpus, self.processor, self.detector.embedding_model,
        )

        # Target paper closely resembles ref1 -> low novelty expected
        similar_pdf = os.path.join(self.tmp, 'similar.pdf')
        _build_pdf(similar_pdf, [
            'Quantum entanglement enables superdense coding protocols. ' * 6,
        ])
        similar_report = citation_novelty_cli.score_paper_against_citations(
            similar_pdf, build['corpus_id'], self.corpus, self.processor,
            self.detector.embedding_model,
        )
        self.assertLess(similar_report['avg_novelty'], 0.4)

        # Target paper that talks about something entirely different -> high novelty
        novel_pdf = os.path.join(self.tmp, 'novel.pdf')
        _build_pdf(novel_pdf, [
            'Baroque counterpoint requires independent voice leading techniques. ' * 6,
        ])
        novel_report = citation_novelty_cli.score_paper_against_citations(
            novel_pdf, build['corpus_id'], self.corpus, self.processor,
            self.detector.embedding_model,
        )
        self.assertGreater(novel_report['avg_novelty'], 0.5)


if __name__ == '__main__':
    unittest.main()

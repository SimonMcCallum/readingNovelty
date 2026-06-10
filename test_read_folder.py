"""
Tests for the PhD read-folder CLI.

Creates a temp folder with small synthesized PDFs, ingests them, and scores
a 'new' PDF against the corpus. Uses the real embedding model but a stub
LLM-prediction provider to avoid Ollama dependence.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import fitz

from corpus import CorpusStore
from llm_providers import LLMProvider
from novelty_detector import NoveltyDetector
from pdf_processor import PDFProcessor
import read_folder_cli


_CLEAR_API_KEYS = {
    'ANTHROPIC_API_KEY': '', 'OPENAI_API_KEY': '', 'GEMINI_API_KEY': '',
    'OLLAMA_HOST': '', 'OLLAMA_REMOTE_URL': '',
}


def _build_pdf(path: str, paragraphs):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 60
    for para in paragraphs:
        chunked = para
        while chunked:
            page.insert_text((50, y), chunked[:80], fontsize=11, fontname="helv")
            chunked = chunked[80:]
            y += 16
            if y > 780:
                page = doc.new_page(width=595, height=842)
                y = 60
        y += 14
    doc.save(path)
    doc.close()


class EchoPredictor(LLMProvider):
    """Returns its own keyword hint as the prediction — keeps tests deterministic."""
    name = "echo-predictor"

    def generate_prompt(self, chunk, context_before='', context_after=''):
        return chunk[:60]

    def predict_chunk(self, context_before, context_after, hint, target_length_words=150):
        return hint


class TestReadFolderCli(unittest.TestCase):
    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='readfolder_')
        self.read_folder = os.path.join(self.tmp, 'read')
        self.inbox = os.path.join(self.tmp, 'inbox')
        self.corpus_dir = os.path.join(self.tmp, 'corpus')
        os.makedirs(self.read_folder)
        os.makedirs(self.inbox)

        _build_pdf(os.path.join(self.read_folder, 'paper_a.pdf'),
                   ["Quantum entanglement underpins quantum cryptography protocols. " * 4,
                    "Bell inequality tests rule out local hidden variable theories. " * 4])
        _build_pdf(os.path.join(self.read_folder, 'paper_b.pdf'),
                   ["Marine biology examines cephalopod cognition in deep ocean ecosystems. " * 4])

        self.detector = NoveltyDetector(provider=EchoPredictor())
        self.processor = PDFProcessor()
        self.corpus = CorpusStore(self.corpus_dir, embedding_dim=self.detector.embedding_dim)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ingest_folder_imports_all_pdfs(self):
        report = read_folder_cli.ingest_folder(
            self.read_folder, 'phd_test', self.corpus,
            self.detector, self.processor,
        )
        self.assertEqual(report['ingested'], 2)
        self.assertEqual(report['skipped'], 0)
        self.assertEqual(report['failed'], 0)
        info = self.corpus.get_assignment('phd_test')
        self.assertEqual(info['submission_count'], 2)
        self.assertGreater(info['chunk_count'], 0)

    def test_ingest_is_idempotent_by_default(self):
        read_folder_cli.ingest_folder(
            self.read_folder, 'phd_test', self.corpus,
            self.detector, self.processor,
        )
        second = read_folder_cli.ingest_folder(
            self.read_folder, 'phd_test', self.corpus,
            self.detector, self.processor,
        )
        self.assertEqual(second['ingested'], 0)
        self.assertEqual(second['skipped'], 2)

    def test_reingest_forces_overwrite(self):
        read_folder_cli.ingest_folder(
            self.read_folder, 'phd_test', self.corpus,
            self.detector, self.processor,
        )
        forced = read_folder_cli.ingest_folder(
            self.read_folder, 'phd_test', self.corpus,
            self.detector, self.processor, force=True,
        )
        self.assertEqual(forced['ingested'], 2)
        self.assertEqual(forced['skipped'], 0)

    def test_score_new_pdf_against_read_corpus(self):
        read_folder_cli.ingest_folder(
            self.read_folder, 'phd_test', self.corpus,
            self.detector, self.processor,
        )

        # 'New' PDF that closely mirrors paper_a content -> low novelty expected
        similar_path = os.path.join(self.inbox, 'similar.pdf')
        _build_pdf(similar_path, [
            "Quantum entanglement underpins quantum cryptography protocols. " * 4,
            "Bell inequality tests rule out local hidden variable theories. " * 4,
        ])
        out_path = os.path.join(self.inbox, 'annotated.pdf')
        report = read_folder_cli.score_new_pdf(
            similar_path, 'phd_test', self.corpus, self.detector, self.processor,
            alpha=1.0, annotated_output=out_path,
        )
        self.assertEqual(report['alpha'], 1.0)
        self.assertGreater(report['chunks_scored'], 0)
        self.assertIsNone(report['avg_llm_novelty'])
        self.assertLess(report['avg_corpus_novelty'], 0.5)  # similar content
        self.assertTrue(os.path.exists(out_path))

    def test_score_unrelated_pdf_scores_higher_novelty(self):
        read_folder_cli.ingest_folder(
            self.read_folder, 'phd_test', self.corpus,
            self.detector, self.processor,
        )
        novel_path = os.path.join(self.inbox, 'novel.pdf')
        _build_pdf(novel_path, [
            "Baroque keyboard counterpoint requires careful voice independence training. " * 4,
        ])
        report = read_folder_cli.score_new_pdf(
            novel_path, 'phd_test', self.corpus, self.detector, self.processor,
            alpha=1.0,
        )
        self.assertGreater(report['avg_corpus_novelty'], 0.5)

    def test_alpha_blend_uses_both_scores(self):
        read_folder_cli.ingest_folder(
            self.read_folder, 'phd_test', self.corpus,
            self.detector, self.processor,
        )
        target_path = os.path.join(self.inbox, 'target.pdf')
        _build_pdf(target_path, [
            "Quantum entanglement underpins quantum cryptography protocols. " * 4,
        ])
        report = read_folder_cli.score_new_pdf(
            target_path, 'phd_test', self.corpus, self.detector, self.processor,
            alpha=0.5,
        )
        self.assertEqual(report['alpha'], 0.5)
        self.assertIsNotNone(report['avg_llm_novelty'])
        # blended should be roughly midway between corpus and llm components
        blended = report['avg_blended_novelty']
        expected = 0.5 * report['avg_corpus_novelty'] + 0.5 * report['avg_llm_novelty']
        self.assertAlmostEqual(blended, expected, places=5)

    def test_alpha_below_one_with_fallback_provider_raises(self):
        read_folder_cli.ingest_folder(
            self.read_folder, 'phd_test', self.corpus,
            self.detector, self.processor,
        )
        target_path = os.path.join(self.inbox, 'target.pdf')
        _build_pdf(target_path, ["Some text. " * 30])

        fallback_detector = NoveltyDetector()  # picks fallback when env is clean
        with self.assertRaises(RuntimeError):
            read_folder_cli.score_new_pdf(
                target_path, 'phd_test', self.corpus, fallback_detector,
                self.processor, alpha=0.5,
            )


if __name__ == '__main__':
    unittest.main()

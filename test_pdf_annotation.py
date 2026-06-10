"""
Tests for inline PDF annotation with PyMuPDF.

Builds a tiny PDF in-memory, highlights it, and verifies highlights were
actually added (not just a summary page prepended).
"""

import os
import shutil
import tempfile
import unittest

import fitz

from pdf_processor import PDFProcessor


def _build_pdf(path, paragraphs):
    """Create a multi-paragraph PDF with PyMuPDF (no reportlab needed)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 60
    for para in paragraphs:
        page.insert_text((50, y), para, fontsize=11, fontname="helv")
        y += 30
        if y > 780:
            page = doc.new_page(width=595, height=842)
            y = 60
    doc.save(path)
    doc.close()


class TestInlineAnnotation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='annot_test_')
        self.processor = PDFProcessor()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_highlights_added_for_matching_chunks(self):
        original = os.path.join(self.tmp, 'source.pdf')
        _build_pdf(original, [
            "Quantum mechanics describes nature at the smallest scales.",
            "Marine biology studies organisms living in saltwater ecosystems.",
            "Machine learning trains models on labelled data sets.",
        ])
        chunks = [
            {'text': 'Quantum mechanics describes nature at the smallest scales.'},
            {'text': 'Marine biology studies organisms living in saltwater ecosystems.'},
            {'text': 'Machine learning trains models on labelled data sets.'},
        ]
        scores = [0.9, 0.3, 0.6]
        output = os.path.join(self.tmp, 'annotated.pdf')

        report = self.processor.create_annotated_pdf(original, chunks, scores, output)

        self.assertTrue(os.path.exists(output))
        self.assertEqual(report['chunks_highlighted'], 3)
        self.assertEqual(report['chunks_unmatched'], 0)

        doc = fitz.open(output)
        try:
            # First page is the summary page we prepended; original pages follow.
            self.assertGreaterEqual(len(doc), 2)
            highlight_count = 0
            for page in doc:
                for annot in page.annots() or []:
                    if annot.type[0] == fitz.PDF_ANNOT_HIGHLIGHT:
                        highlight_count += 1
            self.assertEqual(highlight_count, 3)
        finally:
            doc.close()

    def test_unmatched_chunk_skipped_silently(self):
        original = os.path.join(self.tmp, 'source.pdf')
        _build_pdf(original, ["Only one paragraph exists in this document."])
        chunks = [
            {'text': 'Only one paragraph exists in this document.'},
            {'text': 'This text does not appear anywhere in the PDF.'},
        ]
        output = os.path.join(self.tmp, 'annotated.pdf')

        report = self.processor.create_annotated_pdf(original, chunks, [0.8, 0.2], output)

        self.assertEqual(report['chunks_highlighted'], 1)
        self.assertEqual(report['chunks_unmatched'], 1)

    def test_summary_page_prepended(self):
        original = os.path.join(self.tmp, 'source.pdf')
        _build_pdf(original, ["A solitary paragraph."])
        output = os.path.join(self.tmp, 'annotated.pdf')

        original_doc = fitz.open(original)
        original_page_count = len(original_doc)
        original_doc.close()

        self.processor.create_annotated_pdf(
            original, [{'text': 'A solitary paragraph.'}], [0.5], output
        )

        doc = fitz.open(output)
        try:
            # Output has exactly one more page than the original (the summary)
            self.assertEqual(len(doc), original_page_count + 1)
            # And the summary page contains the report title
            self.assertIn("Novelty Analysis", doc[0].get_text())
        finally:
            doc.close()


if __name__ == '__main__':
    unittest.main()

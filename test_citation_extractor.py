"""
Tests for the DOI / arXiv id extractor.
"""

import unittest

from citation_extractor import extract_dois, extract_arxiv_ids


class TestDoiExtraction(unittest.TestCase):
    def test_basic_doi(self):
        text = "See Smith et al. (doi:10.1038/nature12373) for context."
        self.assertEqual(extract_dois(text), ['10.1038/nature12373'])

    def test_strips_trailing_punctuation(self):
        text = "as shown in 10.1145/3501385.3543957, the result holds."
        dois = extract_dois(text)
        self.assertEqual(dois, ['10.1145/3501385.3543957'])

    def test_deduplicates_preserving_order(self):
        text = "10.1000/abc and later again 10.1000/abc and 10.1000/xyz."
        self.assertEqual(extract_dois(text), ['10.1000/abc', '10.1000/xyz'])

    def test_case_insensitive_matched_but_lowered(self):
        text = "DOI:10.1000/ABC"
        self.assertEqual(extract_dois(text), ['10.1000/abc'])

    def test_no_dois_returns_empty(self):
        self.assertEqual(extract_dois("just normal prose."), [])


class TestArxivExtraction(unittest.TestCase):
    def test_new_style(self):
        text = "see arXiv:2106.09685 for LoRA."
        self.assertEqual(extract_arxiv_ids(text), ['2106.09685'])

    def test_new_style_with_version(self):
        text = "see arXiv:2106.09685v2 for LoRA."
        self.assertEqual(extract_arxiv_ids(text), ['2106.09685'])

    def test_old_style(self):
        text = "cf. cs.LG/0301234 in the literature."
        self.assertEqual(extract_arxiv_ids(text), ['cs.lg/0301234'])

    def test_deduplicates(self):
        text = "arXiv:2106.09685 and arXiv:2106.09685"
        self.assertEqual(extract_arxiv_ids(text), ['2106.09685'])

    def test_no_arxiv_returns_empty(self):
        self.assertEqual(extract_arxiv_ids("plain prose"), [])


if __name__ == '__main__':
    unittest.main()

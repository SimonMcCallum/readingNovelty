"""
Tests for SemanticScholarClient. No real HTTP calls — all responses mocked.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from citation_fetcher import SemanticScholarClient, CitationFetchError


def _ok(json_body):
    r = MagicMock(status_code=200)
    r.json.return_value = json_body
    return r


def _status(code, json_body=None, text=''):
    r = MagicMock(status_code=code, text=text)
    r.json.return_value = json_body or {}
    return r


class TestNormalizeIdentifier(unittest.TestCase):
    def test_bare_doi_gets_doi_prefix(self):
        self.assertEqual(
            SemanticScholarClient.normalize_identifier('10.1038/nature12373'),
            'DOI:10.1038/nature12373',
        )

    def test_already_prefixed_doi_is_kept(self):
        self.assertEqual(
            SemanticScholarClient.normalize_identifier('DOI:10.x/y'),
            'DOI:10.x/y',
        )

    def test_arxiv_numeric_gets_prefix(self):
        self.assertEqual(
            SemanticScholarClient.normalize_identifier('2106.09685'),
            'ARXIV:2106.09685',
        )

    def test_already_prefixed_arxiv_is_kept(self):
        self.assertEqual(
            SemanticScholarClient.normalize_identifier('ARXIV:2106.09685'),
            'ARXIV:2106.09685',
        )


class TestFetchPaper(unittest.TestCase):
    def test_returns_paper_json(self):
        c = SemanticScholarClient(rate_limit_seconds=0)
        body = {'paperId': 'abc', 'title': 'T', 'abstract': 'A', 'year': 2020}
        with patch.object(c.session, 'get', return_value=_ok(body)) as get:
            paper = c.fetch_paper('10.1038/nature12373')
        self.assertEqual(paper['title'], 'T')
        call = get.call_args
        self.assertIn('/paper/DOI:10.1038/nature12373', call.args[0])

    def test_404_raises(self):
        c = SemanticScholarClient(rate_limit_seconds=0)
        with patch.object(c.session, 'get', return_value=_status(404, text='nope')):
            with self.assertRaises(CitationFetchError):
                c.fetch_paper('10.1/missing')

    def test_429_raises_with_clear_message(self):
        c = SemanticScholarClient(rate_limit_seconds=0)
        with patch.object(c.session, 'get', return_value=_status(429)):
            with self.assertRaises(CitationFetchError) as ctx:
                c.fetch_paper('10.1/x')
        self.assertIn('rate limit', str(ctx.exception).lower())


class TestFetchReferences(unittest.TestCase):
    def test_returns_cited_papers(self):
        c = SemanticScholarClient(rate_limit_seconds=0)
        refs_body = {
            'data': [
                {'citedPaper': {'paperId': 'r1', 'abstract': 'one', 'year': 2018, 'title': 'A'}},
                {'citedPaper': {'paperId': 'r2', 'abstract': 'two', 'year': 2019, 'title': 'B'}},
                {'citedPaper': {}},  # empty cited paper — dropped
                {},                   # no citedPaper key — dropped
            ]
        }
        with patch.object(c.session, 'get', return_value=_ok(refs_body)):
            refs = c.fetch_references('10.x/y')
        ids = [r['paperId'] for r in refs]
        self.assertEqual(ids, ['r1', 'r2'])

    def test_year_max_filters(self):
        c = SemanticScholarClient(rate_limit_seconds=0)
        refs_body = {
            'data': [
                {'citedPaper': {'paperId': 'old', 'year': 2010, 'abstract': '', 'title': ''}},
                {'citedPaper': {'paperId': 'cusp', 'year': 2019, 'abstract': '', 'title': ''}},
                {'citedPaper': {'paperId': 'new', 'year': 2020, 'abstract': '', 'title': ''}},
                {'citedPaper': {'paperId': 'noyear', 'abstract': '', 'title': ''}},
            ]
        }
        with patch.object(c.session, 'get', return_value=_ok(refs_body)):
            refs = c.fetch_references('10.x/y', year_max=2019)
        ids = sorted(r['paperId'] for r in refs)
        # 2010 + 2019 (the cusp is dropped because >= year_max) + noyear (kept)
        self.assertEqual(ids, ['noyear', 'old'])


class TestCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='s2cache_')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_second_call_is_cached(self):
        c = SemanticScholarClient(cache_dir=self.tmp, rate_limit_seconds=0)
        body = {'paperId': 'abc', 'abstract': 'A'}
        with patch.object(c.session, 'get', return_value=_ok(body)) as get:
            c.fetch_paper('10.x/y')
            c.fetch_paper('10.x/y')
        self.assertEqual(get.call_count, 1)  # second call hit the cache

    def test_cache_separate_per_identifier(self):
        c = SemanticScholarClient(cache_dir=self.tmp, rate_limit_seconds=0)
        with patch.object(c.session, 'get', side_effect=[_ok({'a': 1}), _ok({'b': 2})]) as get:
            r1 = c.fetch_paper('10.x/y')
            r2 = c.fetch_paper('10.x/z')
        self.assertNotEqual(r1, r2)
        self.assertEqual(get.call_count, 2)


if __name__ == '__main__':
    unittest.main()

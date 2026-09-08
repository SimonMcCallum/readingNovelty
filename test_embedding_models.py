"""
Tests for embedding model discovery (embedding_models.py) and the CLI helpers.

No network: Hub responses are fixtures, requests is mocked where a fetch is
exercised, and the disk cache is pointed at a temp directory.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import embedding_models as em


def _card(*task_entries):
    """Build a model-index cardData block from (dataset_name, metrics, [config]) tuples."""
    results = []
    for entry in task_entries:
        name, metrics = entry[0], entry[1]
        ds = {'name': name, 'type': name.lower()}
        if len(entry) > 2:
            ds['config'] = entry[2]
        results.append({
            'task': {'type': 'STS'},
            'dataset': ds,
            'metrics': [{'type': k, 'value': v} for k, v in metrics.items()],
        })
    return {'model-index': [{'name': 'x', 'results': results}]}


def _hub_model(model_id, card=None, params=110_000_000, downloads=1000, likes=10,
               library='sentence-transformers', gated=False, tags=()):
    m = {
        'id': model_id, 'downloads': downloads, 'likes': likes,
        'library_name': library, 'gated': gated, 'tags': list(tags),
        'pipeline_tag': 'sentence-similarity',
    }
    if params is not None:
        m['safetensors'] = {'total': params}
    if card is not None:
        m['cardData'] = card
    return m


class TestParsing(unittest.TestCase):
    def test_parse_params(self):
        self.assertEqual(em.parse_params('600M'), 600_000_000)
        self.assertEqual(em.parse_params('1.5B'), 1_500_000_000)
        self.assertEqual(em.parse_params('33000000'), 33_000_000)
        self.assertIsNone(em.parse_params(None))
        self.assertIsNone(em.parse_params(''))
        with self.assertRaises(ValueError):
            em.parse_params('big')

    def test_human_params(self):
        self.assertEqual(em.human_params(33_360_000), '33M')
        self.assertEqual(em.human_params(1_543_268_864), '1.5B')
        self.assertEqual(em.human_params(None), '?')

    def test_split_dataset_name(self):
        self.assertEqual(em.split_dataset_name('MTEB STS17 (en-en)'), ('STS17', 'en-en'))
        self.assertEqual(em.split_dataset_name('MTEB STS22.v2'), ('STS22', None))
        self.assertEqual(em.split_dataset_name('MTEB STSBenchmark', 'default'),
                         ('STSBenchmark', 'default'))
        self.assertEqual(em.split_dataset_name('MTEB STS22 (de)'), ('STS22', 'de'))


class TestCardScores(unittest.TestCase):
    def test_reads_english_sts_spearman_only(self):
        card = _card(
            ('MTEB STSBenchmark', {'cos_sim_pearson': 80.0, 'cos_sim_spearman': 85.0}),
            ('MTEB STS17 (en-en)', {'cosine_spearman': 90.0}),
            ('MTEB STS17 (fr-en)', {'cosine_spearman': 10.0}),   # non-English pair ignored
            ('MTEB STS22 (de)', {'cosine_spearman': 5.0}),       # non-English ignored
            ('MTEB BIOSSES', {'spearman': 0.75}),                # 0..1 scale normalised
        )
        scores = em.sts_scores_from_card(card)
        self.assertEqual(scores, {'STSBenchmark': 85.0, 'STS17': 90.0, 'BIOSSES': 75.0})

    def test_ignores_non_sts_tasks(self):
        card = {'model-index': [{'results': [{
            'task': {'type': 'Retrieval'}, 'dataset': {'name': 'MTEB STS12'},
            'metrics': [{'type': 'ndcg_at_10', 'value': 50}],
        }]}]}
        self.assertEqual(em.sts_scores_from_card(card), {})

    def test_empty_or_missing_card(self):
        self.assertEqual(em.sts_scores_from_card(None), {})
        self.assertEqual(em.sts_scores_from_card({}), {})


class TestRanking(unittest.TestCase):
    def test_ranks_by_mean_sts_and_dedupes_clones(self):
        good = _card(('MTEB STS12', {'cosine_spearman': 80}),
                     ('MTEB STS13', {'cosine_spearman': 90}))
        weak = _card(('MTEB STS12', {'cosine_spearman': 60}),
                     ('MTEB STS13', {'cosine_spearman': 70}))
        models = [
            _hub_model('org/weak', weak, downloads=999_999),
            _hub_model('org/good', good, params=100, downloads=50),
            _hub_model('mirror/good-copy', good, params=100, downloads=10),  # identical clone
            _hub_model('org/no-card', None),
        ]
        ranked = em.rank_from_hub_models(models)
        self.assertEqual([c.model_id for c in ranked], ['org/good', 'org/weak'])
        self.assertAlmostEqual(ranked[0].sts_mean, 85.0)
        self.assertEqual(ranked[0].n_sts_tasks, 2)
        self.assertEqual(ranked[0].source, 'cards')

    def test_candidate_flags(self):
        m = _hub_model('g/gemma', gated='manual', tags=['custom_code'], library='transformers')
        c = em.candidate_from_hub_model(m)
        self.assertTrue(c.gated)
        self.assertTrue(c.custom_code)
        self.assertFalse(c.runs_with_sentence_transformers)

    def test_filter_candidates(self):
        cands = [
            em.EmbeddingCandidate('a/st-small', 80, 9, params=100_000_000, library='sentence-transformers'),
            em.EmbeddingCandidate('b/st-big', 85, 9, params=7_000_000_000, library='sentence-transformers'),
            em.EmbeddingCandidate('c/gated', 84, 9, params=300_000_000, library='sentence-transformers', gated=True),
            em.EmbeddingCandidate('d/remote', 83, 9, params=300_000_000, library='sentence-transformers', custom_code=True),
            em.EmbeddingCandidate('e/api-only', 90, 9, library=None),
            em.EmbeddingCandidate('f/few-tasks', 88, 1, params=100_000_000, library='sentence-transformers'),
        ]
        kept = em.filter_candidates(cands, max_params=em.parse_params('1B'), min_tasks=5)
        self.assertEqual([c.model_id for c in kept], ['a/st-small'])
        kept = em.filter_candidates(cands, allow_gated=True, allow_custom_code=True,
                                    require_sentence_transformers=False)
        self.assertEqual(len(kept), 6)

    def test_aggregate_results_rows(self):
        rows_by_task = {
            'STS12': [
                {'model_name': 'x/one', 'score': 0.80, 'subset': 'default', 'language': ['eng-Latn'], 'is_public': True},
                {'model_name': 'x/two', 'score': 0.70, 'subset': 'default', 'language': ['eng-Latn'], 'is_public': True},
                {'model_name': 'x/private', 'score': 0.99, 'subset': 'default', 'language': ['eng-Latn'], 'is_public': False},
            ],
            'STS17': [
                {'model_name': 'x/one', 'score': 0.90, 'subset': 'en-en', 'language': ['eng-Latn'], 'is_public': True},
                {'model_name': 'x/one', 'score': 0.10, 'subset': 'fr-en', 'language': ['fra-Latn', 'eng-Latn'], 'is_public': True},
                {'model_name': 'x/two', 'score': 0.60, 'subset': 'en-en', 'language': ['eng-Latn'], 'is_public': True},
            ],
            'STS13': [
                {'model_name': 'x/one', 'score': 0.85, 'subset': 'default', 'language': ['eng-Latn'], 'is_public': True},
            ],
        }
        cands = em.aggregate_results_rows(rows_by_task, min_task_fraction=0.6)
        ids = [c.model_id for c in cands]
        self.assertEqual(ids, ['x/one', 'x/two'])          # private model dropped
        self.assertAlmostEqual(cands[0].sts_mean, 85.0)     # (80 + 90 + 85) / 3, fr-en ignored
        self.assertEqual(cands[1].n_sts_tasks, 2)

    def test_rank_from_mteb_dataframe_long_and_wide(self):
        import pandas as pd
        tasks = ('STS12', 'STS13')
        long = pd.DataFrame([
            {'model_name': 'm/a', 'task_name': 'STS12', 'score': 0.8},
            {'model_name': 'm/a', 'task_name': 'STS13', 'score': 0.9},
            {'model_name': 'm/b', 'task_name': 'STS12', 'score': 0.5},
            {'model_name': 'm/b', 'task_name': 'STS13', 'score': 0.6},
        ])
        ranked = em.rank_from_mteb_dataframe(long, tasks)
        self.assertEqual([c.model_id for c in ranked], ['m/a', 'm/b'])
        # mteb's default wide format: one row per task, one column per model
        wide = pd.DataFrame([
            {'task_name': 'STS12', 'm/a': 0.8, 'm/b': 0.5},
            {'task_name': 'STS13', 'm/a': 0.9, 'm/b': 0.6},
            {'task_name': 'Banking77Classification', 'm/a': 0.1, 'm/b': 0.99},  # ignored
        ])
        ranked = em.rank_from_mteb_dataframe(wide, tasks)
        self.assertEqual([c.model_id for c in ranked], ['m/a', 'm/b'])
        self.assertAlmostEqual(ranked[0].sts_mean, 85.0)
        self.assertEqual(ranked[0].n_sts_tasks, 2)
        # Same, with tasks on the index instead of a column
        wide_idx = wide.set_index('task_name')
        ranked = em.rank_from_mteb_dataframe(wide_idx, tasks)
        self.assertEqual([c.model_id for c in ranked], ['m/a', 'm/b'])


class TestCacheAndFetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='emcache_')
        self._env = patch.dict(os.environ, {'EMBEDDING_CACHE_DIR': self.tmp})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cached_json_hits_disk_second_time(self):
        calls = []

        def fetch():
            calls.append(1)
            return {'v': len(calls)}

        first = em._cached_json('k', fetch)
        second = em._cached_json('k', fetch)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        third = em._cached_json('k', fetch, refresh=True)
        self.assertEqual(third, {'v': 2})
        self.assertEqual(len(os.listdir(self.tmp)), 1)

    @patch('embedding_models.requests.get')
    def test_fetch_hub_models_follows_next_link(self, mock_get):
        page1 = MagicMock(status_code=200)
        page1.json.return_value = [_hub_model('a/1'), _hub_model('a/2')]
        page1.links = {'next': {'url': 'https://huggingface.co/api/models?cursor=abc'}}
        page2 = MagicMock(status_code=200)
        page2.json.return_value = [_hub_model('a/3')]
        page2.links = {}
        mock_get.side_effect = [page1, page2]
        models = em.fetch_hub_models(filter_tag='mteb', limit=3)
        self.assertEqual([m['id'] for m in models], ['a/1', 'a/2', 'a/3'])
        self.assertEqual(mock_get.call_count, 2)
        # Second call goes to the cursor URL with no params
        self.assertEqual(mock_get.call_args_list[1].args[0],
                         'https://huggingface.co/api/models?cursor=abc')

    @patch('embedding_models.requests.get')
    def test_fetch_hub_models_raises_on_error_status(self, mock_get):
        mock_get.return_value = MagicMock(status_code=503, text='down')
        with self.assertRaises(em.SourceUnavailable):
            em.fetch_hub_models(limit=1)

    @patch('embedding_models.requests.get')
    def test_results_ranking_raises_when_index_loading(self, mock_get):
        resp = MagicMock(status_code=500, text='the dataset index is loading')
        mock_get.return_value = resp
        with self.assertRaises(em.SourceUnavailable):
            em.fetch_results_ranking(tasks=('STS12',))

    @patch('embedding_models.fetch_cards_ranking')
    @patch('embedding_models.fetch_results_ranking')
    def test_auto_falls_back_to_cards(self, mock_results, mock_cards):
        mock_results.side_effect = em.SourceUnavailable('index loading')
        mock_cards.return_value = [em.EmbeddingCandidate('x/y', 80, 9)]
        cands, source = em.load_ranking('auto')
        self.assertEqual(source, 'cards')
        self.assertEqual(cands[0].model_id, 'x/y')

    @patch('embedding_models.fetch_model_meta')
    def test_enrich_fills_library_and_marks_missing(self, mock_meta):
        mock_meta.side_effect = lambda mid, refresh=False: (
            None if mid == 'cohere/api-only' else _hub_model(mid, params=42, library='sentence-transformers')
        )
        cands = [em.EmbeddingCandidate('a/b', 80, 9, source='results'),
                 em.EmbeddingCandidate('cohere/api-only', 90, 9, source='results')]
        em.enrich_with_hub(cands, top_n=10)
        self.assertEqual(cands[0].library, 'sentence-transformers')
        self.assertEqual(cands[0].params, 42)
        self.assertIsNone(cands[1].library)


class TestEnvUpsert(unittest.TestCase):
    def test_upsert_replaces_commented_line_and_appends(self):
        from embedding_models_cli import _upsert_env
        tmp = tempfile.mkdtemp(prefix='envtest_')
        try:
            path = os.path.join(tmp, '.env')
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write("LOCAL_ONLY=1\n# EMBEDDING_MODEL=all-MiniLM-L6-v2\nOTHER=x\n")
            self.assertEqual(_upsert_env(path, 'EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5'), 'updated')
            self.assertEqual(_upsert_env(path, 'NEW_KEY', 'v'), 'added')
            with open(path, encoding='utf-8') as fh:
                text = fh.read()
            self.assertIn('EMBEDDING_MODEL=BAAI/bge-base-en-v1.5\n', text)
            self.assertNotIn('# EMBEDDING_MODEL', text)
            self.assertTrue(text.endswith('NEW_KEY=v\n'))
            self.assertIn('LOCAL_ONLY=1\n', text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

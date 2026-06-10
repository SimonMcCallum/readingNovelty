"""
Tests for the per-assignment corpus store.

Uses temp directories and the same embedding model the server uses so the
FAISS dimension is honest, but constructs embeddings deterministically to
avoid needing real text-to-vector calls.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from corpus import CorpusStore


def _vec(seed, dim=16):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype('float32')


def _matrix(seeds, dim=16):
    return np.vstack([_vec(s, dim) for s in seeds])


class TestCorpusStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='corpus_test_')
        self.store = CorpusStore(self.tmp, embedding_dim=16)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_corpus_returns_max_novelty(self):
        scores = self.store.score_against_corpus('asg1', _matrix([1, 2, 3]))
        self.assertEqual(scores, [1.0, 1.0, 1.0])

    def test_ensure_assignment_is_idempotent(self):
        self.store.ensure_assignment('asg1', name='First name')
        self.store.ensure_assignment('asg1', name='Other name')
        info = self.store.get_assignment('asg1')
        self.assertEqual(info['name'], 'First name')  # initial name sticks

    def test_add_submission_grows_corpus(self):
        chunks = [{'text': 'a', 'prompt': 'p'}, {'text': 'b', 'prompt': 'q'}]
        rec = self.store.add_submission(
            'asg1', 'sub1', 'student-1', 'a.pdf',
            chunks, _matrix([10, 20]), [0.8, 0.9]
        )
        self.assertEqual(rec['chunk_count'], 2)
        info = self.store.get_assignment('asg1')
        self.assertEqual(info['chunk_count'], 2)
        self.assertEqual(info['submission_count'], 1)

    def test_second_submission_scored_against_first(self):
        chunks = [{'text': 'shared', 'prompt': 'p'}]
        emb = _matrix([42])
        # First submission lands in an empty corpus
        scores1 = self.store.score_against_corpus('asg1', emb)
        self.store.add_submission('asg1', 'sub1', 's1', 'a.pdf', chunks, emb, scores1)

        # Second submission with IDENTICAL embedding must score very low novelty
        scores2 = self.store.score_against_corpus('asg1', emb)
        self.assertLess(scores2[0], 0.1)

        # And a totally different embedding scores high
        scores3 = self.store.score_against_corpus('asg1', _matrix([99]))
        self.assertGreater(scores3[0], scores2[0])

    def test_exclude_submission_id_drops_own_chunks(self):
        chunks = [{'text': 'x', 'prompt': 'p'}]
        emb = _matrix([7])
        scores = self.store.score_against_corpus('asg1', emb)
        self.store.add_submission('asg1', 'sub1', 's1', 'a.pdf', chunks, emb, scores)

        # Without excluding, sub1 finds itself — score near 0
        included = self.store.score_against_corpus('asg1', emb)
        excluded = self.store.score_against_corpus(
            'asg1', emb, exclude_submission_id='sub1'
        )
        self.assertLess(included[0], 0.1)
        # With sub1 excluded the corpus is effectively empty -> full novelty
        self.assertEqual(excluded[0], 1.0)

    def test_persistence_across_instances(self):
        chunks = [{'text': 'x', 'prompt': 'p'}]
        emb = _matrix([5])
        self.store.add_submission('asg1', 'sub1', 's1', 'a.pdf', chunks, emb, [1.0])

        # New store instance reads the same on-disk state
        store2 = CorpusStore(self.tmp, embedding_dim=16)
        self.assertEqual(store2.get_assignment('asg1')['chunk_count'], 1)
        scores = store2.score_against_corpus('asg1', emb)
        self.assertLess(scores[0], 0.1)

    def test_list_and_get_submission(self):
        chunks = [{'text': 'hello', 'prompt': 'p1'}, {'text': 'world', 'prompt': 'p2'}]
        emb = _matrix([1, 2])
        self.store.add_submission('asg1', 'sub1', 'alice', 'a.pdf', chunks, emb, [0.5, 0.6])

        listed = self.store.list_submissions('asg1')
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]['student_id'], 'alice')
        self.assertAlmostEqual(listed[0]['avg_novelty'], 0.55, places=5)

        detail = self.store.get_submission('sub1')
        self.assertEqual(len(detail['chunks']), 2)
        self.assertEqual(detail['chunks'][0]['text'], 'hello')

    def test_reupload_replaces_chunks_but_not_corpus_growth(self):
        """Re-uploading a submission deletes its old chunk rows but FAISS
        rows for the prior version remain (they're orphaned, not addressable).

        This is the intentional simplification for Phase B: corpus only grows
        forward; a 'rescore' workflow would rebuild the index from chunks.
        """
        chunks = [{'text': 'v1', 'prompt': 'p'}]
        emb1 = _matrix([1])
        self.store.add_submission('asg1', 'sub1', 's1', 'a.pdf', chunks, emb1, [1.0])
        self.store.add_submission('asg1', 'sub1', 's1', 'a.pdf',
                                  [{'text': 'v2', 'prompt': 'p'}], _matrix([2]), [1.0])
        detail = self.store.get_submission('sub1')
        self.assertEqual(detail['chunks'][0]['text'], 'v2')


if __name__ == '__main__':
    unittest.main()

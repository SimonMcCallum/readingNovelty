"""
Tests for LLM-predictive novelty and the corpus/LLM blend.

Uses mocked providers so no real LLM is required. The embedding model is
real (small, downloads once) so the cosine numbers are honest.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Force fallback-only mode so the global env doesn't influence discovery
_CLEAR_API_KEYS = {
    'ANTHROPIC_API_KEY': '', 'OPENAI_API_KEY': '', 'GEMINI_API_KEY': '',
    'OLLAMA_HOST': '', 'OLLAMA_REMOTE_URL': '',
}


class TestKeywordHint(unittest.TestCase):
    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_strips_stopwords_and_picks_top_n(self):
        from novelty_detector import NoveltyDetector
        det = NoveltyDetector()
        text = "Quantum entanglement underpins quantum cryptography. Quantum entanglement is real."
        hint = det.keyword_hint(text, n=3)
        # 'quantum' appears most often and isn't a stopword
        self.assertIn('quantum', hint)
        self.assertLessEqual(len(hint.split()), 3)

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_returns_truncated_text_when_no_content_words(self):
        from novelty_detector import NoveltyDetector
        det = NoveltyDetector()
        hint = det.keyword_hint("the a an of")
        self.assertTrue(hint)


class TestCombineScores(unittest.TestCase):
    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_alpha_one_returns_corpus(self):
        from novelty_detector import NoveltyDetector
        out = NoveltyDetector.combine_novelty_scores([0.2, 0.8], [0.9, 0.1], alpha=1.0)
        self.assertEqual(out, [0.2, 0.8])

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_alpha_zero_returns_llm(self):
        from novelty_detector import NoveltyDetector
        out = NoveltyDetector.combine_novelty_scores([0.2, 0.8], [0.9, 0.1], alpha=0.0)
        self.assertEqual(out, [0.9, 0.1])

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_alpha_half_is_midpoint(self):
        from novelty_detector import NoveltyDetector
        out = NoveltyDetector.combine_novelty_scores([0.2, 0.8], [0.8, 0.2], alpha=0.5)
        self.assertEqual(out, [0.5, 0.5])

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_length_mismatch_raises(self):
        from novelty_detector import NoveltyDetector
        with self.assertRaises(ValueError):
            NoveltyDetector.combine_novelty_scores([0.5], [0.5, 0.5], alpha=0.5)

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_alpha_out_of_range_raises(self):
        from novelty_detector import NoveltyDetector
        with self.assertRaises(ValueError):
            NoveltyDetector.combine_novelty_scores([0.5], [0.5], alpha=1.5)


class TestAnalyzeLlmNovelty(unittest.TestCase):
    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_identical_prediction_scores_near_zero(self):
        """If the LLM predicts a passage identical to the actual chunk,
        novelty should be ~0 (cosine ~1)."""
        from novelty_detector import NoveltyDetector
        from llm_providers import LLMProvider

        class EchoProvider(LLMProvider):
            name = "echo"
            def __init__(self, chunk_text):
                self.chunk_text = chunk_text
            def generate_prompt(self, *a, **kw): return "ignored"
            def predict_chunk(self, *a, **kw):
                return self.chunk_text

        chunk_text = "Quantum entanglement allows distant particles to share correlated states."
        det = NoveltyDetector(provider=EchoProvider(chunk_text))
        chunks = [{'text': chunk_text}]
        scores = det.analyze_llm_novelty(chunks)
        self.assertEqual(len(scores), 1)
        self.assertLess(scores[0], 0.05)

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_orthogonal_prediction_scores_high(self):
        """If the LLM predicts something topically unrelated, novelty should be high."""
        from novelty_detector import NoveltyDetector
        from llm_providers import LLMProvider

        class OffTopicProvider(LLMProvider):
            name = "offtopic"
            def generate_prompt(self, *a, **kw): return "ignored"
            def predict_chunk(self, *a, **kw):
                return "Sourdough bread requires a wild yeast starter fed daily with flour and water."

        chunk_text = "Marine biology examines the behaviour of cephalopods in deep ocean trenches."
        det = NoveltyDetector(provider=OffTopicProvider())
        chunks = [{'text': chunk_text}]
        scores = det.analyze_llm_novelty(chunks)
        self.assertGreater(scores[0], 0.3)

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_provider_exception_does_not_crash(self):
        from novelty_detector import NoveltyDetector
        from llm_providers import LLMProvider

        class BrokenProvider(LLMProvider):
            name = "broken"
            def generate_prompt(self, *a, **kw): return "ignored"
            def predict_chunk(self, *a, **kw):
                raise RuntimeError("network down")

        det = NoveltyDetector(provider=BrokenProvider())
        scores = det.analyze_llm_novelty([{'text': 'short text'}])
        self.assertEqual(len(scores), 1)
        # Falls back to the hint as the prediction; non-crashing is the contract.
        self.assertGreaterEqual(scores[0], 0.0)
        self.assertLessEqual(scores[0], 1.0)


class TestPredictiveNoveltyParallel(unittest.TestCase):
    """Verify that thread-pool execution preserves order and result correctness."""

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_parallel_matches_serial(self):
        """Parallel results must match serial for the same provider/inputs."""
        from novelty_detector import NoveltyDetector
        from llm_providers import LLMProvider

        class DeterministicProvider(LLMProvider):
            """Returns a fixed reply derived from the hint — deterministic
            regardless of execution order or concurrency."""
            name = "det"

            def generate_prompt(self, *a, **kw):
                return "ignored"

            def predict_chunk(self, context_before, context_after, hint, target_length_words=150):
                return f"predicted from hint: {hint}"

        det = NoveltyDetector(provider=DeterministicProvider())
        chunks = [{'text': f'Topic {i} discusses some content. ' * 20} for i in range(6)]

        serial = det.analyze_llm_novelty(chunks, max_workers=1)
        parallel = det.analyze_llm_novelty(chunks, max_workers=4)

        self.assertEqual(len(serial), len(parallel))
        for s, p in zip(serial, parallel):
            self.assertAlmostEqual(s, p, places=5)

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_parallel_actually_runs_concurrently(self):
        """With a slow provider, parallel must be measurably faster than serial."""
        import time
        from novelty_detector import NoveltyDetector
        from llm_providers import LLMProvider

        class SlowProvider(LLMProvider):
            name = "slow"

            def generate_prompt(self, *a, **kw):
                return "ignored"

            def predict_chunk(self, *a, **kw):
                time.sleep(0.15)  # cheap I/O simulation
                return "predicted"

        det = NoveltyDetector(provider=SlowProvider())
        chunks = [{'text': f'Chunk {i}.'} for i in range(8)]

        t0 = time.monotonic()
        det.analyze_llm_novelty(chunks, max_workers=1)
        serial_s = time.monotonic() - t0

        t1 = time.monotonic()
        det.analyze_llm_novelty(chunks, max_workers=4)
        parallel_s = time.monotonic() - t1

        # 4-way parallelism on 8 items should be faster than serial. Allow
        # generous headroom: parallel must finish in less than 60% of serial.
        self.assertLess(parallel_s, serial_s * 0.6,
                        f"parallel={parallel_s:.2f}s should be << serial={serial_s:.2f}s")


class TestGeminiProvider(unittest.TestCase):
    def test_predict_chunk_posts_to_api(self):
        from llm_providers import GeminiProvider
        provider = GeminiProvider(api_key='test-key', model='gemini-2.5-flash')

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            'candidates': [{'content': {'parts': [{'text': '  predicted passage  '}]}}]
        }
        with patch('llm_providers.requests.post', return_value=mock_resp) as post:
            result = provider.predict_chunk("before", "after", "topic hint", target_length_words=80)
        self.assertEqual(result, 'predicted passage')
        call = post.call_args
        self.assertIn('gemini-2.5-flash:generateContent', call.args[0])
        self.assertEqual(call.kwargs['params']['key'], 'test-key')
        body = call.kwargs['json']
        self.assertIn('systemInstruction', body)
        self.assertIn('contents', body)

    def test_non_200_raises(self):
        from llm_providers import GeminiProvider
        provider = GeminiProvider(api_key='test-key')
        with patch('llm_providers.requests.post', return_value=MagicMock(status_code=429, text='rate')):
            with self.assertRaises(RuntimeError):
                provider.predict_chunk("a", "b", "c")

    def test_is_cloud_flag_set(self):
        from llm_providers import GeminiProvider
        self.assertTrue(GeminiProvider(api_key='x').is_cloud)

    @patch.dict(os.environ, {
        'LOCAL_ONLY': '0',
        'GEMINI_API_KEY': 'real-key',
    }, clear=True)
    def test_discovered_when_local_only_off(self):
        from llm_providers import discover_providers
        providers = discover_providers()
        self.assertIn('gemini', providers)

    @patch.dict(os.environ, {
        'LOCAL_ONLY': '1',
        'GEMINI_API_KEY': 'real-key',
    }, clear=True)
    def test_not_discovered_when_local_only_on(self):
        from llm_providers import discover_providers
        providers = discover_providers()
        self.assertNotIn('gemini', providers)


if __name__ == '__main__':
    unittest.main()

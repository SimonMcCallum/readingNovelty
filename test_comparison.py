"""
Tests for multi-model comparison functionality.

Uses FallbackProvider so no API keys or external services needed.
"""

import unittest
from unittest.mock import patch, MagicMock
import os

# Keys to remove to force fallback mode (don't clear entire env -- torch needs system vars)
_CLEAR_API_KEYS = {
    'ANTHROPIC_API_KEY': '',
    'OPENAI_API_KEY': '',
    'OLLAMA_HOST': '',
    'OLLAMA_REMOTE_HOST': '',
}


class TestNoveltyDetectorProviderIntegration(unittest.TestCase):
    """Test NoveltyDetector with provider system."""

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_default_provider_is_fallback(self):
        from novelty_detector import NoveltyDetector
        detector = NoveltyDetector()
        self.assertEqual(detector.active_provider.name, "fallback")
        self.assertEqual(detector.llm_type, "fallback")

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_providers_dict_available(self):
        from novelty_detector import NoveltyDetector
        detector = NoveltyDetector()
        self.assertIn('fallback', detector.providers)

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_custom_provider(self):
        from novelty_detector import NoveltyDetector
        from llm_providers import FallbackProvider
        custom = FallbackProvider()
        custom.name = "custom-test"
        detector = NoveltyDetector(provider=custom)
        self.assertEqual(detector.active_provider.name, "custom-test")

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_analyze_novelty_with_fallback(self):
        from novelty_detector import NoveltyDetector
        detector = NoveltyDetector()
        chunks = [
            {'text': 'Quantum physics explores wave-particle duality and uncertainty.', 'word_count': 8},
            {'text': 'Machine learning uses neural networks for pattern recognition.', 'word_count': 8},
            {'text': 'Neural networks in machine learning recognize complex patterns.', 'word_count': 8},
        ]
        scores = detector.analyze_novelty(chunks)
        self.assertEqual(len(scores), 3)
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_analyze_novelty_multi(self):
        from novelty_detector import NoveltyDetector
        detector = NoveltyDetector()
        chunks = [
            {'text': 'The quick brown fox jumps over the lazy dog.', 'word_count': 9},
            {'text': 'A completely different topic about space exploration.', 'word_count': 7},
        ]
        results = detector.analyze_novelty_multi(chunks)
        self.assertIn('fallback', results)
        self.assertEqual(len(results['fallback']), 2)

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_analyze_novelty_multi_specific_providers(self):
        from novelty_detector import NoveltyDetector
        detector = NoveltyDetector()
        chunks = [
            {'text': 'Test text about programming languages.', 'word_count': 5},
            {'text': 'Another text about cooking recipes.', 'word_count': 5},
        ]
        results = detector.analyze_novelty_multi(chunks, provider_names=['fallback'])
        self.assertEqual(list(results.keys()), ['fallback'])

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_analyze_novelty_multi_unknown_provider_skipped(self):
        from novelty_detector import NoveltyDetector
        detector = NoveltyDetector()
        chunks = [
            {'text': 'Test text.', 'word_count': 2},
        ]
        results = detector.analyze_novelty_multi(chunks, provider_names=['nonexistent'])
        self.assertEqual(results, {})

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def test_generate_prompt_falls_back_on_error(self):
        from novelty_detector import NoveltyDetector
        from llm_providers import LLMProvider

        class BrokenProvider(LLMProvider):
            name = "broken"
            def generate_prompt(self, chunk, context_before="", context_after=""):
                raise RuntimeError("provider failed")

        detector = NoveltyDetector()
        result = detector.generate_prompt_for_chunk(
            "test chunk", provider=BrokenProvider()
        )
        self.assertTrue(result.startswith("Write about:"))


class TestServerEndpoints(unittest.TestCase):
    """Test new server endpoints."""

    @patch.dict(os.environ, _CLEAR_API_KEYS)
    def setUp(self):
        from server import app
        self.app = app
        self.client = app.test_client()

    def test_providers_endpoint(self):
        resp = self.client.get('/providers')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('providers', data)
        self.assertIn('fallback', data['providers'])

    def test_compare_endpoint_no_text(self):
        resp = self.client.post('/compare', json={})
        self.assertEqual(resp.status_code, 400)

    def test_compare_endpoint_with_text(self):
        text = ("First paragraph about quantum computing and its applications. " * 20 +
                "\n\n" +
                "Second paragraph about marine biology and ocean ecosystems. " * 20)
        resp = self.client.post('/compare', json={'text': text})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertIn('providers', data)
        self.assertIn('fallback', data['providers'])
        self.assertIn('scores', data['providers']['fallback'])

    def test_analyze_with_provider_param(self):
        text = "Some text about testing. " * 30 + "\n\n" + "Different text about cooking. " * 30
        resp = self.client.post('/analyze?provider=fallback', json={'text': text})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])

    def test_health_still_works(self):
        resp = self.client.get('/health')
        self.assertEqual(resp.status_code, 200)


if __name__ == '__main__':
    unittest.main()

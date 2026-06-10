"""
Tests for LLM provider abstraction.

All tests work without API keys or running Ollama.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock


class TestFallbackProvider(unittest.TestCase):
    """Test the fallback provider (no external dependencies)."""

    def test_generate_prompt(self):
        from llm_providers import FallbackProvider
        provider = FallbackProvider()
        result = provider.generate_prompt("This is a test chunk of text with several words.")
        self.assertTrue(result.startswith("Write about:"))
        self.assertIn("test chunk", result)

    def test_is_available(self):
        from llm_providers import FallbackProvider
        provider = FallbackProvider()
        self.assertTrue(provider.is_available())

    def test_name(self):
        from llm_providers import FallbackProvider
        provider = FallbackProvider()
        self.assertEqual(provider.name, "fallback")

    def test_truncates_to_50_words(self):
        from llm_providers import FallbackProvider
        provider = FallbackProvider()
        long_text = " ".join([f"word{i}" for i in range(100)])
        result = provider.generate_prompt(long_text)
        # Should only contain first 50 words after "Write about: "
        words_in_result = result.replace("Write about: ", "").split()
        self.assertEqual(len(words_in_result), 50)


class TestOllamaProvider(unittest.TestCase):
    """Test Ollama provider with mocked HTTP."""

    def test_init_sets_base_url(self):
        from llm_providers import OllamaProvider
        provider = OllamaProvider(base_url="http://myhost:9999/v1", model="phi3")
        self.assertEqual(provider.base_url, "http://myhost:9999/v1")
        self.assertEqual(provider.model, "phi3")

    def test_default_name(self):
        from llm_providers import OllamaProvider
        provider = OllamaProvider()
        self.assertEqual(provider.name, "ollama")

    def test_custom_name(self):
        from llm_providers import OllamaProvider
        provider = OllamaProvider(name="ollama-local")
        self.assertEqual(provider.name, "ollama-local")

    def test_remote_with_api_key(self):
        from llm_providers import OllamaProvider
        provider = OllamaProvider(
            base_url="https://simonmccallum.org.nz/llm/v1",
            api_key="test-key",
            model="qwen2.5:7b",
            name="ollama-remote"
        )
        self.assertEqual(provider.base_url, "https://simonmccallum.org.nz/llm/v1")
        self.assertEqual(provider.api_key, "test-key")
        self.assertEqual(provider.model, "qwen2.5:7b")

    @patch('llm_providers.requests.get')
    def test_is_available_true(self, mock_get):
        from llm_providers import OllamaProvider
        mock_get.return_value = MagicMock(status_code=200)
        provider = OllamaProvider()
        self.assertTrue(provider.is_available())
        mock_get.assert_called_once_with(
            "http://localhost:11434/v1/models", timeout=5,
            headers={"Authorization": "Bearer ollama"}
        )

    @patch('llm_providers.requests.get')
    def test_is_available_false(self, mock_get):
        from llm_providers import OllamaProvider
        mock_get.side_effect = ConnectionError("refused")
        provider = OllamaProvider()
        self.assertFalse(provider.is_available())

    @patch('llm_providers.requests.get')
    def test_is_available_bad_status(self, mock_get):
        from llm_providers import OllamaProvider
        mock_get.return_value = MagicMock(status_code=500)
        provider = OllamaProvider()
        self.assertFalse(provider.is_available())


class TestAnthropicProvider(unittest.TestCase):
    """Test Anthropic provider with mocked client."""

    @patch('llm_providers.AnthropicProvider.__init__', return_value=None)
    def test_generate_prompt(self, mock_init):
        from llm_providers import AnthropicProvider
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.model = "claude-3-haiku-20240307"

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="  Generated prompt text  ")]
        provider.client = MagicMock()
        provider.client.messages.create.return_value = mock_message

        result = provider.generate_prompt("test chunk", "before", "after")
        self.assertEqual(result, "Generated prompt text")
        provider.client.messages.create.assert_called_once()


class TestOpenAIProvider(unittest.TestCase):
    """Test OpenAI provider with mocked client."""

    @patch('llm_providers.OpenAIProvider.__init__', return_value=None)
    def test_generate_prompt(self, mock_init):
        from llm_providers import OpenAIProvider
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.model = "gpt-3.5-turbo"

        mock_choice = MagicMock()
        mock_choice.message.content = "  Generated prompt text  "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        provider.client = MagicMock()
        provider.client.chat.completions.create.return_value = mock_response

        result = provider.generate_prompt("test chunk", "before", "after")
        self.assertEqual(result, "Generated prompt text")


class TestDiscoverProviders(unittest.TestCase):
    """Test provider discovery from environment."""

    @patch.dict(os.environ, {}, clear=True)
    def test_fallback_always_present(self):
        from llm_providers import discover_providers
        providers = discover_providers()
        self.assertIn('fallback', providers)

    @patch.dict(os.environ, {
        'OLLAMA_HOST': 'localhost',
        'OLLAMA_PORT': '11434',
        'OLLAMA_MODEL': 'phi3',
    }, clear=True)
    def test_discovers_ollama_local(self):
        from llm_providers import discover_providers
        providers = discover_providers()
        self.assertIn('ollama-local', providers)
        self.assertEqual(providers['ollama-local'].model, 'phi3')
        self.assertIn('fallback', providers)

    @patch.dict(os.environ, {
        'LOCAL_ONLY': '0',
        'OLLAMA_REMOTE_URL': 'https://simonmccallum.org.nz/llm/v1',
        'OLLAMA_REMOTE_API_KEY': 'test-key',
        'OLLAMA_REMOTE_MODEL': 'qwen2.5:7b',
    }, clear=True)
    def test_discovers_ollama_remote_when_local_only_off(self):
        from llm_providers import discover_providers
        providers = discover_providers()
        self.assertIn('ollama-remote', providers)
        self.assertEqual(providers['ollama-remote'].base_url, 'https://simonmccallum.org.nz/llm/v1')
        self.assertEqual(providers['ollama-remote'].model, 'qwen2.5:7b')

    @patch.dict(os.environ, {}, clear=True)
    def test_no_api_keys_only_fallback(self):
        from llm_providers import discover_providers
        providers = discover_providers()
        self.assertEqual(list(providers.keys()), ['fallback'])

    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'your_anthropic_api_key_here',
    }, clear=True)
    def test_placeholder_key_ignored(self):
        from llm_providers import discover_providers
        providers = discover_providers()
        self.assertNotIn('anthropic', providers)


class TestLocalOnlyMode(unittest.TestCase):
    """Test that LOCAL_ONLY mode gates cloud and remote providers."""

    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'real-looking-key',
        'OPENAI_API_KEY': 'real-looking-key',
        'OLLAMA_REMOTE_URL': 'https://example.com/llm/v1',
    }, clear=True)
    def test_local_only_default_excludes_cloud_and_remote(self):
        from llm_providers import discover_providers
        providers = discover_providers()
        self.assertNotIn('anthropic', providers)
        self.assertNotIn('openai', providers)
        self.assertNotIn('ollama-remote', providers)
        self.assertIn('fallback', providers)

    @patch.dict(os.environ, {
        'LOCAL_ONLY': '0',
        'OLLAMA_REMOTE_URL': 'https://example.com/llm/v1',
    }, clear=True)
    def test_local_only_off_allows_remote(self):
        from llm_providers import discover_providers
        providers = discover_providers()
        self.assertIn('ollama-remote', providers)

    @patch.dict(os.environ, {
        'LOCAL_ONLY': '1',
        'OLLAMA_HOST': 'localhost',
        'OLLAMA_REMOTE_URL': 'https://example.com/llm/v1',
    }, clear=True)
    def test_local_only_keeps_ollama_local_first(self):
        from llm_providers import discover_providers
        providers = discover_providers()
        # Insertion order = priority; ollama-local must precede fallback
        keys = list(providers.keys())
        self.assertEqual(keys[0], 'ollama-local')
        self.assertNotIn('ollama-remote', providers)

    @patch.dict(os.environ, {'LOCAL_ONLY': 'false'}, clear=True)
    def test_local_only_accepts_false_string(self):
        from llm_providers import _local_only_enabled
        self.assertFalse(_local_only_enabled())

    @patch.dict(os.environ, {}, clear=True)
    def test_local_only_default_on(self):
        from llm_providers import _local_only_enabled
        self.assertTrue(_local_only_enabled())


class TestPromptTemplates(unittest.TestCase):
    """Test shared prompt building."""

    def test_build_user_message_with_context(self):
        from llm_providers import _build_user_message
        msg = _build_user_message("target text", "before context", "after context")
        self.assertIn("target text", msg)
        self.assertIn("before context", msg)
        self.assertIn("after context", msg)

    def test_build_user_message_no_context(self):
        from llm_providers import _build_user_message
        msg = _build_user_message("target text", "", "")
        self.assertIn("[None]", msg)
        self.assertIn("target text", msg)

    def test_context_truncation(self):
        from llm_providers import _build_user_message
        long_before = "x" * 500
        long_after = "y" * 500
        msg = _build_user_message("target", long_before, long_after)
        # context_before should be truncated to last 200 chars
        self.assertNotIn("x" * 300, msg)
        # context_after should be truncated to first 200 chars
        self.assertNotIn("y" * 300, msg)


if __name__ == '__main__':
    unittest.main()

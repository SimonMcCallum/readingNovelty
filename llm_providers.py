"""
LLM Provider Abstraction

Provides a unified interface for different LLM backends used in novelty detection.
Supports Anthropic, OpenAI, Ollama (local and remote), and a fallback mode.
"""

import os
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert at analyzing text and creating prompts.
Given a piece of text and its surrounding context, generate a concise prompt that could
be used to regenerate that specific text. Focus on the key concepts, themes, and
information conveyed."""


def _build_user_message(chunk: str, context_before: str, context_after: str) -> str:
    """Build the standard user message for prompt generation."""
    return f"""Context before:
{context_before[-200:] if context_before else '[None]'}

Target text:
{chunk}

Context after:
{context_after[:200] if context_after else '[None]'}

Generate a concise prompt (2-3 sentences) that captures the essence of the target text
and could be used to regenerate it."""


class LLMProvider:
    """Base class for LLM providers."""

    name: str = "base"

    def generate_prompt(self, chunk: str, context_before: str = "",
                        context_after: str = "") -> str:
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if this provider is ready to use."""
        return True


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate_prompt(self, chunk: str, context_before: str = "",
                        context_after: str = "") -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _build_user_message(chunk, context_before, context_after)}
            ]
        )
        return message.content[0].text.strip()


class OpenAIProvider(LLMProvider):
    """OpenAI ChatGPT provider."""

    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_prompt(self, chunk: str, context_before: str = "",
                        context_after: str = "") -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(chunk, context_before, context_after)}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()


class OllamaProvider(LLMProvider):
    """Ollama provider using OpenAI-compatible API.

    Supports both local Ollama (http://host:port/v1) and remote instances
    behind a reverse proxy (e.g. https://host/llm/v1 with API key).
    """

    def __init__(self, base_url: str = "http://localhost:11434/v1",
                 api_key: str = "ollama", model: str = "llama3.2",
                 name: Optional[str] = None, timeout: int = 60):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.name = name or "ollama"
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self._client

    def is_available(self) -> bool:
        """Check if the endpoint is reachable."""
        try:
            # Try the models endpoint as a health check
            resp = requests.get(f"{self.base_url}/models", timeout=5,
                                headers={"Authorization": f"Bearer {self.api_key}"})
            return resp.status_code == 200
        except Exception:
            return False

    def generate_prompt(self, chunk: str, context_before: str = "",
                        context_after: str = "") -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(chunk, context_before, context_after)}
            ],
            max_tokens=200,
            temperature=0.7,
            timeout=self.timeout
        )
        return response.choices[0].message.content.strip()


class FallbackProvider(LLMProvider):
    """Fallback provider that extracts key words without an LLM."""

    name = "fallback"

    def generate_prompt(self, chunk: str, context_before: str = "",
                        context_after: str = "") -> str:
        words = chunk.split()[:50]
        return f"Write about: {' '.join(words)}"


def discover_providers() -> dict:
    """
    Discover all available LLM providers from environment configuration.

    Returns:
        Dict mapping provider name to LLMProvider instance.
        Priority order: anthropic, openai, ollama-local, ollama-remote, fallback.
    """
    providers = {}

    # Anthropic
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if anthropic_key and anthropic_key != 'your_anthropic_api_key_here':
        try:
            providers['anthropic'] = AnthropicProvider(api_key=anthropic_key)
            logger.info("Discovered Anthropic provider")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic provider: {e}")

    # OpenAI
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key and openai_key != 'your_openai_api_key_here':
        try:
            providers['openai'] = OpenAIProvider(api_key=openai_key)
            logger.info("Discovered OpenAI provider")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider: {e}")

    # Ollama local
    ollama_host = os.getenv('OLLAMA_HOST')
    if ollama_host:
        ollama_port = int(os.getenv('OLLAMA_PORT', '11434'))
        ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2')
        base_url = f"http://{ollama_host}:{ollama_port}/v1"
        provider = OllamaProvider(
            base_url=base_url, model=ollama_model,
            name='ollama-local', timeout=60
        )
        providers['ollama-local'] = provider
        logger.info(f"Discovered Ollama local provider at {base_url}")

    # Ollama remote (supports custom URL for reverse-proxied setups)
    ollama_remote_url = os.getenv('OLLAMA_REMOTE_URL')
    if ollama_remote_url:
        ollama_remote_model = os.getenv('OLLAMA_REMOTE_MODEL', 'qwen2.5:7b')
        ollama_remote_key = os.getenv('OLLAMA_REMOTE_API_KEY', 'ollama')
        provider = OllamaProvider(
            base_url=ollama_remote_url, api_key=ollama_remote_key,
            model=ollama_remote_model, name='ollama-remote', timeout=120
        )
        providers['ollama-remote'] = provider
        logger.info(f"Discovered Ollama remote provider at {ollama_remote_url}")

    # Fallback is always available
    providers['fallback'] = FallbackProvider()

    return providers

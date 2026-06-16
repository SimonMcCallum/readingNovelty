"""
LLM Provider Abstraction

Provides a unified interface for different LLM backends used in novelty detection.
Supports Anthropic, OpenAI, Ollama (local and remote), and a fallback mode.

LOCAL_ONLY mode (default): cloud providers and remote Ollama are excluded from
discovery so that copyrighted content never leaves the host. Set LOCAL_ONLY=0
only when processing material you are licensed to share with third parties.
"""

import os
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def _local_only_enabled() -> bool:
    """LOCAL_ONLY mode forbids cloud providers and remote Ollama. Default on."""
    return os.getenv('LOCAL_ONLY', '1').lower() not in ('0', 'false', 'no', '')


SYSTEM_PROMPT = """You are an expert at analyzing text and creating prompts.
Given a piece of text and its surrounding context, generate a concise prompt that could
be used to regenerate that specific text. Focus on the key concepts, themes, and
information conveyed."""

PREDICT_SYSTEM_PROMPT = """You are filling a missing passage in a longer document.
Given the text that came before, the text that came after, and a short topic hint,
write the most likely missing passage. Match the style and depth of the surrounding
text. Output the passage only — no preamble, no commentary, no quotation marks."""


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


def _build_predict_message(context_before: str, context_after: str,
                           hint: str, target_length_words: int = 150) -> str:
    """Build the cloze-prediction prompt: 'what fills the gap?'"""
    return f"""Context before:
{context_before[-400:] if context_before else '[None]'}

Context after:
{context_after[:400] if context_after else '[None]'}

Topic hint: {hint}

Write the missing passage of approximately {target_length_words} words. Output the
passage text only."""


class LLMProvider:
    """Base class for LLM providers."""

    name: str = "base"
    is_cloud: bool = False  # subclasses set True if calls leave the host

    def generate_prompt(self, chunk: str, context_before: str = "",
                        context_after: str = "") -> str:
        raise NotImplementedError

    def predict_chunk(self, context_before: str, context_after: str,
                      hint: str, target_length_words: int = 150) -> str:
        """
        Predict the most likely passage that fills the gap between
        context_before and context_after given a short topic hint.

        Used by the LLM-predictive novelty signal: a chunk is novel to the
        extent the LLM, given everything *around* it plus a thin hint, can't
        guess what's there.
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if this provider is ready to use."""
        return True


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    name = "anthropic"
    is_cloud = True

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

    def predict_chunk(self, context_before: str, context_after: str,
                      hint: str, target_length_words: int = 150) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max(400, target_length_words * 4),
            system=PREDICT_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _build_predict_message(
                    context_before, context_after, hint, target_length_words
                )},
            ],
        )
        return message.content[0].text.strip()


class OpenAIProvider(LLMProvider):
    """OpenAI ChatGPT provider."""

    name = "openai"
    is_cloud = True

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

    def predict_chunk(self, context_before: str, context_after: str,
                      hint: str, target_length_words: int = 150) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PREDICT_SYSTEM_PROMPT},
                {"role": "user", "content": _build_predict_message(
                    context_before, context_after, hint, target_length_words
                )},
            ],
            max_tokens=max(400, target_length_words * 4),
            temperature=0.5,
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
        # max_tokens=600 so gemma-style thinking models can emit visible
        # output after their internal reasoning budget. Llama, Qwen and other
        # non-thinking models will stop naturally well before this cap.
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(chunk, context_before, context_after)}
            ],
            max_tokens=600,
            temperature=0.7,
            timeout=self.timeout
        )
        return response.choices[0].message.content.strip()

    def predict_chunk(self, context_before: str, context_after: str,
                      hint: str, target_length_words: int = 150) -> str:
        client = self._get_client()
        # See note on generate_prompt above re thinking-model headroom.
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PREDICT_SYSTEM_PROMPT},
                {"role": "user", "content": _build_predict_message(
                    context_before, context_after, hint, target_length_words
                )},
            ],
            max_tokens=max(800, target_length_words * 6),
            temperature=0.5,
            timeout=self.timeout,
        )
        return response.choices[0].message.content.strip()


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the REST API directly (no SDK dependency).

    Free tier exists for gemini-2.5-flash-lite and gemini-2.5-flash. This
    provider is only discovered when LOCAL_ONLY=0 because requests leave the
    host; do not use for copyrighted content.
    """

    name = "gemini"
    is_cloud = True
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _generate(self, system: str, user: str, max_output_tokens: int,
                  temperature: float) -> str:
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
                "temperature": temperature,
            },
        }
        resp = requests.post(
            url, params={"key": self.api_key}, json=body, timeout=self.timeout
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Gemini API returned {resp.status_code}: {resp.text[:300]}"
            )
        payload = resp.json()
        try:
            return payload["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected Gemini response shape: {payload}") from e

    def is_available(self) -> bool:
        try:
            url = f"{self.BASE_URL}/models/{self.model}"
            resp = requests.get(
                url, params={"key": self.api_key}, timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

    def generate_prompt(self, chunk: str, context_before: str = "",
                        context_after: str = "") -> str:
        return self._generate(
            SYSTEM_PROMPT,
            _build_user_message(chunk, context_before, context_after),
            max_output_tokens=200, temperature=0.7,
        )

    def predict_chunk(self, context_before: str, context_after: str,
                      hint: str, target_length_words: int = 150) -> str:
        return self._generate(
            PREDICT_SYSTEM_PROMPT,
            _build_predict_message(context_before, context_after, hint, target_length_words),
            max_output_tokens=max(400, target_length_words * 4),
            temperature=0.5,
        )


class FallbackProvider(LLMProvider):
    """Fallback provider that extracts key words without an LLM."""

    name = "fallback"

    def generate_prompt(self, chunk: str, context_before: str = "",
                        context_after: str = "") -> str:
        words = chunk.split()[:50]
        return f"Write about: {' '.join(words)}"

    def predict_chunk(self, context_before: str, context_after: str,
                      hint: str, target_length_words: int = 150) -> str:
        # Deterministic, non-LLM filler — produces a reasonable embedding
        # neighbour to the context so cosine-distance novelty stays informative
        # in tests without a real LLM, but is poor signal in real use.
        joined = ' '.join(filter(None, [
            context_before.split('.')[-1].strip() if context_before else '',
            hint,
            context_after.split('.')[0].strip() if context_after else '',
        ]))
        return joined or hint


def discover_providers() -> dict:
    """
    Discover available LLM providers from environment configuration.

    In LOCAL_ONLY mode (default), only ollama-local and fallback are eligible;
    cloud providers and remote Ollama are skipped regardless of env config.

    Returns:
        Dict mapping provider name to LLMProvider instance. Insertion order
        reflects priority: ollama-local first when local-only, then fallback.
        When LOCAL_ONLY=0: anthropic, openai, ollama-local, ollama-remote, fallback.
    """
    providers = {}
    local_only = _local_only_enabled()

    # Ollama local — highest priority in LOCAL_ONLY mode
    ollama_host = os.getenv('OLLAMA_HOST')
    if ollama_host:
        ollama_port = int(os.getenv('OLLAMA_PORT', '11434'))
        ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2')
        ollama_timeout = int(os.getenv('OLLAMA_TIMEOUT', '60'))
        base_url = f"http://{ollama_host}:{ollama_port}/v1"
        provider = OllamaProvider(
            base_url=base_url, model=ollama_model,
            name='ollama-local', timeout=ollama_timeout
        )
        providers['ollama-local'] = provider
        logger.info(
            "Discovered Ollama local provider at %s (model=%s, timeout=%ds)",
            base_url, ollama_model, ollama_timeout,
        )

    # Cloud and remote providers — skipped entirely when LOCAL_ONLY is set
    if local_only:
        for var in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'OLLAMA_REMOTE_URL', 'GEMINI_API_KEY'):
            if os.getenv(var) and os.getenv(var) not in (
                'your_anthropic_api_key_here', 'your_openai_api_key_here',
                'your_gemini_api_key_here'
            ):
                logger.warning(
                    "LOCAL_ONLY=1: ignoring %s. Cloud/remote providers are "
                    "disabled to keep submissions on-host.", var
                )
    else:
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key and anthropic_key != 'your_anthropic_api_key_here':
            try:
                providers['anthropic'] = AnthropicProvider(api_key=anthropic_key)
                logger.info("Discovered Anthropic provider")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic provider: {e}")

        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and openai_key != 'your_openai_api_key_here':
            try:
                providers['openai'] = OpenAIProvider(api_key=openai_key)
                logger.info("Discovered OpenAI provider")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")

        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key and gemini_key != 'your_gemini_api_key_here':
            gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
            providers['gemini'] = GeminiProvider(api_key=gemini_key, model=gemini_model)
            logger.info(f"Discovered Gemini provider (model={gemini_model})")

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

    providers['fallback'] = FallbackProvider()

    return providers

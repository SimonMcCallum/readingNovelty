"""
Novelty Detection Module

Uses LLMs and FAISS embeddings to detect novelty in text chunks.

Embedding model selection
-------------------------
The sentence-embedding model is chosen, in order, from the constructor
argument, the EMBEDDING_MODEL environment variable, then the default
(all-MiniLM-L6-v2). Embeddings are always L2-normalised before they are
indexed or compared, so the squared-L2 distances FAISS returns are
2 - 2*cosine regardless of which model produced them. Without that step a
model that does not normalise its output (e.g. the e5 family) would push
every distance far above the novelty scale and saturate scores at 1.0.

Use `python embedding_models_cli.py list` to pull a current ranking of
similarity models from Hugging Face and `... check <model>` to verify one
loads locally before pointing EMBEDDING_MODEL at it.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from llm_providers import (
    LLMProvider, FallbackProvider, discover_providers, select_active_provider,
)

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
_TRUTHY = ('1', 'true', 'yes', 'on')


def resolve_embedding_model_name(explicit: Optional[str] = None) -> str:
    """Constructor argument > EMBEDDING_MODEL env var > project default."""
    return explicit or os.getenv('EMBEDDING_MODEL') or DEFAULT_EMBEDDING_MODEL


class Embedder:
    """Thin wrapper around SentenceTransformer that always normalises output.

    Exposes the two methods the rest of the code base relies on
    (`encode(texts, show_progress_bar=False)` and
    `get_sentence_embedding_dimension()`), so anything that accepted a raw
    SentenceTransformer keeps working.
    """

    def __init__(self, model_name: str, device: Optional[str] = None,
                 trust_remote_code: Optional[bool] = None):
        if trust_remote_code is None:
            trust_remote_code = os.getenv('EMBEDDING_TRUST_REMOTE_CODE', '0').lower() in _TRUTHY
        device = device or os.getenv('EMBEDDING_DEVICE') or None
        kwargs = {}
        if device:
            kwargs['device'] = device
        if trust_remote_code:
            kwargs['trust_remote_code'] = True
        self.name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)
        self.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))
        logger.info(
            "Embedding model: %s (dim=%d, max_seq_length=%s)",
            model_name, self.get_sentence_embedding_dimension(),
            getattr(self.model, 'max_seq_length', '?'),
        )

    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        return np.asarray(self.model.encode(
            list(texts), show_progress_bar=show_progress_bar,
            batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True,
        ), dtype='float32')

    def get_sentence_embedding_dimension(self) -> int:
        # sentence-transformers >= 5 renamed this; keep the old name as our API.
        getter = getattr(self.model, 'get_embedding_dimension', None) \
            or self.model.get_sentence_embedding_dimension
        return int(getter())

    @property
    def max_seq_length(self):
        return getattr(self.model, 'max_seq_length', None)


class NoveltyDetector:
    """Detects novelty in text using LLM and FAISS embeddings."""

    def __init__(self, embedding_model: Optional[str] = None,
                 provider: Optional[LLMProvider] = None):
        """
        Initialize novelty detector.

        Args:
            embedding_model: Sentence-transformers model id. None means use
                EMBEDDING_MODEL from the environment, else all-MiniLM-L6-v2.
            provider: Optional specific LLM provider to use. If None,
                discovers from env and picks the first reachable one
                (LLM_PROVIDER overrides).
        """
        self.embedding_model_name = resolve_embedding_model_name(embedding_model)
        self.embedding_model = Embedder(self.embedding_model_name)
        self.providers = discover_providers()

        if provider:
            self.active_provider = provider
        else:
            self.active_provider = select_active_provider(
                self.providers, preferred=os.getenv('LLM_PROVIDER') or None,
            )

        # Backwards compatibility
        self.llm_type = self.active_provider.name

        logger.info(f"Active provider: {self.active_provider.name}")
        logger.info(f"Available providers: {list(self.providers.keys())}")

    def generate_prompt_for_chunk(self, chunk: str, context_before: str = "",
                                  context_after: str = "",
                                  provider: Optional[LLMProvider] = None) -> str:
        """
        Generate a prompt that could be used to regenerate the given text chunk.

        Args:
            chunk: The text chunk to analyze
            context_before: Text that comes before the chunk
            context_after: Text that comes after the chunk
            provider: Optional provider override

        Returns:
            Generated prompt
        """
        use_provider = provider or self.active_provider
        try:
            return use_provider.generate_prompt(chunk, context_before, context_after)
        except Exception as e:
            logger.error(f"Error generating prompt with {use_provider.name}: {e}")
            fallback = self.providers.get('fallback', FallbackProvider())
            return fallback.generate_prompt(chunk, context_before, context_after)

    def calculate_novelty_score(self, chunk_embedding: np.ndarray,
                               all_embeddings: np.ndarray, chunk_index: int) -> float:
        """
        Calculate novelty score using FAISS similarity search.

        Args:
            chunk_embedding: Embedding of the current chunk
            all_embeddings: Embeddings of all chunks
            chunk_index: Index of current chunk

        Returns:
            Novelty score (0-1, where higher = more novel)
        """
        try:
            # Create FAISS index
            dimension = all_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(all_embeddings.astype('float32'))

            # Search for similar chunks (excluding itself)
            k = min(5, len(all_embeddings))  # Top 5 similar chunks
            distances, indices = index.search(chunk_embedding.reshape(1, -1).astype('float32'), k)

            # Filter out the chunk itself
            valid_distances = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != chunk_index:
                    valid_distances.append(dist)

            if not valid_distances:
                return 1.0  # Completely novel if no similar chunks

            # Calculate novelty: higher distance = more novel
            avg_distance = np.mean(valid_distances)

            # Convert distance to similarity score (0-1)
            # Using exponential decay: novelty = 1 - exp(-distance/scale)
            scale = 2.0
            novelty = 1.0 - np.exp(-avg_distance / scale)

            return float(min(1.0, max(0.0, novelty)))

        except Exception as e:
            logger.error(f"Error calculating novelty score: {e}")
            return 0.5  # Default to medium novelty on error

    def analyze_novelty(self, chunks: List[Dict],
                        provider: Optional[LLMProvider] = None) -> List[float]:
        """
        Analyze novelty of all chunks.

        Args:
            chunks: List of text chunks with metadata
            provider: Optional provider override for this analysis

        Returns:
            List of novelty scores (0-1)
        """
        try:
            use_provider = provider or self.active_provider
            logger.info(f"Analyzing novelty for {len(chunks)} chunks using {use_provider.name}")

            # Generate embeddings for all chunks
            texts = [chunk['text'] for chunk in chunks]
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

            novelty_scores = []

            # For each chunk, generate prompt and calculate novelty
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")

                # Get context
                context_before = chunks[i-1]['text'] if i > 0 else ""
                context_after = chunks[i+1]['text'] if i < len(chunks) - 1 else ""

                # Generate prompt (this helps identify key concepts)
                prompt = self.generate_prompt_for_chunk(
                    chunk['text'],
                    context_before,
                    context_after,
                    provider=use_provider
                )
                logger.debug(f"Generated prompt: {prompt[:100]}...")

                # Get embedding for the prompt
                prompt_embedding = self.embedding_model.encode([prompt])[0]

                # Calculate novelty by comparing prompt embedding to all chunk embeddings
                novelty_score = self.calculate_novelty_score(
                    prompt_embedding,
                    embeddings,
                    i
                )

                novelty_scores.append(novelty_score)
                logger.debug(f"Chunk {i+1} novelty score: {novelty_score:.2f}")

            logger.info("Novelty analysis complete")
            return novelty_scores

        except Exception as e:
            logger.error(f"Error analyzing novelty: {str(e)}", exc_info=True)
            return [0.5] * len(chunks)

    def embed_and_prompt_chunks(
        self, chunks: List[Dict], provider: Optional[LLMProvider] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate LLM prompts and embeddings for a list of chunks.

        Returns (embeddings, prompts) where embeddings are over the LLM-generated
        prompts (matching analyze_novelty's signal). Used by corpus-relative
        scoring so the caller can persist both into the corpus store.
        """
        use_provider = provider or self.active_provider
        prompts = []
        for i, chunk in enumerate(chunks):
            context_before = chunks[i - 1]['text'] if i > 0 else ""
            context_after = chunks[i + 1]['text'] if i < len(chunks) - 1 else ""
            prompts.append(
                self.generate_prompt_for_chunk(
                    chunk['text'], context_before, context_after, provider=use_provider
                )
            )
        embeddings = self.embedding_model.encode(prompts, show_progress_bar=False)
        return np.asarray(embeddings), prompts

    @property
    def embedding_dim(self) -> int:
        return self.embedding_model.get_sentence_embedding_dimension()

    _STOPWORDS = frozenset("""
        a an and are as at be by for from has have he her his i in is it its of
        on or that the their they this to was we were will with you your would
        which not but also if then than these those there been being but more
        most some such other any all can could should may might into about over
        between within without per via etc eg e.g. i.e. ie one two three four
        five also however therefore thus when where while because although
    """.split())

    @classmethod
    def keyword_hint(cls, text: str, n: int = 5) -> str:
        """Cheap, deterministic ~n-word topic hint (no LLM call).

        Used to feed predict_chunk with minimal information about the missing
        passage, so the predictive-novelty score measures what the LLM would
        write without seeing the target — only its topic and neighbours.
        """
        from collections import Counter
        words = [w.lower().strip('.,;:()[]"\'') for w in text.split()]
        content = [w for w in words if len(w) > 3 and w not in cls._STOPWORDS]
        if not content:
            return text[:40]
        top = [w for w, _ in Counter(content).most_common(n)]
        return ' '.join(top)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-10:
            return 0.0
        return float(np.dot(a, b) / denom)

    def analyze_llm_novelty(
        self, chunks: List[Dict], provider: Optional[LLMProvider] = None,
        hint_words: int = 5, max_workers: int = 4,
    ) -> List[float]:
        """
        LLM-predictive novelty: ask the LLM to fill the gap between the
        neighbours given a short hint, then score 1 - cosine(predicted, actual).

        High score = the LLM (with the hint and the neighbours) wrote something
        very different from the actual chunk — either because the chunk is
        novel/surprising or because it's wrong relative to expectations.

        max_workers controls how many predict_chunk calls run concurrently.
        For local Ollama this helps only up to Ollama's parallel-request
        capacity (`OLLAMA_NUM_PARALLEL`, default 1). For Gemini free tier,
        keep it <= 3 to avoid 429s. Pass max_workers=1 for serial behaviour.

        Returns one score per chunk in [0, 1].
        """
        use_provider = provider or self.active_provider
        if use_provider.name == 'fallback':
            logger.warning(
                "analyze_llm_novelty called with fallback provider; results "
                "will not reflect real LLM predictability."
            )

        actual_texts = [c['text'] for c in chunks]
        actual_embeddings = self.embedding_model.encode(actual_texts, show_progress_bar=False)

        def predict_one(idx_chunk: Tuple[int, Dict]) -> str:
            i, chunk = idx_chunk
            context_before = chunks[i - 1]['text'] if i > 0 else ""
            context_after = chunks[i + 1]['text'] if i < len(chunks) - 1 else ""
            hint = self.keyword_hint(chunk['text'], n=hint_words)
            try:
                return use_provider.predict_chunk(
                    context_before, context_after, hint,
                    target_length_words=max(50, len(chunk['text'].split())),
                )
            except Exception as e:
                logger.error("predict_chunk failed on chunk %d: %s", i, e)
                return hint  # degrade gracefully

        if max_workers <= 1 or len(chunks) <= 1:
            predictions = [predict_one((i, c)) for i, c in enumerate(chunks)]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                predictions = list(pool.map(predict_one, enumerate(chunks)))

        predicted_embeddings = self.embedding_model.encode(predictions, show_progress_bar=False)

        scores: List[float] = []
        for actual_emb, pred_emb in zip(actual_embeddings, predicted_embeddings):
            sim = self._cosine(actual_emb, pred_emb)
            scores.append(min(1.0, max(0.0, 1.0 - sim)))
        return scores

    @staticmethod
    def combine_novelty_scores(
        corpus_scores: List[float], llm_scores: List[float], alpha: float
    ) -> List[float]:
        """
        Blend per-chunk corpus novelty with LLM-predictive novelty.

        alpha=1.0 → pure corpus (current Phase B behaviour).
        alpha=0.0 → pure LLM predictability.
        alpha=0.5 → balanced.

        Lengths must match. Both score lists must already be in [0, 1].
        """
        if len(corpus_scores) != len(llm_scores):
            raise ValueError("corpus_scores and llm_scores length mismatch")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        return [
            alpha * c + (1.0 - alpha) * l
            for c, l in zip(corpus_scores, llm_scores)
        ]

    def analyze_novelty_multi(self, chunks: List[Dict],
                              provider_names: Optional[List[str]] = None) -> Dict[str, List[float]]:
        """
        Run novelty analysis with multiple providers for comparison.

        Args:
            chunks: List of text chunks with metadata
            provider_names: List of provider names to use. If None, uses all available.

        Returns:
            Dict mapping provider name to list of novelty scores.
        """
        if provider_names:
            providers_to_use = {
                name: self.providers[name]
                for name in provider_names
                if name in self.providers
            }
        else:
            providers_to_use = self.providers

        results = {}
        for name, provider in providers_to_use.items():
            logger.info(f"Running novelty analysis with provider: {name}")
            try:
                scores = self.analyze_novelty(chunks, provider=provider)
                results[name] = scores
            except Exception as e:
                logger.error(f"Error with provider {name}: {e}")
                results[name] = [0.5] * len(chunks)

        return results

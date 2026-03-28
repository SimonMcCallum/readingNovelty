"""
Novelty Detection Module

Uses LLMs and FAISS embeddings to detect novelty in text chunks.
"""

import logging
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from llm_providers import LLMProvider, FallbackProvider, discover_providers

logger = logging.getLogger(__name__)


class NoveltyDetector:
    """Detects novelty in text using LLM and FAISS embeddings."""

    def __init__(self, embedding_model='all-MiniLM-L6-v2', provider: Optional[LLMProvider] = None):
        """
        Initialize novelty detector.

        Args:
            embedding_model: Name of sentence transformer model to use
            provider: Optional specific LLM provider to use. If None, auto-discovers from env.
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.providers = discover_providers()

        if provider:
            self.active_provider = provider
        else:
            # Use first available provider (priority order from discover_providers)
            self.active_provider = next(iter(self.providers.values()))

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
            embeddings = np.array(embeddings)

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

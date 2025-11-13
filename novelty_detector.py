"""
Novelty Detection Module

Uses LLMs and FAISS embeddings to detect novelty in text chunks.
"""

import os
import logging
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class NoveltyDetector:
    """Detects novelty in text using LLM and FAISS embeddings."""
    
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize novelty detector.
        
        Args:
            embedding_model: Name of sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.llm_client = None
        self._initialize_llm_client()
        
    def _initialize_llm_client(self):
        """Initialize LLM client based on available API keys."""
        # Try Anthropic first
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key and anthropic_key != 'your_anthropic_api_key_here':
            try:
                from anthropic import Anthropic
                self.llm_client = Anthropic(api_key=anthropic_key)
                self.llm_type = 'anthropic'
                logger.info("Initialized Anthropic client")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
        
        # Try OpenAI
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and openai_key != 'your_openai_api_key_here':
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=openai_key)
                self.llm_type = 'openai'
                logger.info("Initialized OpenAI client")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        logger.warning("No LLM client initialized - using fallback mode")
        self.llm_type = 'fallback'
    
    def generate_prompt_for_chunk(self, chunk: str, context_before: str = "", 
                                  context_after: str = "") -> str:
        """
        Generate a prompt that could be used to regenerate the given text chunk.
        
        Args:
            chunk: The text chunk to analyze
            context_before: Text that comes before the chunk
            context_after: Text that comes after the chunk
            
        Returns:
            Generated prompt
        """
        if self.llm_type == 'anthropic':
            return self._generate_prompt_anthropic(chunk, context_before, context_after)
        elif self.llm_type == 'openai':
            return self._generate_prompt_openai(chunk, context_before, context_after)
        else:
            return self._generate_prompt_fallback(chunk, context_before, context_after)
    
    def _generate_prompt_anthropic(self, chunk: str, context_before: str, 
                                   context_after: str) -> str:
        """Generate prompt using Anthropic Claude."""
        try:
            system_prompt = """You are an expert at analyzing text and creating prompts. 
Given a piece of text and its surrounding context, generate a concise prompt that could 
be used to regenerate that specific text. Focus on the key concepts, themes, and 
information conveyed."""
            
            user_message = f"""Context before:
{context_before[-200:] if context_before else '[None]'}

Target text:
{chunk}

Context after:
{context_after[:200] if context_after else '[None]'}

Generate a concise prompt (2-3 sentences) that captures the essence of the target text 
and could be used to regenerate it."""
            
            message = self.llm_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating prompt with Anthropic: {e}")
            return self._generate_prompt_fallback(chunk, context_before, context_after)
    
    def _generate_prompt_openai(self, chunk: str, context_before: str, 
                               context_after: str) -> str:
        """Generate prompt using OpenAI."""
        try:
            system_prompt = """You are an expert at analyzing text and creating prompts. 
Given a piece of text and its surrounding context, generate a concise prompt that could 
be used to regenerate that specific text. Focus on the key concepts, themes, and 
information conveyed."""
            
            user_message = f"""Context before:
{context_before[-200:] if context_before else '[None]'}

Target text:
{chunk}

Context after:
{context_after[:200] if context_after else '[None]'}

Generate a concise prompt (2-3 sentences) that captures the essence of the target text 
and could be used to regenerate it."""
            
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating prompt with OpenAI: {e}")
            return self._generate_prompt_fallback(chunk, context_before, context_after)
    
    def _generate_prompt_fallback(self, chunk: str, context_before: str, 
                                 context_after: str) -> str:
        """Fallback prompt generation without LLM."""
        # Simple extraction of key information
        words = chunk.split()[:50]  # First 50 words
        return f"Write about: {' '.join(words)}"
    
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
            # Normalize distance to 0-1 range
            avg_distance = np.mean(valid_distances)
            
            # Convert distance to similarity score (0-1)
            # Using exponential decay: novelty = 1 - exp(-distance/scale)
            scale = 2.0
            novelty = 1.0 - np.exp(-avg_distance / scale)
            
            return float(min(1.0, max(0.0, novelty)))
            
        except Exception as e:
            logger.error(f"Error calculating novelty score: {e}")
            return 0.5  # Default to medium novelty on error
    
    def analyze_novelty(self, chunks: List[Dict]) -> List[float]:
        """
        Analyze novelty of all chunks.
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            List of novelty scores (0-1)
        """
        try:
            logger.info(f"Analyzing novelty for {len(chunks)} chunks")
            
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
                    context_after
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
            # Return default scores on error
            return [0.5] * len(chunks)

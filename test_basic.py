"""
Simple test script to verify the basic functionality
"""

import sys
import os

# Test imports
print("Testing imports...")
try:
    from pdf_processor import PDFProcessor
    from novelty_detector import NoveltyDetector
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test PDF processor
print("\nTesting PDF processor...")
try:
    processor = PDFProcessor()
    
    # Test chunking
    test_text = """
    This is the first paragraph. It contains some text that will be chunked.
    We want to make sure that the chunking algorithm works properly.
    This paragraph is designed to be around 100 words or more.
    
    This is the second paragraph. It also contains text for testing.
    The chunking should properly separate these paragraphs into meaningful chunks.
    We need enough text here to test the algorithm properly.
    """ * 3  # Repeat to get enough content
    
    chunks = processor.chunk_text(test_text)
    print(f"✓ Chunking works - created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk['word_count']} words")
    
except Exception as e:
    print(f"✗ PDF processor error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test novelty detector
print("\nTesting novelty detector...")
try:
    detector = NoveltyDetector()
    print(f"✓ Novelty detector initialized with LLM type: {detector.llm_type}")
    
    # Test with simple chunks
    test_chunks = [
        {'text': 'This is a unique piece of text about quantum physics.', 'word_count': 10},
        {'text': 'This is another unique text about machine learning algorithms.', 'word_count': 9},
        {'text': 'Similar text about machine learning and algorithms.', 'word_count': 7}
    ]
    
    print("  Running novelty analysis on test chunks...")
    novelty_scores = detector.analyze_novelty(test_chunks)
    print(f"✓ Novelty analysis works - got {len(novelty_scores)} scores")
    
    for i, score in enumerate(novelty_scores):
        print(f"  Chunk {i+1}: novelty score = {score:.2f}")
    
except Exception as e:
    print(f"✗ Novelty detector error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
print("\nNote: For full functionality, configure API keys in .env file")
print("  - ANTHROPIC_API_KEY for Claude")
print("  - OPENAI_API_KEY for ChatGPT")

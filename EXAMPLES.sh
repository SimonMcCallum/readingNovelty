"""
Quick Start Guide for PDF Novelty Detection Server
"""

# Example 1: Health check
curl http://localhost:5000/health

# Example 2: Upload and analyze a PDF
curl -X POST -F "file=@example.pdf" http://localhost:5000/upload

# Example 3: Analyze text directly (for testing)
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is the first paragraph with some interesting content. It talks about artificial intelligence and machine learning. This is a novel concept that has not been discussed before in this document. This is the second paragraph. It contains different information about neural networks and deep learning architectures. This paragraph introduces new concepts about transformers and attention mechanisms."
  }'

# Example 4: Download annotated PDF
curl -O http://localhost:5000/download/annotated_example.pdf

# Example Response from /upload:
# {
#   "success": true,
#   "original_filename": "example.pdf",
#   "annotated_filename": "annotated_example.pdf",
#   "chunks_analyzed": 15,
#   "novelty_scores": [
#     {
#       "chunk_index": 0,
#       "text_preview": "Introduction to the topic...",
#       "novelty_score": 0.85
#     },
#     ...
#   ],
#   "download_url": "/download/annotated_example.pdf"
# }

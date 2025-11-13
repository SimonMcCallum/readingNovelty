# readingNovelty

A PDF novelty detection server that analyzes PDFs and annotates them based on the novelty of each paragraph.

## Overview

This server allows you to:
1. Upload a PDF document
2. Automatically chunk the text into ~100-200 word segments
3. Analyze each chunk for novelty using LLM-based prompt generation and FAISS embeddings
4. Download an annotated PDF with color-coded novelty scores

## Features

- **PDF Upload & Processing**: Upload PDF files for automatic text extraction
- **Smart Chunking**: Intelligently splits text into ~100-200 word chunks
- **LLM-Based Analysis**: Uses Claude/ChatGPT to generate prompts that capture the essence of each text chunk
- **FAISS Similarity Search**: Compares embeddings to determine novelty
- **Annotated PDFs**: Downloads PDFs with color-coded novelty indicators
- **REST API**: Simple HTTP endpoints for integration

## Novelty Detection Algorithm

The system uses the following approach:

1. **Text Chunking**: Break PDF into sections of ~100-200 words
2. **Prompt Generation**: For each chunk, use an LLM (Claude/ChatGPT/Gemini) to analyze the text with surrounding context and generate a prompt that could regenerate that text
3. **Embedding Generation**: Create embeddings for all chunks using sentence transformers
4. **Similarity Search**: Use FAISS to find similar chunks and calculate novelty based on uniqueness
5. **Scoring**: Higher novelty = less similar to other chunks in the document

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SimonMcCallum/readingNovelty.git
cd readingNovelty
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

You need at least one of:
- `ANTHROPIC_API_KEY` for Claude
- `OPENAI_API_KEY` for ChatGPT

4. Run the server:
```bash
python server.py
```

The server will start on `http://localhost:5000`

## API Usage

### Health Check

```bash
curl http://localhost:5000/health
```

### Upload and Analyze PDF

```bash
curl -X POST -F "file=@document.pdf" http://localhost:5000/upload
```

Response:
```json
{
  "success": true,
  "original_filename": "document.pdf",
  "annotated_filename": "annotated_document.pdf",
  "chunks_analyzed": 15,
  "novelty_scores": [
    {
      "chunk_index": 0,
      "text_preview": "Introduction text...",
      "novelty_score": 0.75
    }
  ],
  "download_url": "/download/annotated_document.pdf"
}
```

### Download Annotated PDF

```bash
curl -O http://localhost:5000/download/annotated_document.pdf
```

### Analyze Text Directly (Testing)

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here..."}'
```

## Novelty Score Legend

- **High Novelty (>0.7)**: Green - Unique content, highly novel
- **Medium Novelty (0.4-0.7)**: Yellow - Moderately novel content
- **Low Novelty (0.2-0.4)**: Orange - Somewhat repetitive content
- **Very Low Novelty (<0.2)**: Red - Highly repetitive or common content

## Architecture

- `server.py`: Flask web server with REST API endpoints
- `pdf_processor.py`: PDF text extraction, chunking, and annotation
- `novelty_detector.py`: LLM-based prompt generation and FAISS similarity analysis
- `requirements.txt`: Python dependencies
- `.env`: Configuration and API keys (not tracked in git)

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Linting

```bash
python -m pylint *.py
```

## Configuration

Environment variables (set in `.env`):

- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `OPENAI_API_KEY`: Your OpenAI API key
- `FLASK_ENV`: Environment (development/production)
- `FLASK_DEBUG`: Enable debug mode (1/0)
- `UPLOAD_FOLDER`: Directory for uploaded files (default: uploads)
- `MAX_CONTENT_LENGTH`: Max upload size in bytes (default: 16MB)

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

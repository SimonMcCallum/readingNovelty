"""
PDF Novelty Detection Server

This server allows users to upload PDFs, analyze them for novelty,
and download annotated versions with novelty scores.
"""

import os
import logging
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from novelty_detector import NoveltyDetector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize processors
pdf_processor = PDFProcessor()
novelty_detector = NoveltyDetector()

ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'PDF Novelty Detection Server is running'})


@app.route('/upload', methods=['POST'])
def upload_pdf():
    """
    Upload and analyze a PDF file.
    
    Returns:
        JSON with analysis results and download link for annotated PDF
    """
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved: {filepath}")
        
        # Extract text from PDF
        logger.info("Extracting text from PDF...")
        text_chunks = pdf_processor.extract_and_chunk_text(filepath)
        logger.info(f"Extracted {len(text_chunks)} text chunks")
        
        # Analyze novelty
        logger.info("Analyzing novelty...")
        novelty_scores = novelty_detector.analyze_novelty(text_chunks)
        logger.info("Novelty analysis complete")
        
        # Create annotated PDF
        logger.info("Creating annotated PDF...")
        annotated_filename = f"annotated_{filename}"
        annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        pdf_processor.create_annotated_pdf(filepath, text_chunks, novelty_scores, annotated_filepath)
        logger.info(f"Annotated PDF created: {annotated_filepath}")
        
        # Prepare response
        response = {
            'success': True,
            'original_filename': filename,
            'annotated_filename': annotated_filename,
            'chunks_analyzed': len(text_chunks),
            'novelty_scores': [
                {
                    'chunk_index': i,
                    'text_preview': chunk['text'][:100] + '...' if len(chunk['text']) > 100 else chunk['text'],
                    'novelty_score': score
                }
                for i, (chunk, score) in enumerate(zip(text_chunks, novelty_scores))
            ],
            'download_url': f'/download/{annotated_filename}'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """
    Download an annotated PDF file.
    
    Args:
        filename: Name of the file to download
        
    Returns:
        PDF file
    """
    try:
        filename = secure_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500


@app.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Analyze text directly without PDF upload (for testing).
    
    Expects JSON with 'text' field.
    
    Returns:
        JSON with novelty analysis
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        # Chunk the text
        chunks = pdf_processor.chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks from text")
        
        # Analyze novelty
        novelty_scores = novelty_detector.analyze_novelty(chunks)
        
        response = {
            'success': True,
            'chunks_analyzed': len(chunks),
            'novelty_scores': [
                {
                    'chunk_index': i,
                    'text': chunk['text'],
                    'novelty_score': score
                }
                for i, (chunk, score) in enumerate(zip(chunks, novelty_scores))
            ]
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error analyzing text: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', '0') == '1')

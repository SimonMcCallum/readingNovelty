"""
PDF Novelty Detection Server

This server allows users to upload PDFs, analyze them for novelty,
and download annotated versions with novelty scores.
"""

import os
import logging
from flask import Flask, request, jsonify, send_file, render_template_string
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from novelty_detector import NoveltyDetector
from llm_providers import discover_providers, _local_only_enabled
from corpus import CorpusStore

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

CORPUS_DIR = os.getenv('CORPUS_DIR', 'corpus')
corpus_store = CorpusStore(CORPUS_DIR, embedding_dim=novelty_detector.embedding_dim)

LOCAL_ONLY = _local_only_enabled()


def _log_privacy_banner():
    """Log the active privacy posture and provider so it is obvious in startup logs."""
    active = novelty_detector.active_provider.name
    if LOCAL_ONLY:
        logger.info("PRIVACY: LOCAL_ONLY=1 — cloud and remote providers disabled.")
    else:
        logger.warning(
            "PRIVACY: LOCAL_ONLY=0 — cloud providers may process submissions. "
            "Do not use for copyrighted content."
        )
    logger.info("Active provider: %s", active)
    if active == 'fallback':
        logger.warning(
            "No LLM provider is reachable; using keyword fallback. Assessment "
            "endpoints will refuse to run in this state."
        )


def _assessment_provider_ready() -> bool:
    """True only when a real LLM provider is active (fallback is not acceptable)."""
    active = novelty_detector.active_provider
    if active.name == 'fallback':
        return False
    return active.is_available()


_log_privacy_banner()

ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint. Reports privacy posture and assessment readiness."""
    return jsonify({
        'status': 'healthy',
        'message': 'PDF Novelty Detection Server is running',
        'local_only': LOCAL_ONLY,
        'active_provider': novelty_detector.active_provider.name,
        'assessment_ready': _assessment_provider_ready(),
    })


@app.route('/providers', methods=['GET'])
def list_providers():
    """List available LLM providers and their status."""
    providers_info = {}
    for name, provider in novelty_detector.providers.items():
        providers_info[name] = {
            'name': name,
            'available': provider.is_available(),
            'active': name == novelty_detector.active_provider.name,
        }
        if hasattr(provider, 'model'):
            providers_info[name]['model'] = provider.model
        if hasattr(provider, 'base_url'):
            providers_info[name]['base_url'] = provider.base_url
    return jsonify({
        'local_only': LOCAL_ONLY,
        'providers': providers_info,
    }), 200


@app.route('/compare', methods=['POST'])
def compare_providers():
    """
    Run novelty analysis with multiple providers for comparison.

    Accepts JSON with 'text' field and optional 'providers' list.
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        provider_names = data.get('providers', None)

        chunks = pdf_processor.chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks for comparison")

        results = novelty_detector.analyze_novelty_multi(chunks, provider_names)

        response = {
            'success': True,
            'chunks_analyzed': len(chunks),
            'chunks': [
                {'chunk_index': i, 'text_preview': c['text'][:100]}
                for i, c in enumerate(chunks)
            ],
            'providers': {
                name: {
                    'scores': scores,
                    'avg': sum(scores) / len(scores) if scores else 0
                }
                for name, scores in results.items()
            }
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in comparison: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error during comparison. Please try again.'}), 500


@app.route('/assignments', methods=['POST'])
def create_assignment():
    """Register an assignment so submissions can be scored against its cohort."""
    data = request.get_json(silent=True) or {}
    assignment_id = data.get('assignment_id')
    name = data.get('name')
    if not assignment_id:
        return jsonify({'error': 'assignment_id is required'}), 400
    corpus_store.ensure_assignment(assignment_id, name=name)
    return jsonify({'success': True, 'assignment': corpus_store.get_assignment(assignment_id)}), 201


@app.route('/assignments/<assignment_id>', methods=['GET'])
def get_assignment(assignment_id):
    """Return assignment metadata and submission count."""
    info = corpus_store.get_assignment(assignment_id)
    if info is None:
        return jsonify({'error': 'Assignment not found'}), 404
    return jsonify(info), 200


@app.route('/assignments/<assignment_id>/submissions', methods=['POST'])
def submit_to_assignment(assignment_id):
    """
    Score a submission PDF against the assignment cohort and append it.

    Form fields: file (PDF), student_id (optional), submission_id (optional).
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    if not _assessment_provider_ready():
        return jsonify({
            'error': 'No LLM provider is available for assessment.',
            'detail': f"Active provider is '{novelty_detector.active_provider.name}'.",
        }), 503

    student_id = request.form.get('student_id')
    submission_id = request.form.get('submission_id') or f"{assignment_id}:{secure_filename(file.filename)}"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        chunks = pdf_processor.extract_and_chunk_text(filepath)
        if not chunks:
            return jsonify({'error': 'PDF contained no extractable text'}), 400

        corpus_store.ensure_assignment(assignment_id)
        embeddings, prompts = novelty_detector.embed_and_prompt_chunks(chunks)

        # Score against the prior corpus (excluding any earlier version of
        # this same submission_id, so a re-upload doesn't score itself).
        novelty_scores = corpus_store.score_against_corpus(
            assignment_id, embeddings, exclude_submission_id=submission_id
        )

        for chunk, prompt in zip(chunks, prompts):
            chunk['prompt'] = prompt

        record = corpus_store.add_submission(
            assignment_id=assignment_id,
            submission_id=submission_id,
            student_id=student_id,
            filename=filename,
            chunks=chunks,
            embeddings=embeddings,
            novelty_scores=novelty_scores,
        )

        annotated_filename = f"annotated_{submission_id}_{filename}"
        annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        pdf_processor.create_annotated_pdf(filepath, chunks, novelty_scores, annotated_filepath)

        return jsonify({
            'success': True,
            'submission': record,
            'novelty_scores': [
                {
                    'chunk_index': i,
                    'text_preview': chunk['text'][:100],
                    'novelty_score': score,
                }
                for i, (chunk, score) in enumerate(zip(chunks, novelty_scores))
            ],
            'download_url': f'/download/{annotated_filename}',
            'corpus_priors': corpus_store.get_assignment(assignment_id)['chunk_count'] - len(chunks),
        }), 200

    except Exception as e:
        logger.error(f"Error processing submission: {e}", exc_info=True)
        return jsonify({'error': 'Error processing submission.'}), 500


@app.route('/assignments/<assignment_id>/submissions', methods=['GET'])
def list_submissions(assignment_id):
    """List submissions for an assignment, ordered most-recent first."""
    if corpus_store.get_assignment(assignment_id) is None:
        return jsonify({'error': 'Assignment not found'}), 404
    return jsonify({
        'assignment_id': assignment_id,
        'submissions': corpus_store.list_submissions(assignment_id),
    }), 200


READER_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Novelty: {{ assignment.assignment_id }}</title>
<style>
 body { font-family: -apple-system, Segoe UI, sans-serif; max-width: 960px; margin: 2em auto; color: #222; }
 h1 { margin-bottom: 0.2em; }
 .meta { color: #666; margin-bottom: 1.5em; }
 table { border-collapse: collapse; width: 100%; }
 th, td { padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }
 th { background: #f7f7f8; font-weight: 600; }
 .score { font-variant-numeric: tabular-nums; font-weight: 600; padding: 2px 6px; border-radius: 3px; }
 .s-high { background: #d4f5d4; color: #145214; }
 .s-med { background: #fff6cc; color: #6a5400; }
 .s-low { background: #ffe0c2; color: #7a3a00; }
 .s-vlow { background: #ffd2d2; color: #800000; }
 a { color: #0a58ca; text-decoration: none; }
 a:hover { text-decoration: underline; }
 .empty { color: #999; font-style: italic; }
</style>
</head>
<body>
<h1>{{ assignment.name or assignment.assignment_id }}</h1>
<div class="meta">
 Assignment ID: <code>{{ assignment.assignment_id }}</code> &middot;
 {{ assignment.submission_count }} submissions &middot;
 {{ assignment.chunk_count }} chunks in corpus
</div>
{% if submissions %}
<table>
 <tr><th>Submitted</th><th>Student</th><th>Filename</th><th>Chunks</th><th>Avg novelty</th><th>PDF</th></tr>
 {% for s in submissions %}
 <tr>
  <td>{{ s.submitted_at }}</td>
  <td>{{ s.student_id or '-' }}</td>
  <td><code>{{ s.filename }}</code></td>
  <td>{{ s.chunk_count }}</td>
  <td>
   {% set sc = s.avg_novelty %}
   {% if sc >= 0.7 %}<span class="score s-high">{{ '%.2f'|format(sc) }}</span>
   {% elif sc >= 0.4 %}<span class="score s-med">{{ '%.2f'|format(sc) }}</span>
   {% elif sc >= 0.2 %}<span class="score s-low">{{ '%.2f'|format(sc) }}</span>
   {% else %}<span class="score s-vlow">{{ '%.2f'|format(sc) }}</span>{% endif %}
  </td>
  <td><a href="/download/annotated_{{ s.submission_id }}_{{ s.filename }}">annotated</a></td>
 </tr>
 {% endfor %}
</table>
{% else %}
<p class="empty">No submissions yet.</p>
{% endif %}
</body>
</html>"""


@app.route('/assignments/<assignment_id>/reader', methods=['GET'])
def reader_view(assignment_id):
    """Grader-facing HTML view: cohort ranked by submission time with novelty + PDF links."""
    assignment = corpus_store.get_assignment(assignment_id)
    if assignment is None:
        return jsonify({'error': 'Assignment not found'}), 404
    submissions = corpus_store.list_submissions(assignment_id)
    return render_template_string(
        READER_TEMPLATE, assignment=assignment, submissions=submissions
    )


@app.route('/assignments/<assignment_id>/submissions/<submission_id>', methods=['GET'])
def get_submission_detail(assignment_id, submission_id):
    """Return per-chunk novelty detail for a single submission."""
    sub = corpus_store.get_submission(submission_id)
    if sub is None or sub['assignment_id'] != assignment_id:
        return jsonify({'error': 'Submission not found'}), 404
    return jsonify(sub), 200


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

    if not _assessment_provider_ready():
        return jsonify({
            'error': 'No LLM provider is available for assessment.',
            'detail': (
                f"Active provider is '{novelty_detector.active_provider.name}'. "
                'In LOCAL_ONLY mode, configure OLLAMA_HOST and ensure Ollama is reachable.'
            ),
        }), 503

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

        # Select provider (optional query param)
        provider_name = request.args.get('provider')
        provider = novelty_detector.providers.get(provider_name) if provider_name else None

        # Analyze novelty
        logger.info("Analyzing novelty...")
        novelty_scores = novelty_detector.analyze_novelty(text_chunks, provider=provider)
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
        return jsonify({'error': 'Error processing PDF. Please check the file and try again.'}), 500


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
        return jsonify({'error': 'Error downloading file. Please try again.'}), 500


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

        # Select provider (optional query param)
        provider_name = request.args.get('provider')
        provider = novelty_detector.providers.get(provider_name) if provider_name else None

        # Analyze novelty
        novelty_scores = novelty_detector.analyze_novelty(chunks, provider=provider)
        
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
        return jsonify({'error': 'Error analyzing text. Please check the input and try again.'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', '0') == '1')

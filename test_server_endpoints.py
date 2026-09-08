"""
Flask integration tests for the assignment endpoints.

Mocks the LLM provider with a deterministic stub and points corpus_store
+ UPLOAD_FOLDER at a temp directory, so the tests run without Ollama and
do not pollute the dev corpus.
"""

import os
import shutil
import tempfile
import unittest
from io import BytesIO

import fitz


# Ensure local-only mode + no real providers BEFORE importing server.
os.environ.setdefault('LOCAL_ONLY', '1')
for _k in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY',
           'OLLAMA_HOST', 'OLLAMA_REMOTE_URL'):
    os.environ.pop(_k, None)


def _build_pdf_bytes(paragraphs):
    """Build a small in-memory PDF and return its bytes."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 60
    for para in paragraphs:
        s = para
        while s:
            page.insert_text((50, y), s[:80], fontsize=11, fontname="helv")
            s = s[80:]
            y += 16
            if y > 780:
                page = doc.new_page(width=595, height=842)
                y = 60
        y += 14
    data = doc.tobytes()
    doc.close()
    return data


class _StubProvider:
    """Deterministic stub that satisfies the assessment-ready check."""
    name = "stub"
    is_cloud = False

    def generate_prompt(self, chunk, context_before="", context_after=""):
        return chunk[:60]

    def predict_chunk(self, context_before, context_after, hint, target_length_words=150):
        return hint

    def is_available(self):
        return True


class TestAssignmentEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import server
        from corpus import CorpusStore

        cls.tmp = tempfile.mkdtemp(prefix='server_endpoints_')
        cls.client = server.app.test_client()
        cls.server = server

        # Swap in a stub provider that always satisfies _assessment_provider_ready()
        cls._orig_provider = server.novelty_detector.active_provider
        server.novelty_detector.active_provider = _StubProvider()

        # Redirect corpus + uploads to the temp dir for the entire class
        cls._orig_corpus = server.corpus_store
        server.corpus_store = CorpusStore(
            os.path.join(cls.tmp, 'corpus'),
            embedding_dim=server.novelty_detector.embedding_dim,
        )
        cls._orig_upload = server.app.config['UPLOAD_FOLDER']
        server.app.config['UPLOAD_FOLDER'] = os.path.join(cls.tmp, 'uploads')
        os.makedirs(server.app.config['UPLOAD_FOLDER'], exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.server.novelty_detector.active_provider = cls._orig_provider
        cls.server.corpus_store = cls._orig_corpus
        cls.server.app.config['UPLOAD_FOLDER'] = cls._orig_upload
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # --- /assignments registration ---

    def test_create_assignment_requires_id(self):
        resp = self.client.post('/assignments', json={})
        self.assertEqual(resp.status_code, 400)

    def test_create_and_fetch_assignment(self):
        resp = self.client.post(
            '/assignments',
            json={'assignment_id': 'asg-create-1', 'name': 'Essay 1'},
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.get_json()
        self.assertTrue(body['success'])
        self.assertEqual(body['assignment']['assignment_id'], 'asg-create-1')
        self.assertEqual(body['assignment']['name'], 'Essay 1')
        self.assertEqual(body['assignment']['submission_count'], 0)

        get_resp = self.client.get('/assignments/asg-create-1')
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.get_json()['name'], 'Essay 1')

    def test_get_unknown_assignment_404(self):
        resp = self.client.get('/assignments/does-not-exist')
        self.assertEqual(resp.status_code, 404)

    # --- /submissions ingestion ---

    def test_submit_requires_file(self):
        self.client.post('/assignments', json={'assignment_id': 'asg-sub-1'})
        resp = self.client.post('/assignments/asg-sub-1/submissions')
        self.assertEqual(resp.status_code, 400)

    def test_submit_rejects_non_pdf(self):
        self.client.post('/assignments', json={'assignment_id': 'asg-sub-2'})
        resp = self.client.post(
            '/assignments/asg-sub-2/submissions',
            data={'file': (BytesIO(b'not a pdf'), 'note.txt')},
            content_type='multipart/form-data',
        )
        self.assertEqual(resp.status_code, 400)

    def test_submit_returns_503_when_provider_is_fallback(self):
        from llm_providers import FallbackProvider
        prior = self.server.novelty_detector.active_provider
        self.server.novelty_detector.active_provider = FallbackProvider()
        try:
            self.client.post('/assignments', json={'assignment_id': 'asg-sub-3'})
            resp = self.client.post(
                '/assignments/asg-sub-3/submissions',
                data={'file': (BytesIO(_build_pdf_bytes(['Some text. ' * 30])),
                               'doc.pdf')},
                content_type='multipart/form-data',
            )
            self.assertEqual(resp.status_code, 503)
            body = resp.get_json()
            self.assertIn('No LLM provider', body['error'])
        finally:
            self.server.novelty_detector.active_provider = prior

    def test_submit_scores_first_submission_as_max_novelty(self):
        self.client.post('/assignments', json={'assignment_id': 'asg-sub-4'})
        pdf = _build_pdf_bytes(['Quantum entanglement underpins cryptography. ' * 6])
        resp = self.client.post(
            '/assignments/asg-sub-4/submissions',
            data={
                'file': (BytesIO(pdf), 'alice.pdf'),
                'student_id': 'alice',
                'submission_id': 'asg-sub-4:alice',
            },
            content_type='multipart/form-data',
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body['success'])
        self.assertEqual(body['submission']['submission_id'], 'asg-sub-4:alice')
        # First submitter has an empty prior corpus -> all chunks = 1.0
        for entry in body['novelty_scores']:
            self.assertAlmostEqual(entry['novelty_score'], 1.0, places=5)
        self.assertIn('/download/', body['download_url'])

    def test_default_submission_id_yields_downloadable_annotated_pdf(self):
        """Default submission_id is '<assignment>:<file>'. The ':' must not leak
        into the on-disk name (Windows alternate data stream; and the download
        route strips it), and the reader view must link to the same name."""
        self.client.post('/assignments', json={'assignment_id': 'asg-sub-colon'})
        pdf = _build_pdf_bytes(['Photosynthesis converts light into chemical energy. ' * 6])
        resp = self.client.post(
            '/assignments/asg-sub-colon/submissions',
            data={'file': (BytesIO(pdf), 'bob.pdf'), 'student_id': 'bob'},
            content_type='multipart/form-data',
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body['submission']['submission_id'], 'asg-sub-colon:bob.pdf')
        download_url = body['download_url']
        self.assertNotIn(':', download_url.rsplit('/', 1)[-1])

        dl = self.client.get(download_url)
        self.assertEqual(dl.status_code, 200, download_url)
        self.assertTrue(dl.data.startswith(b'%PDF'))

        reader = self.client.get('/assignments/asg-sub-colon/reader')
        self.assertEqual(reader.status_code, 200)
        self.assertIn(download_url.encode('utf-8'), reader.data)

    def test_health_and_providers_report_embedding_model(self):
        health = self.client.get('/health').get_json()
        self.assertEqual(health['embedding_model'],
                         self.server.novelty_detector.embedding_model_name)
        self.assertEqual(health['embedding_dim'], self.server.novelty_detector.embedding_dim)
        providers = self.client.get('/providers').get_json()
        self.assertEqual(providers['embedding_model'], health['embedding_model'])
        self.assertIn('trusted_hosts', providers)
        self.assertIn('fallback', providers['providers'])
        self.assertIn('leaves_host', providers['providers']['fallback'])

    def test_second_submission_scored_against_first(self):
        self.client.post('/assignments', json={'assignment_id': 'asg-sub-5'})
        same_text = ['Quantum entanglement underpins cryptography. ' * 6]
        pdf1 = _build_pdf_bytes(same_text)
        self.client.post(
            '/assignments/asg-sub-5/submissions',
            data={'file': (BytesIO(pdf1), 'first.pdf'),
                  'student_id': 'first', 'submission_id': 'sub-1'},
            content_type='multipart/form-data',
        )
        pdf2 = _build_pdf_bytes(same_text)
        resp = self.client.post(
            '/assignments/asg-sub-5/submissions',
            data={'file': (BytesIO(pdf2), 'second.pdf'),
                  'student_id': 'second', 'submission_id': 'sub-2'},
            content_type='multipart/form-data',
        )
        body = resp.get_json()
        # Same content as the prior submission -> low novelty for at least
        # one chunk. (Stub-generated prompts won't be perfect, but the
        # embedding space still places them close.)
        scores = [e['novelty_score'] for e in body['novelty_scores']]
        self.assertLess(min(scores), 0.5,
                        f"expected at least one chunk < 0.5, got {scores}")

    # --- listing / reader / detail ---

    def test_list_submissions_returns_recent_first(self):
        self.client.post('/assignments', json={'assignment_id': 'asg-list'})
        pdf = _build_pdf_bytes(['Some text. ' * 30])
        for sid in ('first', 'second'):
            self.client.post(
                f'/assignments/asg-list/submissions',
                data={'file': (BytesIO(pdf), f'{sid}.pdf'),
                      'student_id': sid, 'submission_id': sid},
                content_type='multipart/form-data',
            )
        resp = self.client.get('/assignments/asg-list/submissions')
        self.assertEqual(resp.status_code, 200)
        listed = resp.get_json()['submissions']
        self.assertEqual(len(listed), 2)
        self.assertEqual({s['submission_id'] for s in listed}, {'first', 'second'})

    def test_list_unknown_assignment_404(self):
        resp = self.client.get('/assignments/missing/submissions')
        self.assertEqual(resp.status_code, 404)

    def test_get_submission_detail(self):
        self.client.post('/assignments', json={'assignment_id': 'asg-detail'})
        pdf = _build_pdf_bytes(['Detail probe text. ' * 30])
        self.client.post(
            '/assignments/asg-detail/submissions',
            data={'file': (BytesIO(pdf), 'd.pdf'),
                  'student_id': 's1', 'submission_id': 'd-sub'},
            content_type='multipart/form-data',
        )
        resp = self.client.get('/assignments/asg-detail/submissions/d-sub')
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body['submission_id'], 'd-sub')
        self.assertIn('chunks', body)
        self.assertGreater(len(body['chunks']), 0)
        self.assertIn('text', body['chunks'][0])
        self.assertIn('novelty_score', body['chunks'][0])

    def test_get_submission_with_mismatched_assignment_404(self):
        # detail endpoint should refuse to cross assignment boundaries
        self.client.post('/assignments', json={'assignment_id': 'asg-x'})
        self.client.post('/assignments', json={'assignment_id': 'asg-y'})
        pdf = _build_pdf_bytes(['Boundary test. ' * 30])
        self.client.post(
            '/assignments/asg-x/submissions',
            data={'file': (BytesIO(pdf), 'b.pdf'),
                  'student_id': 's1', 'submission_id': 'b-sub'},
            content_type='multipart/form-data',
        )
        # Request the submission under the WRONG assignment id
        resp = self.client.get('/assignments/asg-y/submissions/b-sub')
        self.assertEqual(resp.status_code, 404)

    def test_reader_view_renders_html(self):
        self.client.post('/assignments', json={'assignment_id': 'asg-reader',
                                               'name': 'Reader Test'})
        resp = self.client.get('/assignments/asg-reader/reader')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Reader Test', resp.data)
        self.assertIn(b'No submissions yet', resp.data)

    def test_reader_view_with_submissions(self):
        self.client.post('/assignments', json={'assignment_id': 'asg-reader-2'})
        pdf = _build_pdf_bytes(['Some text. ' * 30])
        self.client.post(
            '/assignments/asg-reader-2/submissions',
            data={'file': (BytesIO(pdf), 'r.pdf'),
                  'student_id': 'reader-student',
                  'submission_id': 'r-sub'},
            content_type='multipart/form-data',
        )
        resp = self.client.get('/assignments/asg-reader-2/reader')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'reader-student', resp.data)
        # Annotated PDF link should be present
        self.assertIn(b'/download/annotated_r-sub_r.pdf', resp.data)

    # --- /health surfaces the new fields ---

    def test_health_includes_assessment_ready_and_local_only(self):
        resp = self.client.get('/health')
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn('local_only', body)
        self.assertIn('active_provider', body)
        self.assertIn('assessment_ready', body)


if __name__ == '__main__':
    unittest.main()

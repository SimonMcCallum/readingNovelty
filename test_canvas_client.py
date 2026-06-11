"""
Tests for canvas_client.py.

No real Canvas calls — every HTTP method is mocked. Verifies that the
client builds the right URLs/payloads, parses pagination, and picks the
correct PDF attachment.
"""

import io
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from canvas_client import CanvasClient, CanvasError


def _resp(status=200, json_body=None, headers=None, content=b''):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_body if json_body is not None else {}
    r.headers = headers or {}
    r.content = content
    return r


class TestCanvasClientBasics(unittest.TestCase):
    def test_constructor_validates_inputs(self):
        with self.assertRaises(ValueError):
            CanvasClient('', 'tok')
        with self.assertRaises(ValueError):
            CanvasClient('https://x', '')

    def test_constructor_strips_trailing_slash_and_sets_auth(self):
        c = CanvasClient('https://canvas.example.edu/', 'TOKEN')
        self.assertEqual(c.base_url, 'https://canvas.example.edu')
        self.assertEqual(c.session.headers['Authorization'], 'Bearer TOKEN')

    def test_url_passes_through_absolute(self):
        c = CanvasClient('https://x', 'tok')
        self.assertEqual(c._url('https://files.example/abc'), 'https://files.example/abc')
        self.assertEqual(c._url('/courses/1'), 'https://x/api/v1/courses/1')


class TestPagination(unittest.TestCase):
    def test_follows_next_link(self):
        c = CanvasClient('https://x', 'tok')
        page1 = _resp(
            json_body=[{'id': 1}, {'id': 2}],
            headers={'Link': '<https://x/api/v1/things?page=2>; rel="next"'},
        )
        page2 = _resp(json_body=[{'id': 3}], headers={})
        with patch.object(c.session, 'get', side_effect=[page1, page2]) as get:
            rows = list(c._paginate('/things'))
        self.assertEqual([r['id'] for r in rows], [1, 2, 3])
        self.assertEqual(get.call_count, 2)
        # Second call uses the absolute next URL, no params
        second_call = get.call_args_list[1]
        self.assertEqual(second_call.args[0], 'https://x/api/v1/things?page=2')

    def test_raises_on_non_200(self):
        c = CanvasClient('https://x', 'tok')
        with patch.object(c.session, 'get', return_value=_resp(status=401, json_body={'msg': 'no'})):
            with self.assertRaises(CanvasError):
                list(c._paginate('/things'))

    def test_no_next_link_terminates(self):
        c = CanvasClient('https://x', 'tok')
        single = _resp(json_body=[{'id': 1}], headers={'Link': '<...>; rel="last"'})
        with patch.object(c.session, 'get', return_value=single) as get:
            rows = list(c._paginate('/things'))
        self.assertEqual(rows, [{'id': 1}])
        self.assertEqual(get.call_count, 1)


class TestPdfAttachmentPicker(unittest.TestCase):
    def test_picks_pdf_by_extension(self):
        sub = {'attachments': [
            {'display_name': 'notes.docx', 'content-type': 'application/x-docx'},
            {'display_name': 'paper.PDF', 'content-type': 'application/pdf'},
        ]}
        att = CanvasClient.pdf_attachment(sub)
        self.assertEqual(att['display_name'], 'paper.PDF')

    def test_picks_pdf_by_mime_when_extension_missing(self):
        sub = {'attachments': [{'filename': 'paper', 'mime_class': 'pdf'}]}
        att = CanvasClient.pdf_attachment(sub)
        self.assertIsNotNone(att)

    def test_returns_none_when_no_pdf(self):
        sub = {'attachments': [{'display_name': 'video.mp4', 'mime_class': 'video'}]}
        self.assertIsNone(CanvasClient.pdf_attachment(sub))

    def test_returns_none_when_no_attachments(self):
        self.assertIsNone(CanvasClient.pdf_attachment({}))


class TestDownloadAttachment(unittest.TestCase):
    def test_streams_to_file(self):
        c = CanvasClient('https://x', 'tok')
        chunks = [b'hello ', b'world']

        ctx = MagicMock()
        ctx.__enter__.return_value = MagicMock(
            status_code=200,
            iter_content=lambda chunk_size=8192: iter(chunks),
        )
        ctx.__exit__.return_value = False

        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, 'out.pdf')
            with patch.object(c.session, 'get', return_value=ctx):
                c.download_attachment({'url': 'https://files.example/abc'}, dest)
            with open(dest, 'rb') as fh:
                self.assertEqual(fh.read(), b'hello world')

    def test_no_url_raises(self):
        c = CanvasClient('https://x', 'tok')
        with self.assertRaises(CanvasError):
            c.download_attachment({}, '/tmp/whatever')


class TestUploadCommentFile(unittest.TestCase):
    def test_three_step_upload_returns_file_id(self):
        c = CanvasClient('https://x', 'tok')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as fh:
            fh.write(b'PDFDATA')
            file_path = fh.name

        init_resp = _resp(
            status=200,
            json_body={'upload_url': 'https://files.example/slot', 'upload_params': {'k': 'v'}},
        )
        upload_resp = _resp(status=201, json_body={'id': 4242}, content=b'{"id":4242}')

        try:
            with patch.object(c.session, 'post', side_effect=[init_resp, upload_resp]) as post:
                file_id = c.upload_comment_file('C1', 'A1', 'U1', file_path)
            self.assertEqual(file_id, 4242)
            # First call: init slot
            init_call = post.call_args_list[0]
            self.assertIn('/comments/files', init_call.args[0])
            self.assertEqual(init_call.kwargs['data']['name'], os.path.basename(file_path))
            self.assertEqual(init_call.kwargs['data']['size'], len(b'PDFDATA'))
            # Second call: upload body to the slot URL
            upload_call = post.call_args_list[1]
            self.assertEqual(upload_call.args[0], 'https://files.example/slot')
            self.assertEqual(upload_call.kwargs['data'], {'k': 'v'})
        finally:
            os.unlink(file_path)

    def test_missing_upload_url_raises(self):
        c = CanvasClient('https://x', 'tok')
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            file_path = fh.name
        try:
            with patch.object(c.session, 'post', return_value=_resp(json_body={})):
                with self.assertRaises(CanvasError):
                    c.upload_comment_file('C1', 'A1', 'U1', file_path)
        finally:
            os.unlink(file_path)


class TestPreflightHelpers(unittest.TestCase):
    def test_get_self_returns_user(self):
        c = CanvasClient('https://x', 'tok')
        with patch.object(c.session, 'get', return_value=_resp(
            json_body={'id': 42, 'name': 'Alice', 'primary_email': 'a@x.edu'}
        )) as get:
            me = c.get_self()
        self.assertEqual(me['id'], 42)
        call = get.call_args
        self.assertIn('/users/self', call.args[0])

    def test_get_self_bad_token_raises(self):
        c = CanvasClient('https://x', 'tok')
        with patch.object(c.session, 'get', return_value=_resp(status=401, json_body={'errors': 'no'})):
            with self.assertRaises(CanvasError):
                c.get_self()

    def test_get_assignment_returns_metadata(self):
        c = CanvasClient('https://x', 'tok')
        with patch.object(c.session, 'get', return_value=_resp(
            json_body={'id': 12, 'name': 'Essay 1', 'submission_types': ['online_upload']}
        )) as get:
            asg = c.get_assignment('C1', 'A1')
        self.assertEqual(asg['name'], 'Essay 1')
        call = get.call_args
        self.assertIn('/courses/C1/assignments/A1', call.args[0])

    def test_get_assignment_404_raises(self):
        c = CanvasClient('https://x', 'tok')
        with patch.object(c.session, 'get', return_value=_resp(status=404)):
            with self.assertRaises(CanvasError):
                c.get_assignment('C1', 'A1')


class TestPostSubmissionComment(unittest.TestCase):
    def test_attaches_file_ids(self):
        c = CanvasClient('https://x', 'tok')
        with patch.object(c.session, 'put', return_value=_resp(json_body={'ok': True})) as put:
            c.post_submission_comment('C1', 'A1', 'U1', 'hello', file_ids=[1, 2])
        call = put.call_args
        self.assertIn('/courses/C1/assignments/A1/submissions/U1', call.args[0])
        self.assertEqual(call.kwargs['data']['comment[text_comment]'], 'hello')
        self.assertEqual(call.kwargs['data']['comment[file_ids][]'], [1, 2])

    def test_raises_on_error_status(self):
        c = CanvasClient('https://x', 'tok')
        with patch.object(c.session, 'put', return_value=_resp(status=403, json_body={'e': 'no'})):
            with self.assertRaises(CanvasError):
                c.post_submission_comment('C1', 'A1', 'U1', 'hi')


if __name__ == '__main__':
    unittest.main()

"""
Canvas LMS API client (lecturer access token).

Designed for a lecturer to self-serve: no LTI registration, no admin
approval required. Wraps just the surface area the assessment pipeline
needs:
  - list submissions for an assignment
  - download a submission's PDF attachment
  - post a submission comment with an attached file (annotated PDF)

Reference: https://canvas.instructure.com/doc/api/
"""

import os
import logging
from typing import Iterator, List, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class CanvasError(RuntimeError):
    """Raised when the Canvas API returns an unexpected response."""


class CanvasClient:
    """Thin Canvas API wrapper using a per-user access token."""

    def __init__(self, base_url: str, token: str, timeout: int = 30):
        if not base_url or not token:
            raise ValueError("base_url and token are required")
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {token}'})

    def _url(self, path: str) -> str:
        if path.startswith('http://') or path.startswith('https://'):
            return path
        return f"{self.base_url}/api/v1{path}"

    def _paginate(self, path: str, params: Optional[Dict] = None) -> Iterator[Dict]:
        """Follow Canvas's Link-header pagination, yielding each row."""
        url = self._url(path)
        params = dict(params or {})
        params.setdefault('per_page', 50)
        while url:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                raise CanvasError(
                    f"GET {url} returned {resp.status_code}: {resp.text[:300]}"
                )
            for row in resp.json():
                yield row
            params = None  # next-page URLs already carry their own query
            url = self._next_link(resp.headers.get('Link', ''))

    @staticmethod
    def _next_link(link_header: str) -> Optional[str]:
        for part in link_header.split(','):
            segments = part.strip().split(';')
            if len(segments) < 2:
                continue
            url = segments[0].strip().lstrip('<').rstrip('>')
            for seg in segments[1:]:
                if seg.strip() == 'rel="next"':
                    return url
        return None

    # ------- identity + assignment + submission reads -------

    def get_self(self) -> Dict:
        """GET /users/self — proves the access token works and reveals which
        user it belongs to. Cheap preflight check."""
        url = self._url('/users/self')
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            raise CanvasError(
                f"Token check failed: {resp.status_code} {resp.text[:300]}"
            )
        return resp.json()

    def get_assignment(self, course_id: str, assignment_id: str) -> Dict:
        """Fetch a single assignment's metadata (name, due dates, etc.)."""
        url = self._url(f"/courses/{course_id}/assignments/{assignment_id}")
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            raise CanvasError(
                f"Could not fetch assignment {course_id}/{assignment_id}: "
                f"{resp.status_code} {resp.text[:300]}"
            )
        return resp.json()

    def list_submissions(self, course_id: str, assignment_id: str) -> List[Dict]:
        """Return all submissions for an assignment with their attachments."""
        return list(self._paginate(
            f"/courses/{course_id}/assignments/{assignment_id}/submissions",
            params={'include[]': 'submission_comments', 'include[]': 'user'},
        ))

    def download_attachment(self, attachment: Dict, dest_path: str) -> str:
        """Download a Canvas attachment dict's `url` to dest_path."""
        url = attachment.get('url')
        if not url:
            raise CanvasError(f"Attachment has no url: {attachment}")
        with self.session.get(url, stream=True, timeout=self.timeout) as resp:
            if resp.status_code != 200:
                raise CanvasError(
                    f"Attachment download {url} returned {resp.status_code}"
                )
            with open(dest_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return dest_path

    # ------- comment + file upload back -------

    def upload_comment_file(
        self, course_id: str, assignment_id: str, user_id: str, file_path: str
    ) -> int:
        """
        Upload a local file as a submission-comment attachment. Returns the
        Canvas file_id, which is then passed to post_submission_comment().

        Canvas's file upload is a three-step dance:
         1. POST to .../comments/files with name+size to get an upload slot
         2. POST file body to the slot URL
         3. The slot response carries the new file's id
        """
        size = os.path.getsize(file_path)
        name = os.path.basename(file_path)
        init_url = self._url(
            f"/courses/{course_id}/assignments/{assignment_id}"
            f"/submissions/{user_id}/comments/files"
        )
        init_resp = self.session.post(
            init_url, data={'name': name, 'size': size}, timeout=self.timeout
        )
        if init_resp.status_code not in (200, 201):
            raise CanvasError(
                f"File upload init failed: {init_resp.status_code} {init_resp.text[:300]}"
            )
        slot = init_resp.json()
        upload_url = slot.get('upload_url')
        upload_params = slot.get('upload_params') or {}
        if not upload_url:
            raise CanvasError(f"upload_url missing in slot response: {slot}")

        with open(file_path, 'rb') as fh:
            upload_resp = self.session.post(
                upload_url,
                data=upload_params,
                files={'file': (name, fh)},
                timeout=self.timeout,
            )
        if upload_resp.status_code not in (200, 201, 301, 302):
            raise CanvasError(
                f"File upload body failed: {upload_resp.status_code} {upload_resp.text[:300]}"
            )
        body = upload_resp.json() if upload_resp.content else {}
        file_id = body.get('id')
        if file_id is None:
            raise CanvasError(f"file_id missing in upload response: {body}")
        return int(file_id)

    def post_submission_comment(
        self,
        course_id: str,
        assignment_id: str,
        user_id: str,
        text: str,
        file_ids: Optional[List[int]] = None,
    ) -> Dict:
        """Add a submission comment, optionally with file attachments."""
        url = self._url(
            f"/courses/{course_id}/assignments/{assignment_id}/submissions/{user_id}"
        )
        data: Dict = {'comment[text_comment]': text}
        for fid in file_ids or []:
            data.setdefault('comment[file_ids][]', []).append(fid)
        resp = self.session.put(url, data=data, timeout=self.timeout)
        if resp.status_code not in (200, 201):
            raise CanvasError(
                f"Posting comment failed: {resp.status_code} {resp.text[:300]}"
            )
        return resp.json()

    # ------- helpers -------

    @staticmethod
    def pdf_attachment(submission: Dict) -> Optional[Dict]:
        """Pick the first PDF attachment from a submission, if any."""
        for att in submission.get('attachments') or []:
            name = (att.get('display_name') or att.get('filename') or '').lower()
            mime = (att.get('content-type') or att.get('mime_class') or '').lower()
            if name.endswith('.pdf') or 'pdf' in mime:
                return att
        return None

"""
Semantic Scholar client for citation-graph novelty.

Wraps the Graph API to fetch a paper and its references with abstracts.
No auth required for low-volume use (~100 req per 5 min unsigned). Supports
optional API key (set SEMANTIC_SCHOLAR_API_KEY) and a small on-disk cache so
re-runs don't re-hit the API.

Designed to feed CorpusStore: each reference's abstract becomes one entry
in a per-paper corpus partition the target paper is scored against.
"""

import json
import logging
import os
import time
from hashlib import sha1
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class CitationFetchError(RuntimeError):
    """Raised when the Semantic Scholar API returns an unexpected result."""


class SemanticScholarClient:
    """Thin wrapper around the Semantic Scholar Graph API."""

    BASE = "https://api.semanticscholar.org/graph/v1"
    DEFAULT_FIELDS = "title,abstract,year,authors,paperId"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        rate_limit_seconds: float = 1.0,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout = timeout
        self.session = requests.Session()
        if self.api_key:
            self.session.headers['x-api-key'] = self.api_key
        self._last_request_at = 0.0

    @staticmethod
    def normalize_identifier(identifier: str) -> str:
        """Accepts '10.xxx/yyy', 'DOI:10.xxx', or 'ARXIV:1234.5678'.

        Returns the prefixed form Semantic Scholar expects.
        """
        s = identifier.strip()
        upper = s.upper()
        if upper.startswith('DOI:') or upper.startswith('ARXIV:'):
            return s
        if s.startswith('10.'):
            return f"DOI:{s}"
        # arXiv numeric like 2106.09685
        if s.replace('.', '').replace('v', '').isdigit():
            return f"ARXIV:{s}"
        return s

    def _cache_path(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        digest = sha1(key.encode('utf-8')).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{digest}.json")

    def _read_cache(self, key: str):
        path = self._cache_path(key)
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    return json.load(fh)
            except Exception:
                return None
        return None

    def _write_cache(self, key: str, payload):
        path = self._cache_path(key)
        if path:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh)

    def _throttle(self):
        """Spread requests apart per rate_limit_seconds."""
        if self.rate_limit_seconds <= 0:
            return
        gap = time.monotonic() - self._last_request_at
        if gap < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - gap)

    def _get(self, url: str, params: Dict) -> Dict:
        self._throttle()
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
        finally:
            self._last_request_at = time.monotonic()
        if resp.status_code == 404:
            raise CitationFetchError(f"Not found: {url} {params}")
        if resp.status_code == 429:
            raise CitationFetchError("Semantic Scholar rate limit hit; back off and retry.")
        if resp.status_code != 200:
            raise CitationFetchError(
                f"GET {url} returned {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()

    def fetch_paper(self, identifier: str) -> Dict:
        """Return paper metadata for a DOI / arXiv id / Semantic Scholar id."""
        ident = self.normalize_identifier(identifier)
        cache_key = f"paper:{ident}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached
        url = f"{self.BASE}/paper/{ident}"
        payload = self._get(url, {'fields': self.DEFAULT_FIELDS})
        self._write_cache(cache_key, payload)
        return payload

    def fetch_references(
        self,
        identifier: str,
        year_max: Optional[int] = None,
        limit: int = 200,
    ) -> List[Dict]:
        """
        Return the references of `identifier`. Optionally drops anything
        published in/after year_max (useful when modelling "what was knowable
        at submission time").

        Each entry contains: paperId, title, abstract, year, authors. Entries
        with empty abstracts are returned too — callers decide what to keep.
        """
        ident = self.normalize_identifier(identifier)
        cache_key = f"refs:{ident}:limit={limit}"
        cached = self._read_cache(cache_key)
        if cached is None:
            url = f"{self.BASE}/paper/{ident}/references"
            params = {
                'fields': f"citedPaper.{self.DEFAULT_FIELDS}",
                'limit': min(1000, limit),
            }
            payload = self._get(url, params)
            self._write_cache(cache_key, payload)
            cached = payload

        cited_items = []
        for row in (cached.get('data') or []):
            cited = row.get('citedPaper') or {}
            if not cited:
                continue
            if year_max is not None and cited.get('year') is not None:
                try:
                    if int(cited['year']) >= int(year_max):
                        continue
                except (TypeError, ValueError):
                    pass
            cited_items.append(cited)
        return cited_items

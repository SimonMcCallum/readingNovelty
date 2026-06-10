"""
Citation extraction utilities.

This module is the smallest credible step towards citation-graph novelty:
given a PDF (or its extracted text), surface the things we can use to find
its bibliography — DOIs, arXiv ids — and a reduced-fidelity reference list
suitable for follow-up enrichment via CrossRef / Semantic Scholar.

Why this is *not* yet the full citation-graph novelty path:
  Real bibliography extraction from PDFs is a research-grade problem. The
  state of the art is GROBID, which runs a CRF + DL model trained on
  scholarly references. A pure-regex approach (here) reliably finds DOIs
  and arXiv ids but cannot parse "Author, A. and Author, B. (2019). Title.
  Journal, 12(3), 100-110." into structured fields.

Design sketch for full citation-graph novelty:
  1. Extract references via GROBID (`grobid_client` library) -> structured
     refs with title, authors, year, DOI when present.
  2. Resolve missing DOIs via CrossRef bibliographic search.
  3. For each cited work: fetch abstract from CrossRef / Semantic Scholar
     (both are free, open APIs; no key required at low volume).
  4. Build a corpus from those abstracts plus any abstracts of works
     published in the same field strictly before the target's year (via
     Semantic Scholar's citation graph).
  5. Score the target's chunks against that corpus using the existing
     CorpusStore. Use a separate corpus partition per paper so the graph
     stays per-target.

Why we stop here for now:
  GROBID is a 1.5GB Docker image + a separate Java service. Worth doing,
  but a bigger setup than the per-assignment Canvas pipeline this project
  is built around. The DOIs + arXiv ids we extract here let a human (or a
  future agent) bootstrap the graph by hand if needed.
"""

import re
from typing import List, Set


# DOI syntax per CrossRef: 10.<4-9 digits>/<suffix>. The suffix can be almost
# anything printable but commonly stops at whitespace, '>', '"', and ')'.
DOI_PATTERN = re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)

# arXiv: new-style (1234.5678 with optional version), old-style (cs.LG/0301234).
ARXIV_NEW = re.compile(r'\barXiv:?\s*(\d{4}\.\d{4,5})(v\d+)?\b', re.IGNORECASE)
ARXIV_OLD = re.compile(r'\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b')


def _strip_trailing_punct(token: str) -> str:
    """DOIs in PDF text frequently end with ',', '.', or ')'. Drop those."""
    while token and token[-1] in '.,;:)]>"':
        token = token[:-1]
    return token


def extract_dois(text: str) -> List[str]:
    """Return a de-duplicated list of DOIs found in `text`, preserving order."""
    seen: Set[str] = set()
    out: List[str] = []
    for match in DOI_PATTERN.findall(text):
        doi = _strip_trailing_punct(match).lower()
        if doi and doi not in seen:
            seen.add(doi)
            out.append(doi)
    return out


def extract_arxiv_ids(text: str) -> List[str]:
    """Return a de-duplicated list of arXiv ids in `text`, preserving order."""
    seen: Set[str] = set()
    out: List[str] = []
    for m in ARXIV_NEW.finditer(text):
        aid = m.group(1).lower()
        if aid not in seen:
            seen.add(aid)
            out.append(aid)
    for m in ARXIV_OLD.finditer(text):
        aid = m.group(1).lower()
        if aid not in seen:
            seen.add(aid)
            out.append(aid)
    return out


def crossref_metadata_url(doi: str) -> str:
    """The CrossRef metadata URL for a DOI. Public, no auth required.

    Returned shape (GET): {message: {title: [...], author: [...], abstract: ...,
    issued: {date-parts: [[year, ...]]}, container-title: [...]}}

    Note: many CrossRef abstracts are empty; Semantic Scholar
    (api.semanticscholar.org/graph/v1/paper/DOI:<doi>) is the better source
    for abstracts in the no-auth tier.
    """
    return f"https://api.crossref.org/works/{doi}"


def semantic_scholar_url(doi: str) -> str:
    """Semantic Scholar Graph API URL for a DOI. Public, no auth required at
    low volume (1 req/sec without a key)."""
    return f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,abstract,year,authors,references"

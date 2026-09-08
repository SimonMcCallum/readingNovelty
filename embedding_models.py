"""
Embedding model discovery.

Pulls a ranking of sentence-similarity embedding models from Hugging Face so
EMBEDDING_MODEL can be chosen on evidence rather than habit. Everything here
is read-only HTTP against public endpoints and is cached on disk; nothing
about your documents is sent anywhere.

Sources
-------
cards    Hub API: models tagged `mteb`, fetched with their card metadata. MTEB
         Semantic Textual Similarity (STS) scores are read from each card's
         `model-index`. One request per page, no extra dependency. Misses
         models whose cards do not publish results (several 2025+ releases).
results  The `mteb/results` dataset queried through the datasets-server
         filter API. Complete, but the server-side index for this 8.8M-row
         dataset is frequently unavailable ("index is loading").
mteb     The `mteb` package (`pip install mteb`). Downloads the full results
         repository once into ~/.cache/mteb and aggregates locally. Complete
         and authoritative; heavy first run.
popular  Hub API: pipeline_tag=sentence-similarity sorted by downloads or
         likes. Not a quality signal; a sanity check on what people run.

The ranking metric is the mean Spearman correlation (x100) over the English
STS tasks of MTEB(eng, v2). STS is the task family closest to what this
project does: "how far is this paragraph from its nearest neighbours".
"""

import json
import logging
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)

HUB_API = "https://huggingface.co/api/models"
DATASETS_SERVER_FILTER = "https://datasets-server.huggingface.co/filter"

# English STS tasks in MTEB(eng, v2); v1 names are accepted when parsing cards.
MTEB_ENG_V2_STS = (
    'BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15',
    'STSBenchmark', 'STS17', 'STS22.v2',
)
ENGLISH_STS_TASKS = frozenset(
    {'BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
     'STS17', 'STS22', 'STSBenchmark'}
)
_ENGLISH_CONFIGS = {None, '', 'default', 'en', 'en-en', 'eng', 'eng-eng', 'eng_latn'}
_SPEARMAN_KEYS = ('cosine_spearman', 'cos_sim_spearman', 'spearman', 'main_score')
_HUB_EXPAND = ('cardData', 'safetensors', 'library_name', 'gated',
               'downloads', 'likes', 'pipeline_tag', 'tags')

DEFAULT_TTL_HOURS = 24.0


class SourceUnavailable(RuntimeError):
    """A ranking source could not be reached or is temporarily unusable."""


@dataclass
class EmbeddingCandidate:
    model_id: str
    sts_mean: Optional[float] = None    # 0..100, mean Spearman over STS tasks
    n_sts_tasks: int = 0
    params: Optional[int] = None        # parameter count from safetensors metadata
    downloads: Optional[int] = None
    likes: Optional[int] = None
    library: Optional[str] = None
    gated: bool = False
    custom_code: bool = False
    gguf: bool = False                  # quantised llama.cpp export, not loadable here
    pipeline_tag: Optional[str] = None
    source: str = ''

    @property
    def runs_with_sentence_transformers(self) -> bool:
        return self.library == 'sentence-transformers' and not self.gguf

    @property
    def is_popularity_only(self) -> bool:
        return self.source.startswith('popular')

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['runs_with_sentence_transformers'] = self.runs_with_sentence_transformers
        return d


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def parse_params(text: Optional[str]) -> Optional[int]:
    """'600M' -> 600_000_000, '1.5B' -> 1_500_000_000, '33000000' -> int."""
    if text is None or str(text).strip() == '':
        return None
    s = str(text).strip().upper().replace(',', '').replace('_', '')
    m = re.match(r'^(\d+(?:\.\d+)?)\s*([KMBG]?)$', s)
    if not m:
        raise ValueError(f"cannot parse parameter count {text!r} (use e.g. 600M, 1.5B)")
    value = float(m.group(1))
    mult = {'': 1, 'K': 1e3, 'M': 1e6, 'B': 1e9, 'G': 1e9}[m.group(2)]
    return int(value * mult)


def human_params(n: Optional[int]) -> str:
    if n is None:
        return '?'
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(n)


def cache_dir() -> str:
    return os.getenv('EMBEDDING_CACHE_DIR') or os.path.join(
        os.getenv('CORPUS_DIR', 'corpus'), 'hf_cache'
    )


def _cached_json(key: str, fetch: Callable[[], object], ttl_hours: float = DEFAULT_TTL_HOURS,
                 refresh: bool = False):
    """Disk cache for a JSON-serialisable fetch() result."""
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', key)
    path = os.path.join(cache_dir(), f"{safe}.json")
    if not refresh and os.path.exists(path):
        age_h = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_h <= ttl_hours:
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                pass
    data = fetch()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh)
    os.replace(tmp, path)
    return data


def _normalise_score(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v * 100.0 if v <= 1.0 else v


# --------------------------------------------------------------------------
# Source: model cards (Hub API)
# --------------------------------------------------------------------------

def split_dataset_name(name: str, config: Optional[str] = None):
    """'MTEB STS17 (en-en)' -> ('STS17', 'en-en'); 'MTEB STS22.v2' -> ('STS22', None)."""
    n = (name or '').strip()
    if n.upper().startswith('MTEB '):
        n = n[5:].strip()
    cfg = config
    m = re.match(r'^(.*?)\s*\((.*?)\)\s*$', n)
    if m:
        n = m.group(1).strip()
        cfg = cfg or m.group(2).strip()
    n = re.sub(r'\.v\d+$', '', n)
    return n, (cfg.lower() if isinstance(cfg, str) else cfg)


def _pick_spearman(metrics: Iterable[Dict]) -> Optional[float]:
    by_type = {}
    for met in metrics or []:
        t = met.get('type')
        if t and 'value' in met:
            by_type[str(t)] = met['value']
    for key in _SPEARMAN_KEYS:
        if key in by_type:
            return _normalise_score(by_type[key])
    return None


def sts_scores_from_card(card_data: Optional[Dict]) -> Dict[str, float]:
    """English MTEB STS Spearman scores (x100) keyed by task, from model-index."""
    out: Dict[str, float] = {}
    for entry in (card_data or {}).get('model-index') or []:
        for res in entry.get('results') or []:
            if (res.get('task') or {}).get('type') != 'STS':
                continue
            ds = res.get('dataset') or {}
            base, cfg = split_dataset_name(str(ds.get('name') or ''), ds.get('config'))
            if base not in ENGLISH_STS_TASKS or cfg not in _ENGLISH_CONFIGS:
                continue
            score = _pick_spearman(res.get('metrics') or [])
            if score is None:
                continue
            out[base] = max(out.get(base, -1.0), score)
    return out


def candidate_from_hub_model(m: Dict, source: str = 'cards') -> EmbeddingCandidate:
    tags = m.get('tags') or []
    gated = m.get('gated')
    return EmbeddingCandidate(
        model_id=m.get('id') or m.get('modelId') or '',
        params=(m.get('safetensors') or {}).get('total'),
        downloads=m.get('downloads'),
        likes=m.get('likes'),
        library=m.get('library_name'),
        gated=bool(gated) and gated not in ('False', 'false'),
        custom_code='custom_code' in tags,
        gguf='gguf' in tags or (m.get('id') or '').lower().endswith('-gguf'),
        pipeline_tag=m.get('pipeline_tag'),
        source=source,
    )


def rank_from_hub_models(models: Sequence[Dict], source: str = 'cards',
                         dedupe: bool = True) -> List[EmbeddingCandidate]:
    """Rank Hub model records (with cardData) by mean English STS Spearman."""
    cands: List[EmbeddingCandidate] = []
    for m in models:
        scores = sts_scores_from_card(m.get('cardData'))
        if not scores:
            continue
        c = candidate_from_hub_model(m, source=source)
        c.sts_mean = statistics.mean(scores.values())
        c.n_sts_tasks = len(scores)
        cands.append(c)
    return sort_candidates(cands, dedupe=dedupe)


def sort_candidates(cands: Iterable[EmbeddingCandidate], dedupe: bool = True
                    ) -> List[EmbeddingCandidate]:
    """Sort by STS desc then downloads desc; drop re-uploads with identical scores."""
    ordered = sorted(
        cands,
        key=lambda c: (-(c.sts_mean or -1.0), -(c.downloads or 0), c.model_id),
    )
    if not dedupe:
        return ordered
    seen = set()
    kept = []
    for c in ordered:
        sig = (round(c.sts_mean or -1.0, 2), c.n_sts_tasks, c.params)
        if c.sts_mean is not None and c.params is not None and sig in seen:
            continue
        seen.add(sig)
        kept.append(c)
    return kept


def _hub_get(url: str, params: Dict, timeout: float = 90.0) -> requests.Response:
    headers = {}
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    if token:
        headers['Authorization'] = f"Bearer {token}"
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        raise SourceUnavailable(f"{url} returned {resp.status_code}: {resp.text[:200]}")
    return resp


def fetch_hub_models(filter_tag: Optional[str] = 'mteb', pipeline_tag: Optional[str] = None,
                     sort: str = 'downloads', limit: int = 200,
                     expand: Sequence[str] = _HUB_EXPAND, timeout: float = 90.0) -> List[Dict]:
    """Page through the Hub model listing. Follows the Link: rel=next cursor."""
    params = [('sort', sort), ('direction', '-1'), ('limit', str(min(100, limit)))]
    if filter_tag:
        params.append(('filter', filter_tag))
    if pipeline_tag:
        params.append(('pipeline_tag', pipeline_tag))
    for e in expand:
        params.append(('expand[]', e))
    out: List[Dict] = []
    url: Optional[str] = HUB_API
    query: Optional[Dict] = params
    while url and len(out) < limit:
        resp = _hub_get(url, query, timeout=timeout)
        page = resp.json()
        if not isinstance(page, list):
            raise SourceUnavailable(f"unexpected Hub response: {str(page)[:200]}")
        out.extend(page)
        nxt = resp.links.get('next', {}).get('url')
        url, query = (nxt, None) if nxt and page else (None, None)
    return out[:limit]


def fetch_cards_ranking(limit_models: int = 200, refresh: bool = False,
                        ttl_hours: float = DEFAULT_TTL_HOURS) -> List[EmbeddingCandidate]:
    """Rank `mteb`-tagged Hub models by the STS scores in their model cards."""
    def _fetch():
        by_id: Dict[str, Dict] = {}
        for sort in ('downloads', 'likes'):
            for m in fetch_hub_models(filter_tag='mteb', sort=sort, limit=limit_models):
                by_id.setdefault(m.get('id'), m)
        return list(by_id.values())

    models = _cached_json(f"hub_mteb_cards_{limit_models}", _fetch,
                          ttl_hours=ttl_hours, refresh=refresh)
    return rank_from_hub_models(models, source='cards')


# --------------------------------------------------------------------------
# Source: popularity (Hub API)
# --------------------------------------------------------------------------

def fetch_popular(sort: str = 'downloads', limit: int = 50, refresh: bool = False,
                  ttl_hours: float = DEFAULT_TTL_HOURS) -> List[EmbeddingCandidate]:
    """sentence-similarity models by downloads / likes / trendingScore."""
    if sort not in ('downloads', 'likes', 'trendingScore'):
        raise ValueError("sort must be downloads, likes or trendingScore")
    expand = tuple(e for e in _HUB_EXPAND if e != 'cardData')

    def _fetch():
        return fetch_hub_models(filter_tag=None, pipeline_tag='sentence-similarity',
                                sort=sort, limit=limit, expand=expand)

    models = _cached_json(f"hub_popular_{sort}_{limit}", _fetch,
                          ttl_hours=ttl_hours, refresh=refresh)
    cands = [candidate_from_hub_model(m, source=f'popular:{sort}') for m in models]
    return cands


# --------------------------------------------------------------------------
# Source: mteb/results dataset via datasets-server
# --------------------------------------------------------------------------

def _english_result_row(row: Dict) -> bool:
    subset = (row.get('subset') or '').lower()
    langs = [str(l).lower() for l in (row.get('language') or [])]
    if subset not in _ENGLISH_CONFIGS:
        return False
    return not langs or any(l.startswith('eng') for l in langs)


def aggregate_results_rows(rows_by_task: Dict[str, List[Dict]], min_task_fraction: float = 0.7
                           ) -> List[EmbeddingCandidate]:
    """Collapse per-task rows from mteb/results into one candidate per model."""
    per_model: Dict[str, Dict[str, float]] = {}
    for task, rows in rows_by_task.items():
        for row in rows:
            if not _english_result_row(row):
                continue
            if row.get('is_public') is False:
                continue
            name = row.get('model_name')
            score = _normalise_score(row.get('score'))
            if not name or score is None:
                continue
            scores = per_model.setdefault(name, {})
            scores[task] = max(scores.get(task, -1.0), score)
    needed = max(1, math.ceil(min_task_fraction * len(rows_by_task)))
    cands = []
    for name, scores in per_model.items():
        if len(scores) < needed:
            continue
        cands.append(EmbeddingCandidate(
            model_id=name, sts_mean=statistics.mean(scores.values()),
            n_sts_tasks=len(scores), source='results',
        ))
    return sort_candidates(cands, dedupe=False)


def fetch_results_ranking(tasks: Sequence[str] = MTEB_ENG_V2_STS, per_task: int = 100,
                          timeout: float = 20.0, refresh: bool = False,
                          ttl_hours: float = DEFAULT_TTL_HOURS) -> List[EmbeddingCandidate]:
    """Query the mteb/results dataset server-side, one call per STS task."""
    def _fetch():
        rows_by_task: Dict[str, List[Dict]] = {}
        for task in tasks:
            params = {
                'dataset': 'mteb/results', 'config': 'default', 'split': 'train',
                'where': f"\"task_name\"='{task}' AND \"split\"='test'",
                'orderby': '"score" DESC', 'length': str(min(100, per_task)),
            }
            try:
                resp = requests.get(DATASETS_SERVER_FILTER, params=params, timeout=timeout)
            except requests.RequestException as e:
                raise SourceUnavailable(f"datasets-server unreachable: {e}") from e
            if resp.status_code != 200:
                raise SourceUnavailable(
                    f"datasets-server returned {resp.status_code} for {task}: {resp.text[:160]}"
                )
            payload = resp.json()
            if payload.get('error'):
                raise SourceUnavailable(f"datasets-server: {payload['error']}")
            rows_by_task[task] = [r.get('row', {}) for r in payload.get('rows', [])]
        return rows_by_task

    rows_by_task = _cached_json("mteb_results_sts", _fetch, ttl_hours=ttl_hours, refresh=refresh)
    return aggregate_results_rows(rows_by_task)


# --------------------------------------------------------------------------
# Source: mteb package (local aggregation of the full results repository)
# --------------------------------------------------------------------------

def fetch_mteb_ranking(tasks: Sequence[str] = MTEB_ENG_V2_STS,
                       refresh: bool = False) -> List[EmbeddingCandidate]:
    """Aggregate STS results with the `mteb` package.

    First run clones the results repository (~100k files) into MTEB_CACHE
    (default ~/.cache/mteb) and parsing it takes several minutes. Later runs
    reuse the clone unless refresh=True.
    """
    try:
        import mteb  # type: ignore
    except ImportError as e:
        raise SourceUnavailable("pip install mteb to use --source mteb") from e
    import warnings
    mteb_cache = os.path.expanduser(os.getenv('MTEB_CACHE', '~/.cache/mteb'))
    download_latest = refresh or not os.path.isdir(mteb_cache)

    def _fetch():
        # Parsing ~100k result files takes 15-20 minutes; cache the outcome.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = mteb.load_results(tasks=list(tasks), only_main_score=True,
                                        download_latest=download_latest)
            table = results.to_dataframe(format='long')
        return [c.to_dict() for c in rank_from_mteb_dataframe(table, tasks)]

    rows = _cached_json(f"mteb_ranking_{len(tasks)}", _fetch, refresh=refresh)
    return [
        EmbeddingCandidate(model_id=r['model_id'], sts_mean=r['sts_mean'],
                           n_sts_tasks=r['n_sts_tasks'], source='mteb')
        for r in rows
    ]


def rank_from_mteb_dataframe(table, tasks: Sequence[str]) -> List[EmbeddingCandidate]:
    """Turn BenchmarkResults.to_dataframe() output into candidates.

    Long format has model_name / task_name / score columns. Wide format (the
    mteb default) has one row per task and one column per model.
    """
    cols = [str(c) for c in table.columns]
    per_model: Dict[str, Dict[str, float]] = {}
    if 'task_name' in cols and 'score' in cols:
        model_col = 'model_name' if 'model_name' in cols else cols[0]
        for _, row in table.iterrows():
            task = str(row['task_name'])
            if task not in tasks:
                continue
            score = _normalise_score(row['score'])
            if score is None:
                continue
            per_model.setdefault(str(row[model_col]), {})[task] = score
    else:
        # Wide format: rows are tasks (a task_name column or the index),
        # every other column is a model.
        task_col = 'task_name' if 'task_name' in cols else None
        model_cols = [c for c in cols if c != task_col]
        for idx, row in table.iterrows():
            task = str(row[task_col]) if task_col else str(idx)
            if task not in tasks:
                continue
            for c in model_cols:
                score = _normalise_score(row[c])
                if score is not None:
                    per_model.setdefault(str(c), {})[task] = score
    needed = max(1, math.ceil(0.7 * len(tasks)))
    cands = [
        EmbeddingCandidate(model_id=name, sts_mean=statistics.mean(s.values()),
                           n_sts_tasks=len(s), source='mteb')
        for name, s in per_model.items() if len(s) >= needed
    ]
    return sort_candidates(cands, dedupe=False)


# --------------------------------------------------------------------------
# Enrichment + filtering
# --------------------------------------------------------------------------

def fetch_model_meta(model_id: str, refresh: bool = False,
                     ttl_hours: float = DEFAULT_TTL_HOURS) -> Optional[Dict]:
    """Per-model Hub metadata (library, params, gated, tags). None if not on the Hub."""
    expand = tuple(e for e in _HUB_EXPAND if e != 'cardData')

    def _fetch():
        params = [('expand[]', e) for e in expand]
        headers = {}
        token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        if token:
            headers['Authorization'] = f"Bearer {token}"
        resp = requests.get(f"{HUB_API}/{model_id}", params=params, headers=headers, timeout=30)
        if resp.status_code == 404:
            return {'missing': True}
        if resp.status_code != 200:
            raise SourceUnavailable(f"Hub returned {resp.status_code} for {model_id}")
        return resp.json()

    data = _cached_json(f"model_{model_id}", _fetch, ttl_hours=ttl_hours, refresh=refresh)
    return None if data.get('missing') else data


def enrich_with_hub(cands: List[EmbeddingCandidate], top_n: int = 40,
                    refresh: bool = False) -> List[EmbeddingCandidate]:
    """Fill library/params/gated for the first top_n candidates lacking them."""
    done = 0
    for c in cands:
        if done >= top_n:
            break
        if c.library is not None and c.params is not None:
            continue
        done += 1
        try:
            meta = fetch_model_meta(c.model_id, refresh=refresh)
        except SourceUnavailable as e:
            logger.debug("meta for %s unavailable: %s", c.model_id, e)
            continue
        if meta is None:
            c.library = None  # not a Hub repo (API-only model such as Cohere/OpenAI)
            continue
        filled = candidate_from_hub_model(meta, source=c.source)
        c.params = c.params if c.params is not None else filled.params
        c.downloads = filled.downloads
        c.likes = filled.likes
        c.library = filled.library
        c.gated = filled.gated
        c.custom_code = filled.custom_code
        c.gguf = filled.gguf
        c.pipeline_tag = filled.pipeline_tag
    return cands


def filter_candidates(cands: Iterable[EmbeddingCandidate], max_params: Optional[int] = None,
                      allow_gated: bool = False, allow_custom_code: bool = False,
                      require_sentence_transformers: bool = True,
                      min_tasks: int = 0) -> List[EmbeddingCandidate]:
    """Keep only models that can be run locally under the given constraints."""
    out = []
    for c in cands:
        if require_sentence_transformers and not c.runs_with_sentence_transformers:
            continue
        if not allow_gated and c.gated:
            continue
        if not allow_custom_code and c.custom_code:
            continue
        if max_params is not None and c.params is not None and c.params > max_params:
            continue
        # Popularity listings carry no STS scores; the task floor is meaningless there.
        if not c.is_popularity_only and c.n_sts_tasks < min_tasks:
            continue
        out.append(c)
    return out


def load_ranking(source: str = 'auto', refresh: bool = False, limit_models: int = 200
                 ) -> (List[EmbeddingCandidate], str):
    """Fetch a ranking. 'auto' tries results, then cards. Returns (candidates, source_used)."""
    if source == 'auto':
        try:
            cands = fetch_results_ranking(refresh=refresh)
            if cands:
                return cands, 'results'
        except SourceUnavailable as e:
            logger.warning("mteb/results via datasets-server unavailable (%s); "
                           "falling back to model cards.", e)
        return fetch_cards_ranking(limit_models=limit_models, refresh=refresh), 'cards'
    if source == 'results':
        return fetch_results_ranking(refresh=refresh), 'results'
    if source == 'cards':
        return fetch_cards_ranking(limit_models=limit_models, refresh=refresh), 'cards'
    if source == 'mteb':
        return fetch_mteb_ranking(refresh=refresh), 'mteb'
    if source in ('popular', 'downloads'):
        return fetch_popular('downloads', limit=limit_models, refresh=refresh), 'popular:downloads'
    if source == 'likes':
        return fetch_popular('likes', limit=limit_models, refresh=refresh), 'popular:likes'
    if source == 'trending':
        return fetch_popular('trendingScore', limit=limit_models, refresh=refresh), 'popular:trending'
    raise ValueError(f"unknown source {source!r}")


# --------------------------------------------------------------------------
# Local verification
# --------------------------------------------------------------------------

def check_model(model_id: str, trust_remote_code: Optional[bool] = None) -> Dict:
    """Load a model with sentence-transformers and report dim, speed and a sanity check."""
    from novelty_detector import Embedder
    t0 = time.time()
    emb = Embedder(model_id, trust_remote_code=trust_remote_code)
    load_s = time.time() - t0
    probe = [
        "Quantum entanglement underpins quantum cryptography protocols.",
        "Entangled particles are the basis of quantum key distribution.",
        "The recipe calls for two cups of flour and a pinch of salt.",
    ] * 6
    t1 = time.time()
    vecs = emb.encode(probe)
    enc_s = time.time() - t1
    para = float(vecs[0] @ vecs[1])
    unrelated = float(vecs[0] @ vecs[2])
    return {
        'model': model_id,
        'dim': emb.get_sentence_embedding_dimension(),
        'max_seq_length': emb.max_seq_length,
        'load_seconds': round(load_s, 2),
        'encode_seconds_per_100': round(enc_s / len(probe) * 100, 3),
        'cosine_paraphrase': round(para, 3),
        'cosine_unrelated': round(unrelated, 3),
        'separates_paraphrase_from_unrelated': para > unrelated + 0.1,
    }

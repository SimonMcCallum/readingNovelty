"""
Choose the local embedding model from a current Hugging Face ranking.

  python embedding_models_cli.py list                       # best STS models you can run locally
  python embedding_models_cli.py list --max-params 500M     # cap the size
  python embedding_models_cli.py list --source popular      # what people download
  python embedding_models_cli.py check BAAI/bge-base-en-v1.5
  python embedding_models_cli.py set   BAAI/bge-base-en-v1.5   # writes EMBEDDING_MODEL to .env
  python embedding_models_cli.py current

`list` defaults to `--source auto`: the mteb/results dataset (complete) with a
fallback to model-card metadata (fast, no dependency). `--source mteb` uses the
mteb package for an authoritative local aggregation (large first download).

The ranking is by mean Spearman (x100) over the English MTEB STS tasks, which
is the benchmark family closest to this project's "distance to nearest
neighbour" use. Nothing about your documents is sent anywhere.
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import List, Optional

from dotenv import load_dotenv

import embedding_models as em
from novelty_detector import DEFAULT_EMBEDDING_MODEL, resolve_embedding_model_name

logger = logging.getLogger(__name__)


def _print_table(cands: List[em.EmbeddingCandidate], source: str, current: str):
    if source.startswith('popular'):
        print(f"Source: {source}   (popularity only; no quality signal)")
    else:
        print(f"Source: {source}   (ranking = mean Spearman x100 over English MTEB STS tasks)")
    print(f"{'#':>3} {'STS':>6} {'tasks':>5} {'params':>7} {'downloads':>10}  model")
    for i, c in enumerate(cands, 1):
        flags = []
        if c.gated:
            flags.append('gated')
        if c.custom_code:
            flags.append('remote-code')
        if c.gguf:
            flags.append('gguf')
        if not c.runs_with_sentence_transformers:
            flags.append(f"lib={c.library or '?'}")
        mark = ' *' if c.model_id.split('/')[-1] == current.split('/')[-1] else ''
        sts = f"{c.sts_mean:6.2f}" if c.sts_mean is not None else '     -'
        dl = f"{c.downloads:>10,}" if c.downloads is not None else f"{'?':>10}"
        line = f"{i:>3} {sts} {c.n_sts_tasks:>5} {em.human_params(c.params):>7} {dl}  {c.model_id}{mark}"
        if flags:
            line += f"   [{', '.join(flags)}]"
        print(line)
    if any(c.model_id.split('/')[-1] == current.split('/')[-1] for c in cands):
        print("  * = current EMBEDDING_MODEL")


def cmd_list(args) -> int:
    try:
        cands, source = em.load_ranking(args.source, refresh=args.refresh,
                                        limit_models=args.fetch)
    except em.SourceUnavailable as e:
        print(f"Source unavailable: {e}", file=sys.stderr)
        return 2
    if source in ('results', 'mteb'):
        # These sources know scores but not what is runnable; ask the Hub.
        em.enrich_with_hub(cands, top_n=args.enrich, refresh=args.refresh)
    filtered = em.filter_candidates(
        cands,
        max_params=em.parse_params(args.max_params),
        allow_gated=args.allow_gated,
        allow_custom_code=args.allow_remote_code,
        require_sentence_transformers=not args.all_libraries,
        min_tasks=args.min_tasks,
    )
    if source in ('results', 'mteb'):
        # Only enriched rows carry library info; unenriched rows were dropped
        # by the sentence-transformers filter. Say so rather than hide it.
        dropped_unknown = sum(1 for c in cands if c.library is None and c.params is None)
        if dropped_unknown and not args.all_libraries:
            logger.info("%d lower-ranked models were not checked against the Hub "
                        "(raise --enrich to include them).", dropped_unknown)
    shown = filtered[:args.limit]
    current = resolve_embedding_model_name()
    if args.json:
        print(json.dumps({'source': source, 'current': current,
                          'models': [c.to_dict() for c in shown]}, indent=2))
    else:
        _print_table(shown, source, current)
        if shown and not source.startswith('popular'):
            best = shown[0]
            print(f"\nRecommended: {best.model_id}")
            print(f"  verify:  python embedding_models_cli.py check {best.model_id}")
            print(f"  adopt:   python embedding_models_cli.py set {best.model_id}")
    return 0


def cmd_check(args) -> int:
    trust = True if args.trust_remote_code else None
    try:
        report = em.check_model(args.model, trust_remote_code=trust)
    except Exception as e:  # model load errors are varied; show them plainly
        print(f"FAILED to load {args.model}: {e}", file=sys.stderr)
        if 'trust_remote_code' in str(e):
            print("Retry with --trust-remote-code (and set EMBEDDING_TRUST_REMOTE_CODE=1).",
                  file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for k, v in report.items():
            print(f"  {k:38s} {v}")
        if not report['separates_paraphrase_from_unrelated']:
            print("WARNING: paraphrase and unrelated sentences are not well separated; "
                  "this model is a poor fit for novelty scoring.")
    return 0


def _upsert_env(path: str, key: str, value: str) -> str:
    """Set key=value in a dotenv file, replacing an existing (possibly commented) line."""
    lines: List[str] = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            lines = fh.read().splitlines()
    pattern = re.compile(rf'^\s*#?\s*{re.escape(key)}\s*=')
    new_line = f"{key}={value}"
    replaced = False
    for i, line in enumerate(lines):
        if pattern.match(line):
            if not replaced:
                lines[i] = new_line
                replaced = True
            else:
                lines[i] = f"# {line.lstrip('# ')}" if not line.lstrip().startswith('#') else line
    if not replaced:
        lines.append(new_line)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return 'updated' if replaced else 'added'


def cmd_set(args) -> int:
    env_path = args.env_file
    if not os.path.exists(env_path) and os.path.exists('.env.example') and env_path == '.env':
        with open('.env.example', 'r', encoding='utf-8') as src, \
                open(env_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        print("Created .env from .env.example")
    previous = resolve_embedding_model_name()
    action = _upsert_env(env_path, 'EMBEDDING_MODEL', args.model)
    print(f"{action} EMBEDDING_MODEL={args.model} in {env_path}")
    if previous != args.model:
        print(f"Previous model: {previous}")
        print("Existing corpora under CORPUS_DIR were embedded with the previous model. "
              "Rebuild each partition (POST /assignments/<id>/rebuild_index) or point "
              "CORPUS_DIR at a fresh directory before scoring.")
    return 0


def cmd_current(args) -> int:
    name = resolve_embedding_model_name()
    print(f"EMBEDDING_MODEL = {name}" + ("  (project default)" if name == DEFAULT_EMBEDDING_MODEL
                                         and not os.getenv('EMBEDDING_MODEL') else ''))
    print(f"EMBEDDING_DEVICE = {os.getenv('EMBEDDING_DEVICE') or '<auto>'}")
    print(f"EMBEDDING_TRUST_REMOTE_CODE = {os.getenv('EMBEDDING_TRUST_REMOTE_CODE', '0')}")
    print(f"cache dir = {em.cache_dir()}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command', required=True)

    p_list = sub.add_parser('list', help='Rank similarity models from Hugging Face.')
    p_list.add_argument('--source', default='auto',
                        choices=['auto', 'results', 'cards', 'mteb', 'popular', 'likes', 'trending'])
    p_list.add_argument('--limit', type=int, default=20, help='Rows to show (default 20).')
    p_list.add_argument('--fetch', type=int, default=200,
                        help='Hub records to fetch per sort key before ranking (default 200).')
    p_list.add_argument('--enrich', type=int, default=40,
                        help='For results/mteb sources: how many top models to look up on the Hub.')
    p_list.add_argument('--max-params', help='Drop models above this size, e.g. 500M or 1.5B.')
    p_list.add_argument('--min-tasks', type=int, default=5,
                        help='Require at least this many STS tasks behind the mean (default 5).')
    p_list.add_argument('--allow-gated', action='store_true',
                        help='Include models that need a Hugging Face licence click (e.g. Gemma).')
    p_list.add_argument('--allow-remote-code', action='store_true',
                        help='Include models that need trust_remote_code=True.')
    p_list.add_argument('--all-libraries', action='store_true',
                        help='Do not require the sentence-transformers library tag.')
    p_list.add_argument('--refresh', action='store_true', help='Ignore the 24h disk cache.')
    p_list.add_argument('--json', action='store_true')
    p_list.set_defaults(func=cmd_list)

    p_check = sub.add_parser('check', help='Load a model locally and report dim/speed/sanity.')
    p_check.add_argument('model')
    p_check.add_argument('--trust-remote-code', action='store_true')
    p_check.add_argument('--json', action='store_true')
    p_check.set_defaults(func=cmd_check)

    p_set = sub.add_parser('set', help='Write EMBEDDING_MODEL into .env.')
    p_set.add_argument('model')
    p_set.add_argument('--env-file', default='.env')
    p_set.set_defaults(func=cmd_set)

    p_cur = sub.add_parser('current', help='Show the embedding model in effect.')
    p_cur.set_defaults(func=cmd_current)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())

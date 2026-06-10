# readingNovelty

PDF novelty detection that runs locally and integrates with Canvas
assessment workflows. Scores each paragraph of a PDF for how *novel* it
is relative to a chosen reference corpus, then writes an annotated PDF
with colour-coded highlights.

The system supports four use cases from one shared core:

| Use case | Entry point | Reference corpus |
|---|---|---|
| Canvas-style cohort assessment | `server.py` (`POST /assignments/<id>/submissions`) or `assess_cli.py` | Other submissions to the same assignment |
| PhD "what's new in this paper given what I've read" | `read_folder_cli.py` | A folder of PDFs you have already read |
| Published paper vs. its own prior art | `citation_novelty_cli.py` | Abstracts of the paper's references (via Semantic Scholar) |
| Ad-hoc intra-document novelty | `POST /upload` | The PDF's own other paragraphs |

## Privacy posture

Submissions and reference content may be copyrighted. The default mode
(`LOCAL_ONLY=1`) keeps everything on the host:

- Cloud providers (Anthropic, OpenAI, Gemini) and remote Ollama are
  **excluded from provider discovery**, even when their env vars are set.
  The server logs a warning so you can see which vars were ignored.
- Sentence-Transformers embeddings download once, then run offline.
- The corpus (SQLite + per-assignment FAISS) is stored under `./corpus/`.
- Citation-graph mode hits the public Semantic Scholar API for cited
  papers' abstracts. This is anonymous, public-paper metadata — no
  submission content is sent. If even that is unacceptable, skip
  citation-graph mode and use the PhD read-folder mode instead.

Set `LOCAL_ONLY=0` only when processing material you are licensed to send
to a third-party API. The assessment endpoints will still refuse to run
on the fallback (keyword-only) provider so a misconfigured grader path
can never silently degrade.

## Quickstart

1. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure** `.env`:

   ```bash
   cp .env.example .env
   # The defaults work if you run Ollama locally on port 11434.
   ```

3. **Start a local LLM**:

   ```bash
   ollama serve            # in one terminal
   ollama pull qwen2.5:7b  # one-time model fetch
   ```

4. **Start the server** (only needed for the API + reader view):

   ```bash
   python server.py
   ```

   Watch the startup logs for `PRIVACY: LOCAL_ONLY=1` and
   `Active provider: ollama-local`. If you see `Active provider: fallback`,
   Ollama isn't reachable — the assessment endpoints will return 503
   until that's fixed.

5. **Verify the citation-graph pipeline** against the live Semantic
   Scholar API:

   ```bash
   python smoke_test_s2.py
   ```

   ~30 seconds. Builds a real citation corpus for the BERT paper,
   scores two synthesized PDFs against it, asserts that an unrelated
   topic scores higher novelty than a BERT paraphrase.

## How novelty is computed

The shared math: for each chunk of text in the target PDF, embed it,
search a FAISS index for nearest neighbours in the corpus, convert
average L2 distance to a novelty score in `[0, 1]`:

```
novelty = 1 - exp(-avg_distance / 2.0)
```

What differs between modes is **which corpus** the chunk is scored
against:

- **Cohort mode** (`POST /assignments/<id>/submissions`): scored against
  prior submissions to the same assignment. Each chunk goes through an
  LLM "regeneration prompt" first, and the prompt's embedding (not the
  chunk's) is matched against prior prompts. This dampens stylistic
  matches and emphasizes conceptual overlap.
- **PhD mode** (`read_folder_cli.py`): same machinery as cohort mode,
  but the corpus is a folder of read PDFs you control.
- **Citation-graph mode** (`citation_novelty_cli.py`): scored against
  the *abstracts* of the paper's references. Skips the LLM prompt step
  deliberately — we want raw semantic distance to the prior-art
  summaries, not paraphrase-of-paraphrase distance.

You can blend the corpus-based score with an LLM-predictive score using
the `--alpha` knob (`read_folder_cli.py`):

```
final = alpha * corpus_novelty + (1 - alpha) * llm_novelty
```

The LLM-predictive score is `1 - cosine(predicted_chunk, actual_chunk)`,
where the LLM is asked to fill the gap between the chunk's neighbours
given a five-word topic hint. High LLM-novelty means the LLM, even with
the surrounding context, couldn't reconstruct what you actually wrote.
"Interesting" and "incorrect" both manifest as high LLM-novelty — they
are not distinguished by the score alone.

## Canvas assessment workflow

Two integration shapes, both using a *lecturer-owned* Canvas access
token (no LMS admin involvement, no LTI registration):

### Via the CLI (recommended)

```bash
# One time: create a Canvas access token at
# Canvas → Account → Settings → New Access Token.
# Set it as CANVAS_TOKEN in .env (plus CANVAS_BASE_URL).

python assess_cli.py \
    --course 12345 \
    --assignment 67890 \
    --internal-id cs101-essay1-2026 \
    --dry-run    # remove --dry-run when scores look right
```

The CLI fetches every submission, scores each PDF against the cohort
corpus (`internal-id` partitions the corpus on disk), and posts the
annotated PDF + a `[novelty-bot]` summary back as a submission comment.
Subsequent runs skip already-assessed submissions; pass `--rescore` to
force a re-process.

### Via the HTTP API

```bash
# Register an assignment partition
curl -X POST http://localhost:5000/assignments \
    -H "Content-Type: application/json" \
    -d '{"assignment_id": "cs101-essay1-2026", "name": "Essay 1"}'

# Score a single PDF
curl -X POST \
    -F "file=@submission.pdf" \
    -F "student_id=alice" \
    http://localhost:5000/assignments/cs101-essay1-2026/submissions

# Grader's reader view
open http://localhost:5000/assignments/cs101-essay1-2026/reader
```

## PhD "papers I have read" workflow

```bash
# First time — bulk-ingest your read folder
python read_folder_cli.py \
    --read-folder ~/papers/read \
    --corpus-id phd_alice

# Routine use — drop a new PDF into ~/papers/inbox, score it
python read_folder_cli.py \
    --read-folder ~/papers/read \
    --corpus-id phd_alice \
    --new ~/papers/inbox/just_arrived.pdf \
    --alpha 0.7 \
    --output annotated.pdf \
    --report-json report.json
```

`--alpha 1.0` (default) uses only the corpus signal. Lowering it
(`--alpha 0.7`, `--alpha 0.5`, `--alpha 0.0`) blends in the
LLM-predictive signal. Ingestion is idempotent — re-runs skip PDFs
already in the corpus unless you pass `--reingest`.

## Citation-graph novelty workflow

```bash
# Score a paper against the abstracts of its own references
python citation_novelty_cli.py \
    --pdf attention_is_all_you_need.pdf \
    --paper-doi 10.18653/v1/N19-1423 \
    --target-year 2019 \
    --output annotated.pdf \
    --report-json report.json
```

- If you omit `--paper-doi`, the first DOI found inside the PDF text is
  used.
- `--target-year` filters references to those published strictly before
  that year — useful for modelling "what was knowable at submission
  time."
- The Semantic Scholar response cache lives at `corpus/s2_cache/` by
  default so re-runs are fast and respectful of the public API rate
  limit. Set `--rate-limit 1.5` (or higher) if you're hitting 429s
  without an API key. Pass `SEMANTIC_SCHOLAR_API_KEY` in the
  environment for higher limits.

Notes on identifiers:

- Plain DOIs (`10.18653/v1/N19-1423`), arXiv ids (`1706.03762`), and
  prefixed forms (`DOI:...`, `ARXIV:...`) all work. The arXiv *DOI*
  form (`10.48550/arXiv.<id>`) does NOT — pass the arXiv id directly
  or with the `ARXIV:` prefix instead.
- Semantic Scholar does not have abstracts for every cited paper.
  Expect ~60–70% coverage in NLP/ML; humanities and older work fares
  worse. The CLI logs how many references were ingested vs. skipped.

## HTTP endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Privacy posture + active provider + `assessment_ready` |
| GET | `/providers` | List discovered providers |
| POST | `/upload` | Legacy intra-document novelty |
| POST | `/analyze` | Intra-document novelty on a JSON-supplied string |
| POST | `/compare` | Run novelty with multiple providers for A/B |
| POST | `/assignments` | Register an assignment partition |
| GET | `/assignments/<id>` | Assignment metadata + chunk counts |
| POST | `/assignments/<id>/submissions` | Cohort assessment entry point |
| GET | `/assignments/<id>/submissions` | List submissions in order |
| GET | `/assignments/<id>/submissions/<sid>` | Per-chunk novelty detail |
| GET | `/assignments/<id>/reader` | Grader-facing HTML view |
| GET | `/download/<filename>` | Download a generated annotated PDF |

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `LOCAL_ONLY` | `1` | Gate cloud providers (Anthropic, OpenAI, Gemini, ollama-remote). Default on. |
| `OLLAMA_HOST` | `localhost` | Local Ollama hostname. |
| `OLLAMA_PORT` | `11434` | Local Ollama port. |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Local model name. |
| `ANTHROPIC_API_KEY` | unset | Used only when `LOCAL_ONLY=0`. |
| `OPENAI_API_KEY` | unset | Used only when `LOCAL_ONLY=0`. |
| `GEMINI_API_KEY` | unset | Used only when `LOCAL_ONLY=0`. Free-tier `gemini-2.5-flash` works. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Override the Gemini model id. |
| `OLLAMA_REMOTE_URL` | unset | Used only when `LOCAL_ONLY=0`. |
| `CORPUS_DIR` | `corpus` | Where SQLite + per-assignment FAISS indices live. |
| `SEMANTIC_SCHOLAR_API_KEY` | unset | Higher S2 rate limit. Optional; the CLI works without it. |
| `S2_CACHE_DIR` | `<corpus>/s2_cache` | Disk cache for S2 responses. |
| `CANVAS_BASE_URL` | unset | E.g. `https://canvas.example.edu`. Required for `assess_cli.py`. |
| `CANVAS_TOKEN` | unset | Lecturer access token. Required for `assess_cli.py`. |
| `FLASK_DEBUG` | `0` | Flask debug mode. |
| `UPLOAD_FOLDER` | `uploads` | Where uploaded + annotated PDFs live. |
| `MAX_CONTENT_LENGTH` | `16777216` | Max upload size in bytes (16 MiB). |

## Tests

```bash
python -m pytest -q
```

The non-network suite runs in ~2 minutes (113 tests). The live S2 check
is a separate script (`python smoke_test_s2.py`).

## Known limitations

1. **Predictive novelty is serial.** Each chunk triggers one extra LLM
   call. Fine for local Ollama; expect 4+ minutes per 30-chunk PDF on
   Gemini free tier.
2. **Annotation needs literal page-level matches.** PyMuPDF's
   `search_for` cannot match text that spans a page break. Such chunks
   appear in the summary but are not highlighted on the body. The
   per-call report dict carries `chunks_unmatched` so you can surface
   this.
3. **Reupload does not shrink the FAISS index.** Old embedding rows of
   replaced submissions stay searchable until a manual rebuild. Matters
   only if students resubmit often.
4. **First submitter in a fresh cohort scores 1.0 on every chunk.**
   Honest to the empty-corpus case; the submission comment carries the
   `corpus_size_at_scoring` field so graders see the context.
5. **Semantic Scholar abstract coverage is uneven.** Cited papers
   without an abstract are dropped from the citation corpus. The CLI
   logs the count so you know when the signal is weak.
6. **Citation-graph rate limit.** Unkeyed S2 calls are throttled to
   roughly one per second. The default `--rate-limit 1.0` will hit 429s
   on bursty access; bump to 1.5 or set `SEMANTIC_SCHOLAR_API_KEY`.

## Project layout

```
.
├── server.py                  # Flask API + reader view
├── novelty_detector.py        # Core scoring (corpus + LLM-predictive + blend)
├── pdf_processor.py           # Extract, chunk, highlight (PyMuPDF)
├── llm_providers.py           # Provider abstraction + LOCAL_ONLY gating
├── corpus.py                  # SQLite + per-assignment FAISS
├── canvas_client.py           # Lecturer-token Canvas API wrapper
├── assess_cli.py              # Canvas assessment runner
├── read_folder_cli.py         # PhD personal corpus workflow
├── citation_novelty_cli.py    # Citation-graph novelty workflow
├── citation_extractor.py      # DOI / arXiv id extraction from text
├── citation_fetcher.py        # Semantic Scholar client
├── smoke_test_s2.py           # Optional live S2 verification
└── test_*.py                  # Unit + integration tests
```

## License

MIT

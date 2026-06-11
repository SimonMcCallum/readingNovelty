# Testing the Canvas ingest

Step-by-step procedure for the first real run of `assess_cli.py`
against a Canvas instance. Designed to fail safe — at each step there
is a kill switch before any student-visible action.

## 0. Pre-flight checklist

Before you start, confirm:

- `LOCAL_ONLY=1` in `.env` (default).
- Ollama is running locally and `python server.py` prints
  `Active provider: ollama-local` in its startup logs (you don't need
  the server to be *running* for the CLI, but use it once to confirm
  Ollama is reachable).
- You have a **test course** with at least one PDF submission to a
  test assignment. Do **not** point the first run at a live course
  with real students. Most Canvas instances let you self-enrol as
  teacher of a personal sandbox course — use that.

## 1. Create a Canvas access token

In Canvas:

> **Account → Settings → Approved Integrations → New Access Token**

- Purpose: `readingNovelty assessment CLI`
- Expires: pick a date 1–4 weeks out; tokens can always be revoked.

**Copy the token immediately — Canvas only shows it once.** Treat it
like a password: it grants every action your account can take, in
every course you can see.

## 2. Configure environment

Add to `.env`:

```ini
CANVAS_BASE_URL=https://canvas.your-institution.edu
CANVAS_TOKEN=paste_your_token_here
```

`.env` is gitignored — verify with `git check-ignore .env`.

## 3. Find the course id and assignment id

Open the assignment in Canvas. The URL is:

```
https://canvas.your-institution.edu/courses/<COURSE_ID>/assignments/<ASSIGNMENT_ID>
```

Both ids are integers visible in the URL. Note them down.

## 4. Preflight check

This validates your token, fetches the assignment, lists submissions,
and reports *what would happen* — without scoring or posting anything.

```bash
python assess_cli.py \
    --course 12345 \
    --assignment 67890 \
    --internal-id sandbox-test-2026 \
    --preflight
```

You should see something like:

```
Canvas base URL: https://canvas.your-institution.edu
Token belongs to: Your Name (id=987, email=you@x.edu)
Assignment: 'Essay 1' (id=67890)
  Due: 2026-06-30T23:59:00Z
  Submission types: ['online_upload']

Submissions found: 23
  with attachments: 21
  with PDF attachment: 19
  already assessed (will skip unless --rescore): 0
  no attachment (will skip): 2

A real run with these flags would process: 19 submissions
```

**If any of those numbers look wrong, stop here.** Common issues:

- `Token belongs to: <name>` is the wrong account → wrong token.
- `cannot fetch assignment` → wrong course/assignment id, or your
  account is not enrolled as teacher/TA in that course.
- `with PDF attachment: 0` → submissions are not PDFs. The CLI will
  skip them. Either the assignment type is different (Text Entry?
  Website URL?) or students uploaded `.docx` files.

## 5. Dry run

Process submissions locally without posting any comments back to Canvas.

```bash
python assess_cli.py \
    --course 12345 \
    --assignment 67890 \
    --internal-id sandbox-test-2026 \
    --dry-run --verbose
```

For each PDF you should see logs like:

```
INFO assess_cli: DRY RUN — would post on user 555:
[novelty-bot] Novelty assessment vs. cohort
Average novelty: 0.78 (high)
Chunks scored: 18
Corpus size at scoring time: 0 chunks from prior submissions
...
```

Open the temporary working directory (printed in `--verbose`) and
inspect the annotated PDFs before deciding the scores look sensible.

> First-submitter quirk: every chunk scores 1.0 (max novelty) because
> the cohort corpus is empty at scoring time. This is correct — there
> is nothing to compare against yet. Score variability appears from
> submission 2 onward.

## 6. Single-submission real run

The big risk in step 7 is posting *many* comments to real students at
once. Mitigate by processing only one submission first:

```bash
python assess_cli.py \
    --course 12345 \
    --assignment 67890 \
    --internal-id sandbox-test-2026 \
    --limit 1
```

The CLI stops after the first PDF is fully processed and posted.
Verify in Canvas:

1. Open that student's submission in **SpeedGrader**.
2. There should be a new **comment** ending with the `[novelty-bot]`
   summary text.
3. Attached to the comment is the annotated PDF — open it and confirm
   the highlights look right.

## 7. Full cohort

When step 6 looks right:

```bash
python assess_cli.py \
    --course 12345 \
    --assignment 67890 \
    --internal-id sandbox-test-2026
```

Re-runs are safe: every comment carries the marker `[novelty-bot]`,
and the CLI skips any submission with such a comment unless you pass
`--rescore`.

## 8. Recovery and re-runs

- **Re-process everything** (after a config change):
  ```bash
  python assess_cli.py --course ... --assignment ... --internal-id ... --rescore
  ```
- **Forget a botched cohort** (drop the corpus partition and start
  over):
  ```bash
  # Stop the server first if it's running, then:
  rm -rf corpus/sandbox-test-2026
  sqlite3 corpus/corpus.db "DELETE FROM chunks WHERE assignment_id='sandbox-test-2026';
                            DELETE FROM submissions WHERE assignment_id='sandbox-test-2026';
                            DELETE FROM assignments WHERE assignment_id='sandbox-test-2026';"
  ```
- **Remove the bot comments from Canvas**: there is no bulk API. Use
  SpeedGrader to delete them individually, or filter via the Canvas
  data export and ask Canvas Support if there are many.

## 9. Revoke the token when done

Account → Settings → Approved Integrations → click the trash icon
next to the token. The CLI cannot post comments without it.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Token check failed: 401` | Token expired or revoked. Create a new one. |
| `Cannot fetch assignment ... 404` | Wrong course/assignment id, or you don't have permission. |
| `No PDF attachment` (in dry run) | Submissions are .docx, .txt, or quiz responses — CLI handles PDFs only. |
| `Active provider: fallback` at startup | Ollama isn't reachable. Start `ollama serve`. |
| Posted comment but no attached PDF | Canvas's three-step file upload can race. Check the logs for the file upload step. Re-run with `--rescore` if needed. |
| Annotated PDF has no highlights | Chunks may span page breaks. Look at the `chunks_unmatched` count in the logs. |
| Submissions left out | `--limit` was set, or some students have no PDF attachment. Run `--preflight` to see counts. |

## What the CLI does not do

- **It does not assign grades.** Only posts comments. If you want a
  rubric entry or score, build that on top with `PUT /submissions`
  and a `submission[posted_grade]` field — not implemented here.
- **It does not pull from group submissions.** Each student id is
  treated as a separate submitter. Groupwork support would need
  Canvas's group-set endpoints.
- **It does not respect Canvas mute/post policy.** Comments post
  immediately even if the assignment is muted. Consider the policy
  setting before running on real students.

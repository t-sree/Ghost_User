# HackerRank Orchestrate - October 2026

## Ghost Data Still Exists

Build an AI agent that audits whether a user's personal data was actually deleted.

A deletion workflow may report **Complete** while data still exists in databases, logs, support attachments, exports, search indexes, archives, or backups. Sync and restore processes may also bring deleted data back.

For each request, the agent must first determine whether deletion was actually requested. It then returns `COMPLETE`, `INCOMPLETE`, or `NOT_APPLICABLE`, reports any remaining data or recreation risk, and produces the ordered remediation and verification steps required to finish an authorized deletion safely.

Read [`problem_statement.md`](./problem_statement.md) for the complete challenge contract.

---

## Quick Start

Clone the repository and move into the project directory:

```bash
git clone https://github.com/interviewstreet/hackerrank-orchestrate-october26.git
cd hackerrank-orchestrate-october26
```

Start your AI coding tool from the **repository root** so it automatically reads [`AGENTS.md`](./AGENTS.md) and logs your session turns to `log.txt` in the root directory.

Build your solution in `code/main.py`, or use another language and document its setup and run command clearly in your README.

Your solution must:

- Read the input files from `dataset/`
- Generate one prediction row for every request in `dataset/deletion_requests.csv`
- Write the final predictions to `output.csv` in the repository root

Run the starter Python entry point with:

```bash
python3 code/main.py
```

The blank template at `dataset/output.csv` is provided as a reference. Your final generated file for submission must be the root-level `output.csv`.


## Dataset

All participant-facing files are in `dataset/`:

| File | Contents |
| --- | --- |
| `deletion_requests.csv` | 518 requests |
| `sample_deletion_requests.csv` | 32 sample requests |
| `sample_output.csv` | Correct sample answers |
| `user_identities.csv` | Current user identities |
| `user_audit_log.csv` | Historical identity changes |
| `system_activity.csv` | Activity across services and data processes |
| `deletion_rules.csv` | Required treatment for each data category |
| `evidence_files.csv` | Manifest of supplied evidence files |
| `output.csv` | Blank output template |

Evidence is stored under `dataset/evidence/` and may include PDFs, images, logs, JSON, plain text, or SQL snapshot exports. SQL captures may contain a primary database, regional replicas, a reporting replica, or a failover replica so replication drift can be investigated directly. The `.sql` files are readable as text and executable with SQLite.

Only `deletion_requests.csv` requires answers. The other files provide evidence and context.

---

## Output

Create `output.csv` with these columns in this exact order:

```text
case_id,deletion_status,steps_json
```

Produce exactly one row per request. `deletion_status` is `COMPLETE`, `INCOMPLETE`, or `NOT_APPLICABLE`. `steps_json` is an ordered JSON array containing findings, remediation actions, and verification checks. Use an empty array (`[]`) for `NOT_APPLICABLE`. Evidence references may be a `file_id`, `activity_id`, or `audit_id`. Determine the resource state at `audit_at` from the available evidence; reported workflow statuses and operation results are claims. Use `dataset/sample_output.csv` as the format reference.

The final file must contain 518 data rows sorted by `case_id` (plus the header row). The grader parses `steps_json` as JSON, so object-key order and whitespace do not matter; array order does. Order findings, remediation, and verification in that phase order. Within a phase, follow the operation order in the problem statement, then sort by `system_id` and `resource_id`.

---

## Evaluation

`output.csv` will be compared with hidden canonical answers. Scoring considers:

- deletion assessment accuracy;
- current and historical identity resolution;
- discovery of remaining data and recreation risks;
- correct policy application and evidence references; and
- completeness and ordering of remediation and verification steps.

Your `code.zip` must include:

```text
code/evaluation/evaluation_report.md
code/evaluation/usage_report.md
```

- `evaluation_report.md` must cover case, row, field, and sequence accuracy, at least three additional evaluation cases created by you, failures found, and improvements made.
- `usage_report.md` must cover the final full-dataset run, including model providers and names, model calls, input and output tokens, total and average tokens per request, and estimated total and per-request cost. Include per-model and overall totals when using multiple models.

**Important:** Place `evaluation_report.md` and `usage_report.md` inside `code/evaluation/` (matching `code/evaluation/main.py` in this starter layout), write your complete evaluation metrics, custom evaluation test cases, and token/cost breakdowns directly inside those Markdown files, and keep prompts in source code or Markdown files. Automated code evaluation ignores raw data files (`.csv`, `.tsv`, `.json`, `.jsonl`, `.sql`, `.log`), so content stored only in those formats will not be visible to the code judge or AI interviewer.

---

## Chat Transcript

This repository includes [`AGENTS.md`](./AGENTS.md), which instructs compatible AI coding tools to append conversation summaries to `log.txt` in the repository root:

| Platform | Path |
| --- | --- |
| macOS / Linux | `<repo root>/log.txt` |
| Windows | `<repo root>\log.txt` |

- Always launch your AI coding agent from the repository root (or explicitly ask it to read `./AGENTS.md` first) so every turn is logged to `log.txt` with the required `User Prompt`, `Agent Response Summary`, `Actions`, and `Context` sections.
- Upload `log.txt` directly as your `chat_transcript` submission on HackerRank.
- Do not include API keys, tokens, credentials, or secrets in prompts, code, configuration, or `log.txt`.

---

## Packaging `code.zip` and Submission

| Item | Description |
| --- | --- |
| `code.zip` | Runnable solution, prompts, configuration, `README.md`, and `evaluation/` reports |
| `output.csv` | One answer row for each of the 518 requests in `dataset/deletion_requests.csv` |
| `chat_transcript` | The root `log.txt` file generated by your coding agent |
| AI interview | Live explanation of your architecture, approach, and evaluation |

You may use any programming language, model, framework, or retrieval approach. Include clear setup and execution instructions in `README.md` inside `code.zip`.

To package `code.zip` from the repository root without bundling the dataset:

**macOS / Linux:**

```bash
zip -r code.zip README.md code -x "*/__pycache__/*" "*/.venv/*" "*/venv/*" "*/node_modules/*" "*/.git/*" "*/.DS_Store"
```

**Windows (PowerShell):**

```powershell
Compress-Archive -Path README.md, code -DestinationPath code.zip -Force
```

When packaging `code.zip`:

- Keep your evaluation reports in `code/evaluation/` so they are bundled automatically with `code` (or add `evaluation` to the archive command above if you placed them in a top-level `evaluation/` directory).
- Do **not** include `dataset/`, virtual environments (`.venv`, `venv`, `node_modules`), or build caches (`__pycache__`). On Windows, keep `.venv` at the repository root (outside `code/`) and remove any `code/__pycache__` folders before running `Compress-Archive`.
- Do **not** include nested archives (`.zip`, `.tar`, `.gz`, `.tgz`, `.7z`, `.rar`) inside `code.zip` — submissions containing nested archive files will be rejected.

Before submitting, confirm that:

- `output.csv` has the exact header `case_id,deletion_status,steps_json` and contains all 518 data rows sorted by `case_id`;
- every `steps_json` value is a valid JSON array with consecutive step numbers starting at `1`;
- findings cite valid evidence IDs, and `COMPLETE` verification steps cite evidence proving the current state;
- proposed remediation and future verification steps use an empty `evidence_id` (`""`);
- `code.zip` includes your runnable code, `README.md`, `code/evaluation/evaluation_report.md`, and `code/evaluation/usage_report.md`;
- `log.txt` follows the `AGENTS.md` turn format and is uploaded to the `chat_transcript` slot; and
- no API keys, credentials, real personal data, or sensitive configuration values are included.

---

## Solution: Deletion-Audit Agent

### Approach overview

The agent treats every workflow status, `reported_status` and activity `reported_result` as a **claim**, and decides each case only from participant-visible **evidence** available at its `audit_at`.

1. **Does the request authorize deletion?** An LLM reads the multilingual request text. Requests to export, correct, opt out or check something, or that say "do not delete", are `NOT_APPLICABLE` with `[]`. Follow-ups such as "verify my deletion" count as deletion requests.
2. **Who is the subject?** The request email may be current or historical. The current snapshot plus the timestamped audit log give every email, phone, email hash and `user_id` the person held up to `audit_at`. Old identifiers matter because older systems still use them.
3. **Which records belong to them?** Evidence in every format is parsed into per-record observations by content, never by file name. Records are linked by exact identifiers and lookup keys, never fuzzy matching, because the data is full of near-duplicate decoys. Data-catalog links and chained references also link records. Records with no identifier left (a deleted account, already-anonymized analytics) are linked through the deletion workflow's own activity, assigned across all requests by elimination.
4. **What is their state at `audit_at`?** The latest evidence wins. A copy in any replica still counts as present. A full listing proves absence only for the tables or collections it covers. Evidence captured after `audit_at` is ignored.
5. **What does policy require?** Each record maps to a `deletion_rules.csv` category, which gives the required final state: deleted, redacted, anonymized, retained, restore blocked, or data flow disabled. Tombstones keep only the stable `user_id`.
6. **Verdict and plan.** If any record violates its requirement, the case is `INCOMPLETE`:
   - **Findings** cite the proving `file_id` or `activity_id`.
   - **Remediation** is safely ordered: containment first (disable data flows, block restores), so deleted data cannot be recreated, then delete, redact or anonymize.
   - **Verifications** cover every relevant record's required final state.

   Otherwise the case is `COMPLETE`, with only verification steps, each citing the evidence that proves it.

The core is deterministic Python. The LLM is used only where code cannot do the job reliably: request intent (16 batched calls) and reading 11 screenshots and 4 PDFs. All answers are cached, so runs are reproducible and cheap, about $0.15 in total.

### Project layout

```text
code/
  main.py              pipeline entry point: intent, identity, audit_case, output contract, usage report
  llm.py               provider-agnostic LLM client, cache, fallbacks, token/cost meter
  evidence.py          evidence parsers (SQL, JSON, logs, text, media) -> observations + listings
  reconstruct.py       record linking and state at audit_at
  policy.py            deletion_rules.csv categories and required final states
  planner.py           findings, remediation, verification, verdict
  tests.py             phase test gates (python code/tests.py -v)
  requirements.txt
  cache/               cached model answers (makes the run reproducible without a key)
  evaluation/
    main.py            evaluation workflow (python code/evaluation/main.py)
    evaluation_report.md, usage_report.md, test_findings.md
    cache/             model answers used only by the evaluation
```

### Setup

Requirements: Python 3.10+ (developed and tested on Python 3.14, Windows 11). Run all commands from the **repository root**. The challenge `dataset/` folder must be at the repository root next to `code/`; `code.zip` intentionally does not include it.

```bash
python -m venv .venv                                 # optional; keep .venv outside code/
.venv\Scripts\activate                               # Windows   (macOS/Linux: source .venv/bin/activate)
python -m pip install -r code/requirements.txt       # pypdf, openai
```

On macOS/Linux, use `python3` where `python` is shown.

Optional: create an uncommitted `.env` at the repository root (it is git-ignored and never packaged):

```text
OPENAI_API_KEY=...            # or ANTHROPIC_API_KEY with AUDIT_LLM_PROVIDER=anthropic
AUDIT_LLM_PROVIDER=openai     # openai (default) | anthropic
AUDIT_TEXT_MODEL=...          # optional override, default gpt-4.1-mini-2025-04-14
AUDIT_VISION_MODEL=...        # optional override, default gpt-4.1-2025-04-14
```

**No key is needed to reproduce the submitted `output.csv`.** Every model answer is cached in `code/cache/` (keyed by a sha256 of provider, model, prompt and image), so a clean run makes zero API calls and writes a byte-identical file. A key is only needed after changing a prompt, model or evidence file.

### Run

```bash
python code/main.py               # writes output.csv (518 rows) and code/evaluation/usage_report.md
python code/evaluation/main.py    # writes code/evaluation/evaluation_report.md (samples, custom cases, model comparison)
python code/tests.py -v           # phase test gates (unit, end-to-end, fallback paths)
```

Output on stdout: row counts by status and the number of fallbacks taken (`fallbacks: 0` on the normal path). With the cache, `main.py` takes about 5 seconds and the test suite about 45 seconds. A fresh run with a key and an empty cache makes 31 model calls and takes about 2 minutes.

(`python -m unittest code/...` does not work because the folder name `code` shadows the standard-library module.)

### How it works

| Stage | File | What it does |
|---|---|---|
| Request intent | `code/main.py` | LLM classifies the 398 distinct multilingual request texts (batches of 25) as deletion / non-deletion; a rule-based classifier is the fallback. Non-deletion gives `NOT_APPLICABLE`, `[]`. |
| LLM client | `code/llm.py` | Provider-agnostic (OpenAI / Anthropic) with temperature 0, a disk cache, retries with fallback to the other provider, and a token and cost meter. |
| Identity | `code/main.py` | Current snapshot plus timestamped audit-log history, giving every email, phone, email hash and `user_id` held up to `audit_at`. |
| Evidence | `code/evidence.py` | Content-driven parsers (never file-name specific) for SQL snapshots (run in in-memory SQLite, replicas kept apart), JSON exports and bundles, 7 log layouts (trace-id joins lookup keys to results), pipe-delimited text, and media: screenshots through a vision model and PDFs through extracted text. Output: per-resource observations plus full listings that prove absence. |
| Linking + state | `code/reconstruct.py` | Links records to the subject through identifiers, lookup keys, catalog link events and chained resources. Records with no identifier (deleted accounts, anonymized analytics) are linked through the deletion workflow's own activity, assigned across all requests by elimination. The state at `audit_at` comes from evidence; workflow results are only claims. |
| Policy | `code/policy.py` | Maps each record to a `deletion_rules.csv` category (explicit record type, then collection, record shape, system) and the final state the rule requires. |
| Plan | `code/planner.py` | Findings (with evidence), remediation (containment first: disable flows and block restores before deleting, redacting or anonymizing), and verification of every relevant record. The status is `COMPLETE` only if nothing violates the policy. |
| Output | `code/main.py` | A single validator enforces the output contract before `output.csv` is written; rows follow `case_id` order. |

Only participant files under `dataset/` are read. Sample answers are used only by the evaluation and the tests; pipeline code contains no case, evidence or file names (a test checks this). Evidence captured after a case's `audit_at` is never used, not even to decide which person a record belongs to (AGENTS.md §6.2).

### Fallbacks (degraded modes)

A run always produces a complete, valid `output.csv`. Every fallback taken is listed under *Fallbacks and warnings* in `code/evaluation/usage_report.md` and counted on stdout (`fallbacks: N`).

| Failure | What happens |
|---|---|
| API error or timeout | Retries with backoff (4 attempts), then the other provider if its key is set (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). |
| No key and the answer is not cached | **Request intent:** a rule-based multilingual classifier (it agrees with the LLM on all 398 texts, but the LLM stays primary because rules generalize worse). **PDFs:** read from their extracted text without a model. **Screenshots:** skipped with a recorded reason; other evidence still applies. |
| Classifier answer is malformed or incomplete | The batch is split in halves and retried; anything still unlabeled gets the rule-based classifier. |
| Corrupt cache entry | Treated as a cache miss. |
| Evidence file missing, unreadable or malformed | Only that file is skipped (recorded in the report). Bad bytes are decoded leniently. A broken SQL dump is replayed statement by statement, keeping the valid statements. |
| One case raises an error | That case becomes `NOT_APPLICABLE` with a warning; the other cases are unaffected. |
| A row fails the output contract | Repaired (canonical order, evidence fields fixed). If it cannot be made valid, it is written as `NOT_APPLICABLE`. |
| `COMPLETE` but no record has state evidence | Verifications cite the evidence or activity row that links each record to the subject, flagged as a warning. |
| Request email matches no identity | `NOT_APPLICABLE` with a warning (no record can be tied to the person). |

Worst case, measured with no API key and an empty cache: all 518 rows valid, sample status 32/32, step precision 0.956 and recall 0.907 (normal mode: 0.973 / 0.922). The `Phase9Fallbacks` tests exercise each path.

### Results

See `code/evaluation/evaluation_report.md`:
- **32 public samples:** status accuracy 100%, step F1 0.947 (precision 0.973, recall 0.922), exact sequence 21/32.
- **Custom cases:** 4/4 pass. They are injected into a temporary copy of the dataset and deleted after scoring.
- **Model comparison:** gpt-4.1-mini and gpt-4o-mini give identical labels.
- **Cost:** about $0.15 in total for the run behind `output.csv` (`code/evaluation/usage_report.md`).

The remaining sample differences are data limits: records that appear only in evidence captured after `audit_at`, or nowhere, and same-batch accounts with no distinguishing evidence. Every issue found during development, and how it was fixed, is listed in `code/evaluation/test_findings.md`.

### Packaging

`code.zip` contains `README.md` and `code/` only: no dataset, `.env`, virtual environment or `__pycache__`.

```powershell
# Windows
Remove-Item -Recurse -Force code\__pycache__, code\evaluation\__pycache__ -ErrorAction SilentlyContinue
Compress-Archive -Path README.md, code -DestinationPath code.zip -Force
```

```bash
# macOS / Linux
zip -r code.zip README.md code -x "*/__pycache__/*" "*/.venv/*" "*/venv/*" "*/.DS_Store"
```

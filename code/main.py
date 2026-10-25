"""Ghost Data Still Exists - deletion-audit agent.

Run from the repository root:
    python code/main.py [--requests dataset/deletion_requests.csv] [--out output.csv]
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import planner
from evidence import EvidenceStore, make_media_extractor
from llm import LLM, LLMError, LLMUnavailable
from llm import PRICES as LLM_PRICES
from policy import Policy
from reconstruct import Reconstructor

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_REQUESTS = DATASET_DIR / "deletion_requests.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output.csv"
REQUEST_FILES = ("deletion_requests.csv", "sample_deletion_requests.csv")

# ---------------------------------------------------------------------------
# Output contract (problem_statement.md, "Allowed Operations" and "Rules")
# ---------------------------------------------------------------------------

OUTPUT_HEADER = ["case_id", "deletion_status", "steps_json"]
STATUSES = ("COMPLETE", "INCOMPLETE", "NOT_APPLICABLE")
PHASES = ("FINDING", "REMEDIATION", "VERIFICATION")
STEP_FIELDS = ("step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id")

# (phase, operation, state) in the canonical within-phase order.
OPERATIONS = [
    ("FINDING", "REPORT_REMAINING_RECORD", "PRESENT"),
    ("FINDING", "REPORT_EXPOSED_PII", "PRESENT"),
    ("FINDING", "REPORT_ACTIVE_DATA_FLOW", "ACTIVE"),
    ("FINDING", "REPORT_RESTORE_RISK", "ACTIVE"),
    ("REMEDIATION", "DISABLE_DATA_FLOW", "DISABLED"),
    ("REMEDIATION", "BLOCK_RESTORE", "BLOCKED"),
    ("REMEDIATION", "DELETE_RECORD", "ABSENT"),
    ("REMEDIATION", "REDACT_RECORD", "REDACTED"),
    ("REMEDIATION", "ANONYMIZE_RECORD", "ANONYMIZED"),
    ("VERIFICATION", "VERIFY_RECORD_ABSENT", "ABSENT"),
    ("VERIFICATION", "VERIFY_RECORD_PRESENT", "PRESENT"),
    ("VERIFICATION", "VERIFY_RECORD_REDACTED", "REDACTED"),
    ("VERIFICATION", "VERIFY_RECORD_ANONYMIZED", "ANONYMIZED"),
    ("VERIFICATION", "VERIFY_DATA_FLOW_DISABLED", "DISABLED"),
    ("VERIFICATION", "VERIFY_RESTORE_BLOCKED", "BLOCKED"),
]
OPERATION_PHASE = {op: phase for phase, op, _ in OPERATIONS}
OPERATION_STATE = {op: state for _, op, state in OPERATIONS}
OPERATION_RANK = {op: i for i, (_, op, _) in enumerate(OPERATIONS)}


def step_sort_key(step):
    """Phase, then operation order, then system_id and resource_id ascending."""
    return (OPERATION_RANK[step["operation"]], step["system_id"], step["resource_id"])


def make_step(operation, system_id, resource_id, evidence_id=""):
    return {
        "step_no": 0,
        "phase": OPERATION_PHASE[operation],
        "system_id": system_id,
        "resource_id": resource_id,
        "operation": operation,
        "state": OPERATION_STATE[operation],
        "evidence_id": evidence_id,
    }


def order_steps(steps):
    """Sort steps canonically, drop exact duplicates and renumber from 1."""
    unique = {}
    for step in steps:
        unique.setdefault((step["operation"], step["system_id"], step["resource_id"]), step)
    ordered = sorted(unique.values(), key=step_sort_key)
    return [dict(step, step_no=i) for i, step in enumerate(ordered, start=1)]


def validate_row(case_id, status, steps):
    """Return a list of contract violations for one output row (empty if valid)."""
    errors = []
    if status not in STATUSES:
        return [f"{case_id}: unknown deletion_status {status!r}"]
    if not isinstance(steps, list):
        return [f"{case_id}: steps_json is not a JSON array"]
    if status == "NOT_APPLICABLE":
        if steps:
            errors.append(f"{case_id}: NOT_APPLICABLE must have steps_json=[]")
        return errors
    if not steps:
        errors.append(f"{case_id}: {status} must have at least one step")

    for i, step in enumerate(steps, start=1):
        where = f"{case_id} step {i}"
        if not isinstance(step, dict) or set(step) != set(STEP_FIELDS):
            errors.append(f"{where}: must contain exactly the fields {STEP_FIELDS}")
            continue
        if step["step_no"] != i:
            errors.append(f"{where}: step_no is {step['step_no']!r}, expected {i}")
        op = step["operation"]
        if op not in OPERATION_RANK:
            errors.append(f"{where}: unknown operation {op!r}")
            continue
        if step["phase"] != OPERATION_PHASE[op] or step["state"] != OPERATION_STATE[op]:
            errors.append(f"{where}: {op} requires phase {OPERATION_PHASE[op]} and state {OPERATION_STATE[op]}")
        for field in ("system_id", "resource_id", "evidence_id"):
            if not isinstance(step[field], str):
                errors.append(f"{where}: {field} must be a string")
        if not step["system_id"] or not step["resource_id"]:
            errors.append(f"{where}: system_id and resource_id must be non-empty")

        if status == "COMPLETE":
            if step["phase"] != "VERIFICATION":
                errors.append(f"{where}: COMPLETE rows may only contain VERIFICATION steps")
            if not step["evidence_id"]:
                errors.append(f"{where}: COMPLETE verification must cite evidence")
        elif step["phase"] == "FINDING" and not step["evidence_id"]:
            errors.append(f"{where}: findings must cite evidence")
        elif step["phase"] != "FINDING" and step["evidence_id"]:
            errors.append(f"{where}: future remediation/verification must use an empty evidence_id")

    if not errors:
        keys = [step_sort_key(s) for s in steps]
        if keys != sorted(keys):
            errors.append(f"{case_id}: steps are not in canonical phase/operation/system/resource order")
        if len(set(keys)) != len(keys):
            errors.append(f"{case_id}: duplicate steps")
    if status == "INCOMPLETE" and not any(s.get("phase") == "FINDING" for s in steps if isinstance(s, dict)):
        errors.append(f"{case_id}: INCOMPLETE rows must report at least one finding")
    return errors


def case_sort_key(case_id):
    """Numeric-aware order (a 2-digit id sorts before a 3-digit one), matching deletion_requests.csv."""
    prefix, _, number = case_id.rpartition("_")
    return (prefix, int(number)) if number.isdigit() else (case_id, -1)


def serialize_steps(steps):
    return json.dumps(
        [{field: step[field] for field in STEP_FIELDS} for step in steps],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def write_output(results, expected_case_ids, path=DEFAULT_OUTPUT):
    """Validate every row against the contract, then write output.csv sorted by case_id."""
    errors = []
    missing = set(expected_case_ids) - set(results)
    extra = set(results) - set(expected_case_ids)
    if missing:
        errors.append(f"missing rows for {len(missing)} cases, e.g. {sorted(missing)[:5]}")
    if extra:
        errors.append(f"unexpected rows for {sorted(extra)[:5]}")
    for case_id, (status, steps) in results.items():
        errors.extend(validate_row(case_id, status, steps))
    if errors:
        raise ValueError("output contract violated:\n  " + "\n  ".join(errors[:50]))

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(OUTPUT_HEADER)
        for case_id in sorted(results, key=case_sort_key):
            status, steps = results[case_id]
            writer.writerow([case_id, status, serialize_steps(steps)])


def read_output(path):
    """Read an output.csv-shaped file into {case_id: (status, steps)}."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != OUTPUT_HEADER:
            raise ValueError(f"{path}: header must be {','.join(OUTPUT_HEADER)}")
        return {row["case_id"]: (row["deletion_status"], json.loads(row["steps_json"])) for row in reader}


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Identity resolution (user_identities.csv + user_audit_log.csv)
# ---------------------------------------------------------------------------

def norm_email(value):
    return (value or "").strip().lower()


def norm_phone(value):
    return "".join(ch for ch in (value or "") if ch.isdigit())


def norm_name(value):
    return " ".join((value or "").split()).casefold()


def email_hash(email):
    return hashlib.sha256(norm_email(email).encode("utf-8")).hexdigest()


@dataclass
class Identity:
    """Every identifier a user has held up to a point in time."""

    user_id: str
    names: set = field(default_factory=set)
    emails: set = field(default_factory=set)
    phones: set = field(default_factory=set)
    email_hashes: set = field(default_factory=set)
    audit_ids: list = field(default_factory=list)  # identity changes that produced historical values


class IdentityIndex:
    def __init__(self, identities_rows, audit_rows):
        self.snapshot = {row["user_id"]: row for row in identities_rows}
        self.changes = {}
        for row in sorted(audit_rows, key=lambda r: (r["event_time"], r["audit_id"])):
            self.changes.setdefault(row["user_id"], []).append(row)
        # email -> {user_id: [times the email was known to belong to that user]}
        self.email_owners = {}
        for user_id, row in self.snapshot.items():
            self._add_owner(row["email"], user_id, row["updated_at"])
        for row in audit_rows:
            if row["field_name"] == "email":
                self._add_owner(row["old_value"], row["user_id"], row["event_time"])
                self._add_owner(row["new_value"], row["user_id"], row["event_time"])

    @classmethod
    def load(cls, dataset_dir=DATASET_DIR):
        return cls(read_csv(dataset_dir / "user_identities.csv"), read_csv(dataset_dir / "user_audit_log.csv"))

    def _add_owner(self, email, user_id, when):
        if email:
            self.email_owners.setdefault(norm_email(email), {}).setdefault(user_id, []).append(when)

    def identity_at(self, user_id, at):
        """Identifiers held at any time up to `at`.

        The snapshot is the latest state; changes after `at` are rolled back so their new
        values are not used, while their old values (still valid at `at`) are kept.
        """
        snap = self.snapshot[user_id]
        current = {"name": snap["name"], "email": snap["email"], "phone": snap["phone"], "email_hash": snap["email_hash"]}
        held = {key: {value} for key, value in current.items() if value}
        audit_ids = []
        for change in reversed(self.changes.get(user_id, [])):
            key, old, new = change["field_name"], change["old_value"], change["new_value"]
            if change["event_time"] > at:
                if current.get(key) == new:
                    held[key].discard(new)
                    current[key] = old
                held.setdefault(key, set()).add(old)
            else:
                held.setdefault(key, set()).update(v for v in (old, new) if v)
                audit_ids.append(change["audit_id"])
        identity = Identity(user_id=user_id, audit_ids=sorted(audit_ids, key=case_sort_key))
        identity.names = {norm_name(v) for v in held.get("name", ()) if v}
        identity.emails = {norm_email(v) for v in held.get("email", ()) if v}
        identity.phones = {norm_phone(v) for v in held.get("phone", ()) if v}
        identity.email_hashes = {v.strip().lower() for v in held.get("email_hash", ()) if v}
        identity.email_hashes |= {email_hash(e) for e in identity.emails}
        return identity

    def all_emails(self):
        return set(self.email_owners)

    def resolve(self, email, at):
        """Map a request email (current or historical) to the identity of its owner at `at`."""
        owners = {
            user_id: times
            for user_id, times in self.email_owners.get(norm_email(email), {}).items()
            if min(times) <= at
        }
        if len(owners) != 1:
            return None
        return self.identity_at(next(iter(owners)), at)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Request authorization (LLM, multilingual)
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = """You are a privacy-operations analyst. For each customer message, decide \
whether it authorizes DELETION of the customer's account or personal data (erasure, permanent \
deletion, closing the account with removal of personal data, "right to be forgotten"). Messages may \
be written in any language.

A message about a deletion the person already requested also authorizes it: asking to verify, \
confirm, audit, check, finish or retry their account deletion, or to make sure nothing identifying \
them was left behind, is a deletion request (the deletion must be completed if it is not).

It is NOT a deletion authorization when the person asks only for something else, even if the wording \
mentions the account or data: a copy/export of their data or invoices, access to what is stored, \
correcting or updating details, unsubscribing or opting out of marketing, pausing or temporarily \
deactivating, a question about policy, or when they explicitly ask to keep their profile or data. \
If a message both asks for deletion and for something else, it authorizes deletion."""

CLASSIFY_PROMPT = """Classify each message. Return JSON:
{{"results": [{{"id": <id>, "authorizes_deletion": true|false, "intent": "deletion|export|access|correction|marketing_opt_out|deactivation|question|other", "language": "ISO 639-1 code"}}]}}
Include every id exactly once.

Messages:
{messages}"""

CLASSIFY_BATCH_SIZE = 25


def classify_requests(llm, texts, warnings=None):
    """Map each distinct request text to {authorizes_deletion, intent, language, source}.

    Happy path: fixed batches of 25 (stable cache keys). Fallbacks: a batch whose answer is
    malformed or incomplete is split and retried; texts the model still cannot label (or every
    text, when no provider is reachable) get the rule-based multilingual classifier.
    """
    warnings = warnings if warnings is not None else []
    distinct = sorted(set(texts))
    labels = {}
    for start in range(0, len(distinct), CLASSIFY_BATCH_SIZE):
        labels.update(_classify_batch(llm, distinct[start:start + CLASSIFY_BATCH_SIZE], warnings))
    return labels


def _classify_batch(llm, batch, warnings):
    unavailable = False
    try:
        messages = "\n".join(json.dumps({"id": i, "text": text}, ensure_ascii=False) for i, text in enumerate(batch))
        answer = llm.complete_json(CLASSIFY_SYSTEM_PROMPT, CLASSIFY_PROMPT.format(messages=messages))
        results = answer.get("results", []) if isinstance(answer, dict) else []
        by_id = {r.get("id"): r for r in results if isinstance(r, dict)}
    except LLMError as exc:
        by_id, unavailable = {}, isinstance(exc, LLMUnavailable)
    labels = {}
    for i, text in enumerate(batch):
        result = by_id.get(i)
        if isinstance(result, dict) and isinstance(result.get("authorizes_deletion"), bool):
            labels[text] = dict(result, source="llm")
    missing = [t for t in batch if t not in labels]
    if missing and len(batch) > 1 and not unavailable:
        mid = (len(missing) + 1) // 2
        warnings.append(f"classifier: {len(missing)}/{len(batch)} labels missing or malformed; retrying in halves")
        for part in (missing[:mid], missing[mid:]):
            if part:
                labels.update(_classify_batch(llm, part, warnings))
    elif missing:
        warnings.append(f"classifier: rule-based fallback for {len(missing)} text(s)"
                        + (" (no LLM provider available)" if unavailable else ""))
        for text in missing:
            labels[text] = rule_based_label(text)
    return labels


# Rule-based fallback classifier (used only when the model cannot answer). Lowercased substrings.
KEEP_OR_NEGATION = (
    "not asking for deletion", "not a request to erase", "don't delete", "do not delete", "do not close",
    "don't close", "keep the account", "leave the account", "leave my profile", "keep my account",
    "still use", "still need the account", "still want", "continue using", "without removing",
    "nicht löschen", "bestehen bleiben", "no borre", "no eliminen", "ne supprimez pas", "não apague",
    "削除しないで", "삭제하지",
)
NON_DELETION_INTENT = (
    "copy of", "a copy", "export", "download", "portable format", "invoice", "what personal information",
    "information request", "correct", "misspelled", "update", "change the email", "replace it",
    "out of date", "unsubscribe", "promotional", "marketing", "newsletter", "reset", "can't access",
    "kopie", "copia", "copie", "corrig", "berichtig", "marketing lists only",
)
DELETION_INTENT = (
    "delet", "erase", "erasure", "remov",
    "close my account", "close this account", "close the account", "closed", "forgotten", "gone",
    "supprim", "suppression", "eliminar", "eliminación", "borrar", "löschen", "lösch", "excluir", "exclusão",
    "apagar", "cancella", "削除", "삭제", "удал",
)


def rule_based_label(text):
    lower = text.lower()
    deletion = any(k in lower for k in DELETION_INTENT)
    other = any(k in lower for k in NON_DELETION_INTENT)
    negated = any(k in lower for k in KEEP_OR_NEGATION)
    # Deletion intent wins unless it is negated or the request is scoped to something else "only"
    # (a message that both deletes and asks for something else still authorizes deletion).
    authorizes = deletion and not negated and not (other and " only" in lower)
    return {"authorizes_deletion": bool(authorizes), "intent": "deletion" if authorizes else "other",
            "language": None, "source": "rules"}


class Context:
    """Participant-facing data shared by every case."""

    def __init__(self, dataset_dir=DATASET_DIR, llm=None):
        self.llm = llm or LLM()
        self.warnings = []       # every fallback taken during the run (reported in usage_report.md)
        self.identities = IdentityIndex.load(dataset_dir)
        self.activity = read_csv(dataset_dir / "system_activity.csv")
        self.evidence = EvidenceStore(dataset_dir, self.activity, make_media_extractor(self.llm),
                                      known_emails=self.identities.all_emails())
        self.warnings += [f"evidence {fid} skipped: {reason}" for fid, reason in self.evidence.errors]
        self.policy = Policy(read_csv(dataset_dir / "deletion_rules.csv"))
        # Classify every provided request text in one fixed batching so cache keys are stable
        # whichever request file is being run.
        self.requests = [row for name in REQUEST_FILES if (dataset_dir / name).exists()
                         for row in read_csv(dataset_dir / name)]
        self.request_labels = classify_requests(self.llm, [row["request_text"] for row in self.requests], self.warnings)
        self._reconstructor = None

    @property
    def reconstructor(self):
        """Built once over every deletion request, so workflow runs are assigned across all of them."""
        if self._reconstructor is None:
            cases = []
            for request in self.requests:
                identity = self.identities.resolve(request["email"], request["audit_at"])
                if identity is not None and self.authorizes_deletion(request):
                    cases.append((request, identity))
            self._reconstructor = Reconstructor(self, cases)
        return self._reconstructor

    def authorizes_deletion(self, request):
        if request["request_text"] not in self.request_labels:
            self.request_labels.update(classify_requests(self.llm, [request["request_text"]], self.warnings))
        return self.request_labels[request["request_text"]]["authorizes_deletion"]


def audit_case(request, ctx):
    """Decide one case: intent -> identity -> linked records and their state -> policy -> plan."""
    if not ctx.authorizes_deletion(request):
        return "NOT_APPLICABLE", []
    identity = ctx.identities.resolve(request["email"], request["audit_at"])
    if identity is None:
        # No subject to audit: no record can be tied to this person, so nothing is authorized.
        ctx.warnings.append(f"{request['case_id']}: email matches no known identity at audit_at; NOT_APPLICABLE")
        return "NOT_APPLICABLE", []
    states = ctx.reconstructor.reconstruct(request, identity)
    status, steps, assessments = planner.plan(request, states, ctx.policy, make_step, order_steps)
    if status == "COMPLETE" and not steps:
        status, steps = complete_without_proof(request, assessments, ctx)
    return status, steps


def complete_without_proof(request, assessments, ctx):
    """Fallback for a deletion with no violation and no state evidence for any linked record.

    The verification is the best participant-visible reference available: the evidence or
    activity row that links each record to the subject. Flagged as a warning; without any
    linked record the request cannot be audited and is reported NOT_APPLICABLE.
    """
    steps = [
        make_step(planner.REMEDIATION[a.requirement.final_state][2], a.state.system_id, a.state.resource_id,
                  a.state.link_evidence)
        for a in assessments if a.state.link_evidence and a.state.system_id
    ]
    if steps:
        ctx.warnings.append(f"{request['case_id']}: COMPLETE with no state evidence; verifications cite the linking reference")
        return "COMPLETE", order_steps(steps)
    ctx.warnings.append(f"{request['case_id']}: no record could be linked to the subject; NOT_APPLICABLE")
    return "NOT_APPLICABLE", []


def repair_row(case_id, status, steps):
    """Bring a row that fails the contract back into it (re-order, fix evidence fields)."""
    steps = [s for s in steps if s.get("operation") in OPERATION_RANK and s.get("system_id") and s.get("resource_id")]
    steps = order_steps([make_step(s["operation"], s["system_id"], s["resource_id"], str(s.get("evidence_id") or ""))
                         for s in steps])
    if status == "INCOMPLETE":
        if not any(s["phase"] == "FINDING" and s["evidence_id"] for s in steps):
            status = "COMPLETE"
        else:
            steps = order_steps([dict(s, evidence_id=s["evidence_id"] if s["phase"] == "FINDING" else "") for s in steps
                                 if s["phase"] != "FINDING" or s["evidence_id"]])
    if status == "COMPLETE":
        steps = order_steps([s for s in steps if s["phase"] == "VERIFICATION" and s["evidence_id"]])
    if status != "NOT_APPLICABLE" and not steps:
        return "NOT_APPLICABLE", []
    return status, steps


def run(requests_path=DEFAULT_REQUESTS, output_path=DEFAULT_OUTPUT, ctx=None):
    """Audit every request; one failing case never stops the run or invalidates output.csv."""
    requests = read_csv(requests_path)
    ctx = ctx or Context()
    results = {}
    for req in requests:
        case_id = req["case_id"]
        try:
            status, steps = audit_case(req, ctx)
        except Exception as exc:  # noqa: BLE001 - isolate the case, keep the run going
            ctx.warnings.append(f"{case_id}: audit failed ({type(exc).__name__}: {str(exc)[:160]}); NOT_APPLICABLE")
            status, steps = "NOT_APPLICABLE", []
        if validate_row(case_id, status, steps):
            ctx.warnings.append(f"{case_id}: row failed the output contract and was repaired")
            status, steps = repair_row(case_id, status, steps)
        results[case_id] = (status, steps)
    write_output(results, [req["case_id"] for req in requests], output_path)
    return results


USAGE_REPORT = REPO_ROOT / "code" / "evaluation" / "usage_report.md"


def write_usage_report(ctx, results, path=USAGE_REPORT):
    """Token and cost summary of the run that produced output.csv (generated, not hand-written)."""
    rows = ctx.llm.meter.summary()
    n = len(results)
    counts = {s: sum(1 for status, _ in results.values() if status == s) for s in STATUSES}
    tot = {k: sum(r[k] for r in rows) for k in ("calls", "cached", "input_tokens", "output_tokens",
                                                   "total_calls", "total_input_tokens", "total_output_tokens",
                                                   "cost_usd", "total_cost_usd")}
    lines = [
        "# Usage Report",
        "",
        "Generated by `python code/main.py` at the end of the full-dataset run that wrote `output.csv`.",
        "",
        "## Run",
        "",
        f"- Requests processed: **{n}** ({counts['COMPLETE']} COMPLETE, {counts['INCOMPLETE']} INCOMPLETE, "
        f"{counts['NOT_APPLICABLE']} NOT_APPLICABLE)",
        f"- Provider: **{ctx.llm.provider}**; models: text `{ctx.llm.models['text']}`, vision/documents `{ctx.llm.models['vision']}`",
        "- LLM use is limited to (1) classifying the 398 distinct request texts of `deletion_requests.csv` and "
        "`sample_deletion_requests.csv` in batches of 25, and (2) extracting facts from the 11 screenshots (vision) "
        "and 4 PDFs (extracted text). Everything else (identity, evidence parsing, state reconstruction, policy, "
        "planning) is deterministic code with no model calls.",
        "- Every answer is cached in `code/cache/` (sha256 of provider, model, prompt and image). A rerun makes "
        "zero API calls and reproduces `output.csv` byte for byte.",
        "",
        "## Model calls and tokens",
        "",
        "*Total* is the full model cost behind this output: calls made in this run plus the original calls behind "
        "cached answers it reused. *This run* counts only API calls made by this process.",
        "",
        "| Provider | Model | Calls (total) | Input tokens | Output tokens | Total tokens | Est. cost (USD) | Calls this run | Cost this run (USD) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['provider']} | `{r['model']}` | {r['total_calls']} | {r['total_input_tokens']:,} | {r['total_output_tokens']:,} | "
            f"{r['total_input_tokens'] + r['total_output_tokens']:,} | {r['total_cost_usd']:.4f} | {r['calls']} | {r['cost_usd']:.4f} |"
        )
    total_tokens = tot["total_input_tokens"] + tot["total_output_tokens"]
    lines += [
        f"| **All** | | **{tot['total_calls']}** | **{tot['total_input_tokens']:,}** | **{tot['total_output_tokens']:,}** | "
        f"**{total_tokens:,}** | **{tot['total_cost_usd']:.4f}** | **{tot['calls']}** | **{tot['cost_usd']:.4f}** |",
        "",
        "## Per request",
        "",
        f"- Average tokens per request: **{total_tokens / n:,.1f}** "
        f"({tot['total_input_tokens'] / n:,.1f} input + {tot['total_output_tokens'] / n:,.1f} output)",
        f"- Estimated cost per request: **${tot['total_cost_usd'] / n:.6f}** (total ${tot['total_cost_usd']:.4f} / {n} requests)",
        f"- Model calls per request: {tot['total_calls'] / n:.3f} (shared batches and shared evidence, not one call per request)",
        "",
        "## Pricing assumptions",
        "",
        "Estimated with list prices per 1M tokens (input / output): "
        + "; ".join(f"`{m}` ${p[0]:.2f} / ${p[1]:.2f}" for m, p in sorted(LLM_PRICES.items()) if any(r['model'] == m for r in rows))
        + ". Development runs (prompt iterations) are not included; see `evaluation_report.md`.",
        "",
        "## Fallbacks and warnings",
        "",
    ]
    events = run_warnings(ctx)
    sources = Counter(label.get("source", "llm") for label in ctx.request_labels.values())
    lines.append(f"- Request labels by source: {dict(sorted(sources.items()))} (`llm` = model, `rules` = rule-based fallback)")
    if events:
        lines.append(f"- {len(events)} fallback event(s) in this run:")
        lines += [f"  - {e}" for e in events[:100]]
    else:
        lines.append("- None: every answer came from the primary path (cache or primary provider), every evidence "
                     "file parsed, and every row passed the output contract unmodified.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_warnings(ctx):
    return list(ctx.warnings) + list(ctx.llm.events)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit deletion requests and write output.csv.")
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    ctx = Context()
    results = run(args.requests, args.out, ctx)
    counts = {s: sum(1 for status, _ in results.values() if status == s) for s in STATUSES}
    print(f"wrote {len(results)} rows to {args.out}  {counts}")
    events = run_warnings(ctx)
    print(f"fallbacks: {len(events)}" + ("" if not events else "  (first: " + events[0][:120] + ")"))
    if args.requests.resolve() == DEFAULT_REQUESTS.resolve():
        write_usage_report(ctx, results)
        print(f"wrote {USAGE_REPORT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())

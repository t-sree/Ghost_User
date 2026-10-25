"""Evaluation workflow: score the pipeline and write code/evaluation/evaluation_report.md.

Run from the repository root:
    python code/evaluation/main.py [--skip-model-comparison]

1. Scores the pipeline on the 32 public samples against dataset/sample_output.csv:
   status accuracy, step-level P/R/F1, field-level accuracy, sequence accuracy, per-case diffs.
2. Runs custom evaluation cases. They are defined below as data, injected into a temporary copy
   of the dataset, audited by the unchanged pipeline, compared with hand-derived expected
   outputs, and the temporary files are deleted afterwards (the report keeps them in full).
3. Compares request-classifier models (accuracy vs tokens vs cost).
4. Profiles the 518-case output (no labels exist for it) and embeds the findings log.
The pipeline is imported, not re-implemented.
"""

import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import main as pipeline  # noqa: E402
from llm import LLM  # noqa: E402

REPORT = EVAL_DIR / "evaluation_report.md"
FINDINGS = EVAL_DIR / "test_findings.md"
EVAL_CACHE = EVAL_DIR / "cache"          # LLM answers used only by the evaluation
STEP_FIELDS = ("phase", "system_id", "resource_id", "operation", "state", "evidence_id")
COMPARISON_MODEL = "gpt-4o-mini-2024-07-18"   # gpt-4.1-nano is retired for this account (404)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def step_key(step):
    return tuple(step[f] for f in STEP_FIELDS)


def lcs(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            table[i + 1][j + 1] = table[i][j] + 1 if x == y else max(table[i][j + 1], table[i + 1][j])
    return table[-1][-1]


def align(truth_steps, predicted_steps):
    """Pair each expected step with a predicted step on the same (phase, resource_id), else None."""
    unused = list(predicted_steps)
    pairs = []
    for t in truth_steps:
        match = next((p for p in unused if p["phase"] == t["phase"] and p["resource_id"] == t["resource_id"]), None)
        if match is None:
            match = next((p for p in unused if p["phase"] == t["phase"] and p["system_id"] == t["system_id"]), None)
        if match is not None:
            unused.remove(match)
        pairs.append((t, match))
    return pairs


def score(truth, predicted):
    """truth / predicted: {case_id: (status, steps)}."""
    m = Counter()
    field_hits = Counter()
    per_case = []
    lcs_ratios = []
    for case_id in sorted(truth, key=pipeline.case_sort_key):
        t_status, t_steps = truth[case_id]
        p_status, p_steps = predicted[case_id]
        want, got = Counter(map(step_key, t_steps)), Counter(map(step_key, p_steps))
        hit = sum((want & got).values())
        m["cases"] += 1
        m["status_ok"] += t_status == p_status
        m["exact"] += t_steps == p_steps
        m["want"] += sum(want.values())
        m["got"] += sum(got.values())
        m["hit"] += hit
        t_seq, p_seq = [step_key(s) for s in t_steps], [step_key(s) for s in p_steps]
        ratio = 1.0 if not t_seq and not p_seq else 2 * lcs(t_seq, p_seq) / (len(t_seq) + len(p_seq))
        lcs_ratios.append(ratio)
        for t, p in align(t_steps, p_steps):
            for f in ("step_no",) + STEP_FIELDS:
                field_hits[f] += p is not None and p[f] == t[f]
        per_case.append({
            "case_id": case_id, "truth_status": t_status, "pred_status": p_status,
            "truth_steps": len(t_steps), "pred_steps": len(p_steps), "matched": hit,
            "exact": t_steps == p_steps, "lcs": ratio,
            "missing": sorted(set(want) - set(got)), "extra": sorted(set(got) - set(want)),
        })
    precision = m["hit"] / m["got"] if m["got"] else 1.0
    recall = m["hit"] / m["want"] if m["want"] else 1.0
    return {
        "cases": m["cases"], "status_accuracy": m["status_ok"] / m["cases"],
        "status_ok": m["status_ok"], "exact": m["exact"], "sequence_accuracy": m["exact"] / m["cases"],
        "mean_lcs": sum(lcs_ratios) / len(lcs_ratios),
        "step_precision": precision, "step_recall": recall,
        "step_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "want": m["want"], "got": m["got"], "hit": m["hit"],
        "field_accuracy": {f: field_hits[f] / m["want"] for f in ("step_no",) + STEP_FIELDS},
        "per_case": per_case,
    }


# ---------------------------------------------------------------------------
# Custom evaluation cases (fictional people; injected into a temporary dataset copy)
# ---------------------------------------------------------------------------

def _verify(op, system, resource, evidence=""):
    return {"operation": op, "system_id": system, "resource_id": resource, "evidence_id": evidence}


CUSTOM_CASES = [
    {
        "case_id": "case_c01",
        "title": "Non-deletion request written in German (data copy, keep the account)",
        "tests": "multilingual intent classification; NOT_APPLICABLE must have `[]` even though the subject has deletable data",
        "identity": {"user_id": "user_9001", "name": "Greta Lindqvist", "email": "greta.lindqvist@example.net",
                     "phone": "+46 70 555 0101", "updated_at": "2026-05-01T09:00:00Z"},
        "audit_log": [],
        "request": {"email": "greta.lindqvist@example.net", "requested_at": "2026-11-01T09:00:00Z",
                    "reported_status": "NOT_STARTED", "reported_at": "2026-11-02T11:00:00Z", "audit_at": "2026-11-06T12:00:00Z",
                    "request_text": "Bitte senden Sie mir eine Kopie aller Daten, die Sie über mich gespeichert haben. "
                                    "Mein Konto soll bestehen bleiben, bitte nichts löschen."},
        "activity": [],
        "evidence": [],
        "expected": ("NOT_APPLICABLE", []),
    },
    {
        "case_id": "case_c02",
        "title": "Request made with the current email; a search copy and an active sync still use the old email",
        "tests": "historical-identifier resolution (audit log), workflow claims SUCCESS but evidence shows the copy, "
                 "active data flow, containment-before-deletion ordering",
        "identity": {"user_id": "user_9002", "name": "Tomas Okafor", "email": "tomas.okafor@example.net",
                     "phone": "+234 803 555 0202", "updated_at": "2026-06-10T08:00:00Z"},
        "audit_log": [{"audit_id": "audit_9002", "user_id": "user_9002", "event_time": "2026-06-10T08:00:00Z",
                       "field_name": "email", "old_value": "t.okafor.old@example.net", "new_value": "tomas.okafor@example.net"}],
        "request": {"email": "tomas.okafor@example.net", "requested_at": "2026-11-01T09:00:00Z",
                    "reported_status": "COMPLETE", "reported_at": "2026-11-02T11:00:00Z", "audit_at": "2026-11-06T12:00:00Z",
                    "request_text": "Please delete my account and everything linked to my old and new email addresses."},
        "activity": [
            ("activity_90021", "2026-11-02T10:30:00Z", "DELETE_RECORD", "deletion_worker", "account_service", "account_90002", "SUCCESS",
             "Deletion worker submitted primary profile account_90002 to account_service."),
            ("activity_90022", "2026-11-02T11:20:00Z", "DELETE_RECORD", "deletion_worker", "search_service", "search_90002", "SUCCESS",
             "Deletion worker received the final result for search_90002 from search_service."),
            ("activity_90023", "2026-11-02T11:40:00Z", "CHECK_DATA_FLOW", "deletion_worker", "integration_service", "data_flow_90002", "SUCCESS",
             "Connector status check confirmed that data flow data_flow_90002 remains enabled."),
        ],
        "evidence": [{
            "file_id": "evidence_9002", "path": "evidence/records/custom_service_export_20261105.json",
            "origin": "service_resource_export", "captured_at": "2026-11-05T10:00:00Z",
            "content": {"exported_at": "2026-11-05T10:00:00Z", "region": "global", "service_exports": {
                "account_service": {"profile_lookups": [
                    {"observed_at": "2026-11-05T09:00:00Z", "profile": None, "resource_id": "account_90002", "status_code": 404}]},
                "search_service": {"index_queries": [
                    {"observed_at": "2026-11-05T09:10:00Z", "resource_id": "search_90002", "query": "t.okafor.old@example.net",
                     "hits": {"total": 1, "documents": [{"resource_id": "search_90002", "display_name": "Tomas Okafor",
                                                         "email": "t.okafor.old@example.net", "phone": None,
                                                         "email_hash": None, "customer_reference": None}]}}]},
                "integration_service": {"connectors": [
                    {"observed_at": "2026-11-05T09:20:00Z", "resource_id": "data_flow_90002", "connector_name": "search_profile_indexer",
                     "customer_email": "t.okafor.old@example.net", "customer_reference": None, "enabled": True,
                     "last_delivery_status": "delivered"}]},
            }},
        }],
        "expected": ("INCOMPLETE", [
            {"phase": "FINDING", "operation": "REPORT_REMAINING_RECORD", "system_id": "search_service", "resource_id": "search_90002", "evidence_id": "evidence_9002"},
            {"phase": "FINDING", "operation": "REPORT_ACTIVE_DATA_FLOW", "system_id": "integration_service", "resource_id": "data_flow_90002", "evidence_id": "evidence_9002"},
            {"phase": "REMEDIATION", "operation": "DISABLE_DATA_FLOW", "system_id": "integration_service", "resource_id": "data_flow_90002", "evidence_id": ""},
            {"phase": "REMEDIATION", "operation": "DELETE_RECORD", "system_id": "search_service", "resource_id": "search_90002", "evidence_id": ""},
            {"phase": "VERIFICATION", **_verify("VERIFY_RECORD_ABSENT", "account_service", "account_90002")},
            {"phase": "VERIFICATION", **_verify("VERIFY_RECORD_ABSENT", "search_service", "search_90002")},
            {"phase": "VERIFICATION", **_verify("VERIFY_DATA_FLOW_DISABLED", "integration_service", "data_flow_90002")},
        ]),
    },
    {
        "case_id": "case_c03",
        "title": "Tombstone + active legal hold, with a later capture that must be ignored",
        "tests": "ACCOUNT_TOMBSTONE keeps only user_id (REDACTED), LEGAL_HOLD_RECORD retained (PRESENT), subject-scoped listing "
                 "proves the account absent, evidence captured after audit_at never changes the answer",
        "identity": {"user_id": "user_9003", "name": "Aiko Brennan", "email": "aiko.brennan@example.net",
                     "phone": "+81 90 5555 0303", "updated_at": "2026-04-01T08:00:00Z"},
        "audit_log": [],
        "request": {"email": "aiko.brennan@example.net", "requested_at": "2026-11-01T09:00:00Z",
                    "reported_status": "COMPLETE", "reported_at": "2026-11-02T11:00:00Z", "audit_at": "2026-11-06T12:00:00Z",
                    "request_text": "Please close and delete my account. Keep only what you are legally required to keep."},
        "activity": [
            ("activity_90031", "2026-11-02T10:35:00Z", "DELETE_RECORD", "deletion_worker", "account_service", "account_90003", "SUCCESS",
             "Deletion worker submitted primary profile account_90003 to account_service."),
            ("activity_90032", "2026-11-02T11:30:00Z", "REDACT_RECORD", "deletion_worker", "identity_service", "tombstone_90003", "SUCCESS",
             "Policy action completed for tombstone_90003."),
        ],
        "evidence": [
            {"file_id": "evidence_9003", "path": "evidence/records/custom_snapshot_bundle_20261105.json",
             "origin": "records_assurance_snapshot_bundle", "captured_at": "2026-11-05T12:00:00Z",
             "content": {"exported_at": "2026-11-05T12:00:00Z", "source": "records_assurance_service", "snapshots": [
                 {"captured_at": "2026-11-05T12:00:00Z", "data_store": "account_service", "collection": "customer_profiles", "records": []},
                 {"captured_at": "2026-11-05T12:00:00Z", "data_store": "identity_service", "collection": "deleted_customer_references",
                  "records": [{"resource_id": "tombstone_90003", "subject_reference": "user_9003", "name": None, "email": None,
                               "phone": None, "email_hash": None, "account_status": "DELETED"}]},
                 {"captured_at": "2026-11-05T12:00:00Z", "data_store": "legal_service", "collection": "retained_order_records",
                  "records": [{"resource_id": "legal_hold_90003", "subject_reference": "user_9003", "order_reference": "ORD-20269003",
                               "retention_reason": "payment_dispute_and_tax_record", "retained_until": "2033-12-31T23:59:59Z"}]},
             ]}},
            {"file_id": "evidence_9004", "path": "evidence/records/custom_snapshot_bundle_20261120.json",
             "origin": "records_assurance_snapshot_bundle", "captured_at": "2026-11-20T12:00:00Z",
             "content": {"exported_at": "2026-11-20T12:00:00Z", "source": "records_assurance_service", "snapshots": [
                 {"captured_at": "2026-11-20T12:00:00Z", "data_store": "identity_service", "collection": "deleted_customer_references",
                  "records": [{"resource_id": "tombstone_90003", "subject_reference": "user_9003", "name": "Aiko Brennan",
                               "email": "aiko.brennan@example.net", "phone": None, "email_hash": None, "account_status": "DELETED"}]},
             ]}},
        ],
        "expected": ("COMPLETE", [
            {"phase": "VERIFICATION", **_verify("VERIFY_RECORD_ABSENT", "account_service", "account_90003", "evidence_9003")},
            {"phase": "VERIFICATION", **_verify("VERIFY_RECORD_PRESENT", "legal_service", "legal_hold_90003", "evidence_9003")},
            {"phase": "VERIFICATION", **_verify("VERIFY_RECORD_REDACTED", "identity_service", "tombstone_90003", "evidence_9003")},
        ]),
    },
    {
        "case_id": "case_c04",
        "title": "Primary profile deleted but a failover replica still holds it; a backup can still restore",
        "tests": "replica drift in a SQL snapshot (primary listing must not disprove the failover copy), restore risk cited "
                 "from the workflow's own restore-access check, BLOCK_RESTORE before DELETE",
        "identity": {"user_id": "user_9004", "name": "Rafael Mendes", "email": "rafael.mendes@example.net",
                     "phone": "+55 11 95555 0404", "updated_at": "2026-03-01T08:00:00Z"},
        "audit_log": [],
        "request": {"email": "rafael.mendes@example.net", "requested_at": "2026-11-01T09:00:00Z",
                    "reported_status": "COMPLETE", "reported_at": "2026-11-02T11:00:00Z", "audit_at": "2026-11-06T12:00:00Z",
                    "request_text": "Supprimez définitivement mon compte et mes données personnelles, y compris dans les sauvegardes."},
        "activity": [
            ("activity_90041", "2026-11-02T10:40:00Z", "DELETE_RECORD", "deletion_worker", "account_service", "account_90004", "SUCCESS",
             "Deletion worker submitted primary profile account_90004 to account_service."),
            ("activity_90042", "2026-11-02T10:41:00Z", "SYNC_RECORD", "account_service", "account_failover_replica", "account_90004", "SUCCESS",
             "Replication worker reported its final checkpoint. Target resource: account_90004."),
            ("activity_90043", "2026-11-02T11:50:00Z", "CHECK_RESTORE_ACCESS", "deletion_worker", "backup_service", "backup_90004", "SUCCESS",
             "Restore-access check confirmed that archive backup_90004 can still be restored."),
        ],
        "evidence": [{
            "file_id": "evidence_9005", "path": "evidence/databases/custom_accounts_failover_replica_20261105.sql",
            "origin": "customer_account_failover_replica_capture", "captured_at": "2026-11-05T23:00:00Z",
            "content": (
                "PRAGMA foreign_keys=ON;\nBEGIN TRANSACTION;\n"
                "CREATE TABLE customer_profiles_failover_replica (resource_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, full_name TEXT NOT NULL, "
                "email TEXT NOT NULL, phone TEXT NOT NULL, account_status TEXT NOT NULL, updated_at TEXT NOT NULL);\n"
                "INSERT INTO \"customer_profiles_failover_replica\" VALUES('account_90004','user_9004','Rafael Mendes',"
                "'rafael.mendes@example.net','+55 11 95555 0404','ACTIVE','2026-11-05T22:45:00Z');\n"
                "CREATE TABLE customer_profiles_primary (resource_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, full_name TEXT NOT NULL, "
                "email TEXT NOT NULL, phone TEXT NOT NULL, account_status TEXT NOT NULL, updated_at TEXT NOT NULL);\n"
                "COMMIT;\n"
            ),
        }],
        "expected": ("INCOMPLETE", [
            {"phase": "FINDING", "operation": "REPORT_REMAINING_RECORD", "system_id": "account_service", "resource_id": "account_90004", "evidence_id": "evidence_9005"},
            {"phase": "FINDING", "operation": "REPORT_RESTORE_RISK", "system_id": "backup_service", "resource_id": "backup_90004", "evidence_id": "activity_90043"},
            {"phase": "REMEDIATION", "operation": "BLOCK_RESTORE", "system_id": "backup_service", "resource_id": "backup_90004", "evidence_id": ""},
            {"phase": "REMEDIATION", "operation": "DELETE_RECORD", "system_id": "account_service", "resource_id": "account_90004", "evidence_id": ""},
            {"phase": "VERIFICATION", **_verify("VERIFY_RECORD_ABSENT", "account_service", "account_90004")},
            {"phase": "VERIFICATION", **_verify("VERIFY_RESTORE_BLOCKED", "backup_service", "backup_90004")},
        ]),
    },
]


def expected_steps(case):
    status, steps = case["expected"]
    full = [pipeline.make_step(s["operation"], s["system_id"], s["resource_id"], s["evidence_id"]) for s in steps]
    return status, [dict(s, step_no=i) for i, s in enumerate(full, start=1)]


def _append_csv(path, header, rows):
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in rows:
            writer.writerow([row[h] for h in header] if isinstance(row, dict) else row)


def build_custom_dataset(root):
    """Copy the participant dataset to `root` and inject the custom cases (fictional data only)."""
    dataset = root / "dataset"
    shutil.copytree(pipeline.DATASET_DIR, dataset)
    identities = []
    for case in CUSTOM_CASES:
        ident = dict(case["identity"])
        ident["email_hash"] = pipeline.email_hash(ident["email"])
        identities.append(ident)
    _append_csv(dataset / "user_identities.csv", ["user_id", "name", "email", "phone", "email_hash", "updated_at"], identities)
    _append_csv(dataset / "user_audit_log.csv", ["audit_id", "user_id", "event_time", "field_name", "old_value", "new_value"],
                [a for c in CUSTOM_CASES for a in c["audit_log"]])
    _append_csv(dataset / "system_activity.csv", None, [a for c in CUSTOM_CASES for a in c["activity"]])
    manifest = []
    for case in CUSTOM_CASES:
        for ev in case["evidence"]:
            target = dataset / ev["path"]
            content = ev["content"]
            target.write_text(content if isinstance(content, str) else json.dumps(content, indent=2), encoding="utf-8")
            manifest.append([ev["file_id"], ev["path"], ev["origin"], ev["captured_at"]])
    _append_csv(dataset / "evidence_files.csv", None, manifest)
    requests = dataset / "custom_requests.csv"
    header = ["case_id", "email", "requested_at", "reported_status", "reported_at", "audit_at", "request_text"]
    with open(requests, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for case in CUSTOM_CASES:
            writer.writerow([case["case_id"]] + [case["request"][h] for h in header[1:]])
    return dataset, requests


def run_custom_cases():
    """Audit the custom cases in a temporary dataset copy; the copy is deleted afterwards."""
    eval_llm = LLM(cache_dir=EVAL_CACHE)
    with tempfile.TemporaryDirectory(prefix="ghost_custom_cases_") as tmp:
        dataset, requests_path = build_custom_dataset(Path(tmp))
        ctx = pipeline.Context(dataset_dir=dataset)
        requests = pipeline.read_csv(requests_path)
        ctx.request_labels.update(pipeline.classify_requests(eval_llm, [r["request_text"] for r in requests]))
        # The custom requests take part in run assignment like any other request.
        ctx.requests.extend(requests)
        predicted = {r["case_id"]: pipeline.audit_case(r, ctx) for r in requests}
        for case_id, (status, steps) in predicted.items():
            errors = pipeline.validate_row(case_id, status, steps)
            if errors:
                raise AssertionError(errors)
    # TemporaryDirectory removed every injected file at this point.
    truth = {c["case_id"]: expected_steps(c) for c in CUSTOM_CASES}
    return truth, predicted, eval_llm.meter.summary()


# ---------------------------------------------------------------------------
# Model comparison (request classification)
# ---------------------------------------------------------------------------

def compare_classifiers(ctx):
    texts = sorted({r["request_text"] for r in ctx.requests})
    sample_texts = {r["request_text"] for r in pipeline.read_csv(pipeline.DATASET_DIR / "sample_deletion_requests.csv")}
    primary = {t: ctx.request_labels[t]["authorizes_deletion"] for t in texts}
    alt_llm = LLM(cache_dir=EVAL_CACHE)
    alt_llm.models["text"] = COMPARISON_MODEL
    alt = {t: v["authorizes_deletion"] for t, v in pipeline.classify_requests(alt_llm, texts).items()}
    eval_requests = pipeline.read_csv(pipeline.DEFAULT_REQUESTS)
    intake_non_deletion = sum(a["event_type"] == "CLASSIFY_REQUEST" and "non-deletion" in a["details"] for a in ctx.activity)
    rows = []
    for name, labels, meter in (("primary", primary, ctx.llm.meter.summary()), ("comparison", alt, alt_llm.meter.summary())):
        model = ctx.llm.models["text"] if name == "primary" else COMPARISON_MODEL
        usage = next((r for r in meter if r["model"] == model), None)
        rows.append({
            "model": model,
            "sample_accuracy": sum(labels[t] for t in sample_texts) / len(sample_texts),   # every sample authorizes deletion
            "eval_non_deletion": sum(not labels[r["request_text"]] for r in eval_requests),
            "agreement": sum(labels[t] == primary[t] for t in texts) / len(texts),
            "tokens": (usage["total_input_tokens"] + usage["total_output_tokens"]) if usage else 0,
            "cost": usage["total_cost_usd"] if usage else 0.0,
            "calls": usage["total_calls"] if usage else 0,
        })
    disagreements = [t for t in texts if alt[t] != primary[t]]
    return rows, intake_non_deletion, disagreements


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt_step(key):
    phase, system, resource, op, state, evidence = key
    return f"`{op}` {system}/{resource}" + (f" ({evidence})" if evidence else "")


def write_report(sample, custom, custom_usage, comparison, profile, findings_md):
    s = sample
    lines = [
        "# Evaluation Report",
        "",
        "Generated by `python code/evaluation/main.py`. The pipeline under test is `python code/main.py`; this script "
        "imports it unchanged.",
        "",
        "## Summary",
        "",
        "| Metric | Public samples (32 cases) | Custom cases |",
        "|---|---:|---:|",
        f"| Case-level `deletion_status` accuracy | **{s['status_accuracy']:.1%}** ({s['status_ok']}/{s['cases']}) | "
        f"{custom['status_accuracy']:.1%} ({custom['status_ok']}/{custom['cases']}) |",
        f"| Step-level precision | {s['step_precision']:.3f} ({s['hit']}/{s['got']}) | {custom['step_precision']:.3f} |",
        f"| Step-level recall | {s['step_recall']:.3f} ({s['hit']}/{s['want']}) | {custom['step_recall']:.3f} |",
        f"| Step-level F1 | **{s['step_f1']:.3f}** | {custom['step_f1']:.3f} |",
        f"| Sequence accuracy (exact ordered `steps_json`) | **{s['sequence_accuracy']:.1%}** ({s['exact']}/{s['cases']}) | "
        f"{custom['sequence_accuracy']:.1%} ({custom['exact']}/{custom['cases']}) |",
        f"| Mean sequence similarity (LCS) | {s['mean_lcs']:.3f} | {custom['mean_lcs']:.3f} |",
        "",
        "A step matches only when all of `phase`, `system_id`, `resource_id`, `operation`, `state` and `evidence_id` "
        "match; steps are compared as multisets per case. Sequence accuracy requires the whole ordered array to be identical.",
        "",
        "## Field-level accuracy (public samples)",
        "",
        "Each expected step is aligned with the predicted step of the same phase and resource (else same phase and system); "
        "an expected step with no aligned prediction counts as wrong in every field.",
        "",
        "| Field | Accuracy |",
        "|---|---:|",
    ]
    for field, value in s["field_accuracy"].items():
        lines.append(f"| `{field}` | {value:.3f} |")
    lines += ["", "## Per-case results (public samples)", "",
              "| Case | Expected | Predicted | Steps exp/pred/matched | Exact | Differences |", "|---|---|---|---|---|---|"]
    for c in s["per_case"]:
        diff = "; ".join([f"missing {fmt_step(k)}" for k in c["missing"]] + [f"extra {fmt_step(k)}" for k in c["extra"]]) or "-"
        lines.append(f"| {c['case_id']} | {c['truth_status']} | {c['pred_status']} | {c['truth_steps']}/{c['pred_steps']}/{c['matched']} | "
                     f"{'yes' if c['exact'] else 'no'} | {diff} |")
    lines += [
        "",
        "### Why the remaining sample differences remain",
        "",
        "- **Underivable from participant data at `audit_at` (8 resources):** one account id and two audit records appear "
        "nowhere in participant files; five audit records appear only in evidence captured after the case's `audit_at`. "
        "The pipeline never uses later evidence, not even for linking (see F-24).",
        "- **Irreducible ties:** some same-batch requests' accounts have no evidence that tells them apart; pairing is "
        "deterministic but can swap them (F-25).",
        "",
        "## Custom evaluation cases",
        "",
        "Four cases created for this evaluation (fictional people, ids in the 9000 range). `code/evaluation/main.py` "
        "copies the dataset into a temporary directory, injects each case's identity, audit-log rows, activity rows "
        "and evidence files, runs the unchanged pipeline on them, compares with the expected output below, and deletes "
        "the temporary copy. Nothing is written into `dataset/`.",
        "",
    ]
    for case in CUSTOM_CASES:
        row = next(c for c in custom["per_case"] if c["case_id"] == case["case_id"])
        exp_status, exp_steps = expected_steps(case)
        lines += [
            f"### {case['case_id']}: {case['title']}",
            "",
            f"**Tests:** {case['tests']}.",
            "",
            f"**Request** ({case['request']['email']}, requested {case['request']['requested_at']}, reported "
            f"`{case['request']['reported_status']}` at {case['request']['reported_at']}, audit_at {case['request']['audit_at']}):",
            f"> {case['request']['request_text']}",
            "",
        ]
        if case["audit_log"]:
            lines.append("**Identity history:** " + "; ".join(f"{a['audit_id']}: {a['field_name']} {a['old_value']} -> {a['new_value']} at {a['event_time']}" for a in case["audit_log"]))
            lines.append("")
        if case["activity"]:
            lines.append("**Injected activity** (workflow claims):")
            lines.append("")
            for a in case["activity"]:
                lines.append(f"- `{a[0]}` {a[1]} {a[2]} {a[5]} -> {a[6]}: {a[7]}")
            lines.append("")
        if case["evidence"]:
            lines.append("**Injected evidence:**")
            lines.append("")
            for ev in case["evidence"]:
                kind = "SQL snapshot" if ev["path"].endswith(".sql") else "JSON export"
                lines.append(f"- `{ev['file_id']}` ({kind}, captured {ev['captured_at']}): `{ev['path']}`")
            lines.append("")
        lines.append(f"**Expected:** `{exp_status}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(exp_steps, indent=1))
        lines.append("```")
        lines.append("")
        lines.append(f"**Result:** predicted `{row['pred_status']}`, {row['matched']}/{row['truth_steps']} steps matched, "
                     f"exact sequence: **{'PASS' if row['exact'] and row['pred_status'] == row['truth_status'] else 'FAIL'}**"
                     + ("" if row["exact"] else " (missing: " + ", ".join(map(fmt_step, row["missing"])) + "; extra: " + ", ".join(map(fmt_step, row["extra"])) + ")"))
        lines.append("")
    usage_tokens = sum(r["total_input_tokens"] + r["total_output_tokens"] for r in custom_usage)
    lines += [f"Custom-case classification used {usage_tokens:,} tokens (cached in `code/evaluation/cache/`).", ""]

    rows, intake, disagreements = comparison
    lines += [
        "## Model comparison: request classifier",
        "",
        "The same prompt and batching run with two OpenAI models over all 398 distinct request texts. Every public sample "
        f"authorizes deletion; the intake service's activity log records {intake} non-deletion classifications for the "
        "evaluation set, a label-free sanity check.",
        "",
        "| Model | Sample accuracy | Evaluation NOT_APPLICABLE | Agreement with primary | Calls | Tokens | Est. cost (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| `{r['model']}` | {r['sample_accuracy']:.1%} | {r['eval_non_deletion']} | {r['agreement']:.1%} | "
                     f"{r['calls']} | {r['tokens']:,} | {r['cost']:.4f} |")
    lines += [""]
    if disagreements:
        lines.append(f"Texts where the models disagree ({len(disagreements)}):")
        lines.append("")
        for t in disagreements[:10]:
            lines.append(f"- {t[:160]}")
        lines.append("")
    if rows:
        primary_row, alt_row = rows
        if alt_row["agreement"] == 1.0 and alt_row["sample_accuracy"] >= primary_row["sample_accuracy"]:
            verdict = (f"Both models produce identical labels on every text; `{alt_row['model']}` would cut classification "
                       f"cost from ${primary_row['cost']:.4f} to ${alt_row['cost']:.4f}. The task is easy for both, so the "
                       f"choice is cost, not accuracy. `{primary_row['model']}` remains configured because the submitted "
                       f"`output.csv` and its cache were produced with it; switching is one environment variable "
                       f"(`AUDIT_TEXT_MODEL`) and changes no prediction.")
        else:
            verdict = (f"`{primary_row['model']}` stays primary: the comparison model disagrees on {len(disagreements)} texts "
                       f"and/or scores lower on the samples.")
        lines += [f"Decision: {verdict} Screenshot and PDF extraction uses the `gpt-4.1` tier because one misread "
                  "character breaks exact identifier matching (F-17, F-28).", ""]
    lines += [
        "## Evaluation-set profile (518 cases, no labels)",
        "",
        f"- Status distribution: {profile['statuses']}",
        f"- Steps per case: mean {profile['mean_steps']:.2f}, max {profile['max_steps']}",
        f"- Operations: {profile['operations']}",
        f"- Evidence ids cited: {profile['evidence_cited']} distinct",
        "",
        "## Failures found and improvements made",
        "",
        "Every issue found by the phase test gates (`python code/tests.py -v`) or by data profiling, in the order "
        "found. Status: FIXED (fixed and covered by a test), NOTED (data fact the design relies on), OPEN (known limitation).",
        "",
        findings_md,
        "",
        "## Reproduce",
        "",
        "```",
        "python code/main.py              # writes output.csv and code/evaluation/usage_report.md",
        "python code/evaluation/main.py   # writes this report",
        "python code/tests.py -v          # phase test gates",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def profile_output(path=pipeline.DEFAULT_OUTPUT):
    output = pipeline.read_output(path)
    statuses = Counter(status for status, _ in output.values())
    ops = Counter(step["operation"] for _, steps in output.values() for step in steps)
    lengths = [len(steps) for _, steps in output.values()]
    evidence = {step["evidence_id"] for _, steps in output.values() for step in steps if step["evidence_id"]}
    return {"statuses": dict(sorted(statuses.items())), "operations": dict(ops.most_common()),
            "mean_steps": sum(lengths) / len(lengths), "max_steps": max(lengths), "evidence_cited": len(evidence)}


def findings_table():
    text = FINDINGS.read_text(encoding="utf-8")
    start = text.find("| # |")
    return text[start:].strip() if start != -1 else text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the deletion-audit pipeline.")
    parser.add_argument("--skip-model-comparison", action="store_true")
    args = parser.parse_args(argv)

    ctx = pipeline.Context()
    sample_requests = pipeline.read_csv(pipeline.DATASET_DIR / "sample_deletion_requests.csv")
    predicted = {r["case_id"]: pipeline.audit_case(r, ctx) for r in sample_requests}
    sample = score(pipeline.read_output(pipeline.DATASET_DIR / "sample_output.csv"), predicted)

    custom_truth, custom_pred, custom_usage = run_custom_cases()
    custom = score(custom_truth, custom_pred)

    comparison = compare_classifiers(ctx) if not args.skip_model_comparison else ([], 0, [])
    write_report(sample, custom, custom_usage, comparison, profile_output(), findings_table())

    print(f"samples: status {sample['status_ok']}/{sample['cases']}  exact {sample['exact']}/{sample['cases']}  "
          f"step P {sample['step_precision']:.3f} R {sample['step_recall']:.3f} F1 {sample['step_f1']:.3f}")
    print(f"custom:  status {custom['status_ok']}/{custom['cases']}  exact {custom['exact']}/{custom['cases']}")
    for c in custom["per_case"]:
        if not c["exact"]:
            print("  ", c["case_id"], "missing", c["missing"], "extra", c["extra"])
    print(f"wrote {REPORT.relative_to(pipeline.REPO_ROOT)}")
    return 0 if custom["exact"] == custom["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())

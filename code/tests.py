"""Phase test gates. Run from the repository root:
    python code/tests.py -v
(`python -m unittest code/...` does not work: `code` clashes with the stdlib module.)
"""

import csv
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence  # noqa: E402
import llm  # noqa: E402
import main  # noqa: E402
import planner  # noqa: E402
import policy  # noqa: E402
import reconstruct  # noqa: E402

SAMPLE_REQUESTS = main.DATASET_DIR / "sample_deletion_requests.csv"
SAMPLE_OUTPUT = main.DATASET_DIR / "sample_output.csv"


class Phase0OutputContract(unittest.TestCase):
    def setUp(self):
        self.samples = main.read_output(SAMPLE_OUTPUT)

    def test_validator_accepts_all_sample_rows(self):
        for case_id, (status, steps) in self.samples.items():
            self.assertEqual(main.validate_row(case_id, status, steps), [], case_id)

    def test_order_steps_reproduces_sample_order(self):
        for case_id, (_, steps) in self.samples.items():
            shuffled = list(reversed(steps))
            self.assertEqual(main.order_steps(shuffled), steps, case_id)

    def _incomplete(self):
        return [
            main.make_step("REPORT_REMAINING_RECORD", "account_service", "account_1", "evidence_01"),
            main.make_step("DELETE_RECORD", "account_service", "account_1"),
            main.make_step("VERIFY_RECORD_ABSENT", "account_service", "account_1"),
        ]

    def test_rejects_wrong_order(self):
        steps = self._incomplete()
        steps = [dict(s, step_no=i) for i, s in enumerate(reversed(steps), start=1)]
        self.assertTrue(main.validate_row("c", "INCOMPLETE", steps))

    def test_rejects_step_no_gap(self):
        steps = main.order_steps(self._incomplete())
        steps[2]["step_no"] = 5
        self.assertTrue(main.validate_row("c", "INCOMPLETE", steps))

    def test_rejects_unknown_operation(self):
        steps = main.order_steps(self._incomplete())
        steps[1]["operation"] = "PURGE_EVERYTHING"
        self.assertTrue(main.validate_row("c", "INCOMPLETE", steps))

    def test_rejects_evidence_on_future_step(self):
        steps = main.order_steps(self._incomplete())
        steps[1]["evidence_id"] = "evidence_01"
        self.assertTrue(main.validate_row("c", "INCOMPLETE", steps))

    def test_rejects_complete_without_evidence_or_with_findings(self):
        verify = main.order_steps([main.make_step("VERIFY_RECORD_ABSENT", "account_service", "account_1")])
        self.assertTrue(main.validate_row("c", "COMPLETE", verify))
        self.assertTrue(main.validate_row("c", "COMPLETE", main.order_steps(self._incomplete())))

    def test_rejects_not_applicable_with_steps(self):
        self.assertTrue(main.validate_row("c", "NOT_APPLICABLE", main.order_steps(self._incomplete())))

    def test_round_trip_sample_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            main.write_output(self.samples, list(self.samples), path)
            self.assertEqual(main.read_output(path), self.samples)

    def test_full_run_writes_518_sorted_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.csv"
            main.run(main.DEFAULT_REQUESTS, path)
            with open(path, encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], main.OUTPUT_HEADER)
        case_ids = [r[0] for r in rows[1:]]
        request_ids = [r["case_id"] for r in main.read_csv(main.DEFAULT_REQUESTS)]
        self.assertEqual(len(case_ids), 518)
        self.assertEqual(case_ids, sorted(case_ids, key=main.case_sort_key))
        self.assertEqual(case_ids, request_ids)  # same order as the input file


class Phase1IdentityResolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = main.IdentityIndex.load()
        cls.requests = main.read_csv(main.DEFAULT_REQUESTS) + main.read_csv(SAMPLE_REQUESTS)

    def test_every_request_resolves_to_exactly_one_user(self):
        unresolved = [r["case_id"] for r in self.requests if self.index.resolve(r["email"], r["audit_at"]) is None]
        self.assertEqual(unresolved, [])

    def test_request_email_is_among_resolved_identifiers(self):
        for r in self.requests:
            identity = self.index.resolve(r["email"], r["audit_at"])
            self.assertIn(main.norm_email(r["email"]), identity.emails, r["case_id"])

    def test_snapshot_hashes_are_sha256_of_email(self):
        for user_id, row in self.index.snapshot.items():
            self.assertEqual(main.email_hash(row["email"]), row["email_hash"], user_id)

    def test_historical_and_current_email_resolve_to_same_user(self):
        change = next(c for rows in self.index.changes.values() for c in rows if c["field_name"] == "email")
        far_future = "9999-12-31T00:00:00Z"
        old = self.index.resolve(change["old_value"], far_future)
        new = self.index.resolve(change["new_value"], far_future)
        self.assertEqual(old.user_id, change["user_id"])
        self.assertEqual(new.user_id, change["user_id"])
        self.assertIn(main.norm_email(change["old_value"]), new.emails)
        self.assertIn(change["audit_id"], new.audit_ids)

    def test_changes_after_cutoff_are_not_used(self):
        change = next(c for rows in self.index.changes.values() for c in rows if c["field_name"] == "email")
        before = self.index.identity_at(change["user_id"], "0000-01-01T00:00:00Z")
        self.assertIn(main.norm_email(change["old_value"]), before.emails)
        self.assertNotIn(change["audit_id"], before.audit_ids)
        if main.norm_email(change["new_value"]) != main.norm_email(self.index.snapshot[change["user_id"]]["email"]):
            return  # a later change replaced it; rollback of the latest value is covered by the next assertion
        self.assertNotIn(main.norm_email(change["new_value"]), before.emails)

    def test_phone_and_name_normalization(self):
        self.assertEqual(main.norm_phone("+34 612 847 593"), "34612847593")
        self.assertEqual(main.norm_name("  Mateo   Ríos "), "mateo ríos")


class Phase2EvidenceNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = main.Context()
        cls.store = cls.ctx.evidence

    def test_every_structured_file_yields_observations_or_coverage(self):
        produced = {o.file_id for o in self.store.observations} | {c.file_id for c in self.store.coverage}
        structured = [fid for fid, row in self.store.manifest.items()
                      if Path(row["file_path"]).suffix.lower() in evidence.STRUCTURED_PARSERS]
        self.assertEqual(len(structured), 79)
        self.assertEqual([fid for fid in structured if fid not in produced], [])

    def test_only_media_files_are_left_for_phase_3(self):
        suffixes = {Path(self.store.manifest[fid]["file_path"]).suffix.lower() for fid in self.store.unparsed}
        self.assertTrue(suffixes <= evidence.MEDIA_SUFFIXES, suffixes)

    def test_every_observation_has_system_and_resource(self):
        bad = [(o.file_id, o.resource_id) for o in self.store.observations if not o.system_id or not o.resource_id]
        self.assertEqual(bad, [])

    def test_learned_systems_match_sample_answers(self):
        for case_id, (_, steps) in main.read_output(SAMPLE_OUTPUT).items():
            for step in steps:
                self.assertEqual(self.store.systems.system_for(step["resource_id"]), step["system_id"], case_id)

    def _obs(self, file_id, resource_id):
        found = [o for o in self.store.by_resource[resource_id] if o.file_id == file_id and o.exists is not None]
        self.assertTrue(found, (file_id, resource_id))
        return found[0]

    def test_log_trace_join_and_flags(self):
        obs = self._obs("evidence_94", "data_flow_4713")
        self.assertEqual(obs.attrs.get("enabled"), "false")
        self.assertIn("user_49", obs.lookup_keys | obs.user_refs)

    def test_log_absence_and_lookup_key(self):
        obs = self._obs("evidence_94", "account_2064")
        self.assertFalse(obs.exists)
        self.assertIn("charlotte.wright@outlook.com", obs.lookup_keys)

    def test_sql_replica_row_and_listing_coverage(self):
        obs = self._obs("evidence_54", "account_005")
        self.assertTrue(obs.exists)
        self.assertIn("failover", obs.container)
        listings = [c for c in self.store.coverage if c.file_id == "evidence_54"]
        self.assertTrue(any("account_005" in c.resource_ids for c in listings))
        self.assertTrue(all("account_001" not in c.resource_ids for c in listings))

    def test_empty_scoped_snapshot_is_coverage(self):
        listing = [c for c in self.store.coverage if c.file_id == "evidence_90"]
        self.assertEqual([(c.system_id, c.resource_ids) for c in listing], [("account_service", set())])

    def test_readback_signals_are_interpreted(self):
        """F-43: readbacks that return a record, and explicit availability, decide existence."""
        infer = evidence.infer_exists
        self.assertTrue(infer({"event": "message_read", "resource_id": "support_1", "message_text": "[REDACTED]"}))
        self.assertTrue(infer({"event": "analytics_profile_read", "resource_id": "analytics_1", "direct_identifiers": "0"}))
        self.assertTrue(infer({"event": "connector_heartbeat", "resource_id": "data_flow_1", "enabled": "true"}))
        self.assertTrue(infer({"event": "deletion_marker_check", "resource_id": "support_1", "marker_found": "false"}))
        self.assertFalse(infer({"event": "export_download_check_completed", "resource_id": "export_1", "available": "false"}))
        self.assertIsNone(infer({"event": "profile_lookup_started", "resource_id": "account_1", "lookup_key": "a@b.co"}))

    def test_json_null_payload_is_absence(self):
        nulls = [o for o in self.store.observations if o.attrs.get("status_code") == 404]
        self.assertTrue(nulls)
        self.assertTrue(all(o.exists is False for o in nulls))

    def test_visible_respects_audit_cutoff(self):
        resource_id = "account_7566"
        all_obs = self.store.by_resource[resource_id]
        cutoff = min(o.observed_at for o in all_obs)
        visible = self.store.visible(resource_id, cutoff)
        self.assertTrue(all(o.observed_at <= cutoff and self.store.captured_at(o.file_id) <= cutoff for o in visible))
        self.assertEqual(self.store.visible(resource_id, "0000-01-01T00:00:00Z"), [])


class Phase3LLMLayer(unittest.TestCase):
    """Runs from the committed cache: must make zero API calls."""

    @classmethod
    def setUpClass(cls):
        cls.ctx = main.Context()
        cls.store = cls.ctx.evidence

    def test_rerun_makes_zero_api_calls(self):
        summary = self.ctx.llm.meter.summary()
        self.assertTrue(summary)
        self.assertEqual(sum(row["calls"] for row in summary), 0, summary)

    def test_cache_has_no_stale_entries(self):
        cached = {p.stem for p in llm.CACHE_DIR.glob("*.json")}
        self.assertEqual(cached, self.ctx.llm.used_keys)

    def test_every_media_file_is_extracted(self):
        media = [fid for fid, row in self.store.manifest.items()
                 if Path(row["file_path"]).suffix.lower() in evidence.MEDIA_SUFFIXES]
        self.assertEqual(len(media), 15)
        produced = {o.file_id for o in self.store.observations} | {f["file_id"] for f in self.store.facts}
        self.assertEqual([fid for fid in media if fid not in produced], [])
        self.assertEqual(self.store.unparsed, [])

    def test_media_times_never_exceed_capture_time(self):
        for obs in self.store.observations:
            if obs.container == "media":
                self.assertLessEqual(obs.observed_at, self.store.captured_at(obs.file_id), obs.resource_id)

    def test_media_id_swap_is_normalized(self):
        answer = {"records": [{"resource_id": "user_7", "user_refs": ["analytics_12"], "emails": ["A@B.co"]}]}
        obs, _, facts = evidence.media_answer_to_observations("f", answer, "2026-01-01T00:00:00Z", self.store.systems)
        self.assertEqual((obs[0].resource_id, obs[0].user_refs, obs[0].emails), ("analytics_12", {"user_7"}, {"a@b.co"}))
        self.assertEqual(facts, [])

    def test_every_request_is_labelled(self):
        for name in main.REQUEST_FILES:
            for row in main.read_csv(main.DATASET_DIR / name):
                self.assertIn(row["request_text"], self.ctx.request_labels, row["case_id"])

    def test_all_sample_requests_authorize_deletion(self):
        # Every sample answer is COMPLETE or INCOMPLETE, never NOT_APPLICABLE.
        for row in main.read_csv(SAMPLE_REQUESTS):
            self.assertTrue(self.ctx.authorizes_deletion(row), row["case_id"])

    def test_non_deletion_count_matches_intake_classifications(self):
        # Sanity cross-check: the intake service logged one non-deletion classification per such request.
        predicted = sum(not self.ctx.authorizes_deletion(r) for r in main.read_csv(main.DEFAULT_REQUESTS))
        logged = sum(a["event_type"] == "CLASSIFY_REQUEST" and "non-deletion" in a["details"] for a in self.ctx.activity)
        self.assertEqual(predicted, logged)


class Phase4StateReconstruction(unittest.TestCase):
    """Resource discovery and state at audit_at, scored against what the sample answers imply."""

    FINDING_FACT = {
        "REPORT_REMAINING_RECORD": ("exists", True), "REPORT_EXPOSED_PII": ("exposed", True),
        "REPORT_ACTIVE_DATA_FLOW": ("enabled", True), "REPORT_RESTORE_RISK": ("restore_allowed", True),
    }
    VERIFY_FACT = {
        "VERIFY_RECORD_ABSENT": ("exists", False), "VERIFY_RECORD_PRESENT": ("exists", True),
        "VERIFY_RECORD_REDACTED": ("exposed", False), "VERIFY_RECORD_ANONYMIZED": ("exposed", False),
        "VERIFY_DATA_FLOW_DISABLED": ("enabled", False), "VERIFY_RESTORE_BLOCKED": ("restore_allowed", False),
    }

    @classmethod
    def setUpClass(cls):
        cls.ctx = main.Context()
        cls.rec = cls.ctx.reconstructor
        cls.truth = main.read_output(SAMPLE_OUTPUT)
        cls.samples = {}
        for row in main.read_csv(SAMPLE_REQUESTS):
            identity = cls.ctx.identities.resolve(row["email"], row["audit_at"])
            cls.samples[row["case_id"]] = (row, identity, cls.rec.reconstruct(row, identity))

    def test_resource_discovery_on_samples(self):
        want = got = hit = 0
        for case_id, (_, _, states) in self.samples.items():
            expected = {s["resource_id"] for s in self.truth[case_id][1]}
            found = {s.resource_id for s in states}
            want, got, hit = want + len(expected), got + len(found), hit + len(expected & found)
        recall, precision = hit / want, hit / got
        print(f"\n    resource discovery: recall {hit}/{want}={recall:.3f}  precision {hit}/{got}={precision:.3f}")
        self.assertGreaterEqual(recall, 0.84)
        self.assertGreaterEqual(precision, 0.90)

    def test_states_and_evidence_match_sample_answers(self):
        bad, ev_bad = [], []
        for case_id, (_, _, states) in self.samples.items():
            status, steps = self.truth[case_id]
            by_id = {s.resource_id: s for s in states}
            reported = {s["resource_id"] for s in steps if s["phase"] == "FINDING"}
            for step in steps:
                state = by_id.get(step["resource_id"])
                if state is None:
                    continue
                if step["phase"] == "FINDING":
                    attr, want = self.FINDING_FACT[step["operation"]]
                elif step["phase"] == "VERIFICATION" and step["resource_id"] not in reported:
                    attr, want = self.VERIFY_FACT[step["operation"]]
                else:
                    continue
                fact = getattr(state, attr)
                value = None if fact is None else fact.value
                if not (value == want or (value is None and status == "INCOMPLETE")):
                    bad.append((case_id, step["resource_id"], attr, want, fact))
                if step["evidence_id"] and (fact is None or fact.evidence_id != step["evidence_id"]):
                    ev_bad.append((case_id, step["resource_id"], step["evidence_id"], fact))
        self.assertEqual(bad, [])
        self.assertEqual(ev_bad, [])

    def test_no_fact_uses_evidence_after_audit_at(self):
        captured = {fid: row["captured_at"] for fid, row in self.ctx.evidence.manifest.items()}
        activity_time = {row["activity_id"]: row["event_time"] for row in self.ctx.activity}
        for request in self.ctx.requests:
            identity = self.ctx.identities.resolve(request["email"], request["audit_at"])
            if identity is None or not self.ctx.authorizes_deletion(request):
                continue
            for state in self.rec.reconstruct(request, identity):
                for fact in (state.exists, state.exposed, state.user_ref, state.enabled, state.restore_allowed):
                    if fact is not None:
                        when = captured.get(fact.evidence_id) or activity_time.get(fact.evidence_id)
                        self.assertIsNotNone(when, fact)
                        self.assertLessEqual(when, request["audit_at"], (request["case_id"], state.resource_id, fact))

    def test_run_resources_are_assigned_to_at_most_one_request(self):
        owners = {}
        for case_id, resources in self.rec.run_assignment.items():
            for resource_id in resources:
                self.assertNotIn(resource_id, owners, (resource_id, owners.get(resource_id), case_id))
                owners[resource_id] = case_id

    def test_reconstruction_is_deterministic(self):
        again = main.Context().reconstructor
        self.assertEqual(again.run_assignment, self.rec.run_assignment)

    def test_subject_scoped_listing_does_not_disprove_other_subjects(self):
        # A bundle that lists records only for the subjects it covers must not mark everyone else absent.
        scoped = [c for c in self.ctx.evidence.coverage if c.subject_scoped]
        self.assertTrue(scoped)
        for case_id, (_, identity, states) in self.samples.items():
            for state in states:
                if state.exists is not None and not state.exists.value:
                    cov = [c for c in scoped if c.file_id == state.exists.evidence_id]
                    if cov:
                        self.assertIn(state.exists.evidence_id, self.rec.subject_files(identity, self.samples[case_id][0]["audit_at"]))

    def test_restore_scheduler_and_identifier_count_signals(self):
        """F-43: restore authorization decisions, connector scheduling, identifier counts."""
        R = reconstruct.Reconstructor
        obs = lambda attrs: evidence.Observation("f", "x_1", None, "2026-01-01T00:00:00Z", True, "", attrs)
        self.assertFalse(R._restore_value(obs({"decision": "deny", "policy": "privacy_deletion"})))
        self.assertTrue(R._restore_value(obs({"eligible": "true", "deletion_block": "false"})))
        self.assertFalse(R._enabled_value(obs({"scheduler_registered": "false"})))
        self.assertEqual(reconstruct._count("1"), 1)
        self.assertEqual(reconstruct._count(None), 0)

    def test_ocr_punctuation_snap_is_unambiguous(self):
        obs = evidence.Observation("f", "data_flow_1", None, "2026-01-01T00:00:00Z", True, "media", emails={"ana.lee@x.com"})
        fixed = evidence.snap_ocr_emails([obs], {"ana-lee@x.com", "bob@x.com"})
        self.assertEqual((fixed, obs.emails), (1, {"ana-lee@x.com"}))
        obs2 = evidence.Observation("f", "data_flow_2", None, "2026-01-01T00:00:00Z", True, "media", emails={"ana.lee@x.com"})
        evidence.snap_ocr_emails([obs2], {"ana-lee@x.com", "analee@x.com"})  # two twins: leave untouched
        self.assertEqual(obs2.emails, {"ana.lee@x.com"})


class Phase5Policy(unittest.TestCase):
    VERIFY_STATE = {
        "VERIFY_RECORD_ABSENT": "ABSENT", "VERIFY_RECORD_PRESENT": "PRESENT", "VERIFY_RECORD_REDACTED": "REDACTED",
        "VERIFY_RECORD_ANONYMIZED": "ANONYMIZED", "VERIFY_DATA_FLOW_DISABLED": "DISABLED", "VERIFY_RESTORE_BLOCKED": "BLOCKED",
    }

    @classmethod
    def setUpClass(cls):
        cls.ctx = main.Context()
        cls.policy = cls.ctx.policy

    def test_required_final_states_match_sample_verification(self):
        truth = main.read_output(SAMPLE_OUTPUT)
        checked, bad = 0, []
        for row in main.read_csv(SAMPLE_REQUESTS):
            identity = self.ctx.identities.resolve(row["email"], row["audit_at"])
            want = {s["resource_id"]: self.VERIFY_STATE[s["operation"]]
                    for s in truth[row["case_id"]][1] if s["phase"] == "VERIFICATION"}
            for state in self.ctx.reconstructor.reconstruct(row, identity):
                if state.resource_id in want:
                    checked += 1
                    got = self.policy.requirement(state, row["audit_at"]).final_state
                    if got != want[state.resource_id]:
                        bad.append((row["case_id"], state.resource_id, want[state.resource_id], got))
        self.assertGreaterEqual(checked, 90)
        self.assertEqual(bad, [])

    def test_every_category_is_a_rule_or_data_flow(self):
        rules = {r["data_category"] for r in main.read_csv(main.DATASET_DIR / "deletion_rules.csv")}
        for request in self.ctx.requests:
            identity = self.ctx.identities.resolve(request["email"], request["audit_at"])
            if identity is None or not self.ctx.authorizes_deletion(request):
                continue
            for state in self.ctx.reconstructor.reconstruct(request, identity):
                req = self.policy.requirement(state, request["audit_at"])
                self.assertIn(req.category, rules | {"DATA_FLOW"}, (request["case_id"], state.resource_id))

    # -- one test per rule branch, on synthetic states ------------------------------

    def _state(self, resource_id, system_id, record_types=(), containers=(), attrs=None, activity=(), retention=None):
        state = reconstruct.ResourceState(resource_id, system_id, "direct")
        state.record_types = set(record_types)
        state.containers = set(containers)
        state.retention_until = retention
        state.activity = list(activity)
        if attrs is not None:
            state.observations = [evidence.Observation("f", resource_id, system_id, "2026-01-01T00:00:00Z", True,
                                                       next(iter(containers), ""), attrs)]
        return state

    def _final(self, state, at="2026-09-01T00:00:00Z"):
        return self.policy.requirement(state, at)

    def test_tombstone_is_redacted_not_deleted(self):
        req = self._final(self._state("tombstone_1", "identity_service", containers=["deleted_customer_references"]))
        self.assertEqual((req.category, req.final_state), ("ACCOUNT_TOMBSTONE", "REDACTED"))

    def test_backup_is_retained_with_restore_blocked(self):
        req = self._final(self._state("backup_1", "backup_service", containers=["backup_catalog"]))
        self.assertEqual((req.category, req.final_state), ("BACKUP_ARCHIVE", "BLOCKED"))

    def test_data_flow_must_be_disabled(self):
        self.assertEqual(self._final(self._state("data_flow_1", "integration_service")).final_state, "DISABLED")

    def test_security_audit_log_retained_only_within_retention(self):
        inside = self._state("audit_record_1", "security_service", retention="2030-01-01T00:00:00Z")
        expired = self._state("audit_record_2", "security_service", retention="2025-01-01T00:00:00Z")
        self.assertEqual(self._final(inside).final_state, "PRESENT")
        self.assertEqual(self._final(expired).final_state, "ABSENT")

    def test_legal_hold_retained_while_active(self):
        req = self._final(self._state("legal_hold_1", "legal_service", retention="2033-12-31T23:59:59Z"))
        self.assertEqual((req.category, req.final_state), ("LEGAL_HOLD_RECORD", "PRESENT"))

    def test_analytics_is_anonymized(self):
        req = self._final(self._state("analytics_1", "analytics_platform", containers=["customer_analytics_reports"]))
        self.assertEqual(req.final_state, "ANONYMIZED")

    def test_contact_lookup_is_a_secondary_identifier(self):
        state = self._state("analytics_2", "analytics_platform", containers=["analytics_platform"],
                            attrs={"event": "contact_lookup_completed", "correlation_key": "abc"})
        req = self._final(state)
        self.assertEqual((req.category, req.final_state), ("SECONDARY_IDENTIFIER", "ABSENT"))

    def test_support_attachment_deleted_and_conversation_redacted(self):
        attachment = self._state("support_1", "support_platform", containers=["ticket_attachments"],
                                 attrs={"file_name": "x.pdf", "uploaded_by": "a@b.co"})
        conversation = self._state("support_2", "support_platform", containers=["support_conversations"], attrs={"message_text": "hi"})
        self.assertEqual(self._final(attachment).final_state, "ABSENT")
        self.assertEqual(self._final(conversation).final_state, "REDACTED")

    def test_conflicting_labels_prefer_the_more_specific(self):
        state = self._state("support_3", "support_platform", record_types=["SUPPORT_MESSAGE", "SUPPORT_ATTACHMENT"], containers=["media"])
        self.assertEqual(self._final(state).category, "SUPPORT_ATTACHMENT")

    def test_workflow_delete_disambiguates_generic_support_listing(self):
        row = {"source_system": "deletion_worker", "event_type": "DELETE_RECORD"}
        state = self._state("support_4", "support_platform", containers=["support_records:support_records"],
                            attrs={"message_excerpt": "x"}, activity=[row])
        self.assertEqual(self._final(state).category, "SUPPORT_ATTACHMENT")
        explicit = self._state("support_5", "support_platform", record_types=["SUPPORT_MESSAGE"], activity=[row])
        self.assertEqual(self._final(explicit).category, "SUPPORT_MESSAGE")  # explicit type is not overridden


class Phase6PlannerAndVerdict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = main.Context()
        cls.truth = main.read_output(SAMPLE_OUTPUT)
        cls.samples = {r["case_id"]: (r, main.audit_case(r, cls.ctx)) for r in main.read_csv(SAMPLE_REQUESTS)}
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output_path = Path(cls.tmp.name) / "output.csv"
        main.run(main.DEFAULT_REQUESTS, cls.output_path, cls.ctx)
        cls.output = main.read_output(cls.output_path)
        cls.requests = {r["case_id"]: r for r in main.read_csv(main.DEFAULT_REQUESTS)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _key(step):
        return (step["phase"], step["operation"], step["system_id"], step["resource_id"], step["evidence_id"])

    def test_sample_scores(self):
        status_ok = exact = want = got = hit = 0
        for case_id, (_, (status, steps)) in self.samples.items():
            t_status, t_steps = self.truth[case_id]
            status_ok += status == t_status
            exact += steps == t_steps
            w, g = Counter(map(self._key, t_steps)), Counter(map(self._key, steps))
            want, got, hit = want + sum(w.values()), got + sum(g.values()), hit + sum((w & g).values())
        print(f"\n    samples: status {status_ok}/32  exact {exact}/32  step P {hit/got:.3f} R {hit/want:.3f}")
        self.assertGreaterEqual(status_ok, 31)
        self.assertGreaterEqual(exact, 20)
        self.assertGreaterEqual(hit / got, 0.95)
        self.assertGreaterEqual(hit / want, 0.90)

    def test_every_evaluation_row_is_valid_and_present(self):
        self.assertEqual(sorted(self.output, key=main.case_sort_key), [r["case_id"] for r in main.read_csv(main.DEFAULT_REQUESTS)])
        for case_id, (status, steps) in self.output.items():
            self.assertEqual(main.validate_row(case_id, status, steps), [], case_id)

    def test_cited_evidence_exists_and_precedes_audit_at(self):
        when = {fid: row["captured_at"] for fid, row in self.ctx.evidence.manifest.items()}
        when.update({row["activity_id"]: row["event_time"] for row in self.ctx.activity})
        when.update({row["audit_id"]: row["event_time"] for row in main.read_csv(main.DATASET_DIR / "user_audit_log.csv")})
        for case_id, (_, steps) in self.output.items():
            for step in steps:
                if step["evidence_id"]:
                    self.assertIn(step["evidence_id"], when, (case_id, step))
                    self.assertLessEqual(when[step["evidence_id"]], self.requests[case_id]["audit_at"], (case_id, step))

    def test_plans_are_consistent(self):
        """Every finding is remediated, every remediated resource is verified, containment comes first."""
        for case_id, (status, steps) in self.output.items():
            if status != "INCOMPLETE":
                continue
            found = {(s["system_id"], s["resource_id"]) for s in steps if s["phase"] == "FINDING"}
            fixed = {(s["system_id"], s["resource_id"]) for s in steps if s["phase"] == "REMEDIATION"}
            verified = {(s["system_id"], s["resource_id"]) for s in steps if s["phase"] == "VERIFICATION"}
            self.assertEqual(found, fixed, case_id)
            self.assertTrue(fixed <= verified, case_id)
            ops = [s["operation"] for s in steps if s["phase"] == "REMEDIATION"]
            containment = [i for i, op in enumerate(ops) if op in ("DISABLE_DATA_FLOW", "BLOCK_RESTORE")]
            changes = [i for i, op in enumerate(ops) if op not in ("DISABLE_DATA_FLOW", "BLOCK_RESTORE")]
            if containment and changes:
                self.assertLess(max(containment), min(changes), case_id)

    def test_requests_with_account_candidates_get_an_account(self):
        """Overlapping batches must not starve a request of its account (F-41)."""
        rec = self.ctx.reconstructor
        for case_id, request in self.requests.items():
            if not self.ctx.authorizes_deletion(request):
                continue
            has_candidate = any(rec.is_account(r["resource_id"]) for r in rec.run_events(request))
            identity = self.ctx.identities.resolve(request["email"], request["audit_at"])
            linked = rec.links_for(request, identity)
            if has_candidate:
                self.assertTrue(any(rec.is_account(r) for r in linked), case_id)

    def test_non_deletion_requests_are_not_applicable(self):
        for case_id, request in self.requests.items():
            if not self.ctx.authorizes_deletion(request):
                self.assertEqual(self.output[case_id], ("NOT_APPLICABLE", []))

    def test_run_is_deterministic(self):
        again = Path(self.tmp.name) / "again.csv"
        main.run(main.DEFAULT_REQUESTS, again, main.Context())
        self.assertEqual(again.read_bytes(), self.output_path.read_bytes())

    # -- assess() branches on synthetic states ----------------------------------------

    def _assess(self, final, category="X", **facts):
        state = reconstruct.ResourceState("r_1", "s", "direct")
        for name, value in facts.items():
            setattr(state, name, reconstruct.Fact(value, f"ev_{name}", "2026-01-01T00:00:00Z"))
        requirement = policy.Requirement(category, None, "", final, "")
        return planner.assess(state, requirement)

    def test_assess_branches(self):
        self.assertEqual(self._assess("ABSENT", exists=True).violation.evidence_id, "ev_exists")
        self.assertEqual(self._assess("ABSENT", exists=False).proof.evidence_id, "ev_exists")
        self.assertIsNotNone(self._assess("REDACTED", "ACCOUNT_TOMBSTONE", exists=True, exposed=True).violation)
        tomb = self._assess("REDACTED", "ACCOUNT_TOMBSTONE", exists=True, exposed=False, user_ref=True)
        self.assertIsNone(tomb.violation)  # a tombstone may keep the stable user_id
        anon = self._assess("ANONYMIZED", "ANALYTICS_PROFILE", exists=True, exposed=False, user_ref=True)
        self.assertIsNotNone(anon.violation)  # analytics still attributable through user_id
        self.assertIsNotNone(self._assess("DISABLED", enabled=True).violation)
        self.assertEqual(self._assess("DISABLED", enabled=False).proof.evidence_id, "ev_enabled")
        self.assertIsNotNone(self._assess("BLOCKED", restore_allowed=True).violation)
        self.assertIsNotNone(self._assess("BLOCKED", exists=True, exposed=True).violation)  # no block in evidence
        self.assertIsNone(self._assess("PRESENT", exists=True).violation)


class Phase7Evaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(main.__file__).parent / "evaluation"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("evaluation_main", Path(main.__file__).parent / "evaluation" / "main.py")
        cls.ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ev)

    def test_custom_cases_pass_and_temp_files_are_deleted(self):
        before = set(Path(tempfile.gettempdir()).glob("ghost_custom_cases_*"))
        truth, predicted, _ = self.ev.run_custom_cases()
        self.assertGreaterEqual(len(truth), 3)
        for case_id in truth:
            self.assertEqual(predicted[case_id], truth[case_id], case_id)
        self.assertEqual(set(Path(tempfile.gettempdir()).glob("ghost_custom_cases_*")), before)
        self.assertFalse((main.DATASET_DIR / "custom_requests.csv").exists())

    def test_scorer_on_known_inputs(self):
        step = main.make_step("VERIFY_RECORD_ABSENT", "account_service", "account_1", "evidence_1")
        truth = {"c": ("COMPLETE", main.order_steps([step]))}
        self.assertEqual(self.ev.score(truth, truth)["step_f1"], 1.0)
        wrong = {"c": ("COMPLETE", main.order_steps([dict(step, evidence_id="evidence_2")]))}
        result = self.ev.score(truth, wrong)
        self.assertEqual((result["step_precision"], result["field_accuracy"]["evidence_id"], result["field_accuracy"]["resource_id"]), (0.0, 0.0, 1.0))

    def test_report_has_required_sections(self):
        report = self.ev.REPORT.read_text(encoding="utf-8")
        for heading in ("Case-level `deletion_status` accuracy", "Step-level precision", "## Field-level accuracy",
                        "Sequence accuracy", "## Custom evaluation cases", "## Failures found and improvements made",
                        "## Model comparison"):
            self.assertIn(heading, report)
        self.assertGreaterEqual(report.count("**Result:** predicted"), 3)
        usage = (Path(main.__file__).parent / "evaluation" / "usage_report.md").read_text(encoding="utf-8")
        for heading in ("Model calls and tokens", "Average tokens per request", "Estimated cost per request", "| **All** |"):
            self.assertIn(heading, usage)


class Phase8SubmissionChecklist(unittest.TestCase):
    """README pre-submission checklist, automated."""

    ROOT = main.REPO_ROOT
    SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{16,}|sk-ant-[A-Za-z0-9_\-]{16,}|OPENAI_API_KEY\s*=\s*\S{8,}|AKIA[0-9A-Z]{16}")

    def test_root_output_csv_matches_contract(self):
        path = self.ROOT / "output.csv"
        self.assertTrue(path.exists(), "run python code/main.py first")
        with open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], main.OUTPUT_HEADER)
        self.assertEqual([r[0] for r in rows[1:]], [r["case_id"] for r in main.read_csv(main.DEFAULT_REQUESTS)])
        for case_id, (status, steps) in main.read_output(path).items():
            self.assertEqual(main.validate_row(case_id, status, steps), [], case_id)

    def test_reports_exist_where_the_readme_requires(self):
        for name in ("evaluation_report.md", "usage_report.md"):
            self.assertTrue((self.ROOT / "code" / "evaluation" / name).exists(), name)

    def test_no_secrets_in_packaged_sources(self):
        files = [self.ROOT / "README.md", *(self.ROOT / "code").rglob("*")]
        for path in files:
            if path.is_file() and path.suffix in (".py", ".md", ".txt", ".json", ".csv", ".toml", ".cfg"):
                self.assertIsNone(self.SECRET_RE.search(path.read_text(encoding="utf-8", errors="ignore")), path)

    def test_env_file_is_git_ignored(self):
        self.assertIn(".env", (self.ROOT / ".gitignore").read_text(encoding="utf-8").split())

    def test_log_entries_have_tool_and_model(self):
        log = (self.ROOT / "log.txt").read_text(encoding="utf-8")
        entries = [e for e in re.split(r"\n(?=## \[)", log) if e.strip().startswith("## [")]
        self.assertGreater(len(entries), 5)
        for entry in entries:
            header = entry.splitlines()[0]
            self.assertEqual(len(re.findall(r"^tool=\S.*$", entry, re.M)), 1, header)
            self.assertEqual(len(re.findall(r"^model=\S.*$", entry, re.M)), 1, header)
            self.assertNotIn("model=unknown", entry, header)

    def test_code_zip_contents(self):
        import zipfile
        path = self.ROOT / "code.zip"
        if not path.exists():
            self.skipTest("code.zip not built yet")
        names = [n.replace("\\", "/") for n in zipfile.ZipFile(path).namelist()]
        for required in ("README.md", "code/main.py", "code/evaluation/main.py",
                         "code/evaluation/evaluation_report.md", "code/evaluation/usage_report.md", "code/requirements.txt"):
            self.assertIn(required, names)
        for name in names:
            self.assertFalse(name.startswith("dataset/") or "/dataset/" in name, name)
            self.assertNotIn("__pycache__", name)
            self.assertFalse(name.endswith((".zip", ".tar", ".gz", ".tgz", ".7z", ".rar")), name)
            self.assertNotEqual(Path(name).name, ".env", name)
            self.assertNotIn(".venv", name)


class Phase9Fallbacks(unittest.TestCase):
    """Unhappy paths: every failure degrades to a valid output.csv instead of aborting."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.saved_env = {k: os.environ.get(k) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}

    def tearDown(self):
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def _no_keys(self):
        os.environ["OPENAI_API_KEY"] = ""      # set before LLM() so .env cannot supply a key
        os.environ["ANTHROPIC_API_KEY"] = ""

    def test_no_api_key_and_empty_cache_still_writes_a_valid_output(self):
        self._no_keys()
        ctx = main.Context(llm=llm.LLM(cache_dir=self.root / "empty_cache"))
        results = main.run(main.DEFAULT_REQUESTS, self.root / "output.csv", ctx)
        self.assertEqual(len(results), 518)
        for case_id, (status, steps) in main.read_output(self.root / "output.csv").items():
            self.assertEqual(main.validate_row(case_id, status, steps), [], case_id)
        self.assertEqual({l["source"] for l in ctx.request_labels.values()}, {"rules"})
        self.assertTrue(any("skipped" in w for w in ctx.warnings))           # screenshots need a model
        truth = main.read_output(SAMPLE_OUTPUT)
        ok = sum(main.audit_case(r, ctx)[0] == truth[r["case_id"]][0] for r in main.read_csv(SAMPLE_REQUESTS))
        self.assertGreaterEqual(ok, 31)

    def test_rule_based_classifier_matches_the_model_labels(self):
        ctx = main.Context()
        disagree = [t for t, l in ctx.request_labels.items() if main.rule_based_label(t)["authorizes_deletion"] != l["authorizes_deletion"]]
        self.assertLessEqual(len(disagree), 4, disagree)

    def test_malformed_classifier_answer_is_split_and_retried(self):
        class HalfAnswers:
            calls = 0
            def complete_json(self, system, prompt, **_):
                HalfAnswers.calls += 1
                ids = [json.loads(line)["id"] for line in prompt.split("Messages:\n", 1)[1].splitlines() if line.strip()]
                keep = ids if len(ids) == 1 else ids[: len(ids) // 2]       # drops half of every batch
                return {"results": [{"id": i, "authorizes_deletion": True, "intent": "deletion"} for i in keep]}
        warnings = []
        labels = main.classify_requests(HalfAnswers(), [f"please delete my account {i}" for i in range(8)], warnings)
        self.assertEqual(len(labels), 8)
        self.assertTrue(all(l["source"] == "llm" for l in labels.values()))
        self.assertGreater(HalfAnswers.calls, 1)
        self.assertTrue(any("retrying in halves" in w for w in warnings))

    def test_provider_fallback_and_corrupt_cache_entry(self):
        os.environ["OPENAI_API_KEY"] = "test-key-not-used"
        os.environ["ANTHROPIC_API_KEY"] = "test-key-not-used"
        client = llm.LLM(provider="openai", cache_dir=self.root / "cache")
        calls = []
        def fake_call(provider, model, *args):
            calls.append(provider)
            if provider == "openai":
                raise RuntimeError("simulated outage")
            return {"ok": True}, {"input_tokens": 1, "output_tokens": 1}
        client._call = fake_call
        self.assertEqual(client.complete_json("s", "p"), {"ok": True})
        self.assertEqual(calls, ["openai", "anthropic"])
        self.assertTrue(any("fallback provider anthropic" in e for e in client.events))
        for path in (self.root / "cache").glob("*.json"):
            path.write_text("{not json", encoding="utf-8")                     # corrupt the cache entry
        self.assertEqual(client.complete_json("s", "p"), {"ok": True})          # treated as a miss, re-asked
        self.assertTrue(any("unreadable cache entry" in e for e in client.events))

    def test_bad_and_missing_evidence_files_are_isolated(self):
        dataset = self.root / "dataset"
        shutil.copytree(main.DATASET_DIR, dataset)
        (dataset / "evidence" / "records" / "broken.json").write_text("{ this is not json", encoding="utf-8")
        sql = "CREATE TABLE customer_profiles (resource_id TEXT, email TEXT);\nINSERT INTO customer_profiles VALUES ('account_1','a@b.co');\nTHIS IS NOT SQL;\n"
        (dataset / "evidence" / "databases" / "partly_broken.sql").write_text(sql, encoding="utf-8")
        with open(dataset / "evidence_files.csv", "a", encoding="utf-8", newline="") as handle:
            handle.write("evidence_8001,evidence/records/broken.json,test,2026-01-01T00:00:00Z\n")
            handle.write("evidence_8002,evidence/records/does_not_exist.json,test,2026-01-01T00:00:00Z\n")
            handle.write("evidence_8003,evidence/databases/partly_broken.sql,test,2026-01-01T00:00:00Z\n")
        ctx = main.Context(dataset_dir=dataset)
        skipped = {fid for fid, _ in ctx.evidence.errors}
        self.assertEqual(skipped, {"evidence_8001", "evidence_8002"})
        self.assertIn("evidence_8003", {o.file_id for o in ctx.evidence.observations})   # good statements kept
        results = main.run(main.DEFAULT_REQUESTS, self.root / "output.csv", ctx)
        self.assertEqual(len(results), 518)

    def test_a_crashing_case_does_not_stop_the_run(self):
        ctx = main.Context()
        real = ctx.reconstructor.reconstruct
        victim = next(r["case_id"] for r in main.read_csv(main.DEFAULT_REQUESTS) if ctx.authorizes_deletion(r))
        def flaky(request, identity):
            if request["case_id"] == victim:
                raise RuntimeError("simulated bug")
            return real(request, identity)
        ctx.reconstructor.reconstruct = flaky
        results = main.run(main.DEFAULT_REQUESTS, self.root / "output.csv", ctx)
        self.assertEqual(len(results), 518)
        self.assertEqual(results[victim], ("NOT_APPLICABLE", []))
        self.assertTrue(any(victim in w and "audit failed" in w for w in ctx.warnings))

    def test_invalid_rows_are_repaired(self):
        finding = main.make_step("REPORT_REMAINING_RECORD", "search_service", "search_1", "evidence_1")
        fix = dict(main.make_step("DELETE_RECORD", "search_service", "search_1"), evidence_id="evidence_9")
        verify = main.make_step("VERIFY_RECORD_ABSENT", "search_service", "search_1")
        status, steps = main.repair_row("c", "INCOMPLETE", [verify, fix, finding])        # wrong order + bad evidence
        self.assertEqual(main.validate_row("c", status, steps), [])
        status, steps = main.repair_row("c", "INCOMPLETE", [fix, verify])                 # no finding left
        self.assertEqual((status, steps), ("NOT_APPLICABLE", []))


class NoHardcodedAnswers(unittest.TestCase):
    """Pipeline code must not branch on case ids, evidence ids or evidence file names."""

    def test_pipeline_sources_are_generic(self):
        names = {Path(row["file_path"]).name for row in main.read_csv(main.DATASET_DIR / "evidence_files.csv")}
        pattern = re.compile(r"\bcase_\d+|\bevidence_\d+|\bactivity_\d+|\baudit_\d+")
        for source in ("main.py", "evidence.py", "llm.py", "reconstruct.py", "policy.py", "planner.py"):
            text = (Path(main.__file__).parent / source).read_text(encoding="utf-8")
            self.assertEqual(pattern.findall(text), [], source)
            self.assertEqual([n for n in names if n in text], [], source)


if __name__ == "__main__":
    unittest.main()

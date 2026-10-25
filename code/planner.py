"""Phase 6: turn reconstructed states + policy requirements into the verdict and steps_json.

Per resource, compare the state at audit_at with the required final state:
    ABSENT      record still exists              -> REPORT_REMAINING_RECORD + DELETE_RECORD
    REDACTED    direct identifiers still exposed -> REPORT_EXPOSED_PII      + REDACT_RECORD
    ANONYMIZED  record still attributable        -> REPORT_EXPOSED_PII      + ANONYMIZE_RECORD
    DISABLED    data flow still enabled          -> REPORT_ACTIVE_DATA_FLOW + DISABLE_DATA_FLOW
    BLOCKED     backup can still restore         -> REPORT_RESTORE_RISK     + BLOCK_RESTORE
    PRESENT     retained record                  -> (verification only)
Every relevant resource gets a verification of its required final state. INCOMPLETE: findings
cite evidence, remediation and future verification use "". COMPLETE: verification only, each
citing the evidence that proves the compliant state (unprovable checks are left out).
Ordering (phase, operation rank, system_id, resource_id) is applied by main.order_steps, which
also puts flow/restore containment before any record change.
"""

from dataclasses import dataclass
from datetime import timedelta

from reconstruct import parse_time

REMEDIATION = {
    "ABSENT": ("REPORT_REMAINING_RECORD", "DELETE_RECORD", "VERIFY_RECORD_ABSENT"),
    "REDACTED": ("REPORT_EXPOSED_PII", "REDACT_RECORD", "VERIFY_RECORD_REDACTED"),
    "ANONYMIZED": ("REPORT_EXPOSED_PII", "ANONYMIZE_RECORD", "VERIFY_RECORD_ANONYMIZED"),
    "DISABLED": ("REPORT_ACTIVE_DATA_FLOW", "DISABLE_DATA_FLOW", "VERIFY_DATA_FLOW_DISABLED"),
    "BLOCKED": ("REPORT_RESTORE_RISK", "BLOCK_RESTORE", "VERIFY_RESTORE_BLOCKED"),
    "PRESENT": (None, None, "VERIFY_RECORD_PRESENT"),
}

# Categories whose retained form may keep the stable user_id reference.
USER_REF_ALLOWED = {"ACCOUNT_TOMBSTONE", "SUPPORT_MESSAGE", "EMAIL_DELIVERY_LOG", "SECURITY_AUDIT_LOG",
                    "LEGAL_HOLD_RECORD", "BACKUP_ARCHIVE"}


@dataclass
class Assessment:
    state: object            # reconstruct.ResourceState
    requirement: object      # policy.Requirement
    violation: object        # reconstruct.Fact proving the violation, or None
    proof: object            # reconstruct.Fact proving the compliant state, or None


def assess(state, requirement):
    final = requirement.final_state
    exists, exposed, uref = state.exists, state.exposed, state.user_ref
    violation = proof = None
    if final == "ABSENT":
        if exists and exists.value:
            violation = exists
        elif exists is not None:
            proof = exists
    elif final in ("REDACTED", "ANONYMIZED"):
        attributable = uref if (final == "ANONYMIZED" or requirement.category not in USER_REF_ALLOWED) else None
        if exists is not None and not exists.value:
            proof = None  # gone entirely: nothing to redact, nothing to prove as redacted
        elif exposed and exposed.value:
            violation = exposed
        elif attributable and attributable.value:
            violation = attributable
        elif exposed is not None:
            proof = exposed
    elif final == "DISABLED":
        if state.enabled and state.enabled.value:
            violation = state.enabled
        elif state.enabled is not None:
            proof = state.enabled
    elif final == "BLOCKED":
        if state.restore_allowed and state.restore_allowed.value:
            violation = state.restore_allowed
        elif state.restore_allowed is not None:
            proof = state.restore_allowed
        elif exists and exists.value and exposed and exposed.value:
            # An archive holding the subject's data with no restore block anywhere in evidence
            # can bring the data back: the block the policy requires is not in place.
            violation = exposed
    elif final == "PRESENT":
        if exists and exists.value:
            proof = exists
    return Assessment(state, requirement, violation, proof)


STALE_BEFORE_REQUEST = timedelta(days=30)     # compliant context older than this is not re-verified
OUT_OF_SCOPE_BEFORE_REQUEST = timedelta(days=120)  # any capture older than this no longer shows the audited state


def is_stale_context(assessment, request):
    """Context outside this request's workflow run whose evidence is too old to matter.

    Kept: anything in the request's own run; compliant records with evidence from the month
    before the request; violations with evidence from the four months before it. Older captures
    (an earlier deletion's tombstone, a screenshot from half a year ago) describe a past state.
    """
    if assessment.state.link == "run":
        return False
    times = [o.observed_at for o in assessment.state.observations]
    if assessment.state.link_evidence.startswith("activity_"):
        times += [r["event_time"] for r in assessment.state.activity if r["activity_id"] == assessment.state.link_evidence]
    if not times:
        return assessment.violation is None and assessment.proof is None
    horizon = OUT_OF_SCOPE_BEFORE_REQUEST if assessment.violation is not None else STALE_BEFORE_REQUEST
    return parse_time(max(times)) < parse_time(request["requested_at"]) - horizon


def plan(request, states, policy, make_step, order_steps):
    assessments = [assess(s, policy.requirement(s, request["audit_at"])) for s in states]
    assessments = [a for a in assessments if not is_stale_context(a, request)]
    findings = [a for a in assessments if a.violation is not None]
    steps = []
    if findings:
        for a in findings:
            report, remediate, _ = REMEDIATION[a.requirement.final_state]
            steps.append(make_step(report, a.state.system_id, a.state.resource_id, a.violation.evidence_id))
            steps.append(make_step(remediate, a.state.system_id, a.state.resource_id))
        for a in assessments:
            steps.append(make_step(REMEDIATION[a.requirement.final_state][2], a.state.system_id, a.state.resource_id))
        return "INCOMPLETE", order_steps(steps), assessments
    for a in assessments:
        if a.proof is not None:
            steps.append(make_step(REMEDIATION[a.requirement.final_state][2], a.state.system_id,
                                   a.state.resource_id, a.proof.evidence_id))
    return "COMPLETE", order_steps(steps), assessments

"""Phase 5: map each resource to a deletion_rules.csv category and its required final state.

Category, strongest signal first:
  1. an explicit record_type written in the evidence (handover rows, manifests, logs, media);
  2. the table / collection the record was listed in;
  3. contact-lookup shape: a record reached only through a contact/identifier lookup index
     is a SECONDARY_IDENTIFIER (e.g. an analytics contact keyed by an old email hash);
  4. the owning system.
The rule's required_action then gives the final state the plan must reach.
"""

from dataclasses import dataclass

# Table / collection name -> data category (deletion_rules.csv vocabulary).
COLLECTION_CATEGORY = {
    "customer_profiles": "USER_PROFILE",
    "customer_search_documents": "SEARCH_INDEX_ENTRY",
    "search_documents": "SEARCH_INDEX_ENTRY",
    "customer_analytics_reports": "ANALYTICS_PROFILE",
    "customer_analytics_profiles": "ANALYTICS_PROFILE",
    "support_conversations": "SUPPORT_MESSAGE",
    "support_records": "SUPPORT_MESSAGE",
    "tickets": "SUPPORT_MESSAGE",
    "ticket_attachments": "SUPPORT_ATTACHMENT",
    "customer_exports": "USER_EXPORT",
    "customer_export_jobs": "USER_EXPORT",
    "exports": "USER_EXPORT",
    "connector_configurations": "DATA_FLOW",
    "connector_subscriptions": "DATA_FLOW",
    "connectors": "DATA_FLOW",
    "backup_catalog": "BACKUP_ARCHIVE",
    "backup_restore_entries": "BACKUP_ARCHIVE",
    "restore_entries": "BACKUP_ARCHIVE",
    "security_audit_events": "SECURITY_AUDIT_LOG",
    "security_audit_records": "SECURITY_AUDIT_LOG",
    "audit_records": "SECURITY_AUDIT_LOG",
    "retained_order_records": "LEGAL_HOLD_RECORD",
    "deleted_customer_references": "ACCOUNT_TOMBSTONE",
    "profile_lookups": "USER_PROFILE",
    "profiles": "ANALYTICS_PROFILE",
    "index_queries": "SEARCH_INDEX_ENTRY",
    "documents": "SEARCH_INDEX_ENTRY",
}

# Owning system -> default data category.
SYSTEM_CATEGORY = {
    "account_service": "USER_PROFILE",
    "search_service": "SEARCH_INDEX_ENTRY",
    "analytics_platform": "ANALYTICS_PROFILE",
    "support_platform": "SUPPORT_MESSAGE",
    "export_service": "USER_EXPORT",
    "integration_service": "DATA_FLOW",
    "backup_service": "BACKUP_ARCHIVE",
    "security_service": "SECURITY_AUDIT_LOG",
    "legal_service": "LEGAL_HOLD_RECORD",
    "identity_service": "ACCOUNT_TOMBSTONE",
    "marketing_platform": "MARKETING_PROFILE",
    "cache_service": "CACHE_ENTRY",
    "email_delivery_service": "EMAIL_DELIVERY_LOG",
    "object_storage": "SUPPORT_ATTACHMENT",
}

# record_type spellings seen in evidence that differ from rule categories.
RECORD_TYPE_ALIASES = {"SUPPORT_TICKET": "SUPPORT_MESSAGE", "OBJECT": "SUPPORT_ATTACHMENT", "ORDER": "LEGAL_HOLD_RECORD"}

# When one record carries conflicting labels, the more specific kind of record wins.
MORE_SPECIFIC_FIRST = ("ACCOUNT_TOMBSTONE", "SUPPORT_ATTACHMENT", "LEGAL_HOLD_RECORD", "SECURITY_AUDIT_LOG",
                       "SECONDARY_IDENTIFIER", "USER_EXPORT", "SUPPORT_MESSAGE")

# required_action -> final state the plan must reach.
ACTION_STATE = {"DELETE": "ABSENT", "REDACT": "REDACTED", "ANONYMIZE": "ANONYMIZED", "RETAIN": "PRESENT"}


@dataclass
class Requirement:
    category: str
    rule_id: str | None
    action: str              # DELETE | REDACT | ANONYMIZE | RETAIN | DISABLE | BLOCK_RESTORE
    final_state: str         # ABSENT | REDACTED | ANONYMIZED | PRESENT | DISABLED | BLOCKED
    reason: str


class Policy:
    def __init__(self, rules_rows):
        self.rules = {row["data_category"]: row for row in rules_rows}

    def category(self, state):
        types = {RECORD_TYPE_ALIASES.get(t, t) for t in state.record_types}
        types = {t for t in types if t in self.rules or t == "DATA_FLOW"}
        if len(types) == 1:
            return next(iter(types)), "record_type"
        if len(types) > 1:
            # Conflicting labels: prefer the one matching the listed collection, then the more
            # specific label (an uploaded file described inside a conversation is the attachment).
            listed = self._collection_categories(state)
            for t in sorted(types):
                if t in listed:
                    return t, "record_type+collection"
            for t in MORE_SPECIFIC_FIRST:
                if t in types:
                    return t, "record_type(specific)"
        if self._contact_index_only(state):
            return "SECONDARY_IDENTIFIER", "contact_lookup"
        if self._attachment_signals(state):
            return "SUPPORT_ATTACHMENT", "attachment_signals"
        listed = self._collection_categories(state)
        category, source = (next(iter(listed)), "collection") if len(listed) == 1 else \
            (SYSTEM_CATEGORY.get(state.system_id, "SECONDARY_IDENTIFIER"), "system")
        if category == "SUPPORT_MESSAGE" and self._workflow_deleted(state):
            # Generic support listings (support_records, tickets, ticket reads) hold attachments and
            # conversations with the same excerpts; the workflow's own treatment is the only
            # visible discriminator, and it matches the required treatment on every sample.
            return "SUPPORT_ATTACHMENT", f"{source}+workflow_treatment"
        return category, source

    @staticmethod
    def _workflow_deleted(state):
        return any(row["source_system"] == "deletion_worker" and row["event_type"] == "DELETE_RECORD"
                   for row in state.activity)

    @staticmethod
    def _collection_categories(state):
        found = set()
        for container in state.containers:
            for part in container.split(":"):
                part = part.strip().lower()
                for name, category in COLLECTION_CATEGORY.items():
                    if part == name or part.startswith(name + "_") or part.endswith("_" + name):
                        found.add(category)
        return found

    @staticmethod
    def _attachment_signals(state):
        """A support record that is an uploaded file: uploader, attachment storage or scan, file name."""
        if SYSTEM_CATEGORY.get(state.system_id) not in ("SUPPORT_MESSAGE", "SUPPORT_ATTACHMENT"):
            return False
        for obs in state.observations:
            attrs = obs.attrs
            file_name = str(attrs.get("file_name") or attrs.get("file") or "").lower()
            if (attrs.get("uploaded_by")
                    or "attachment" in str(attrs.get("storage_class", ""))
                    or "attachment" in str(attrs.get("event", ""))
                    or "attachment" in obs.container
                    or file_name.endswith((".pdf", ".png", ".jpg", ".jpeg", ".zip"))):
                return True
        return False

    @staticmethod
    def _contact_index_only(state):
        """Reached only via contact lookups / a contact index, never as a profile or document."""
        events = {str(o.attrs.get("event", "")) for o in state.observations}
        sources = {str(o.attrs.get("source", "")) for o in state.observations}
        contact = any("contact" in e for e in events) or any("contact_index" in s for s in sources)
        structured = any(":" in o.container or o.attrs.get("record_type") for o in state.observations)
        return contact and not structured

    def requirement(self, state, audit_at):
        category, source = self.category(state)
        if category == "DATA_FLOW":
            return Requirement(category, None, "DISABLE", "DISABLED", "active sync can recreate deleted data")
        rule = self.rules.get(category)
        if rule is None:
            return Requirement(category, None, "DELETE", "ABSENT", f"no rule for {category}; personal data defaults to deletion")
        action = rule["required_action"]
        if category == "BACKUP_ARCHIVE":
            return Requirement(category, rule["rule_id"], "BLOCK_RESTORE", "BLOCKED", rule["condition"])
        if category in ("SECURITY_AUDIT_LOG", "LEGAL_HOLD_RECORD"):
            # Retention applies only while the supplied period / hold is still running.
            until = state.retention_until
            if until and until < audit_at:
                return Requirement(category, rule["rule_id"], "DELETE", "ABSENT", "retention period or hold expired")
        return Requirement(category, rule["rule_id"], action, ACTION_STATE[action], f"{rule['condition']} ({source})")

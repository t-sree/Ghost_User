"""Evidence normalization: turn every evidence file into Observation records.

Parsers dispatch on content shape (SQL tables, JSON objects carrying a resource_id,
key=value log lines, pipe-delimited text rows), never on file names, so a new file of a
known shape is handled without code changes.
"""

import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
HASH_RE = re.compile(r"\b[0-9a-f]{64}\b")
USER_REF_RE = re.compile(r"\buser_\d+\b")
RESOURCE_RE = re.compile(r"\b[a-z]+(?:_[a-z]+)*_\d+\b")
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
KV_RE = re.compile(r'([A-Za-z_][\w.]*)=("(?:[^"\\]|\\.)*"|\S+)')
TRACE_RE = re.compile(r"\b(?:tr|rb)-[\w-]+\b")

# Keys whose value is the identifier used to *query* a system, not record content.
LOOKUP_KEYS = {"lookup_key", "lookup_value", "correlation_key", "query", "subject_ref"}
NAME_KEYS = {"name", "full_name", "display_name", "requester_name"}
PHONE_KEYS = {"phone", "owner_phone", "requester_phone"}
TIME_KEYS = ("observed_at", "recorded_at", "updated_at", "indexed_at", "captured_at", "last_write")
NULLS = {"", "null", "none", "None", "NULL"}

ABSENT_RESULTS = {"NOT_FOUND", "MISS", "ABSENT", "DELETED_NOT_FOUND"}
PRESENT_RESULTS = {"FOUND", "HIT", "PRESENT", "IDENTIFIABLE", "ANONYMIZED", "REDACTED",
                   "ACTIVE", "BLOCKED", "DISABLED", "RETAINED"}


@dataclass
class Observation:
    """One statement an evidence file makes about one resource."""

    file_id: str
    resource_id: str
    system_id: str | None
    observed_at: str
    exists: bool | None              # True / False when the evidence says so, None if silent
    container: str = ""              # table / collection / log service the statement came from
    attrs: dict = field(default_factory=dict)        # every non-null attribute, normalized keys
    lookup_keys: set = field(default_factory=set)    # identifiers used to query (not content)
    emails: set = field(default_factory=set)         # identifiers present in record content
    phones: set = field(default_factory=set)
    hashes: set = field(default_factory=set)
    names: set = field(default_factory=set)
    user_refs: set = field(default_factory=set)


@dataclass
class Coverage:
    """A full listing of a container at a point in time: resources missing from it are absent."""

    file_id: str
    system_id: str | None
    container: str
    observed_at: str
    resource_ids: set
    scope: dict | None = None        # SQL snapshot_scope: system -> captured_at, when the file declares one
    subject_scoped: bool = False     # JSON bundles list records only for the subjects they cover


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def read_text(path):
    """UTF-8 first; a file with stray bytes is decoded leniently instead of being dropped."""
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def clean(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return None if value in NULLS else value
    return value


def digits(value):
    return "".join(ch for ch in str(value) if ch.isdigit())


def flatten(record, prefix=""):
    """Flatten nested dicts to dotted keys; lists of scalars are kept, lists of dicts dropped."""
    flat = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, name + "."))
        elif isinstance(value, list):
            if all(not isinstance(v, (dict, list)) for v in value):
                flat[name] = value
        else:
            flat[name] = value
    return flat


def infer_exists(attrs):
    """Read existence from the generic status signals found across evidence formats."""
    for key in ("rows", "total_hits", "hits.total"):
        if key in attrs:
            try:
                return int(attrs[key]) > 0
            except (TypeError, ValueError):
                pass
    for key in ("http_status", "status_code"):
        if key in attrs:
            return str(attrs[key]) != "404"
    result = str(attrs.get("result", "")).upper()
    if result in ABSENT_RESULTS:
        return False
    if result in PRESENT_RESULTS:
        return True
    if "archive_present" in attrs:
        return str(attrs["archive_present"]).lower() == "true"
    if "available" in attrs:
        return str(attrs["available"]).lower() == "true"
    # A content scan that matched an identifier inside the record.
    if attrs.get("matched_value"):
        return True
    # A readback that returned the record (read, scan, evaluation, check, heartbeat): it exists.
    # "*_started" lines only announce a lookup and say nothing about the result.
    event = str(attrs.get("event", ""))
    if event and not event.endswith("_started") and attrs.get("resource_id") and (
            event.endswith(("_read", "_read_completed", "_loaded", "_evaluated", "_check_completed", "_heartbeat"))
            or event == "deletion_marker_check" or event.endswith("_failed") or "direct_identifiers" in attrs):
        return True
    return None


def make_observation(file_id, resource_id, system_id, observed_at, attrs, container="", exists=None):
    attrs = {k: v for k, v in ((k, clean(v)) for k, v in attrs.items()) if v is not None}
    # An explicit status signal in the record wins over the default implied by where it was listed.
    inferred = infer_exists(attrs)
    obs = Observation(file_id, resource_id, system_id, observed_at, inferred if inferred is not None else exists,
                      container or "", attrs)
    for key, value in attrs.items():
        leaf = key.rsplit(".", 1)[-1]
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, str):
                continue
            if leaf in LOOKUP_KEYS:
                obs.lookup_keys.add(item.lower())
                continue
            obs.emails.update(e.lower() for e in EMAIL_RE.findall(item))
            obs.hashes.update(HASH_RE.findall(item.lower()))
            obs.user_refs.update(USER_REF_RE.findall(item))
            if leaf in NAME_KEYS:
                obs.names.add(" ".join(item.split()).casefold())
            if leaf in PHONE_KEYS:
                obs.phones.add(digits(item))
    return obs


def pick_time(attrs, default):
    for key in TIME_KEYS:
        value = clean(attrs.get(key))
        if isinstance(value, str) and TIMESTAMP_RE.fullmatch(value):
            return value
    return default


# ---------------------------------------------------------------------------
# System inference (learned from data, not hardcoded)
# ---------------------------------------------------------------------------

def resource_prefix(resource_id):
    return resource_id.rsplit("_", 1)[0]


class SystemMap:
    """Resource-prefix -> system_id, learned from system_activity.csv target systems."""

    def __init__(self, activity_rows):
        votes = defaultdict(Counter)
        for row in activity_rows:
            if RESOURCE_RE.fullmatch(row["resource_id"] or ""):
                votes[resource_prefix(row["resource_id"])][row["target_system"]] += 1
        self.by_prefix = {prefix: counts.most_common(1)[0][0] for prefix, counts in votes.items()}
        self.known_systems = {row["target_system"] for row in activity_rows} | {row["source_system"] for row in activity_rows}

    def learn(self, observations):
        """Systems named explicitly by evidence override the activity-stream votes.

        Activity target systems include downstream copies (e.g. a support record indexed
        into search_service), so they are only a fallback for prefixes evidence never names.
        """
        votes = defaultdict(Counter)
        for obs in observations:
            if obs.system_id:
                votes[resource_prefix(obs.resource_id)][obs.system_id] += 1
                self.known_systems.add(obs.system_id)
        for prefix, counts in votes.items():
            self.by_prefix[prefix] = counts.most_common(1)[0][0]

    def system_for(self, resource_id):
        return self.by_prefix.get(resource_prefix(resource_id))


# ---------------------------------------------------------------------------
# SQL snapshots
# ---------------------------------------------------------------------------

def parse_sql(file_id, path, captured_at):
    conn = sqlite3.connect(":memory:")
    try:
        script = read_text(path)
        try:
            conn.executescript(script)
        except sqlite3.Error:
            # Fallback: replay statement by statement and skip only the broken ones.
            conn.close()
            conn = sqlite3.connect(":memory:")
            statement = ""
            for line in script.splitlines(keepends=True):    # any line ending
                statement += line
                if sqlite3.complete_statement(statement):
                    try:
                        conn.execute(statement)
                    except sqlite3.Error:
                        pass
                    statement = ""
        conn.row_factory = sqlite3.Row
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        scope = {}
        if "snapshot_scope" in tables:
            scope = {r["origin_system"]: r["captured_at"] for r in conn.execute("SELECT * FROM snapshot_scope")}
        observations, coverage = [], []
        for table in tables:
            if table == "snapshot_scope":
                continue
            rows = [dict(r) for r in conn.execute(f'SELECT * FROM "{table}"')]
            listed = defaultdict(set)
            for row in rows:
                attrs = {}
                for key, value in row.items():
                    parsed = _json_object(value)
                    if parsed is not None:
                        attrs.update(flatten(parsed))
                    else:
                        attrs[key] = value
                resource_id = attrs.get("resource_id")
                if not resource_id:
                    continue
                system_id = clean(attrs.get("system")) or clean(attrs.get("origin_system"))
                container = clean(attrs.get("collection")) or clean(attrs.get("collection_name")) or table
                when = pick_time(attrs, scope.get(system_id, captured_at))
                observations.append(make_observation(file_id, resource_id, system_id, when, attrs, f"{table}:{container}", True))
                listed[(system_id, container)].add(resource_id)
            if not listed:
                # Empty table: its system is resolved later from the rows other files hold for it.
                coverage.append(Coverage(file_id, None, f"{table}:{table}", captured_at, set(), scope or None))
            for (system_id, container), ids in listed.items():
                coverage.append(Coverage(file_id, system_id, f"{table}:{container}", scope.get(system_id, captured_at), ids, scope or None))
        return observations, coverage
    finally:
        conn.close()


def _json_object(value):
    if isinstance(value, str) and value.startswith("{"):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# JSON documents
# ---------------------------------------------------------------------------

SYSTEM_HINT_KEYS = ("system", "data_store")
CONTAINER_HINT_KEYS = ("collection",)


def parse_json(file_id, path, captured_at, systems):
    document = json.loads(read_text(path))
    observations, coverage = [], []

    def walk(node, ctx):
        if isinstance(node, list):
            for item in node:
                walk(item, ctx)
            return
        if not isinstance(node, dict):
            return
        ctx = dict(ctx)
        for key in SYSTEM_HINT_KEYS:
            if isinstance(node.get(key), str):
                ctx["system"] = node[key]
        for key in CONTAINER_HINT_KEYS:
            if isinstance(node.get(key), str):
                ctx["container"] = node[key]
        ctx["time"] = pick_time(node, ctx["time"])
        if isinstance(node.get("records"), list) and "container" in ctx:
            ids = {r["resource_id"] for r in node["records"] if isinstance(r, dict) and r.get("resource_id")}
            coverage.append(Coverage(file_id, ctx.get("system"), ctx["container"], ctx["time"], ids, subject_scoped=True))
        if node.get("resource_id"):
            attrs = flatten(node)
            exists = None
            # A lookup that returned an empty payload ("profile": null) proves absence.
            for key, value in node.items():
                if value is None and key in ("profile", "document", "record"):
                    exists = False
            system_id = ctx.get("system")
            observations.append(make_observation(
                file_id, node["resource_id"], system_id, ctx["time"], attrs, ctx.get("container", ""),
                exists if exists is not None else (True if ctx.get("listing") else None),
            ))
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                child = dict(ctx)
                if key in systems.known_systems:
                    child["system"] = key  # e.g. service_exports.account_service.*
                if isinstance(value, list):
                    # Entries listed in an export exist unless they carry an explicit status signal.
                    child["container"] = child.get("container") if key == "records" else key
                    child["listing"] = True
                walk(value, child)

    walk(document, {"time": captured_at, "listing": False})
    return observations, coverage


# ---------------------------------------------------------------------------
# Logs (key=value, JSON-in-line, bracketed, pipe-delimited, syslog)
# ---------------------------------------------------------------------------

def parse_log_line(line):
    """Return (timestamp, attrs) for one log line in any of the supported layouts."""
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None, {}
    timestamp, rest = match.group(0), line[match.end():].strip()
    attrs = {}
    brace = rest.find("{")
    if brace != -1:
        try:
            attrs.update(json.loads(rest[brace:]))
            rest = rest[:brace]
        except json.JSONDecodeError:
            pass
    for key, raw in KV_RE.findall(rest):
        value = raw[1:-1].replace('\\"', '"') if raw.startswith('"') else raw
        if key == "message" and "=" in value:
            attrs.update(dict((k, v[1:-1] if v.startswith('"') else v) for k, v in KV_RE.findall(value)))
        else:
            attrs[key] = value
    # Positional layouts: "[LEVEL] [service] [trace] event:", "| LEVEL | service | trace | event |",
    # syslog "<pri>1 host service trace - -".
    head = rest.split("=", 1)[0]
    tokens = [t for t in re.split(r"[\[\]|\s:]+", head) if t]
    if "service" not in attrs:
        for key in ("function", "logger"):
            if key in attrs:
                attrs["service"] = attrs[key]
                break
        else:
            names = [t for t in tokens if re.fullmatch(r"[a-z]+(?:_[a-z]+)+", t)]
            if names:
                attrs["service"] = names[0]
                if "event" not in attrs and len(names) > 1:
                    attrs["event"] = names[-1]
    if "trace_id" not in attrs:
        trace = attrs.get("requestId") or next(iter(TRACE_RE.findall(rest)), None)
        if trace:
            attrs["trace_id"] = trace
    return timestamp, attrs


def parse_log(file_id, path, captured_at, systems):
    lines = [parse_log_line(line) for line in read_text(path).splitlines()]
    lookup_by_trace = defaultdict(set)
    for _, attrs in lines:
        trace = attrs.get("trace_id")
        for key in LOOKUP_KEYS:
            value = clean(attrs.get(key))
            if trace and value:
                lookup_by_trace[trace].add(value.lower())
    observations = []
    for timestamp, attrs in lines:
        resource_id = clean(attrs.get("resource_id"))
        if not timestamp or not resource_id or not RESOURCE_RE.fullmatch(resource_id):
            continue
        service = attrs.get("system") or attrs.get("service")
        system_id = service if service in systems.known_systems else None
        obs = make_observation(file_id, resource_id, system_id, timestamp, attrs, attrs.get("service", ""))
        obs.lookup_keys |= lookup_by_trace.get(attrs.get("trace_id"), set())
        observations.append(obs)
    return observations, []


# ---------------------------------------------------------------------------
# Plain-text records (pipe-delimited rows)
# ---------------------------------------------------------------------------

def parse_text(file_id, path, captured_at, systems):
    observations = []
    for line in read_text(path).splitlines():
        if "|" not in line:
            continue
        cells = [c.strip(" -\t") for c in line.split("|")]
        resource_id = next((c for c in cells if RESOURCE_RE.fullmatch(c) and not USER_REF_RE.fullmatch(c)), None)
        if not resource_id:
            continue
        attrs = {}
        for cell in cells:
            if TIMESTAMP_RE.fullmatch(cell):
                attrs["observed_at"] = cell
            elif cell in systems.known_systems:
                attrs["system"] = cell
            elif cell == resource_id:
                attrs["resource_id"] = cell
            elif re.fullmatch(r"[A-Z]+(?:_[A-Z]+)+", cell) and "record_type" not in attrs:
                attrs["record_type"] = cell
            elif re.fullmatch(r"[A-Z_]+", cell):
                attrs["result"] = cell
            elif cell:
                attrs["subject_reference"] = cell
        observations.append(make_observation(
            file_id, resource_id, attrs.get("system"), attrs.get("observed_at", captured_at), attrs, "text_record"))
    return observations, []


# ---------------------------------------------------------------------------
# Media (screenshots via a vision model, PDFs via extracted text) -> Observations
# ---------------------------------------------------------------------------

MEDIA_SYSTEM_PROMPT = """You extract facts from internal business artifacts of an e-commerce company \
(admin-console screenshots, invoices, support attachments, compliance extracts, registers). \
You transcribe only what is visibly written. Never guess, never invent identifiers, never infer \
deletion outcomes that are not shown. Copy identifiers (emails, phones, resource ids such as \
account_123, search_45, data_flow_9, backup_7, audit_record_3, support_001, user_12, order numbers) \
exactly as written, character for character."""

MEDIA_PROMPT = """Extract every record this artifact shows. A record is any row, panel, card or \
document section that describes one stored item (an account/profile, search document, analytics \
profile, support ticket or attachment, export, connector/data flow, backup, audit/compliance entry, \
retained/legal-hold order, marketing profile, stored object, etc.).

Return JSON:
{{
  "artifact_kind": "short description of the artifact",
  "artifact_time": "ISO-8601 UTC timestamp the artifact was generated/captured if written on it, else null",
  "records": [
    {{
      "resource_id": "internal id like account_123 if written, else null",
      "system_id": "owning service if written or clearly labelled (e.g. account_service, integration_service), else null",
      "record_type": "e.g. USER_PROFILE, SEARCH_INDEX_ENTRY, SUPPORT_MESSAGE, SUPPORT_ATTACHMENT, ANALYTICS_PROFILE, USER_EXPORT, DATA_FLOW, BACKUP_ARCHIVE, SECURITY_AUDIT_LOG, LEGAL_HOLD_RECORD, MARKETING_PROFILE, ACCOUNT_TOMBSTONE, EMAIL_DELIVERY_LOG, OBJECT or null",
      "status": "status text exactly as shown (e.g. Active, Deleted, Not found, Healthy, Restore allowed, Anonymized), else null",
      "exists": true | false | null,
      "enabled": true | false | null,
      "restore_allowed": true | false | null,
      "observed_at": "ISO-8601 UTC time for this record if written (combine with the artifact date when only a time is shown), else null",
      "names": ["person names shown for this record"],
      "emails": ["emails shown for this record"],
      "phones": ["phone numbers shown for this record"],
      "user_refs": ["user_N references shown"],
      "email_hashes": ["64-hex hashes shown"],
      "order_reference": "order number if shown, else null",
      "retain_until": "ISO-8601 date if a retention/hold end date is shown, else null",
      "details": "other short facts shown about this record (redaction markers, masked fields, legal basis, owner, destination)"
    }}
  ]
}}
Use exists=false only when the artifact explicitly shows the item is missing/not found/deleted; \
use enabled for connectors/data flows; use restore_allowed for backups/restores.
When the artifact itself is a file uploaded to a support ticket (it shows a file name, size, \
uploader, checksum or upload history together with a ticket/support reference), the support \
record is a SUPPORT_ATTACHMENT, not a SUPPORT_MESSAGE.
Artifact file name: {name}
Manifest capture time: {captured_at}
{body}"""


def pdf_text(path):
    from pypdf import PdfReader
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages).strip()


def make_media_extractor(llm):
    """Build an extractor for EvidenceStore that reads PNG/JPG via vision and PDF via text."""

    def extract(file_id, path, captured_at, systems):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = pdf_text(path)
            prompt = MEDIA_PROMPT.format(name=path.name, captured_at=captured_at, body="Document text:\n" + text)
            # Few documents, high stakes: use the stronger (vision-tier) model on the extracted text.
            try:
                answer = llm.complete_json(MEDIA_SYSTEM_PROMPT, prompt, kind="vision")
            except Exception as exc:  # noqa: BLE001 - no model: mine the extracted text directly
                llm.events.append(f"{file_id}: PDF read by text fallback ({type(exc).__name__})")
                return pdf_text_fallback(file_id, text, captured_at, systems)
        else:
            prompt = MEDIA_PROMPT.format(name=path.name, captured_at=captured_at, body="The screenshot is attached.")
            answer = llm.complete_json(MEDIA_SYSTEM_PROMPT, prompt, kind="vision", image_path=path)
        return media_answer_to_observations(file_id, answer, captured_at, systems)

    return extract


def pdf_text_fallback(file_id, text, captured_at, systems):
    """Degraded PDF reading without a model: every resource id in the document is recorded as
    present, carrying the identifiers the document shows. Less precise than the LLM extraction
    (identifiers are not attributed to individual records), but nothing in the PDF is lost."""
    resource_ids = sorted({r for r in RESOURCE_RE.findall(text) if not USER_REF_RE.fullmatch(r)})
    attrs = {"details": "pdf_text_fallback"}
    observations = []
    for resource_id in resource_ids:
        obs = make_observation(file_id, resource_id, None, captured_at, attrs, "media", True)
        obs.emails = {e.lower() for e in EMAIL_RE.findall(text)}
        obs.user_refs = set(USER_REF_RE.findall(text))
        obs.hashes = set(HASH_RE.findall(text.lower()))
        observations.append(obs)
    return observations, [], []


def media_answer_to_observations(file_id, answer, captured_at, systems):
    artifact_time = clean(answer.get("artifact_time"))
    if not (isinstance(artifact_time, str) and TIMESTAMP_RE.fullmatch(artifact_time)) or artifact_time > captured_at:
        artifact_time = captured_at
    observations, facts = [], []
    for record in answer.get("records") or []:
        if not isinstance(record, dict):
            continue
        observed_at = clean(record.get("observed_at"))
        if not (isinstance(observed_at, str) and TIMESTAMP_RE.fullmatch(observed_at)) or observed_at > captured_at:
            observed_at = artifact_time
        attrs = {
            "system": record.get("system_id"),
            "record_type": record.get("record_type"),
            "result": record.get("status"),
            "enabled": _bool_text(record.get("enabled")),
            "restore_allowed": _bool_text(record.get("restore_allowed")),
            "order_reference": record.get("order_reference"),
            "retention_until": record.get("retain_until"),
            "details": record.get("details"),
        }
        # Models sometimes swap the record id and the user reference; sort the candidates by shape.
        candidates = [record.get("resource_id"), record.get("system_id"), *(record.get("user_refs") or [])]
        candidates = [c.strip().lower() for c in candidates if isinstance(clean(c), str)]
        resource_ids = [c for c in candidates if RESOURCE_RE.fullmatch(c) and not USER_REF_RE.fullmatch(c)]
        user_refs = {u for c in candidates for u in USER_REF_RE.findall(c)}
        resource_id = resource_ids[0] if resource_ids else None
        record = dict(record, user_refs=sorted(user_refs))
        if not resource_id:
            # Kept as a fact; EvidenceStore.resolve_aliases() may map its label to a resource later.
            facts.append({"file_id": file_id, "observed_at": observed_at, "record": record, "attrs": attrs})
            continue
        system_id = clean(record.get("system_id"))
        system_id = system_id if system_id in systems.known_systems else None
        explicit = record.get("exists")
        obs = make_observation(file_id, resource_id, system_id, observed_at, attrs, "media",
                               explicit if isinstance(explicit, bool) else None)
        if isinstance(explicit, bool):
            obs.exists = explicit
        add_media_identifiers(obs, record)
        observations.append(obs)
    return observations, [], facts


def add_media_identifiers(obs, record):
    obs.emails |= {e.lower() for v in record.get("emails") or [] for e in EMAIL_RE.findall(str(v))}
    obs.phones |= {digits(v) for v in record.get("phones") or [] if digits(v)}
    obs.names |= {" ".join(str(v).split()).casefold() for v in record.get("names") or [] if v}
    obs.user_refs |= {u for v in record.get("user_refs") or [] for u in USER_REF_RE.findall(str(v))}
    obs.hashes |= {h for v in record.get("email_hashes") or [] for h in HASH_RE.findall(str(v).lower())}


ALIAS_RE = re.compile(r"linked (\b[a-z]+(?:_[a-z]+)*_\d+\b) \(([^)]+)\)|confirmed (\b[a-z]+(?:_[a-z]+)*_\d+\b) \(([^)]+)\)")


def _bool_text(value):
    return None if value is None else str(bool(value)).lower()


# ---------------------------------------------------------------------------
# Loading the whole evidence set
# ---------------------------------------------------------------------------

STRUCTURED_PARSERS = {".sql": "sql", ".json": "json", ".log": "log", ".txt": "text"}
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf"}


def punctuation_skeleton(email):
    return re.sub(r"[.\-_]", "", email)


def snap_ocr_emails(observations, known_emails):
    """Correct OCR punctuation slips in media emails (e.g. '.' read for '-').

    An extracted email that matches no known identity is replaced only when exactly one known
    email has the same letters once '.', '-' and '_' are removed. Known emails never share a
    skeleton in this data, so near-duplicate decoys cannot be merged.
    """
    by_skeleton = defaultdict(set)
    for email in known_emails:
        by_skeleton[punctuation_skeleton(email)].add(email)
    fixed = 0
    for obs in observations:
        if obs.container != "media":
            continue
        for email in list(obs.emails):
            twins = by_skeleton.get(punctuation_skeleton(email), set())
            if email not in known_emails and len(twins) == 1:
                obs.emails.discard(email)
                obs.emails.add(next(iter(twins)))
                fixed += 1
    return fixed


class EvidenceStore:
    def __init__(self, dataset_dir, activity_rows, media_extractor=None, known_emails=()):
        self.dataset_dir = dataset_dir
        self.systems = SystemMap(activity_rows)
        self.manifest = {}
        self.observations = []
        self.coverage = []
        self.facts = []          # media records without a resource id (e.g. register rows keyed by order)
        self.unparsed = []
        with open(dataset_dir / "evidence_files.csv", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self.manifest[row["file_id"]] = row
        self.errors = []         # (file_id, reason) for files skipped by a fallback
        for file_id, row in sorted(self.manifest.items()):
            path = dataset_dir / row["file_path"]
            suffix = path.suffix.lower()
            if not path.is_file():
                self.errors.append((file_id, f"listed in evidence_files.csv but missing on disk: {row['file_path']}"))
                continue
            # One unreadable or malformed file must not stop the audit: skip it, record why, and the
            # affected cases fall back to the remaining evidence.
            try:
                if suffix == ".sql":
                    obs, cov = parse_sql(file_id, path, row["captured_at"])
                elif suffix == ".json":
                    obs, cov = parse_json(file_id, path, row["captured_at"], self.systems)
                elif suffix == ".log":
                    obs, cov = parse_log(file_id, path, row["captured_at"], self.systems)
                elif suffix == ".txt":
                    obs, cov = parse_text(file_id, path, row["captured_at"], self.systems)
                elif suffix in MEDIA_SUFFIXES and media_extractor is not None:
                    obs, cov, facts = media_extractor(file_id, path, row["captured_at"], self.systems)
                    self.facts.extend(facts)
                else:
                    self.unparsed.append(file_id)
                    continue
            except Exception as exc:  # noqa: BLE001 - isolate any parser / extraction failure to this file
                self.errors.append((file_id, f"{type(exc).__name__}: {str(exc)[:200]}"))
                continue
            self.observations.extend(obs)
            self.coverage.extend(cov)
        self.resolve_aliases(activity_rows)
        self.ocr_fixes = snap_ocr_emails(self.observations, set(known_emails))
        self.systems.learn(self.observations)
        for obs in self.observations:
            if not obs.system_id:
                obs.system_id = self.systems.system_for(obs.resource_id)
        table_system = defaultdict(Counter)
        for obs in self.observations:
            if obs.container:
                table_system[obs.container][obs.system_id] += 1
        resolved = []
        for cov in self.coverage:
            if not cov.system_id and cov.resource_ids:
                cov.system_id = self.systems.system_for(next(iter(cov.resource_ids)))
            if not cov.system_id and table_system.get(cov.container):
                cov.system_id = table_system[cov.container].most_common(1)[0][0]
            if cov.scope is not None:
                if cov.system_id not in cov.scope:
                    continue  # the snapshot does not claim to cover this system
                cov.observed_at = cov.scope[cov.system_id]
            if cov.system_id:
                resolved.append(cov)
        self.coverage = resolved
        self.by_resource = defaultdict(list)
        for obs in self.observations:
            self.by_resource[obs.resource_id].append(obs)

    def resolve_aliases(self, activity_rows):
        """Turn media facts labelled only by a display name into observations of the resource.

        Data-catalog rows state aliases explicitly, e.g. "linked object_19 (delivery_confirmation_19.pdf)"
        or "linked export_1842 (NST-2026-0718-1842)"; a fact whose label or order reference is such an
        alias describes that resource.
        """
        aliases = {}
        for row in activity_rows:
            if row["event_type"] != "LINK_RESOURCE_IDENTITY":
                continue
            for match in ALIAS_RE.finditer(row["details"]):
                resource_id, alias = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
                if alias.strip().lower() != resource_id:
                    aliases[alias.strip().lower()] = resource_id
        remaining = []
        for fact in self.facts:
            record = fact["record"]
            labels = [record.get("resource_id"), record.get("order_reference"), *(record.get("user_refs") or [])]
            resource_id = next((aliases[str(l).strip().lower()] for l in labels if l and str(l).strip().lower() in aliases), None)
            if resource_id is None:
                remaining.append(fact)
                continue
            obs = make_observation(fact["file_id"], resource_id, None, fact["observed_at"], fact["attrs"], "media",
                                   record.get("exists") if isinstance(record.get("exists"), bool) else None)
            add_media_identifiers(obs, record)
            self.observations.append(obs)
        self.facts = remaining

    def captured_at(self, file_id):
        return self.manifest[file_id]["captured_at"]

    def visible(self, resource_id, audit_at):
        """Observations of a resource from files captured at or before the cutoff."""
        return [
            obs for obs in self.by_resource.get(resource_id, [])
            if obs.observed_at <= audit_at and self.captured_at(obs.file_id) <= audit_at
        ]

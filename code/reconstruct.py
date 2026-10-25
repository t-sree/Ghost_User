"""Phase 4: find each case's resources and reconstruct their state at audit_at.

Linking, in order of strength:
  1. direct  - evidence visible at audit_at carries one of the subject's identifiers
               (email, email hash, phone, user_id reference, lookup key) or a data-catalog
               LINK_RESOURCE_IDENTITY event names them;
  2. chained - a linked record or its activity names another resource (e.g. a search
               document's source_resource_id, "restore repopulated search_N");
  3. run     - the deletion workflow's own activity for the request. Runs of different
               requests overlap in time, so run resources are assigned by elimination
               across all requests: evidence owners first, one account per request,
               at most one resource per system per request.

State: evidence decides, activity rows are claims. For each resource the latest evidence
file that speaks about it wins; within that file any replica still holding the record
means PRESENT; a full listing (Coverage) that omits the record means ABSENT.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from evidence import EMAIL_RE, HASH_RE, RESOURCE_RE, USER_REF_RE, digits

# Deletion-run windows relative to the request's reported_at. The workflow submits the primary
# profile shortly before reporting (samples: -75 min .. +10 min), then works through downstream
# systems for up to a few hours.
ACCOUNT_BEFORE = timedelta(minutes=80)
ACCOUNT_AFTER = timedelta(minutes=15)
ACCOUNT_TYPICAL_OFFSET = timedelta(minutes=-60)   # samples: -75 .. -46 min for workflow-generated runs
ACCOUNT_FIT_BUCKET = timedelta(minutes=30)        # coarse enough that one batch's requests tie
RUN_BEFORE = timedelta(minutes=80)
RUN_AFTER = timedelta(hours=3)
AUDIT_READBACK = timedelta(days=2)  # evidence captured this close before audit_at is that audit's readback

# Words in a request that name a kind of system (last-resort tie-break for unbalanced run groups).
SYSTEM_HINTS = {
    "backup_service": ("backup", "restore", "sauvegarde", "copia de seguridad", "respaldo", "バックアップ", "백업"),
    "analytics_platform": ("analytics", "statistic", "analyse", "analítica", "estadística"),
    "search_service": ("search", "recherche", "búsqueda", "busca", "検索", "검색"),
    "support_platform": ("support", "ticket", "soporte", "suporte", "サポート"),
    "export_service": ("export", "download", "archive i requested", "exportación", "téléchargement"),
    "integration_service": ("integration", "connector", "sync", "intégration", "integración"),
}
TRUE_VALUES = {"true", "1", "yes", "enabled", "active"}
FALSE_VALUES = {"false", "0", "no", "disabled", "blocked", "stopped"}


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def flag(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


@dataclass
class Fact:
    value: object
    evidence_id: str
    at: str


@dataclass
class ResourceState:
    resource_id: str
    system_id: str
    link: str                                   # direct | chained | run
    link_evidence: str = ""
    exists: Fact | None = None                  # record present / absent
    exposed: Fact | None = None                 # subject's direct identifiers still in the record
    user_ref: Fact | None = None                # subject's user_id still referenced
    enabled: Fact | None = None                 # data flow active
    restore_allowed: Fact | None = None         # backup can restore
    record_types: set = field(default_factory=set)
    containers: set = field(default_factory=set)
    retention_until: str | None = None             # latest stated retention end, if any
    observations: list = field(default_factory=list)  # visible observations at audit_at
    activity: list = field(default_factory=list)   # activity rows about this resource up to audit_at


class IdentifierOwners:
    """Reverse index: identifier -> user_ids that ever held it (to spot other people's records)."""

    def __init__(self, identities):
        self.owner = defaultdict(set)
        far_future = "9999-12-31T23:59:59Z"
        for user_id in identities.snapshot:
            ident = identities.identity_at(user_id, far_future)
            for value in ident.emails | ident.email_hashes | ident.phones | {user_id}:
                self.owner[value].add(user_id)

    def users_in(self, obs):
        users = set()
        for value in obs.emails | obs.hashes | obs.phones | obs.user_refs | obs.lookup_keys:
            users |= self.owner.get(value, set())
        return users


def subject_keys(identity):
    return identity.emails | identity.email_hashes | identity.phones | {identity.user_id}


def observation_matches(obs, identity):
    return bool(
        obs.emails & identity.emails or obs.hashes & identity.email_hashes or obs.phones & identity.phones
        or identity.user_id in obs.user_refs or obs.lookup_keys & subject_keys(identity)
    )


class Reconstructor:
    def __init__(self, ctx, cases):
        """`cases`: list of (request, identity) for every deletion-authorizing request."""
        self.ctx = ctx
        self.store = ctx.evidence
        self.owners = IdentifierOwners(ctx.identities)
        self.activity_by_resource = defaultdict(list)
        for row in ctx.activity:
            self.activity_by_resource[row["resource_id"]].append(row)
        self.link_events = [row for row in ctx.activity if row["event_type"] == "LINK_RESOURCE_IDENTITY"]
        # resource -> [(resource whose activity mentions it, event time)]
        self.mentioned_by = defaultdict(list)
        for row in ctx.activity:
            for other in RESOURCE_RE.findall(row["details"]):
                if other != row["resource_id"] and not USER_REF_RE.fullmatch(other):
                    self.mentioned_by[other].append((row["resource_id"], row["event_time"]))
        self.account_system = self.store.systems.system_for("account_0")
        self._subject_files = {}
        self.run_assignment = self._assign_runs(cases)

    # -- visibility -------------------------------------------------------

    def visible(self, obs, at):
        return obs.observed_at <= at and self.store.captured_at(obs.file_id) <= at

    def subject_files(self, identity, at):
        """Files with visible content about the subject (subject-scoped listings apply to them)."""
        key = (identity.user_id, at)
        if key not in self._subject_files:
            self._subject_files[key] = {
                o.file_id for o in self.store.observations if self.visible(o, at) and observation_matches(o, identity)
            }
        return self._subject_files[key]

    # -- linking ----------------------------------------------------------

    def direct_links(self, identity, at):
        links = {}
        for obs in self.store.observations:
            if self.visible(obs, at) and observation_matches(obs, identity):
                links.setdefault(obs.resource_id, obs.file_id)
        for row in self.link_events:
            if row["event_time"] > at:
                continue
            details = row["details"].lower()
            if {e.lower() for e in EMAIL_RE.findall(details)} & identity.emails or identity.user_id in USER_REF_RE.findall(details):
                links.setdefault(row["resource_id"], row["activity_id"])
        return links

    def chained_links(self, linked, at):
        """Resources named inside linked records (attributes or activity details)."""
        found = {}
        for resource_id in linked:
            for obs in self.store.by_resource.get(resource_id, []):
                if not self.visible(obs, at):
                    continue
                for key, value in obs.attrs.items():
                    if key != "resource_id" and isinstance(value, str) and RESOURCE_RE.fullmatch(value) \
                            and not USER_REF_RE.fullmatch(value) and value != resource_id:
                        found.setdefault(value, obs.file_id)
            for row in self.activity_by_resource.get(resource_id, []):
                if row["event_time"] <= at and row["event_type"] != "LINK_RESOURCE_IDENTITY":
                    for other in RESOURCE_RE.findall(row["details"]):
                        if other != resource_id and not USER_REF_RE.fullmatch(other):
                            found.setdefault(other, row["activity_id"])
        return {r: e for r, e in found.items() if r not in linked and self.store.systems.system_for(r)}

    def run_events(self, request):
        reported = parse_time(request["reported_at"])
        events = []
        for row in self.ctx.activity:
            if row["source_system"] != "deletion_worker" or row["event_time"] > request["audit_at"]:
                continue
            offset = parse_time(row["event_time"]) - reported
            if self.is_account(row["resource_id"]):
                if -ACCOUNT_BEFORE <= offset <= ACCOUNT_AFTER:
                    events.append(row)
            elif -RUN_BEFORE <= offset <= RUN_AFTER:
                events.append(row)
        return events

    def is_account(self, resource_id):
        return self.store.systems.system_for(resource_id) == self.account_system

    def owners_of(self, resource_id, at, hop=True):
        """Users that visible evidence, catalog links or a related resource tie the resource to."""
        users = set()
        for obs in self.store.by_resource.get(resource_id, []):
            if self.visible(obs, at):
                users |= self.owners.users_in(obs)
        for row in self.activity_by_resource.get(resource_id, []):
            if row["event_type"] == "LINK_RESOURCE_IDENTITY" and row["event_time"] <= at:
                users |= set(USER_REF_RE.findall(row["details"]))
        if hop:
            # e.g. "Restore job repopulated downstream resource search_N" on another user's backup.
            for other in self.mentioned_by.get(resource_id, ()):
                if other[1] <= at:
                    users |= self.owners_of(other[0], at, hop=False)
        return users

    def foreign(self, resource_id, identity, at):
        """True when visible evidence ties the resource to someone other than the subject."""
        users = self.owners_of(resource_id, at)
        return bool(users) and identity.user_id not in users

    def has_evidence(self, resource_id, at, within=None):
        """Visible evidence of the resource, optionally captured no earlier than `within` before `at`."""
        floor = (parse_time(at) - within) if within else None
        return any(
            self.visible(o, at) and (floor is None or parse_time(self.store.captured_at(o.file_id)) >= floor)
            for o in self.store.by_resource.get(resource_id, [])
        )

    def _assign_runs(self, cases):
        """Assign deletion-run resources to requests by elimination across all requests.

        Rules, applied until nothing changes:
          - a resource directly linked by evidence belongs to that request only;
          - a request holds at most one resource per system (exactly one account);
          - a resource claimed by exactly one request goes to it; a request left with
            exactly one account candidate takes it.
        Symmetric leftovers are paired by id order only when the number of resources equals
        the number of requests still missing that system; otherwise they stay unassigned.
        """
        system = self.store.systems.system_for
        direct, candidates, audit_at = {}, {}, {}
        self.request_text = {request["case_id"]: request["request_text"] for request, _ in cases}
        self.reported_at = {request["case_id"]: request["reported_at"] for request, _ in cases}
        self.run_time = {}
        for request, identity in cases:
            cid, at = request["case_id"], request["audit_at"]
            audit_at[cid] = at
            direct[cid] = self.direct_links(identity, at)
            candidates[cid] = {}
            for row in self.run_events(request):
                rid = row["resource_id"]
                if rid in direct[cid] or not self.foreign(rid, identity, at):
                    candidates[cid].setdefault(rid, row["activity_id"])
                    self.run_time.setdefault((cid, rid), row["event_time"])

        owned = defaultdict(set)
        for cid, links in direct.items():
            for rid in links:
                owned[rid].add(cid)
        assigned = {cid: {} for cid in candidates}
        for cid, cands in candidates.items():
            for rid in list(cands):
                if cid in owned.get(rid, ()):
                    assigned[cid][rid] = cands.pop(rid)
                elif rid in owned:
                    del cands[rid]

        def held_systems(cid):
            return {system(r) for r in assigned[cid]} | {system(r) for r in direct[cid]}

        while True:
            taken = {rid for res in assigned.values() for rid in res}
            for cid, cands in candidates.items():
                have = held_systems(cid)
                for rid in list(cands):
                    if rid in taken or system(rid) in have:
                        del cands[rid]
            claimants = defaultdict(set)
            for cid, cands in candidates.items():
                for rid in cands:
                    claimants[rid].add(cid)
            progress = False
            for rid, cids in sorted(claimants.items()):
                if len(cids) == 1:
                    cid = next(iter(cids))
                    if system(rid) not in held_systems(cid):
                        assigned[cid][rid] = candidates[cid].pop(rid)
                        progress = True
            for cid, cands in candidates.items():
                accounts = [r for r in cands if self.is_account(r)]
                if len(accounts) == 1 and self.account_system not in held_systems(cid):
                    assigned[cid][accounts[0]] = cands.pop(accounts[0])
                    progress = True
            if not progress:
                break

        # Symmetric leftovers: within each connected group of (requests, resources) of one system,
        # pair by id order when the counts balance (or always for accounts: every request has one).
        for component_cases, component_rids in self._components(candidates, system):
            sys_id = system(next(iter(component_rids)))
            needing = sorted((c for c in component_cases if sys_id not in held_systems(c)),
                             key=lambda c: self._case_order_key(c, assigned))
            if not (len(needing) == len(component_rids) or sys_id == self.account_system):
                # Unbalanced: a request that explicitly talks about this kind of system ("you may
                # keep backups but they must not restore my data") is the only remaining signal.
                hinted = [c for c in needing if self._mentions_system(c, sys_id)]
                if len(hinted) != len(component_rids):
                    continue
                needing = hinted
            if sys_id == self.account_system:
                self._match_accounts(needing, component_rids, candidates, assigned, audit_at)
                continue
            free = sorted(component_rids, key=_numeric_key)
            for cid, rid in zip(needing, free):
                if rid in candidates[cid]:
                    assigned[cid][rid] = candidates[cid].pop(rid)
        # Unresolved candidates are kept only as a last resort for requests left with nothing.
        self.leftovers = {cid: dict(cands) for cid, cands in candidates.items() if cands}
        return assigned

    def _match_accounts(self, needing, rids, candidates, assigned, audit_at):
        """Pair requests with accounts inside one tied component, best pairs first.

        Pair quality: (1) the account was looked up by readback evidence captured shortly before
        this request's own audit_at; (2) the deletion time fits the workflow's typical offset
        (primary profile submitted ~1 h before reported_at), in coarse 30-minute buckets: that
        separates overlapping batches, while inside one batch (same reported_at) every request
        fits equally and pairs fall back to id order (lowest account with the first request).
        """
        pairs = []
        for cid in needing:
            reported = parse_time(self.reported_at[cid])
            for rid in rids:
                if rid not in candidates[cid]:
                    continue
                when = parse_time(self.run_time[(cid, rid)])
                fit_bucket = int(abs((when - reported) - ACCOUNT_TYPICAL_OFFSET) / ACCOUNT_FIT_BUCKET)
                seen = self.has_evidence(rid, audit_at[cid], AUDIT_READBACK)
                pairs.append(((not seen, fit_bucket, _numeric_key(rid), self._case_order_key(cid, assigned)), cid, rid))
        done_cases, done_rids = set(), set()
        for _, cid, rid in sorted(pairs):
            if cid in done_cases or rid in done_rids:
                continue
            assigned[cid][rid] = candidates[cid].pop(rid)
            done_cases.add(cid)
            done_rids.add(rid)

    def _mentions_system(self, case_id, system_id):
        text = self.request_text.get(case_id, "").lower()
        return any(word in text for word in SYSTEM_HINTS.get(system_id, ()))

    @staticmethod
    def _components(candidates, system):
        """Connected components of the request-resource candidate graph, per system."""
        edges = defaultdict(set)
        for cid, cands in candidates.items():
            for rid in cands:
                edges[("c", cid, system(rid))].add(("r", rid))
                edges[("r", rid)].add(("c", cid, system(rid)))
        seen, components = set(), []
        for node in sorted(edges):
            if node in seen or node[0] != "r":
                continue
            stack, cases, rids = [node], set(), set()
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                (rids.add(current[1]) if current[0] == "r" else cases.add(current[1]))
                stack.extend(edges[current] - seen)
            components.append((cases, rids))
        return components

    def _case_order_key(self, case_id, assigned):
        accounts = sorted((r for r in assigned[case_id] if self.is_account(r)), key=_numeric_key)
        return _numeric_key(accounts[0]) if accounts else (case_id, 0)

    def links_for(self, request, identity):
        at = request["audit_at"]
        links = {rid: ("direct", ev) for rid, ev in self.direct_links(identity, at).items()}
        for rid, ev in self.run_assignment.get(request["case_id"], {}).items():
            links.setdefault(rid, ("run", ev))
        if not links:
            # Nothing linked at all: fall back to this request's own unresolved run candidates.
            for rid, ev in self.leftovers.get(request["case_id"], {}).items():
                links.setdefault(rid, ("run", ev))
        for rid, ev in self.chained_links(links, at).items():
            links.setdefault(rid, ("chained", ev))
        return links

    # -- state ------------------------------------------------------------

    def state(self, resource_id, link, link_evidence, identity, at):
        system_id = self.store.systems.system_for(resource_id)
        rs = ResourceState(resource_id, system_id, link, link_evidence)
        rs.activity = [row for row in self.activity_by_resource.get(resource_id, []) if row["event_time"] <= at]
        visible = [o for o in self.store.by_resource.get(resource_id, []) if self.visible(o, at)]
        rs.observations = visible
        for obs in sorted(visible, key=lambda o: o.observed_at):
            until = obs.attrs.get("retention_until") or obs.attrs.get("retained_until")
            if isinstance(until, str) and until[:4].isdigit():
                rs.retention_until = until

        # Existence: latest file that makes a statement; any present copy in it wins.
        statements = defaultdict(list)    # file_id -> [(time, exists)]
        for obs in visible:
            rs.containers.add(obs.container)
            if obs.attrs.get("record_type"):
                rs.record_types.add(str(obs.attrs["record_type"]).upper())
            if obs.exists is not None:
                statements[obs.file_id].append((obs.observed_at, obs.exists))
        # A full listing proves absence only for what it covers: any listing of the system when
        # the record was never observed explicitly, otherwise only listings of a table or
        # collection the record was actually seen in (a primary-table listing says nothing
        # about a failover copy; a conversations listing says nothing about an attachment).
        subject_files = self.subject_files(identity, at)
        seen_in = {o.container for o in visible if o.exists is not None}
        for cov in self.store.coverage:
            if cov.system_id == system_id and resource_id not in cov.resource_ids \
                    and cov.observed_at <= at and self.store.captured_at(cov.file_id) <= at \
                    and (not cov.subject_scoped or cov.file_id in subject_files) \
                    and (not seen_in or cov.container in seen_in):
                statements[cov.file_id].append((cov.observed_at, False))
        if statements:
            # Files that speak about this record or this subject (an explicit row or lookup, a
            # subject's bundle) outrank generic table listings that merely omit it; then latest.
            explicit = {o.file_id for o in visible if o.exists is not None} | subject_files
            file_id = max(statements, key=lambda f: (f in explicit, max(t for t, _ in statements[f]), f))
            rows = statements[file_id]
            rs.exists = Fact(any(e for _, e in rows), file_id, max(t for t, _ in rows))

        # Content facts come from the latest observation that shows the record's content.
        content = sorted((o for o in visible if o.exists), key=lambda o: (o.observed_at, o.file_id))
        if content:
            latest_time = content[-1].observed_at
            latest_file = content[-1].file_id
            same = [o for o in content if o.file_id == latest_file and o.observed_at == latest_time] or content[-1:]
            exposed = any(o.emails & identity.emails or o.hashes & identity.email_hashes or o.phones & identity.phones
                          or o.names & identity.names or str(o.attrs.get("result", "")).upper() == "IDENTIFIABLE"
                          or _count(o.attrs.get("direct_identifiers")) > 0   # a scan reporting remaining identifiers
                          for o in same)
            rs.exposed = Fact(exposed, latest_file, latest_time)
            rs.user_ref = Fact(any(identity.user_id in o.user_refs for o in same), latest_file, latest_time)

        # Data flows and restore paths: latest explicit evidence, then the workflow's own checks.
        rs.enabled = self._latest_flag(visible, self._enabled_value)
        rs.restore_allowed = self._latest_flag(visible, self._restore_value)
        for row in sorted(rs.activity, key=lambda r: r["event_time"]):
            details = row["details"].lower()
            if row["event_type"] == "CHECK_DATA_FLOW" and rs.enabled is None:
                value = True if "remains enabled" in details else False if "disabled" in details else None
                if value is not None:
                    rs.enabled = Fact(value, row["activity_id"], row["event_time"])
            if row["event_type"] == "CHECK_RESTORE_ACCESS" and rs.restore_allowed is None:
                value = True if "can still be restored" in details else False if "blocked" in details or "cannot" in details else None
                if value is not None:
                    rs.restore_allowed = Fact(value, row["activity_id"], row["event_time"])
        return rs

    @staticmethod
    def _enabled_value(obs):
        if "enabled" in obs.attrs:
            return flag(obs.attrs["enabled"])
        if "scheduler_registered" in obs.attrs:      # still scheduled = still able to sync
            return flag(obs.attrs["scheduler_registered"])
        result = str(obs.attrs.get("result", "")).upper()
        if str(obs.attrs.get("record_type", "")).upper() == "DATA_FLOW" and result in ("ACTIVE", "DISABLED"):
            return result == "ACTIVE"
        return None

    @staticmethod
    def _restore_value(obs):
        attrs = obs.attrs
        if "restore_allowed" in attrs:
            return flag(attrs["restore_allowed"])
        if "deletion_block" in attrs:
            blocked = flag(attrs["deletion_block"])
            return None if blocked is None else (not blocked and flag(attrs.get("eligible", "true")) is not False)
        decision = str(attrs.get("decision", "")).lower()   # restore authorization outcome
        if decision in ("deny", "denied", "block", "blocked"):
            return False
        if decision in ("allow", "allowed", "approve", "approved"):
            return True
        result = str(attrs.get("result", "")).upper()
        if str(attrs.get("record_type", "")).upper() == "BACKUP_ARCHIVE" and result in ("ACTIVE", "BLOCKED"):
            return result == "ACTIVE"
        return None

    @staticmethod
    def _latest_flag(visible, getter):
        best = None
        for obs in visible:
            value = getter(obs)
            if value is not None and (best is None or (obs.observed_at, obs.file_id) > (best.at, best.evidence_id)):
                best = Fact(value, obs.file_id, obs.observed_at)
        return best

    def reconstruct(self, request, identity):
        at = request["audit_at"]
        return [
            self.state(rid, link, ev, identity, at)
            for rid, (link, ev) in sorted(self.links_for(request, identity).items())
        ]


def _count(value):
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _numeric_key(resource_id):
    prefix, _, number = resource_id.rpartition("_")
    return (prefix, int(number)) if number.isdigit() else (resource_id, 0)

"""Application database: SQLite in WAL mode, one connection per thread, a single
writer lock, and explicit schema versions.

CONCURRENCY. Session runner threads, the alert worker and HTTP handler threads all
write. Reads use the calling thread's own connection (WAL: readers never block the
writer); every write goes through `AppDatabase.write()`, which serialises writers
behind one re-entrant lock and wraps the work in a `BEGIN IMMEDIATE` transaction.
The audit chain (`anvesh.app.audit`) relies on that serialisation.

SCHEMA VERSIONS. `MIGRATIONS` is an ordered list; the database records the applied
versions in `schema_migrations`. Opening a database written by a NEWER build than
this one is refused rather than silently downgraded.

INVARIANTS ENFORCED IN THE SCHEMA (not just in services), so a bug or a hand-written
script cannot break them:
  - the audit log and evidence assets and plate readings are append-only
    (UPDATE/DELETE abort);
  - a SANDBOX registry lookup or SANDBOX challan handoff cannot attach to a case whose
    data class is `operational`; conversely a case that already carries sandbox
    records cannot be promoted to `operational`.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from pathlib import Path

SCHEMA_VERSION = 4

_V1 = """
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('administrator','investigator','reviewer','officer')),
    password_hash TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_login_at TEXT
);

CREATE TABLE auth_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0,
    remote_addr TEXT
);

CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    updated_by TEXT
);

CREATE TABLE cameras (
    camera_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    nominal_fps REAL,
    location_label TEXT,
    latitude REAL,
    longitude REAL,
    utc_offset_minutes INTEGER NOT NULL DEFAULT 330,
    data_class TEXT NOT NULL CHECK (data_class IN ('operational','demo','simulated')),
    enabled INTEGER NOT NULL DEFAULT 1,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    created_by TEXT
);

CREATE TABLE processing_sessions (
    session_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL REFERENCES cameras(camera_id),
    run_id TEXT NOT NULL,
    state TEXT NOT NULL,
    data_class TEXT NOT NULL CHECK (data_class IN ('operational','demo','simulated')),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    time_basis TEXT NOT NULL CHECK (time_basis IN ('declared_recording_start','processing_clock','live_capture')),
    recording_start_utc TEXT NOT NULL,
    source_snapshot_json TEXT NOT NULL,
    stats_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    requested_by TEXT
);
CREATE INDEX idx_sessions_camera ON processing_sessions(camera_id, started_at);

CREATE TABLE incidents (
    incident_id TEXT PRIMARY KEY,
    natural_key TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL REFERENCES processing_sessions(session_id),
    camera_id TEXT NOT NULL REFERENCES cameras(camera_id),
    event_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('detected','under_investigation','confirmed','dismissed','escalated','closed')),
    severity TEXT NOT NULL CHECK (severity IN ('info','low','medium','high','critical')),
    severity_basis TEXT NOT NULL,
    confidence_tier TEXT NOT NULL,
    confidence_basis TEXT NOT NULL,
    evidence_completeness TEXT NOT NULL,
    start_offset_s REAL NOT NULL,
    detected_offset_s REAL NOT NULL,
    end_offset_s REAL,
    start_utc TEXT NOT NULL,
    detected_utc TEXT NOT NULL,
    end_utc TEXT,
    time_basis TEXT NOT NULL,
    location_json TEXT NOT NULL DEFAULT '{}',
    track_ids_json TEXT NOT NULL DEFAULT '[]',
    involved_json TEXT NOT NULL DEFAULT '[]',
    indicators_json TEXT NOT NULL DEFAULT '{}',
    disruption_json TEXT NOT NULL DEFAULT '{}',
    candidate_causes_json TEXT NOT NULL DEFAULT '[]',
    uncertainty_json TEXT NOT NULL DEFAULT '[]',
    related_json TEXT NOT NULL DEFAULT '[]',
    detector TEXT NOT NULL,
    detector_version TEXT NOT NULL,
    detector_config_json TEXT NOT NULL DEFAULT '{}',
    motion_space TEXT NOT NULL,
    geometry_validity TEXT NOT NULL,
    speed_basis TEXT,
    data_class TEXT NOT NULL CHECK (data_class IN ('operational','demo','simulated')),
    assigned_to TEXT,
    investigation_notes TEXT NOT NULL DEFAULT '',
    established_cause TEXT,
    established_cause_by TEXT,
    alert_status TEXT NOT NULL DEFAULT 'none',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    update_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_incidents_type ON incidents(event_type, detected_utc);
CREATE INDEX idx_incidents_camera ON incidents(camera_id, detected_utc);
CREATE INDEX idx_incidents_session ON incidents(session_id);

CREATE TABLE incident_timeline (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    ts_utc TEXT NOT NULL,
    offset_s REAL,
    kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}',
    actor TEXT NOT NULL
);
CREATE INDEX idx_timeline_incident ON incident_timeline(incident_id, entry_id);

CREATE TABLE evidence_assets (
    asset_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    kind TEXT NOT NULL CHECK (kind IN ('frame','frame_annotated','clip','plate_crop','thumbnail','report')),
    relpath TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mime TEXT NOT NULL,
    offset_s REAL,
    frame_index INTEGER,
    track_id TEXT,
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL
);
CREATE INDEX idx_assets_incident ON evidence_assets(incident_id);

CREATE TABLE plate_readings (
    reading_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    track_id TEXT,
    source TEXT NOT NULL CHECK (source IN ('ocr','manual_entry','officer_verification','officer_correction','unreadable')),
    raw_text TEXT,
    normalized_text TEXT,
    format_valid INTEGER NOT NULL DEFAULT 0,
    confidence REAL,
    alternatives_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL CHECK (status IN ('unreadable','needs_review','ocr_candidate','officer_entered','officer_verified','officer_corrected')),
    engine TEXT,
    crop_asset_id TEXT,
    frame_index INTEGER,
    offset_s REAL,
    supersedes TEXT,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL
);
CREATE INDEX idx_plates_incident ON plate_readings(incident_id, created_at);

CREATE TABLE cases (
    case_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    data_class TEXT NOT NULL CHECK (data_class IN ('operational','demo','simulated')),
    state TEXT NOT NULL CHECK (state IN ('detected','awaiting_evidence_verification','awaiting_plate_verification',
        'ready_for_review','approved_for_submission','rejected','needs_further_review','submitted',
        'submission_failed','challan_reference_received','closed')),
    offence_code TEXT,
    offence_description TEXT,
    rule_reference TEXT,
    reason TEXT NOT NULL DEFAULT '',
    limitations TEXT NOT NULL DEFAULT '',
    track_id TEXT,
    evidence_verified_by TEXT,
    evidence_verified_at TEXT,
    plate_verified_by TEXT,
    plate_verified_at TEXT,
    registry_override_reason TEXT,
    approved_by TEXT,
    approved_at TEXT,
    external_reference TEXT,
    external_reference_kind TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE UNIQUE INDEX idx_cases_incident_offence ON cases(incident_id, offence_code);
CREATE INDEX idx_cases_state ON cases(state, updated_at);

CREATE TABLE case_transitions (
    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    from_state TEXT,
    to_state TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    ts TEXT NOT NULL
);
CREATE INDEX idx_transitions_case ON case_transitions(case_id, transition_id);

CREATE TABLE registry_lookups (
    lookup_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    plate TEXT NOT NULL,
    adapter TEXT NOT NULL,
    adapter_kind TEXT NOT NULL CHECK (adapter_kind IN ('sandbox','production')),
    status TEXT NOT NULL CHECK (status IN ('found','not_found','error','not_connected','denied','invalid_request')),
    is_simulated INTEGER NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    source_label TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    retrieved_at TEXT NOT NULL
);
CREATE INDEX idx_lookups_case ON registry_lookups(case_id, retrieved_at);

CREATE TABLE handoffs (
    handoff_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    adapter TEXT NOT NULL,
    adapter_kind TEXT NOT NULL CHECK (adapter_kind IN ('sandbox','export','production')),
    idempotency_key TEXT NOT NULL,
    case_version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending','succeeded','failed','exported')),
    is_simulated INTEGER NOT NULL,
    external_reference TEXT,
    response_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    artifact_relpath TEXT,
    submitted_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE UNIQUE INDEX idx_handoff_idem ON handoffs(idempotency_key) WHERE status IN ('succeeded','exported','pending');
CREATE INDEX idx_handoffs_case ON handoffs(case_id, created_at);

CREATE TABLE alert_deliveries (
    delivery_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    rule_id TEXT,
    channel TEXT NOT NULL,
    adapter_kind TEXT NOT NULL CHECK (adapter_kind IN ('test','webhook','email')),
    recipient TEXT NOT NULL,
    subject TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending','sent','failed','abandoned')),
    delivery_confirmed INTEGER NOT NULL DEFAULT 0,
    is_test INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 4,
    next_attempt_at TEXT,
    last_error TEXT,
    confirmation TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_alerts_status ON alert_deliveries(status, next_attempt_at);
CREATE INDEX idx_alerts_incident ON alert_deliveries(incident_id);

CREATE TABLE audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    actor_role TEXT,
    action TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    details_json TEXT NOT NULL,
    remote_addr TEXT,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX idx_audit_entity ON audit_log(entity_type, entity_id);

CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;

CREATE TRIGGER assets_no_update BEFORE UPDATE ON evidence_assets BEGIN SELECT RAISE(ABORT, 'evidence_assets are immutable'); END;
CREATE TRIGGER assets_no_delete BEFORE DELETE ON evidence_assets BEGIN SELECT RAISE(ABORT, 'evidence_assets are immutable'); END;

CREATE TRIGGER plates_no_update BEFORE UPDATE ON plate_readings BEGIN SELECT RAISE(ABORT, 'plate_readings are immutable; supersede with a new reading'); END;
CREATE TRIGGER plates_no_delete BEFORE DELETE ON plate_readings BEGIN SELECT RAISE(ABORT, 'plate_readings are immutable'); END;

CREATE TRIGGER transitions_no_update BEFORE UPDATE ON case_transitions BEGIN SELECT RAISE(ABORT, 'case_transitions are append-only'); END;
CREATE TRIGGER transitions_no_delete BEFORE DELETE ON case_transitions BEGIN SELECT RAISE(ABORT, 'case_transitions are append-only'); END;

CREATE TRIGGER lookups_no_update BEFORE UPDATE ON registry_lookups BEGIN SELECT RAISE(ABORT, 'registry_lookups are immutable'); END;
CREATE TRIGGER lookups_no_delete BEFORE DELETE ON registry_lookups BEGIN SELECT RAISE(ABORT, 'registry_lookups are immutable'); END;

-- A sandbox record must never attach to an operational case.
CREATE TRIGGER lookups_sandbox_isolation BEFORE INSERT ON registry_lookups
WHEN NEW.adapter_kind = 'sandbox' AND (SELECT data_class FROM cases WHERE case_id = NEW.case_id) = 'operational'
BEGIN SELECT RAISE(ABORT, 'sandbox registry lookups cannot attach to an operational case'); END;
CREATE TRIGGER handoffs_sandbox_isolation BEFORE INSERT ON handoffs
WHEN NEW.adapter_kind = 'sandbox' AND (SELECT data_class FROM cases WHERE case_id = NEW.case_id) = 'operational'
BEGIN SELECT RAISE(ABORT, 'sandbox handoffs cannot attach to an operational case'); END;
CREATE TRIGGER cases_no_promotion BEFORE UPDATE OF data_class ON cases
BEGIN SELECT RAISE(ABORT, 'a case data class cannot be changed'); END;
"""

_V2 = """
-- v2 (2026-10-20): (a) a case remembers WHICH plate reading its plate verification covered, so a plate changed after
-- verification can no longer ride on the old verification; (b) retention tombstones: evidence FILES may be removed under the
-- retention policy, but the evidence_assets row (and its SHA-256) stays and a purge record is added.
ALTER TABLE cases ADD COLUMN plate_verified_reading_id TEXT;

-- Best-effort backfill for cases verified under v1: the newest verified, unsuperseded reading for the case's incident/track.
UPDATE cases SET plate_verified_reading_id = (SELECT p.reading_id FROM plate_readings p WHERE p.incident_id = cases.incident_id AND (cases.track_id IS NULL OR p.track_id = cases.track_id) AND p.status IN ('officer_verified','officer_corrected') AND NOT EXISTS (SELECT 1 FROM plate_readings q WHERE q.supersedes = p.reading_id) ORDER BY p.created_at DESC, p.reading_id DESC LIMIT 1) WHERE plate_verified_by IS NOT NULL;

CREATE TABLE evidence_purges (
    purge_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL UNIQUE REFERENCES evidence_assets(asset_id),
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    kind TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    relpath TEXT NOT NULL,
    reason TEXT NOT NULL,
    purged_at TEXT NOT NULL,
    purged_by TEXT NOT NULL,
    file_existed INTEGER NOT NULL
);
CREATE INDEX idx_purges_incident ON evidence_purges(incident_id);

CREATE TRIGGER purges_no_update BEFORE UPDATE ON evidence_purges BEGIN SELECT RAISE(ABORT, 'evidence_purges are append-only'); END;
CREATE TRIGGER purges_no_delete BEFORE DELETE ON evidence_purges BEGIN SELECT RAISE(ABORT, 'evidence_purges are append-only'); END;
"""

_V3 = """
-- v3 (2026-10-21): registry results are bound to the plate reading they were made for, and the DATABASE refuses to move an
-- operational case into `approved_for_submission` unless an authorised (production, non-simulated) registry result exists for the
-- verified plate reading. Pre-v3 lookups keep reading_id NULL and therefore never support an operational approval.
ALTER TABLE registry_lookups ADD COLUMN reading_id TEXT;

CREATE TRIGGER cases_operational_approval_needs_registry
BEFORE UPDATE OF state ON cases
WHEN NEW.data_class = 'operational' AND NEW.state = 'approved_for_submission' AND OLD.state <> 'approved_for_submission'
  AND NOT EXISTS (SELECT 1 FROM registry_lookups l WHERE l.case_id = NEW.case_id AND l.adapter_kind = 'production' AND l.is_simulated = 0 AND l.status = 'found' AND l.reading_id IS NOT NULL AND l.reading_id = NEW.plate_verified_reading_id)
BEGIN SELECT RAISE(ABORT, 'an operational case cannot be approved without an authorised registry result for its verified plate reading'); END;
"""

_V4 = """
-- v4 (2026-10-21): account security. Privileged provisioning goes through two-person account requests; admin-provisioned and reset
-- passwords are temporary (must be changed at first login); demo accounts are flagged so they can never act on operational cases;
-- sessions record their last use so an idle timeout can be enforced.
ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN is_demo INTEGER NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN password_changed_at TEXT;
ALTER TABLE auth_sessions ADD COLUMN last_seen_at TEXT;

CREATE TABLE account_requests (
    request_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('create_user','change_role','reset_password','activate_user')),
    target_user_id TEXT,
    target_username TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL CHECK (status IN ('pending','approved','rejected','expired','failed')),
    requested_by TEXT NOT NULL,
    requested_by_user_id TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    decided_by TEXT,
    decided_at TEXT,
    decision_note TEXT,
    result TEXT
);
CREATE INDEX idx_account_requests_status ON account_requests(status, requested_at);
CREATE TRIGGER account_requests_no_delete BEFORE DELETE ON account_requests BEGIN SELECT RAISE(ABORT, 'account_requests are append-only'); END;
CREATE TRIGGER account_requests_no_reopen BEFORE UPDATE OF status ON account_requests WHEN OLD.status <> 'pending' BEGIN SELECT RAISE(ABORT, 'a decided account request cannot change'); END;
"""

MIGRATIONS = [(1, _V1), (2, _V2), (3, _V3), (4, _V4)]


class SchemaTooNew(RuntimeError):
    pass


class AppDatabase:
    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._connections: list = []
        self._conn_lock = threading.Lock()
        self._migrate()

    # -- connections -------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        with self._conn_lock:
            self._connections.append(conn)
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._connect()
            self._local.conn = c
        return c

    def close(self) -> None:
        with self._conn_lock:
            for c in self._connections:
                with contextlib.suppress(Exception):
                    c.close()
            self._connections.clear()
        self._local = threading.local()

    # -- migrations --------------------------------------------------------

    def _migrate(self) -> None:
        conn = self.conn
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        applied = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations")}
        newest = max(applied) if applied else 0
        if newest > SCHEMA_VERSION:
            raise SchemaTooNew(
                f"database {self.path} is schema version {newest}, newer than this build ({SCHEMA_VERSION}); "
                "refusing to open it"
            )
        pending = [(v, sql) for v, sql in MIGRATIONS if v not in applied]
        if pending and applied:
            self._backup_before_migration(conn, newest)
        for version, sql in pending:
            with self._write_lock:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    for statement in _split_script(sql):
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                        (version,),
                    )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise

    def _backup_before_migration(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Migrations are one-way (older builds refuse a newer schema), so a populated database is copied first.
        The copy uses SQLite's online backup API (consistent under WAL) and never overwrites an earlier backup."""
        target = self.path.with_name(f"{self.path.stem}.pre-v{from_version + 1}-migration{self.path.suffix}.bak")
        if target.exists():
            return
        dest = sqlite3.connect(str(target))
        try:
            conn.backup(dest)
        finally:
            dest.close()

    def schema_version(self) -> int:
        row = self.conn.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()
        return int(row["v"] or 0)

    # -- access ------------------------------------------------------------

    @contextlib.contextmanager
    def write(self):
        """Serialised write transaction. Nested use joins the outer transaction."""
        with self._write_lock:
            conn = self.conn
            if conn.in_transaction:
                yield conn
                return
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def query(self, sql: str, params=()) -> list:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params=()):
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def execute(self, sql: str, params=()):
        with self.write() as conn:
            return conn.execute(sql, params)


def _split_script(sql: str) -> list:
    """Split a migration script into statements, keeping CREATE TRIGGER ... END; blocks whole."""
    out, buf, in_trigger = [], [], False
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buf.append(line)
        if stripped.upper().startswith("CREATE TRIGGER"):
            in_trigger = True
        if in_trigger:
            if stripped.upper().endswith("END;"):
                out.append("\n".join(buf))
                buf, in_trigger = [], False
        elif stripped.endswith(";"):
            out.append("\n".join(buf))
            buf = []
    if buf:
        out.append("\n".join(buf))
    return out


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def loads(text, default=None):
    if text is None or text == "":
        return default
    return json.loads(text)

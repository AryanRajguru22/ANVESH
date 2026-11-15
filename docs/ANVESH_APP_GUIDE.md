# ANVESH application guide

The application layer (`anvesh/app/`) turns the perception core into an operational, auditable workflow:

```
source ─► ControlledSource ─► StreamingPipeline (reader/worker/consumer, bounded queues)
            file | webcam | rtsp | rtmp |        │ worker: YOLOv8 + ByteTrack ─► StreamingPerceptionEngine
            http_stream | synthetic              │ consumer: closed window ─► WindowReasoner
                                                 ▼
              detectors (window + frame level) ─► IncidentService ─► evidence capture (key frames, clip, hashes)
                                                       │                    │
                                                       ▼                    ▼
                                alert routing (test/webhook/email)    plates ─► registry ─► case state machine ─► hand-off
                                                       └──────── append-only, hash-chained audit log ────────┘
              HTTP API (/api/v1) + dashboard (/)  ·  SQLite (WAL) · evidence files on disk
```

## Install and launch

Uses the existing `.venv` (CPython 3.11, the pinned `requirements-lock.txt` environment). **No new dependency** is
required: the server is standard-library `http.server` + `sqlite3`. Detector weights: `models/yolov8n.pt`.

```bash
.venv\Scripts\python.exe scripts\run_app.py --demo-users
```

Opens `http://127.0.0.1:8750/`. First run creates users with **random** passwords in
`<data-dir>/initial_credentials.txt` (default `outputs/app/`, gitignored). Without `--demo-users` only `admin` is created.
Change passwords from the API (`POST /api/v1/auth/password`) and delete the file.

Environment variables (all optional): `ANVESH_DATA_DIR`, `ANVESH_HOST`, `ANVESH_PORT`, `ANVESH_MODEL`, `ANVESH_MEDIA_ROOTS`
(extra directories file sources may read), `ANVESH_SECURE_COOKIES=1` (behind HTTPS), `ANVESH_DISTINCT_APPROVER=0` (turn off
four-eyes), `ANVESH_ALERT_WEBHOOK_URL/_SECRET`, `ANVESH_SMTP_*`, `ANVESH_REGISTRY_URL/_API_KEY`, `ANVESH_CHALLAN_URL/_API_KEY`,
`ANVESH_OCR_MODEL_DIR`, `ANVESH_PLATE_MODEL`. Nothing secret is read from the repository.

Detector weights: the default is the COCO-pretrained `models/yolov8n.pt`. `ANVESH_MODEL` overrides that path; `ANVESH_FINETUNED_MODEL`
(an ANVESH fine-tuned checkpoint, see `docs/ANVESH_MODEL_TRAINING_REPORT.md`) takes precedence over both. A configured path that does not
exist is an error — ANVESH never downloads weights and never silently falls back to a different model. `GET /ready` reports which
detector (`pretrained` / `finetuned`) is configured; evidence records carry a `sha256-<12 hex>` fingerprint of the actual checkpoint file.

The server binds to 127.0.0.1. Binding elsewhere exposes an API that is authenticated but not TLS-terminated; put it behind a
reverse proxy with HTTPS first.

## Roles

| Role | Can do | Cannot do |
|---|---|---|
| administrator | cameras, users, settings, offences, alert rules, audit | approve/submit cases, see registry owner details, verify evidence |
| investigator | start/stop sessions, work incidents, create cases, enter plates, registry lookup (with owner details) | verify evidence/plates, approve |
| reviewer | verify evidence and plates, reject / request more | create cases, approve, see owner details |
| officer | approve, submit/export, close, registry lookup, audit view | verify evidence/plates (four-eyes), manage cameras |

Four-eyes (default on): the approving officer must differ from the evidence verifier, the plate verifier and the case creator; a plate
verifier must differ from whoever entered/read the plate.

## Data classes (simulation isolation)

Every camera, session, incident and case carries `data_class`: `operational`, `demo` or `simulated`. Sandbox registry lookups and
sandbox challan hand-offs are refused **by the database schema** for an `operational` case; non-operational incidents may only use the
test alert adapter; a case's class can never change. Synthetic sources are always `simulated`. Registering a camera as `operational` is
the operator's declaration that its footage is real evidential material; with no production registry/challan configured, such cases can
only be exported (manual hand-off).

## Time

Stored instants are UTC. A recorded file with no declared `recording_start_utc` has **no known wall-clock time**: incidents are stamped
with the processing clock and flagged `time_basis = processing_clock` (the video offset is authoritative); an operational case cannot be
approved on that basis. Live sources use the capture clock (`live_capture`).

## Camera configuration (`config`)

`stop_criterion` (`step_speed` default, `displacement` opt-in), `max_stream_s`, `realtime_factor`, `recording_start_utc`, `thresholds`
(`window_s`, `dwell_s`, `long_stop_s`, `max_capacity_vehicles`, `stop_radius_px`, `collision_min_iou`, `collision_min_contact_s`,
`queue_peer_stops` - how many vehicles stationary together read as a queue rather than an obstruction, default 3, untuned),
`direction_zones`, `lanes` (with `allowed_classes`), `signal` (stop line + `schedule`/`manual`/`external`), `turn_rules`,
`speed_limit_kmh`, `calibration` (reference points; metric only with a `validation` record). Invalid configs are rejected with 422. The
Cameras page shows, per camera, which event categories can run and what is missing (`GET /api/v1/capabilities?camera_id=`).

## Integrations

| Integration | State here | To connect |
|---|---|---|
| OCR | not installed | `pip install easyocr` + place its weights (ANVESH never downloads them); needs your approval |
| Plate detector | no weights | place a YOLO plate model at `models/plate_detector.pt` (licence/choice is a decision) |
| Vehicle registry | **not connected**; sandbox = synthetic fixture | departmental authorisation + a gateway implementing the contract in `anvesh/app/registry.py` |
| e-Challan | **not connected**; sandbox simulator + evidence-package export | the system is operated by MoRTH/NIC; no public API documentation was found; access is by departmental arrangement. Contract in `anvesh/app/handoff.py` is ANVESH-defined and untested |
| Alerts | test adapter (local file) | set webhook/SMTP variables; delivery is *confirmed* only on a 2xx / SMTP acceptance |

## API

Versioned under `/api/v1`; errors are `{"error": {"code", "message", "details"}}`; lists are `{items,total,limit,offset}`.
Auth: `Authorization: Bearer <token>` or the HttpOnly `anvesh_session` cookie (state changes then need `X-Requested-With: anvesh`).
Main groups: `auth`, `cameras`, `uploads`, `sessions` (start/stop/pause/resume, `stream.mjpg`, `snapshot.jpg`, `video`, `signal`),
`incidents` (+ notes, cause, plates, cases), `evidence/{id}/file|verify`, `cases` (+ actions, registry-lookup, handoff, report.html,
package), `alerts`, `overview`, `congestion`, `stats`, `audit`, `users`, `settings`, `retention` (`report`, `plan`, `purge`), `capabilities`,
`system/status`, `health`, `ready`. Error statuses worth knowing: 400 `bad_length`/`bad_json`, 411 (chunked bodies or missing length on upload), 413,
409 (state/policy conflicts, `self_lockout`, retention), 410 (`purged` / `file_missing` evidence), 422 (validation, including wrong-typed fields).

## Evidence and integrity

Per incident: unannotated key frames (the evidentiary originals), annotated frames, a thumbnail and an annotated review clip (H.264
where available; verified playable in Chrome), each written once, SHA-256-hashed and recorded in an immutable table. The frame ring is
bounded (JPEG, ≤10 fps, window + dwell + 30 s). `GET /evidence/{id}/verify` re-hashes a file. The audit log is append-only
(SQL-enforced) and hash-chained; `GET /audit/verify` re-derives the chain. This is tamper-*evident*, not tamper-proof.

## Tests

`pytest tests/app` (foundation/auth/audit, incidents, detectors incl. engine-level simulated scenarios, plates, registry, case workflow,
hand-off, alerts, ingestion/resilience, HTTP API contract, and - added 2026-10-20 - `test_security_hardening`, `test_retention`,
`test_event_logic`, `test_plates_robustness`). Streaming: `tests/streaming` (now incl. `test_synthetic_load.py`). Training tooling: `tests/training`.
Exact counts and the full-suite result are in `docs/ANVESH_EXECUTION_REPORT.md` section 9.T. Commands: `docs/ANVESH_RELEASE_READINESS.md` section 7.

## Security model and assumptions (updated 2026-10-20)

- **Authentication**: scrypt password hashes (per-user salt, 8-256 characters), 256-bit random session tokens stored only as SHA-256, 8 h sessions, revoked on
  password change / deactivation. Bearer token or HttpOnly `SameSite=Strict` cookie; cookie-authenticated state changes need `X-Requested-With: anvesh` and a
  matching `Origin`. Failed logins are throttled per (username, source), per source and per username (5 / 30 / 25 failures, 5-minute lock; in memory, bounded; unknown names are tracked identically so lockout does not reveal accounts). Sessions last 8 h and end after 30 minutes idle; a role change, deactivation or password change signs the user out.
- **Authorisation**: every route declares a permission checked first in the handler (matrix in `anvesh/app/auth.py`); the role matrix enforces separation of duties
  (administrator cannot approve or see owner details); privileged accounts need two administrators - see `docs/ANVESH_ACCOUNT_SECURITY.md` and the residual risks in `docs/ANVESH_RELEASE_READINESS.md` section 3.
- **Input**: JSON bodies <= 1 MiB, must be objects; Content-Length must be a non-negative integer; `Transfer-Encoding` is refused (411, connection closed);
  every optional field is type-checked (a non-boolean `active` is a 422, never coerced); free-text fields <= 20,000 characters; case-action bodies forward only
  `reading_id`, `allow_nonstandard`, `external_reference` (the former `registry_override_reason` is rejected with 422: there is no registry override). Idle or half-open
- **Files**: uploads must have a video extension, are size-capped (2 GiB default), opened by OpenCV before acceptance, and stored under a random name in
  `uploads/`. A recorded-file camera must be a video file inside an allowed media root (uploads, the repository, `ANVESH_MEDIA_ROOTS`) and **outside the application data directory except its `uploads/`** - this stops a camera from exposing the database, `initial_credentials.txt` or evidence. The rule is re-checked when a session starts and when a replay is served. Static files come only from `anvesh/app/static`; evidence paths are resolved against the evidence root.
- **Sensitive data**: stream credentials are redacted in every API response, audit entry and snapshot; passwords and tokens never appear in logs or audit details; owner details from the registry are shown only to roles with `registry.view_owner`.

## Retention (implemented 2026-10-20)

Evidence retention defaults to 365 days (Administration -> Retention, or `PUT /settings`). Nothing is deleted automatically.
`GET /retention/plan` (permission `retention.manage`, administrator) lists the incidents past retention that are eligible, those that are protected and why (still open, or any case not closed/rejected), and a `plan_id`.
`POST /retention/purge {"plan_id", "reason"}` removes exactly the evidence **files** in that plan (the plan must still match; a reason of >= 10 characters is required; a retention period below 30 days cannot be used to purge).
The `evidence_assets` rows and their SHA-256 stay, an append-only `evidence_purges` row records who/when/why, and the API then answers 410 `purged` for those files; incident, case, plate and audit rows are untouched. Audit expiry is not implemented. The dashboard shows the report; the plan/purge calls are API-only (no UI was added). Back up `evidence/` before the first purge.

## Schema versions and rollback

Schema v2 (this session) adds `cases.plate_verified_reading_id` and `evidence_purges`. A populated v1 database is copied to `<db>.pre-v2-migration.sqlite3.bak` before the migration runs; a v1 build refuses a v2 database. Roll back by restoring that copy with the previous build.


## Approval invariant: registry result required (2026-10-21)

An operational case cannot be approved, retried (`retry_submission`) or handed off unless **all** hold: the case's current plate reading is verified by someone other than whoever entered it; a registry lookup is **bound to that reading** (`registry_lookups.reading_id`) and says `found`; for an `operational` case the lookup came from the **configured authorised (production) adapter** and is not simulated/sandbox; it is fresh (`ANVESH_REGISTRY_MAX_AGE_S`, default 24 h), well-formed, for the same plate text, and confirmed by the append-only audit log. A manual entry, OCR confidence, an officer's note or a sandbox result never satisfies it. There is no override. A blocked attempt returns 422 (`case_rule`) with the failing items, leaves the case unchanged and is audited (`case.approve_blocked`, `case.retry_submission_blocked`, `handoff.blocked`). Enforced in `CaseService`, re-checked in `HandoffService`, and by the database trigger `cases_operational_approval_needs_registry`. Demo/simulated cases follow the same shape but accept the labelled SANDBOX result. With no authorised registry configured, operational approval stays blocked: connect one (department authorisation, `ANVESH_REGISTRY_URL`/`_API_KEY`, a gateway implementing the contract in `anvesh/app/registry.py`) - nothing else enables it.

## Accounts, demo accounts, recovery (2026-10-21)

See `docs/ANVESH_ACCOUNT_SECURITY.md`. In short: investigators/reviewers are created immediately by an administrator with a temporary password; administrators/officers need a second administrator to approve (`GET /users/requests`, `POST /users/requests/{id}/approve|reject`); `POST /users/{id}/unlock` clears this process's lockout counters; `scripts/manage_users.py --data-dir <dir>` is the audited offline recovery path. `run_app.py --demo-users` creates flagged **demo accounts** that cannot act on operational cases. Environment: `ANVESH_REGISTRY_MAX_AGE_S`, `ANVESH_SESSION_IDLE_S`.

## Retention screen

Administration -> Retention now shows the report, a **purge-plan preview** (files that would be deleted with size and hash prefix, protected incidents with reasons, files already purged) and, for `retention.manage`, an execution form that stays disabled until a reason of >= 10 characters, the exact plan id and an acknowledgement are entered. Audit records are never deleted.

## Session 6 additions (2026-10-31): live speed, calibration kind, hardened detectors, demo kit

* **Presentation kit:** `demo/README.md` (runbook and walkthrough), `scripts/run_demo.py` (disposable data dir `demo/data`, camera `demo-motorway`), `demo/camera_config.json`, `demo/SOURCE.md` (video provenance and licence status), `demo/CALIBRATION.md`.
* **Speed:** `anvesh/app/livespeed.py` estimates a per-track speed (least-squares slope of calibrated ground points over 1.6 s, with an error band, EMA-smoothed, boxes clipped by the frame edge discarded) and is the single source for the dashboard overlay, the over-speed detector and the sudden-braking detector. New camera config key `speed_zone` (image polygon) limits where speed is shown and judged. It never yields a number without a validated calibration.
* **Calibration kind:** `ValidationEvidenceKind.MARKING_STANDARD_HOLDOUT` = scale from a published road-marking module with a held-out check; reported as "APPROXIMATE metric ... site NOT surveyed"; over-speed events from it are confidence `low`. It is not `SURVEY_HOLDOUT`.
* **Over-speed** now requires a smoothed speed above `limit + 5 km/h` whose error band also clears the limit, sustained for 2 s, inside the speed zone (the previous window-mean of summed path length was inflated by box jitter on real footage). **Sudden braking** now requires a significant, quick, monotonic fall of the smoothed speed (>= 25 km/h from >= 40 km/h, >= 3.5 m/s^2 over 1.0-2.5 s); the previous 0.5 s differencing produced about 200 false events in 160 s of real footage.
* **Collision candidate** gates (each with a regression test): approach (closing distance), sudden stop within 2.5 s of first contact, not part of a stationary queue (>= `queue_peer_stops` neighbours already stopped at contact), not a whole-scene standstill (>= 60 % of the other vehicles stationary). Two vehicles stopping together is never enough. Still experimental; no real collision has been seen.
* **Live view:** progress bar, source fps, queue depth, calibration label, tracked-vehicle table (speed with +/-, estimated plate pixel size and readability), live event feed under the video. Event labels read "candidate" where the evidence is indirect (`A.title`).
* **Plates:** unchanged - no detector, no OCR engine, manual entry only. The live table states "unreadable (too small)" when the estimated plate is under 40 px (a heuristic: plate ~ 27 % of the box width).


## Session 7 additions (2026-10-31): presentation freeze

* Camera config: `thresholds.congestion_close_grace_s` (free-flow gap that ends a congestion episode; shorter gaps are merged) and an optional `tracker` block `{"conf": 0.01..<1, "config": "<repo-relative>.yaml"}` (absent = library defaults; opt-in, see `docs/ANVESH_SESSION7_REPORT.md` section 3).
* Stopped-vehicle incidents carry `indicators.queue_member`, `queue_group_id`, `queue_group_peak_stationary`; the queue reading is latched for the whole stop. Congestion incidents carry `non_free_s`, `interruptions`, `clearing_since_s`.
* The incident list shows SIMULATED and "queue member" chips; the incident page links "events with this track" (a tracklet history, not a vehicle history).
* `demo/README.md` is the 13-step presentation runbook; helpers: `demo/tools/summarize_run.py`, `ui_resolutions.py`, `restart_check.py`, `track_continuity.py`.

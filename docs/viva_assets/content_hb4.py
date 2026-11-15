from pathlib import Path
from docbuilder import *

FIGS = Path(__file__).resolve().parent / "figures"
SCR = Path(__file__).resolve().parent / "screens"
F = lambda n: FIGS / n
CODE_TEXT = 'GET  /api/v1/sessions/SES-2626B4805A\n     -> 200 {state: live, runtime: {processing_fps: 24.9, frames_dropped: 0, ...}}\nGET  /api/v1/incidents?data_class=demo&limit=50\n     -> 200 {items: [...], total: 18, limit: 50, offset: 0}\nPOST /api/v1/cameras/demo-motorway/sessions   -> 201 (session.control; audited)\nPOST /api/v1/cases/CASE-.../actions  {action: approve}\n     -> 422 {error: {code: case_rule, ...}} if the registry rule is not met'
blocks = []
A = blocks.append

# =============================================================== CH 11
A(H1("Chapter 11 - Number Plate Detection and OCR"))
A(CALLOUT("limit", "**On the 640 x 360 demo footage no plate is ever read automatically.** Since Session 8 the officer screen has a live plate reader (a YOLOv8n plate detector trained for 12 epochs on a Vietnamese plate dataset, validation mAP50 0.98 on that dataset only, plus RapidOCR) that tries a read only on vehicles at least 110 px wide; it was exercised on photographs, not CCTV. OCR on saved *incident evidence* (this chapter) is still not connected. "
          "Plates are typed in by an officer. The dashboard says so on every incident: *\"OCR is not running in this build\"*.", "Honest status"))
A(H2("11.1 The ideas"))
A(TABLE(["Term", "Meaning", "Where it would sit in ANVESH"], [
    ["**ANPR / ALPR**", "*Automatic Number Plate Recognition* = find the plate, read its characters", "the whole chain below"],
    ["**Plate detection**", "a detector (e.g. a YOLO model trained on plates) that draws a box around the plate", "`YoloPlateDetector`, expects `models/plate_detector.pt` - **no weights exist**"],
    ["**Plate crop**", "the plate box cut out of the frame, then enlarged and contrast-normalised (`preprocess_crop`: grayscale, upscale to >= 64 px high, CLAHE)", "input to OCR"],
    ["**OCR**", "*Optical Character Recognition* - turns the crop into text with a per-reading confidence", "`EasyOcrEngine` needs the `easyocr` package and its weights in `ANVESH_OCR_MODEL_DIR` - **not installed**; the app uses `UnavailableOcr`"],
    ["**Format check**", "does the text look like an Indian plate (state code + digits + letters + 4 digits, or a BH-series plate)?", "`validate_format` (structural validity only: a well-formed mark may not exist)"],
    ["**Registry verification**", "ask a vehicle database whether that plate exists and who it belongs to", "different from OCR; sandbox only (Chapter 14)"],
], [1.1, 3, 2.4], "Plate pipeline vocabulary"))
A(H2("11.2 Why preprocessing and image size matter"))
A(P("OCR needs a plate that is tens of pixels wide. In `demo.mp4` (640 x 360) the median vehicle box is **28 px** wide and a plate is about **27 %** of the box width, so roughly **8 px** - a few blobs, not letters. "
    "Even near-field cars (about 130 px wide) give about 35 px. OCR typically needs several times that (the app labels < 40 px *unreadable*, < 80 px *marginal*, >= 80 px *candidate size*). "
    "Upscaling a tiny crop adds pixels but not information, so preprocessing cannot rescue it. The live table therefore says **\"unreadable (too small)\"** for each tracked vehicle."))
A(H2("11.3 The plate workflow that does exist"))
A(TABLE(["Step", "Status value", "Who", "Notes"], [
    ["Run OCR on evidence (button)", "-", "investigator", "returns *unavailable* with reasons; never invents a reading"],
    ["Enter plate manually", "`officer_entered`", "investigator / officer", "e.g. the placeholder `MH12AB1234`; format-checked; **not** read from the footage"],
    ["Verify plate", "`officer_verified` / `officer_corrected`", "a **different** reviewer from whoever entered it", "readings are immutable rows; a correction supersedes the previous reading"],
    ["Unreadable / needs review", "`unreadable`, `needs_review`", "-", "OCR outputs can only be `ocr_candidate` (confidence >= 0.85 and valid format), `needs_review` or `unreadable` - **never verified**"],
], [1.7, 1.5, 1.4, 2.2], "Plate status flow"))
A(P("**Manual correction is not automatic recognition**, and **registry verification is not OCR**: OCR (or typing) gives a *claimed* plate text; the registry lookup asks whether that text belongs to a real vehicle. Operational approval needs the verified plate **and** a fresh authorised registry result bound to that reading."))

# =============================================================== CH 12
A(H1("Chapter 12 - Incident, Evidence and Investigation"))
A(FIG(F("fig_lifecycle.png"), 1.0, "Lifecycle from detector event to the sandbox hand-off."))
A(H2("12.1 Event versus incident, duplicates, severity"))
A(BUL(["**Event** = an in-memory `DetectedEvent` produced by a detector (event type, indicators, times, severity, key frames, uncertainty). **Incident** = the stored, reviewable row built from it by `IncidentService.upsert` (`INC-YYYYMMDD-XXXXXX`) with a timeline.",
       "**Duplicates:** each event has a `natural_key` (e.g. the stop episode id, `congestion:<camera>:<start>:<n>`, `wrong_way:<track>:<zone>`), unique per session. A stop episode is re-delivered every window; the database sees the **same** key and updates the one incident.",
       "**Severity** is set by the detector's own stated rule, shown as *severity basis* on the page: stopped vehicle low/medium/high (Chapter 9), congestion low/medium/high, wrong-way high, collision candidate high, braking and rash medium. It is a priority label, **not** a legal seriousness.",
       "**Confidence tier** (low / medium / high) is a heuristic label with a stated basis, not a probability.",
       "**Closure:** an incident's *status* moves through `detected -> under_investigation -> confirmed / dismissed / escalated -> closed` (an officer action with a note; a dismissal or closure requires a note). Separately, the *event* has an end time when the detector sees the episode finish (e.g. the stopped car moves again); this triggers an \"episode ended\" evidence frame."]))
A(H2("12.2 Evidence capture"))
A(NUM(["**Frame ring:** while the video plays, the session keeps a ring of recent frames as **JPEG (quality 85, at most 10 per second)**, sized by *seconds* (window 10 s + dwell 3 s + 30 s = 43 s). Window-level events are only detected after a window closes, so this is how the *start* of the event can still be shown.",
       "**Key frames:** each detector lists labelled offsets (e.g. \"stop begins\", \"stop confirmed\"). The ring frame nearest each offset is saved **unannotated** - the evidentiary original (`kind = frame`) - and an annotated copy with boxes of the involved tracks.",
       "**Clip:** a review clip around the event (`clip_window`, e.g. stop start - 5 s to confirmed + 5 s) is encoded from ring frames at 10 fps, **H.264 (avc1)** first, then VP8/VP9/mp4v as fallbacks if the encoder is missing (the harmless `libopenh264` message). It is checked to be readable.",
       "**Track association:** each asset records the `track_id` and the box at that moment; the incident lists the involved tracks as role `observed` or `potentially_involved` - never as established cause.",
       "**Hashes:** each file is written with exclusive create, `fsync`ed, SHA-256 hashed and inserted in `evidence_assets` (an immutable table; UPDATE/DELETE rejected by SQL triggers). An `evidence.add` entry with the hash goes into the audit chain.",
       "**Completeness:** `set_evidence_completeness` grades the incident (`full` / partial / none) from the assets present."]))
A(H2("12.3 Verification, tampering, purge, audit"))
A(FIG(F("fig_chain.png"), 1.0, "Evidence hashes and the audit hash chain."))
A(TABLE(["Situation", "What ANVESH does"], [
    ["Verify an asset (`GET /evidence/{id}/verify`; reviewer's **Verify evidence**)", "re-reads the file, recomputes SHA-256, compares with the recorded value -> `ok` or `sha256 mismatch` / `file missing`"],
    ["The file is modified on disk", "verification **fails**; case reports/exports embed only files whose hash matches (`read_verified`); the case's evidence step cannot pass. Note: plain *viewing* of a file does not re-hash it - verification is an explicit action."],
    ["Evidence purged under retention", "only the **file** is deleted; the `evidence_assets` row and its SHA-256 stay, an append-only `evidence_purges` row records who/when/why, the API answers **410 purged**; incident, case, plate and audit rows are untouched"],
    ["Audit entries kept", "every login, session start/stop, evidence add, case action, lookup, hand-off, purge and chain verification; owner PII and secrets are never written"],
], [2.4, 4], "Integrity behaviour"))
A(P("**What does a tampered file look like?** Change one byte of a stored JPEG: its SHA-256 changes completely, so the stored hash no longer matches and *Verify* fails. Changing the *stored hash* too is blocked by the immutable table; changing the audit row that recorded it breaks the chain from that row onward."))
A(P("**Tamper-evident, not tamper-proof:** someone with write access to both the database file and the code could rewrite the whole chain. The production mitigation (publishing the head hash outside the database) is not implemented."))

# =============================================================== CH 13
A(H1("Chapter 13 - Database, Backend and Frontend"))
A(H2("13.1 Stack"))
A(TABLE(["Layer", "What is actually used"], [
    ["Backend language / framework", "Python 3.11, **no web framework**: `http.server.ThreadingHTTPServer` (one thread per request, daemon threads, per-request timeout against slow-loris) + hand-written router in `anvesh/app/routes.py`"],
    ["Frontend", "static **HTML + vanilla JavaScript + CSS** (`anvesh/app/static/index.html`, `app.js`, `views_ops.js`, `views_cases.js`, `views_admin.js`); hash-based routing (`#/overview`, `#/live/<session>` ...); no React, no build step"],
    ["Database", "**SQLite** via `sqlite3`, WAL journal, `synchronous=NORMAL`, foreign keys on, 30 s busy timeout, one connection per thread, a single serialised writer; schema version 4 with migrations and a backup before migrating"],
    ["API", "REST-style JSON under `/api/v1`; errors as `{\"error\": {\"code\", \"message\", \"details\"}}`; lists as `{items, total, limit, offset}`"],
    ["Live video", "`GET /api/v1/sessions/{id}/stream.mjpg` - **MJPEG** (a stream of JPEG images) shown in a normal `<img>` tag"],
    ["Live data refresh", "**polling**: `A.poll(fn, ms)` with `setTimeout` - the live page asks `/sessions/{id}` and `/incidents` every **2.5 s**; overview/camera lists every **4 s**. **No WebSockets and no Server-Sent Events.**"],
    ["Evidence", "`GET /api/v1/evidence/{asset_id}/file` (permission `view.evidence`; 410 if purged/missing); the browser plays the H.264 clip in a `<video>` tag"],
], [1.5, 5], "Implementation facts"))
A(H2("13.2 Main pages (navigation)"))
A(P("Overview | Cameras | Live monitoring | Incidents | Violations / e-challan | Congestion | History & audit | Administration. Which links appear depends on the user's permissions. A banner on every page says *RESEARCH / DEMO USE ONLY*; status chips show **OCR: unavailable, Registry: not connected, e-Challan: not connected, Alerts: test adapter only, GPU**."))
A(FIG(SCR / "live.jpg", 0.78, "Live monitoring page (real demo video, fresh run): annotated video, telemetry, tracked-vehicle table. Screenshot from demo/runs/walkthrough_freeze."))
A(H2("13.3 REST in simple terms"))
A(P("**REST** (*Representational State Transfer*) means the browser asks for named resources with ordinary HTTP verbs. A request is a verb plus a URL; the reply is a status code plus JSON."))
A(CODE(CODE_TEXT))
A(P("Status codes worth knowing: 401 not signed in, 403 no permission (or demo account on operational data, or password change required), 409 conflict (state/policy), 410 evidence purged/missing, 422 validation or case rule, 429 too many login attempts."))
A(H2("13.4 How the dashboard updates during playback"))
A(NUM(["The session thread processes frames and keeps the latest annotated JPEG in memory.", "The browser's `<img src=.../stream.mjpg>` is a long-lived response; the server keeps sending the newest annotated JPEG (a new preview frame is rendered at most every 0.1 s).",
       "Every 2.5 s the page calls `GET /sessions/{id}`; the reply carries the runtime numbers (fps, queue depth, latency, tracked vehicles, congestion trend). The same tick lists the session's incidents for the live event feed.",
       "When an incident appears the feed shows it with its thumbnail; clicking opens `#/incidents/<id>`, which fetches the incident, evidence list and timeline."]))
A(H2("13.5 What the database stores"))
A(TABLE(["Table", "Holds"], [
    ["`users`, `auth_sessions`, `account_requests`", "accounts (scrypt hash, role, demo flag, must-change-password), hashed session tokens, two-person account requests"],
    ["`cameras`, `processing_sessions`", "camera config (JSON), session state, final statistics (`stats_json`, e.g. 5,552 unique tracks)"],
    ["`incidents`, `incident_timeline`", "incidents with indicators, uncertainty, track ids, candidate causes; step-by-step timeline"],
    ["`evidence_assets`, `evidence_purges`", "immutable evidence rows with SHA-256; retention tombstones"],
    ["`plate_readings`", "immutable reading history (superseded, never edited)"],
    ["`cases`, `case_transitions`", "violation cases and every transition (append-only)"],
    ["`registry_lookups`, `handoffs`", "immutable lookup results (sandbox flag) and hand-off records"],
    ["`alert_deliveries`, `settings`, `audit_log`", "alert attempts, configuration, the hash-chained audit log"],
], [2.2, 4.1], "Tables (anvesh/app/db.py)"))
A(P("Database **triggers** enforce rules even if code is bypassed: audit/evidence/plate/transition/lookup rows cannot be updated or deleted; sandbox lookups and hand-offs are refused for an `operational` case; a case's data class cannot be promoted; an operational case cannot reach approval without a registry result."))
A(CALLOUT("def", "**If the server restarts:** SQLite keeps all incidents, cases, evidence rows and audit entries (verified in Session 7: 46 incidents and 2 cases unchanged; audit chain intact). A session that was running is marked `interrupted`, not live; a stopped session stays stopped. The live video and in-memory login throttle counters do not survive.", "Persistence"))

# =============================================================== CH 14
A(H1("Chapter 14 - Registry and Challan Workflow"))
A(CALLOUT("limit", "There is **no real government registry integration** and **no real e-challan submission**. The sandbox registry is a synthetic fixture (`configs/sandbox_registry.json`); the sandbox challan is a simulator producing `TEST-CHALLAN-...` references. "
          "An operational case cannot be approved without a fresh result from an *authorised* registry adapter, which is not configured.", "Read first"))
A(H2("14.1 The officer workflow, button by button"))
A(TABLE(["#", "Who", "Where / button (exact label)", "Resulting state"], [
    ["1", "investigator", "Incidents -> filter *all data classes* = `demo` -> open a stopped-vehicle incident -> **Open violation case from this incident**", "case `detected`"],
    ["2", "investigator", "Violations / e-challan -> the case -> **Submit for evidence verification**", "`awaiting_evidence_verification`"],
    ["3", "reviewer", "**Verify evidence** (hashes re-checked; needs frames + clip)", "`awaiting_plate_verification`"],
    ["4", "investigator", "incident page -> **Enter plate manually** (`MH12AB1234`, DEMO PLACEHOLDER) -> Save", "reading `officer_entered`"],
    ["5", "reviewer2", "**Verify plate** (a different person from step 4)", "`ready_for_review`"],
    ["6", "officer", "**Look up registration (sandbox)**", "lookup row, labelled SANDBOX / synthetic"],
    ["7", "officer", "**Approve for submission** (four-eyes: different from the evidence verifier, plate verifier and case creator)", "`approved_for_submission`"],
    ["8", "officer", "**Submit to SANDBOX simulator**", "`submitted` -> `challan_reference_received` (`TEST-CHALLAN-...`)"],
    ["9", "administrator", "History & audit -> Audit log -> **Verify hash chain**", "toast \"Audit chain intact: N entries\""],
], [0.3, 1, 3.7, 1.7], "Workflow from incident to sandbox challan"))
A(FIG(F("fig_case.png"), 1.0, "Case state machine and the users involved."))
A(H2("14.2 Registry rules"))
A(BUL(["A registry lookup needs an **officer-verified** plate reading (never a raw OCR candidate). If the registry is unavailable the result is `not_connected` / `error` / `denied` - **never a guessed result**.",
       "For an **operational** case: the lookup must come from the configured authorised adapter, be for the same plate text, be fresh (`ANVESH_REGISTRY_MAX_AGE_S`, default 24 h), well-formed, and confirmed in the audit log. A manual entry, OCR confidence, an officer note or a sandbox result **never** satisfies it.",
       "**There is no override.** An attempt to pass `registry_override_reason` is rejected with 422. The rule is enforced in `CaseService`, re-checked in `HandoffService`, and by a database trigger.",
       "For demo/simulated cases the same shape applies but the labelled SANDBOX result is accepted. Demo accounts cannot write operational data at all (403).",
       "Owner name/address/contact fields come back only to roles with `registry.view_owner` (investigator, officer); the reviewer and administrator never see them."]))
A(FIG(SCR / "handoff.jpg", 0.8, "Case page after the sandbox hand-off (demo/runs/walkthrough_freeze/13_handoff.png)."))

# =============================================================== CH 15
A(H1("Chapter 15 - Security, Accounts and Retention"))
A(H2("15.1 Roles and permissions (`anvesh/app/auth.py`)"))
A(TABLE(["Permission", "administrator", "investigator", "reviewer", "officer"], [
    ["view overview / cameras / incidents / cases / evidence / congestion", "yes", "yes", "yes", "yes"],
    ["manage cameras, users, settings, alert rules, retention", "**yes**", "no", "no", "no"],
    ["start / stop / pause sessions", "yes", "yes", "no", "no"],
    ["update incidents, record established cause", "no", "yes", "update only", "yes"],
    ["create a case; enter plates", "no", "yes", "plates only", "plates only"],
    ["verify evidence / verify plate", "no", "no", "**yes**", "no"],
    ["approve / submit / close a case", "no", "no", "no", "**yes**"],
    ["registry look-up; see owner details", "no", "yes + owner", "look-up only", "yes + owner"],
    ["view audit log; verify audit chain", "yes + verify", "no", "no", "view only"],
], [3.2, 1, 1, 1, 1], "Role matrix (separation of duties)"))
A(H2("15.2 Authentication and sessions"))
A(BUL(["**Password hashing: scrypt** from `hashlib` with N = 2^17, r = 8, p = 1 (about 128 MiB and 0.4 s per hash), a 16-byte random salt; the parameters are stored in the hash string, weaker old hashes are upgraded at the next successful login, and an absurdly large stored cost is rejected.",
       "**Password policy:** 10-256 characters (the code constant `MIN_PASSWORD_LENGTH` is 10), a short deny-list of common passwords, not the username, different from the current one.",
       "**Temporary passwords:** an administrator-created investigator/reviewer gets a temporary password; until it is changed the API answers 403 `password_change_required` to everything except `/auth/me`, `/auth/password`, `/auth/logout`, `/health`, `/ready`.",
       "**Login throttling:** failures are counted per (username, source address) = 5, per source = 30, per username = 25; each lock lasts 5 minutes; unknown names are treated the same so lockout does not reveal accounts. Counters live in server memory (lost on restart).",
       "**Sessions:** 256-bit random token, only its SHA-256 stored; **8 h** lifetime, **30 min** idle timeout; HttpOnly + `SameSite=Strict` cookie; state-changing requests also need `X-Requested-With: anvesh` and a matching `Origin`. A role change, deactivation or password change **revokes** the user's sessions.",
       "**Two-person rule for privileged accounts:** investigators/reviewers are created immediately; **officer** and **administrator** accounts need a request by one administrator approved by a *different* one (72 h expiry). Offline break-glass: `scripts/manage_users.py --data-dir <dir>` (audited as `cli:<os user>`).",
       "**Demo-account isolation:** accounts created by `--demo-users` are flagged `is_demo`, labelled \"(DEMO ACCOUNT)\", and cannot write to operational data; demo status cannot be laundered through account creation."]))
A(CALLOUT("limit", "**MFA (multi-factor authentication) is not implemented** - deliberately, with no fake substitute. Reason (account-security doc): a TOTP secret needs authenticated encryption with a key outside the database, and no vetted cryptography library was approved for the pinned environment. A design exists on paper. Other open items: HTTP on localhost only (TLS proxy needed), throttling per process, no independent security review.", "Open item"))
A(H2("15.3 Retention and reviewed purge"))
A(BUL(["Default evidence retention **365 days**; **nothing is deleted automatically**.",
       "`GET /retention/plan` (administrator) lists incidents past retention that are eligible, those **protected** (still open, or with any case not closed/rejected) and a `plan_id`.",
       "`POST /retention/purge {plan_id, reason}` removes exactly the files in that plan; the plan must still match; a reason of >= 10 characters is required; a retention period below 30 days cannot be used to purge; the Retention screen needs the exact plan id and an acknowledgement.",
       "Hashes, tombstones and audit rows remain; audit expiry is not implemented. Back up `evidence/` before the first purge."]))

# =============================================================== CH 16
A(H1("Chapter 16 - Performance, Testing and Results"))
A(H2("16.1 Kinds of tests and why they matter"))
A(TABLE(["Kind", "What it checks", "Example in ANVESH"], [
    ["**Unit tests**", "one function or class in isolation, fast, many", "`tests/app/test_live_speed.py` (11): regression slope, clipped boxes, no calibration -> no number"],
    ["**Integration tests**", "several modules together", "`tests/app/test_session_synthetic.py`: scripted tracks -> incident -> evidence -> case"],
    ["**End-to-end tests**", "the whole system through its public interface", "`demo/tools/officer_workflow.py` (API) and `presenter_walkthrough.py` (browser)"],
    ["**Browser tests**", "the real UI in headless Chrome (Playwright)", "`ui_acceptance.py`, `ui_resolutions.py`"],
    ["**Synthetic scenarios**", "scripted boxes for events real footage does not contain", "sim-collision, sim-wrong-way, sim-queue"],
    ["**Full-video acceptance**", "a complete real run with measured performance", "the 20-minute run, `demo/runs/perf_freeze.csv`"],
], [1.2, 2.4, 2.9], "Test types"))
A(H2("16.2 Verified Session 7 results"))
A(TABLE(["Check", "Result", "What it covers"], [
    ["Full test suite (9 sequential pytest processes)", "**1,381 passed, 0 failed** (was 1,350; +31 new)", "app, streaming, perception, world, safety, storage, evidence, hypotheses, propagation, top-level tests"],
    ["Presenter click-through (1920 x 1080, no API shortcuts)", "**23 / 23** steps", "fresh start, real video, SIMULATED collision/wrong-way/queue, case to sandbox challan, audit, hash chain, no JS errors"],
    ["Officer workflow (API)", "**15 / 15**", "evidence re-hashes, manual plate, OCR unavailable, override -> 422, SANDBOX label, official hand-off 409, demo account 403, audit chain (407 rows)"],
    ["Browser acceptance on a live session", "**12 / 12**", "stream decodes 640x360, progress/fps/queue/calibration notice, evidence images, clip plays, banner, no overflow, no JS errors"],
    ["Page layout", "**36 / 36**", "9 pages x 4 resolutions (1920x1080, 1366x768, 1280x720, 1024x768)"],
    ["Restart persistence", "verified", "46 incidents and 2 cases unchanged; audit chain intact"],
    ["Full 20-minute run", "30,001 frames, 0 dropped, **24.89 FPS**", "Chapter 4"],
], [2.4, 1.8, 2.4], "Verification summary"))
A(CALLOUT("limit", "**What these results do not show:** detection or tracking **accuracy** on the demo video (no labels); that the detectors are right on real violations (none occurred); behaviour on a real CCTV stream (RTSP only tested with fake captures); throughput beyond 25 fps; multi-hour stability; security under attack. "
          "Passing tests prove the code does what its tests say, and the workflow rules hold - not that the real world matches the assumptions.", "Limits"))

# =============================================================== CH 17
A(H1("Chapter 17 - Technologies Used"))
A(P("Taken from `pyproject.toml`, `requirements-lock.txt`, the `.venv` package list and the imports in `anvesh/`. Technologies that are installed on the development PC but not used by ANVESH (FastAPI, Flask, uvicorn, pydantic, argon2, cryptography) are deliberately omitted."))
A(TABLE(["Technology", "Where used", "Why used", "What it does in ANVESH"], [
    ["Python 3.11.9", "everything", "readable, rich ML ecosystem", "all application and perception logic"],
    ["Ultralytics 8.4.132 (YOLOv8, ByteTrack)", "`perception/tracking.py`, `detection.py`", "ready-made fast detector + tracker; AGPL-3.0 licence question open", "`model.track()` = detection + association + raw IDs"],
    ["PyTorch 2.13.0+cu130, CUDA", "inside Ultralytics", "GPU inference", "runs YOLOv8n on the RTX 3050 Laptop GPU"],
    ["OpenCV 5.0.0.93", "`ingestion.py`, `evidence.py`, `runner.py`, `calibration.py`", "decoding, drawing, encoding", "`VideoCapture`, `imencode`, `VideoWriter` (H.264), `findHomography`, CLAHE"],
    ["NumPy 2.4.6", "calibration, plates", "array maths", "3 x 3 homography, matrix-vector products"],
    ["lap 0.5.13", "inside ByteTrack matching", "fast linear assignment", "`lapjv` one-to-one track/detection assignment"],
    ["SQLite (`sqlite3`, WAL)", "`app/db.py`", "zero-install, transactional, single file", "persistence of all records; triggers enforce rules"],
    ["`http.server.ThreadingHTTPServer`", "`app/server.py`", "no extra dependency", "REST API, MJPEG, static files"],
    ["`hashlib` SHA-256 / scrypt", "`evidence.py`, `audit.py`, `auth.py`", "standard, audited primitives", "evidence hashes, audit chain, password hashing"],
    ["Vanilla JavaScript, HTML, CSS", "`app/static/`", "simple, no build step", "dashboard SPA with polling and MJPEG"],
    ["PyYAML 6.0.3, requests 2.34.2, polars 1.44.1, matplotlib 3.11.1, Pillow 12.3.0, psutil 7.2.2", "tooling and evaluation scripts", "supporting libraries", "config files, evaluation tables/plots, memory monitor"],
    ["pytest 9.1.1, Playwright (system Python)", "tests, `demo/tools/`", "automation", "unit / browser acceptance checks"],
    ["Dempster-Shafer fusion (own code)", "`fusion/ds_fusion.py`, `hypotheses/`", "combine uncertain evidence", "belief/plausibility ranking of H1-H6"],
    ["Kinematic shock-wave formula (own code)", "`propagation/shockwave.py`", "classic traffic-flow theory", "implemented, not used in the single-camera demo"],
], [1.7, 1.7, 1.5, 2], "Technology inventory (versions from the project .venv)"))

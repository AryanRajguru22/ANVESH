# ANVESH — acceptance report (operational MVP)

Date: 2026-10-18 · Environment: Windows 11, CPython 3.11.9 `.venv` (pinned `requirements-lock.txt`), RTX 3050 6 GB laptop GPU.
Baseline when work began: `main` @ `b6d5ecd` plus the uncommitted M-A hardening changes (kept untouched); 915 tests passing.
Nothing has been committed or pushed.

## 1. What genuinely works end to end

Demonstrated by `scripts/demo_e2e.py` (through the same HTTP API the dashboard uses, as four different users) and re-verified after a server restart:

1. A real recorded video (`demo.mp4`, 120 s processed) → the **real YOLOv8n + ByteTrack** engine → stop episodes and congestion state → **14 persisted incidents** (13 `stopped_vehicle`, 1 `congestion`), each with unannotated key frames, annotated frames, thumbnail and an annotated H.264 clip, all SHA-256-hashed.
2. Incident detail in the dashboard: evidence viewer, timeline, uncertainty, candidate vehicles (role `observed`), explicit "no established cause".
3. Investigator opens a case → reviewer verifies evidence → plate entered (placeholder, see §7) and verified by a different reviewer → officer registry lookup (**sandbox**) → approval (four-eyes enforced) → **sandbox** hand-off returning a `TEST-CHALLAN-…` reference → evidence package (zip + manifest) and printable report.
4. After stopping and restarting the server: the case (7 transitions), 8 evidence assets (hashes intact) and the audit chain (176 entries) are all present and valid (`--verify-only`).
5. Live annotated view (MJPEG) in Chrome: boxes, track ids, classes, image-space heading arrows, telemetry; evidence clip playback verified in Chrome (H.264 and VP9).

## 2. Install and launch (from `D:\ANVESH`)

```
.venv\Scripts\python.exe scripts\run_app.py --demo-users     # http://127.0.0.1:8750/ ; passwords in outputs\app\initial_credentials.txt
.venv\Scripts\python.exe scripts\demo_e2e.py                 # reproduce the demo (server must be running)
```
No new dependency (standard-library server). Details: `docs/ANVESH_APP_GUIDE.md`. A fresh-machine install was **not** re-run (the lock file's own caveats apply).

## 3. Sources actually tested

| Source | Tested how | Real-world validation |
|---|---|---|
| Recorded files (`demo.mp4` 640×360, `traffic.mp4` 1280×720, uploads) | real inference; upload → camera → session over HTTP | yes, on those files |
| HTTP MJPEG stream (`http_stream` adapter) | local test server, decoded by OpenCV/FFmpeg | **not** a camera |
| RTSP / RTMP / webcam | injected fake captures: down at start → reconnect loop, mid-stream outage (new identity session, lost frames counted), stall detection, retry exhaustion (one camera fails, another unaffected), bounded queues with counted drops | **none** — no real RTSP/RTMP server, camera or webcam was used |
| Pause/resume (files only), stop | tested; "jump to incident" in source-video replay verified in Chrome on `demo.mp4` | — |
| Webcam | adapter code only (DirectShow capture factory); **never opened or exercised** | none |
| Missing detector weights | session ends `failed` with a clear message; **no download is attempted** (tested) | — |

## 4. Event categories (full table in the dashboard: Administration → Event capabilities)

| Category | State | Real footage |
|---|---|---|
| Stopped vehicle / long stop | operational (image-space; displacement criterion is opt-in per camera) | yes — `demo.mp4`: 13 stops with the displacement rule, **0 with the default per-step rule** (the audit's known fragility); `traffic.mp4` 90 s: 27. Many are queue stops; precision is **unmeasured** |
| Congestion | operational (count/occupancy proxy in image space) | yes — free-flow → building → congested → dissipating on `demo.mp4` |
| Possible collision (candidate) | experimental | **no collision in any available footage**. 0 candidates in 120 s `demo.mp4` and 90 s dense `traffic.mp4` (not proof of low false-positive rate); detects the scripted simulated collision |
| Wrong-way, wrong-lane, rash driving (lane changes), red-light, illegal turn | implemented, need configured zones/lanes/signal/turn rules | unit-tested + simulated scenarios only; not demonstrated on real footage |
| Overspeed, sudden braking | implemented; need **validated** metric geometry | tested incl. a regression test that a fitted-but-unvalidated homography yields **no** speed events; only a SYNTHETIC-validated fixture fires |
| Helmet, seatbelt | **no detector** (needs a trained classifier) | shown as unavailable |

Red-light uses a configured schedule / manual / external signal state — the signal colour is **not** read from video.

## 5. Evidence and reports
Key frames (originals + annotated), thumbnail, annotated clip, hashes, immutable table, `/evidence/{id}/verify`; incident JSON report; case printable HTML and zip package with a manifest. The frame ring is bounded (JPEG ≤10 fps, window+dwell+30 s).

## 6. ANPR / OCR
**Does not run.** No OCR engine or plate-detector weights are installed and the footage has no legible plates. Built and tested: format validation/normalisation, an EasyOCR adapter (downloads disabled), a YOLO plate-detector hook (`models/plate_detector.pt`), status rules (OCR is never "verified"), officer entry with distinct provenance, corrections that preserve the original, four-eyes verification. Needs your decision: installing EasyOCR and its weights, and choosing/licensing a plate detector.

## 7. Registry and e-challan
- **No authorised registry is connected.** Sandbox = synthetic fixture (`configs/sandbox_registry.json`), labelled everywhere; the schema refuses it on operational cases. The production adapter is an unverified gateway contract.
- **No official e-challan integration exists or was tested.** The system is operated by MoRTH/NIC (Vahan/Sarathi integrated); no public API documentation was found ([NIC e-Challan news](https://informatics.nic.in/news/1288/), [NIC e-Gov products](https://informatics.nic.in/files/websites/april-2025/echallan.php)), so no endpoint was invented. Supported routes: evidence-package export (manual submission + officer-recorded reference) and the labelled sandbox simulator.
- In the demo, the plate `MH12AB1234` is a **placeholder typed to exercise the workflow**, not read from the footage.

## 8. Dashboard functions that work
Login/roles; overview; camera add/edit/upload/start/stop/pause/capabilities; live annotated view + telemetry + source replay; incident centre (filter, detail, status, notes, assign, officer-recorded cause, alerts, plates, case creation); case centre (checklist, actions, registry lookup, hand-off, report/package, history, audit); congestion trends; history/stats/audit with chain verification; administration (integrations status, users, thresholds, offences, alert rules, retention report). Every screen reads real API data; unavailable capabilities are shown as such.

## 9. Tests
`pytest` on the final tree: **1044 passed**, 0 failed, 0 skipped (915 legacy + 129 app tests across foundation/auth/audit, incidents, detectors, plates, registry, cases, hand-off, alerts, ingestion, HTTP contract). Finished sessions release their frame ring, history, preview and detector (tested), and at most 30 terminal sessions stay in memory. A real server bug found during browser testing (unread POST bodies corrupting keep-alive connections) is fixed with a regression test.

## 10. Performance (this machine, recorded files, accelerated — not paced real-time)
- `demo.mp4` 640×360: 38–46 fps processing, detect+track p50 19 ms / p95 27 ms, 0 dropped frames, including evidence capture.
- `traffic.mp4` 1280×720: ~21.7 fps (below its 25 fps source rate) with evidence capture; `frame_end_to_end` p50 275 ms includes queueing in file mode and is not a live-latency figure.
- Real-time operation on live cameras is **not** demonstrated; GPU/CPU: CUDA used here; a CPU-only run was not measured.

## 11. Known limitations (security, privacy, accuracy)
- Image-space only: no calibrated camera exists in this repo; no km/h, no lane-level congestion without configured lanes.
- Stop/queue discrimination, collision, wrong-way etc. have no measured precision/recall.
- Track ids are session-scoped; a tracker break can split one vehicle.
- No vehicle-counting line (concurrent and distinct-track counts only).
- **No map view**: camera coordinates are shown as text only.
- Recorded files without a declared start time have processing-clock times; operational approval is blocked on that.
- Retention is configured and reported; **no purge** is implemented (evidence/audit are immutable by design).
- Audit chain is tamper-evident, not tamper-proof (head hash is exposed for external anchoring).
- Server is HTTP on localhost; TLS must be provided by a proxy for any network exposure. Login lockout state is held in memory only (lost on restart); there is no MFA.
- Ultralytics is AGPL-3.0 (licensing is an open decision from earlier audits).
- OpenCV prints harmless FFmpeg `libopenh264` messages when it falls back to the Windows H.264 encoder.

## 12. Files
New: `anvesh/app/` (config, db, audit, auth, cameras, capabilities, detectors, synthetic, evidence, incidents, runner, alerts, plates, registry, cases, handoff, reports, settings, services, bootstrap, server, routes, static/{index.html,app.css,app.js,views_*.js,boot.js}), `tests/app/*`, `configs/sandbox_registry.json`, `scripts/run_app.py`, `scripts/demo_e2e.py`, `docs/ANVESH_APP_GUIDE.md`, this report. Modified by this work: `README.md`, `pyproject.toml` (package-data). The pre-existing uncommitted M-A modifications and untracked data/audits were left untouched. Generated, gitignored: `outputs/app/`, `outputs/demo/`.

## 12a. Git status (nothing staged, committed or pushed)
`git status --short`: 70 entries (62 before this work). Of the 31 untracked entries, the new ones from this work are `anvesh/app/`, `tests/app/`, `configs/sandbox_registry.json`, `scripts/run_app.py`, `scripts/demo_e2e.py`, `docs/ANVESH_APP_GUIDE.md`, `docs/ANVESH_ACCEPTANCE_REPORT.md`; the rest were already untracked. Modified files are the pre-existing M-A set plus `README.md` and `pyproject.toml`.

## 13. Needs from you
OCR/plate-model approval; departmental authorisation and API details for the registry and e-challan; real camera/RTSP access to validate live ingestion; a per-camera calibration with survey hold-out evidence for any speed claim; the rule references for the offence catalogue; webhook/SMTP details for real alerts; a licensing decision on Ultralytics; authorisation before any commit/push.


## 14. Addendum 2026-10-20 (fourth session) - read this before relying on sections 1-13

Sections 1-13 are the 2026-10-18 record and are unchanged. What changed since, verified in `docs/ANVESH_EXECUTION_REPORT.md` section 9:

- **Corrections to section 11 ("Known limitations")**: retention now has a guarded purge of evidence *files* (plan -> reviewed `plan_id` -> purge; rows, hashes and the audit trail are kept; audit expiry is still not implemented). The test count in section 9 is superseded by execution report 9.T.
- **Security defects found and fixed** (each with a regression test): negative/non-numeric Content-Length thread hang, chunked-body smuggling, no connection timeout, **a file camera could expose the database and credentials file through the replay endpoint**, unvalidated camera updates, 500s on wrong-typed JSON, truthiness coercion of `active`, plate verification not following a later plate correction, export packages that could carry altered evidence. One open policy question (registry override on operational cases) and two organisational risks (admin-created accounts, per-username lockout) are listed in `docs/ANVESH_RELEASE_READINESS.md` section 3.
- **Schema v2**: `cases.plate_verified_reading_id`, `evidence_purges`. A populated v1 database is backed up before migrating; v1 builds refuse v2.
- **Event logic**: a stationary-group rule lowers the severity of stops made by >= 3 vehicles together (queue vs obstruction); `queue_peer_stops` is untuned. Section 4's "Many are queue stops; precision is unmeasured" still stands.
- **Streaming**: corrupt live frames are skipped and counted instead of ending the session; a recorded file that ends short of its container frame count is flagged. Still **no real RTSP / RTMP / webcam validation**; the dashboard still polls (+ MJPEG), there is no WebSocket/SSE.
- **Still true and still blocking**: no OCR engine, no plate model, no registry, no e-challan, no map, no real camera, no local-footage benchmark, fine-tuned detector not promoted, licence decisions open.
- Demo re-run on a fresh data directory: Part A/B passed and persistence after restart verified (details in execution report 9.D). `outputs/app` was not touched.

## 15. Addendum 2026-10-21 (fifth session)

- **Registry approval policy (project-owner decision): there is no registry override.** An operational case cannot be approved, retried or handed off without a verified plate and a fresh, audit-confirmed result from the configured authorised (non-sandbox) registry bound to exactly that reading; a database trigger repeats the rule. Because no authorised registry exists here, **operational approval is blocked by design**. Demo/simulated cases still work with the labelled SANDBOX result (section 1 item 3 now includes a sandbox lookup bound to the verified plate before approval). Sections 1-14 describing an override are historical.
- **Accounts:** administrator/officer provisioning needs a second administrator (or the audited offline CLI); temporary passwords; scrypt N=2^17; per-(user, source) throttling; 30-minute idle timeout; flagged demo accounts that cannot act on operational cases. **MFA is not implemented.** Throttling is in memory only. See `docs/ANVESH_ACCOUNT_SECURITY.md`.
- **Retention screen** added (plan preview, protected records, gated purge); not driven in a browser this session.
- **Product status** is stated in the API, the dashboard banner and the start-up output: research/demo only. Licensing (Ultralytics, MIO-TCD, OCR weights) is undecided.
- **Tests:** all test directories ran this session in sequential groups: 1324 passed, 0 failed. Demo re-run on a fresh data directory: passed; persistence after restart verified. Details: execution report section 10.

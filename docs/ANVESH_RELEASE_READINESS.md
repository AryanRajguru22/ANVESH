# ANVESH release readiness (acceptance matrix, blockers, licences, authorisations)

Date: 2026-10-20 (fourth session). Companion to `docs/ANVESH_ACCEPTANCE_REPORT.md` (what works) and `docs/ANVESH_EXECUTION_REPORT.md` section 9 (what was done and measured).
Statuses: **VERIFIED** = exercised by a test or a run in this repository and the evidence is cited; **PARTIAL** = implemented, evidence is limited (say how);
**BLOCKED-FOOTAGE** = needs your local CCTV footage; **BLOCKED-EXTERNAL** = needs data, a licence, a credential or an authorisation we do not have;
**NOT BUILT** = does not exist.

## 1. Acceptance matrix

| Area | Status | Evidence / limit |
|---|---|---|
| Login, roles, scrypt password hashing, hashed session tokens, lockout, CSRF for cookie sessions | VERIFIED | `tests/app/test_foundation.py`, `test_api.py`, `test_security_hardening.py`; unit + live checks against a running server (execution report 9.A) |
| Request framing (negative / non-numeric Content-Length, chunked bodies, idle connections) | VERIFIED (fixed this session) | `test_security_hardening.py`; live: `Content-Length: -1` -> 400, chunked -> 411 |
| File-source disclosure (database / credentials readable through replay) | VERIFIED (fixed this session) | file cameras must be video files inside an allowed root and outside the data dir except uploads; re-checked at session start and at replay |
| Malformed JSON / wrong-typed fields on every mutating route | VERIFIED (fixed this session) | fuzz test: 4 roles x every POST/PATCH/PUT route x ~250 bodies, no HTTP 500; a planted-fault self-check proves the harness can see one |
| Four-eyes / role separation for case approval | VERIFIED + 1 gap fixed | a plate changed after plate verification can no longer ride on the old verification (schema v2 `plate_verified_reading_id`) |
| Audit log append-only + hash chain | VERIFIED (tamper-evident only) | head hash exposed; tail truncation is not detectable without anchoring it externally |
| Incident -> evidence -> case -> sandbox hand-off -> package, restart persistence | VERIFIED | `scripts/demo_e2e.py` on a fresh data dir (60 s of `demo.mp4`, real YOLOv8n + ByteTrack): 2 incidents, 1 case to `challan_reference_received` (sandbox), audit chain ok, persistence verified after restart |
| Evidence export integrity | VERIFIED (fixed this session) | a package is refused if any file differs from its registered SHA-256; purged files are listed, not silently omitted |
| Retention: expiry of evidence files | VERIFIED (new) | plan -> reviewed `plan_id` -> purge; tombstones keep hashes; open cases protect; audit rows never deleted. Audit expiry NOT BUILT (by design, see 3) |
| Streaming: incremental processing, bounded queues, drop accounting, isolation, shutdown, reconnect | VERIFIED on synthetic / fake sources | 205 streaming tests; **not** real webcam / RTSP / RTMP acceptance |
| Corrupt frames / truncated files | VERIFIED (fixed this session) | a corrupt live frame is skipped and counted (30 in a row = reconnect); a recorded file that ends short of the container frame count is flagged |
| Throughput / latency | PARTIAL | `scripts/stream_synthetic_bench.py`, 5 repeats, stub tracker: plumbing only. 2 of 6 scenarios flagged UNSTABLE on this laptop. No detector, camera or dashboard latency claim |
| Dashboard live update mechanism | VERIFIED (what it is) | polling (`setTimeout`, 4 s) + MJPEG for the annotated view. No WebSocket / SSE exists. UI not re-driven in a browser this session (no JS changed) |
| Stopped vs queued traffic | PARTIAL | stationary-group rule added (lowers severity, still raises the incident); threshold `queue_peer_stops` = 3 is untuned; no real-footage precision |
| Overspeed / sudden braking | BLOCKED-FOOTAGE | fail closed without survey-validated geometry; nothing metric is claimed |
| Wrong-way / wrong-lane / red-light / illegal turn / rash driving | PARTIAL | unit + simulated only; need per-camera zones/lanes/signal feed; no real example |
| Possible collision | PARTIAL (experimental) | no real collision in any footage; false-positive rate unmeasured |
| Helmet / seatbelt | NOT BUILT | no model |
| Plate OCR / plate detection | NOT BUILT as a working capability | adapters + handling tested with doubles only; no engine installed, no plate model exists (the only training attempt died in epoch 1, no checkpoint) |
| Vehicle registry | BLOCKED-EXTERNAL | sandbox fixture only |
| e-Challan | BLOCKED-EXTERNAL | sandbox simulator + evidence-package export only; nothing is ever sent |
| Fine-tuned vehicle model | NOT PROMOTED | gate says DO_NOT_PROMOTE; needs labelled local footage + your approval |
| Local-CCTV benchmark | BLOCKED-FOOTAGE | tooling tested; refuses to give a verdict without labelled frames; one-run test locks now tested |
| Accident detection model | NOT BUILT | design + validators + evaluator only (`docs/ANVESH_ACCIDENT_RESEARCH_PIPELINE.md`) |
| Map view | NOT BUILT | camera coordinates are text only (no fake map) |
| Full-suite regression | VERIFIED in groups (2026-10-21) | 1324 passed, 0 failed across every test directory, run as 8 sequential processes (execution report 10.6). A single-invocation run has still not been completed (memory guard on 2026-10-20) |

## 2. Release blockers (must be cleared before any operational use)

1. **No real camera has ever been connected.** RTSP / RTMP / webcam paths are fake-capture tested only. Needs: a real camera or stream, a test window, and stream credentials. (BLOCKED-EXTERNAL / FOOTAGE)
2. **Detector accuracy on your cameras is unknown.** Needs the labelled local benchmark (about 400 labelled frames across >= 3 cameras; see the benchmark doc). Until then the COCO-pretrained default stays and no accuracy claim may be made.
3. **Licences** (section 4): Ultralytics AGPL-3.0 vs enterprise licence, and the MIO-TCD NonCommercial / ShareAlike terms for the fine-tuned weights, need a legal decision before an external or departmental deployment.
4. **Authorisations** (section 5): registry, e-challan, alerts, and any camera network access are not available to this project.
5. **TLS, network exposure and operations**: the server is HTTP on localhost. Any network exposure needs a TLS reverse proxy, `ANVESH_SECURE_COOKIES=1`, a connection limit at the proxy, backups of the data directory and a decision on MFA / SSO. (open)
6. **Account security gaps**: MFA is not implemented; throttling is in-memory and per source address; accounts are provisioned by two administrators or the offline CLI (section 3, P2/P3).
7. **Time basis**: operational approval requires a declared `recording_start_utc` (recorded files) - unchanged and enforced.

## 3. Security findings, resolved and unresolved

Resolved this session (each has a regression test; details in execution report 9.A): negative / non-numeric Content-Length thread hang; chunked-body request smuggling on keep-alive connections;
no per-connection timeout (slow-loris); **a file camera could be registered on the database or the credentials file and then downloaded through the replay endpoint by any user who may view evidence**;
unvalidated camera update fields; 500s on malformed nested config / wrong-typed JSON; `active: 0` / `"false"` bypassing the self-lockout guard and being coerced by truthiness; case-action body keys colliding with service
arguments; unbounded login username / failure table; plate verification not following a later plate correction; export packages that would carry altered evidence under a consistent-looking manifest.

Unresolved / documented, not fixed:

- **P1 - registry override on operational cases: RESOLVED 2026-10-21 (project-owner decision).** The override is abolished. An operational case cannot be approved, retried or handed off unless its current plate reading is verified and a fresh, well-formed, audit-confirmed result from the configured AUTHORISED (non-sandbox) registry, bound to exactly that reading, says `found`; a database trigger enforces the same rule. No authorised registry exists in this product, so operational approval is **blocked by design** until one is connected (`ANVESH_REGISTRY_URL/_API_KEY`). Details: execution report section 10.1 and `tests/app/test_registry_policy.py`. (Earlier text: `registry_override_reason` used to satisfy the registry item.)
- **P2 - administrators minting officers: MITIGATED 2026-10-21.** Creating, re-enabling or resetting the password of an `administrator`/`officer` account, or moving anyone into or out of those roles, is now a request that a DIFFERENT active administrator must approve (the requester cannot approve their own); the last active administrator cannot be removed; passwords set for others are temporary. Residual risk: two colluding administrators, or any holder of host access (the offline `scripts/manage_users.py` is the audited break-glass path). See `docs/ANVESH_ACCOUNT_SECURITY.md`.
- **P3 - lockout / MFA: PARTLY MITIGATED 2026-10-21.** Throttling is now per (username, source), per source and per username, with a monitored, bounded table; a known user is not locked out of their own desk by an attacker elsewhere. Still in process memory only (lost on restart, not shared between processes, per source address so a proxy hides clients). **MFA is NOT implemented** (no vetted crypto library in the pinned environment; none was hand-rolled): design and remaining work in `docs/ANVESH_ACCOUNT_SECURITY.md`.
- Audit chain: tamper-*evident*; deleting the newest rows is only detectable if the head hash was recorded elsewhere. Audit expiry is not implemented (deleting rows breaks the chain); it needs an offline export + anchor + new segment procedure.
- `ThreadingHTTPServer` has no connection cap (only the per-request timeout); uploaded video is parsed by FFmpeg/OpenCV, so untrusted uploads carry decoder-vulnerability risk - restrict who has `camera.manage` and keep OpenCV patched.
- A clip spanning a mid-stream resolution change would be written at the first frame's size (frames of another size are dropped by the writer). Not reproduced; noted.
- Evidence package filenames are `<case>-v<version>.zip`; re-exporting the same version overwrites the previous export file (the evidence itself is never overwritten).

## 4. Licence review (primary sources read 2026-10-20 unless noted)

| Item | Licence found | Consequence |
|---|---|---|
| Ultralytics YOLOv8 (code, `yolov8n.pt`, training) | AGPL-3.0, or a paid Ultralytics Enterprise License "for development and production use" that bypasses AGPL's open-source requirements ([repo](https://github.com/ultralytics/ultralytics)); the page does not say how trained weights are licensed | Running ANVESH as a network service on AGPL code implies offering the corresponding source to its users unless an enterprise licence is bought, or the detector is replaced by a permissively licensed one. Counsel / Ultralytics to confirm. Open since earlier audits |
| MIO-TCD dataset | **CC BY-NC-SA 4.0** ([dataset page](https://tcd.miovision.com/)); the page does not address trained models | The fine-tuned checkpoint is derived from NonCommercial data: do not deploy it commercially; whether a police department's use is "non-commercial" and what ShareAlike means for weights is a legal question, so keep it non-default (it already is) |
| COCO-pretrained base weights | per Ultralytics distribution terms (above) | as above |
| `keremberke/license-plate-object-detection` plate set | CC BY 4.0 per its README ("Provided by undefined"); Vietnamese plates | attribution chain and plate region unverified; not suitable as an Indian plate detector |
| EasyOCR (code) | Apache-2.0; the page states **no licence for the pretrained weights** (CRAFT detector etc.) ([repo](https://github.com/JaidedAI/EasyOCR)) | verify the weights' terms before installing |
| PaddleOCR | Apache-2.0 for the project; pretrained-model terms are on their model cards ([repo](https://github.com/PaddlePaddle/PaddleOCR)) | not installed; check cards |
| Tesseract | Apache-2.0 code; traineddata licence is in the separate tessdata repo ([repo](https://github.com/tesseract-ocr/tesseract)) | not installed; check tessdata |
| CADP (accident frames) | non-commercial, research use; raw video not shared (earlier session; not re-confirmable by search today) | research only |
| Indian plate OCR datasets | none usable: Indian_LPR not public, IEEE DataPort subscription, Kaggle set unverifiable without login | no Indian OCR training data is available to this project |

## 5. Integrations that need authorisation or credentials

| Integration | What is needed from you / the department | Where it plugs in |
|---|---|---|
| Vehicle registry (Vahan-type lookup) | written departmental authorisation, endpoint + API key, the data-sharing terms | `ANVESH_REGISTRY_URL/_API_KEY`; `anvesh/app/registry.py` gateway contract (ANVESH-defined, untested against any real service) |
| e-Challan | arrangement with the operator (NIC / MoRTH), endpoint + key, legal sign-off on the offence catalogue and rule references | `ANVESH_CHALLAN_URL/_API_KEY`; until then evidence-package export and officer-entered reference |
| Alerts | webhook URL + secret, or SMTP host/credentials | `ANVESH_ALERT_WEBHOOK_*`, `ANVESH_SMTP_*` |
| Live cameras | network access, RTSP/RTMP URLs + credentials, retention / privacy approval for the footage | camera `source_uri` (credentials are redacted in every response and log) |
| OCR / plate model | approval to install EasyOCR (or another engine), its weights' licence check, and a plate model trained on **Indian** plates | `ANVESH_OCR_MODEL_DIR`, `models/plate_detector.pt` |
| Offence rule references | the legal reference per offence | Administration -> Offences |

## 6. Rollback

- **Code**: nothing was committed or staged, so rollback is by file: the changes are listed in the execution report 9.Z; `git diff` / `git status` show them, and no tracked file was reverted.
- **Database**: the first start on a schema-v1 database applies migration v2 (one way: a v1 build refuses a v2 database). A consistent copy is written first to `<db>.pre-v2-migration.sqlite3.bak` next to the database (test: `test_schema_v2_migration_backs_up...`). To roll back: stop the server, restore that file over the database and use the previous build.
- **Retention purge** removes evidence files and cannot be undone except from a backup of the data directory: back up `evidence/` before the first purge.
- **Detector**: default is unchanged; `ANVESH_FINETUNED_MODEL` unset -> pretrained. No checkpoint was overwritten.

## 7. Test and verification commands

```
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs\pytest_tmp\full        # whole suite (about 5-6 min, needs ~3 GB free RAM)
.venv\Scripts\python.exe -m pytest tests\app -q -p no:cacheprovider                                 # application (~4 min)
.venv\Scripts\python.exe -m pytest tests\streaming tests\training -q -p no:cacheprovider            # streaming + training tooling (~1 min)
.venv\Scripts\python.exe scripts\run_app.py --data-dir <NEW dir> --port 8761 --demo-users            # never point a test run at outputs\app
.venv\Scripts\python.exe scripts\demo_e2e.py --base http://127.0.0.1:8761 --data-dir <NEW dir> --seconds 60 --report <NEW file>
.venv\Scripts\python.exe scripts\demo_e2e.py --base http://127.0.0.1:8761 --data-dir <NEW dir> --verify-only      # after a restart
.venv\Scripts\python.exe scripts\stream_synthetic_bench.py --repeats 5 --out <file>                 # plumbing benchmark; read the STABLE/UNSTABLE column
.venv\Scripts\python.exe scripts\validate_yolo_dataset.py --data <data.yaml>                         # dataset validator
.venv\Scripts\python.exe scripts\train_detector.py --name <new> --epochs N --preflight-only          # resource check; no training
```
No linter/formatter is installed in the pinned environment (ruff / black / flake8 / mypy absent); `python -m py_compile` on every changed file is the only static check run.

## 8. Remaining release blockers, in order

1. Local CCTV footage + labels -> detector decision (benchmark doc section 1).
2. Licence decision (Ultralytics, MIO-TCD) with counsel.
3. A real camera session (RTSP) on a test window, including a deliberate disconnect.
4. TLS proxy + backup + MFA (needs an approved dependency) + trusted-proxy handling for per-source throttling.
5. Indian-plate training data and an approved OCR engine, or keep plates manual-only.
6. Registry / e-challan authorisation (or ship export-only).


## 9. Update 2026-10-21 (fifth session)

| Area | Status | Evidence / limit |
|---|---|---|
| Operational approval requires an authorised, non-sandbox registry result for the verified plate | VERIFIED (enforced in service, hand-off and database trigger) | `tests/app/test_registry_policy.py` (31 tests). **No authorised registry exists, so operational approval is blocked by design** (BLOCKED-EXTERNAL) |
| Registry override | REMOVED | 422 for the old field; legacy overrides on pre-existing cases are shown as "LEGACY, not a registry result" |
| Two-person provisioning of administrator/officer accounts; last-administrator protection; temporary passwords; offline recovery CLI | VERIFIED | `tests/app/test_account_security.py` (49 tests); `docs/ANVESH_ACCOUNT_SECURITY.md` |
| Password hashing at OWASP scrypt minimum with upgrade-on-login | VERIFIED | live server reports N=2^17; production constants asserted in tests (the suite lowers the live cost for speed only) |
| Login throttling (user+source / source / user), idle timeout, session revocation | VERIFIED (in-memory scope) | lost on restart, per process, per source address |
| Multi-factor authentication | **NOT BUILT** | no vetted crypto library installed and none hand-rolled; design in the account-security doc section 5 |
| Demo accounts cannot WRITE to operational material (cases, registry, hand-off, plates, incidents, sessions, cameras, retention/offence settings) and cannot launder demo status into real accounts | VERIFIED for writes; reads are not restricted | flagged in schema v4; `test_account_security.py`; **pre-v4 accounts are not auto-flagged: the 2026-10-18 `outputs/app` accounts migrate as non-demo with plaintext passwords on disk - flag/disable them (`manage_users.py set-demo`)** |
| Research/demo-only status visible in API, dashboard and start-up | VERIFIED | `/system/status` `product`, banner, `run_app.py` output |
| Retention screen (preview, protected records, gated purge) | PARTIAL | API behaviour tested; dashboard script syntax-checked only, not driven in a browser |
| Licensing | **OPEN - not decided** | Ultralytics AGPL-3.0 vs enterprise licence; MIO-TCD CC BY-NC-SA 4.0 vs fine-tuned weights (a licence for one dependency does not grant the other); OCR weights; CADP |

Release blockers now, in order: (1) authorised registry access + departmental authorisation; (2) licence decisions; (3) labelled local footage and a detector decision; (4) a real camera session; (5) MFA (approved dependency) and trusted-proxy handling behind TLS; (6) e-challan authorisation; (7) Indian plate data/OCR or keep plates manual.


## 10. Update 2026-10-31 (sixth session: presentation product)

Full report: `docs/ANVESH_SESSION6_REPORT.md`. Status stays **RESEARCH / DEMONSTRATION ONLY**; every blocker in sections 2 and 8 is unchanged (no authorised registry, no real camera, licences undecided, MFA not built, plates manual).

| Area | Status | Evidence / limit |
|---|---|---|
| 20-minute recorded-video session through the real app (YOLOv8n + ByteTrack, GPU) | VERIFIED | `demo.mp4` first 1200 s at video speed: 30,001 frames, 0 dropped, 24.95 fps average, frame p50/p95/p99 27.7/37.5/47.2 ms, server RSS 1.38-1.62 GB with no growth trend; two complete runs gave identical incident counts (27) |
| Per-track speed in km/h (dashboard, over-speed, braking) | PARTIAL - APPROXIMATE | scale from the UK lane-marking standard (new `MARKING_STANDARD_HOLDOUT` kind), hold-out RMSE 0.43 m, road identity and lane width assumed, near field only; not a survey |
| Over-speed / sudden braking on real footage | NO EVENTS in 20 min | rewritten (smoothed, significant, sustained); the previous versions produced ~200 false braking events in 160 s and inflated over-speed events; unit-tested only for positives |
| Collision candidate | EXPERIMENTAL, gated | queue / standstill / gradual-stop / not-closing gates, 6 new tests; no real collision exists to test against; 0 candidates in 20 min of real footage; fires on the SIMULATED fixture and not on the SIMULATED queue |
| Browser acceptance | VERIFIED in headless Chrome (Playwright) | 12/12 checks: live stream decodes, evidence images load, evidence clip plays, no JS errors, no overflow at 1440 and 390 px |
| Officer workflow, sandbox hand-off, no override, restart persistence, audit chain | VERIFIED on the final run | `demo/tools/officer_workflow.py` 15/15; persistence VERIFIED after restart (317 audit rows) |
| Full test suite | 1350 passed, 0 failed, 0 skipped | 9 sequential pytest processes (`demo/tools/run_test_groups.sh`) |
| Plates, accident model, wrong-way on real footage | NOT DONE / NOT DEMONSTRATED | see the session report |


## 11. Update 2026-10-31 (seventh session: presentation freeze)

Full report: `docs/ANVESH_SESSION7_REPORT.md`. Status unchanged: **RESEARCH / DEMONSTRATION ONLY**; every blocker above stands.

| Area | Status | Evidence / limit |
|---|---|---|
| 20-minute real-time session, fresh data dir, presenter walkthrough in a browser beside it | VERIFIED | 30,001 frames, 0 dropped, 24.89 fps, frame p50/p95/p99 27.4/38.9/48.1 ms, RSS 1.37-1.57 GB (+1.8 % first to last quarter); 18 real-footage incidents (14 stopped, 4 congestion) |
| Stop episodes whose closure is learned one window late | FIXED | 2 of 14 stop incidents on the real footage never received an end; `StopEpisodeTracker.views` / `evict_before`; 2 tests |
| Queue stops re-rated "isolated obstruction" when the queue clears | FIXED | queue reading latched per episode, DISSIPATING counts as queue context, `queue member` badge; 4 tests |
| Congestion flapping (13 episodes in 20 min) | FIXED (policy value) | close grace 60 s on the demo camera -> 4 episodes; severity from non-free time; the proxy's assumed capacity is disclosed on the incident |
| Tracker low-score association | MEASURED, NOT ADOPTED | `tracker: {conf, config}` camera setting exists; conf 0.10 = -8 % ids but +10 % boxes shifting the congestion proxy; `match_thresh` 0.9 = -18 % ids but cannot be shown not to join vehicles without labels |
| Full test suite | 1381 passed, 0 failed | 9 sequential processes |
| Browser: 12/12 acceptance on a live session, 36/36 page-size checks (1920x1080 .. 1024x768), presenter click-through (no API shortcuts) | VERIFIED | `demo/tools/ui_acceptance.py`, `ui_resolutions.py`, `presenter_walkthrough.py` |
| Restart persistence (incidents, cases, audit chain) | VERIFIED | `demo/tools/restart_check.py` |

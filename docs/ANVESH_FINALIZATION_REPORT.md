# ANVESH finalization report (Session 9, 15 November 2026)

**Status: finalized as a research / demonstration system.** It is implemented and tested. It is **not** validated against ground truth for speed, congestion,
stop incidents, wrong-way, collision or cause ranking, it is not production-ready, and it is not cleared for enforcement or deployment on any road or railway.

## 1. Project overview

ANVESH turns fixed-camera traffic video into **reviewable, evidence-backed incident candidates** and walks an officer through an audited review. It watches a
camera, tracks vehicles, derives lanes / approximate speed / stopped vehicles / congestion / jam-cause candidates / wrong-way and collision candidates, saves
hashed evidence, and keeps a hash-chained audit log. Candidate causes are ranked with a Dempster-Shafer fusion core that is allowed to say "insufficient evidence".
Nothing it produces is a finding: the reviewing officer is the final authority; registry and challan results are synthetic sandbox data.

## 2. Final architecture (as implemented)

```
video file / synthetic scene / live URL
  -> frame source (paced at video speed for files)           anvesh/perception/ingestion.py, anvesh/streaming/
  -> bounded frame queue (capacity 8, policy "block")        anvesh/streaming/queues.py
  -> YOLOv8n + ByteTrack (Ultralytics 8.4.132, model.track)  anvesh/perception/tracking.py
  -> session identity, windows (10 s), traffic state,         anvesh/streaming/, anvesh/perception/traffic_state.py
     stop episodes, congestion hysteresis
  -> per-frame app analytics: lanes, live speed, wrong-way,   anvesh/app/{officer,detectors,livespeed,jamcause,plate_reader}.py
     collision candidate, jam-cause candidate, plates
  -> incidents + hashed evidence + case state machine +       anvesh/app/{incidents,evidence,cases,audit,registry,handoff}.py
     hash-chained audit (SQLite, WAL)
  -> HTTP API (stdlib ThreadingHTTPServer) + MJPEG stream     anvesh/app/{server,routes}.py
  -> officer screen (/officer) and full dashboard (/advanced) anvesh/app/static/  (vanilla JS, 1-2 s polling)
```
Core reasoning (single- and two-camera evidence, fusion, ranking, propagation, feedback, evaluation harness) lives in `anvesh/{evidence,fusion,hypotheses,propagation,feedback,evaluation,world}`.
No WebSocket, no SSE, no external web framework. The application runs **one camera per session**; two-camera fusion is implemented and unit-tested, not exercised by the shipped app.

## 3. Completed in this session (Session 9) and in Session 8

- **Session 8** (commits `629242d`, `976e4fd`, `dfb4046`, `6ba8786`): video-first officer screen state line, pause/stop states, stale-poll guard, stream reconnect; MJPEG stream tests; quiet client disconnects; earlier sessions' core changes, tests, scripts and reports committed.
- **Session 9** (this report; commit in section 11):
  - **Defect fixed:** `officer.js` could leave two sessions running after rapid source switching (interleaved start/stop calls). Starts are now serialised and coalesced (last click wins). Found by `demo/tools/ui_lifecycle.py` (11/12 -> 12/12, run three times).
  - Documentation: handbook rebuilt with three new chapters (20 officer screen / streaming / session lifecycle, 21 analytical validity, 22 performance and testing) and stale plate/OCR statements corrected; panel guide rebuilt with a new officer-screen demonstration section and screenshots captured from the running app; README rewritten (status, configuration, documentation, layout, tests); this report; `docs/benchmarks/`.
  - Tools: `demo/tools/sample_cpu.py`, `demo/tools/summarize_realtime.py`, `docs/viva_assets/capture_screens.py`; `monitor_run.py`'s CPU column fixed (it always read 0).
  - `.gitignore`: personal coursework files and documents embedding footage of unverified licence.

## 4. Tests (verified this session)

`bash demo/tools/run_test_groups.sh demo/runs/session9_final.log` (nine sequential pytest processes), run after all Python changes; the only later code change was the `officer.js` start-serialisation fix (JavaScript, not covered by pytest, verified by the browser lifecycle check below):

| Group | Result |
|---|---|
| app-1 (alerts, api, foundation, incidents) | 38 passed |
| app-2 (cases, registry, plates) | 110 passed |
| app-3 (detectors, event logic, synthetic sessions, ingestion, live speed, lifecycle, tracker config) | first run **1 failed, 108 passed**; the failure was `test_ingestion.py::test_mid_stream_outage_reconnects_with_new_identity_session_and_counts_loss`, a 20 s wait that timed out while the machine was slow; the same group rerun alone: **109 passed** |
| app-4 (officer layer, MJPEG stream, retention, security, accounts) | 99 passed |
| streaming + training | 306 passed |
| perception | 193 passed |
| world + safety + storage | 132 passed |
| evidence .. evaluation | 229 passed |
| top-level | 187 passed |
| **Total** | **1,403 passed** (counting the rerun of app-3); no test was skipped or weakened |

The one timing-bounded test is flaky under load (it passed in the Session 8 baseline and final runs and in the rerun). Not fixed: it tests reconnect timing with real sleeps.
Also: all modules compile and import (`compileall`, import of every `anvesh.*` module), `pip check` is clean.
Browser checks (real Chrome, real server, real `demo.mp4` through the real pipeline): `ui_lifecycle.py` **12/12** (three runs after the fix), officer-screen layout 6/6 sizes with no overflow or JS errors (`ui_viewports.py`; mobile layout was not otherwise in scope).
Not run: Docker build (the Dockerfile is untested), live RTSP / webcam, any non-Windows OS.

## 5. Performance (measured; see `docs/benchmarks/`)

### 5.1 20-minute real-time run (first 1,200 s of `demo.mp4`, `realtime_factor` 1.0, `--no-simulated --autostart`)

| Measure | Value |
|---|---|
| Wall time / video time | **1,377.9 s** / 1,200.0 s |
| Frames received / processed / dropped | 30,001 / 30,001 / **0** (frame queue drops 0, high-water 8 of 8) |
| Average processing FPS | **21.77** (source 25.00) |
| Frame processing p50 / p95 / p99 (mean 38.8) | 34.9 / 71.7 / 105.6 ms |
| detect + track p50 / p95 | 28.1 / 52.4 ms |
| End-to-end p50 / p95 | 43.3 / 663.9 ms (reader-produced frame to processing finished, **including queue wait**; excludes browser display) |
| Server RSS | 929-1,745 MB; first-quarter mean 1,565, last-quarter mean 1,589 (+1.5 %); **peak sampled 1,745 MB** |
| GPU (nvidia-smi, 10 s samples) | utilisation mean 27 %, max sampled 70 %; memory 133-177 MiB total used (looks low; **not cross-checked**) |
| CPU (server process tree) | mean 103 %, max sampled 321 % of one core (16 logical cores); system mean 33 % |
| Stalls | 0 live samples without frame progress; 0 sampler errors; no crash |
| Incidents / tracks / events | 18 / 5,552 session-scoped track ids / 18 |

**Real time was kept for about the first 740 s of video (25.0 fps, empty queue) and was lost from about 800 s** (15-21 fps, queue full, source held back by the blocking queue, nothing dropped). **The cause is not established.**
The machine was a laptop on the Balanced power plan that Windows reported as discharging (on battery) after the run; the user's browser, Docker and a vendor agent were running; and a later fast-mode run was about 2.5x slower than the same code earlier the same day. This is therefore a measurement on a shared, possibly power-limited machine, **not a clean capacity figure**, and it does not support the claim "keeps real time for 20 minutes". A clean repeat needs the charger connected, a high-performance power plan and nothing else running; one run gives no confidence interval.
Because the per-frame callback (officer overlay drawing + JPEG encoding) is inside the measured frame-processing time, part of the difference to Session 7 may be code rather than machine; that was not isolated.

### 5.2 Fast mode (kept separate, pacing removed)

First 150 s: 3,751 frames, 0 dropped, **41.4 fps**, frame processing p50 13.4 / p95 32.0 ms, detect+track p50 10.5 ms; end-to-end p50 145 ms is a queue-wait artefact. A full-length fast run was **aborted after about 580 s** at about 17 fps (p50 about 54 ms) because the machine was contended; kept as `s9_fast_full_ABORTED_contended.csv`, no figure derived.

### 5.3 Comparison with the historical Session 7 run (31 Oct 2026, **not reproduced**)

| | Session 7 (historical) | Session 9 (measured) |
|---|---|---|
| Frames processed / dropped | 30,001 / 0 | 30,001 / 0 |
| Average FPS | 24.89 | 21.77 |
| Frame latency p50 / p95 | 27.4 / 38.9 ms (means of 10 s samples) | 34.9 / 71.7 ms (exact over all frames) |
| RSS | 1,365-1,573 MB, +1.8 % | 929-1,745 MB, +1.5 % |
| Incidents / track ids | 18 / 5,552 | 18 / 5,552 |

What **did** reproduce: zero dropped frames, flat memory, and the identical analytic output (18 incidents, 5,552 track ids) on the same footage. What did **not**: the timing.

## 6. Analytical validity (details in handbook chapter 21)

- **Speed:** least-squares slope of homography-projected ground points over 1.6 s, near-field `speed_zone` only. Scale comes from the **assumed** UK lane-marking standard (2 m dash + 7 m gap, lane width 3.65 m); the site was not surveyed and a homography absorbs a uniform scale, so the true scale error (plausibly +/-10 % or more) is **not** in the displayed +/- band. Plausibility check only (median 97 km/h on a motorway); no radar / GPS truth. Unreliable: far field, the away carriageway, short or fragmented tracks, off-plane geometry.
- **Congestion:** an **occupancy proxy** (`mean_concurrency / 20`, capacity assumed) plus mean speed when metric speed exists; thresholds 8 / 3 m/s and occupancy 0.6 / 0.3; 10 s windows with hysteresis; episode close grace 30 s (60 s in the demo). It can read "congested" while traffic flows at 55-65 km/h. It is not independently verified congestion.
- **Fusion / ranking:** reliability-discounted Dempster-Shafer over H1-H6 (H7 = ignorance), shared-uncertainty blend for two cameras, ranking by (belief, plausibility, id), outcomes RANKED / HIGH_CONFLICT / INSUFFICIENT_EVIDENCE. Belief is not probability. Image-space or unvalidated cameras give `insufficient_evidence` by design. Code docstrings and the handbook were checked against each other.
- **Ground truth:** none exists for speed, congestion, stop incidents, wrong-way, collision, ranking or the plate reader on CCTV, so **no accuracy, precision, recall or F1 is claimed for them**. Historical detector / tracker scores (CityFlow c006 recall 0.633, IDF1 lower bound 0.209; MIO-TCD mAP50 0.870 for the **non-promoted** fine-tuned model) are from `docs/M-A_FOUNDATION_REPORT.md` and `docs/ANVESH_MODEL_TRAINING_REPORT.md` and were not re-run. Needed: a surveyed distance or radar pass; labelled free/slow/congested intervals; a fully labelled hour of stop and queue events; licensed crash and wrong-way clips; independently established causes for ranking.

## 7. Documentation delivered

| Item | Where | In git? |
|---|---|---|
| This report, Session 6-9 reports, README, app guide, runbook | `docs/`, `README.md`, `demo/README.md` | yes |
| Benchmarks (raw samples + summaries) | `docs/benchmarks/` | yes |
| Build scripts and text of handbook and panel guide, screenshot capture | `docs/viva_assets/*.py`, `numbers.json` | yes |
| Technical viva handbook (66 pages, 22 chapters) | `docs/ANVESH_COMPLETE_TECHNICAL_VIVA_HANDBOOK.pdf/.docx` | **no** |
| Panel presentation guide (23 pages) | `docs/ANVESH_PANEL_PRESENTATION_GUIDE.pdf/.docx` | **no** |
| Screenshots, diagrams | `docs/viva_assets/screens/`, `figures/` | **no** |

**Why not committed:** the handbook, guide and screenshots embed frames of `demo.mp4`, whose origin and licence are unverified (`demo/SOURCE.md`: "not redistributable, not committed"). Rebuild locally: start the demo (`run_demo.py --reset --one-click`), run `python docs\viva_assets\capture_screens.py --base http://127.0.0.1:8760`, then `python docs\viva_assets\build_handbook.py` and `build_guide.py`. The PBL report files in the repository root are personal coursework documents (name and roll number) and are git-ignored. `docs/demo_reference/` (CityFlow frames, dataset licence not reviewed) and `stitch/` (design mock-ups) are git-ignored for the same reason. The DOCX layouts were generated but **not opened in Word** in this session; the PDFs were checked by rendering pages.

## 8. Known limitations

Approximate speed (assumed scale); proxy congestion; fragmented track ids (5,552 ids in 20 min, the median box is 28 px; ids are not vehicle counts); detector accuracy on real cameras unmeasured; wrong-way and collision shown only on SIMULATED scenes; plate reader unvalidated on CCTV and silent on the 640 x 360 demo (needs vehicles >= 110 px wide), OCR on saved incident evidence not connected; registry / e-challan / alert channels are sandbox or "not connected"; no MFA; one camera per session; Ultralytics is AGPL-3.0 and MIO-TCD is CC BY-NC-SA, plate-dataset licence chain unverified; `demo.mp4` licence unverified; real-time capacity on this laptop not established (section 5.1); Dockerfile untested; Windows only; live RTSP / webcam never validated on hardware.

## 9. Remaining work (outside what this session could do)

A clean real-time benchmark on a dedicated, plugged-in machine (and a repeat to give a spread); ground-truth annotation for speed / congestion / stops; licensed replacement footage; deployment hardening (HTTPS, MFA, authorised registry); re-checking the DOCX layouts in Word.

## 10. Ownership of results

Every number above is read from `docs/benchmarks/*`, `demo/runs/session9_final.log` or the cited earlier reports; none is typed from memory except the Session 8 fast-mode figures, which were transcribed from a terminal printout and are marked as such in `docs/benchmarks/s8_fast_summary.json`.

## 11. Git

See the final commit hash and push status in the completion message and in `git log` (branch `officer-screen`, remote `origin` = https://github.com/AryanRajguru22/ANVESH).

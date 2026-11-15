# ANVESH Session 7 report - presentation freeze (2026-10-31)

RESEARCH / DEMONSTRATION ONLY. Branch `main`, HEAD `b6d5ecd` (unchanged). Nothing was staged, committed, pushed, stashed, reset or cleaned.

## 1. What changed and why

| Area | Finding | Change | Status |
|---|---|---|---|
| Stop episodes (core) | **Defect.** A stop episode whose track vanished near the end of window k is still "open" in k, is closed during window k+1 with `end < start(k+1)`, and `episode_views` drops it; the engine's per-frame eviction also deleted it. 2 of 14 real stop incidents stayed "still open" for the rest of the stream. | `StopEpisodeTracker.views` reports an episode that was last handed out open and has since closed once, as CLOSED; `evict_before` keeps it until reported (`anvesh/safety/stop_episodes.py`). 2 tests. | fixed, verified on the real footage (all 14 now have an end) |
| Stopped vehicle (app) | **Defect.** The incident is re-emitted every window and the last emission is stored. A stop that began in a congested queue was re-rated "medium, outside congested traffic" in the window where the congestion level fell to DISSIPATING/FREE_FLOW (INC at 135 s in the first run): a queued vehicle reported as an isolated obstruction as its queue cleared. DISSIPATING was not treated as queue context. | Queue reading is **latched per stop episode**; DISSIPATING counts as queue context; every queue-context stop carries `queue_member`, `queue_group_id`, `queue_group_peak_stationary`; the incident list shows a **queue member** badge. One incident per vehicle is kept (a human must be able to see each). 4 tests. | fixed |
| Congestion (app) | 13 separate episodes over 20 min, several opening in the same window they closed in: one stop-and-go regime cut up by single free-flow windows. | `CongestionDetector` **close grace** (`congestion_close_grace_s`, code default 30 s, demo 60 s): free-flow windows stay in the trend, the episode continues if congestion returns, otherwise it closes with the end back-dated to the first free-flow window. Severity uses `non_free_s`, never the span. New indicators `non_free_s`, `interruptions`, `clearing_since_s`. 7 tests. The core `CongestionStateTracker` was NOT touched. | fixed: 13 -> 4 episodes |
| Congestion honesty | "Congested" is declared from an occupancy proxy (mean concurrency / an **assumed** capacity of 20 vehicles). On this 6-lane view it fires while the mean speed of all vehicles is 55-65 km/h (e.g. 230 s: dense but flowing). | Not retuned (would be tuning to a narrative). The incident now states it: "In N of M congested/building windows the mean speed ... was still X-Y km/h ... dense, not necessarily slow traffic". Capability note and README updated. | disclosed |
| Tracking | Root cause of part of the fragmentation: `model.track(conf=0.25)` removes every box below 0.25 before ByteTrack sees it, which disables ByteTrack's second association stage on low-score boxes (`track_low_thresh` 0.1). | New validated per-camera setting `tracker: {"conf", "config"}` (repo-relative yaml, fail closed); library defaults untouched. **Measured, not adopted** (section 3). | opt-in |
| UI | Simulated incidents showed a real-footage "Detector validation" note. No vehicle history. | Note replaced by "scripted SIMULATED scenario" text for simulated data; **SIMULATED** chip in the list; **events with this track** link in *Vehicles* with the tracklet caveat; congestion page shows non-free time and merged gaps. | done |

## 2. Before / after on the same footage (first 300 s of `demo.mp4`, default tracker, no labels)

| | Before (session 6) | After |
|---|---|---|
| stopped-vehicle incidents | 14 (13 low + 1 "medium, isolated") | 14, all `low`, all `queue member`, one queue group |
| stop incidents never given an end | 2 | 0 |
| congestion episodes in 300 s | 1 (20-180 s) | 1 (20-170 s, end back-dated, closed after 60 s of free flow) |

Full 20 minutes (real time, fresh data dir): real-footage incidents **27 -> 18** (stopped 14 -> 14, congestion 13 -> 4). Congestion episodes (start-end, non-free seconds): 20-170 s (150), 360-680 s (190 of a 320 s span, 5 short gaps merged), 840-950 s (50 of 110, 2 gaps), 1020-1200 s (150 of 180, 2 gaps; still open at the end of the stream). Replaying the first run's level sequence: grace 10 s -> 13 episodes, 20 s -> 10, 30 s -> 7, 40 s -> 7, 60 s -> 4. The footage alternates building / congested / dissipating roughly every 50-60 s, so 60 s is a policy value, stated as such.

## 3. Track continuity study (`demo/tools/track_continuity.py`, 180 s from 60 s, same detector, no labels -> proxies only)

| Variant | raw ids | tracks < 1 s | median len s | near-field births | near-field ends | re-link candidates | size jumps | tracks with class change |
|---|---|---|---|---|---|---|---|---|
| baseline (conf 0.25, default yaml) | 808 | 346 | 2.08 | 161 | 256 | 62 | 2 | 167 |
| conf 0.10 | 744 | 268 | 2.34 | 151 | 233 | 53 | 2 | 201 |
| conf 0.10 + match_thresh 0.9 | 660 | 179 | 2.68 | 111 | 147 | 26 | 5 | 197 |
| conf 0.10 + buffer 50 + match 0.9 | 653 | 173 | 2.72 | 110 | 145 | 25 | 5 | 198 |

(Near-field = rows >= 200 and away from the frame border; far-field vehicles legitimately appear in the interior at the horizon.) Identity-jump proxy (centre moves > 0.5 box widths in one frame) was 0 everywhere. Run time is ~10 ms/frame in all variants.

**Decision.** conf 0.10 gives -8 % ids and -15 % near-field re-link candidates but adds about 10 % boxes per frame; in a 300 s app run it pushed the count-based congestion proxy over its threshold for flowing traffic (episode 10-280 s instead of 20-170 s; frame at 230 s is dense and moving). The extra boxes cannot be validated without labels, so **the demo keeps the default tracker**; the setting stays available (`demo/camera_config.json` documents it). `match_thresh` 0.9 is the larger gain but is a looser matcher: ids fall because tracks are joined, and with no labels it cannot be shown that different vehicles are not being joined (size jumps rose 2 -> 5, small numbers). It was NOT adopted. The remaining fragmentation is mostly short far-field tracks and detector flicker at 640x360 (median vehicle box 28 px). Identity stays a tracklet; nothing merges tracklets.

## 4. Full acceptance run (fresh `run_demo.py --reset`, real time, nothing else on the GPU; the presenter walkthrough ran in a browser beside it)

| Metric | Value |
|---|---|
| Frames processed / dropped | **30,001 / 0** (video 1200 s at 25 fps) |
| Processing fps (average) | 24.89 |
| Frame latency | p50 27.4 ms, p95 38.9 ms, p99 48.1 ms (means of the 10 s samples; maxima 27.9 / 40.8 / 51.5) |
| Server RSS | 1,365-1,573 MB; first-quarter mean 1,460, last-quarter mean 1,486 (**+1.8 %**) |
| GPU | 138-160 MB reported by `nvidia-smi`, utilisation <= 64 % (the memory figure looks low and was not cross-checked) |
| Frame queue | capacity 8, high water 8, nothing dropped |
| Frame ring | at most 13.5 MB |
| Real-footage incidents | 18 (14 stopped, 4 congestion); no over-speed / braking / aggressive / wrong-way / collision on the real footage |
| Raw data | `demo/runs/perf_freeze.csv`, `demo/runs/freeze_summary.json` |

Incremental: the engine is fed frame by frame by the session thread at `realtime_factor` 1.0 (progress bar, queue depth, streaming MJPEG); incidents appear in the live feed as the video reaches them. Nothing is read from a result file.

## 5. Verification

| Check | Command | Result |
|---|---|---|
| All tests, 9 sequential processes | `bash demo/tools/run_test_groups.sh demo/runs/test_groups_freeze.log` | **1381 passed, 0 failed** (38 + 110 + 109 + 77 + 306 + 193 + 132 + 229 + 187; was 1350, +31 new tests) |
| Presenter click-through, 1920x1080, no API shortcuts | `python demo/tools/presenter_walkthrough.py --base http://127.0.0.1:8762 --data-dir demo/data-walk` (fresh `--reset` dir) | **23/23 steps passed** with the final script (an earlier version failed one step - my audit check looked for the oldest rows in a newest-first view - which was rewritten and then exercised both in `--audit-only` mode and in this full pass). Covers: fresh start shows no incidents; real video start; SIMULATED collision (exactly 1), SIMULATED wrong-way (exactly 1), SIMULATED queue (8 queue members, no collision); case -> evidence verify -> manual plate -> second reviewer -> sandbox registry -> approve -> SANDBOX hand-off `TEST-CHALLAN-...`; audit entry for the hand-off and hash chain verify; no JavaScript errors |
| Officer workflow (API) | `demo/tools/officer_workflow.py --data-dir demo/data` | **15/15**: evidence re-hashes, manual plate recorded as `manual_entry`, OCR unavailable, registry override 422, registry labelled SANDBOX, official hand-off 409, demo account cannot register an operational camera (403), audit chain verifies (407 rows) |
| Browser acceptance against a live session | `python demo/tools/ui_acceptance.py` | **12/12** (stream decodes 640x360, progress/fps/queue/calibration notice, evidence images, clip plays, banner, no overflow at 1440/390, no JS errors). The first attempt ran against a *finished* session and failed its 3 live-page checks; it was re-run on a live one |
| Page-by-page layout | `python demo/tools/ui_resolutions.py` | **36/36** page-size checks (9 pages x 1920x1080, 1366x768, 1280x720, 1024x768): banner, no sideways scroll, no broken image, no clipped element, 0 JS errors |
| Restart persistence | `demo/tools/restart_check.py snapshot/compare` | **VERIFIED**: 46 incidents, 2 cases and their states unchanged; audit chain intact (571 -> 579 rows, the extra rows are the logins of the check); a stopped session stays stopped |
| Evidence | `officer_workflow` + browser | every asset re-hashes; clips decode and play in Chrome |
| Stale incidents | walkthrough step 0 | a fresh `--reset` shows "No incidents match." |

## 6. Preservation

* `models/yolov8n.pt` f59b3d83...ccfc83b36, `anvesh_mio5_yolov8n_best.pt` cdc3cf10...ba7f73bf21, `..._last.pt` 0ca32c41...d035119, `weights/yolo26n.pt` 9b09cc8b... - all unchanged. Default detector unchanged. Held-out MIO-TCD split not run, lock files untouched.
* `outputs/app` (database last written 18 Oct) untouched; every run used `demo/data*` directories.
* Git: HEAD `b6d5ecd`, nothing staged/committed/pushed/stashed. One tracked file newly modified by this session: `tests/streaming/test_ma1_stop_episodes.py` (+2 tests); `anvesh/safety/stop_episodes.py` was already modified.
* I stopped only my own servers, by exact PID (two attempts to stop by pattern / image name were refused by the permission classifier and were not worked around).
* No network downloads, no installs, no training.

## 7. Files

Modified: `anvesh/safety/stop_episodes.py`, `anvesh/app/detectors.py`, `anvesh/app/cameras.py`, `anvesh/app/runner.py`, `anvesh/app/capabilities.py`, `anvesh/app/static/views_cases.js`, `views_admin.js`, `tests/streaming/test_ma1_stop_episodes.py`, `demo/README.md` (rewritten to the 13 presentation steps), `demo/tools/build_demo_config.py`, `demo/camera_config.json` (regenerated: + `congestion_close_grace_s` 60), `demo/tools/presenter_walkthrough.py`, `demo/tools/monitor_run.py`, `demo/tools/run_test_groups.sh`.
New: `tests/app/test_event_lifecycle.py` (11), `tests/app/test_tracker_config.py` (18 collected: 4 functions, 14 parametrised cases), `demo/tools/track_continuity.py`, `demo/tools/trackers/*.yaml` (study variants), `demo/tools/summarize_run.py`, `demo/tools/ui_resolutions.py`, `demo/tools/restart_check.py`, this report.

## 8. Known limitations (unchanged unless stated)

* Congestion is an occupancy proxy with an assumed capacity; dense flowing traffic can read "congested" (now disclosed on the incident). Per-carriageway analysis would be the real fix.
* The 60 s close grace is a policy value tuned on one video's level sequence.
* Stop incidents are still one per vehicle (14 in one queue); "same vehicle after a tracker break" is not established.
* Stops are judged on the approximate metric scale over the whole frame (extrapolated outside the near field).
* Detector accuracy unmeasured; ids fragment; class of a track can flicker.
* Speed approximate (UK lane-marking assumption, site not surveyed, +/- excludes scale uncertainty); no over-speed/braking/aggressive/wrong-way/collision event in the real footage.
* Wrong-way and collision exist only as SIMULATED scenarios; collision is experimental, no trained accident model.
* No plate detection/OCR; manual plate is a DEMO PLACEHOLDER. No real registry/challan. MFA not implemented.
* `demo.mp4` origin and licence unknown (classroom use only, never committed or redistributed).
* The GPU-memory figure from `nvidia-smi` (138-160 MB) looks implausibly low and was not cross-checked.

## 9. Recommendations

1. Rehearse once from `--reset` with the printed steps; do not wait for events the footage does not contain.
2. Label ~200 frames of the footage; use them to decide `conf` 0.10 and `match_thresh` 0.9 properly, to set a per-carriageway congestion capacity, and to score pretrained vs fine-tuned.
3. Collapse queue stops into one expandable queue incident in the UI once the one-per-vehicle policy is reviewed.
4. Survey two known distances to make speed defensible; then tune over-speed and braking on labelled examples.
5. Licensed footage with legible plates, a verified plate OCR licence, real collision clips, MFA, and an authorised registry/e-challan remain the blockers for anything beyond a demonstration.

## 10. Disposable data

`demo/data` (freeze run + walkthrough + restart check), `demo/data-fast` (300 s runs), `demo/data-final`, `data-run1`, `data-ft`, `data-test`, `probe_data` (older sessions) are all git-ignored and safe to delete (about 550 MB). `demo/*.log|err|txt` likewise.

Note: `demo/data` also holds a second, short real-video session (about 2 minutes, started over the API for the live-page browser check and then stopped) and the walkthrough's cases, so its 46-incident restart snapshot is not the incident count of the 20-minute run (18 real-footage incidents; the rest are SIMULATED scenarios and the second session).

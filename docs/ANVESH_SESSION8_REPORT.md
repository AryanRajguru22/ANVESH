# ANVESH Session 8 report - recovery and video-first UI (2026-11-15)

RESEARCH / DEMONSTRATION ONLY. Branch `officer-screen`, HEAD at start `a74a400` (not `b6d5ecd`). Remote `origin` = github.com/AryanRajguru22/ANVESH (the URL in the brief returned "not found").

## Recovery
The interrupted session ended cleanly after commit `a74a400` (officer screen, 2026-11-14). Nothing newer was half-written. Many earlier-session files (46 modified core files, tests/app, scripts, demo tools, docs) had never been committed; they are preserved and committed in this session. Handbook and panel guide (2026-11-01) predate the officer screen; they describe the `/advanced` dashboard flow, which still exists. They were not regenerated.

## Baseline (before changes)
`bash demo/tools/run_test_groups.sh` -> 1,381 passed in 9 groups. A first attempt overlapped two runs sharing a temp dir and gave false errors; the clean rerun passed.

## Changes
- Officer screen: status line above the video (state pill, camera, session id, recorded/SIMULATED), PAUSED message, correct Stopped/Interrupted/Finished wording, stale-poll guard (a late answer for a previous session is dropped), stream reconnect (3 tries), side column 360 px (video 1022 px wide at 1440), phone layout with a scrollable source strip. Overflow at 320 px fixed (grid min-width).
- Lanes tile now summarises ("10 vehicles in lanes now; busiest AWAY-2 (3)").
- Dashboard live view: compact phone header (video from y=687 to ~513 at 320 px; chip-line fix applied afterwards, not re-measured); outdated "no plate detector installed" text corrected.
- Server: vanished clients no longer print tracebacks (`_QuietThreadingServer`).
- New tests: `tests/app/test_mjpeg_stream.py` (5: whole ordered frames, ends on stop, session isolation, thread freed on abrupt disconnect, 404/401) and a lane-tile test. New tools: `demo/tools/ui_viewports.py`, `demo/tools/ui_lifecycle.py`.

## Verified (real pipeline, real demo.mp4, headless Chrome)
- Officer screen at 320/375/390/768/1024/1440: no overflow, video on screen, no JS errors (6/6).
- `ui_lifecycle.py` 12/12: frames reach the page, LIVE, clock advances, pause/resume, stop (server state `stopped`), stream request ends, 8 quick source switches leave the server responsive with one session running.
- Tracker facts confirmed in Ultralytics 8.4.132: class-level ID counter, track_buffer 30 frames, conf 0.25 passed to predict hides the low-score stage. Not retuned (Session 7 measured conf 0.10: rejected).

## Final test run (after the last code change)
`bash demo/tools/run_test_groups.sh demo/runs/session8_final.log`: 38 + 110 + 109 + 77 + 306 + 193 + 132 + 229 + 187 = **1,381 passed, 0 failed**, plus `tests/app/test_officer_layer.py` and `tests/app/test_mjpeg_stream.py` (**22 passed**, now listed in group app-4 of the runner; the runner previously omitted the officer-layer file). Total 1,403.

## Performance (measured this session; fast mode, nothing else running, RTX GPU, ultralytics 8.4.132)
First 150 s of `demo.mp4`, `run_demo.py --fast`, single session, default tracker:

| Metric | Value |
|---|---|
| Frames processed / dropped | 3,751 / 0 |
| Throughput | 41.4 fps (wall 90.7 s). Fast mode is not real time; the Session 7 real-time run (24.89 fps = source rate) is historical |
| detect+track | p50 10.5 ms, p95 23.6 ms, p99 31.8 ms (max 5.5 s = first-frame model warm-up) |
| frame processing | p50 13.4 ms, p95 32.0 ms, p99 44.8 ms |
| frame end-to-end | p50 145 ms, p95 276 ms - includes waiting in a full queue (capacity 8, blocking policy), so it is a fast-mode artefact, not live latency |
| Events / tracks | 14 event candidates, 723 session-scoped track ids (not unique vehicles) |

Memory/GPU were not sampled in this run (Session 7 figures stand, historical).

## Remaining / not done
Handbook, panel guide and screenshots were not regenerated (they describe the `/advanced` flow). Dashboard 320/390 px was not re-audited after the last chip-line CSS rule. Limitations unchanged: speed approximate (assumed lane-marking pitch), congestion is an occupancy proxy (assumed capacity 20), plates unreadable at 640x360, no ground-truth validation, research/demonstration build and not deployment-ready.

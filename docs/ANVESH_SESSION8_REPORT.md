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

## Not done (usage limit)
Final full 9-group test rerun after the last edits; quiet-machine performance re-measurement (historical Session 7 figures stand: 24.89 fps, p50 27.4 ms, p95 38.9 ms - historical, not re-measured); dashboard 320/390 px re-audit after the final CSS line; screenshot/handbook refresh; push. Handbook/panel guide untouched. Limitations unchanged: speed approximate, congestion is an occupancy proxy, plates unreadable at 640x360, nothing validated against ground truth, not deployment-ready.

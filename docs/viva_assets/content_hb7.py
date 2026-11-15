from pathlib import Path
from docbuilder import *

blocks = []
A = blocks.append

# =============================================================== CH 20
A(H1("Chapter 20 - The Officer Screen, Live Video and the Session Lifecycle (added in Sessions 8-9)"))
A(CALLOUT("note", "This chapter describes code that exists in `anvesh/app/` today. Where something is **measured** the source is named; where it is not measured, it says so.", "Scope"))
A(H2("20.1 Two front ends over one backend"))
A(TABLE(["Screen", "URL", "Purpose", "Files"], [
    ["Officer screen", "`/officer` (and `/` when started with `--one-click`)", "one page: choose a source, watch the annotated video, see lanes / speed / wrong-way / accident / plate / jam-cause tiles and the alert list", "`static/officer.html`, `officer.css`, `officer.js`"],
    ["Full dashboard", "`/advanced` (and `/` without `--one-click`)", "cameras, live monitoring, incidents, cases, evidence verification, audit, administration", "`static/index.html`, `app.js`, `views_*.js`"],
], [1.1, 1.6, 2.6, 1.7], "The two screens"))
A(P("Both are plain HTML + vanilla JavaScript served by the Python standard library (`http.server.ThreadingHTTPServer`, one thread per request) with SQLite in WAL mode behind it. "
    "There is **no** WebSocket and **no** server-sent events: the page *polls* JSON (officer screen every 1 s, dashboard live view every 2 s) and shows the video with an ordinary `<img>` whose source is an MJPEG stream. "
    "Passwords are hashed with `hashlib.scrypt`; the content-security policy forbids inline scripts, which is why the JavaScript is in separate files."))
A(H2("20.2 From a video file to a picture on the screen"))
A(TABLE(["Stage", "What happens", "Live or batch?", "Where"], [
    ["1 Source", "a camera row (file, synthetic scene, or live URL) is read frame by frame; a recorded file is paced at `realtime_factor` 1.0 (video speed) unless `--fast`", "live-paced (recorded file)", "`streaming/ingestion.py`, `app/cameras.py`"],
    ["2 Detect + track", "`model.track()`: YOLO detection + NMS + ByteTrack on the GPU", "live, per frame", "`perception/tracking.py`"],
    ["3 Session thread", "one daemon thread per session feeds the streaming engine; frames pass through a bounded queue (capacity 8, blocking policy; high-water and drops are counted)", "live", "`app/runner.py`, `streaming/queues.py`"],
    ["4 Analytics", "windows of tracks -> traffic state, congestion, stop episodes; per-frame lane map, speed, jam-cause candidate, wrong-way and collision detectors", "live, per frame / per window", "`perception/traffic_state.py`, `app/detectors.py`, `app/officer.py`, `app/jamcause.py`"],
    ["5 Overlay", "boxes, track ids, lanes, speed and jam hull are drawn on the same frame and JPEG-encoded into one slot, `last_jpeg`", "live", "`app/officer.py`, `app/runner.py`"],
    ["6 Persistence", "an incident is written to SQLite with hashed evidence (frames, annotated frame, clip, thumbnail) as it is raised", "live", "`app/incidents.py`, `app/evidence.py`"],
    ["7 API", "`GET /sessions/{id}/officer` returns the tiles and alerts; `GET /sessions/{id}/stream.mjpg` streams `last_jpeg`", "live", "`app/routes.py`"],
    ["8 Browser", "the officer page polls the JSON and the `<img>` shows the stream", "live", "`static/officer.js`"],
], [0.9, 3.0, 1.1, 1.8], "Execution path (nothing here reads a result file)"))
A(P("Not everything is live: the **registry / challan** results are *synthetic sandbox* data, the three test cameras (`sim-jam`, `sim-wrong-way`, `sim-accident`) are *scripted scenes* labelled SIMULATED, and a finished session shown later is *historical* data read from SQLite. "
    "The page labels a session as `recorded footage, processed live` (a file played at video speed) or `SIMULATED scene`."))
A(H2("20.3 The MJPEG stream and why it cannot grow without bound"))
A(BUL([
    "The session keeps **one** slot, `last_jpeg` - not a queue. The stream handler loops every 80 ms and writes the slot only if it holds a *new* object. A slow browser therefore skips frames; it never makes the server buffer them.",
    "Each part is `--frame`, `Content-Type: image/jpeg`, `Content-Length`, then the bytes, so a frame is always whole. Frames from one session come out in order and never repeat.",
    "When the session reaches a terminal state (`stopped`, `completed`, `failed`, `interrupted`) the slot is cleared and, two seconds later, the handler ends the response. A stopped session is not an endless feed.",
    "A closed tab raises `BrokenPipe` / `ConnectionReset` in the handler, which then returns; the thread is freed. A custom `_QuietThreadingServer` logs vanished clients at debug level instead of printing a traceback.",
    "An unknown session id gives 404; no sign-in gives 401. Every stream is looked up by its own session id, so two sessions cannot share a picture.",
    "All of this is tested over a real socket in `tests/app/test_mjpeg_stream.py` (5 tests: whole ordered frames, end on stop, isolation between two sessions, handler threads released after 5 abrupt disconnects, 404 / 401).",
]))
A(H2("20.4 Session states and what the screen shows"))
A(TABLE(["State", "Meaning", "Officer screen"], [
    ["starting / connecting", "model loading or source opening", "pill STARTING, message \"Starting... loading the detector\""],
    ["live", "frames are flowing", "pill LIVE (green), clock and progress advance"],
    ["paused", "recorded sources only (live sources refuse pause)", "pill PAUSED and the message \"Paused - press Resume to continue\""],
    ["reconnecting / stalled", "a live source stopped delivering", "pill RECONNECTING / STALLED"],
    ["completed", "the file ended", "end card \"Finished\""],
    ["stopped", "an operator pressed Stop", "end card \"Stopped\" (not \"Finished\")"],
    ["failed / interrupted", "an error, or the server restarted", "end card with the first error / \"Processing was interrupted\""],
], [1.4, 2.6, 2.8], "States"))
A(P("Safeguards in `officer.js`: a poll answer for a session that is no longer the selected one is **dropped** (so a slow reply cannot overwrite a newer screen); if the stream connection breaks while the session is still running the page reconnects up to three times; "
    "starting another source first stops every other running session, so one video is processed at a time; and the last frame stays on the screen only under an end card that says so. "
    "`demo/tools/ui_lifecycle.py` drives this in real Chrome against the real server (start, pause, resume, stop, eight quick source switches) and passed 12 of 12 checks in Session 8."))
A(H2("20.5 The one-click demonstration sign-in"))
A(P("`POST /auth/demo` exists only if the server was started with `--one-click` **and** listens on a loopback address **and** the caller is on loopback. Otherwise the route does not exist. It signs in the flagged DEMO investigator account whose password was generated randomly into `demo/data/initial_credentials.txt` (git-ignored)."))
A(H2("20.6 The jam-cause candidate, the lane map and the plate reader"))
A(BUL([
    "**Jam-cause candidate** (`jamcause.py`): vehicles that moved less than a fraction of their own width over 2 s are STOPPED; stopped vehicles within a few widths of each other form a QUEUE; the vehicle at the downstream front is the *candidate*, with three reported facts (did it stop first, how many queued behind it, a Spearman order correlation). Rated `likely` or `possible`. It works in image space and does not know *why* the front vehicle stopped. It is always worded as a candidate.",
    "**Lane map**: operator-drawn lane polygons; a vehicle's ground point decides its lane. The tile summarises (`10 vehicles in lanes now; busiest AWAY-2 (3)`).",
    "**Plate reader** (`plate_reader.py`, real-footage sources only): a YOLOv8n plate detector (12 epochs on a Vietnamese plate dataset; its own validation mAP50 0.98, which says nothing about other plates or CCTV) plus RapidOCR, on its own thread with a queue of 2. It tries a vehicle only when the box is at least 110 px wide, accepts a read only when detector and OCR confidence are high and the text looks like a plate (4-10 letters/digits), and calls a read *confirmed* only when a second read agrees. On the 640 x 360 demo clip vehicles are about 28 px wide, so it never fires and the tile says \"Too small\". Every read is a candidate for an officer to check.",
]))

# =============================================================== CH 21
A(H1("Chapter 21 - Analytical Validity: What the Numbers Do and Do Not Mean"))
A(H2("21.1 Speed estimation"))
A(P("Pipeline: ground-contact point of each tracked box (bottom centre, never the centroid) -> homography to road-plane metres -> least-squares slope of position over the latest samples INSIDE the calibrated zone (<= 1.6 s, >= 8 samples, >= 0.4 s) (`livespeed.py`; `perception/speed_estimation.py` for the engine) -> per-track smoothing. "
    "Safety invariant (M-A): a km/h value exists **only** when the camera's geometry is validated as metric; otherwise the system carries unit-less image-space motion and the screen shows \"not calibrated\"."))
A(TABLE(["Assumption / error source", "Effect"], [
    ["Scale from the UK lane-marking standard: 2 m dash + 7 m gap (9 m pitch), lane width 3.65 m. The site was **not surveyed** and the road's identity is unverified", "any wrong pitch scales every speed by the same factor (the shorter 1 m + 5 m marking would read about 65 km/h instead of about 97). A homography absorbs a uniform scale, so the hold-out test cannot detect it. Plausibly +/-10 % or more, **not** in the displayed +/- band"],
    ["Planar road assumption, one camera, one homography", "a crest, a curve or the central reservation breaks the plane; only the toward-camera carriageway was fitted"],
    ["Fit quality: RMSE 0.094 m on 10 points; hold-out dash 0.43 m", "checks self-consistency of the geometry, not the metric scale"],
    ["Pixel scale grows with distance: about 0.42 m/px at row 225, 0.10 m/px at row 320", "far vehicles are noisy, so the displayed speed is limited to a near-field `speed_zone` (rows >= 225); boxes clipped by the frame edge are discarded"],
    ["Ground point from a bounding box, not the tyre contact", "a few pixels of jitter become km/h; the least-squares fit, outlier removal and the plausibility gate (`MAX_PLAUSIBLE_SPEED_KMH`) limit it; implausible tracks are rejected"],
    ["Timestamps come from the video clock", "a time-lapsed or speed-altered video would scale all speeds; this footage looks real-time but that is not verified"],
], [3.2, 3.6], "Speed: assumptions and sources of error"))
A(P("Reliable only for tracks inside the speed zone with at least 0.4 s (8 samples) of unclipped in-zone history (a fast vehicle crosses the ~20 m zone in ~0.7 s). Unreliable: far field, the away-from-camera carriageway, short tracks, track ID swaps, anything outside the fitted plane. "
    "Check done in `demo/tools/speed_plausibility.py` (first 300 s): the Session 10 audit (`docs/ANVESH_SPEED_AUDIT_FINDINGS.md`) found that the fit window used to reach into the extrapolated far field (inflating speeds ~13-17 %); after the fix the median is 94.9 km/h, p90 109.3, trucks 88.1 - a plausibility check, **not an accuracy measurement**. There is no radar or GPS ground truth for this clip. The over-speed threshold (112.7 km/h = 70 mph) is a demonstration value."))
A(H2("21.2 Congestion"))
A(P("A window (default 10 s, non-overlapping) of tracks is summarised by `aggregate_traffic_state`. What is actually measured:"))
A(BUL([
    "**vehicle count** = tracks overlapping the window; **flow** = count / window length;",
    "**mean concurrency** = average number of simultaneously present tracks; **occupancy proxy** = `min(1, mean_concurrency / max_capacity_vehicles)` with `max_capacity_vehicles = 20` **assumed** (it is *not* the fraction of road length occupied);",
    "**mean speed** only from tracks whose speed was measured, and only on a validated-metric camera. With no usable speed the rule falls back to occupancy alone (the image-space rule).",
]))
A(TABLE(["Rule", "Value"], [
    ["World-space low pressure", "mean speed >= 8 m/s (about 29 km/h) and occupancy < 0.6"],
    ["World-space high pressure", "mean speed <= 3 m/s (about 11 km/h) **or** occupancy >= 0.6"],
    ["Image-space (no speed)", "occupancy >= 0.6 high, <= 0.3 low, otherwise medium"],
    ["Hysteresis", "`CongestionStateTracker` turns window pressures into a level (free flow / building / congested / dissipating) so one window cannot flip the state"],
    ["App-level episode", "`CongestionDetector` keeps an episode open through a close grace (code default 30 s, demo 60 s) so one stop-and-go regime is one episode"],
], [2.2, 4.6], "Congestion thresholds (code defaults)"))
A(CALLOUT("limit", "**Congestion here is an occupancy / count proxy, not independently verified congestion.** On the 6-lane demo view it can fire while the mean speed of all vehicles is 55-65 km/h (dense but flowing). Capacity 20 is an assumption, one count covers six lanes, and the 60 s grace is a policy value taken from one video. "
        "False positives: dense free-flowing traffic, many far-field detections flickering in and out. False negatives: slow traffic with few tracks. No labelled congestion data exists to measure either.", "Honest status"))
A(H2("21.3 Evidence fusion and hypothesis ranking (what is implemented)"))
A(BUL([
    "**Evidence builder** (`evidence/evidence_builder.py`) turns the measured window (state, stops, speeds, geometry) into `EvidenceItem`s for the closed hypothesis set **H1 downstream bottleneck, H2 stopped/disabled vehicle, H3 collision-like obstruction, H4 lane blockage (non-incident), H5 upstream demand surge, H6 signal / intersection restriction**. H7 \"insufficient evidence\" is not a cause: it is the ignorance mass.",
    "**Reliability** (`evidence/reliability.py`) scores each camera and window as a documented heuristic weighting of observation count, tracking continuity, calibration status and temporal coverage, times a completeness multiplier for lost frames (not a probability); **Shafer discounting** (`fusion/ds_fusion.py`) moves mass towards ignorance in proportion to unreliability.",
    "**Dempster-Shafer combination** of per-camera mass functions with a **shared-uncertainty blend** (a conservative element-wise-max fallback when both cameras share a degradation such as fog or glare) and explicit handling of total conflict. This cross-camera step is implemented and unit-tested on synthetic and CityFlow-derived data; the **application runs one camera**, so the shipped product exercises the *single-camera discounted* path only.",
    "**Ranking** (`hypotheses/ranking.py`) sorts by (belief desc, plausibility desc, id asc) and returns `RANKED`, `HIGH_CONFLICT` or `INSUFFICIENT_EVIDENCE`. Belief and plausibility are Dempster-Shafer quantities, **not probabilities**; the code never prints a percentage.",
    "**Abstention is a feature**: for an image-space or unvalidated-geometry camera nothing can be tested, so `insufficient_evidence` is the correct result. H1's score is the heuristic `0.5 + 0.5 * occupancy`. A hypothesis that could not be tested is excluded, never counted as contradicted. A ranking is a candidate list for a human; it is not a cause. An *established* cause can only be set by an authorised officer with a written basis.",
]))
A(H2("21.4 Validation status - what has and has not been measured against ground truth"))
A(TABLE(["Output", "Ground truth available?", "What exists (historical, from the named report)"], [
    ["Detector (YOLOv8n)", "Yes, partly", "MIO-TCD held-out test: fine-tuned mAP50 0.870 vs pretrained 0.215 (5-class taxonomy; **not promoted**: on CityFlow S02 CCTV it was slightly worse). CityFlow c006: recall 0.633, precision only a lower bound (GT lists only vehicles seen by two or more cameras). `docs/M-A_FOUNDATION_REPORT.md`, `docs/ANVESH_MODEL_TRAINING_REPORT.md`"],
    ["Tracker (ByteTrack)", "Yes, partly", "CityFlow c006: IDF1 >= 0.209 (lower bound), IDR 0.514, 42 ID switches. Session 7 proxy study on `demo.mp4` (no labels): the default tracker was kept"],
    ["Speed (km/h)", "**No**", "no radar / GPS / surveyed distance; only a plausibility check"],
    ["Congestion level", "**No**", "no labelled congestion intervals"],
    ["Stopped-vehicle incidents", "**No**", "14 stops in 20 min on `demo.mp4` were reviewed by eye, not scored"],
    ["Wrong-way, collision", "**No real events**", "shown only on scripted SIMULATED scenes (none occurs in the real footage)"],
    ["Hypothesis ranking", "**No**", "no annotated causes; `insufficient_evidence` is the usual honest outcome"],
    ["Plate reader", "Own plate dataset only", "mAP50 0.98 on its validation split (Vietnamese plates); not evaluated on CCTV"],
], [1.4, 1.3, 4.1], "Ground-truth status"))
A(P("**No accuracy, precision, recall or F1 is claimed for speed, congestion, stop incidents, wrong-way, collision or ranking.** "
    "What would be needed: (1) a surveyed distance on the road or a radar / GPS pass for speed; (2) a few hundred frames or intervals labelled by two annotators as free / slow / congested, with the camera's lane count and the real capacity; "
    "(3) every stop event on one hour of footage labelled with start / end and cause, to score precision and recall of the stop and queue logic; (4) crash and wrong-way clips with licences that allow use; "
    "(5) for ranking, incidents whose cause was independently established by an officer. This session did **not** build such a dataset."))

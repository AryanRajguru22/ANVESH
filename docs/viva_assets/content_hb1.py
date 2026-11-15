from pathlib import Path
from docbuilder import *

FIGS = Path(__file__).resolve().parent / "figures"
SCR = Path(__file__).resolve().parent / "screens"
F = lambda n: FIGS / n

blocks = []
A = blocks.append

# =============================================================== READ ME FIRST
A(H1("How to use this handbook"))
A(P("This handbook is written for **you** - a B.Tech CSE student who has to explain, demonstrate and defend ANVESH in front of a teacher and an evaluation panel. "
    "Every statement was checked against the repository at `D:\\ANVESH` (source code, configuration, run logs, the Session 6 and Session 7 reports and the installed library code), "
    "not against what a typical traffic system might do. Where something could not be established it says **\"Not established in the current implementation\"**."))
A(TABLE(["Marker", "Meaning"], [
    ["**Real**", "Runs on the actual 20-minute video while it plays; produced by code in the repository."],
    ["**Approximate**", "Real code but the number is only a rough estimate (for example speed in km/h)."],
    ["**Experimental**", "Code exists and is unit-tested on scripted data; never validated on real events."],
    ["**Simulated**", "Scripted boxes from a synthetic camera (sim-collision, sim-wrong-way, sim-queue). Always labelled SIMULATED in the app."],
    ["**Not built / not connected**", "OCR on saved incident evidence, real registry, real e-challan, MFA, a trained accident model (a live plate reader exists on the officer screen: Chapter 20)."],
], [1.3, 4.5], "Status words used throughout"))
A(CALLOUT("tip", "Reading plan: if you have one evening, read Chapters 1, 3, 4, 6, 8, 9, 12 and the revision sheets in Chapter 20, then skim Chapter 18 (the viva questions). "
          "The demonstration itself is in the second PDF (panel presentation guide).", "Short on time?"))
A(CALLOUT("limit", "ANVESH is a RESEARCH / DEMONSTRATION system. It is not approved for traffic-police deployment, real challans or legal decisions. "
          "The demo video's origin and licence are unverified (classroom use only). Every registry and challan result in the demo is SIMULATED.", "Status"))
A(TABLE(["Source of truth used", "What it provided"], [
    ["`docs/ANVESH_SESSION7_REPORT.md`, `ANVESH_SESSION6_REPORT.md`", "20-minute run results, defects fixed, track-continuity study, verification counts"],
    ["`demo/README.md`, `SOURCE.md`, `CALIBRATION.md`, `camera_config.json`", "13-step runbook, video facts, calibration method and the real configuration values"],
    ["`docs/ANVESH_APP_GUIDE.md`, `ANVESH_ACCOUNT_SECURITY.md`, `ANVESH_RELEASE_READINESS.md`", "roles, security model, retention, licences, blockers"],
    ["`docs/ANVESH_MODEL_TRAINING_REPORT.md`", "MIO-TCD fine-tuning and evaluation"],
    ["`anvesh/**` source and `.venv` Ultralytics 8.4.132 source", "exact thresholds, algorithms, ID allocation, ByteTrack internals"],
    ["`demo/data/anvesh_app.sqlite3` (read-only) and `demo/runs/*.csv`", "stored statistics of the 20-minute run: 5,552 track IDs, latency percentiles, jam-cause outcome"],
], [2.6, 3.4], "Where the facts come from"))

# =============================================================== CH 1
A(H1("Chapter 1 - Understanding ANVESH"))
A(H2("1.1 What ANVESH stands for"))
A(P("**Not established in the current implementation.** No file in the repository defines an acronym expansion. "
    "(Outside the repository: \"Anvesh\" is a Sanskrit/Hindi word commonly translated as *search* or *investigation* - that fits the product's purpose, "
    "but it is general knowledge, not something the repository states, so do not present it as an official expansion.) "
    "The product's own tagline in the dashboard is **\"Road event investigation\"**, and the login screen says *\"Road event investigation and traffic intelligence\"*."))
A(H2("1.2 The problem"))
A(P("A city has hundreds of CCTV cameras but only a few people watching them. Normal CCTV gives you **pictures**, not **answers**. When something goes wrong - a vehicle stalls in a lane, a queue suddenly forms, two cars stop together - "
    "an officer must (1) notice, (2) find the right camera and time, (3) work out what happened, (4) preserve proof, and (5) follow a formal process before anyone is penalised. Each step is slow and manual."))
A(TABLE(["Why ordinary CCTV monitoring is not enough", "What ANVESH does about it"], [
    ["Humans cannot watch every camera continuously; attention drops after minutes.", "Detects and tracks every vehicle automatically on every frame and raises **candidate** events."],
    ["Finding an event afterwards means scrubbing hours of video.", "Each event becomes an **incident** with time, camera, key frames, an annotated clip and a timeline."],
    ["Recorded video is easy to dispute: \"was it edited?\"", "Evidence files are written once and SHA-256 hashed; the audit log is hash-chained and can be re-verified."],
    ["A jam shows *that* traffic is slow, not *why*.", "Congestion episodes are linked to nearby incidents and a cause ranking that is allowed to say \"insufficient evidence\"."],
    ["Fines need careful, separated roles (investigator, reviewer, approving officer).", "A case state machine enforces four-eyes review and records every step."],
], [1, 1], "Problem versus contribution"))
A(H2("1.3 What traffic officers struggle with"))
A(BUL(["**Detection delay** - an obstruction is noticed only after a complaint.", "**Evidence quality** - screenshots without hashes, without the moment of the event, or with unclear vehicle identity.",
       "**Cause uncertainty** - a queue can be caused by a crash, a stalled vehicle, a signal, or simply high demand.", "**Process discipline** - who verified what, and when, is hard to prove later."]))
A(H2("1.4 Intended users"))
A(TABLE(["Role in the app", "Real-world equivalent", "What they do in ANVESH"], [
    ["investigator", "control-room operator / investigating officer", "starts monitoring, opens incidents, opens a case, enters the plate manually"],
    ["reviewer", "independent verifier", "verifies the evidence files, verifies the plate (a different person from the one who entered it)"],
    ["officer", "approving traffic officer", "sandbox registry look-up, approval, sandbox hand-off, can read the audit log"],
    ["administrator", "system administrator", "cameras, users, retention, verifies the audit hash chain; cannot approve cases or see owner details"],
], [1, 1.5, 3], "Roles"))
A(H2("1.5 Problem statement, objectives, scope, outcomes"))
A(CALLOUT("def", "**Problem statement (our wording, based on the documents):** Traffic cameras produce video that is too large to watch and too weak as evidence. "
          "There is a need for a system that automatically turns a fixed-camera video into **reviewable, evidence-backed incident candidates** and takes an officer through a controlled, auditable investigation - without ever deciding enforcement by itself.", "Problem statement"))
A(P("**Main objectives**"))
A(NUM(["Detect and track road vehicles in real time on a fixed camera video.", "Estimate where and how fast vehicles move, with honest error labelling.",
       "Raise candidate events: stopped vehicle, congestion, over-speed, sudden braking, wrong-way, wrong-lane, rash driving, possible collision.",
       "Preserve verifiable evidence (frames, clip, hashes) and a tamper-evident audit trail.", "Support an officer-led case workflow ending in a **sandbox** challan reference.",
       "State capability limits openly (what is real, approximate, experimental, simulated, missing)."]))
A(P("**Scope (what is in):** a single fixed camera processed from a recorded file (also written for RTSP / webcam / HTTP streams, but only tested with fake captures), the web dashboard, the HTTP API, SQLite storage, "
    "accounts and roles, retention, and the sandbox registry / challan simulators."))
A(P("**Out of scope / not built:** plate OCR on saved incident evidence (the officer screen has a live plate reader that only fires on vehicles at least 110 px wide), a real vehicle registry, the real e-challan system, a trained accident model, multi-factor authentication (MFA), multi-camera operation in the app, and any live CCTV hardware acceptance test."))
A(P("**Expected outcomes:** a working demonstration that processes a 20-minute video at video speed with zero dropped frames, produces real stopped-vehicle and congestion incidents with evidence, "
    "shows the other detectors on clearly labelled simulated cameras, and walks an incident to a sandbox challan with a verifiable audit trail."))
A(H2("1.6 Real-world use cases"))
A(BUL(["Control-room alerting: \"vehicle stopped in the fast lane for 30 s\" instead of an unnoticed stall.", "Post-incident review: jump straight to the clip, the stopped vehicle's track and the congestion that followed.",
       "Evidence handling: officers can prove a clip is the original file (hash verification).", "Policy analysis (future): per-carriageway congestion statistics once calibrated cameras are available."]))
A(H2("1.7 Monitoring vs traffic intelligence vs road-event investigation"))
A(TABLE(["Idea", "Question it answers", "Example in ANVESH"], [
    ["Traffic **monitoring**", "What is on the road right now?", "live annotated video, vehicles in view, tracked-vehicle table"],
    ["Traffic **intelligence**", "What is the traffic state and trend?", "congestion level per 10 s window, flow, occupancy proxy, mean speed"],
    ["Road-event **investigation**", "What happened, what proves it, and what should a human do next?", "incident with evidence + uncertainty, case workflow, audit trail"],
], [1.2, 2, 3], "Three levels of capability"))
A(H2("1.8 A simple example of an officer using ANVESH"))
A(NUM(["The investigator signs in and starts the camera **Demo motorway**. The live page shows boxes, track numbers and the congestion strip.",
       "At about 1:50 of video a **stopped vehicle** incident appears (rated *low*, with a **queue member** badge, because the road is congested).",
       "The investigator opens it: reads the observed event, the uncertainty list, and plays the annotated clip; original frames and SHA-256 hashes are listed.",
       "They click **Open violation case from this incident**, then **Submit for evidence verification**. A reviewer verifies the evidence; the investigator enters a plate manually; a *second* reviewer verifies the plate.",
       "The officer looks up the SANDBOX registry, approves, and submits to the SANDBOX simulator, receiving a `TEST-CHALLAN-...` reference.",
       "The administrator opens *History & audit* and verifies the hash chain."]))
A(CALLOUT("warn", "In this demonstration the stopped vehicles are queue members on a congested carriageway. They are real detections, but ANVESH itself rates them *low* and says they are most likely queued traffic, not obstructions. "
          "Do not describe them as violations.", "Be honest about the example"))

# =============================================================== CH 2
A(H1("Chapter 2 - Complete System Architecture"))
A(P("ANVESH has three layers. **Perception** turns pixels into tracked vehicles. **Reasoning** turns tracked vehicles into candidate events. The **application layer** turns events into incidents, evidence, cases and an audit trail that people can use."))
A(FIG(F("fig_architecture.png"), 0.9, "End-to-end architecture. Red boxes are simulated or not connected."))
A(H2("2.1 Components, inputs and outputs"))
A(TABLE(["Component", "Receives", "Produces", "Where in code"], [
    ["Video source + decoder", "a file path / RTSP URL / webcam index", "`Frame(index, timestamp, image)` paced to 25 fps (`realtime_factor` 1.0)", "`anvesh/perception/ingestion.py` (`VideoFileFrameSource`); `anvesh/app/cameras.py`"],
    ["Bounded queue", "frames", "frames (BLOCK for files, DROP_OLDEST for live)", "`anvesh/streaming/queues.py`"],
    ["Detector + tracker", "BGR image", "boxes, class, score, raw track ID", "`anvesh/perception/tracking.py` (`ByteTrackYoloTracker`) + Ultralytics"],
    ["TrackManager", "raw-ID detections", "session identities, ground point, trajectories", "`anvesh/perception/track_manager.py`"],
    ["Streaming engine", "tracked detections", "track store, 10 s `WindowSnapshot`s, stop episodes, latency stats", "`anvesh/streaming/engine.py`"],
    ["Calibration / speed", "ground points, 10 reference points", "metric position (m), speed (km/h) with error band", "`anvesh/world/calibration.py`, `anvesh/app/livespeed.py`"],
    ["Frame detectors", "per-frame observations", "over-speed, braking, wrong-way, wrong-lane, rash, collision events", "`anvesh/app/detectors.py`"],
    ["Window reasoner + detectors", "closed window", "traffic state, congestion level, stopped / congestion events, cause ranking", "`anvesh/streaming/reasoning.py`, `anvesh/perception/traffic_state.py`"],
    ["IncidentService", "detected events", "one incident row per event key + timeline", "`anvesh/app/incidents.py`"],
    ["Evidence store", "frames from a ring buffer", "JPEG frames, annotated frames, clip, SHA-256 rows", "`anvesh/app/evidence.py`"],
    ["Database + audit", "all writes", "SQLite WAL file, hash-chained audit rows", "`anvesh/app/db.py`, `anvesh/app/audit.py`"],
    ["HTTP API + dashboard", "browser requests", "JSON, MJPEG video, static JS/CSS", "`anvesh/app/server.py`, `routes.py`, `static/*.js`"],
    ["Case, registry, hand-off", "officer actions", "state transitions, sandbox lookups, `TEST-CHALLAN-...`", "`cases.py`, `registry.py`, `handoff.py`"],
], [1.2, 1.4, 2, 2.2], "Component map"))
A(H2("2.2 Technologies actually used"))
A(P("Versions are taken from the project virtual environment `D:\\ANVESH\\.venv` (Python 3.11.9) and `requirements-lock.txt`. Technologies that merely happen to be installed on the machine (for example FastAPI or Flask) are **not** used by ANVESH."))
A(TABLE(["Technology", "Version", "Role in ANVESH"], [
    ["Python", "3.11.9", "all application and perception code"],
    ["Ultralytics (YOLOv8 + built-in ByteTrack)", "8.4.132 (AGPL-3.0)", "vehicle detection and tracking (`model.track(...)`)"],
    ["PyTorch + CUDA", "2.13.0+cu130", "runs YOLOv8n on the GPU (NVIDIA RTX 3050 6 GB Laptop)"],
    ["OpenCV (`opencv-python`)", "5.0.0.93", "video decode, resizing/drawing, JPEG/clip encoding, `findHomography`"],
    ["NumPy", "2.4.6", "matrices (homography), array maths"],
    ["lap", "0.5.13", "linear assignment solver used by ByteTrack matching"],
    ["SQLite (Python `sqlite3`)", "stdlib", "database in WAL mode"],
    ["`http.server.ThreadingHTTPServer`", "stdlib", "HTTP API and static files (no web framework)"],
    ["`hashlib` (SHA-256, scrypt)", "stdlib", "evidence hashes, audit chain, password hashing"],
    ["Vanilla JavaScript + CSS", "-", "dashboard (`anvesh/app/static/`), no React/Vue"],
    ["pytest, psutil, Playwright (tests only)", "9.1.1 / 7.2.2 / -", "automated tests, resource monitor, headless-Chrome acceptance checks"],
], [2.2, 1.1, 3.2], "Technology inventory (short form; full table in Chapter 17)"))

# =============================================================== CH 3
A(H1("Chapter 3 - What Happens to One Frame?"))
A(P("This is the most important chapter for the viva. Follow the numbered steps against the flowchart. The order is taken from `StreamingPerceptionEngine.process_frame` and `ProcessingSession` (`anvesh/streaming/engine.py`, `anvesh/app/runner.py`)."))
A(FIG(F("fig_frame_flow.png"), 1.0, "The journey of one frame through ANVESH. Three threads cooperate: reader, worker and consumer."))
A(H2("3.1 Step by step"))
A(NUM([
    "**Source supplies a frame.** `VideoFileFrameSource` reads the next frame with OpenCV. For a recorded file at `realtime_factor` 1.0 it **sleeps** until the moment that frame would be shown in real time (frame `i` is released at time `i / fps`). So the 20-minute video really takes 20 minutes.",
    "**Decoder output.** The frame becomes `Frame(index, timestamp = index / fps, image, capture_wall_s, decode_s)`. Median decode time is about 1.1 ms.",
    "**Frame is queued.** A bounded queue of capacity 8 holds frames between the reader thread and the worker thread (Chapter 4).",
    "**Windows that are due are closed first.** Before the new frame's detections are added, any 10-second window whose end time has passed is closed from the store as it was. This guarantees a window never depends on the frame that triggered it.",
    "**YOLOv8n + ByteTrack run.** `model.track(image, persist=True, tracker='bytetrack.yaml', classes=[...], conf=0.25)`. Ultralytics resizes/pads the image to the model's input size (default `imgsz` 640; ANVESH does not override it), runs the network on the GPU, applies the 0.25 confidence filter and non-maximum suppression (IoU 0.7), then ByteTrack associates the boxes with existing tracks. Output: for each vehicle a box, class, confidence and (once confirmed) a raw track ID.",
    "**Class mapping.** Model class IDs are mapped **by name** to ANVESH vehicle classes (`ClassMap`): bicycle, car, motorcycle, bus, truck. Anything else (person, traffic light ...) is dropped.",
    "**Identity and ground point.** `TrackManager` converts the raw ByteTrack ID into a **session identity** and records the **ground point** = bottom-centre of the box, with its timestamp.",
    "**`on_frame` callback** (still the worker thread): updates `TrackHistory`, adds a JPEG to the evidence **frame ring**, and (at most every 0.1 s) renders the annotated preview frame with boxes, IDs, class, direction arrows and, inside the speed zone, a speed in km/h.",
    "**Frame-level detectors** run on the updated tracks: over-speed, sudden braking, wrong-way, wrong-lane, rash driving, possible collision.",
    "**When a window closes** (every 10 s of video) the consumer thread builds the traffic state (count, flow, occupancy proxy, mean speed), advances the congestion state machine, closes stop episodes and runs the cause ranking. Stopped-vehicle and congestion detectors emit events from it.",
    "**An incident is created or updated.** `IncidentService.upsert` uses the event's *natural key*; the same event re-sent never creates a second row.",
    "**Evidence is captured.** Key frames are taken from the frame ring, annotated copies and an H.264 review clip are written, and each file's SHA-256 is stored.",
    "**Data is persisted** in SQLite and an audit entry is chained.",
    "**The dashboard sees it.** The browser polls `/api/v1/sessions/{id}` every 2.5 s (and lists/incidents every 4 s); the video pane is an MJPEG `<img>` stream of annotated frames."]))
A(H2("3.2 Vocabulary: frame, detection, track, vehicle, event, incident, case, evidence"))
A(TABLE(["Term", "Meaning in ANVESH", "Tiny example"], [
    ["**Frame**", "one image of the video, with an index and timestamp", "frame 3,750 = 150.0 s at 25 fps"],
    ["**Detection**", "one box found by YOLO in one frame (class + confidence)", "\"car, 0.91, box (210, 190)-(290, 250)\""],
    ["**Bounding box**", "rectangle `(x1, y1, x2, y2)` in pixels around a detection", "28 px wide for a median vehicle in demo.mp4"],
    ["**Track**", "detections in consecutive frames linked as the same moving object", "the same car over 80 frames"],
    ["**Track ID**", "the label of a track (two layers - Chapter 6)", "raw ID 7 -> displayed `00051`"],
    ["**Vehicle**", "the real-world car. ANVESH does **not** know vehicles, only tracks", "one real car may produce 3 tracks"],
    ["**Event**", "something a detector noticed (a Python object)", "`stopped_vehicle` for track 00051"],
    ["**Incident**", "a stored, reviewable candidate built from an event, with evidence", "`INC-20261031-...`"],
    ["**Case**", "an officer-led violation work item built on one incident", "`CASE-...` in state `awaiting_plate_verification`"],
    ["**Evidence asset**", "one stored file for an incident (raw frame, annotated frame, clip) with a hash", "`AST-...`, SHA-256 `ef577ce...`"],
], [1.1, 3, 2], "Core vocabulary"))
A(CALLOUT("say", "\"A frame is a picture, a detection is a box in a picture, a track links boxes across pictures, and an incident is what a detector builds from tracks. "
          "ANVESH never claims that a track is a real vehicle - one vehicle can break into several tracks.\"", "One-sentence answer"))

# =============================================================== CH 4
A(H1("Chapter 4 - FPS, Video Processing and Real-Time Performance"))
A(H2("4.1 Definitions"))
A(TABLE(["Term", "Meaning", "In ANVESH"], [
    ["**Frame (picture)**", "one image of the video", "640 x 360 pixels"],
    ["**FPS (frames per second)**", "how many frames per second", "source 25 fps"],
    ["**Source FPS**", "frame rate stored in the video", "25.0"],
    ["**Inference FPS**", "frames the detector alone could handle per second = 1 / (detect+track time)", "p50 detect+track 24.9 ms -> about 40 fps (derived)"],
    ["**End-to-end processing FPS**", "frames fully processed per second of wall time", "24.89 (monitor) / 24.99 (stored)"],
    ["**Frame latency**", "time one frame spends in the system", "p50 27.4 ms, p95 38.9 ms"],
    ["**Queue depth**", "frames waiting between two threads right now", "0-2 sampled; capacity 8"],
    ["**Dropped frame**", "a frame that was discarded before processing", "0 of 30,001"],
    ["**Bounded queue**", "a queue with a maximum size", "capacity 8 (frames), 16 (windows)"],
    ["**Back-pressure**", "a full queue makes the *producer* wait", "BLOCK policy for files"],
], [1.3, 2.3, 2.3], "Performance vocabulary"))
A(FORMULA(r"\mathrm{FPS}=\frac{\mathrm{processed\ frames}}{\mathrm{processing\ time}}", "FPS formula. Example: 30,001 frames / 1,200.66 s = 24.99 FPS (stored session statistic)."))
A(FORMULA(r"\mathrm{frame\ interval}=\frac{1}{\mathrm{FPS}}\qquad \frac{1}{24.89}=0.04018\ \mathrm{s}\approx 40.2\ \mathrm{ms}\quad(\frac{1}{25}=40.0\ \mathrm{ms})", "Frame interval"))
A(FORMULA(r"\mathrm{latency}=t_{\mathrm{completion}}-t_{\mathrm{arrival}}", "Latency. In the code two latencies are recorded (see 4.3)."))
A(FORMULA(r"\mathrm{real\ time\ factor}=\frac{\mathrm{video\ seconds\ processed}}{\mathrm{wall\ seconds\ used}}=\frac{1200.0}{1200.66}\approx 0.9995", "Real-time factor of the 20-minute run."))
A(P("**FPS versus real-time factor.** FPS counts frames per second of *wall clock*. The real-time factor compares *video time* with wall time. At 25 fps source video they are proportional (RTF = FPS / 25). A system that reads a 25 fps file as fast as it can may have FPS = 80 and RTF = 3.2: it finishes early. "
    "In the presentation mode ANVESH deliberately **paces** to RTF = 1.0, so average FPS is expected to sit near 25 - it measures \"kept up\", not \"maximum speed\"."))
A(H2("4.2 Incremental processing, bounded memory and back-pressure"))
A(FIG(F("fig_queue.png"), 1.0, "Three threads joined by bounded queues (anvesh/streaming/pipeline.py, queues.py)."))
A(BUL(["**No whole-video preprocessing.** The reader thread yields one frame at a time; the worker detects and tracks that frame immediately. Detection, events and evidence appear *while the video plays*; nothing is read from a result file (Session 7 report section 4).",
       "**Bounded memory by design.** Frame queue 8, window queue 16, a frame ring limited by *seconds* (window 10 s + dwell 3 s + 30 s = 43 s, JPEG, at most 10 fps, measured <= 13.5 MB), trajectories trimmed to the history actually needed, finished tracks evicted, latency kept in a fixed 20,000-sample reservoir.",
       "**If inference is slower than the source:** for a *file* the policy is **BLOCK** - the reader waits, so no frame is dropped and results equal a batch replay (the video just plays slower than real time). For a *live* stream the policy is **DROP_OLDEST** - the oldest queued frame is discarded and counted in `frames_dropped`, and the affected windows are marked degraded.",
       "**GPU:** Ultralytics chooses the CUDA device automatically (ANVESH sets no `device`); the dashboard chip shows \"GPU\" when `torch.cuda.is_available()`. The first frame includes CUDA / model warm-up - the largest single frame time in the run was 2,187 ms, which the reports attribute to first-frame warm-up (\"likely, not confirmed\").",
       "**Keeping a 20-minute session alive:** the session runs on its own daemon thread; every stage's exception is captured; a failing detector is logged and skipped without stopping perception (`_detector_error`); resources are released at the end (`_release`)."]))
A(H2("4.3 What exactly is timed"))
A(TABLE(["Recorded stage (`LatencyRecorder`)", "Measures", "20-minute p50 / p95 (stored)"], [
    ["`decode`", "time to read one frame from the file", "1.09 / 1.68 ms"],
    ["`detect_track`", "`model.track()` = GPU YOLO + NMS + ByteTrack", "24.9 / 33.1 ms"],
    ["`store_update`", "identity assignment + track store", "0.10 / 0.22 ms"],
    ["`frame_processing`", "whole `process_frame` call, start to end (**our \"completion minus start\"**)", "27.9 / 38.6 ms"],
    ["`frame_end_to_end`", "from the moment the source released the frame to the end of `process_frame` (includes waiting in the queue)", "28.3 / 39.9 ms"],
    ["`windowing`", "building a window snapshot (121 windows)", "34.8 / 56.4 ms"],
], [1.7, 3.2, 1.4], "Latency stages (demo/data session SES-2626B4805A)"))
A(CALLOUT("note", "The headline figures in the Session 7 report (p50 **27.4 ms**, p95 **38.9 ms**, p99 48.1 ms) are *means of the 10-second monitor samples*. The session's own stored statistics (27.9 / 38.6 / 46.6 ms) come from a 20,000-sample random reservoir, so they are marked `percentiles_exact: false` for a 30,001-frame run. "
          "The two agree within about 0.5 ms. Quote the Session 7 numbers and know the difference.", "Two slightly different sets of numbers"))
A(P("Processing FPS is computed in `runtime()` as `frames_processed / (now - started_wall)`, where `started_wall` is set when the session object is created. So the average includes start-up (opening the video, loading the model, warm-up); "
    "that is why the monitor's final value is **24.89** while the stored session figure, measured over 1,200.66 s, is **24.99**. Both are \"about 25\" because the video is paced."))
A(H2("4.4 The Session 7 results and what they mean"))
A(FIG(F("fig_perf.png"), 1.0, "20-minute acceptance run, from demo/runs/perf_freeze.csv (119 samples every 10 s)."))
A(TABLE(["Result", "Value", "What it tells you", "What it does NOT prove"], [
    ["Frames processed / dropped", "30,001 / 0", "every frame of the 1,200 s (frames 0 to 30,000) was handled; the bounded queue never overflowed", "that every vehicle in them was detected"],
    ["Average processing FPS", "24.89", "kept up with a 25 fps source in real time", "maximum throughput: pacing caps it near 25"],
    ["Latency p50 / p95", "27.4 / 38.9 ms", "typical frame done in 27 ms; 95 % within 39 ms - both under the 40.0 ms frame interval", "that no single frame was slower (maximum was 2,187 ms, warm-up)"],
    ["Memory (RSS)", "1,365-1,573 MB; +1.8 %", "no sign of a leak over 20 minutes; bounded buffers work", "that memory is flat over hours or days"],
    ["Queue", "capacity 8, high water 8, nothing dropped", "the worker occasionally lagged a little but never enough to overflow", "headroom on a slower machine"],
], [1.4, 1.2, 2.5, 2.1], "Interpretation of the performance run"))
A(P("**Why p95 latency is not average FPS.** Average FPS says how many frames finish per second *in the long run*. p95 latency says how long a *single* frame takes in its slow tail. "
    "If every frame took exactly 27.9 ms, throughput could reach 1/0.0279 = **35.8 FPS** (a derived estimate; the run was paced, so the ceiling was not measured). Throughput of 24.89 and p95 = 38.9 ms are consistent: a slow frame (38.9 ms) still finishes before the next one is due (40.2 ms), so the queue stays near empty."))
A(CALLOUT("say", "\"The 20-minute test shows that ANVESH processes frames incrementally with bounded memory and keeps up with a 25 fps source on my laptop GPU. It does not by itself show detection accuracy - "
          "a system can run fast and still be wrong, so accuracy is a separate question that needs labelled footage.\"", "How to say it"))
A(CALLOUT("limit", "GPU memory reported by `nvidia-smi` during the run (138-160 MB) looks implausibly low for a CUDA process and was **not cross-checked** (Session 7 report). Do not quote it as a result. Real-time was shown on one laptop, one video, one run (the earlier Session 6 run gave 24.95 FPS).", "Honest limits"))

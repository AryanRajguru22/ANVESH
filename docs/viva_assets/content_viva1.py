from pathlib import Path
from docbuilder import *

FIGS = Path(__file__).resolve().parent / "viva_figs"
blocks = []
A = blocks.append

# ====================================================================== 0
A(H1("0. Read this first: the questions you could not answer last time"))
A(P("You were asked: *how many frames per second does it process, which model, which algorithm, how are IDs assigned, and why are the ID numbers limited?* Here are short, honest answers. Every later part explains them properly."))
A(TABLE(["Question", "Short answer you can say out loud"], [
    ["**How many frames per second?**", "\"The video is 25 fps. In our last 20-minute real-time test the system averaged **21.8 fps** and processed all 30,001 frames with none dropped, but it kept up with 25 fps only for the first ~12 minutes of video; after that it fell to 15-21 fps. That test ran on a shared laptop that was probably on battery, so it is not a clean capacity number. With the video pacing removed (fast mode, 150 s) it reached **41.4 fps**.\" (Section 10)"],
    ["**Which model?**", "\"**YOLOv8n** (the *nano* size of YOLO version 8, 3.16 million parameters), the **pretrained COCO weights** from Ultralytics. We did **not** train it. We fine-tuned a copy on MIO-TCD for 8 epochs (2.97 h) as an experiment, but it was **not promoted** because it was slightly worse on real CCTV.\" (Section 4)"],
    ["**Which algorithm?**", "\"Detection: YOLOv8n. Tracking: **ByteTrack** (Kalman filter + IoU matching + Hungarian assignment). Speed: homography to road metres + least-squares slope over 1.6 s. Congestion: occupancy proxy with hysteresis. Causes: **Dempster-Shafer** evidence fusion that is allowed to say 'insufficient evidence'.\""],
    ["**How are IDs assigned?**", "\"ByteTrack gives each new track the next integer from a counter. ANVESH then gives every track its own **session serial** (`session-t00001`, `t00002`...). It is a *temporary label for one uninterrupted track*, not the vehicle's identity or number plate.\" (Section 5)"],
    ["**Why are ID numbers 'limited'?**", "\"They are not capped. The counter just keeps counting (we saw **5,552** track IDs in 20 minutes). Two things make it *look* limited: ByteTrack's ids **restart at 1** after a reset and have **gaps**, and the printed form `t00001` is only a minimum width, not a maximum. Also, a car lost for more than 30 frames (1.2 s) comes back with a **new** ID, so the number of IDs is **not** the number of cars.\""],
], [1.5, 5.3], "The five answers"))
A(CALLOUT("tip", "Golden rule for the viva: say what is **implemented**, what is **measured**, what is **assumed** and what is **not validated**. Examiners respect a clearly stated limit; they punish a claim you cannot defend.", "How to answer"))

# ====================================================================== 1
A(H1("1. The project in plain English"))
A(H2("1.1 What is ANVESH and what problem does it solve?"))
A(P("**ANVESH** (full title: *Vehicle Behaviour Analysis and Traffic Forensics System*) takes video from a fixed road camera and turns it into **reviewable, evidence-backed incident candidates**: a vehicle stopped in a lane, a traffic jam, a vehicle going the wrong way, a possible collision, and a possible *reason* for a jam. It keeps proof (saved frames and clips with a cryptographic hash) and lets an officer review each case with a tamper-evident audit log."))
A(P("The problem: a city has far more CCTV cameras than people watching them, and a camera only **records**; it does not tell you *what happened* or *why traffic stopped*. After an incident an officer must find the right camera and time, work out what happened, and keep proof nobody can dispute. That is slow and error-prone."))
A(TABLE(["Question", "Answer"], [
    ["Why is analysing vehicle behaviour important?", "Stopped vehicles, wrong-way driving and jams cause crashes and delays; early, evidence-backed alerts help officers act and later explain what happened."],
    ["Limits of traditional monitoring", "Humans watch many screens and miss events; searching hours of video is slow; plain detection counts cars but does not explain causes; evidence is easy to dispute."],
    ["Our solution", "Automatic detection and tracking, measured behaviour (lanes, approximate speed, stops, congestion), candidate causes with honest uncertainty, hashed evidence, and an officer-led review workflow."],
    ["What is different", "Not a new detector. The contribution is the **integration**: (1) candidate-cause reasoning that **abstains** when evidence is weak (Dempster-Shafer with reliability discounting); (2) **evidence integrity** (SHA-256 + hash-chained audit + four-eyes approval); (3) **honesty gates** (no km/h without calibration, simulated scenes labelled, nothing presented as proven)."],
    ["Who benefits / applications", "Traffic police control rooms, highway operators, accident investigators, road-safety researchers. Examples: stopped-vehicle alerts, jam analysis, post-incident review."],
    ["Current limitations", "Research/demo system: speed is approximate, congestion is a proxy, no ground-truth validation, one camera per session, not production-ready (Section 13)."],
], [2.0, 4.8], "Project idea"))
A(H2("1.2 Say it in 30 seconds, 1 minute and 2 minutes"))
A(CALLOUT("note", "\"ANVESH watches a fixed traffic camera, finds events such as stopped vehicles, jams and possible collisions, and saves tamper-proof evidence so an officer can review what happened. It does not claim certainty: everything is a candidate for a human to confirm.\"", "30 seconds"))
A(CALLOUT("note", "\"Cameras record but they do not explain. ANVESH runs a YOLO detector and a ByteTrack tracker on the video, turns the tracks into measurements - lanes, approximate speed, stopped vehicles, congestion - and raises incident candidates with saved, hashed evidence. For jams it ranks possible causes using Dempster-Shafer evidence fusion, and it is allowed to answer 'insufficient evidence'. An officer reviews each case in a dashboard with an audit trail. It is a research prototype: we are honest that speed is approximate and nothing is validated against ground truth.\"", "1 minute"))
A(CALLOUT("note", "Add to the minute: \"The video is processed live, frame by frame: a decoder feeds a queue, the GPU runs YOLOv8n and ByteTrack, and the officer screen shows the annotated frames as an MJPEG stream while a small JSON feed updates the tiles. Evidence goes into SQLite with a SHA-256 hash and a hash-chained audit log. We tested it with 1,403 automated tests and a 20-minute real-time run. The detector is pretrained; the new work is the reasoning, evidence and workflow around it.\"", "2 minutes (what to add)"))
A(H2("1.3 A 5-minute walkthrough"))
A(NUM(["**Problem** (30 s): cameras record, they do not explain; evidence is easy to dispute.",
       "**Start the demo** (`Start ANVESH Demo.bat`): the officer screen opens; five cards, each labelled REAL VIDEO or SIMULATED.",
       "**Click the real motorway video**: the status line shows LIVE, the camera and the session id; boxes and track numbers are drawn on the frames the detector is processing *now*.",
       "**Point at the tiles**: lanes, speed (marked approximate), wrong-way, accident, plate (\"too small\" at 640x360), jam cause.",
       "**Pause/Resume/Stop**: the screen says PAUSED / Stopped; the server really ends the session and the stream.",
       "**Click 'Test: jam from a breakdown'** (SIMULATED): the jam-cause tile names a *candidate* at the front of the queue and says why.",
       "**Open the full dashboard**: incidents with hashed evidence, a case, four-eyes approval, the audit chain.",
       "**Close with limits**: approximate speed, proxy congestion, no ground truth, research only."]))

# ====================================================================== 2
A(H1("2. Complete system architecture"))
A(FIG(FIGS / "fig_architecture.png", 1.0, "ANVESH data flow. Top row: the live per-frame path. Middle: windows, incidents, evidence reasoning. Bottom: API and the two screens."))
A(H2("2.1 Each component: what, why, in, out, who uses it"))
A(TABLE(["Component (code)", "What it does and why", "Input -> Output", "Used by"], [
    ["**Frame source** `perception/ingestion.py`", "Reads video frames with OpenCV; a recorded file is paced at video speed (`realtime_factor` 1.0). Needed so processing behaves like a live camera.", "file / synthetic scene / URL -> `Frame(image, timestamp, index)`", "queue"],
    ["**Frame queue** `streaming/queues.py`", "Bounded buffer (8 frames, policy *block*): if processing is slower, the reader waits; memory cannot grow. Counts drops and high-water.", "frames -> frames", "engine"],
    ["**Detector** YOLOv8n", "Finds vehicles in one image.", "image -> boxes, scores, class ids", "tracker"],
    ["**Tracker** ByteTrack", "Links boxes across frames so a vehicle keeps an ID.", "detections -> boxes with raw track ids", "TrackManager"],
    ["**TrackManager** `perception/track_manager.py`", "Gives each track a session-scoped serial identity, keeps its history and occlusion state.", "raw ids -> `session-t00001` + trajectory", "analytics, windows"],
    ["**Analytics** `app/officer.py`, `detectors.py`, `livespeed.py`, `jamcause.py`", "Per frame: lane of each vehicle, speed, stopped vehicles, wrong-way, collision candidate, jam-cause candidate, plates.", "tracks -> tiles, alerts, incident candidates", "API, storage"],
    ["**Windows + traffic state** `streaming/windows.py`, `perception/traffic_state.py`", "Every 10 s: vehicle count, flow, occupancy, mean speed, congestion level with hysteresis.", "tracks -> `TrafficState`", "congestion detector, evidence"],
    ["**Evidence + fusion + ranking** `evidence/`, `fusion/`, `hypotheses/`", "Scores hypotheses H1-H6, discounts by camera reliability, combines with Dempster-Shafer, ranks (may abstain).", "window state -> ranked candidate causes", "incident record"],
    ["**Storage** SQLite (WAL)", "Incidents, evidence files with SHA-256, cases, users, audit chain.", "incidents -> rows + files", "dashboard"],
    ["**API/server** `app/server.py`, `routes.py`", "Python standard-library HTTP server; JSON routes and the MJPEG stream.", "HTTP requests -> JSON / JPEG stream", "browser"],
    ["**Officer screen / dashboard** `app/static/`", "Vanilla JavaScript pages: show video, tiles, alerts; cases and evidence.", "JSON + stream -> what the officer sees", "the officer"],
], [1.5, 2.6, 1.7, 1.0], "Components"))
A(H2("2.2 One video, from entry to the final screen"))
A(NUM(["You click **Real motorway CCTV**. The page sends `POST /api/v1/cameras/demo-motorway/sessions`. The server checks your role, creates a session row (`SES-...`) and starts **one daemon thread** for it.",
       "The thread opens `demo.mp4` with OpenCV and builds the detector + tracker (a warm-up at server start makes the first session faster). Status: *starting* -> *live*.",
       "Each frame (640x360, every 40 ms at 25 fps) is decoded and put into the frame queue.",
       "The engine takes a frame, runs `model.track()`: YOLOv8n finds vehicles (confidence >= 0.25, five vehicle classes), ByteTrack attaches raw IDs.",
       "TrackManager maps raw IDs to session serials and appends the ground point (bottom centre of each box) to each track's history.",
       "Analytics use the history: lane polygons, speed (inside the speed zone), stopped vehicles, jam cause... Every 10 s a window closes and the congestion level is updated.",
       "The annotated frame (boxes, ids, lanes, banner) is JPEG-encoded into one slot, `last_jpeg`. An incident, if raised, is stored with its frames, clip and SHA-256 hash.",
       "The browser shows frames through `<img src=\".../stream.mjpg\">` and polls `GET /sessions/{id}/officer` once a second for tiles and alerts.",
       "You press Stop: `POST .../stop`; the session thread ends, state becomes *stopped*, the stream closes, the page shows the **Stopped** end card."]))

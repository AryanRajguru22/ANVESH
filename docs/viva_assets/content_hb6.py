from pathlib import Path
from docbuilder import *

FIGS = Path(__file__).resolve().parent / "figures"
F = lambda n: FIGS / n
blocks = []
A = blocks.append

# =============================================================== CH 19
A(H1("Chapter 19 - Honest Limitations and Future Scope"))
A(H2("19.1 What is working, approximate, experimental, simulated, unimplemented, not connected"))
A(TABLE(["Category", "Features"], [
    ["**Working** (real, on the footage)", "frame-by-frame processing at video speed; YOLOv8n detection; ByteTrack tracking and session identities; counts and direction arrows; lane assignment (image-space); stopped-vehicle and congestion incidents; evidence frames, clip, SHA-256; incident lifecycle; case workflow; audit chain; retention plan/purge; roles, throttling, sessions"],
    ["**Approximate**", "speed in km/h (marking-standard calibration, near field only, +/- excludes scale error); stop judgement on the far carriageway (extrapolated scale); congestion level (occupancy proxy against an assumed capacity of 20); vehicle class (coarse COCO classes)"],
    ["**Experimental**", "possible-collision candidate (gates added, never validated); over-speed, sudden-braking and aggressive-driving candidates (policy thresholds, no event in the real footage); jam-cause ranking (reports insufficient evidence)"],
    ["**Simulated**", "sim-wrong-way, sim-collision, sim-queue cameras; every registry result and challan reference (sandbox fixture / `TEST-CHALLAN-...`)"],
    ["**Unimplemented**", "OCR on saved incident evidence (a live plate reader exists on the officer screen, unvalidated on CCTV); a trained accident model; MFA; red-light / illegal-turn / helmet / seat-belt on this scene; audit expiry; multi-camera operation in the app; vehicle re-identification"],
    ["**External integrations not available**", "authorised vehicle registry (Vahan-style), official e-challan system, live alert channels (webhook/e-mail only as a test adapter), real CCTV/RTSP hardware"],
], [1.6, 4.9], "Capability status"))
A(H2("19.2 The limitations in detail"))
A(TABLE(["Limitation", "What it means for you"], [
    ["Camera calibration", "assumed lane-marking standard, site not surveyed; speed is approximate; scale uncertainty plausibly +/-10 % or more and not in the +/- band"],
    ["Tracking fragmentation", "5,552 tracklets in 20 min; ID counts are not vehicle counts; the median box is 28 px; nothing merges tracklets"],
    ["Unmeasured detector accuracy", "no labels for demo.mp4; MIO-TCD scores do not transfer; the fine-tuned model is not promoted"],
    ["Congestion proxy", "capacity 20 is assumed; one count for six lanes; dense but flowing traffic can read congested; the 60 s grace is a policy value from one video"],
    ["One incident per stopped vehicle", "14 incidents for one queue; they are grouped (`queue_group_id`) but not collapsed"],
    ["No accident model", "the collision candidate is a heuristic; zero real examples"],
    ["No plate reading on the demo footage", "plates are about 8 px wide, so the live plate reader (officer screen, needs vehicles >= 110 px wide) never fires there; incident-evidence OCR is not connected; manual entry only"],
    ["No real registry / challan", "sandbox only; operational approval is blocked by design"],
    ["No MFA", "accounts rely on scrypt passwords, throttling and the two-person rule"],
    ["Licensing", "Ultralytics AGPL-3.0; MIO-TCD CC BY-NC-SA (NonCommercial); demo.mp4 origin unknown (classroom use only); OCR weight licences unverified"],
    ["No real CCTV acceptance test", "RTSP, webcam and HTTP sources are tested only with fake captures; no real camera was ever connected"],
    ["Operations", "HTTP on localhost only; login counters in memory; no independent security review; one session per camera"],
], [1.7, 4.8], "Limitation checklist"))
A(H2("19.3 Realistic future improvements (in a sensible order)"))
A(NUM(["**Label about 200 frames** of the demo footage; then score pretrained vs fine-tuned, decide conf 0.10 and match_thresh 0.9, and measure tracking properly.",
       "**Survey two known distances** on a real road to turn approximate speed into a defensible measurement; then tune over-speed and braking on labelled examples.",
       "**Per-carriageway congestion** with a measured capacity, and one expandable queue incident instead of one per vehicle.",
       "**Higher-resolution, licensed footage** (plate width >= 60-80 px), then an Indian-plate detector and OCR with verified licences.",
       "**Real collision / near-miss clips** with a clear licence for a trained accident model, evaluated by incident.",
       "**MFA** with an approved library, TLS proxy, trusted-proxy throttling, an independent security review.",
       "**Real camera test** (RTSP) and an authorised registry / e-challan gateway through the department.",
       "**Legal decisions** on the AGPL licence and the NonCommercial dataset, or a permissively licensed detector."]))

# =============================================================== CH 20
A(H1("Chapter 20 - Quick Revision Sheets"))
A(H2("Sheet 1 - One-page project summary"))
A(BUL(["**What:** ANVESH turns a fixed-camera video into evidence-backed road-event **candidates** and walks an officer through a controlled, audited investigation. Research/demo only.",
       "**How:** YOLOv8n detects, ByteTrack tracks, a homography gives metres, detectors raise events, IncidentService stores one incident per event, evidence is SHA-256 hashed, a case state machine enforces four-eyes, the audit log is hash-chained.",
       "**Real in the demo:** detection/tracking, approximate speed, stopped-vehicle (14) and congestion (4) incidents, evidence, case workflow, audit. **Simulated:** wrong-way, collision, queue cameras; registry; challan.",
       "**Honest gaps:** no plate OCR, no real registry/challan, no trained accident model, no MFA, accuracy unmeasured, video licence unknown."]))
A(H2("Sheet 2 - Architecture in one breath"))
A(P("Video source -> decoder (paced 25 fps) -> bounded queue (8) -> **YOLOv8n + ByteTrack** (GPU) -> **TrackManager** (identity, ground point) -> **streaming engine** (store, 10 s windows) -> frame detectors and window detectors -> **IncidentService** -> evidence store -> **SQLite + audit chain** -> **HTTP API** (stdlib) -> **dashboard** (vanilla JS, polling + MJPEG) -> officer -> case -> registry (sandbox) -> challan (sandbox)."))
A(H2("Sheet 3 - The pipeline"))
A(TABLE(["Stage", "Thread", "Key numbers"], [
    ["decode + pace", "reader", "1.1 ms, 25 fps, RTF 1.0"], ["queue", "-", "capacity 8, BLOCK (file)"], ["detect + track", "worker", "24.9 ms p50, conf 0.25, NMS IoU 0.7"],
    ["identity + store + frame detectors", "worker", "0.1 ms store; ground point = bottom-centre"], ["window reasoning", "consumer", "10 s windows, 121 in 20 min"],
    ["incident + evidence + audit", "either", "ring 43 s, <= 10 fps JPEG, SHA-256"], ["dashboard", "browser", "poll 2.5-4 s, MJPEG"],
], [2.3, 1, 3], "Pipeline cheat-sheet"))
A(H2("Sheet 4 - YOLO and ByteTrack"))
A(BUL(["YOLO = single-pass detector; **v8n** = nano, ~3.2 M parameters; conf >= 0.25; NMS IoU 0.7; classes bicycle, car, motorcycle, bus, truck (COCO ids 1, 2, 3, 5, 7), mapped by name.",
       "IoU = overlap / union. ByteTrack: Kalman prediction -> stage 1 (high boxes, cost 1 - IoU x score <= 0.8) -> stage 2 (low boxes 0.1-0.25, IoU >= 0.5) -> unconfirmed (0.7) -> new track if score >= 0.25. Lost for 30 frames.",
       "IDs: raw ByteTrack int (first = 1, class-global counter, reset per tracker) -> ANVESH identity `<run>-s000-t<serial>` (serial from 0). **Tracklet, not vehicle.** 5,552 IDs in 20 min.",
       "Fine-tuned model: MIO-TCD mAP@50 0.215 -> 0.870, CityFlow recall 0.748 vs 0.718 -> **not promoted**."]))
A(H2("Sheet 5 - FPS and performance formulas"))
A(FORMULA(r"\mathrm{FPS}=\frac{\mathrm{frames}}{\mathrm{time}}=\frac{30001}{1200.66}=24.99\qquad \Delta t=\frac{1}{\mathrm{FPS}}=40.2\ \mathrm{ms}", None))
A(FORMULA(r"\mathrm{latency}=t_{\mathrm{done}}-t_{\mathrm{start}}", None))
A(P("30,001 frames, 0 dropped, 24.89 FPS (monitor), p50 27.4 ms, p95 38.9 ms, p99 48.1 ms, RSS 1,365-1,573 MB (+1.8 %), queue capacity 8. p95 < frame interval, so it keeps up. FPS != RTF = video seconds / wall seconds."))
A(H2("Sheet 6 - Speed and homography formulas"))
A(FORMULA(r"X=\frac{h_{11}u+h_{12}v+h_{13}}{h_{31}u+h_{32}v+h_{33}},\quad Y=\frac{h_{21}u+h_{22}v+h_{23}}{h_{31}u+h_{32}v+h_{33}}", None))
A(FORMULA(r"v_{\mathrm{km/h}}=3.6\,\frac{d}{t}", None))
A(P("H from 10 lane-dash points: fit RMSE 0.094 m, hold-out RMSE 0.43 m; speed = slope of least-squares line over 1.6 s, band = 2 x rms/sqrt(Stt) x 3.6 + 1 km/h. Speed zone rows >= 225. APPROXIMATE: assumed 9 m pitch and 3.65 m lanes, scale uncertainty not in the band."))
A(H2("Sheet 7 - Traffic-event logic"))
A(TABLE(["Event", "Rule (numbers from code)", "Demo"], [
    ["Stopped", "within 0.75 m of mean of last 3 s, for >= 3 s; long stop >= 30 s; queue -> low", "14 real (low, queue)"],
    ["Congestion", "10 s window: high if speed <= 3 m/s or occupancy >= 0.6 (concurrency/20); hysteresis; close after 60 s free flow", "4 real"],
    ["Wrong-way", "zone displacement >= 60 px, >= 1.5 s, cos <= -0.7", "SIMULATED"],
    ["Over-speed", "> limit + 5 km/h, lower band > limit, 2 s, in zone; limit 112.7", "none"],
    ["Sudden braking", "from >= 40 km/h, drop >= 25 km/h, >= 3.5 m/s^2 in 1.0-2.5 s", "none"],
    ["Aggressive", ">= 3 lane changes in 10 s", "none"],
    ["Collision", "contact IoU >= 0.10 for 1 s + closing + joint stop 2 s within 2.5 s; not queue (3) / standstill (60 %)", "SIMULATED"],
], [1.1, 4.3, 1.1], "Event rules"))
A(H2("Sheet 8 - Officer workflow"))
A(P("investigator: open incident -> **Open violation case** -> **Submit for evidence verification** -> reviewer: **Verify evidence** -> investigator: **Enter plate manually** -> reviewer2: **Verify plate** -> officer: **Look up registration (sandbox)** -> **Approve for submission** -> **Submit to SANDBOX simulator** (`TEST-CHALLAN-...`) -> admin: **Verify hash chain**. No override; official hand-off 409; operational approval blocked."))
A(H2("Sheet 9 - Technologies"))
A(P("Python 3.11.9 | Ultralytics 8.4.132 (YOLOv8n, ByteTrack) | PyTorch 2.13.0+cu130 (RTX 3050 6 GB Laptop) | OpenCV 5.0.0.93 | NumPy 2.4.6 | lap 0.5.13 | SQLite WAL | stdlib `ThreadingHTTPServer` | `hashlib` SHA-256 + scrypt | vanilla JS, polling + MJPEG | pytest 1,381 tests."))
A(H2("Sheet 10 - Limitations and future scope"))
A(P("Approximate speed; fragmenting IDs; unmeasured accuracy; proxy congestion; simulated wrong-way/collision; no OCR / registry / challan / accident model / MFA; unknown video licence; no real CCTV test.  **Next:** label 200 frames, survey 2 distances, per-carriageway congestion, licensed hi-res footage, MFA, real RTSP test."))
A(H2("Sheet 11 - The ten numbers to remember"))
A(TABLE(["#", "Number", "Meaning"], [
    ["1", "**30,001 / 0**", "frames processed / dropped in the 20-minute run"], ["2", "**24.89 FPS**", "average processing rate on a 25 fps source"], ["3", "**27.4 / 38.9 ms**", "latency p50 / p95 (frame interval 40.2 ms)"],
    ["4", "**5,552**", "track IDs (tracklets) in 20 min - not vehicles"], ["5", "**0.094 m / 0.43 m**", "calibration fit RMSE / hold-out RMSE"], ["6", "**0.25 / 30 / 0.8**", "ByteTrack conf, buffer frames, match threshold"],
    ["7", "**18**", "real incidents: 14 stopped + 4 congestion"], ["8", "**1,381**", "tests passed (23/23 walkthrough, 15/15 workflow, 12/12 UI, 36/36 layouts)"], ["9", "**0.75 m, 3 s**", "stopped-vehicle radius and dwell"],
    ["10", "**+1.8 %**", "memory growth over the run (1,460 -> 1,486 MB)"],
], [0.3, 1.5, 4.5], "Ten numbers"))
A(H2("Sheet 12 - A five-minute project explanation"))
A(CALLOUT("say", "**(0:00)** Cities have far more cameras than people to watch them, and CCTV only records; it does not tell an officer what happened or prove it. ANVESH is my attempt at a road-event investigation system built on a fixed camera video.\n"
          "**(0:45)** Each frame is read at video speed and passed to YOLOv8n, a small real-time detector, and ByteTrack, which links detections across frames and gives tracks. A homography calibrated from lane markings converts the bottom-centre of each box to metres, so I can estimate speed - approximate, because the markings are an assumption and the site was not surveyed.\n"
          "**(1:45)** On those tracks, detectors raise candidates: stopped vehicles, congestion, over-speed, sudden braking, wrong-way, and an experimental collision candidate. In my 20-minute real video the system finds 14 stopped vehicles - all queue members - and 4 congestion episodes. Events that never occur in the footage, like wrong-way and collision, I show on clearly labelled simulated cameras.\n"
          "**(2:45)** Every incident keeps original frames and an annotated clip with SHA-256 hashes. An investigator opens a case, a reviewer verifies the evidence, a second reviewer verifies the manually entered plate, and an officer approves and submits to a sandbox challan simulator. Every step is in a hash-chained audit log I can verify live.\n"
          "**(3:45)** It runs frame by frame with bounded queues: 30,001 frames, none dropped, 24.89 FPS, 27 ms median latency, and memory flat over 20 minutes. What it does not do: read plates, talk to a real registry or challan system, use a trained accident model, or support MFA. Detector accuracy on this video is unmeasured.\n"
          "**(4:30)** Next steps are labelled footage, a surveyed calibration, per-carriageway congestion and licensed higher-resolution video. It is a research prototype and says so on every screen.", "Five-minute explanation (read it aloud once)"))

glossary = [
    ["ANPR", "Automatic Number Plate Recognition: detect a plate and read its characters."], ["Audit log", "append-only record of actions; each row hashes the previous one."], ["Back-pressure", "a full queue makes the producer wait."],
    ["Bounded queue", "queue with a maximum size (frames 8, windows 16)."], ["Bounding box", "rectangle (x1, y1, x2, y2) around a detection."], ["ByteTrack", "tracking-by-detection method that also uses low-score boxes."],
    ["Calibration", "finding the mapping between image pixels and road metres."], ["Candidate", "an automatic suggestion for human review, not a finding."], ["Confidence", "model score 0-1 for a detection; also a heuristic tier label for incidents."],
    ["Congestion hysteresis", "state machine that needs confirmation before changing level."], ["CUDA / GPU", "NVIDIA parallel computing platform used by PyTorch."], ["Dempster-Shafer", "method to combine uncertain evidence with belief and plausibility."],
    ["Dwell", "time a vehicle stays stopped (3 s required)."], ["Evidence asset", "stored file for an incident with a SHA-256."], ["FPS", "frames per second."], ["Ground point", "bottom-centre of a box, the estimated contact with the road."],
    ["Homography", "3 x 3 matrix mapping image plane to road plane."], ["IoU", "Intersection over Union of two boxes."], ["Kalman filter", "predicts a track's next position from its motion and corrects it with each detection."],
    ["Latency", "completion time minus start time for a frame."], ["MFA", "multi-factor authentication (not implemented)."], ["MJPEG", "video as a stream of JPEG images; used for the live view."],
    ["mAP", "mean average precision; @50 uses IoU 0.50, @50-95 averages 0.50-0.95."], ["NMS", "Non-Maximum Suppression of overlapping boxes."], ["OCR", "Optical Character Recognition (not installed)."], ["Occupancy proxy", "mean concurrency / assumed capacity (20)."],
    ["Queue member", "stopped vehicle judged part of queued traffic."], ["Raw track ID", "integer from ByteTrack's counter."], ["Real-time factor", "video seconds processed per wall second."], ["RMSE", "root-mean-square error."],
    ["RTSP", "Real Time Streaming Protocol for IP cameras."], ["Sandbox", "synthetic test registry / challan simulator; nothing real is contacted."], ["scrypt", "memory-hard password hashing function."], ["SHA-256", "256-bit cryptographic hash."],
    ["Simulated", "scripted boxes from a synthetic camera; labelled SIMULATED."], ["Speed zone", "near-field polygon where km/h is shown and judged."], ["Tracklet", "one uninterrupted tracker run; what an ANVESH 'track ID' names."], ["WAL", "SQLite write-ahead-logging journal mode."],
    ["Window", "10 s slice of the timeline used for traffic state and stop episodes."], ["YOLOv8n", "Ultralytics YOLO version 8, nano size."],
]

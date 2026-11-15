from docbuilder import *

blocks = []
A = blocks.append
n = [0]


def Q(cat, pairs):
    A(H3(cat))
    out = []
    for q, a in pairs:
        n[0] += 1
        out.append((f"Q{n[0]}. {q}", a))
    A(QA(out))


A(H1("14. Technical viva question bank"))
A(P("Each answer starts with the **direct answer**, then the reason. *Follow-up:* shows the question an examiner often asks next. Numbers are the verified ones from Sections 4-13."))

Q("1. Project basics", [
    ("What is ANVESH in one sentence?", "A system that processes fixed-camera traffic video live, finds vehicle events (stopped vehicle, jam, wrong-way, possible collision), saves hashed evidence and lets an officer review them. *Follow-up: is it finished?* It is a finished research/demo system, not production."),
    ("What exactly happens when a video enters the system?", "A session thread decodes frames into a bounded queue; YOLOv8n detects vehicles and ByteTrack tracks them; TrackManager gives session IDs; analytics measure lanes, speed, stops and congestion; incidents are saved with SHA-256-hashed evidence; the browser shows the annotated frames as MJPEG and polls JSON for tiles."),
    ("Is the video really processed live or are results pre-computed?", "Live. The server decodes at video speed and runs the model on each frame; the frame counter rises with the clock. Not live: the sandbox registry/challan (synthetic) and finished-session history (read from SQLite)."),
])
Q("2. Problem statement and novelty", [
    ("What problem does it solve?", "Cameras record but do not explain; officers must search video, work out what happened and keep disputable-proof evidence. ANVESH automates the finding and preserves the proof."),
    ("What is new in your approach?", "Not a new detector. The integration: candidate-cause reasoning that can abstain (Dempster-Shafer + reliability), evidence integrity with an audit chain and four-eyes approval, and honesty gates (no km/h without calibration, SIMULATED labels). We reuse existing YOLO/ByteTrack/DS mathematics."),
    ("Who would use it?", "Traffic-police control rooms, highway operators, accident investigators, road-safety researchers."),
])
Q("3. System architecture", [
    ("Draw the architecture.", "Source -> frame queue -> YOLOv8n -> ByteTrack -> TrackManager -> analytics (+10 s windows) -> incidents/evidence (SQLite) -> HTTP API (JSON + MJPEG) -> officer screen / dashboard. Evidence builder -> reliability -> DS fusion -> ranking sits beside the incident step."),
    ("Why a bounded queue of 8 with 'block'?", "To cap memory and apply backpressure: if processing is slower than the video, the reader waits instead of buffering forever. Cost: the session lags the video (no frame is dropped). *Follow-up: why not drop frames?* Dropping breaks speed windows and stop timing."),
    ("Why one thread per session?", "Simple isolation: each session has its own source, tracker, track history and `last_jpeg`; the HTTP server uses a thread per request."),
])
Q("4. Python and programming", [
    ("Why Python?", "Best ML/video ecosystem and fastest to develop and test; heavy work runs in compiled code (PyTorch/OpenCV on GPU)."),
    ("How do you avoid blocking the video while the UI is busy?", "Processing runs in its own session thread; HTTP handlers only read the latest results; plate OCR runs in a separate thread with a queue of 2."),
    ("Why the standard-library web server, not FastAPI?", "No extra dependency, small attack surface, enough for JSON + MJPEG. Trade-off: no async, hand-written routing; for many cameras/users we would move to an async framework."),
])
Q("5. Computer vision fundamentals", [
    ("Classification vs detection?", "Classification gives one label for the whole image; detection gives a box, class and confidence for every object."),
    ("What is a bounding box, confidence and class id?", "Box = `[x1,y1,x2,y2]` pixels around the object; confidence = network's certainty (0-1); class id = integer label (COCO: 2 = car)."),
    ("What is IoU?", "Intersection area divided by union area of two boxes (0-1). Used by NMS (remove duplicates, threshold 0.7) and by the tracker (match boxes)."),
    ("What is Non-Maximum Suppression?", "Keep the best-scoring box among heavily overlapping ones of the same class, delete the rest, repeat."),
    ("What is the confidence threshold and why 0.25?", "Boxes below it are dropped. 0.25 is the Ultralytics default for prediction; we did not tune it. Lower (0.10) added boxes that inflated the congestion proxy, so we kept 0.25."),
])
Q("6. YOLO and object detection", [
    ("Which YOLO exactly?", "YOLOv8n (version 8, nano), COCO-pretrained, 3,157,200 parameters, `models/yolov8n.pt`."),
    ("What does the 'n' mean and why nano?", "The smallest size (n < s < m < l < x). It runs in ~10-28 ms per frame on a laptop GPU, which real time needs; larger sizes are more accurate but slower."),
    ("Why YOLO and not Faster R-CNN?", "YOLO is single-stage and fast; Faster R-CNN is two-stage, often more accurate on small objects but usually much slower per frame. We did not benchmark R-CNN ourselves."),
    ("Which classes do you use?", "COCO bicycle, car, motorcycle, bus, truck (ids 1, 2, 3, 5, 7). Pedestrians are excluded by the `classes=` filter."),
    ("What input size?", "640 (Ultralytics default); a 640x360 frame is letterboxed (aspect kept, padded to a multiple of 32)."),
])
Q("7. Model training and inference", [
    ("Did you train this model yourself? Why?", "The detector we run is **pretrained** (COCO). We only fine-tuned a copy as an experiment (not used). Training from scratch needs big labelled data and weeks of GPU; our contribution is the reasoning on top."),
    ("How many epochs / hours / what dataset did you train?", "Fine-tune experiment: 8 epochs, 2.97 h, MIO-TCD (88,270 train / 10,865 val / 10,865 test), AdamW lr 0.001, batch 16, imgsz 640, RTX 3050 laptop. The base model's COCO training details are not recorded in our repo. Plate detector: 12 epochs on 6,176 train images."),
    ("What accuracy did the fine-tune get and why not use it?", "MIO-TCD test mAP@50 0.870 vs 0.215 pretrained, but on real CCTV (CityFlow) recall was 0.718 vs 0.748, so it did not generalise better. Not promoted."),
    ("Difference between training, fine-tuning and inference?", "Training learns weights; fine-tuning continues training from pretrained weights on new data; inference just runs the finished model. ANVESH does inference."),
    ("What is the detector's accuracy on your demo video?", "**Not measured** - the video has no labels. We only have indirect checks and published numbers on other datasets."),
])
Q("8. Object tracking and ID assignment", [
    ("How does the tracker assign IDs?", "ByteTrack matches each new detection to a predicted track by IoU (Kalman prediction + optimal assignment). Matched keeps its ID; an unmatched confident detection starts a new track with the next integer. ANVESH then maps it to a session serial `SES-...-t00001`."),
    ("Detection vs tracking?", "Detection is per frame with no memory; tracking links boxes across frames so a vehicle keeps one ID."),
    ("Why can IDs switch?", "ByteTrack uses motion and overlap only. If boxes overlap or cross, the predicted box may match the wrong vehicle; if a vehicle is hidden > 30 frames its track is deleted and it returns with a new ID."),
    ("What happens when a car is hidden behind another?", "Its track is marked lost and kept up to 30 frames (1.2 s at 25 fps) and predicted forward; if re-detected near the prediction it gets the same ID, otherwise a new one."),
    ("Why do ID numbers look limited / how many IDs can there be?", "No cap. The counter keeps increasing (5,552 tracks in 20 min). Raw ByteTrack IDs restart at 1 on reset and have gaps; `t00001` is a minimum width, not a limit."),
    ("Is the track ID the plate number?", "No. It is a temporary label for one tracklet inside one session. The plate is a separate record entered or read and verified by an officer."),
    ("Does the same ID mean the same vehicle across cameras?", "No. ANVESH does not re-identify vehicles; identities are per session and per camera."),
])
Q("9. OpenCV and video processing", [
    ("What does OpenCV do here?", "Decodes video, draws boxes/lanes/banners, encodes JPEG for the stream, fits the homography (`cv2.findHomography`)."),
    ("How do you get the frame time?", "From the source timestamp (frame index / fps for files); speed and stop timing use these timestamps, not wall-clock."),
    ("What if the video is very large or high resolution?", "Per-frame cost grows with resolution (YOLO letterboxes to 640 though); a long video only takes longer. Frames are never all held in memory (bounded queue, ~13.5 MB frame ring)."),
])
Q("10. Speed estimation and formulas", [
    ("How do you calculate speed?", "Bottom-centre ground point -> homography to metres -> least-squares slope of position vs time over the latest samples inside the calibrated zone (up to 1.6 s, in practice ~0.5-0.7 s at motorway speed) -> `3.6 x sqrt(vx^2 + vy^2)` km/h, outliers removed, no extra smoothing."),
    ("Why not pixel distance / FPS?", "Pixel size depends on distance (0.42 m/px far vs 0.10 m/px near in our calibration), so pixels do not give metres."),
    ("How does FPS affect speed?", "Not the formula: we use timestamps. FPS only changes the number of samples in the window (about 25 per second of video at 25 fps)."),
    ("Convert m/s to km/h?", "Multiply by 3.6 (3600 s per hour / 1000 m per km). 22.5 m/s = 81 km/h."),
    ("How accurate is it? Validated?", "Not validated: no ground-truth speed. After the audit fix the median is 94.9 km/h and trucks 88 km/h - plausibility only. Scale is assumed from lane-marking standard; error plausibly +/-10 % or more beyond the displayed band."),
    ("What if the camera angle changes?", "The homography is no longer valid and all speeds are wrong until recalibrated; the system does not detect this automatically."),
])
Q("11. Homography and camera calibration", [
    ("What is a homography?", "A 3x3 matrix that maps points of one plane (the road in the image) to another plane (top-down road in metres)."),
    ("What is perspective distortion?", "Equal real distances look smaller far from the camera."),
    ("How did you calibrate?", "Measured 10 lane-dash points in the image, assigned metres from the UK 2 m + 7 m marking standard, fitted H by least squares (RMSE 0.094 m); 4 hold-out points within 0.43 m. The scale itself is assumed."),
    ("Why only a 'speed zone'?", "Far-field pixels are coarse and the plane assumption is weaker; trusted region is rows >= 225 of the toward-camera carriageway."),
])
Q("12. Congestion detection", [
    ("How do you define congestion?", "A 10 s window is 'high pressure' if mean speed <= 3 m/s or occupancy >= 0.6, where occupancy = mean concurrency / 20 (assumed capacity). Hysteresis smooths it into free/building/congested/dissipating."),
    ("Count or speed based?", "Both when speed is calibrated; occupancy (count-based) only when speed is unavailable. On our demo it is effectively count-based, so it is a proxy."),
    ("Traffic jam vs normal slowing? Red light?", "We cannot fully distinguish. Queue context lowers stop severity; signals are not read from video (configured/manual only)."),
    ("How are pedestrians excluded?", "Only bicycle, car, motorcycle, bus, truck classes are kept; 'person' is filtered out."),
    ("Can it detect wrong-side driving?", "Yes, with a configured direction zone; tested only on simulated scenes (none in the real footage)."),
])
Q("13. Evidence fusion", [
    ("What is evidence fusion and why?", "Combining several uncertain pieces of evidence into one assessment. One observation can be wrong or missing, and we need to track uncertainty explicitly."),
    ("How does your system handle conflicting evidence?", "Dempster's conflict K is reported; if K is near 1 we do not divide, return pure ignorance and flag HIGH_CONFLICT; a shared-condition blend stops two cameras from over-agreeing."),
    ("What do H1-H7 mean?", "H1 bottleneck, H2 stopped vehicle, H3 collision-like obstruction, H4 non-incident lane blockage, H5 demand surge, H6 signal restriction, H7 = insufficient evidence (the ignorance mass)."),
    ("Is the top hypothesis the true cause?", "No. It is the best-supported among testable hypotheses; belief is not probability; only an officer can establish a cause."),
])
Q("14. Dempster-Shafer theory", [
    ("Why Dempster-Shafer, not Bayes?", "DS can say 'unknown' without forcing probabilities to sum over all causes and without priors we do not have; ignorance is explicit. Cost: scores are heuristic, and the rule has known conflict problems."),
    ("What is a belief mass, frame of discernment, Bel and Pl?", "Mass m(H) = evidence committed exactly to H; frame = the set {H1..H6}; Bel(H) = m(H); Pl(H) = m(H) + m(Theta) = how much is not ruled out."),
    ("Write the combination rule.", "K = sum_{p != q} m1(Hp) m2(Hq); m(Hk) = [m1(Hk)m2(Hk) + m1(Hk)m2(Theta) + m1(Theta)m2(Hk)] / (1-K); m(Theta) = m1(Theta)m2(Theta)/(1-K)."),
    ("What is discounting?", "Moving mass to ignorance for an unreliable camera: m'(H) = r m(H); m'(Theta) = 1 - r(1 - m(Theta))."),
    ("Is cross-camera fusion running in the app?", "Implemented and unit-tested; the app runs one camera per session, so the shipped product uses the single-camera discounted path."),
])
Q("15. Backend and frontend", [
    ("Which backend and frontend?", "Python `http.server` (threaded) + SQLite WAL; HTML/CSS/vanilla JavaScript. No FastAPI/React/WebSocket/SSE."),
    ("How does the officer screen get updates?", "Polling `GET /sessions/{id}/officer` every 1 s for tiles/alerts, and an MJPEG `<img>` for video."),
    ("How do pause, resume and stop work?", "POST pause/resume/stop on the session; pause only for recorded files; stop closes the source and joins the thread; the page shows PAUSED/Stopped."),
    ("How do you keep sessions isolated?", "Separate session id, thread, tracker and `last_jpeg`; streams looked up by id; the officer page drops stale replies and serialises starts."),
])
Q("16. APIs and video streaming", [
    ("What is MJPEG and why use it?", "Motion JPEG: one HTTP response of consecutive full JPEGs shown in an `<img>`. No player or codec needed; each frame independent; fine for a demo. Less efficient than H.264."),
    ("What if the browser closes?", "The stream handler gets BrokenPipe/ConnectionReset and exits, freeing the thread; the processing session continues."),
    ("What if the stream drops?", "The page reconnects up to 3 times while the session is running."),
])
Q("17. FPS and performance", [
    ("What is the FPS?", "Video 25 fps. Real-time run: 21.77 fps average, 30,001 frames, 0 dropped; 25 fps kept for the first ~740 s. Fast mode: 41.4 fps (not real time)."),
    ("FPS vs latency?", "FPS = throughput (frames finished per second); latency = time one frame takes. Frame processing p50 34.9 ms, p95 71.7 ms."),
    ("What are p50 and p95?", "Half / 95 % of frames were faster than that value; p95 shows the slow tail."),
    ("Is it real time?", "Not for the whole 20 minutes in our measurement: it fell behind after ~12 min (mean 38.8 ms vs 40 ms budget with a long tail). Cause not established (shared laptop, probably battery)."),
    ("GPU or CPU? Memory?", "Detection on the GPU (RTX 3050 6 GB); peak RSS 1,745 MB; GPU memory reading not cross-checked."),
])
Q("18. Testing", [
    ("How many tests and what do they prove?", "1,403 pass. They prove the code meets its specification on synthetic/scripted cases; they do not prove speed or congestion accuracy."),
    ("How do you test streaming and session isolation?", "Real-socket tests: whole ordered frames, stream ends on stop, two sessions never share frames, threads freed after abrupt disconnects, 404/401."),
])
Q("19. Reliability and security", [
    ("How is evidence protected?", "SHA-256 of every file; a *separate* verify route re-hashes; exports only include files that still match; a hash-chained audit log; four-eyes approval."),
    ("How are passwords stored?", "scrypt hashes, lockout, hashed session tokens; the demo one-click sign-in exists only on loopback."),
])
Q("20. Limitations and future scope", [
    ("Biggest limitation?", "No ground truth: speed, congestion and cause ranking are unvalidated, and speed rests on an assumed scale."),
    ("Most technically challenging part?", "Making identities, timing and uncertainty honest: tracklet identities, time-based occlusion, calibrated-only speed and abstaining ranking."),
    ("Is it production-ready?", "No - a tested research/demo system; needs labelled data, per-camera validation, dedicated hardware, MFA and legal approvals."),
    ("What would you do next?", "Label real frames, survey the calibration, rerun the benchmark on a plugged-in idle machine, calibrate the DS scores, multi-camera operation."),
])

# ====================================================================== 15 REVISION
A(H1("15. Quick revision sheet"))
A(TABLE(["Item", "Remember"], [
    ["**Definition**", "ANVESH turns fixed-camera video into reviewable, evidence-backed incident candidates with honest uncertainty; an officer is the final authority. Research/demo, not production."],
    ["**Pipeline (one line)**", "video -> frame queue(8) -> YOLOv8n -> ByteTrack -> TrackManager -> analytics (+10 s windows) -> incidents + SHA-256 evidence + audit -> API (JSON + MJPEG) -> officer screen / dashboard"],
    ["**Model / tracker**", "**YOLOv8n** COCO-pretrained (3.16 M params, conf 0.25, imgsz 640, NMS IoU 0.7, classes 1,2,3,5,7); **ByteTrack** (buffer 30 frames, match 0.8, new-track 0.25)"],
    ["**Formulas**", "IoU = overlap / union;  (x,y,w)=H(u,v,1), X=x/w;  v = LSQ slope over in-zone samples;  km/h = 3.6 sqrt(vx^2+vy^2);  occupancy = min(1, concurrency/20);  discount m' = r m, m'(Theta)=1-r(1-m(Theta));  Dempster K and /(1-K);  Bel=m, Pl=m+m(Theta)"],
    ["**IDs**", "ByteTrack int counter -> session serial `SES-..-t00001`; tracklet, not vehicle; no cap; new ID if lost > 30 frames; plate number is separate"],
    ["**Benchmark (20 min real time)**", "30,001 frames, 0 dropped, 21.77 fps (source 25), wall 1,377.9 s, frame p50/p95 34.9/71.7 ms, peak RSS 1,745 MB. 25 fps kept ~740 s then fell behind; cause not established. Fast mode 41.4 fps (150 s). Session 7's 24.89 fps not reproduced"],
    ["**Tests**", "1,403 passed (9 groups); 12-check browser lifecycle; 1,381 at the previous presentation"],
    ["**Main contribution**", "Integrated, honest candidate-cause reasoning (DS, can abstain) + tamper-evident evidence workflow + live officer screen"],
    ["**Improvements since 2 Nov**", "Officer screen, jam-cause candidate, plate reader, status line, pause/stop states, stale-poll guard, stream reconnect, serialised starts, MJPEG tests, benchmark, docs, everything committed and pushed"],
    ["**Top 5 limitations**", "1 no ground truth for speed/congestion/causes; 2 assumed speed scale; 3 congestion is a proxy; 4 detector accuracy unmeasured, IDs fragment; 5 real time not kept to the end on this laptop, one camera per session, sandbox registry"],
    ["**Top 5 future work**", "1 label data and validate; 2 survey calibration; 3 clean benchmark on dedicated hardware; 4 calibrate fusion scores, multi-camera; 5 MFA, HTTPS, licences, authorised registry"],
], [1.5, 5.3], "One-page revision"))
A(H2("Glossary (simple English)"))
A(TABLE(["Term", "Meaning"], [
    ["Bounding box", "Rectangle around an object: left, top, right, bottom pixels."], ["Confidence", "How sure the model is about a detection (0-1)."],
    ["IoU", "Overlap divided by combined area of two boxes."], ["NMS", "Removes duplicate overlapping boxes, keeps the best."],
    ["Kalman filter", "Predicts where a moving box will be next from its past motion."], ["Hungarian / assignment", "Finds the best one-to-one matching of tracks to detections."],
    ["Tracklet", "One uninterrupted run of a track; may be only part of a vehicle's journey."], ["Homography", "3x3 matrix mapping a plane in the image to a plane in the world."],
    ["Ground point", "Bottom centre of a box: where the vehicle touches the road."], ["Occupancy proxy", "Count-based estimate of how full the road view is; not measured road occupancy."],
    ["Hysteresis", "Needs sustained change before a state flips; avoids flicker."], ["Belief mass", "Share of evidence committed to one hypothesis."],
    ["Ignorance m(Theta)", "Evidence not assigned to any hypothesis: 'unknown'."], ["Conflict K", "Mass two sources put on contradicting hypotheses."],
    ["MJPEG", "Stream of JPEG images in one HTTP response."], ["FPS / latency / p95", "Frames per second / time for one frame / 95th percentile."],
    ["Fine-tuning", "Continue training a pretrained model on new data."], ["Ground truth", "Independently measured correct answers used to score a system."],
], [1.7, 5.1], "Glossary"))

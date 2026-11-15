from pathlib import Path
from docbuilder import *

FIGS = Path(__file__).resolve().parent / "figures"
F = lambda n: FIGS / n
blocks = []
A = blocks.append

# =============================================================== CH 5
A(H1("Chapter 5 - YOLO: Vehicle Detection"))
A(H2("5.1 The ideas, in simple words"))
A(TABLE(["Term", "Full form / meaning", "Why ANVESH needs it"], [
    ["**Object detection**", "find *what* objects are in an image and *where* (a box around each)", "to find vehicles on every frame"],
    ["**YOLO**", "*You Only Look Once* - one neural-network pass over the whole image predicts all boxes, so it is fast", "real-time operation at 25 fps on a laptop GPU"],
    ["**YOLOv8n**", "version 8 of YOLO from Ultralytics; **n = nano**, the smallest of the sizes n, s, m, l, x (about 3.16 million parameters in our file)", "smallest and fastest; leaves GPU time for tracking and encoding"],
    ["**Bounding box**", "`(x1, y1, x2, y2)` in pixels", "position and size of a vehicle"],
    ["**Class probability / confidence**", "the model's score (0 to 1) that the box contains a given class", "weak boxes (< 0.25) are dropped"],
    ["**IoU**", "*Intersection over Union* = overlap area / combined area of two boxes", "used by NMS, by ByteTrack matching, and to score a detector"],
    ["**NMS**", "*Non-Maximum Suppression* - keep the best of several overlapping boxes", "one box per vehicle"],
    ["**Precision / recall**", "precision = real among drawn; recall = found among real", "how to describe detector quality"],
    ["**mAP@50 / mAP@50-95**", "*mean average precision* at IoU 0.50, or averaged over 0.50-0.95", "the standard score on the MIO-TCD test set"],
], [1.3, 3, 2.2], "Detector vocabulary"))
A(FIG(F("fig_yolo.png"), 1.0, "What the detector does to one frame and what a detection looks like."))
A(FIG(F("fig_iou_nms.png"), 1.0, "IoU, NMS and the quality words used in the training report."))
A(FORMULA(r"\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}", "IoU. Worked example: two 10 x 8 boxes overlapping 5 x 5: 25 / (80 + 80 - 25) = 0.185."))
A(H2("5.2 What the model receives and produces"))
A(BUL(["**Input:** one BGR image (640 x 360 in the demo). Ultralytics resizes and pads it to a network input size (default `imgsz` 640; ANVESH does not override it).",
       "**Raw output:** thousands of candidate boxes, each with a score per class. **After** the confidence filter (0.25) and NMS (IoU 0.7, Ultralytics default; `max_det` 300) the final list remains.",
       "**Per detection:** box `(x1, y1, x2, y2)`, class index, confidence. ANVESH wraps it as `RawDetection(frame_index, timestamp, class_name, vehicle_class, confidence, bbox_xyxy)`.",
       "**Ground point:** `RawDetection.ground_point` = `(x1 + width/2, y2)`, the bottom-centre of the box, used for lanes, speed and stop detection (Chapter 7)."]))
A(H2("5.3 The actual ANVESH detector implementation"))
A(P("The detector is called through the tracker: `ByteTrackYoloTracker.update()` calls `model.track(image, persist=True, tracker='bytetrack.yaml', classes=<ids>, conf=0.25, verbose=False)` (`anvesh/perception/tracking.py`). "
    "A stand-alone `YoloDetector` (predict only) also exists but the application does not use it."))
A(P("**Class mapping.** `ClassMap.from_model_names(model.names)` matches class **names** (case-insensitive) with ANVESH's `VehicleClass` values, never hard-coded indices, because the two checkpoints number classes differently:"))
A(TABLE(["ANVESH class", "COCO pretrained (default) index", "Fine-tuned MIO-TCD index"], [
    ["bicycle", "1", "0"], ["car", "2", "1"], ["motorcycle", "3", "2"], ["bus", "5", "3"], ["truck", "7", "4"],
    ["(ignored) person, traffic light, ... / `other_vehicle`", "other COCO classes", "5 `other_vehicle` (trained but ignored by ClassMap)"],
    ["auto_rickshaw", "no COCO equivalent", "only a checkpoint trained with that class could produce it - none exists here"],
], [1.7, 1.7, 1.9], "Class mapping resolved by name"))
A(P("In the 20-minute run only three classes appeared: car 4,339 tracks, truck 714, bus 499. Lorries are sometimes called \"bus\" and the class of one track can flicker."))
A(H2("5.4 Which checkpoint is the default?"))
A(TABLE(["Checkpoint", "File", "SHA-256 (start)", "Status"], [
    ["COCO-pretrained YOLOv8n", "`models/yolov8n.pt`", "`f59b3d83...ccfc83b36`", "**default** (`anvesh/app/config.py`)"],
    ["Fine-tuned on MIO-TCD (best)", "`models/finetuned/anvesh_mio5_yolov8n_best.pt`", "`cdc3cf10...ba7f73bf21`", "**not promoted**; opt-in via `ANVESH_FINETUNED_MODEL` or `run_demo.py --detector finetuned`"],
    ["Fine-tuned (last epoch)", "`..._last.pt`", "`0ca32c41...d035119`", "kept for reference"],
    ["YOLO26n", "`weights/yolo26n.pt`", "`9b09cc8b...`", "present, not used"],
], [1.5, 2.1, 1.3, 2], "Checkpoint inventory (all unchanged during the presentation work)"))
A(P("ANVESH never downloads weights and never silently falls back: a configured but missing fine-tuned path is an error (`anvesh/perception/weights.py`). Evidence records carry `sha256-<12 hex>` of the checkpoint actually used."))
A(H2("5.5 Fine-tuning: dataset, classes, split, results"))
A(TABLE(["Item", "Value (from `docs/ANVESH_MODEL_TRAINING_REPORT.md`)"], [
    ["Dataset", "MIO-TCD localization; converted by `scripts/prepare_mio_tcd.py`"],
    ["Split", "88,270 train / 10,865 validation / 10,865 test images"],
    ["Training classes", "bicycle, car, motorcycle, bus, truck, other_vehicle (6)"],
    ["Run", "8 of 8 epochs, 2.97 h, `--batch 16 --imgsz 640 --patience 4 --close-mosaic 2 --workers 2`; **not converged** (val mAP@50 still rising)"],
    ["Licence", "dataset **CC BY-NC-SA 4.0** (non-commercial; share-alike); Ultralytics code/weights AGPL-3.0 or a paid enterprise licence"],
    ["Held-out test (evaluated once, lock file)", "mAP@50 **0.215 -> 0.870**; mAP@50-95 **0.142 -> 0.651** (pretrained -> fine-tuned, 5-class taxonomy)"],
    ["CityFlowV2 S02 real CCTV (recall at IoU 0.5, no labels for class)", "pooled recall **0.748 pretrained vs 0.718 fine-tuned** - not better; wins 1 of 4 cameras"],
], [1.4, 4.2], "Training and evaluation facts"))
A(H2("5.6 Why the fine-tuned model was not promoted"))
A(NUM(["MIO-TCD scores measure agreement with MIO-TCD labels on MIO-TCD-style images, not with ANVESH's cameras. MIO-TCD also defines \"truck\" and \"car\" differently from COCO, which penalises the pretrained baseline for naming, not only for seeing.",
       "On real CCTV (CityFlow) recall did not improve (0.718 vs 0.748).",
       "There is no labelled footage of the ANVESH demo video, so no accuracy comparison is possible.",
       "No downstream comparison (tracking / speed / incidents) and no valid end-to-end latency comparison existed when the decision was made.",
       "An unexplained metric gap against Ultralytics' own `val()` (later traced to multi-label NMS on validation) and a licence question (NonCommercial data) remain."]))
A(H3("What the two detectors did on the demonstration footage (proxy evidence only)"))
A(TABLE(["Proxy (180 s of demo.mp4, 60-240 s, no labels)", "Pretrained", "Fine-tuned"], [
    ["Mean detections per frame", "12.1", "12.6"], ["Distinct track IDs", "808", "615"], ["Tracks shorter than 1 s", "361", "151"], ["Median track length", "1.96 s", "2.84 s"], ["Mean confidence", "0.598", "0.704"],
], [3, 1.2, 1.2], "Session 6 comparison (`demo/runs/detector_compare.json`)"))
A(P("The fine-tuned model produced fewer, longer tracks and more confident boxes, and a visual side-by-side looked similar. **But** a 300 s application run with it changed the incident counts (19 stopped + 4 congestion versus 14 + 1 for the same period). "
    "Without labels we cannot say whether those extra events are correct, so the default was left unchanged."))
A(CALLOUT("warn", "These are *proxy* numbers. Confidence is the model's own score, not accuracy; fewer IDs can mean better continuity **or** wrongly merged vehicles. Never say \"the fine-tuned model is more accurate on our video\".", "How not to misuse this table"))

# =============================================================== CH 6
A(H1("Chapter 6 - ByteTrack and Vehicle IDs"))
A(H2("6.1 Why detection alone is not enough"))
A(P("YOLO looks at each frame **independently**. Frame 100 says \"a car at (210, 190)\" and frame 101 says \"a car at (212, 191)\" - nothing says these are the same car. "
    "To measure speed, dwell time, lane changes or how long a vehicle has been stopped, ANVESH needs the **same object linked over time**. That linking is **tracking**; the label it produces is the **track ID**."))
A(H2("6.2 What ByteTrack does"))
A(P("**ByteTrack** (Zhang et al., 2022; \"associating every detection box\") is a tracking-by-detection method. ANVESH uses the implementation built into Ultralytics 8.4.132 (`ultralytics/trackers/byte_tracker.py`, `bytetrack.yaml`); no custom tracking algorithm is written in ANVESH."))
A(FIG(F("fig_tracking.png"), 0.95, "Top: one vehicle, lost for two frames and re-found. Bottom: the stages of ByteTrack association in this Ultralytics version."))
A(H3("Step by step, with numbers"))
A(NUM(["**Predict.** Every existing track has a **Kalman filter** (`KalmanFilterXYAH`): an 8-number state *(x, y, aspect ratio, height, and their velocities)* with a constant-velocity motion model. Before matching it predicts where each box should be *now*.",
       "**Split detections.** Boxes with score >= `track_high_thresh` (0.25) are *high*; boxes between `track_low_thresh` (0.1) and 0.25 are *low*.",
       "**Stage 1 matching** (tracks vs *high* boxes). Build a cost matrix with `cost = 1 - IoU x score` (because `fuse_score: True`) and solve the best one-to-one assignment with `lap.lapjv`. A pair is accepted only if its cost <= `match_thresh` (0.8), i.e. IoU x score >= 0.2.",
       "**Stage 2 matching** (still-unmatched tracks vs *low* boxes) using IoU only, accepted if IoU >= 0.5. This is ByteTrack's idea: a weak box is useful when it overlaps an existing track.",
       "**Unconfirmed tracks** (born last frame) are matched against remaining high boxes with a looser threshold 0.7; if they fail they are removed - so a one-frame false detection never becomes an output track.",
       "**New tracks.** A leftover *high* box with score >= `new_track_thresh` (0.25) starts a new track and receives the next raw ID. It is output only after it is matched once more (except on the very first frame).",
       "**Lost tracks.** A track with no match becomes *Lost* but keeps being predicted. If it is matched again within `track_buffer` (30) frames it keeps its ID; after 30 frames (1.2 s at 25 fps) it is removed."]))
A(P("**Worked example of one match.** A track predicts the box `(100, 100, 140, 130)` (40 x 30 = 1200 px^2). The detector reports `(104, 101, 145, 131)` with score 0.80 (41 x 30 = 1230 px^2). "
    "The overlap is x 104..140 (36) by y 101..130 (29) = 1044 px^2; the union is 1200 + 1230 - 1044 = 1386; IoU = **0.753**. Cost = 1 - 0.753 x 0.80 = **0.398**, which is <= 0.8, so the detection is assigned to that track and the Kalman filter is corrected with it."))
A(CALLOUT("def", "**Track states in the library:** New (just created) -> Tracked (matched) -> Lost (not matched, kept alive) -> Removed. \"Confirmed\" in this document means `is_activated = True`: matched at least a second time, which is the only condition under which ByteTrack reports an ID.", "States"))
A(TABLE(["Setting (from `bytetrack.yaml`, unchanged)", "Value", "Meaning"], [
    ["`track_high_thresh`", "0.25", "stage-1 score threshold"], ["`track_low_thresh`", "0.1", "lowest score used in stage 2"], ["`new_track_thresh`", "0.25", "minimum score to start a track"],
    ["`track_buffer`", "30", "frames a lost track is kept (NOT scaled by fps in this version: ANVESH's `TrackManager` converts it to seconds = (30 + 1.5) / fps = 1.26 s at 25 fps)"],
    ["`match_thresh`", "0.8", "maximum cost for a stage-1 match (higher = looser)"], ["`fuse_score`", "True", "cost = 1 - IoU x score"],
    ["Frame rate passed to the tracker", "none", "this Ultralytics version has no frame-rate parameter for ByteTrack; the buffer is in frames"],
], [2, 0.8, 3.7], "ByteTrack configuration actually in force"))
A(CALLOUT("warn", "ANVESH calls `model.track(conf=0.25)`, so every box under 0.25 is removed **before** ByteTrack sees it. The second matching stage (low-score boxes, 0.1-0.25) therefore normally gets no input. This was found in Session 7; the optional per-camera setting `tracker: {conf, config}` can lower it, but the demo keeps the library defaults.", "Important finding"))
A(H2("6.3 The ID questions - answered from the code"))
A(FIG(F("fig_id_layers.png"), 0.95, "Two ID layers. Most confusion in a viva comes from mixing them up."))
A(TABLE(["#", "Question", "Answer (verified in the code)"], [
    ["1", "Who generates the IDs: YOLO, ByteTrack or ANVESH?", "**Both of the last two, in two layers.** YOLO produces no IDs. ByteTrack (Ultralytics `BaseTrack.next_id()`) creates the *raw* ID. ANVESH's `TrackManager` creates the *session identity* shown to users."],
    ["2", "Per frame, video, camera or session?", "Raw ID: from a **class-global counter** that is reset to 0 whenever a `BYTETracker` is constructed (once per processing session in practice). ANVESH identity: unique per **(camera, session, serial)**; each session gets its own random run id (the demo session SES-2626B4805A -> `2626b4805a-s000-t00000`)."],
    ["3", "Integers or strings?", "Raw: integer in the tracker, emitted inside a float32 array and converted with `str(int(id))`. ANVESH: a **string**, `<run>-s000-t<serial:05d>`; the dashboard shows only the serial, e.g. `00051`."],
    ["4", "Start from zero or one?", "Raw: `_count` starts at 0 and `next_id()` **increments first**, so the first raw ID is **1**. ANVESH serial: starts at **0** (`t00000`)."],
    ["5", "How is the next ID allocated?", "Raw: when a leftover high-score box starts a new track (`activate()`), the counter increments - even for tracks that are removed before they are ever output, so raw IDs have gaps. ANVESH: `_next_serial += 1` when a raw ID that has no live state first appears."],
    ["6", "Fixed maximum number of IDs?", "No explicit maximum in either layer. Python integers are unbounded; the serial is printed with at least 5 digits (more if needed)."],
    ["7", "What at the numeric limit?", "No overflow in Python. The practical limit is that ByteTrack's output array is **float32**, which stores integers exactly only up to 2^24 = 16,777,216. At the observed 5,552 IDs per 20 min (about 4.6 per second) that is roughly **42 days of one continuous session** (derived from the library code, not tested; ANVESH sessions end far earlier)."],
    ["8", "Can IDs be reused?", "Not within one tracker run (the counter only grows). They restart from 1 when a tracker is created or reset. ANVESH never reuses a serial inside a manager, and a returning raw ID opens a **new** serial."],
    ["9", "Reused after a track expires?", "No. After removal the ID is gone for good; a similar vehicle later gets a fresh ID."],
    ["10", "Vehicle leaves and returns?", "If it returns within 30 frames and overlaps the Kalman-predicted box enough to match, it keeps the same raw ID. Otherwise it gets a **new** ID."],
    ["11", "Is it recognised as the same physical vehicle?", "**No.** ByteTrack uses only box position/size/score, no appearance (no re-identification). ANVESH states this in every incident: *\"Same physical vehicle after a tracker break is not established.\"*"],
    ["12", "Can two vehicles ever share an ID?", "Not at the same time inside one tracker (assignment is one-to-one). Over time an **ID switch** can make one ID follow two different vehicles (crossing paths, occlusion). Also, because the raw counter is class-global, two ByteTrack trackers alive in one process could disturb each other's numbering - a risk derived from the code that does not arise in the demo (one real-video session at a time; simulated cameras use no ByteTrack). `TrackManager` counts same-frame duplicates and stale restarts as diagnostics."],
    ["13", "After the application restarts?", "Raw IDs restart at 1; new sessions get a new random run id, so identities cannot collide with old ones. Stored incidents keep their old track-id text."],
    ["14", "Persistent across cameras?", "No. The identity key contains the camera id but nothing links two cameras (multi-camera fusion is not part of the app)."],
    ["15", "How are simultaneous vehicles distinguished?", "By the one-to-one assignment (`lap.lapjv`) of detections to tracks using IoU, so each box in a frame belongs to at most one track."],
    ["16", "Why thousands of IDs in 20 minutes?", "IDs name *tracklets*, and tracklets fragment: median vehicle box is 28 px wide at 640 x 360, far-field vehicles flicker, and lorries change class. 808 IDs in 180 s scales to about 5,390 in 1,200 s - the measured value is **5,552** (car 4,339, truck 714, bus 499), consistent with that."],
    ["17", "Track ID vs real-world identity?", "A track ID names one uninterrupted tracker run on one box stream. A real-world vehicle identity (plate, owner) needs plate reading and registry lookup, neither of which works automatically here."],
], [0.3, 1.8, 4.6], "The 17 ID questions"))
A(CALLOUT("say", "\"The 5,552 is the number of tracker identities created in 20 minutes, not the number of cars. One car that is hidden behind a lorry for two seconds can come back as a new identity, "
          "so I never use the ID count as a vehicle count - the dashboard even says 'session-scoped identities, not unique vehicles'.\"", "How to say it"))
A(H2("6.4 The tracking-continuity experiment (Session 7)"))
A(P("Question: can we cut the fragmentation by changing ByteTrack's settings? `demo/tools/track_continuity.py` re-ran the same 180 s of video (starting at 60 s) with the same detector and four settings. There are no labels, so only proxies can be counted."))
A(TABLE(["Variant", "Raw IDs", "Tracks < 1 s", "Median length", "Re-link candidates", "Size jumps"], [
    ["Baseline (conf 0.25, default yaml)", "808", "346", "2.08 s", "62", "2"],
    ["conf 0.10", "744", "268", "2.34 s", "53", "2"],
    ["conf 0.10 + match_thresh 0.9", "660", "179", "2.68 s", "26", "5"],
    ["conf 0.10 + buffer 50 + match 0.9", "653", "173", "2.72 s", "25", "5"],
], [2.8, 0.8, 1, 1, 1.1, 0.9], "Track-continuity study (proxies, 180 s)"))
A(H3("Why the aggressive settings were not adopted"))
A(BUL(["**conf 0.10:** about -8 % IDs and -15 % near-field re-link candidates, but about **10 % more boxes** per frame. In a 300 s application run the extra boxes pushed the *occupancy-based* congestion proxy over its threshold for flowing traffic (episode 10-280 s instead of 20-170 s). The extra boxes cannot be validated without labels.",
       "**match_thresh 0.9:** the biggest drop (660 IDs) but it is a *looser* matcher: IDs fall because tracks are joined, and without labels we cannot show that two different vehicles are not being joined. The size-jump proxy rose from 2 to 5 (small numbers, but the direction is a warning).",
       "**buffer 50:** almost no extra gain (653 vs 660).",
       "Decision: the demo keeps the **default tracker**; the setting stays available (`tracker` block in the camera config). Nothing merges tracklets."]))
A(P("Interpretation: most remaining fragmentation is **short far-field tracks and detector flicker at 640 x 360**, not a tracker bug. That is why the sensible next step is labelled footage (about 200 frames) rather than more parameter tuning."))
A(H3("A tiny example"))
A(TABLE(["Frame", "Detector", "Tracker output"], [
    ["1", "car A seen (score 0.8)", "new track, raw ID 7 created (tentative: not yet output)"],
    ["2", "car A seen", "matched -> confirmed; output **ID 7**; ANVESH opens tracklet serial 12"],
    ["3", "car A seen", "ID 7 -> serial 12"], ["4-5", "car A hidden behind a lorry", "track 7 Lost, box predicted; nothing output for it"],
    ["6", "car A seen again, overlaps the prediction", "re-found: still **ID 7** -> serial 12 (same tracklet)"],
    ["50+", "car A hidden for 40 frames", "ID 7 removed after 30 lost frames; on return -> **new ID 91** -> new serial"],
], [0.5, 2.5, 3.5], "One car through time"))

# =============================================================== CH 7
A(H1("Chapter 7 - Camera Geometry, Lanes and Homography"))
A(H2("7.1 Why pixels are not metres"))
A(P("A video frame is a grid of **pixels**. The road is a flat surface measured in **metres**. The camera looks along the road at a slant, so the same 10 m of road is many pixels near the camera and only a few at the horizon (**perspective distortion**). "
    "To convert you need a rule that maps each pixel on the road surface to a point on a metric map. That rule is a **homography**, found by **camera calibration**."))
A(TABLE(["Term", "Meaning"], [
    ["**Pixel coordinates**", "`(u, v)` = column, row in the 640 x 360 image; origin top-left, v grows downward"],
    ["**Image plane vs road plane**", "the picture versus the flat road seen from above; ANVESH assumes the road is a plane"],
    ["**Camera calibration**", "finding the mapping between the two using known reference points"],
    ["**Reference points**", "image points whose real road coordinates are known (here: lane-marking dash ends)"],
    ["**Homography (3 x 3 matrix H)**", "a projective transformation that maps points of one plane to another; 8 free numbers (H[3][3] fixed at 1)"],
    ["**RMSE**", "*root-mean-square error* = sqrt(mean of squared errors); a typical error size, in metres here"],
    ["**Calibration validation**", "checking the mapping on points that were *not* used to fit it (a hold-out)"],
], [1.3, 4.3], "Geometry vocabulary"))
A(FIG(F("fig_perspective.png"), 0.55, "Perspective: equal road distances shrink towards the horizon."))
A(H2("7.2 The homography equation"))
A(FORMULA(r"x'=h_{11}u+h_{12}v+h_{13},\quad y'=h_{21}u+h_{22}v+h_{23},\quad w'=h_{31}u+h_{32}v+h_{33}", None))
A(FORMULA(r"X=\frac{x'}{w'},\qquad Y=\frac{y'}{w'}", "Applying a homography H = [h_ij] (`CalibrationProfile.image_to_world`). (u, v) = pixel, (X, Y) = metres on the road; dividing by w' handles perspective."))
A(P("**How many points?** Each point pair gives 2 equations; H has 8 unknowns, so **4 points** (no three in a line) determine it exactly. More points give an over-determined system solved in the least-squares sense: ANVESH calls `cv2.findHomography(image_pts, world_pts, method=0)` (plain least squares, no RANSAC) in `fit_planar_homography` and refuses fewer than 4 points or a collinear set."))
A(H2("7.3 The actual ANVESH calibration (`demo/camera_config.json`)"))
A(TABLE(["Item", "Value"], [
    ["Reference", "UK *Traffic Signs Manual* ch. 5, Table 2-2 / Fig. 2-3: motorway lane line = diagram **1005.1** = **2000 mm mark + 7000 mm gap** (9 m pitch); stud spacing 18 m agrees"],
    ["Assumed (not verified)", "the road is built to that diagram; lane width **3.65 m**; planar road; the video is not time-altered; that the road is a UK road at all (inferred from the picture)"],
    ["Measurement", "bright dashes of the temporal-median background (vehicles removed) fitted with a principal axis for sub-pixel ends; toward-camera carriageway dashes 0-3"],
    ["Fit", "10 points (dashes 0-2), world x in {0, 3.65} m, y in {2, 9, 11, 18, 20} m. **Fit RMSE 0.094 m** (re-computed: 0.0940 m, max 0.147 m)"],
    ["Hold-out", "4 end points of dash 3 (27 m and 29 m) **not** used: **RMSE 0.43 m** (tolerance 1.0 m)"],
    ["Validation kind", "`marking_standard_holdout` - deliberately *not* `survey_holdout`"],
    ["Speed zone", "polygon (358,225), (506,225), (640,315), (640,360), (397,360) - rows >= 225 of the toward-camera carriageway"],
], [1.3, 4.5], "Calibration facts"))
A(CODE("H = [[-0.075600   0.056210   17.911958]\n     [ 0.004465   0.147293  -56.051430]\n     [ 0.000815  -0.010360    1.000000]]      # image (u, v) -> road (X, Y) metres, fitted from the 10 points"))
A(P("**Worked example.** Pixel (480, 300): x' = -0.0756 x 480 + 0.05621 x 300 + 17.912 = -1.513; y' = 0.004465 x 480 + 0.14729 x 300 - 56.051 = -9.720; w' = 0.000815 x 480 - 0.01036 x 300 + 1 = -1.717. "
    "So X = -1.513 / -1.717 = **0.88 m** across and Y = -9.720 / -1.717 = **5.66 m** along the road."))
A(FIG(F("fig_homography.png"), 1.0, "Left: the 640 x 360 image with the 10 reference points (red), lane polygons and speed zone. Right: the same points on the metric road plane."))
A(FIG(F("fig_mpp.png"), 0.6, "Along-road metres per pixel from H at image column 480: about 0.07 m at the bottom row and 0.42 m at row 225."))
A(H3("Why this is self-consistency, not ground truth"))
A(BUL(["The hold-out checks that consecutive dashes, a planar road and perspective fit together (0.43 m). It does **not** check the assumed 9 m pitch: a homography can absorb any uniform scale factor. If the true marking were the shorter diagram 1005 (1 m + 5 m) all speeds would read about 65 km/h instead of about 97.",
       "The plausibility check (first 300 s): median per-track speed 97 km/h (p10 87, p90 107, max 116) - the right order of magnitude for a UK motorway, which makes the 9 m assumption *reasonable*, not proved.",
       "**Why speed is labelled APPROXIMATE:** unverified scale assumptions. The displayed +/- band (about twice the regression standard error plus 1 km/h) covers only measurement noise; the scale uncertainty (plausibly +/-10 % or more) is **not** inside it."]))
A(H2("7.4 Which point of a vehicle is measured?"))
A(P("The box centre floats in the air for a tall vehicle; the part touching the road is the **bottom edge**. ANVESH therefore uses the **ground point = bottom-centre of the box**, `(x1 + w/2, y2)`. "
    "Limitations (stated in `types.py`): it is still a 2-D point, and an occluded bottom edge biases it upward. Boxes touching the frame edge are clipped, so their bottom-centre is *not* the ground contact and they are discarded as speed sources (3 px edge margin)."))
A(H2("7.5 Lanes and zones"))
A(P("The demo camera configures six **lane polygons** (TOWARD-3-offside, TOWARD-2, TOWARD-1-nearside, AWAY-1-nearside, AWAY-2, AWAY-3-offside) and two **direction zones** (toward-camera expecting direction (0.6, 0.8); away-from-camera expecting (0.6, -0.8)), all drawn by hand in image pixels. "
    "`assign_lane(ground_point, regions)` is a point-in-polygon test of the ground point. Lanes are **configured**, not detected by a neural network."))
A(CALLOUT("limit", "Calibration is for the toward-camera carriageway only. The stopped-vehicle and congestion logic use the same homography over the **whole frame**, so stops on the away-from-camera carriageway (all 14 real stops) are judged on an *extrapolated* scale (stop radius 0.75 m; far-field pixels are several decimetres). Every stopped-vehicle incident says so.", "Geometry limit"))

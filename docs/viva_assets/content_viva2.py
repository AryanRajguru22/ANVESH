from pathlib import Path
from docbuilder import *

FIGS = Path(__file__).resolve().parent / "viva_figs"
blocks = []
A = blocks.append

# ====================================================================== 3 TECHNOLOGY
A(H1("3. Every technology we used, and why"))
A(P("Versions are from `requirements-lock.txt` / `pyproject.toml` (checked in this repository). **No FastAPI, Flask, React, WebSocket or SSE is used** - say this clearly if asked."))
A(TABLE(["Technology (version)", "What it is / does", "Why we chose it", "Alternative and why not", "Where used"], [
    ["**Python** 3.11.9", "Programming language", "Best ecosystem for ML and video; fast to build and test", "C++ (faster, much slower to develop)", "everything"],
    ["**Ultralytics YOLOv8n** (ultralytics 8.4.132)", "Real-time object detector (+ built-in ByteTrack)", "Fast on a laptop GPU, simple API, pretrained on COCO, one call does detect+track", "Faster R-CNN: two-stage, often more accurate on small objects but usually much slower per frame (not benchmarked by us); SSD: older, weaker", "`perception/tracking.py`"],
    ["**PyTorch** 2.13.0+cu130", "Deep-learning engine that runs the network on the GPU (CUDA 13)", "Required by Ultralytics; GPU gives ~10-28 ms per frame", "TensorFlow/ONNX: would need conversion", "under YOLO"],
    ["**ByteTrack**", "Tracker that links detections over time", "Needs no appearance model, fast, handles low-score boxes", "DeepSORT: adds a ReID network, slower; SORT: simpler, more ID switches", "inside Ultralytics, config `bytetrack.yaml`"],
    ["**OpenCV** 5.0.0.93", "Video decode, drawing, JPEG encode, `findHomography`", "Standard, fast, one library for all image tasks", "FFmpeg directly: no drawing / geometry", "`ingestion.py`, `officer.py`, `calibration.py`"],
    ["**NumPy** 2.4.6; **lap** 0.5.13", "Arrays; linear-assignment solver used by ByteTrack matching", "Required by the stack", "SciPy `linear_sum_assignment`: slower here", "everywhere; tracker"],
    ["**http.server** (`ThreadingHTTPServer`)", "Python's built-in web server, one thread per request", "No dependency, easy to audit, enough for a demo; stream and JSON are simple", "FastAPI/Flask: nicer routing and async, but extra dependencies and more moving parts", "`app/server.py`, `routes.py`"],
    ["**SQLite** (WAL mode, `sqlite3`)", "Embedded database", "Zero setup, transactions, one file, good for one server", "PostgreSQL: needed for many servers / heavy concurrency", "`app/db.py`"],
    ["**HTML + CSS + vanilla JavaScript**", "The two screens", "No build step, small, works under a strict content-security policy", "React/Vue: more tooling than a demo needs", "`app/static/`"],
    ["**MJPEG** over HTTP", "A stream of JPEG images in one response", "Works in a plain `<img>` tag, no player code, each frame independent", "WebRTC/HLS: smoother, but far more complex", "`routes.py` `session_stream`"],
    ["**hashlib** (SHA-256, scrypt)", "Evidence hashing; password hashing", "In the standard library; scrypt is memory-hard", "bcrypt/argon2: extra packages", "`evidence.py`, `auth.py`"],
    ["**RapidOCR** 1.4.4 + onnxruntime 1.32.0", "Text recognition (plates)", "Light, runs without a second deep-learning stack", "EasyOCR: heavier; not connected for incident evidence", "`plate_reader.py`"],
    ["**pytest** 9.1.1, **psutil** 7.2.2, Playwright (tools only)", "Tests; memory/CPU sampling; browser checks", "Standard", "-", "`tests/`, `demo/tools/`"],
], [1.25, 1.35, 1.6, 1.6, 1.0], "Technology choices (verified against the repository)"))

# ====================================================================== 4 YOLO
A(H1("4. Object detection: YOLO and the model"))
A(H2("4.1 What detection is, and what YOLO returns"))
A(P("**Classification** says *what is in the picture* (one label). **Object detection** says *what and where*: it draws a **bounding box** around each object and gives a **class** and a **confidence**. **YOLO** (*You Only Look Once*) does this in one pass through a neural network, which is why it is fast enough for video."))
A(P("For one frame YOLOv8n returns a list like this (illustrative values, real format): box `[x1, y1, x2, y2] = [312, 188, 356, 214]` pixels, **confidence** `0.87` (how sure the network is, 0 to 1), **class id** `2` (= *car* in COCO). ANVESH keeps only vehicle classes: COCO ids **1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck**. *Person* (0) is not in the list, so pedestrians are never tracked as vehicles. The next stage (ByteTrack) receives these boxes with their scores."))
A(H2("4.2 The exact model"))
A(TABLE(["Question", "Verified answer"], [
    ["Which model?", "**YOLOv8n** - Ultralytics YOLO **version 8**, size **n (nano)**: the smallest, fastest variant. File `models/yolov8n.pt`, SHA-256 `f59b3d83...`. Loaded in code as `YOLO(\"models/yolov8n.pt\")`; task `detect`, **80 classes**, **3,157,200 parameters** (counted from the file in this session)."],
    ["Pretrained?", "**Yes.** Pretrained by Ultralytics on the **COCO** dataset (80 everyday classes). The base training details (epochs, hours) are **not recorded in this repository**; we did not train it."],
    ["Did we train it ourselves?", "**The default model: no.** We experimented with **fine-tuning (transfer learning)** a copy: `anvesh_mio5_yolov8n_best.pt` - see 4.4. It is **not used** unless `ANVESH_FINETUNED_MODEL` is set."],
    ["Input size and preprocessing", "Ultralytics default `imgsz = 640`: the 640x360 frame is **letterboxed** (resized keeping aspect ratio, padded to a multiple of 32), pixel values scaled to 0-1, sent to the GPU as a tensor. We do not set a custom size."],
    ["Thresholds", "`conf = 0.25` (we pass it). NMS IoU threshold `0.7`, max 300 detections per image (Ultralytics defaults, not changed by us)."],
    ["How unwanted detections are filtered", "(1) confidence below 0.25 dropped; (2) `classes=[1,2,3,5,7]` filter; (3) NMS removes duplicate boxes; (4) boxes with zero width/height are ignored by the tracker."],
], [1.6, 5.2], "The model"))
A(H2("4.3 IoU, confidence threshold and NMS (simple)"))
A(FORMULA(r"\mathrm{IoU}(A,B)=\frac{\mathrm{area}(A\cap B)}{\mathrm{area}(A\cup B)}", "Intersection over Union: 0 = no overlap, 1 = identical boxes"))
A(BUL(["**Confidence threshold**: ignore a box if the network is less than 25 % sure. Lower = more detections and more false alarms; higher = fewer boxes and more misses.",
       "**NMS (Non-Maximum Suppression)**: the network proposes many overlapping boxes for one car. Keep the highest-scoring box, delete any other box of the same class whose IoU with it is above 0.7, repeat.",
       "Example: boxes A (score 0.9) and B (score 0.6) on one car with IoU 0.8: B is removed. Two different cars with IoU 0.1 both stay."]))
A(H2("4.4 Training, pretraining, fine-tuning, inference, validation"))
A(TABLE(["Term", "Meaning"], [
    ["Training from scratch", "Learn all weights from random numbers on a big labelled dataset (days/weeks). **We did not do this.**"],
    ["Pretraining", "The general training someone else did (Ultralytics on COCO). Our `yolov8n.pt`."],
    ["Fine-tuning / transfer learning", "Start from pretrained weights and continue training on a *new* dataset so it fits a new task. **We did this as an experiment.**"],
    ["Validation", "Measuring the model on labelled images *not used to learn*, to see how good it is (and to pick the best checkpoint)."],
    ["Inference", "Using the finished model on new images. **This is what ANVESH does at run time.**"],
], [1.6, 5.2], "Vocabulary"))
A(TABLE(["Fact (from `docs/ANVESH_MODEL_TRAINING_REPORT.md` and `args.yaml`)", "Value"], [
    ["Run / dataset", "`anvesh_mio5_v1b`; **MIO-TCD** localization converted to 6 classes: **88,270 train / 10,865 val / 10,865 test images**"],
    ["Epochs / time / hardware", "**8 of 8 epochs, 2.97 hours**; GPU device `0` (the RTX 3050 6GB Laptop GPU of this machine)"],
    ["Optimiser / learning rate / batch", "**AdamW**, `lr0 = 0.001`, `lrf = 0.01`, momentum 0.937, weight decay 0.0005, batch 16, `imgsz` 640, mixed precision, early-stop patience 4"],
    ["Loss", "Ultralytics detection loss: box (CIoU), classification (binary cross-entropy) and distribution-focal loss - logged as `box_loss`, `cls_loss`, `dfl_loss` in `results.csv`. *Iterations* are not recorded separately: 88,270 images / batch 16 is about 5,517 per epoch (derived)."],
    ["Result on its own data", "MIO-TCD held-out **test mAP@50: 0.870 (fine-tuned) vs 0.215 (pretrained)**; test evaluated once. Training was not converged (val mAP still rising at epoch 8)."],
    ["Result on real CCTV", "CityFlow S02 (no class labels): pooled **recall 0.718 fine-tuned vs 0.748 pretrained** -> **not promoted**, default stays the pretrained model."],
    ["Plate detector (separate model)", "YOLOv8n fine-tuned **12 epochs** on a Vietnamese plate dataset (6,176 / 1,765 / 882 images); final validation mAP@50 **0.98** on that dataset only; licence chain unverified; not validated on CCTV."],
], [3.2, 3.6], "What we actually trained"))
A(CALLOUT("tip", "**\"Did you train this model yourself, or use a pretrained one? Why?\"** - \"The detector in ANVESH is the **pretrained YOLOv8n COCO model**. Training a detector from scratch needs a large labelled dataset and weeks of GPU time, and our contribution is the reasoning built on top of detection. We did try transfer learning: we fine-tuned a copy on MIO-TCD for 8 epochs (about 3 hours on a laptop GPU). It scored much higher on MIO-TCD (mAP50 0.870 vs 0.215) but was slightly worse on real CCTV footage (recall 0.718 vs 0.748), so we did not promote it and we say so. Detection accuracy on our own demo video is **not measured** because it has no labels.\"", "Your answer"))

# ====================================================================== 5 TRACKING
A(H1("5. Tracking and vehicle IDs"))
A(H2("5.1 Detection vs tracking"))
A(P("Detection works on **one frame at a time** and has no memory: the same car in frame 1 and frame 2 is just two unrelated boxes. **Tracking** adds memory: it decides *which box in this frame is the same vehicle as which box in the last frame* and gives it a consistent number. Without tracking you cannot measure speed, count a vehicle once, or notice that it stopped."))
A(H2("5.2 ByteTrack, step by step (as in the installed Ultralytics 8.4.132 code)"))
A(NUM(["**Predict.** Each existing track has a **Kalman filter** (state: centre x, y, aspect ratio, height, and their velocities). It predicts where the box should be in this frame.",
       "**Split detections** by score: *high* (>= 0.25) and *low* (between 0.1 and 0.25). *Note: ANVESH passes `conf=0.25` to YOLO, so low-score boxes never reach the tracker - ByteTrack's second stage is effectively unused (we measured conf 0.10 and did not adopt it: it inflated the congestion proxy).*",
       "**First association.** Cost = 1 - IoU (blended with the detection score) between every predicted box and every high-score detection. An optimal assignment solver (`lap`, Jonker-Volgenant - same result as the Hungarian algorithm) finds the cheapest one-to-one pairing; pairs with cost above `match_thresh = 0.8` are rejected (roughly IoU below ~0.2).",
       "**Second association** (IoU only, threshold 0.5): leftover tracks are matched to low-score boxes - this is ByteTrack's idea, to rescue partly hidden cars.",
       "**Update or lose.** Matched tracks are updated. Unmatched tracks become **lost** and are kept for `track_buffer = 30` frames (**1.2 s at 25 fps**); after that they are deleted.",
       "**New tracks.** A detection with no track and score >= `new_track_thresh = 0.25` starts a new *tentative* track; it is confirmed when it is matched again in the next frame. Each new track takes the **next integer** from a global counter."]))
A(FIG(FIGS / "fig_tracker.png", 1.0, "The 5-frame example: a track is kept by matching boxes; a hidden vehicle's track is kept for up to 30 frames."))
A(H2("5.3 What happens in the five-frame example"))
A(TABLE(["Frame", "What the tracker does"], [
    ["1", "Car A detected (conf 0.8). No tracks exist, so a new track is created: **ID 1**."],
    ["2", "A has moved. The Kalman filter predicted its new box; IoU with the detection is high -> **same ID 1**."],
    ["3", "Car B enters. No track overlaps it -> **new track, ID 2**. A keeps ID 1."],
    ["4", "A passes behind B: no detection for A. Its track becomes *lost* but is **kept** (up to 30 frames) and keeps being predicted. B keeps ID 2. *Risk:* if B's box overlaps A's predicted box, the tracker could give A's ID to B (an **ID switch**)."],
    ["5", "A reappears where the Kalman filter predicted it -> matched -> **ID 1 again**. If it had been hidden **more than 30 frames**, its track would be gone and it would get a **new ID (3)**."],
], [0.6, 6.2], "Tracker walk-through (illustrative scene, real algorithm)"))
A(H2("5.4 How ANVESH turns tracker IDs into session identities"))
A(P("A **raw ByteTrack ID** is just the next integer from a counter inside the library. It restarts at 1 after a reset, may be re-emitted for a revived track, and has **gaps** (ids given to tentative tracks that never got confirmed). It is not safe to store as a vehicle identity. So `TrackManager` creates its own key:"))
A(CODE("identity key = (camera_id, session_id, serial)\nprinted as        SES-ABC123-t00001, SES-ABC123-t00002, ...\nserial           = per-session counter, +1 per new track, never reused\nraw ByteTrack id = kept only for audit"))
A(BUL(["`t00001` uses a *minimum width* of 5 digits (`:05d`); `t123456` is valid. **There is no upper limit** (Python integers). In the 20-minute run the session created **5,552** tracks.",
       "If a raw ID comes back after its state was closed, it opens a **new** serial - it can never overwrite an old one. A stream reconnect or tracker reset starts a **new session**.",
       "An identity is a **tracklet** (one uninterrupted run of the tracker). The same physical car can have several identities (hidden > 1.2 s, far-field flicker, reconnect), and ANVESH never merges them. **5,552 IDs in 20 minutes is not 5,552 cars.** The screen therefore says *\"tracks this session - not unique vehicles\"*.",
       "Concurrent \"vehicles in view\" is a different number: tracks currently present."]))
A(TABLE(["Term", "What it is", "Lifetime"], [
    ["**Detection** (no ID)", "One box in one frame from YOLO", "one frame"],
    ["**Raw track ID / tracker ID**", "ByteTrack's integer for a box-stream", "until reset; may have gaps; can be revived"],
    ["**Session track ID** (`SES-...-t00001`)", "ANVESH's identity for a tracklet", "one session of one camera"],
    ["**Vehicle registration (plate) number**", "The real-world identity of a vehicle, e.g. `MH12AB1234`", "permanent; ANVESH never derives it from a track ID - a plate is a separate, officer-verified record"],
], [2.0, 3.0, 1.8], "Four different things - do not mix them up"))
A(H2("5.5 Can IDs switch or be confused?"))
A(P("**Yes.** ByteTrack uses only motion and box overlap, no appearance. IDs can switch when vehicles overlap or cross, fragment when a vehicle is hidden for more than 30 frames, or change class label when the detector flickers. On the CityFlow ground truth the raw tracker scored IDF1 >= 0.209 (a lower bound) with 42 ID switches on one camera (historical, `M-A_FOUNDATION_REPORT`). We reduced harm by using time-based occlusion handling and by not merging tracklets, but we did **not** improve tracking accuracy itself."))

from pathlib import Path
from docbuilder import *

FIGS = Path(__file__).resolve().parent / "viva_figs"
blocks = []
A = blocks.append

# ====================================================================== 6 SPEED
A(H1("6. Speed estimation: formulas and mathematics"))
A(H2("6.1 Why pixels are not metres"))
A(P("A box moving 10 pixels does **not** mean a fixed distance. A camera looks along the road, so near objects look big and far objects look small (**perspective distortion**). From our fitted calibration, one pixel of vertical movement is about **0.42 m at image row 225 (far)** but only about **0.10 m at row 320 (near)**. The same 10-pixel move is 4.2 m far away and 1.0 m near the camera. So we first convert image points into **road-plane metres**."))
A(FIG(FIGS / "fig_homography.png", 1.0, "A homography maps the camera's view of the (flat) road onto a top-down plane in metres."))
A(H2("6.2 The method we actually use"))
A(NUM(["**Ground point.** For each tracked box use the **bottom centre**: `(u, v) = ((x1 + x2) / 2, y2)`. The box centre floats above the road; the bottom edge is where the tyres touch the ground. Boxes touching the frame edge are discarded (a clipped bottom edge is not the ground).",
       "**Homography.** A 3x3 matrix `H`, fitted once per camera by `cv2.findHomography` (least squares) from 10 known road points (lane-marking dashes), maps pixels to metres.",
       "**Window.** Take the track's latest ground points **inside the calibrated speed zone only** (up to 1.6 s back; no time gap above 0.3 s, no clipped boxes, no impossible jump). Needs at least **8 samples** spanning at least **0.4 s**. At motorway speed the zone is only ~20 m long, so a fast vehicle gives a window of about 0.5-0.7 s (13-17 samples).",
       "**Least-squares slope.** Fit a straight line of position against time for each axis; isolated outlier samples (more than 3 robust sigmas, at least 0.25 m, from the line) are removed and the fit repeated. The slopes `vx, vy` are the velocity in m/s.",
       "**Convert** to km/h. **No extra smoothing** (the fit already averages; extra smoothing would hide real braking). If the error band is wider than 10 km/h the number is **withheld** ('too imprecise')."]))
A(FORMULA(r"(x,\,y,\,w)^{T}=H\,(u,\,v,\,1)^{T},\qquad X=\frac{x}{w},\quad Y=\frac{y}{w}", "Pixel (u, v) to road metres (X, Y)"))
A(FORMULA(r"v_X=\frac{\sum_i (t_i-\bar t)\,X_i}{\sum_i (t_i-\bar t)^2},\qquad v_Y\ \mathrm{the\ same\ with}\ Y_i", "Least-squares slope over the window (m/s)"))
A(FORMULA(r"\mathrm{speed}_{km/h}=3.6\,\sqrt{v_X^{2}+v_Y^{2}},\qquad \pm\left(2\,\frac{\mathrm{rms}}{\sqrt{\sum (t_i-\bar t)^2}}\cdot 3.6+1\right)", "Speed and the displayed error band (about 95 % of the slope uncertainty + 1 km/h)"))
A(P("Symbols: `u, v` pixel column/row; `H` the 3x3 calibration matrix; `X, Y` metres on the road; `t_i` the timestamp of sample *i*; `bar t` the mean time; `3.6` converts m/s to km/h (1 m/s = 3600 m / 1000 m per h); `rms` the root-mean-square distance of the points from the fitted line (a noise measure)."))
A(H2("6.3 Worked example (illustrative numbers, not a recording)"))
A(P("A car's ground points projected to the road at t = 0, 0.1, 0.2, 0.3, 0.4 s give Y = 6.0, 8.5, 11.0, 13.5, 16.0 m (X constant). Slope: (16.0 - 6.0) / 0.4 = **25 m/s**. Speed = 25 x 3.6 = **90 km/h**. Using the pixel movement instead of `H` would give a wrong answer that changes with distance from the camera."))
A(H2("6.4 Why it is approximate, and when it fails"))
A(TABLE(["Reason", "Effect"], [
    ["The **scale is assumed**: UK lane-marking standard (2 m dash + 7 m gap = 9 m pitch, lane width 3.65 m). The road was not surveyed and its identity is unverified.", "If the real pitch were 6 m, every speed would be wrong by the same factor (about 63 instead of 95 km/h). Plausibly **+/-10 % or more**, **not** included in the displayed +/- band. A homography cannot detect a uniform scale error."],
    ["Calibration fit: RMSE **0.094 m** (10 points); hold-out dash end points within **0.43 m**", "Shows the geometry is self-consistent, not that the metric scale is right."],
    ["Flat-road assumption, one homography, near-field **speed zone** only (rows >= 225)", "Curves, crests, far field: unreliable; speed is shown only inside the magenta zone."],
    ["Box jitter (bottom edge moves a few pixels)", "Averaged by the least-squares fit; isolated outliers are removed; tracks with rms > 2.5 m or > 250 km/h are rejected; an error band above 10 km/h is withheld; clipped or short tracks give no number."],
    ["Camera moved / different angle", "`H` is wrong -> all speeds wrong. Re-calibration is needed. No automatic detection of this."],
], [3.4, 3.4], "Assumptions and failure cases"))
A(CALLOUT("limit", "**Validation:** there is **no ground-truth speed** (no radar, GPS or surveyed distance), so the **absolute scale is not verified**. What *was* verified (speed audit, `docs/ANVESH_SPEED_AUDIT_FINDINGS.md`): timing is exact (constant 25.000 fps, video timestamps only); the formula and units are right; and a real defect was found and fixed - the fit window used to reach back into the *extrapolated* far field (local speed read 1.10x at rows 195-225 and 1.22x at rows 170-195 versus 1.0 in the calibrated rows), inflating displayed speeds by ~13-17 %. After the fix the median of 689 estimates is 94.9 km/h (was 111.4), the share above 112.7 km/h fell from 47 % to 6 %, and the `truck` class median is 88 km/h (UK HGV limiter ~90 km/h - plausibility, not validation). **FPS does not change the formula**: it uses real timestamps. Without a validated calibration the system shows **no km/h at all**.", "Be honest"))

# ====================================================================== 7 CONGESTION
A(H1("7. Congestion and behaviour analysis"))
A(H2("7.1 Congestion: what is actually measured"))
A(P("Every **10 s window** is summarised from the tracks present in it:"))
A(FORMULA(r"\mathrm{occupancy}=\min\left(1,\ \frac{\mathrm{mean\ concurrency}}{20}\right)", "Occupancy proxy; 20 = assumed capacity (max_capacity_vehicles)"))
A(BUL(["**mean concurrency** = average number of vehicles present at the same time in the window; **flow** = vehicles / window seconds; **mean speed** only from tracks whose speed was measured on a validated camera.",
       "**Rule (calibrated camera):** low pressure if mean speed >= 8 m/s (29 km/h) and occupancy < 0.6; high if mean speed <= 3 m/s (11 km/h) **or** occupancy >= 0.6; else medium. **Without speed:** occupancy >= 0.6 high, <= 0.3 low, else medium.",
       "**Hysteresis** (a small state machine) turns window values into FREE_FLOW / BUILDING / CONGESTED / DISSIPATING so one window cannot flip the state. An episode closes only after 30 s of free flow (60 s in the demo) so one stop-and-go jam is one incident."]))
A(CALLOUT("limit", "**This is a proxy, not verified congestion.** Capacity 20 is assumed; one count covers six lanes; on the demo clip it can say *congested* while vehicles move at 55-65 km/h (dense but flowing). Pedestrians are not counted (only 5 vehicle classes are tracked). We have no labelled congestion data.", "Honest status"))
A(H2("7.2 Other behaviours (all implemented, mostly unvalidated on real events)"))
A(TABLE(["Behaviour", "Rule actually coded", "Status"], [
    ["**Stopped vehicle**", "Stays within **0.75 m** (metric) or **7.5 px** (image) of its recent mean for a **3 s** dwell, i.e. slower than 0.5 m/s. Severity is lowered if it stops inside a queue (>= 3 stationary vehicles or congestion level BUILDING/CONGESTED).", "real footage: 14 stops in 20 min, reviewed by eye, not scored"],
    ["**Wrong-way**", "Net ground-point direction in a configured zone opposes the zone's expected direction for >= a minimum duration; duplicates within 3 s / 120 px suppressed as the same vehicle.", "only on SIMULATED scenes; needs configured zones"],
    ["**Possible collision**", "Two vehicles **moving and closing**, boxes overlap (IoU >= 0.10) for >= 1 s, then **both stationary within 2.5 s**; rejected if >= 60 % of >= 4 neighbours are also stopped (a queue/standstill).", "heuristic; **no real collision exists in our footage**"],
    ["**Jam-cause candidate**", "Stopped vehicles close together form a queue; the vehicle at the downstream front is the candidate; reports whether it stopped first, how many queued behind, and a Spearman order correlation; rated *likely* or *possible*.", "candidate only; does not know *why* it stopped"],
    ["Over-speed / braking / lane change", "Use the calibrated speed inside the speed zone (demo limit 112.7 km/h = 70 mph).", "no real event in the footage"],
], [1.3, 4.0, 1.5], "Behaviour detectors"))
A(P("**Traffic signals:** ANVESH does **not** read signals from video. A queue at a red light can look like congestion; the signal state is a configured/manual/external input used for red-light detection and the H6 hypothesis, not an automatic distinction."))

# ====================================================================== 8 FUSION
A(H1("8. Evidence fusion and hypothesis ranking"))
A(P("**The problem:** when traffic jams, *why*? A stopped vehicle? A crash? Heavy demand? One measurement is rarely enough and may be wrong. **Evidence fusion** combines several pieces of evidence while keeping track of how unsure we are. **Dempster-Shafer (DS) theory** does this with *belief masses* instead of probabilities, and has an explicit place for 'I don't know'."))
A(TABLE(["Hypothesis", "Meaning (from `hypotheses/cause_model.py`)"], [
    ["H1", "downstream bottleneck (congestion with no obstruction found)"], ["H2", "stopped / disabled vehicle"], ["H3", "collision-like obstruction"],
    ["H4", "lane blockage, non-incident"], ["H5", "upstream demand surge"], ["H6", "signal / intersection restriction"],
    ["H7", "**insufficient evidence** - not a cause: it is the leftover ignorance mass"],
], [0.8, 6.0], "The closed set H1-H7"))
A(H2("8.1 How it works, step by step"))
A(NUM(["**Evidence.** For each window the evidence builder scores each hypothesis between 0 and 1 from measured facts. Example rules: H1 = (0.5 + 0.5 x occupancy) x coverage if congested and no stopped vehicle; H2 = 0.8 if exactly one vehicle is stopped, 0.2 (contradicting) if two or more. A hypothesis that cannot be tested gets *no* support.",
       "**Mass assignment.** Supporting scores become masses on single hypotheses; if they add to more than 1 they are scaled; the **rest is ignorance** `m(Theta)`. No evidence anywhere = pure ignorance = *insufficient evidence*.",
       "**Reliability discounting.** A camera's reliability `r` (0-1; weights 0.25 each for observation count, tracking continuity, calibration, temporal coverage, times a completeness factor) shrinks its claims:",
       "**Combine two cameras** with Dempster's rule, blended with a conservative fallback when both suffer the same condition (fog, glare).",
       "**Belief / plausibility and rank.** Rank by belief, then plausibility, then id; outcome is RANKED, HIGH_CONFLICT or INSUFFICIENT_EVIDENCE."]))
A(FORMULA(r"m'(H_k)=r\,m(H_k),\qquad m'(\Theta)=1-r\left(1-m(\Theta)\right)", "Shafer discounting: less reliable camera -> more mass moves to ignorance"))
A(FORMULA(r"K=\sum_{p\neq q} m_1(H_p)\,m_2(H_q)", "Dempster's rule, step 1: conflict K = mass the two sources put on different hypotheses"))
A(FORMULA(r"m(H_k)=\frac{m_1(H_k)\,m_2(H_k)+m_1(H_k)\,m_2(\Theta)+m_1(\Theta)\,m_2(H_k)}{1-K}", "Step 2: combined mass of hypothesis k (agreement, or one source being unsure)"))
A(FORMULA(r"m(\Theta)=\frac{m_1(\Theta)\,m_2(\Theta)}{1-K}", "Step 3: remaining ignorance (both sources unsure)"))
A(FORMULA(r"\mathrm{Bel}(H_k)=m(H_k),\qquad \mathrm{Pl}(H_k)=m(H_k)+m(\Theta)", "Belief (proven-by-evidence) and plausibility (not ruled out) on this singleton frame"))
A(P("**Symbols:** `m1, m2` the two cameras' mass functions; `H_k` hypothesis k; `Theta` the whole set = ignorance; `K` the **conflict** (mass the two cameras put on *different* hypotheses); dividing by `1 - K` renormalises. If `1 - K` is almost 0 (total conflict) the code refuses to divide and returns pure ignorance, so conflict stays visible and nothing becomes NaN. With two cameras sharing a degradation the result is blended: `lambda x DS + (1 - lambda) x element-wise-max`, lambda = 0.4."))
A(H2("8.2 Worked example (illustrative numbers)"))
A(P("Camera 1: `m(H1) = 0.6`, `m(Theta) = 0.4`. Camera 2: `m(H1) = 0.3`, `m(H2) = 0.5`, `m(Theta) = 0.2`. Conflict `K = 0.6 x 0.5 = 0.30`, so `1 - K = 0.70`. Then `m(H1) = (0.18 + 0.12 + 0.12) / 0.7 = 0.60`, `m(H2) = (0 + 0 + 0.4 x 0.5) / 0.7 = 0.286`, `m(Theta) = 0.08 / 0.7 = 0.114` (sum 1.0). **Bel(H1) = 0.60, Pl(H1) = 0.714**: H1 ranks first, but 0.114 stays as 'unknown'. *Discounting check:* r = 0.8 turns `m(H1) = 0.6` into 0.48 and `m(Theta) = 0.4` into 1 - 0.8 x 0.6 = 0.52."))
A(CALLOUT("limit", "**Ranking is not proof.** The top hypothesis is only the best supported among those that *could be tested*. Belief is a DS quantity, **not a probability** (the code never prints a percentage). The scores (0.8, 0.5 + 0.5 x occupancy, ...) are **hand-set heuristics, not learned or calibrated**. **Cross-camera fusion is implemented and unit-tested, but the running application uses one camera per session**, so the shipped product uses the single-camera discounted path; on image-space or uncalibrated cameras the honest result is `insufficient_evidence`. A cause can be *established* only by an authorised officer with a written basis.", "Limits"))

# ====================================================================== 9 BACKEND
A(H1("9. Backend, frontend and live video"))
A(H2("9.1 Start -> video on screen"))
A(NUM(["Button click -> `fetch` `POST /api/v1/cameras/{id}/sessions` (headers `X-Requested-With`; cookie session; role must have `session.control`). Server answers **202** with `{session_id, state: starting}`; refuses a second active session on the same camera.",
       "The page calls `attach()`: sets `<img src=\"/api/v1/sessions/{id}/stream.mjpg\">` and starts polling `GET /sessions/{id}/officer` every **1 s**.",
       "The session thread decodes frames, runs the pipeline (Section 2) and keeps the latest annotated JPEG in **one slot** (`last_jpeg`).",
       "The stream handler (one HTTP thread per viewer) checks the slot every **80 ms** and writes each *new* JPEG as a multipart part. The JSON poll returns tiles, alerts, state, progress."]))
A(H2("9.2 MJPEG in one paragraph"))
A(P("**MJPEG = Motion JPEG.** The server sends one long HTTP response of type `multipart/x-mixed-replace; boundary=frame`; each part is a complete JPEG (`--frame`, `Content-Type: image/jpeg`, `Content-Length`, bytes). The browser replaces the image in an `<img>` tag every time a part arrives. It is **not** a video file: there is no seeking and no inter-frame compression (so it uses more bandwidth than H.264), but it needs no player, any single frame is viewable, and it is trivial to implement - ideal for a demonstration. Because there is only one slot, a slow browser skips frames; it can never make the server buffer without limit."))
A(H2("9.3 Sessions, pause, stop, isolation, errors"))
A(TABLE(["Topic", "How it works (verified in code/tests)"], [
    ["States", "starting, connecting, live, paused, reconnecting, stalled, completed, stopped, failed, interrupted. Terminal: stopped, completed, failed, interrupted."],
    ["Pause / resume", "Only **recorded files** can pause (`source.pause()`); live sources refuse. The screen shows PAUSED and a message."],
    ["Stop", "`POST .../stop` sets a stop flag, closes the source, joins the thread; state `stopped`; the stream ends ~2 s later."],
    ["Isolation", "Every session has its own id, thread, track manager and `last_jpeg`; the stream is looked up by session id (tested with two sessions). The officer screen also stops other running sources before starting a new one, and starts are **serialised** (fix for a race found in testing)."],
    ["Errors", "Detector/engine exceptions are recorded in the session's `errors`; the state becomes **failed** and the screen shows the first error. A stalled/lost live source goes to *reconnecting* with backoff and starts a **new identity session**."],
    ["Client disconnect", "A closed tab raises BrokenPipe/ConnectionReset in the stream handler; it returns and the thread is freed (tested with 5 abrupt disconnects). The processing session keeps running."],
    ["Stream reconnection", "If the `<img>` connection breaks while the session is still running, the page reconnects up to **3 times**, 1.5 s apart."],
    ["Stale data", "A poll answer for a session that is no longer selected is **dropped**; the page never mixes sessions."],
    ["Security", "scrypt passwords, hashed session tokens, lockout, CSRF header, role-based permissions, four-eyes approval, SHA-256 evidence verification via a *separate* verify route, hash-chained audit log, one-click demo sign-in only on loopback."],
], [1.3, 5.5], "Lifecycle and reliability"))

# ====================================================================== 10 PERFORMANCE
A(H1("10. Performance, FPS and latency"))
A(P("Source video: 640x360 at **25 fps** = one frame every **40 ms**. To keep real time the system must process a frame in under 40 ms **on average**. Hardware: **NVIDIA RTX 3050 6GB Laptop GPU, Intel Core i5-13450HX (10 cores / 16 threads), 15.7 GB RAM, Windows 11**; the GPU is used for detection (CUDA)."))
A(TABLE(["Measure (20-minute real-time run, Session 9)", "Value", "Meaning"], [
    ["Frames received / processed / dropped", "30,001 / 30,001 / **0**", "the frame queue *blocks* when full, so nothing is thrown away; the session just lags"],
    ["Wall time for 1,200 s of video", "**1,377.9 s**", "it ran 15 % slower than real time overall"],
    ["Average processing FPS", "**21.77**", "frames processed / wall seconds (source = 25)"],
    ["Real time kept?", "first ~740 s at 25.0 fps, then 15-21 fps", "after ~800 s the queue stayed full (8/8)"],
    ["Frame-processing latency p50 / p95 / p99", "34.9 / 71.7 / 105.6 ms (mean 38.8)", "time the engine spends on one frame: windowing + detect/track + callbacks (incl. drawing and JPEG)"],
    ["detect + track p50 / p95", "28.1 / 52.4 ms", "the GPU model call"],
    ["End-to-end p50 / p95", "43.3 / 663.9 ms", "decoded frame -> processed, **including waiting in the queue**; excludes the browser"],
    ["Memory (RSS)", "up to 1,745 MB; first -> last quarter +1.5 %", "no leak seen"],
    ["GPU / CPU", "GPU util mean 27 % (max sampled 70 %); server CPU mean 103 % of one core", "10 s samples; GPU memory reading (133-177 MiB) looks low and was **not cross-checked**"],
    ["Fast mode (150 s, no pacing)", "**41.4 fps**, p50 13.4 / p95 32.0 ms", "capacity test at that moment; **not** real time"],
], [2.5, 2.0, 2.3], "Verified benchmark (docs/benchmarks/, finalization report)"))
A(TABLE(["Term", "Simple meaning"], [
    ["Input FPS", "How fast the video is *recorded* (25)."], ["Processing FPS", "How many frames we *finish* per second of wall time."],
    ["Real-time throughput", "Processing FPS >= input FPS, so the system never falls behind."],
    ["Latency", "How long **one frame** takes from start to finish. FPS and latency differ: you can process 25 frames/s, each taking 200 ms, if frames overlap."],
    ["p50 / p95", "50 % / 95 % of frames were faster than this. p95 = 71.7 ms means 1 frame in 20 took longer; the **tail** is what makes a mean of 38.8 ms (close to the 40 ms budget) fall behind."],
    ["Queue wait", "Time a decoded frame waits because the processor is busy; it is part of end-to-end latency."],
], [1.8, 5.0], "Terms"))
A(CALLOUT("limit", "**Defend it like this:** \"We measured all 30,001 frames with none dropped, 21.8 fps average, 1.7 GB peak memory. We did **not** keep 25 fps for the whole video: after about 12 minutes it fell behind. The laptop was shared and probably on battery, so the cause is **not established** and this is not a clean capacity figure. The earlier 24.89 fps result (Session 7) was **not reproduced**. A large video is processed at the same per-frame cost; only a longer run time.\" A repeat on a plugged-in, idle machine is still to be done.", "Honest benchmark answer"))

# ====================================================================== 11 WHAT CHANGED
A(H1("11. What changed since the previous presentation (2 November)"))
A(P("Sources: Git history (`a74a400` 14 Nov; `629242d`, `976e4fd`, `dfb4046`, `6ba8786` 15 Nov Session 8; `d0ec041` 15 Nov Session 9), `docs/ANVESH_SESSION8_REPORT.md`, `docs/ANVESH_FINALIZATION_REPORT.md`. The 2 November state was the full dashboard (`/advanced`) at git `b6d5ecd` plus an **uncommitted** working tree (sessions 3-7); everything is now committed and pushed."))
A(TABLE(["Earlier version (2 Nov)", "Current version", "Improvement"], [
    ["Only the full dashboard (21-step demo)", "**Officer screen** `/officer`: one click starts a real session; 6 tiles (lanes, speed, wrong-way, accident, plate, jam cause) and an alert list", "Simple, video-first, faster to demo"],
    ["No cause hint for a jam", "**Jam-cause candidate** (front-of-queue vehicle, stopped-first, order correlation)", "Explains *why*, always worded as candidate"],
    ["Plates entered by hand only", "Live **plate reader** (YOLOv8n plate detector + RapidOCR), real footage only, >= 110 px vehicles", "Added; unvalidated on CCTV; silent on the 640x360 demo"],
    ["Wrong-way / accident shown only via the old simulated cameras", "Three scripted scenes (jam, wrong-way, accident) labelled **SIMULATED**", "Honest demonstration of detectors real footage never triggers"],
    ["State not obvious on the live page", "**Status line** (LIVE / PAUSED / STOPPED, camera, session id, recorded/SIMULATED); distinct *Paused* / *Stopped* messages", "Clear state; Stop no longer says 'Finished'"],
    ["Slow reply could overwrite a newer screen; stream could freeze", "Stale-poll guard; stream **reconnect** (3 tries); starts **serialised** (found by the browser lifecycle test: two sessions could stay running)", "Session isolation and reliability"],
    ["Server printed tracebacks for closed tabs; no stream tests", "Quiet disconnect handling; **5 MJPEG stream tests** over a real socket", "Verified stream termination, ordering, isolation, thread release"],
    ["`/advanced` text said no plate detector exists", "Text corrected", "Documentation accuracy"],
    ["1,381 tests; Session 7 benchmark only (24.89 fps)", "**1,403 tests**; new 20-minute real-time measurement with CPU/GPU/memory sampling; fast mode kept separate", "Honest, reproducible measurements (and a finding: real time was not kept to the end)"],
    ["Work only on disk (not committed)", "31 commits, pushed to GitHub branch `officer-screen`; `.gitignore` keeps unlicensed material out", "Reproducible, shareable"],
    ["Handbook 20 chapters", "Handbook 22 chapters (officer screen, analytical validity, performance and testing) and a new officer-screen section in the panel guide", "Up-to-date documentation"],
], [2.2, 3.1, 1.5], "Earlier vs current"))
A(CALLOUT("tip", "**\"What have you improved since your previous presentation?\"** - \"Mainly usability, reliability and honesty. We added a one-screen officer view that starts a real session with one click and shows the processed video with live state, a jam-cause candidate and a plate reader. We found and fixed real bugs: a late reply could overwrite the screen, the video stream could freeze after a drop, and quick source switching could leave two sessions running. We added stream and lifecycle tests - 1,403 tests now - and ran a 20-minute real-time benchmark, which showed we do not keep 25 fps for the whole video on this laptop; we report that rather than hide it. Everything is committed and on GitHub.\"", "Your answer"))

# ====================================================================== 12 TESTING
A(H1("12. Testing and reliability"))
A(TABLE(["Kind", "What it is", "Example in ANVESH"], [
    ["**Unit test**", "Tests one small function/module alone", "IoU / homography / DS combination / speed slope (`tests/world`, `tests/fusion`, `tests/evidence`)"],
    ["**Integration test**", "Several modules together", "streaming engine + windows + identity (`tests/streaming`)"],
    ["**End-to-end test**", "Whole path through the real pipeline", "scripted scene -> detectors -> incident -> evidence -> hash verify (`test_session_synthetic.py`); browser lifecycle with real Chrome + real `demo.mp4` (`demo/tools/ui_lifecycle.py`, 12 checks)"],
    ["**Regression test**", "Locks a fixed bug so it cannot return", "stop-episode late closure, stale poll, the MJPEG tests, queue-vs-collision gates"],
], [1.3, 2.2, 3.3], "Kinds of tests"))
A(P("**Final verified count: 1,403 tests passed** (run in nine groups: 38 + 110 + 109 + 99 + 306 + 193 + 132 + 229 + 187). One timing-bounded ingestion test failed once when the machine was slow and passed on rerun - we report that. Streaming is tested over a real socket (frames whole and in order, stream ends when the session stops, two sessions never share frames, handler threads freed after 5 abrupt disconnects, 404/401). Pause/resume/stop and a closed browser are exercised by the browser lifecycle script."))
A(CALLOUT("limit", "**A passing suite does not prove the speeds, congestion levels or causes are accurate.** It proves the code does what we *specified*, on synthetic scenes and scripted trackers. Accuracy needs ground truth, which we do not have for these outputs. The real detector is exercised by the benchmark and browser checks, not by unit tests.", "What the count does not mean"))

# ====================================================================== 13 LIMITS
A(H1("13. Limitations and future scope"))
A(TABLE(["Area", "Limitation (current)", "Needed / future (not done)"], [
    ["Detection", "Pretrained COCO model; accuracy on our footage **not measured**; small far vehicles (median box 28 px) missed/flicker; no auto-rickshaw class", "Label ~200-500 frames of the real cameras; compare models fairly"],
    ["Tracking", "IDs fragment (5,552 in 20 min); ID switches in overlaps; no appearance model", "Re-identification, tuned association, labelled tracking ground truth"],
    ["Speed", "Assumed scale (UK marking), not surveyed; near field only; **no ground truth**", "Survey two distances or use radar/GPS runs; per-camera calibration tool"],
    ["Congestion", "Occupancy proxy, assumed capacity 20; not verified", "Labelled free/slow/congested intervals; per-lane capacity"],
    ["Fusion", "Hand-set heuristic scores; one camera per session; cross-camera path only unit-tested", "Calibrate scores on labelled incidents; run multi-camera"],
    ["Events", "Wrong-way and collision only on simulated scenes", "Licensed real clips with labels"],
    ["Deployment", "Research/demo; sandbox registry/challan; no MFA; Windows only; Docker untested; real-time not kept to the end on this laptop; licences (Ultralytics AGPL, MIO-TCD non-commercial, demo video unknown)", "Dedicated server, HTTPS, MFA, authorised registry, licence review, long-run soak on target hardware"],
], [1.1, 3.4, 2.3], "Honest limitations vs future work"))
A(CALLOUT("tip", "**\"Is your project production-ready?\"** - \"No. It is a working, tested **research and demonstration system**. The pipeline, evidence workflow and screens work end to end, and we have 1,403 passing tests and a measured 20-minute run. But speed, congestion and cause ranking are not validated against ground truth, our real-time capacity on this laptop was not clean, the registry and challan are simulated, and there is no multi-factor authentication. To go to production we would need labelled data, a validated calibration per camera, a dedicated server and legal approvals.\"", "Your answer"))

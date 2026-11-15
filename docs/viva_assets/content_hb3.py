from pathlib import Path
import json
from docbuilder import *

FIGS = Path(__file__).resolve().parent / "figures"
F = lambda n: FIGS / n
NJ = json.load(open(FIGS / "numbers.json"))
SE = NJ["speed_example"]
blocks = []
A = blocks.append

# =============================================================== CH 8
A(H1("Chapter 8 - Speed Estimation"))
A(P("ANVESH has **two** speed calculations. They share the homography but serve different purposes; mixing them up is a common viva trap."))
A(TABLE(["", "Live speed (`anvesh/app/livespeed.py`)", "Window speed (`anvesh/perception/speed_estimation.py`)"], [
    ["Used for", "the km/h shown in the live table / video, the **over-speed** and **sudden-braking** detectors", "the per-window **mean speed** that feeds the congestion level and traffic state"],
    ["Method", "least-squares **slope** of metric position against time over the last 1.6 s (1.2 s for braking)", "robust speed = total path length / total elapsed time over the track's valid steps (`compute_robust_speed`)"],
    ["Where", "only inside the near-field **speed zone**", "any track in the frame (whole frame, including the extrapolated far field)"],
    ["Error band", "yes: about 2 x standard error + 1 km/h", "standard deviation of per-track speeds across the window"],
    ["Gate", "min 6 samples, span >= 0.8 s, residual RMS <= 2.5 m, <= 250 km/h", "min 3 samples, >= 0.3 s elapsed, step gap <= 0.6 s, <= 200 km/h else `OUTLIER`"],
], [0.9, 2.8, 2.8], "Two speed paths"))
A(H2("8.1 The calculation, step by step"))
A(NUM(["**Positions.** For each track the engine keeps `(timestamp, ground point)` samples; the timestamp is `frame_index / fps`, not the wall clock, so processing delay cannot distort speed.",
       "**Image to road.** Each ground point (u, v) goes through the homography H to metric `(X, Y)` (Chapter 7).",
       "**Distance and time.** Between two samples: distance = sqrt(dX^2 + dY^2) metres; time = dt seconds.",
       "**Velocity.** v = distance / time in m/s; **km/h = m/s x 3.6**.",
       "**Noise-robust version.** A two-point difference amplifies box jitter, so live speed fits a straight line to *all* samples in the window by least squares and uses its slope."]))
A(FORMULA(r"v_{\mathrm{km/h}}=\frac{d_{\mathrm{metres}}}{t_{\mathrm{seconds}}}\times 3.6", "Speed (km/h) = Distance (metres) / Time (seconds) x 3.6"))
A(FORMULA(r"v_X=\frac{\sum_i (t_i-\bar t)\,X_i}{\sum_i (t_i-\bar t)^2},\quad v_Y=\frac{\sum_i (t_i-\bar t)\,Y_i}{\sum_i (t_i-\bar t)^2},\quad v=3.6\sqrt{v_X^2+v_Y^2}", "Live speed = slope of a least-squares line through the road-plane positions (LiveSpeed.estimate)."))
A(FORMULA(r"\mathrm{err}=2\cdot\frac{\mathrm{rms}}{\sqrt{S_{tt}}}\cdot 3.6+1\ \ \mathrm{km/h},\qquad S_{tt}=\sum_i (t_i-\bar t)^2", "Displayed +/- band: about 95 % of the slope's noise plus a 1 km/h floor. rms = residual of the points about the line."))
A(H3("Worked numerical example (computed with the real H)"))
A(P(f"A car travels at a true **80 km/h** (22.2 m/s) for 1.0 s, 26 frames at 25 fps, along x = 1.8 m. Each ground point is projected to pixels (from about ({SE['p_first_px'][0]:.0f}, {SE['p_first_px'][1]:.0f}) to ({SE['p_last_px'][0]:.0f}, {SE['p_last_px'][1]:.0f})) and given about 1 pixel of random box jitter, then converted back through H."))
A(TABLE(["Method", "Result", "Comment"], [
    ["Two-point speed, frames 0 and 1", f"{SE['naive_first_two_kmh']:.1f} km/h", "+9 % error from 1 px of jitter over only 0.04 s"],
    ["Least-squares slope (ANVESH)", f"**{SE['fit_kmh']:.1f} km/h** +/- {SE['err_kmh']:.1f}", f"rms residual {SE['rms_m']:.2f} m; S_tt = {SE['stt']:.2f} s^2; band = 2 x {SE['rms_m']:.3f}/sqrt({SE['stt']:.2f}) x 3.6 + 1"],
], [2, 1.5, 3], "Why a regression is used"))
A(FIG(F("fig_speed_example.png"), 0.62, "Simulated track through the real homography: 26 noisy points and the fitted slope."))
A(H2("8.2 How the implementation handles the difficult parts"))
A(TABLE(["Issue", "Handling (from `LiveSpeed`)"], [
    ["Smoothing", "exponential moving average per track, alpha 0.5 (`ema`); 1.0 (none) for the braking detector"],
    ["Track history", "only samples from the last `window_s` (1.6 s) are used; needs >= 6 samples spanning >= 0.8 s, else `insufficient_history`"],
    ["Noise", "residual RMS above 2.5 m -> `rejected_unstable`; speed above 250 km/h -> `rejected_implausible`"],
    ["Lost tracks", "a track with no samples gives no number; per-track smoothing state is dropped when the track leaves view (`forget_absent`)"],
    ["Frame timing", "uses frame timestamps (index/fps); frame drops would show as longer dt, not wrong speed"],
    ["Invalid calibration", "`no_metric_calibration`: **no number is ever produced** without a validated calibration (the speed-safety invariant, `supports_metric_speed`)"],
    ["Near-field zone", "only the last sample must be inside the polygon; outside -> `outside_speed_zone`"],
    ["Camera-edge artefacts", "boxes within 3 px of the frame border are discarded (a lorry leaving the frame looked like \"sudden braking\" before this rule)"],
], [1.5, 4.8], "Robustness measures"))
A(H2("8.3 Results and limits"))
A(BUL(["Plausibility check (first 300 s): median 97 km/h, p10 87, p90 107, max 116 km/h - right order of magnitude for a motorway.",
       "No over-speed or braking events occurred in the 20-minute real run (over-speed demonstration threshold 112.7 km/h = 70 mph; the event also needs the estimate to exceed the limit + 5 km/h with its own error band above the limit for 2 s). Those detectors are shown only with scripted tracks in unit tests.",
       "A vehicle at about 100 km/h crosses the roughly 22 m speed zone in under a second, so the zone is short in time; that is why speeds need a good fit from few samples.",
       "The window mean speed used by congestion uses path length; Session 6 found that this style of calculation is inflated by box jitter (up to >132 km/h readings the per-track estimator never saw). **Inference, not measured:** part of the 55-65 km/h reading in congested windows may be such inflation."]))
A(H2("8.4 Questions you will be asked"))
A(QA([
    ("Why can speed not be measured directly from pixels?", "A pixel is not a fixed length. The same 1 pixel step is 0.07 m near the bottom of our picture and 0.42 m at row 225. Speed needs real distance, so the pixel must first be converted to the road plane."),
    ("Why does a vehicle look smaller as it moves away?", "Perspective: the apparent size is roughly inversely proportional to distance from the camera. The same effect compresses road distances towards the horizon, which is why the homography is not a simple scale."),
    ("Why is camera height important?", "It sets the viewing angle and therefore how strongly the road is foreshortened. A higher, steeper camera has less distortion and better accuracy; ANVESH has no height measurement - the homography absorbs it implicitly from the 10 reference points."),
    ("Why is calibration necessary?", "Without known points on the road we cannot say how many metres one pixel is, so any km/h would be a guess. ANVESH therefore refuses to show km/h without a validated calibration."),
    ("Why is the current speed only approximate?", "The scale comes from an assumed lane-marking standard (9 m dash pitch, 3.65 m lane width), the site was not surveyed, and the +/- band excludes that scale uncertainty (plausibly +/-10 % or more)."),
    ("How would a traffic department make it more accurate?", "Survey two known distances on the actual road (or use the road's official marking drawing), replace the validation record with a `survey_holdout` one, calibrate each camera, and tune thresholds on labelled examples. No code change is needed other than the configuration."),
]))

# =============================================================== CH 9
A(H1("Chapter 9 - Traffic Events and Violations"))
A(CALLOUT("warn", "**A detected condition is not a confirmed offence.** Every detector outputs a *candidate*: indicators, a heuristic confidence tier (not a probability), an uncertainty list and the evidence. "
          "A human reviewer decides; a case, registry check and approval follow. ANVESH never issues a challan.", "Read this before the detectors"))
A(TABLE(["Detector", "Where in `detectors.py`", "Demo status"], [
    ["Stopped vehicle + long stop", "`StoppedVehicleDetector`", "**real**: 14 incidents, all queue members rated low"],
    ["Congestion + queue membership", "`CongestionDetector`", "**real**: 4 episodes in 20 min"],
    ["Wrong-way", "`WrongWayDetector`", "configured + unit-tested; **SIMULATED** only (1 event)"],
    ["Wrong-lane", "`WrongLaneDetector`", "needs `allowed_classes` per lane - not applicable to this road"],
    ["Over-speed", "`OverspeedDetector`", "candidate from approximate speed; **none** in the footage"],
    ["Sudden braking", "`BrakingDetector`", "candidate; **none** in the footage"],
    ["Aggressive driving (rash)", "`RapidLaneChangeDetector`", "candidate; **none** in the footage"],
    ["Illegal turn / red light", "`IllegalTurnDetector`, `RedLightDetector`", "implemented + unit-tested; need turn rules / stop line + signal state; not applicable to this scene"],
    ["Possible collision", "`CollisionCandidateDetector`", "**experimental**; SIMULATED only (1 event); 0 candidates in 20 min"],
    ["Post-incident disruption", "`IncidentService._link_disruption`", "congestion within 120 s after another incident is linked (temporal only)"],
], [1.8, 2.2, 2.6], "Detector overview"))
A(H2("9.1 Stopped vehicle and long-duration stop"))
A(TABLE(["Item", "Detail"], [
    ["Input signals", "ground-point trajectories in metric space, 10 s window snapshots, congestion level, count of stationary vehicles"],
    ["Algorithm", "**displacement criterion** (`stop_criterion: displacement`): a vehicle is stopped while each ground point stays within a **0.75 m** radius of the mean of the last **3.0 s**; a run becomes an *episode* once it lasts `dwell_seconds` = **3.0 s**; a silence up to 3 s is bridged if the car reappears inside the radius; samples outside the radius must persist 1.0 s to end the run"],
    ["Formula idea", "a car at constant speed v sits v x W / 2 from the mean of its last W seconds, so stopped means v <= 2R/W = 2 x 0.75 / 3 = **0.5 m/s (1.8 km/h)**"],
    ["Escalation", "`long_duration_stop` once the stop lasts >= `long_stop_s` = **30 s** (code default)"],
    ["Severity", "**low** if the congestion level is BUILDING / CONGESTED / DISSIPATING (queue) **or** >= `queue_peer_stops` = 3 vehicles are stationary together; otherwise **medium**; **high** when >= 30 s outside a queue"],
    ["Queue latch", "the queue judgement is latched for the whole stop, so a queued car is not re-rated an \"isolated obstruction\" when its queue clears; `queue_group_id` groups stops within 30 s of each other"],
    ["Confidence", "heuristic tier from dwell vs the required 3 s (score), not a probability"],
    ["False-positive risks", "queues (the main one), signal stops, tracker breaks, extrapolated metric scale on the far carriageway, a crawl in dense traffic"],
    ["Demo status", "14 incidents, all `low` + **queue member**, one queue group (the first stop begins at 111 s), all on the away-from-camera carriageway; before the Session 7 fix 2 of them never got an end time"],
], [1.2, 5], "Stopped vehicle"))
A(FIG(F("fig_stop.png"), 0.85, "The displacement criterion: stopped = staying inside a small circle around the recent mean position."))
A(H2("9.2 Congestion, queue and the occupancy proxy"))
A(P("Every 10 s window the reasoner counts the tracks, computes the **mean concurrency** (average number of vehicles in view = total dwell of all tracks / window length), converts it to an **occupancy proxy** "
    "and advances a four-state hysteresis machine."))
A(FORMULA(r"\mathrm{occupancy}=\min\!\left(1,\ \frac{\mathrm{mean\ concurrency}}{20}\right)", "Occupancy proxy; 20 = max_capacity_vehicles, an ASSUMED capacity. A mean of 12 vehicles in view gives 0.6, the high threshold."))
A(FIG(F("fig_congestion.png"), 1.0, "Congestion states and the raw pressure rule (anvesh/perception/traffic_state.py)."))
A(BUL(["**Hysteresis** (damping) means one noisy window cannot flip FREE_FLOW <-> CONGESTED: the system must pass through BUILDING or DISSIPATING and be re-confirmed.",
       "**Episode:** `CongestionDetector` opens an incident when the level leaves FREE_FLOW (BUILDING = low severity), escalates when CONGESTED (**medium**, or **high** after >= 300 s not free-flowing) and closes after the camera stays FREE_FLOW for `congestion_close_grace_s` (code default 30 s; **60 s in the demo**). The end is back-dated to the first free-flow window; severity uses `non_free_s`, never the span.",
       "**Result:** 13 flapping episodes became **4**: 20-170 s, 360-680 s, 840-950 s, 1020-1200 s (still open at the end of the stream). A grace of 10/20/30/40/60 s gave 13/10/7/7/4 episodes - 60 s is a *policy value*, tuned on one video's pattern.",
       "**Why a dense but moving road can read congested:** the rule is `speed <= 3 m/s OR occupancy >= 0.6`. With six lanes in view, 12 vehicles in the picture is easy even at 55-65 km/h. The incident says so: \"dense, not necessarily slow traffic\". It was **not** retuned, because that would be tuning to a story. Per-carriageway analysis would be the real fix.",
       "**Queue membership** of a stopped vehicle uses two signals: the camera's congestion level, or a stationary group (>= 3)."]))
A(H2("9.3 Wrong-way and wrong-lane"))
A(TABLE(["Item", "Wrong-way", "Wrong-lane"], [
    ["Inputs", "ground points in a configured direction zone; expected direction vector per zone", "lane polygons with `allowed_classes`; vehicle class from the detector"],
    ["Algorithm", "net image-space displacement of the track while inside the zone; cosine between displacement and the expected direction", "ground point inside a lane that excludes the vehicle's class"],
    ["Thresholds", "displacement >= **60 px**, >= **1.5 s** in zone, >= 3 samples, **cos <= -0.7** (about 134 degrees against); a second event within 3 s and 120 px of an already-flagged vehicle is treated as the same vehicle under a new id", "dwell >= 3.0 s"],
    ["Confidence / severity", "severity high; tier **medium** if displacement >= 120 px and >= 3 s, else low", "severity medium, tier low"],
    ["False positives", "tracker identity switch, perspective (direction measured in image space), mis-configured zone", "class errors (van vs truck), occlusion biasing the ground point"],
    ["Demo", "zones configured for the demo road; no wrong-way vehicle occurs; **SIMULATED camera sim-wrong-way** gives exactly 1 event", "not applicable (no allowed-class rules on this road)"],
], [1.1, 2.8, 2.4], "Direction detectors"))
A(H2("9.4 Over-speed, sudden braking and aggressive driving"))
A(TABLE(["Item", "Over-speed", "Sudden braking", "Aggressive-driving candidate"], [
    ["Needs", "validated calibration + `speed_limit_kmh`", "validated calibration", "lane polygons"],
    ["Rule", "smoothed speed > limit + **5 km/h** AND (speed - error) > limit, continuously for **2.0 s**, inside the speed zone", "1.2 s regression speeds sampled each 0.25 s: start >= **40 km/h**; drop >= **25 km/h** and larger than the combined error bands; deceleration >= **3.5 m/s^2** over **1.0-2.5 s**; fall roughly monotonic (a rebound > 5 km/h rejects)", ">= **3** debounced lane changes within **10 s** (a new lane must hold for 0.5 s)"],
    ["Limit used", "112.7 km/h (70 mph) = a **demonstration** threshold; the real signed limit is unknown", "-", "-"],
    ["Confidence", "tier **low** (approximate calibration)", "tier low; severity medium; per-track 30 s debounce", "tier low, severity medium"],
    ["False positives", "scale error; tracker switch", "tracker identity switch or box change can mimic a speed drop; braking is not an offence by itself", "box jitter near a lane boundary"],
    ["Demo", "none (max track speed 116 km/h in the profiled 300 s < 117.7)", "none", "none"],
], [0.9, 1.9, 2.4, 1.6], "Speed-derived detectors"))
A(CALLOUT("note", "**Defects found and fixed in Session 6** (this is a strength to mention): sudden braking produced **199 false events in 160 s** (0.5 s differencing of jittery boxes); over-speed produced **38 events in 270 s** with speeds the estimator never saw (path-length inflation); "
          "a lorry leaving the frame looked like braking. All three were fixed with regression-based speeds, error bands, edge-clip guards and the thresholds above (and regression tests).", "Honesty credit"))
A(H2("9.5 Possible collision (experimental)"))
A(FIG(F("fig_collision.png"), 1.0, "Every gate must pass. Gates 6 and 7 suppress 'two vehicles stopping together'."))
A(P("All quantities are in **box widths per second**, a scale-free unit that does not depend on camera distance or calibration. Box overlap alone never qualifies: it can be perspective or occlusion. "
    "The detector is **experimental**: no collision exists in the footage, no trained accident model exists, and no accuracy is claimed. In the demonstration it is run only on the **SIMULATED** `sim-collision` camera (1 event; it lists the gates it passed) and the `sim-queue` camera (8 stopped cars, **0** collisions)."))
A(H2("9.6 Illegal turn, red light and post-incident disruption"))
A(BUL(["**Red-light:** a track crosses a configured stop line in the approach direction while the signal has been red for >= 1.0 s. The signal state is *not* read from the video; it comes from a schedule, a manual setting or an external feed, and with an unknown state the detector emits nothing. Not applicable here (no signal).",
       "**Illegal turn:** a track enters a rule's entry zone and later its exit zone within 20 s. Needs configured rules; not applicable here.",
       "**Post-incident disruption:** a congestion episode starting within **120 s** after another (non-congestion) incident in the same session is linked as `temporally_associated_with`, always with `causal_claim: false`. Proximity is not causation."]))

# =============================================================== CH 10
A(H1("Chapter 10 - Jam-Cause Investigation"))
A(H2("10.1 The vocabulary"))
A(TABLE(["Term", "Meaning in ANVESH"], [
    ["**Observed congestion**", "the window-level state machine says BUILDING / CONGESTED (a proxy, not a cause)"],
    ["**Stopped vehicle**", "one tracklet that stayed within 0.75 m for >= 3 s (maybe in a queue)"],
    ["**Obstruction**", "a stopped vehicle that is *not* explained by a queue - the candidate for blocking traffic"],
    ["**Possible collision**", "experimental candidate with the gates of Chapter 9"],
    ["**Candidate cause**", "a ranked hypothesis H1-H6 with belief and plausibility; never a probability"],
    ["**Confirmed (established) cause**", "set **only by an authorised officer** with their name and a written basis; detectors never set it"],
], [1.5, 4.8], "Observation versus cause"))
A(H2("10.2 How evidence supports a hypothesis"))
A(P("ANVESH has a closed list of hypotheses (`anvesh/hypotheses/cause_model.py`): **H1** downstream bottleneck, **H2** stopped / disabled vehicle, **H3** collision-like obstruction, **H4** lane blockage (non-incident), "
    "**H5** upstream demand surge, **H6** signal / intersection restriction, and **H7 insufficient evidence** (not a seventh answer but the *ignorance* mass on the whole set). For each window the evidence builder checks observable signals against each hypothesis's signature and records what it found."))
A(P("Evidence is combined with **Dempster-Shafer** theory, a way to combine uncertain evidence without forcing it into probabilities: each item adds *mass* to the hypotheses it supports and the remaining mass stays on \"don't know\". "
    "Per camera, evidence is discounted by a reliability score (a weighted heuristic of observation count, tracking continuity, calibration and temporal coverage). The output is **belief** (how much evidence directly supports a hypothesis) and **plausibility** (how much does not contradict it)."))
A(BUL(["The ranking is **allowed to abstain**: outcomes are `ranked`, `insufficient_evidence` (almost all mass is ignorance, or fewer than 2 hypotheses could be evaluated) and `high_conflict`.",
       "**Missing evidence is not contradicting evidence:** a signal that cannot be computed gets zero mass, not a vote against.",
       "A single-camera ranking is discounted further; with one camera there is no cross-camera confirmation."]))
A(H2("10.3 Why the selected video produced no ranked cause"))
A(P("All four real-footage congestion incidents stored `outcome: insufficient_evidence`, with H1 belief **0.0** and plausibility **1.0** (read from the database): the evidence for every hypothesis was too thin to rank anything. "
    "Specifically: (a) the footage shows **no obstruction or crash**, so H2/H3 have no support; (b) one camera, no downstream or upstream second camera for a bottleneck comparison; (c) with a single camera and the proxy-based load signal there is nothing that favours H1 or H5 over the others. "
    "The incident therefore says: *\"No candidate cause could be ranked from the available evidence.\"* It does not invent one."))
A(CALLOUT("say", "\"Reporting 'no candidate cause' is a feature. A system that always names a cause would be wrong most of the time. ANVESH ranks only when evidence exists, and it keeps 'insufficient evidence' visible.\"", "How to say it"))
A(H2("10.4 How more evidence sources would help in future"))
A(BUL(["**A second camera** upstream/downstream: lets H1 (bottleneck) and H5 (demand surge) be separated by comparing flow and density on both sides, and enables the kinematic shock-wave estimate v = (q2 - q1) / (k2 - k1) implemented in `propagation/shockwave.py` (not used in the single-camera demo).",
       "**Per-lane geometry and signals** to test lane blockage (H4) and signal restriction (H6); the evidence builder's documented V1 notes mark these as not computable from one window.",
       "**A validated accident / obstruction detector** to feed H2/H3 with real evidence.",
       "**Officer feedback** (the feedback module) comparing predictions with what was later observed.",
       "**Labelled data** to calibrate how much belief each evidence type deserves."]))

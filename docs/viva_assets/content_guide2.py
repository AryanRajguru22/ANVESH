from pathlib import Path
from docbuilder import *

FIGS = Path(__file__).resolve().parent / "figures"
F = lambda n: FIGS / n
blocks = []
A = blocks.append

# =============================================================== 4 DIFFICULT MOMENTS
A(H1("4. Exact words for difficult moments"))
A(P("Say these confidently and without apologising. A limitation stated with its technical reason sounds like engineering; the same limitation discovered by the panel sounds like a mistake."))
A(TABLE(["Moment", "What to say"], [
    ["**Why the speed is approximate**", "\"Speed needs real distances, and a camera gives pixels. I calibrated the pixels using motorway lane markings - two-metre dashes every nine metres - and an assumed 3.65-metre lane width. I could not survey the site, so the scale is an assumption. A check on held-out dashes agrees to 43 centimetres, which shows the geometry is consistent, but it cannot prove the nine-metre rule, and the plus-or-minus I display does not include that uncertainty. So I show a tilde and call it approximate. With two surveyed distances it would become defensible, with no code change.\""],
    ["**Why there are no real wrong-way events in the main video**", "\"Nobody drives the wrong way in that footage - I measured it, no wrong-way, collision, braking or over-speed event occurs in the 20 minutes. The rule is configured and unit-tested, and I demonstrate it on a scripted camera that is labelled SIMULATED everywhere. It proves the rule works, not how often it is right on a real road.\""],
    ["**Why collision uses a simulated source**", "\"There is no crash in the video and I have no licensed crash footage or trained accident model. The collision detector is an experimental candidate with several gates, tested on scripted tracks. I show it, and I show that a queue of eight stopped cars does not trigger it, but I claim no accuracy.\""],
    ["**Why number-plate OCR is not available**", "\"Two reasons. The video is 640 by 360, so a plate is about 8 pixels wide - there is nothing to read. And the live plate reader only fires on vehicles at least 110 pixels wide, and OCR on saved incident evidence is not connected: the plate dataset I had was Vietnamese, there is no Indian training data, and the licence of pretrained OCR weights is unverified. So plates are typed in by an officer and verified by a second person; the screen says 'unreadable (too small)'.\""],
    ["**Why the registry is sandbox-only**", "\"A real registry needs departmental authorisation, which a student project cannot have. I built the adapter interface and a synthetic fixture, clearly labelled, and a rule that an operational case cannot be approved without a fresh authorised result - there is no override.\""],
    ["**Why a queue does not automatically mean an accident**", "\"Vehicles stop together in every traffic jam. A collision needs approach, contact and a sudden joint stop, and it must not be explained by a queue or a whole-scene standstill. In the simulated queue, eight cars stopped bumper to bumper produce no collision.\""],
    ["**Why there are multiple track IDs for the same vehicle**", "\"A track ID names one uninterrupted tracker run. If a car is hidden behind a lorry or a far box flickers, the tracker may lose it and later start a new ID. In 20 minutes it created 5,552 IDs for perhaps one or two thousand vehicles. I tested looser settings; they reduced IDs but I cannot prove, without labels, that they do not merge different cars, so I kept the defaults.\""],
    ["**Why it is still research / demo only**", "\"Detector accuracy on real cameras is unmeasured, licences are open - Ultralytics is AGPL and my fine-tuning data is non-commercial - there is no real camera test, no MFA, and no authorised registry or e-challan. The tests pass and the workflow is sound, but those are the blockers before any operational use.\""],
], [1.5, 5], "Scripts for difficult moments"))

# =============================================================== 5 QUESTIONS
A(H1("5. Panel question preparation (30 demonstration questions)"))
A(P("Each answer has three parts: a **concise** answer (say this first), a **deeper** answer if the teacher follows up, and **what to point at** on the screen."))
Q = [
    ("Is the video really processed live?", "It is a recorded file replayed at real speed; detection, tracking, events and evidence are computed while it plays.", "The reader releases frame i at time i/25 s, so the 20 minutes take 20 minutes. There is no whole-video preprocessing and nothing is read from a result file; the same code could take an RTSP stream (not tested on real hardware).", "progress bar and `frames processed` counter rising on the Live page"),
    ("How many frames per second do you process?", "About 25 - the same as the source: 24.89 average over the 20-minute run, 0 dropped.", "FPS = processed frames / time. It is paced to real time, so it shows 'keeps up', not maximum speed. Unpaced it could do roughly 36 FPS (derived from 27.8 ms per frame).", "`processing fps` and `frames ... 0 dropped` in the telemetry"),
    ("What does latency 27 / 38 ms mean?", "Half of the frames finish within 27 ms and 95 % within 39 ms - both under the 40 ms frame interval.", "It is completion time minus start time of `process_frame`. A second figure, end-to-end, also counts waiting in the queue and is about 28 ms median.", "`latency p50 p95`"),
    ("What is the queue depth?", "How many frames are waiting between the decoder and the detector right now; capacity is 8.", "Bounded so memory cannot grow. For a file the policy is BLOCK (reader waits, nothing lost); for live cameras DROP_OLDEST (counted).", "`queue depth now` and `queue high water`"),
    ("What do the numbers on the boxes mean?", "The track identity, the class and, in the magenta zone, speed with a tilde.", "The number is ANVESH's session serial for the tracklet; the raw ByteTrack ID is different and is kept only for audit. It is not a vehicle count.", "a box label and the Tracked vehicles table"),
    ("Why do IDs keep changing?", "Tracks break when a vehicle is hidden or its box flickers; a returning vehicle gets a new ID.", "ByteTrack keeps a lost track for 30 frames (1.2 s) and matches by box overlap, with no appearance model. The distinct-track counter says it is not unique vehicles.", "the line `distinct tracks this session (session-scoped identities, not unique vehicles)`"),
    ("How accurate is the detection?", "Not measured on this video - there are no labels.", "MIO-TCD scores (mAP@50 0.870 fine-tuned) do not transfer; on CityFlow CCTV the fine-tuned model was not better, so the default pretrained model is kept. The first next step is labelling about 200 frames.", "the chip `detector accuracy on your cameras is unmeasured` in the banner"),
    ("How do you get km/h from pixels?", "A homography from 10 lane-marking points converts pixels to metres; speed is the slope of position over time.", "H maps the bottom-centre of the box to the road plane; a least-squares line over 1.6 s avoids jitter; km/h = m/s x 3.6. Fit RMSE 0.094 m, hold-out 0.43 m.", "the magenta speed zone and the geometry notice"),
    ("Why is speed shown only in the magenta zone?", "That is the part of the road where the calibration is trusted.", "Outside it the scale is extrapolated; the stop logic still uses it, which is why far-carriageway stops carry an uncertainty note.", "magenta polygon on the video"),
    ("Is 97 km/h correct?", "It is plausible, not verified.", "Median 97 km/h is the right order for a UK motorway; if the markings were a different standard the same vehicles would read about 65 km/h. That is why it is labelled approximate.", "`~` in the speed label"),
    ("How do you decide the road is congested?", "From a count and occupancy proxy per 10 s window, not from speed alone.", "Mean number of vehicles in view / assumed capacity 20 >= 0.6, or mean speed <= 3 m/s; four states with hysteresis; an episode closes after 60 s of free flow.", "Congestion page strip and the `congestion` incident indicators"),
    ("A moving road can read congested - is that a bug?", "It is a disclosed limitation of the proxy.", "Six lanes in view make 12 vehicles easy at 55-65 km/h. I did not retune it to hide this; the incident says 'dense, not necessarily slow'. Per-carriageway capacity is the fix.", "the uncertainty bullets of a congestion incident"),
    ("How does it tell a queue from an obstruction?", "If the camera is congested or at least three vehicles are stopped together, the stop is a queue member and rated low.", "A lone stopped car in free flow is rated medium, and high after 30 s. The reading is latched so a queued car is not re-rated when the queue clears. The thresholds are untuned policy values.", "the `queue member` chip and severity basis"),
    ("Why 14 incidents for one queue?", "One incident per stopped vehicle, so a human can see each.", "They share a `queue_group_id`; collapsing them into one expandable incident is a listed improvement.", "the incident list filtered to `demo`"),
    ("What is an incident versus an event?", "An event is what a detector produced; an incident is the stored, de-duplicated record with evidence.", "Each event has a natural key; re-sent events update the same row, so no duplicates.", "an incident's Timeline"),
    ("What is the evidence?", "Original frames, annotated frames and a clip, each with a SHA-256.", "Written once, hash in an immutable table, re-verifiable; the original frames are the evidentiary copy.", "Evidence tabs and the sha256 under the clip"),
    ("How do you know evidence was not changed?", "Re-hash the file and compare with the stored hash.", "A change gives a different hash; the stored hash cannot be updated; creation is chained in the audit log. Tamper-evident, not tamper-proof.", "reviewer's **Verify evidence**; admin's **Verify hash chain**"),
    ("Why simulated cameras?", "To show detectors that the real video never triggers.", "Scripted boxes through the same engine and detectors; every record is labelled SIMULATED and the data class cannot become operational.", "purple SIMULATED chip and the NON-PRODUCTION banner"),
    ("Does the collision detector work on real crashes?", "Unknown - none was available.", "It is experimental with gates; the false-positive rate in dense traffic is uncharacterised; a trained model would need licensed crash clips.", "`gates_passed` in the indicators"),
    ("What stops two people doing the whole case?", "The four-eyes rule and the role matrix.", "Approver must differ from the evidence verifier, plate verifier and case creator; the plate verifier differs from the enterer; the administrator cannot approve.", "switching users at each step"),
    ("Why is the plate typed in?", "The live plate reader needs vehicles at least 110 px wide and OCR on saved evidence is not connected, and the footage is too low-resolution.", "A plate is about 8 px wide; manual entry is recorded as `officer_entered` and needs verification by another person.", "the Plate (est. px) column: *unreadable (too small)*"),
    ("Is the registry real?", "No - a synthetic fixture labelled SANDBOX.", "The production adapter is a generic connector that is 'not connected' without authorisation; operational approval stays blocked and there is no override.", "the SANDBOX label on the lookup; chip *Registry: not connected*"),
    ("Is the challan real?", "No - a simulator returns TEST-CHALLAN-...", "The official hand-off returns 409; sandbox hand-offs are rejected for operational cases by a database trigger.", "the `TEST-CHALLAN-` reference"),
    ("What happens if the server crashes?", "Cases, evidence and audit survive; the running session is marked interrupted.", "SQLite WAL on disk; verified by restart check (46 incidents, 2 cases unchanged, chain intact).", "restart without `--reset` and reopen the case"),
    ("What does MFA add and why not done?", "A second factor for sign-in - not implemented.", "Secure TOTP storage needs authenticated encryption and an approved library; no fake MFA was added.", "the banner text *MFA is not implemented*"),
    ("Is this legal to use?", "Only as a research demo.", "Ultralytics is AGPL-3.0, the fine-tuning data is NonCommercial, and the video's origin is unverified, so it is classroom use only.", "banner: *licences are not cleared*"),
    ("Could it run on 100 cameras?", "Not as built.", "One session per camera on one laptop GPU; scaling needs lower fps, batching, several GPUs and a server database - a design, not a tested result.", "one session in `Active processing` on Overview"),
    ("What if FPS falls on a live camera?", "The oldest frames are dropped and counted, affected windows are marked degraded.", "For a recorded file the queue blocks and the video just slows, so no results are lost.", "`frames dropped` and `queue depth`"),
    ("What are you most proud of / what would you do next?", "The honesty engineering: no number without calibration, abstaining cause ranking, audited workflow.", "Next: label 200 frames, survey two distances, per-carriageway congestion, licensed higher-resolution footage for plates, MFA, a real RTSP test.", "the banner and the Congestion uncertainty text"),
    ("What was hardest?", "Making detectors honest on real footage.", "Sudden braking raised 199 false events in 160 s and over-speed 38 in 270 s until I replaced frame differencing with regression-based, error-banded speeds; the stop-episode end bug left 2 of 14 stops 'still open' until fixed.", "(talk through, no screen)"),
]
for i, (q, c, d, p) in enumerate(Q, 1):
    A(QA([(f"P{i}. {q}", f"**Concise:** {c}\n**If pressed:** {d}\n**Point at:** {p}")]))

# =============================================================== 6 TIMING
A(H1("6. Presentation timing"))
A(P("The 20-minute video can run in the background. **Start it as soon as you have signed in** and keep presenting; the spoken parts never wait for it to finish. Congestion appears from about 0:20 of video and the first stopped-vehicle incidents from about 1:50, so real incidents exist after two minutes of talking."))
A(FIG(F("fig_schedule.png"), 1.0, "Three timing plans. The video starts in the first minute and runs underneath all of them."))
A(H2("6.1 Five-minute compressed demonstration"))
A(TABLE(["Time", "Do", "Say"], [
    ["0:00-0:40", "Opening speech (short version); sign in; **Start processing** on Demo motorway", "problem, solution, research/demo-only"],
    ["0:40-1:40", "Live page: boxes, IDs, FPS / latency, one speed label", "\"Everything is computed as the video plays: 25 FPS, 0 dropped.\""],
    ["1:40-2:20", "Congestion page; real incidents list (open one stopped-vehicle incident)", "queue member, low severity, hashed evidence"],
    ["2:20-3:10", "Start `sim-collision` and `sim-queue`; show 1 collision, 0 collisions from the queue", "gates; simulated"],
    ["3:10-4:20", "Case: open -> submit (investigator); sign in reviewer -> verify; show the remaining states from the pre-run case or narrate; officer sandbox hand-off", "four-eyes, sandbox, no override"],
    ["4:20-5:00", "Admin: **Verify hash chain**; closing line", "tamper-evident audit; honest limits"],
], [1, 3.4, 2], "5-minute plan"))
A(CALLOUT("tip", "In five minutes you cannot do five sign-ins live. Narrate the reviewer and second-reviewer steps and show the case page's transition list instead, or rehearse beforehand and say which steps you are skipping.", "Shortcut"))
A(H2("6.2 Ten-minute standard demonstration"))
A(TABLE(["Time", "Do"], [
    ["0:00-1:00", "Opening speech; sign in; point at banner and chips; Cameras -> **Start processing**"],
    ["1:00-3:00", "Live page: detection and tracking, IDs are tracklets, FPS / queue / latency, approximate speed in the magenta zone"],
    ["3:00-4:20", "Congestion page (proxy limit, no ranked cause); open a real stopped-vehicle incident (appears from 1:50)"],
    ["4:20-5:30", "Evidence: clip, original frames, SHA-256, events with this track, uncertainty text"],
    ["5:30-6:50", "SIMULATED wrong-way, collision and queue (start all three; incidents appear within seconds)"],
    ["6:50-9:00", "Case: investigator -> reviewer -> investigator (plate) -> reviewer2 -> officer (lookup, approve, hand-off)"],
    ["9:00-10:00", "Admin: audit log, **Verify hash chain**; closing speech"],
], [1, 5.4], "10-minute plan"))
A(H2("6.3 Fifteen-minute detailed demonstration"))
A(TABLE(["Time", "Do"], [
    ["0:00-1:30", "Opening speech and architecture figure (three layers)"],
    ["1:30-4:00", "Live view in depth: detection, ByteTrack, why IDs fragment, FPS, latency, queue, bounded memory, speed method and its limits"],
    ["4:00-6:00", "Congestion: the proxy, hysteresis, why a moving road can read congested, why no cause is ranked"],
    ["6:00-8:00", "Real stopped-vehicle incident: queue vs obstruction, evidence, hashes, track history"],
    ["8:00-10:00", "SIMULATED wrong-way, collision (gates) and queue; say what each proves and does not"],
    ["10:00-13:00", "Full case workflow with four users; plate placeholder; sandbox registry; hand-off; blocked official path"],
    ["13:00-14:20", "Audit log, **Verify hash chain**; optional restart persistence"],
    ["14:20-15:00", "Closing speech"],
], [1, 5.4], "15-minute plan"))
A(H2("6.4 What to do during playback"))
A(BUL(["While waiting for the first stop (before 1:50): architecture, banner, speed explanation, the simulated cameras, the case on a *simulated* incident if you are short of time.",
       "Use *Pause* only if you must talk for a long time - it stops the video clock (file sources only).",
       "At the end of the video the session shows `completed`, 0 dropped frames and the final statistics."]))

# =============================================================== 7 TROUBLESHOOTING
A(H1("7. Backup and troubleshooting"))
A(CALLOUT("warn", "`--reset` deletes **only** `demo/data` (disposable demo data; it refuses any other directory). Real data in `outputs/app` is never touched. Do **not** delete other folders, and do not use `--fast` in the presentation.", "Safe commands only"))
A(TABLE(["Problem", "Likely cause", "What to do (documented)"], [
    ["App fails to start", "port busy; missing model weights; wrong working directory", "run from `D:\\ANVESH`; use `--port 8765` if busy; weights must exist at `models/yolov8n.pt` (ANVESH never downloads them)"],
    ["Browser cannot connect", "server not running or wrong address", "check the terminal shows `ANVESH DEMO is running: http://127.0.0.1:8760/`; use exactly that address (localhost only)"],
    ["Video does not start", "a session is already active; camera not enabled", "Cameras -> stop the existing session, or restart; message *camera already has an active session*"],
    ["GPU unavailable", "chip shows CPU instead of GPU", "close other GPU applications; check `nvidia-smi`; expect slower processing - say so honestly"],
    ["Processing FPS drops / frames dropped", "other GPU load; thermal throttling", "close other GPU applications; recorded files do not drop frames (the video just slows); monitor with `demo/tools/monitor_run.py --pid <server pid> --out <csv>`"],
    ["Incident does not appear immediately", "stopped / congestion events are detected after a window closes (10 s) plus dwell", "wait; stops appear from about 1:50 of video, congestion from about 0:20; show simulated cameras meanwhile"],
    ["Dashboard shows old incidents", "started without `--reset`", "restart with `" + ".venv\\Scripts\\python.exe scripts\\run_demo.py --reset" + "`"],
    ["Evidence clip does not load", "browser codec / clip still being written", "reload the incident; the clip is H.264 and was verified playable in Chrome; use the *Annotated frames* and *Original frames* tabs; the `libopenh264` message in the server log is harmless"],
    ["Simulated camera does not start", "its session already finished or exists", "stop/refresh Cameras and press **Start processing** again; each finishes in seconds and leaves exactly 1 wrong-way, 1 collision, 8 queue incidents"],
    ["Sandbox workflow fails", "wrong role, same person in two steps, a step skipped", "follow the states in order; use a different user for evidence and plate verification and the officer; a blocked action returns 422 and is audited"],
    ["Server disconnects / terminal closed", "process stopped", "restart **without** `--reset` to keep cases, evidence and audit; a live session is then `interrupted`; start the video again"],
], [1.4, 1.8, 3.2], "Troubleshooting"))
A(H2("7.1 Application restart procedure"))
A(NUM(["Press `Ctrl+C` in the server terminal (wait for `stopping...`).", "Restart: `cd D:\\ANVESH` then `.venv\\Scripts\\python.exe scripts\\run_demo.py` (keeps previous demo data) or add `--reset` for an empty dashboard.",
       "Reload the browser, sign in again, and start the video from Cameras.", "To re-check everything through the API once the server is running: `.venv\\Scripts\\python.exe demo\\tools\\officer_workflow.py --data-dir demo\\data` (repeats steps 6-12 as a dry run)."]))
A(H2("7.2 If the live demo cannot run at all"))
A(BUL(["Show the walkthrough screenshots saved in `demo/runs/walkthrough_freeze/` (15 numbered PNGs from a fresh run: cameras, live video, incidents, incident detail, case, plate, registry, approval, hand-off, audit, audit verified).",
       "Show the Session 7 report numbers (30,001 frames, 0 dropped, 24.89 FPS, p50 27.4 ms, p95 38.9 ms) and explain the architecture figure.",
       "Say plainly that the live run failed; the verified evidence is the 23/23 click-through, the 15/15 workflow check and the 1,381 passing tests."]))

# =============================================================== 8 CLOSING
A(H1("8. Closing speech"))
A(CALLOUT("say", "To summarise: ANVESH turns a fixed-camera video into reviewable road-event candidates, with evidence that can be checked and a workflow that cannot be shortcut. "
          "In the 20-minute run I showed today it processed every one of 30,001 frames at about 25 frames per second, tracked vehicles, estimated approximate speeds, found real congestion and stopped-vehicle incidents, and carried one of them through evidence review, two independent verifications, a sandbox registry check and a sandbox challan - with every step in an audit log we verified at the end.\n"
          "I have been equally careful about what it is not. Speed is approximate, track IDs are not vehicles, detector accuracy on this footage is unmeasured, wrong-way and collision are shown only on simulated cameras, plates are typed by hand, and the registry and challan are simulated.\n"
          "The next steps are clear: label a few hundred frames to measure accuracy, survey two distances to make speed defensible, measure congestion per carriageway, test with licensed higher-resolution footage for plate reading and with a real camera stream, and add multi-factor authentication. "
          "The contribution I am proudest of is that ANVESH is designed to be honest: it gives an officer evidence and uncertainty, and it leaves the decision to a person. Thank you - I am happy to take questions.", "Closing speech - about 75 seconds"))
A(CALLOUT("note", "Everything in this guide was checked against `demo/README.md`, the Session 6 / 7 reports and the app source. If the screen disagrees with this text, trust the screen and the runbook. "
          "Known README differences: the button is **Start processing** (README says *Start*); the simulated cameras finish within seconds of wall time (the README's 15 / 30 / 40 s are video seconds).", "Verification note"))

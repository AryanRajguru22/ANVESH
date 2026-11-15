from pathlib import Path
from docbuilder import *

FIGS = Path(__file__).resolve().parent / "figures"
SCR = Path(__file__).resolve().parent / "screens"
F = lambda n: FIGS / n
blocks = []
A = blocks.append

START = ".venv\\Scripts\\python.exe scripts\\run_demo.py --reset"

# =============================================================== 1 STRATEGY
A(H1("1. Presentation strategy"))
A(P("This guide is your **speaking script and operating manual**. It follows the 13-step runbook in `demo/README.md` (each step was driven in headless Chrome by `demo/tools/presenter_walkthrough.py`: 23/23 passed in Session 7). "
    "The companion *Complete Technical Viva Handbook* explains why everything works; this guide tells you what to click and say."))
A(CALLOUT("warn", "ANVESH is **RESEARCH / DEMONSTRATION ONLY**. The demo video's origin and licence are unverified (classroom use only). Every registry and challan result is **SIMULATED**. "
          "Say these facts yourself, early, before anyone asks.", "Two sentences to own from the start"))
A(H2("1.1 The story in four beats"))
A(TABLE(["Beat", "Goal", "How to say it"], [
    ["**Problem**", "make the panel feel the gap", "Cameras record; they do not tell an officer what happened or prove it. Searching hours of video is slow and evidence is easy to dispute."],
    ["**Solution**", "one sentence, no jargon", "ANVESH watches a fixed camera, finds events as *candidates*, saves hashed evidence, and walks an officer through an audited review."],
    ["**Architecture**", "three layers, not thirty boxes", "Perception (YOLOv8n + ByteTrack) -> reasoning (events) -> application (incidents, evidence, cases, audit). Show only the architecture figure from the handbook, 30 seconds."],
    ["**Live proof**", "show, do not tell", "Start the 20-minute video **first**, then explain while it runs. Everything on the live page is produced while the video plays - nothing is precomputed."],
], [1, 1.6, 3.6], "Narrative"))
A(H2("1.2 Going from talk to demo"))
A(NUM(["Say the opening speech (Section 2) with the dashboard already open on the sign-in screen.", "Sign in as `investigator` and say: *\"I will start the real 20-minute video now and keep talking while it runs.\"*",
       "Press **Start processing**. From here the video is your background clock: congestion appears from about 0:20 of video, stopped-vehicle incidents from about 1:50.",
       "Use the minutes while it runs for the live page, the simulated cameras and the case workflow. Never stand silent waiting for an event.",
       "Come back to real incidents once they have appeared."]))
A(CALLOUT("tip", "Do **not** use `--fast` (testing only) and do not wait for over-speed, braking, aggressive-driving, wrong-way or collision events in the real video: **none occurs** (measured). Show them on the SIMULATED cameras.", "Golden rules"))

# =============================================================== 2 OPENING SPEECH
A(H1("2. Opening speech (60-90 seconds)"))
A(CALLOUT("say", "Good morning. My project is called ANVESH - a road event investigation and traffic intelligence system.\n"
          "The problem is simple. A city has far more CCTV cameras than people watching them, and CCTV only records. When a vehicle stops in a lane or traffic suddenly jams, an officer has to find the camera, search the video, work out what happened and keep proof that nobody can dispute.\n"
          "ANVESH takes a fixed-camera video and does that work step by step. A YOLOv8n detector finds the vehicles, ByteTrack follows them from frame to frame, and a calibration from lane markings converts positions to metres so I can estimate speed. On top of that, detectors raise candidate events: stopped vehicles, congestion, over-speed, sudden braking, wrong-way driving and an experimental collision candidate.\n"
          "Each event becomes an incident with the original frames, an annotated clip and SHA-256 hashes. An investigator opens a case, a reviewer verifies the evidence, a second reviewer verifies the plate, and an officer approves it to a sandbox challan - every step recorded in a tamper-evident audit log.\n"
          "I want to be clear about what this is: a research and demonstration system. Plate reading, the vehicle registry and the e-challan are not connected - the registry and challan you will see are simulated - and the speed is approximate. Let me show you how it works on a real 20-minute video.", "Opening speech - about 85 seconds spoken"))
A(P("**Why it works:** it names the problem in one breath, lists the technologies without lecturing, describes the officer workflow, and states the limitations *before* the panel finds them. Practise it until it flows; do not read it."))

# =============================================================== 3 WALKTHROUGH
A(H1("3. The live demonstration walkthrough"))
A(P("Before you start: the camera feed is a **recorded file replayed at video speed**; the dashboard is a browser page served from `http://127.0.0.1:8760/`; five demo accounts exist (`investigator`, `reviewer`, `reviewer2`, `officer`, `admin`) with random passwords in `demo/data/initial_credentials.txt` - open that file only on your own screen, never on a projector. "
    "The sign-in screen offers no shortcuts. The four-eyes rule means you must **sign out / sign in** between roles (every action also opens a small *Confirm* dialog)."))
A(TABLE(["Runbook step (`demo/README.md`)", "Parts below"], [
    ["1 Start ANVESH", "3.1"], ["2 Sign in", "3.2, 3.3"], ["3 Start the 20-minute video", "3.4, 3.5"], ["4 Live monitoring", "3.6 - 3.9"], ["5 Real stopped / congestion incident", "3.10"], ["6 Evidence and vehicle history", "3.11"],
    ["7 SIMULATED wrong-way", "3.12"], ["8 SIMULATED collision", "3.13"], ["9 SIMULATED queue", "3.14"], ["10 Create and review a case", "3.15 - 3.17"], ["11 Sandbox registry and challan", "3.18, 3.19"], ["12 Audit trail", "3.20"], ["13 Stop and reset safely", "3.21"],
], [3, 1.5], "Mapping to the 13-step runbook"))

def step(num, title, action, screen, say, tech, expect, trans):
    A(H2(f"3.{num} {title}"))
    A(TABLE(["Item", "What to include"], [["**Action**", action], ["**Screen**", screen], ["**What to say**", say], ["**Technical explanation**", tech], ["**Expected result**", expect], ["**Transition**", trans]], [1, 5.3]))

step(1, "Starting the application",
     "Terminal 1: `cd D:\\ANVESH` then `" + START + "`. Wait for the line `ANVESH DEMO is running: http://127.0.0.1:8760/`. Open that address in Chrome, full screen (1920x1080 or 1366x768 both checked). Terminal 2: `type demo\\data\\initial_credentials.txt` (private screen only).",
     "a terminal printing `registered camera demo-motorway`, the DEMO accounts and the safety banner; then the sign-in screen (title ANVESH, \"Road event investigation and traffic intelligence\", username and password, **Sign in**).",
     "\"I start the system on a disposable data folder, so the dashboard starts empty - no incident from an earlier run can appear. Nothing is processed until I press start.\"",
     "`run_demo.py` builds the services on `demo/data` (git-ignored), creates flagged DEMO accounts, registers the camera `demo-motorway` (`demo.mp4` + `demo/camera_config.json`) and three SIMULATED cameras, and serves the dashboard with the standard-library `ThreadingHTTPServer`. `--reset` deletes **only** `demo/data` (it refuses any other directory or one without the `.anvesh-demo-data` marker); `outputs/app` is never touched.",
     "the line `ANVESH DEMO is running`, then a clean sign-in page. No GPU work yet.",
     "\"Let me sign in as the investigator - the person who runs monitoring and opens cases.\"")
step(2, "Logging in",
     "Type `investigator` and the password from the file -> **Sign in**.",
     "the dashboard Overview: left menu (Overview, Cameras, Live monitoring, Incidents, Violations / e-challan, Congestion, History & audit), top-right *Investigator (DEMO ACCOUNT)*, **Sign out**.",
     "\"Each role sees only what it may do. This is the investigator: start monitoring, open cases, enter plates. A reviewer verifies, an officer approves, an administrator manages the system.\"",
     "Passwords are stored as scrypt hashes; the session token is 256-bit random and only its SHA-256 is stored; the cookie is HttpOnly and SameSite=Strict; 8 h lifetime, 30 min idle; five wrong attempts lock that user/address for 5 minutes.",
     "Overview opens; the menu shows only permitted pages.",
     "\"Before anything else, notice the banner at the top.\"")
step(3, "Explaining the research / demo-only banner",
     "Point at the amber banner and the chips; do not click.",
     "amber banner: *RESEARCH / DEMO USE ONLY - not cleared for operational enforcement: no authorised registry or e-challan is connected, detector accuracy on your cameras is unmeasured, licences are not cleared, and MFA is not implemented. You are signed in with a DEMO account.* A blue banner: *Detections are automated CANDIDATES for human review, not findings.* Chips: **OCR: unavailable, Registry: not connected, e-Challan: not connected, Alerts: test adapter only, GPU**.",
     "\"The system tells the truth about itself on every page. OCR is unavailable, the registry and e-challan are not connected, alerts go to a test adapter, and everything it raises is a candidate for a human to review. The GPU chip shows my laptop's NVIDIA GPU is being used.\"",
     "The banner is built into the front-end layout; the chips come from `GET /api/v1/system/status` (integration states and `torch.cuda.is_available()`).",
     "banner and five chips visible on every page.",
     "\"Now the camera. Cameras -> Demo motorway.\"")
step(4, "Opening the real demonstration camera",
     "Left menu **Cameras**; find **Demo motorway (fixed overpass camera, recorded)**; read the geometry line.",
     "the camera card with a `demo data` class chip and the geometry note *APPROXIMATE metric - scaled from a published lane-marking module ... site NOT surveyed*; below it the three SIMULATED cameras (`sim-collision`, `sim-wrong-way`, `sim-queue`) and a **Start processing** button on each.",
     "\"This is a real recorded video from a fixed camera above a motorway, 640 by 360 pixels at 25 frames per second. Its source and licence are unverified, so I use it for classroom demonstration only. The speed here is approximate and labelled so.\"",
     "The camera row stores source file, `camera_config.json` (calibration, speed zone, lanes, direction zones, speed limit, 1200 s cap, congestion grace 60 s) and the data class `demo`.",
     "camera visible with its config summary.",
     "\"I will press Start now and talk while it runs.\"")
step(5, "Starting the 20-minute video",
     "Click **Start processing** on Demo motorway (Confirm if asked). The live page opens (or *Live monitoring* -> open).",
     "the session page: annotated video on the left, telemetry on the right, progress bar `Video 0.0s of 1200.0s`.",
     "\"It runs for 20 minutes at real speed. Frame by frame it is detected, tracked, analysed and saved while it plays. Nothing is precomputed; I could pause it, but I will not.\"",
     "`ProcessingSession` starts three threads (reader, worker, consumer) joined by bounded queues; the file source paces frames to `realtime_factor` 1.0; YOLOv8n + ByteTrack run on the GPU; the first frames include model warm-up.",
     "state chips `live` and `engine live`, `frames processed` rising, `0 dropped`.",
     "\"Let me show you what you are looking at.\"")
step(6, "Explaining live detection and tracking",
     "Point at boxes, numbers and arrows on the video; point at the **Tracked vehicles** table.",
     "coloured boxes with `00051 car`-style labels, yellow direction arrows, green lane polygons, a magenta speed zone; table columns ID, Class (confidence), Speed, Plate (est. px), Plate status.",
     "\"Each box is a detection from YOLOv8n. The number is the track identity ByteTrack gives it so I can follow the same vehicle across frames. Notice it says 'distinct tracks this session - not unique vehicles': if a car is hidden for two seconds it gets a new number.\"",
     "`model.track(conf=0.25, classes=[bicycle, car, motorcycle, bus, truck])`; ByteTrack uses a Kalman filter and IoU matching; ANVESH maps the raw ID to a session identity; the arrow is the image-space heading of the ground point over 1 s.",
     "boxes follow vehicles; class labels (car / truck / bus) change occasionally; plate status says *unreadable (too small)*.",
     "\"Next to it are the performance numbers - those matter for real time.\"")
step(7, "Explaining FPS, processing speed and latency",
     "Point at the telemetry block: source fps, processing fps, frames, queue depth, latency p50 / p95, queue high water.",
     "`source fps 25.0`, `processing fps ~24.9 (average since start)`, `frames N processed, 0 dropped`, `queue depth 0 / 8`, `latency p50 p95 27 / 38 ms`, `queue high water frames 8/8, dropped 0`.",
     "\"The source is 25 frames per second, so a frame arrives every 40 milliseconds. ANVESH processes about 25 frames per second, with a typical frame taking 27 milliseconds and 95 percent under 39, so it keeps up. In my 20-minute test: 30,001 frames, none dropped, memory flat. That proves it runs in real time on my laptop; it does not prove accuracy.\"",
     "FPS = processed frames / time since the session object was created; latency = `frame_processing` (start to end of `process_frame`); the frame queue has capacity 8 with BLOCK policy for files, so a slow worker would slow the video, never drop frames.",
     "numbers stable; processing fps early in the run reads slightly under 25 (warm-up) and settles near 24.9-25.",
     "\"One more number on the video: speed.\"")
step(8, "Showing approximate speed",
     "Wait for a vehicle in the magenta zone (bottom right, toward the camera); point at the `~97 km/h`-style label and the table's Speed column with +/-.",
     "labels with a tilde, for example `00xx car 96km/h~`, only inside the magenta near-field zone; elsewhere the table says *outside speed zone*.",
     "\"The tilde means approximate. The scale comes from motorway lane markings - two-metre dashes with seven-metre gaps - and an assumed lane width, because I could not survey the site. The plus-or-minus covers measurement noise, not that scale assumption, so the real uncertainty is larger, perhaps ten percent. I would not use this to issue a speeding ticket.\"",
     "10 reference points -> homography H (fit RMSE 0.094 m, hold-out 0.43 m) -> least-squares slope of ground positions over 1.6 s x 3.6; shown only inside the speed zone and only because a validated calibration exists.",
     "typical values 85-110 km/h (median about 97 km/h in the profiled segment).",
     "\"Now the traffic state: the Congestion page.\"")
step(9, "Opening congestion analysis",
     "Left menu **Congestion** (the level strip and trend). At least 30 s of video must have elapsed.",
     "a level strip per 10 s window (free flow / building / congested / dissipating) and a trend; the first episode starts at 20 s.",
     "\"The level is a count and occupancy proxy against an assumed capacity of 20 vehicles - it is not a speed measurement. So dense traffic that is still moving can read 'congested'; the incident says so. And because there is no obstruction or crash in this footage, the cause ranking says 'no candidate cause could be ranked'. It does not invent one.\"",
     "per window: mean concurrency / 20 = occupancy; congested if mean speed <= 3 m/s or occupancy >= 0.6; hysteresis through BUILDING / DISSIPATING; an episode closes after 60 s of free flow; cause ranking uses Dempster-Shafer and returned `insufficient_evidence`.",
     "chips such as `building`, `congested`; later the first congestion incident (video 20-170 s) in Incidents.",
     "\"While more video plays, let me show you what an incident looks like.\"")
step(10, "Opening a real stopped-vehicle incident",
     "From about 1:50 of video: **Incidents** -> set the filter **all data classes** to `demo` (real footage) -> open a *stopped vehicle* incident.",
     "list rows with thumbnails, a `queue member` chip, severity `low`, status `detected`; the incident page opens with Evidence, Observed event, Uncertainty and limits, Vehicles, Candidate causes.",
     "\"This is a real candidate from the video. ANVESH rated it low and marked it 'queue member', because the road is congested and several vehicles are stopped together: that pattern is queued traffic, not an obstruction. A lone stopped car in free-flowing traffic would be rated medium, or high after 30 seconds. All 14 real stops here are queue members.\"",
     "stop = ground points within a 0.75 m radius of the mean of the last 3 s for >= 3 s; queue = congestion level BUILDING/CONGESTED/DISSIPATING or >= 3 vehicles stationary; the queue reading is latched per stop; one incident per vehicle is kept so a human can see each.",
     "incident page with severity basis *stopped while the camera is congested: most likely queued traffic, not an obstruction*.",
     "\"What makes it evidence is what is stored with it.\"")
step(11, "Showing the evidence and track history",
     "On the incident page: Evidence tabs (Clip, Annotated frames, Original frames) -> play the clip, open an original frame in the lightbox, read the SHA-256; then Vehicles -> **events with this track**; open *Timeline*.",
     "an H.264 clip with a banner `demo-motorway | video t=... `, thumbnails, `sha256 ef577ce...`, the vehicle's track id and the line *a track id names one uninterrupted tracker run, not a vehicle*.",
     "\"The original frames are saved untouched; the annotated frames and clip are review aids. Every file has a SHA-256 hash stored in an immutable table, so anyone can re-check it was not changed. The track id names one tracker run, not a car: if the tracker lost the vehicle, it has another id, so the history can be incomplete - the page says so.\"",
     "frame ring (JPEG, <= 10 fps, 43 s) -> key frames saved with exclusive create + fsync -> SHA-256 -> `evidence_assets` row + audit entry; clip encoded H.264 (`avc1`).",
     "clip plays in Chrome; frames open; hashes visible.",
     "\"The real video has no wrong-way vehicle and no crash, so for those rules I use clearly labelled simulated cameras.\"")
A(FIG(SCR / "live.jpg", 0.62, "What the panel sees at steps 6-8 (screenshot from the freeze walkthrough)."))
step(12, "Opening the simulated wrong-way scenario",
     "**Cameras** -> `sim-wrong-way` -> **Start processing**; then **Incidents** -> **wrong-way (candidate)** with a purple SIMULATED chip.",
     "a banner *NON-PRODUCTION DATA (simulated). This incident came from a scripted simulation, not real footage*; the Observed event lists the zone and direction cosine.",
     "\"This is scripted boxes, not footage. It only shows that the rule works: the zone has an expected direction, and a vehicle travelling more than about 134 degrees against it for 1.5 seconds raises a candidate. It says nothing about how often it is right on a real road.\"",
     "`WrongWayDetector`: net displacement >= 60 px, >= 1.5 s in zone, cosine to expected <= -0.7; duplicates within 3 s / 120 px are the same vehicle under a new id.",
     "exactly **1** wrong-way incident within seconds (the scenario is finite and unpaced; the README's \"about 15 s\" is video seconds).",
     "\"The harder one: a collision.\"")
step(13, "Opening the simulated collision scenario",
     "**Cameras** -> `sim-collision` -> **Start processing**; **Incidents** -> **possible collision (candidate)** (SIMULATED); open it and read the Indicators.",
     "`gates_passed: moving_before_contact, closing, sudden_stop, not_a_stationary_queue, not_a_standstill`; severity high, confidence low.",
     "\"Collision is experimental. I have no trained accident model and no real crash to test on, so no accuracy is claimed. The rule needs several things at once: two vehicles moving and closing, boxes in contact, then a sudden joint stop - and not a queue or a standstill. Box overlap alone never counts.\"",
     "`CollisionCandidateDetector` works in box widths per second (scale-free): IoU >= 0.10 for >= 1 s, moving > 0.8 widths/s before contact, closing >= 0.15 widths, both <= 0.15 widths/s for 2 s within 2.5 s of contact; suppressed for >= 3 stopped neighbours or >= 60 % of others stopped.",
     "exactly **1** possible-collision incident (SIMULATED).",
     "\"The most important test is the opposite one: a queue.\"")
step(14, "Showing that a simulated queue does not create a collision",
     "**Cameras** -> `sim-queue` -> **Start processing**; **Incidents**: stopped-vehicle incidents with *queue member* chips and a congestion incident, and **no** collision.",
     "eight stopped-vehicle incidents (SIMULATED) and a congestion incident; the collision filter shows only the earlier one.",
     "\"Eight cars stopped bumper to bumper. Two vehicles stopping together is not an accident, so ANVESH raises stopped-vehicle and congestion incidents and no collision.\"",
     "boxes touch but nobody was moving before contact (gate 2) and >= 3 neighbours were already stationary (gate 6): `suppressed` counters record why.",
     "8 queue members, 0 new collisions.",
     "\"Now what an officer does with a real incident: the case.\"")
step(15, "Creating a case (investigator)",
     "Open a real stopped-vehicle incident -> in **Violation cases** choose the offence (`STOP-OBS`) -> **Open violation case from this incident** -> Confirm -> open the case under **Violations / e-challan** -> **Submit for evidence verification** -> Confirm.",
     "a case page with state `detected`, then `awaiting_evidence_verification`; a checklist of requirements.",
     "\"A case is a work item built on one incident. ANVESH never decides enforcement. The case moves only through explicit, named, audited steps, and no single person can do the whole chain.\"",
     "`CaseService` state machine; each transition writes `case_transitions` and an audit row; `STOP-OBS` is the offence catalogue entry for stopped vehicles; rule references are the department's to configure (none shipped).",
     "state `awaiting_evidence_verification`.",
     "\"Now a second person: the reviewer.\"")
step(16, "Reviewer evidence verification",
     "**Sign out**, sign in as `reviewer` -> the case -> **Verify evidence** -> Confirm.",
     "state `awaiting_plate_verification`; the check *evidence_captured: N frame(s), 1 clip(s), hashes intact*.",
     "\"The reviewer re-checks the stored files against their hashes and confirms frames and a clip exist. If a file had been altered, this step would fail.\"",
     "`EvidenceStore.verify` re-hashes each asset (SHA-256) and compares to the immutable row.",
     "`awaiting_plate_verification`.",
     "\"Next the plate. As I will explain, nothing reads it automatically.\"")
step(17, "Manual demo plate entry (and second-reviewer verification)",
     "Sign in as `investigator` -> incident page -> **Enter plate manually** -> type the placeholder `MH12AB1234` -> Save. Then sign in as `reviewer2` (a *different* person) -> **Verify plate**.",
     "a plate reading `officer_entered`, then `officer_verified`; the tracked-vehicle table still says *unreadable (too small)*; the case becomes `ready_for_review`.",
     "\"This plate is a demo placeholder, typed by me. It was not read from the footage: at 640 by 360 a plate is about 8 pixels wide and the live plate reader only fires on vehicles at least 110 px wide and OCR on saved evidence is not connected. The plate must be verified by someone other than the person who entered it.\"",
     "`PlateService`: readings are immutable rows; manual entries are format-checked (Indian plate patterns) and marked `officer_entered`; verification needs a different user; OCR output could never be marked verified.",
     "`ready_for_review`.",
     "\"Now the officer and the registry.\"")
step(18, "Sandbox registry lookup",
     "Sign in as `officer` -> the case -> **Look up registration (sandbox)** -> Confirm.",
     "a lookup result labelled **SANDBOX / synthetic** (*synthetic test data, not a real registry result*).",
     "\"There is no real government registry connection. This is a synthetic fixture file. And notice there is no override button: for an operational case, approval would be impossible without a fresh result from an authorised registry.\"",
     "`SandboxRegistryAdapter` reads `configs/sandbox_registry.json`; the lookup is bound to the verified plate reading and audited; a database trigger rejects sandbox lookups on operational cases.",
     "lookup `found` (sandbox) recorded.",
     "\"With that, the officer can approve.\"")
step(19, "Officer approval and sandbox challan hand-off",
     "**Approve for submission** -> Confirm; then **Submit to SANDBOX simulator** -> Confirm.",
     "state `approved_for_submission`, then `submitted` / `challan_reference_received` with a reference `TEST-CHALLAN-...`.",
     "\"The officer must be different from whoever verified the evidence, verified the plate and created the case. The hand-off goes to a simulator that returns a TEST reference. The official e-challan hand-off is blocked and no real system is contacted.\"",
     "four-eyes check in `CaseService`; `HandoffService` re-checks the registry rule; `official` hand-off returns 409; demo accounts cannot write operational data (403).",
     "reference `TEST-CHALLAN-...`, state *challan reference received*.",
     "\"Finally, the proof of everything that just happened: the audit trail.\"")
A(FIG(SCR / "handoff.jpg", 0.5, "Case page after the sandbox hand-off (screenshot from the freeze walkthrough)."))
step(20, "Showing audit history and evidence integrity",
     "Sign in as `admin` -> **History & audit** -> **Audit log** -> **Verify hash chain**.",
     "the audit list (newest first) with each login, session, evidence add, case action, lookup and hand-off; a toast *Audit chain intact: N entries*.",
     "\"Every step was recorded by an append-only log where each row's hash includes the previous row's hash. If anyone edited or deleted a row, verification would point to the first broken row. It is tamper-evident, not tamper-proof: someone who could rewrite the database and the code could rewrite the chain.\"",
     "`hash = SHA-256(prev_hash + canonical(row))`; SQL triggers reject UPDATE/DELETE; `GET /audit/verify` re-derives the chain.",
     "toast with the entry count (in the verified run: 407 rows at that point).",
     "\"That is the whole path from a video frame to an audited sandbox challan.\"")
step(21, "Closing the demonstration",
     "Leave the video running to its end or stop it; return to the Live page: state `completed` at 20:00 with **0 dropped frames**. Optional (1 min): `Ctrl+C` in the server terminal, restart **without** `--reset`, sign in: the case, evidence and audit are still there and a running session shows `interrupted`.",
     "completed session; or the restored data after restart.",
     "\"At 20 minutes: 30,001 frames, zero dropped. If I restart the server the cases, evidence and audit survive - they are in the database.\" Then give the closing speech (Section 8).",
     "`ProcessingSession._release` frees the ring, history and GPU memory; SQLite WAL keeps durable state.",
     "state `completed`; after a restart, `interrupted` for any session that was live.",
     "\"Let me finish with what I would do next.\"")

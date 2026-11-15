# ANVESH presentation demo kit

**RESEARCH / DEMONSTRATION ONLY.** Not approved for traffic-police deployment, real challans or legally binding decisions. Every registry and
challan result in this demo is **SIMULATED**; nothing is sent to any government system.

The kit runs the application on a **disposable data directory** (`demo/data`, git-ignored) - never `outputs/app`. Detection, tracking, events and
evidence are produced **while the video plays** (video speed, `realtime_factor = 1.0`); there is no whole-video preprocessing and no pre-computed result.

| File | Purpose |
|---|---|
| `scripts/run_demo.py` | starts the server on a fresh/disposable data dir, registers the camera and the SIMULATED cameras, optionally auto-starts the session |
| `demo/camera_config.json` | camera configuration: calibration, speed zone, lanes, direction zones, speed limit, 1200 s cap, congestion close grace |
| `demo/SOURCE.md` | the video: provenance, licence status, suitability comparison, which events it can honestly show |
| `demo/CALIBRATION.md` | how the km/h scale was obtained, and exactly how approximate it is |
| `demo/tools/` | reproducible helpers (config builder, plausibility check, scene-cut scan, detector and tracker studies, performance monitor, officer-workflow check, browser checks, test-group runner) |
| `demo/runs/` | outputs of the verification runs (CSV/JSON/screenshots; git-ignored) |

## Accounts (all DEMO accounts; the random passwords are in `demo/data/initial_credentials.txt`)

| User | Role | Used for |
|---|---|---|
| `investigator` | investigator | start/stop the session, open incidents, create the case, enter the plate |
| `reviewer`, `reviewer2` | reviewer | verify evidence (`reviewer`), verify plate (`reviewer2`: a different person) |
| `officer` | officer | sandbox registry lookup, approval, sandbox hand-off; may read the audit log |
| `admin` | administrator | users, retention, cameras, **verify the audit hash chain** |

The sign-in screen offers no shortcuts. The four-eyes rule means one person cannot do the whole chain, so switch users (*Sign out* / *Sign in*) at the steps below.
Open the credentials file on your own screen only; do not paste it into slides, chat or the report.

## The presentation, step by step (each step was driven in headless Chrome: `demo/tools/presenter_walkthrough.py`)

The video session runs 20 minutes and keeps running while you present the simulated cameras and the case workflow. Plan about 18-20 minutes.

### 1. Start ANVESH

```powershell
cd D:\ANVESH
.venv\Scripts\python.exe scripts\run_demo.py --reset
```

Wait for `ANVESH DEMO is running: http://127.0.0.1:8760/`. Open that address in Chrome (full-screen, 1920x1080 or 1366x768 both checked).
`--reset` deletes **only** `demo/data` (it refuses any directory outside `demo/` or without the `.anvesh-demo-data` marker), so the dashboard starts empty: no incident from an
earlier run can appear. Without `--reset` the previous demo data (cases, audit) are kept. `--autostart` starts the video immediately (acceptance runs; for the presentation start it from the dashboard).
`--fast` processes as fast as possible instead of at video speed: **testing only**.

### 2. Sign in

`type demo\data\initial_credentials.txt` in a second terminal, then sign in as `investigator`. Point at the banner: *RESEARCH / DEMO USE ONLY*, and the status chips
(OCR unavailable, Registry not connected, e-Challan not connected, alerts test adapter only).

### 3. Start the 20-minute video

*Cameras* -> `Demo motorway` -> **Start**. Note the geometry line: "APPROXIMATE metric - ... site NOT surveyed". Do **not** use `--fast`.

### 4. Show live monitoring

The live page opens (or *Live monitoring* -> *open*). Left: annotated frames of the video **being processed now** (boxes, stable track IDs, class, direction arrows, speed in km/h with
`~` for approximate, shown only inside the magenta near-field zone), the progress bar and the live event feed. Right: source fps 25, processing fps, frames dropped, queue depth, latency p50/p95
and a table of tracked vehicles (speed with +/- band, plate readability: "unreadable (too small)"). Nothing is pre-computed - *Pause* stops the video clock if you need to talk.
Then *Congestion*: the level strip and trend. Say plainly: the level is a count/occupancy proxy; dense but flowing traffic can read "congested" and the incident says so;
no obstruction or crash appears in this footage, so the cause ranking says "no candidate cause could be ranked" - it does not invent one.

### 5. Open a real stopped-vehicle or congestion incident

From about 1:50 of video the first stopped-vehicle incidents appear (a queue forming on the away-from-camera carriageway; congestion from about 0:20). *Incidents* -> filter *data classes* = `demo data`
(real footage) -> open one. Each is a **candidate**. Stopped vehicles inside a queue are rated `low` and carry a **queue member** badge: they are judged as queued traffic, not as obstructions,
and the same stop is not re-rated "isolated" when the queue clears. Congestion episodes merge free-flow gaps shorter than 60 s into one stop-and-go episode.
**Measured: sudden-braking, over-speed, aggressive-driving, wrong-way and collision events do NOT occur in this video - do not wait for them.** Show them on the SIMULATED cameras (steps 7-9).

### 6. Show the evidence and the vehicle history

On the incident page: *Evidence* (annotated clip, annotated frames, original frames - open a frame in the lightbox), the SHA-256 hashes, *Observed event*, *Uncertainty and limits*, the
indicators (why it fired) and severity/confidence basis. In *Vehicles*, **events with this track** lists the other incidents that involve the same tracker identity in this session. Say what it is:
a track id names one uninterrupted tracker run, not a vehicle; if the tracker lost the vehicle it has another id, so that history can be incomplete. *Timeline* shows each system step.

### 7. The SIMULATED wrong-way camera

*Cameras* -> `sim-wrong-way` -> **Start** (runs about 15 s). *Incidents* -> **wrong-way (candidate)** with a purple **SIMULATED** badge and a "NON-PRODUCTION DATA (simulated)" banner on the page.
The rule needs a configured direction zone (expected travel direction per zone); it fires only for travel > 135 degrees against it. It is scripted boxes, not footage - it shows the rule works, not how often it is right on a road.

### 8. The SIMULATED collision scenario and its gating

*Cameras* -> `sim-collision` -> **Start** (about 30 s). *Incidents* -> **possible collision (candidate)** (SIMULATED). Open it: *Indicators* list the gates it passed - closing before contact,
contact, joint stop within 2.5 s of first contact, not part of a stationary queue, not a whole-scene standstill. It is **experimental**: no trained accident model exists and no accuracy is claimed. Box overlap alone never qualifies.

### 9. The queue scenario does not trigger a collision

*Cameras* -> `sim-queue` -> **Start** (about 40 s): eight vehicles stopped bumper to bumper. *Incidents* shows stopped-vehicle incidents with **queue member** badges and a congestion incident, and **no** collision.
"Two vehicles stopping together is not an accident."

### 10. Create and review a case

Every action opens a small *Confirm* dialog. Investigator: open a stopped-vehicle incident -> **Open violation case from this incident** -> **Submit for evidence verification**.
Sign out / in as `reviewer`: **Verify evidence**. As `investigator`: open the incident -> **Enter plate manually** (type the placeholder `MH12AB1234`) -> Save.
The plate is a **DEMO PLACEHOLDER**, not read from the footage and not registry-verified (the tracked-vehicle table says "unreadable (too small)": a plate is about 8 px wide at 640x360; no plate detector or OCR is installed).
As `reviewer2`: **Verify plate** (a different person).

### 11. Sandbox registry and challan

As `officer`: **Look up registration (sandbox)** (labelled **SANDBOX / synthetic**), **Approve for submission**, **Submit to SANDBOX simulator** (reference `TEST-CHALLAN-...`, state *challan reference received*).
Point at: there is no registry-override control; the official e-challan hand-off is blocked; an operational case could not be approved without a real registry result; demo accounts cannot write operational data.

### 12. The audit trail

As `admin`: *History & audit* -> *Audit log* (append-only, every step above) -> **Verify hash chain** (toast "Audit chain intact: N entries").

### 13. Stop and reset safely

* End of video: the session state turns `completed` at 20:00; *Live monitoring* shows 0 dropped frames and flat memory.
* Optional restart proof (1 minute): `Ctrl+C` in the server terminal, then start it again **without** `--reset`; the case, evidence and audit are still there, and a session that was running is marked `interrupted`, not live.
* Reset for the next run: `.venv\Scripts\python.exe scripts\run_demo.py --reset` (only `demo/data` is deleted). `Ctrl+C` stops the server.

`demo/tools/officer_workflow.py --data-dir demo/data` repeats steps 6-12 through the API and asserts the safety rules (dry run).

## Checks you can re-run (the server must be running)

```powershell
python demo\tools\presenter_walkthrough.py --data-dir demo\data --keep-real-running    # steps 1-12 through the browser (use --keep-real-running during the 20-minute run)
python demo\tools\ui_acceptance.py --data-dir demo\data                                 # live page, evidence images, clip playback, banner, overflow
python demo\tools\ui_resolutions.py --data-dir demo\data                                # every page at 1920x1080, 1366x768, 1280x720, 1024x768
.venv\Scripts\python.exe demo\tools\officer_workflow.py --data-dir demo\data            # API-level workflow and safety rules
bash demo/tools/run_test_groups.sh demo/runs/test_groups.log                            # all tests, 9 sequential processes
```

## What is real, experimental, unavailable

| Capability | Status in this demo |
|---|---|
| Vehicle detection + tracking (YOLOv8n pretrained COCO, ByteTrack) | **real**, runs on the video as it plays; accuracy on this footage is not measured (no labels) |
| Track identity | **tracklets**: ids fragment (about 800 raw ids per 3 minutes of this footage); never a physical vehicle identity |
| Vehicle classes | real but coarse (car/bus/truck from COCO); lorries are sometimes called "bus"; the class of one track can flicker |
| Counts, direction arrows, lane assignment | real, image-space |
| Speed (km/h) | **approximate**: calibrated from the lane-marking standard (see `CALIBRATION.md`), near field only, shown with +/- that excludes the scale uncertainty |
| Stopped vehicle, queue vs. obstruction | heuristic; queue stops rated low and latched for the whole stop; precision unmeasured; judged on the approximate metric scale over the whole frame |
| Congestion level | heuristic count/occupancy proxy against an assumed capacity; flowing dense traffic can read congested; flapping merged by a 60 s close grace |
| Jam cause | hypothesis ranking with stated uncertainty; reports "none" when the evidence is absent |
| Over-speed, sudden braking | candidates from the approximate speed; demonstration threshold 112.7 km/h; no event in this footage |
| Aggressive driving | candidate: 3+ lane changes in 10 s; never a confirmed offence; no event in this footage |
| Wrong-way, wrong-lane | configured and unit-tested; **SIMULATED camera only**, not demonstrated on real footage |
| Collision | experimental candidate with queue/standstill/gradual-stop gates; **SIMULATED only**; no trained model; not validated |
| Plate detection / OCR | **not available**; manual entry only (DEMO PLACEHOLDER) |
| Red light, illegal turn, helmet, seat belt | not applicable / not built |
| Registry, challan | **sandbox only**; no real integration; operational approval blocked by design |
| MFA | **not implemented** |

## Troubleshooting

* *Dashboard shows an old session / old incidents:* you started without `--reset`; the previous data directory is reused. Use `--reset`.
* *"camera already has an active session":* stop the existing session in *Cameras*, or restart with `--reset`.
* *Slow video / dropped frames:* close other GPU applications; `nvidia-smi` should show the GPU in use. The perf CSV tool is `demo/tools/monitor_run.py --pid <server pid> --out <csv>`.
* *Port busy:* `--port 8765`.
* The server prints a harmless `libopenh264 ... Unable to create encoder` message when it falls back to another codec for evidence clips.

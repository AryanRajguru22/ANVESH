# Demonstration video: source, suitability and licence

**Chosen source:** `D:\ANVESH\demo.mp4` (repository root, git-ignored, never committed), first **1200 s (20 min)** of its 34 min 8 s. The
file is processed unmodified; nothing is trimmed, re-encoded, overlaid or annotated. `max_stream_s = 1200` in `demo/camera_config.json` simply
stops the session after 20 minutes.

| Property | Value (measured with OpenCV in this session) |
|---|---|
| Resolution / FPS | 640 x 360, 25 fps |
| Duration / frames | 2048.0 s (34.1 min), 51,201 frames; the demo uses the first 1200 s = 30,000 frames |
| SHA-256 | `fbbefd1b1c55a600e874ef06f84228c8feb67b7ac07999b2d8fbf41ade5291bb` |
| Container tag | `ISO Media file produced by Google Inc. Created on: 03/17/2025` |
| Continuity | scene-cut scan of the first 1200 s: 0 cuts (max histogram distance 0.177 over 2,499 samples; `demo/runs/scene_cuts_demo.json`) - one continuous shot |
| Camera | fixed, elevated (overpass / gantry height), looking along a divided carriageway; daytime, low sun |
| Scene | two carriageways: traffic *away* from the camera on the left, *toward* it on the right; 3 lanes each, dashed lane lines, central barrier; gantry visible |
| Traffic | free flow -> dense queue on the left carriageway (around 100-300 s) -> mixed; cars, vans, lorries, coaches |

## Provenance and licence - READ THIS

* **The origin and licence of `demo.mp4` are NOT verified.** The container tag (`produced by Google Inc.`) is what a file downloaded from
  YouTube carries, so it is very probably a third-party YouTube upload. No uploader, URL, licence or attribution is recorded anywhere in the
  repository (`ANVESH_PRODUCT_GAP_AUDIT.md` already flagged it "source/licence not recorded"). It is therefore treated as **copyright-unknown:
  classroom presentation use only, not redistributable, not published, not committed.**
* The road *looks* like a UK motorway (left-hand traffic, UK-style gantry, dashed lane lines). That is an inference from the picture. It is
  **not** verified, and it matters only for the speed scale (see `CALIBRATION.md`).
* If a licensed replacement is wanted, the shortlist below shows what was found. Nothing was downloaded from the web in this session.

## Candidates considered

| Candidate | Res / FPS | Duration | Camera / content | Plates | Licence | Verdict |
|---|---|---|---|---|---|---|
| **`demo.mp4`** (chosen) | 640x360 / 25 | 34.1 min continuous | fixed overpass, divided carriageway, congestion building and clearing, lane lines visible | **unreadable** (vehicle boxes are 28 px wide at the median; a plate is roughly 8 px) | **unverified** (likely YouTube upload) | Best available continuous fixed-camera source; low resolution; no wrong-way vehicle, no collision observed |
| `traffic.mp4` | 1280x720 / 25 | 5.2 min | **edited montage**: night flyover and day street scenes in 5 minutes (jump cuts, different cameras) - a scene-cut scan (`demo/runs/scene_cuts.json`) confirms it | partly visible on near vehicles | unverified | Rejected: not one continuous camera, too short, no usable lane geometry |
| Intel IoT `car-detection.mp4`, `person-bicycle-car-detection.mp4` (`datasets/recorded/`) | 768x432 / 12 | 30 s, 54 s | short stock-style clips | no | CC BY 4.0 per `datasets/recorded/*/SOURCE.md` | Too short for a 15-20 min demonstration; kept as test fixtures |
| CityFlowV2 / AI City 2022 S02 cameras (`E:\traffic dataset`) | >=960p / 10 | about 3 min per camera | fixed urban intersection cameras with lane/ROI files | marginal at 1080p | NVIDIA AIC2022 licence: non-commercial academic use after the challenge; no commercial or production use; plates and faces must be redacted in any published material | Licensed for academic use, but each clip is about 3 minutes, its supplied homography is **not metric** (`ANVESH_C006_SPEED_VALIDATION_AUDIT.md`), and the frame rate is 10 fps. Not used for the 20-minute demo |
| S7 motorway gantry videos, Gdansk University of Technology (mostwiedzy.pl open data) | 1280x960 / 25 (per search snippet) | 24-hour sessions | fixed gantry camera, oncoming vehicles | possibly readable (Polish plates, not Indian) | listed as CC BY (search snippet only) | **Not verified and not downloaded**: the portal blocks automated fetching and the files are very large. A promising manual download if a licensed replacement is wanted; verify the licence and the plate readability first |
| Wikimedia Commons / stock sites | - | - | a search found no suitable long fixed-camera traffic video | - | - | none usable |

## Which events the video can honestly show

| Event | In this footage? |
|---|---|
| Vehicle detection, tracking, classification, counting | yes |
| Congestion: free flow -> queue -> clearing; stopped vehicles in the queue | yes (left carriageway, from about 100 s) |
| Jam cause analysis | yes, as *candidates* with evidence and uncertainty; the footage shows no obstruction or crash, so a stalled vehicle or collision is not what the system will find |
| Speed estimate (approximate) | near field of the toward-camera carriageway only |
| Over-speed | possible only if a vehicle exceeds 112.7 km/h; a handful at most |
| Sudden braking, aggressive lane-weaving | possible, rare; every event is a candidate for officer review |
| Wrong-way driving | **none expected** (none observed); configured and unit-tested, not demonstrated on real footage |
| Collision / accident | **none observed**; the detector is unit-tested and experimental and must not be presented as validated |
| Red-light / illegal turn | not applicable to this scene (no signal, no junction) |
| Plate reading | **no** (plates are a few pixels wide); the dashboard says "unreadable" |

If the presentation needs wrong-way or collision footage, use a clearly separate, independently sourced clip with a stated source and licence
and say so; never splice sources into one apparent recording.

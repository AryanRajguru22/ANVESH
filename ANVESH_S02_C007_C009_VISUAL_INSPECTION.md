# ANVESH — S02 c007/c009 Visual Inspection (Preliminary Winning Pair)

**Scope:** Pure visual inspection only. No ANVESH pipeline, YOLO, or tracking was run.
Frames were extracted directly from the raw `.avi` files with OpenCV
(`cv2.VideoCapture` read + `cv2.imwrite`) at 0%, 25%, 50%, 75%, and ~97% of each
video's timeline, purely to sample representative moments. No detection, fusion,
calibration, or lane-assignment logic from the ANVESH codebase was invoked.

- `E:\traffic dataset\validation\S02\c007\vdo.avi`
- `E:\traffic dataset\validation\S02\c009\vdo.avi`

Source frames and contact sheets are saved under
`docs/demo_reference/s02_c007_c009_inspection/`.

## Video properties (via OpenCV metadata, no decoding of content)

| Camera | Resolution | FPS | Frame count | Duration |
|---|---|---|---|---|
| c007 | 1920×1080 | 10.0 | 1965 | ~196.5 s |
| c009 | 1920×1080 | 10.0 | 2110 | ~211.0 s |

Both cameras also ship a `roi.jpg` mask (simple horizon-line cutoff, not a lane
map) and a `calibration.txt` homography — neither was used here beyond confirming
both cameras publish a calibrated homography, which is a prerequisite for any
future metric-speed work but says nothing about visual suitability by itself.

## Contact sheets

**c007** (start / 25% / 50% / 75% / end):
`docs/demo_reference/s02_c007_c009_inspection/c007_contact_sheet.jpg`

**c009** (start / 25% / 50% / 75% / end):
`docs/demo_reference/s02_c007_c009_inspection/c009_contact_sheet.jpg`

---

## Observations — c007

Frames reviewed: `c007_start_f0.jpg`, `c007_50pct_f982.jpg`, `c007_75pct_f1473.jpg`,
`c007_end_f1905.jpg`.

c007 looks down on a **signalized intersection / slip-lane merge** onto a road
signed "Dodge St," including:

- A large, fan-shaped **unmarked concrete apron** filling most of the foreground
  (bottom half of the frame) — no lane paint at all in this area.
- A **curved collector/slip lane** in the middle distance with a dashed yellow
  centerline, a painted right-turn arrow, and a painted U-turn arrow at the
  merge point.
- A "ONE WAY" sign and a "LEFT TURN YIELD ON GREEN" signal head, confirming this
  is a multi-directional junction, not a single traffic-flow corridor.
- Vehicles enter/exit from at least three distinct directions and take curved or
  turning paths across the open apron rather than following fixed straight lanes.

**1. Lane markings / lane boundaries**
- Visible only on the curved slip lane (dashed yellow centerline, turn arrows).
- The dominant part of the visible pavement (the concrete apron closest to the
  camera) has **no lane markings whatsoever**.
- Not sufficiently and consistently marked to define stable polygonal lane
  regions across the whole ROI — at most one curved "lane corridor" could be
  drawn, and it would need to bend to follow the slip lane rather than being a
  simple polygon.
- Geometry is **curved/merging/turning**, not straight.

**2. Vehicle visibility**
- Typically 1–3 vehicles visible per sampled frame, occasionally more (up to ~4).
- Occlusion is low — vehicles are well spaced.
- Vehicle bottoms/wheel contact points are clearly visible (steep downward
  viewing angle, good resolution), so contact-point extraction itself is not a
  problem — the problem is that there's no marked lane structure to assign a
  contact point *to* over most of the frame.

**3. Camera viewpoint**
- Elevated, angled view over the intersection.
- Traffic direction is mixed/turning (entering, exiting, turning across the
  apron), not a single dominant through-direction.
- Does not present a clean "upstream corridor" on its own; it reads as a
  termination/merge point.

**4. Image quality**
- Resolution 1920×1080, sharp, no motion blur, no camera shake.
- Lighting: flat overcast winter daylight, consistent across the sampled frames.
- No obstructions in the ROI (trees/hillside are outside the drivable area).

**5. Showcase value**
- Weak for lane visualization (no stable multi-lane polygon structure).
- Weak for speed estimation as a standalone camera (short curved path, turning
  motion, not a straight measured distance).
- Reasonable only as a qualitative "vehicles arriving at / turning through an
  intersection" establishing shot.

---

## Observations — c009

Frames reviewed: `c009_start_f0.jpg`, `c009_50pct_f1054.jpg`,
`c009_75pct_f1581.jpg`, `c009_end_f2045.jpg`.

c009 looks down a **long, straight, divided arterial corridor** toward the same
signalized "Dodge St" intersection (same green street-name sign, same
"LEFT TURN SIGNAL" head and "SPEED LIMIT 45" sign visible in the background of
every sampled frame), with a car dealership lot and billboard along the
roadside.

**1. Lane markings / lane boundaries**
- Clearly visible and consistent across all sampled frames: dashed white lane
  dividers, a solid line delimiting a left-turn-only pocket lane (painted
  "ONLY" text + turn arrows), and a yellow line separating the near-side
  carriageway from a grass median with guardrail.
- Near-side (camera-facing) carriageway: **3 lanes** — 2 through lanes + 1
  left-turn-only pocket lane.
- Far-side carriageway (beyond the median): **2 lanes**, visible but smaller
  in frame.
- Geometry is straight and stable for the full length of the visible road —
  well suited to defining simple polygonal lane regions.
- No merging or splitting within the main through-lanes; the only lane-count
  change is the turn pocket appearing near the stop line, which is itself
  consistently marked.

**2. Vehicle visibility**
- Typically 4–10+ vehicles visible simultaneously, more at the end-of-video
  sample where traffic is queued at a red light.
- Occlusion: low to moderate. In free flow, vehicles are well separated along
  each lane. In the queued frame, adjacent-lane vehicles are close together but
  still visually distinguishable, with wheel/shadow contact points visible for
  the near lanes; the far lane in a queue is more compressed due to camera
  angle.
- Vehicle bottoms are visible for the majority of vehicles in the two nearest
  lanes; the far carriageway (beyond the median) is smaller/more distant and
  contact-point precision there would be lower.

**3. Camera viewpoint**
- Elevated, straight-down-the-corridor view; traffic in the near carriageway
  travels toward the camera (downstream toward the Dodge St intersection),
  opposing carriageway traffic moves away from camera.
- This is a genuine "upstream" vantage of a road corridor feeding the
  intersection area, in contrast to c007's "at the junction" vantage.
- Because both cameras display matching intersection signage/hardware
  ("Dodge St" sign, matching signal-head style, "LEFT TURN YIELD ON GREEN" /
  "LEFT TURN SIGNAL"), they visually appear to be looking at **the same
  Dodge St intersection from two different sides** — c009 upstream on the
  approach, c007 at/just past the junction. This is a visual impression only;
  no ground-truth topology, timestamps, or synchronization were checked.

**4. Image quality**
- Resolution 1920×1080, sharp, no motion blur, no camera shake.
- Lighting: same flat overcast winter daylight as c007, consistent across
  sampled frames.
- Minor background clutter (dealership lot, digital billboard) but it does not
  obstruct the roadway or vehicles.

**5. Showcase value**
- Strong for lane visualization: stable, straight, clearly painted multi-lane
  geometry that would render well as polygons.
- Strong for speed estimation groundwork: long straight measured corridor,
  consistent single-direction motion per carriageway, multiple vehicles
  available at varying distances/speeds.

---

## Cross-camera (pair) suitability

- Both cameras appear to reference the **same physical intersection** (matching
  "Dodge St" signage and signal hardware visible in both), which supports an
  upstream (c009) → downstream/junction (c007) narrative framing for a demo.
- However, the two cameras are not visually equivalent in showcase value:
  c009 carries essentially all of the clean multi-lane, straight-corridor value;
  c007 contributes an intersection/turning-movement view with weak lane
  structure over most of its visible pavement.
- A pairing built around "vehicle travels down the corridor in c009, then
  arrives at the junction seen in c007" is a plausible qualitative narrative,
  but it is **not** a clean two-point, same-lane-geometry speed-estimation
  demo, since c007 does not offer a comparable marked-lane path.
- No claim is made here about actual geographic adjacency, distance, or time
  synchronization between the two feeds — that would require checking dataset
  metadata/ground truth, not visual inspection.

## Limitations of this inspection

- Only 5 sparse timestamps per video were sampled; intermittent conditions
  (e.g., heavier traffic, glare, a stopped vehicle blocking a lane) elsewhere
  in the ~200 s clips were not surveyed.
- No metric speed, lane width, or physical geometry is claimed — homography
  files exist for both cameras but were not applied or validated here.
- Occlusion/lane-assignment suitability is a visual impression from single
  frames, not a measurement of ANVESH's actual detector/tracker behavior on
  this footage.
- `roi.jpg` for each camera is a coarse horizon-cutoff mask, not a lane map,
  so it added no lane-specific information beyond what the frames already show.

## Final verdict

- **c009: KEEP** — clean, straight, multi-lane corridor with stable markings;
  strong candidate for both lane visualization and speed-estimation showcase.
- **c007: REJECT** for lane-visualization / speed-estimation showcase purposes
  as currently framed — most of its visible pavement is an unmarked apron, and
  vehicle motion is curved/turning rather than along a measurable straight path.
  It may still be usable as a qualitative "arrival at the intersection"
  cutaway shot, not as a quantitative lane/speed demo camera.
- **Pair (c007 + c009) as a "preliminary winning pair" for a lane/speed ANVESH
  demo: REJECT as-is.** Recommend either (a) keeping c009 as the primary
  showcase camera and pairing it with a different downstream camera that has
  comparably clear lane markings, or (b) re-scoping c007's role to a
  qualitative/turning-movement vignette rather than a lane-speed measurement
  scene.

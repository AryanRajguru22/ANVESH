# ANVESH — S02/c006 Lane Coverage Analysis

Read-only analysis. No source file, test, lane config, dataset, or
ANVESH-V0 file was modified while producing this document. Only `Read`,
`Bash` (read-only: loading JSON, running standalone numpy/OpenCV scripts
against copies of frames, reading the dataset video/ROI file), and
image-generation into a **new** folder
(`docs/demo_reference/s02_c006_lane_coverage_analysis/`) were performed.
`configs/lanes/s02_c006.toml`, every `anvesh/` module, and the CityFlow
dataset itself are untouched.

## Data used

- **Primary evidence**: `ui/data/anvesh_s02_c006_data.json`, the exact
  30-second bounded smoke-test output already produced by Commit 9A
  (`scripts/run_s02_c006_showcase.py --sample-seconds 30`, run against
  `E:\traffic dataset\validation\S02\c006\vdo.avi`). This was **not**
  regenerated — it was already sitting on disk from the prior session and
  is internally consistent with the numbers reported in Commit 9A's
  report (300 frames, 102 tracks, 3,926 box observations, 652
  `assigned` / 3,274 `unknown_outside_regions`).
- **Reference stills**: `docs/demo_reference/s02_c006_c008_inspection/`
  (already-existing, pre-Commit-9A frames from this camera, same ones
  the current `configs/lanes/s02_c006.toml` vertices were originally
  measured against).
- **One direct read of the live dataset video** (`vdo.avi`, frame 0 only,
  via `cv2.VideoCapture` — read-only) to pull the *actual* YOLO bounding
  boxes for a real frame, as an independent sanity check against the
  reference stills (see §2/finding 2 below). No frame was written back
  into the dataset directory.
- `E:\traffic dataset\validation\S02\c006\roi.jpg` (CityFlow's own
  region-of-interest mask) — read for context only; it marks broad
  "drivable area" for the whole benchmark, not per-lane geometry, so it
  does not by itself resolve any question below.

## Method

Lane assignment in this project is computed **per detection** (see
`anvesh/perception/lane_assignment.py`'s own docstring), so every box in
`ui/data/anvesh_s02_c006_data.json` already carries a ground point
(`bbox_xyxy` bottom-center) and a `lane_status`. I regrouped every box by
`track_id`, then classified each of the 102 resulting tracks by where its
ground-point trajectory actually sits and how it moves over time (not by
its `lane_status` alone, since that is exactly the label under
evaluation). Full methodology detail and category assignment logic is in
the per-track table (§ Appendix); the categories are:

| Category | How identified | Tracks | Observations |
|---|---|---|---|
| **near-side LEFT** (L1's intended target lane) | y increases over time (approaching camera), x in the through-lane band | 12 | 865 |
| **near-side RIGHT** (physically real, pavement-marked "ONLY" turn lane) | same motion pattern, but a distinct, consistently-offset x-band | 10 | 605 |
| **far-side carriageway** (opposing direction, beyond the median) | y *decreases* over time, x follows a strong S-curve | 17 | 812 |
| **static/background** (parked cars, distant intersection traffic, likely some false positives) | bounding-box span < 30px over the whole track lifetime | 46 | 1,251 |
| **other/ambiguous/partial** | short-lived or transitional tracks that don't cleanly fall in the above | 17 | 393 |

12+10+17+46+17 = **102**, 865+605+812+1,251+393 = **3,926** — matches the
smoke-test totals exactly.

---

## 1. Number of unique tracks

**102** unique `track_id`s were produced by the real YOLOv8+ByteTrack
pipeline over the 30.0s / 300-frame sample. Of these, only **22** (12 +
10, ≈22%) are real, moving, near-side-carriageway vehicles relevant to
this camera's own lane geometry. The other 80 tracks are either the
opposing (far-side) carriageway (17), static/background/likely-spurious
detections (46 — parked cars in the visible retail parking lots, and
distant vehicles near the signalized intersection well beyond this
camera's own roadway), or short/ambiguous partial tracks (17, mostly
<15 observations, entering/leaving the frame edge or briefly occluded).
**Do not read "102 tracks" as "102 vehicles that should have gotten a
lane"** — the vast majority were never in c006's own carriageway.

## 2. Per-track detail

The full 102-row table (observations, %L1, %unknown, category, class)
is in the Appendix. The two categories that matter for lane geometry:

### near-side LEFT (L1's own target lane) — 12 tracks, 865 observations

| track_id | n_obs | class | %L1 | %unknown | trajectory note |
|---|---|---|---|---|---|
| 16 | 119 | car | 26.1 | 73.9 | full corridor, top→bottom of frame |
| 496 | 104 | car | 21.2 | 78.8 | full corridor |
| 11 | 98 | car | 55.1 | 44.9 | full corridor |
| 147 | 84 | car | 33.3 | 66.7 | full corridor |
| 429 | 77 | car | 32.5 | 67.5 | full corridor |
| 5 | 75 | car | 46.7 | 53.3 | full corridor |
| 543 | 71 | **truck** | 31.0 | 69.0 | full corridor |
| 633 | 47 | car | 53.2 | 46.8 | full corridor |
| 310 | 45 | **bus** | 46.7 | 53.3 | full corridor |
| 254 | 39 | **bus** | 30.8 | 69.2 | full corridor |
| 271 | 57 | **truck** | 52.6 | 47.4 | full corridor |
| 180 | 49 | car | 10.2 | 89.8 | **curves further left than the rest** (x=353 at y=440 vs. this group's usual 564–611); track ends mid-frame (y=576, never reaches the bottom) — consistent with a vehicle peeling off into the **left-turn pocket**, not continuing straight |

Every one of these 12 tracks visibly follows the **same** stable lane —
low lateral spread at any fixed row (see §3) — for its entire visible
lifetime, and every vehicle class present (car/truck/bus) uses it. This
is unambiguously L1's real target lane, and the current polygon only
captures 35.8% of its own observations (see §4).

### near-side RIGHT (physically real, pavement-marked "ONLY" turn lane) — 10 tracks, 605 observations

| track_id | n_obs | class | %L1 | %unknown | trajectory note |
|---|---|---|---|---|---|
| 7 | 89 | car | 51.7 | 48.3 | full corridor, never curves right |
| 134 | 84 | car | 34.5 | 65.5 | full corridor, never curves right |
| 473 | 64 | car | 37.5 | 62.5 | full corridor |
| 437 | 55 | car | 43.6 | 56.4 | full corridor |
| 298 | 39 | car | 53.8 | 46.2 | full corridor |
| 278 | 56 | car | 50.0 | 50.0 | full corridor |
| 333 | 45 | car | 60.0 | 40.0 | full corridor |
| 388 | 48 | car | 56.2 | 43.8 | full corridor |
| 285 | 56 | car | 46.4 | 53.6 | full corridor |
| 190 | 69 | car | 40.6 | 59.4 | full corridor |

Every one of these 10 tracks is **exclusively "car"** class (no
truck/bus observed there in this 30s sample), and — critically — **none
of them ever curves toward the right-turn arrow**. Each one drives dead
straight through the entire visible corridor, alongside (not behind or
merging into) the LEFT-stream traffic. This is the single most important
and least expected finding of this analysis (detailed in §3–§4).

Two individually-confirming tracks stand out at the edges of the visible
lane structure but were not folded into a group of their own because
each is a single short track, not a repeated pattern: **track 180**
(above, peeling left) and, more speculatively, **track 43** (`(362,781)
→ (192,1075)`, 8 obs, further left again at a lower y than track 180's
last point) — plausibly *also* a left-turn-pocket user, or a vehicle on
the cross-street revealed at the very bottom of frame. Neither changes
any conclusion below; both are correctly `unknown_outside_regions` today
and should stay that way.

---

## 3. How many distinct near-side lanes are actually in use?

**At least two**, not one — and this is not a subtle statistical
artifact. Bucketing every near-side ground point by row (y) shows two
tight, well-separated clusters at *every* sampled row, with a clean
~90–150px gap containing essentially zero points between them:

| y-row | LEFT cluster (x) | gap | RIGHT cluster (x) |
|---|---|---|---|
| 220 | 769–822 | ✓ empty | 879–918 |
| 260 | 720–773 | ✓ empty | 853–884 |
| 300 | 674–726 | ✓ empty | 832–863 |
| 340 | 640–695 | ✓ empty | 820–842 |
| 440 | 564–611 | ✓ empty | 783–811 |

This is confirmed **visually, against a real frame from the actual
analyzed video** (not a reference still from a different timestamp) —
see `docs/demo_reference/s02_c006_lane_coverage_analysis/
02_real_frame_two_lanes_confirmed.jpg`: frame 0 of the smoke-test sample
shows a queue of six vehicles (tracks 3/5/11/10/13/18) stacked in the
true through lane, and track 7 (this analysis's "RIGHT-stream") sitting
a full lane-width to the right, directly over the "ONLY" pavement text
with the right-turn arrow visible just ahead of it. This is two real,
physically distinct, simultaneously-occupied lanes — not a bounding-box
or vehicle-class artifact (confirmed further by the fact that trucks/
buses appear in the LEFT-stream group but never in the RIGHT-stream
group, which is exactly what you'd expect from real, independent lane
choice, not a systematic detection bias).

`docs/demo_reference/s02_c006_lane_coverage_analysis/
03_two_empirical_streams.jpg` overlays every LEFT/RIGHT ground point for
the full 30s sample against the reference still: the two streams
converge cleanly toward the vanishing point at the top and diverge
toward the camera, following the road's real curvature, exactly as two
real lane centerlines would.

**Conclusion**: the near-side carriageway shows **2 lanes with
substantial through-moving traffic** in this 30-second sample (roughly
even split: 12 vs. 10 tracks), not the 1 lane (`L1`) the current config
models. A 3rd lane (the left-turn pocket) has at least one plausible user
(track 180, possibly 43) but no confirmed *through*-traffic — consistent
with its "ONLY" designation actually holding in practice for left turns.

---

## 4. Candidate polygons for additional lanes — if defensible

Two distinct issues were found, and they call for different fixes:

### 4a. L1's own polygon is not lane-pure (this is the higher-priority, unambiguous fix)

Checking every LEFT-stream (L1's own real lane) point against `L1`'s
actual quadrilateral bounds, interpolated per-row:

| | count | % of LEFT-stream's 865 obs |
|---|---|---|
| Inside L1's polygon | 310 | 35.8% |
| Left of L1's left edge (same y-range) | 96 | 11.1% (avg. 52px too far right) |
| Right of L1's right edge | 0 | 0.0% |
| Outside L1's y-range entirely (y<220 or y>440) | 459 | 53.1% |

And, more concerning: checking every RIGHT-stream (the *other*, marked
lane) point the same way, **280 of 605 (46.3%) already fall inside L1's
current polygon** — confirmed by cross-checking against the pipeline's
own recorded `lane_status` (0 discrepancies). In other words, **L1's
current quadrilateral is simultaneously too narrow on the left (missing
64% of its own lane) and too wide on the right (already leaking into the
neighboring marked-turn lane for nearly half of that lane's
observations)** — visible directly in
`docs/demo_reference/s02_c006_lane_coverage_analysis/
04_candidate_polygons_illustrative.jpg` (current L1 in green sits
straddling both empirical streams; a tighter, left-shifted box — labeled
`L1'`, drawn only for illustration — would cleanly bound just the LEFT
stream).

This is a geometry-fit problem in the *existing, single* lane, not a
"need more lanes" problem, and it means a portion of what's currently
reported as `L1` speed/position data may actually belong to a vehicle in
the adjacent restricted lane.

### 4b. A second lane candidate exists, but with a real caveat

If the project wants to represent the RIGHT lane at all, its empirical
footprint (`docs/demo_reference/s02_c006_lane_coverage_analysis/
04_candidate_polygons_illustrative.jpg`, magenta `L2 candidate` box) is
a defensible **candidate** in the sense that it corresponds to a real,
consistently-used strip of pavement, not a hallucinated region. It is
**not** a clean-cut "yes, add it" the way item 6 below explains.

**No candidate is proposed for**: the left-turn pocket (only 1–2
single-track, non-repeated observations; no stable through-traffic
population the way LEFT/RIGHT-stream have 12 and 10 independent tracks
each), or the far-side carriageway (see §5).

---

## 5. Regions that should remain excluded

- **Left-turn-only pocket.** No repeated, stable through-traffic
  population was observed there (only isolated tracks, at least one of
  which — 180 — visibly *exits* the frame early rather than continuing
  to the bottom, consistent with actually turning). Excluding it remains
  correct.
- **Far-side (opposing) carriageway.** Confirmed directly:
  `docs/demo_reference/s02_c006_lane_coverage_analysis/
  05_farside_scurve_confirmed.jpg` plots all 17 identified far-side
  tracks (812 observations) and shows exactly the pronounced S-curve
  the existing config's rationale describes — a straight-edged
  quadrilateral genuinely cannot fit this path without either clipping
  real lane pavement or bleeding into the median at different points
  along the curve. The exclusion is correct and the evidence directly
  supports it.
- **The intersection apron / crosswalk area** (roughly y > 440–600 in
  this framing, per the existing config's own stop-bar rationale).
  22–28% of both near-side streams' observations fall below y=440; some
  of this pavement is genuinely a shared crosswalk/apron area, not lane
  pavement in the same sense, and the existing design's caution here is
  reasonable. (Some of this y-range could still be safely reclaimed —
  see §8 — but the *bottom-most* part, closest to the stop line, should
  stay excluded.)
- **Static/background detections** (46 tracks, mostly parked vehicles in
  the visible retail lots and distant traffic near the signalized
  intersection well beyond c006's own carriageway). These are not a lane
  geometry question at all — no polygon should ever cover the parking
  lots or the distant cross-street, and none of the current or proposed
  lane geometry does.

---

## 6. Can turn pockets safely be configured as lanes?

**Not as a blanket "yes," and this is the one place this analysis
recommends caution over the raw numbers.**

The RIGHT-stream lane is real, physically distinct, pavement-marked with
an exclusive right-turn arrow and "ONLY" text (confirmed both in the
original reference-frame inspection and independently re-confirmed here
against a live video frame), and empirically shows **zero observed turns
across the entire 30-second, full-visible-corridor sample** — every one
of its 10 tracks drives straight the whole way. That is a genuinely
surprising, non-obvious result, and it is real: it is not an artifact of
vehicle class, bounding-box noise, or polygon fitting.

But "no observed turn in the visible window" is not the same as "this is
a stable through lane." The pavement paint is an unambiguous legal
signal that a vehicle occupying that space is expected to turn right
somewhere ahead — quite possibly just past the edge of this camera's
field of view (this corridor runs through a dense retail strip; a
driveway or the signalized intersection itself is plausibly just beyond
frame). ANVESH's lane-assignment/speed design explicitly assumes a lane
is "a stable path a vehicle travels along" for the *whole* track, not
just the portion a fixed camera happens to see — and this analysis
cannot observe what happens after a vehicle leaves the frame.

**Conclusion: turn pockets should not be configured as general-purpose
lanes on the strength of this data alone.** The RIGHT-stream's real,
repeated usage as a de facto through lane *within this camera's field of
view* is worth documenting (as this report does) and worth a human
traffic-engineering judgment call (e.g., checking whether a driveway
exists just downstream, or reviewing a longer time window across
multiple signal cycles) — but it should not be silently promoted to "L2"
with the same semantics as L1 without that judgment call, because doing
so would assert a stability guarantee the data cannot actually back up
beyond the frame edge.

---

## 7. Is a polygonal image-space model still appropriate?

**Yes, for a short, near-linear window of one lane at a time — which is
exactly the scope the original L1 design deliberately chose — but it
does not scale cleanly to "cover more of the same lane" or "add a second
lane" without more care than a single quadrilateral provides.**

Evidence:
- Both real streams visibly **curve** as y increases (converging near
  the top, diverging toward the camera) — a real road curvature effect,
  confirmed in the far-side carriageway even more strongly (§5). A
  straight-edged quadrilateral is only a good local approximation over a
  limited y-span; §4a's finding (L1 too narrow at y=220, too wide at
  y=440) is at least partly explained by trying to linearly interpolate
  a single lane's edges across a span where the real edges do not move
  linearly.
- Near the vanishing point (small y), the two real lanes visually
  converge to within a few dozen pixels of each other — a polygon
  extended too far upward risks re-merging what should be two separate
  regions, and per-pixel localization noise (YOLO bbox jitter, small
  distant objects) grows relatively larger there.
- This camera **already has a CityFlow-supplied calibration profile**
  (`anvesh/world/calibration.py`, Commit 5) — meaning a *world-space*
  lane definition (fixed real-world lane geometry in metres, projected
  into image space per frame via the existing homography) is technically
  available and would naturally handle the curvature/perspective
  problem that a hand-measured image-space quadrilateral cannot,
  without needing multiple stacked segments. This is a **larger,
  separate piece of work** than anything scoped here — noted as the
  long-term-correct direction, not a near-term action item.

For a **modest** extension of L1's own y-range (see §8, "upward" only),
a single corrected quadrilateral is likely still adequate. For covering
the *entire* visible corridor depth (y≈150 to y≈1000+) accurately, or
for safely adding a second lane without the purity problems found in
§4a, a single quadrilateral per lane is no longer the right tool.

---

## 8. Before/after theoretical coverage estimates

All numbers below are **measured directly** from the existing 30s
sample's real ground points (never fabricated), restricted to the 22
near-side tracks (1,470 observations) that are actually L1's concern —
background/far-side/static tracks are correctly excluded from L1's
scope regardless of any change and are not part of these estimates.

| Scenario | Near-side observations assigned | % |
|---|---|---|
| **Current** (`L1` as configured today) | 590 / 1,470 | 40.1% |
| — of which correctly the LEFT lane | 310 / 865 | 35.8% |
| — of which is RIGHT-lane leakage (§4a) | 280 / 605 | 46.3% *(wrong-lane, not a genuine gain)* |
| **Scenario 1** — re-fit L1 to cleanly bound only the LEFT stream, extended in y-range to capture near-all of its own lane, RIGHT lane still excluded | ≈865 / 1,470 | ≈58.8%, and (unlike today) **lane-pure** |
| **Scenario 2** — Scenario 1 + a distinctly-labeled, caveated L2 for the RIGHT lane | ≈1,470 / 1,470 | ≈100%, contingent on the §6 judgment call |

The honest headline: today's 40.1% "coverage" already includes a
material fraction of **mislabeled** RIGHT-lane traffic being reported as
L1. Scenario 1 (fix L1's own vertices, still one lane) is a **strict
correctness improvement** even though its raw percentage (58.8%) is only
modestly higher than today's 40.1% — it trades away the false 46.3%
RIGHT-lane "credit" for a much higher, and now *genuine*, 100% (all of
LEFT-stream, ideally) capture of L1's real lane. Scenario 2 is the only
path to materially higher coverage, and it is gated on the §6 turn-lane
judgment call, not on this analysis.

---

## Recommendation

**B) Add specific additional lanes — with the following, evidence-scoped
qualifications, not a blanket "add L2 now":**

1. **Highest-confidence action (not scored against a specific letter
   above, since it touches L1 itself): re-measure and tighten `L1`'s own
   vertices.** The current polygon is demonstrably not lane-pure — too
   narrow on its own lane's left edge, too wide on its right edge (§4a).
   This should happen regardless of any other decision, and is the
   single highest-leverage, lowest-risk change available.
2. **Extend `L1`'s y-range upward** (smaller y, further from camera) to
   capture more of its own already-correct lane; keep the *downward*
   (near-camera, near-intersection) boundary conservative, unchanged
   from today's reasoning (§5, §8).
3. **Do not add `L2` for the marked "ONLY" lane without an explicit human
   traffic-engineering decision first** (§6) — the data supports that
   this pavement is real and heavily used as a de facto through lane
   *within camera view*, but cannot rule out that every one of those
   vehicles turns immediately after leaving the frame. If a decision is
   made to add it, it must carry an explicit caveat distinguishing it
   from L1's "stable general lane" semantics, not reuse them silently.
4. **Confirm and keep excluded**: the left-turn pocket, the far-side
   carriageway, and the intersection apron/crosswalk area nearest the
   stop line (§5).
5. **Longer-term** (§7): consider a calibrated world-space lane
   definition for this camera (it already has a CityFlow homography)
   rather than continuing to hand-fit image-space quadrilaterals, if
   this camera's lane geometry is going to keep growing in scope.

None of the above has been implemented — this document is analysis and
recommendation only, per the task's constraints.

---

## Appendix — full 102-track table

| track_id | n_obs | %L1 | %unknown | category | class |
|---|---|---|---|---|---|
| 12 | 300 | 0.0 | 100.0 | static/background | car |
| 14 | 218 | 0.0 | 100.0 | static/background | car |
| 19 | 161 | 0.0 | 100.0 | static/background | car |
| 16 | 119 | 26.1 | 73.9 | near-side LEFT (L1 target) | car |
| 496 | 104 | 21.2 | 78.8 | near-side LEFT (L1 target) | car |
| 506 | 104 | 0.0 | 100.0 | static/background | car |
| 11 | 98 | 55.1 | 44.9 | near-side LEFT (L1 target) | car |
| 7 | 89 | 51.7 | 48.3 | near-side RIGHT (marked ONLY) | car |
| 134 | 84 | 34.5 | 65.5 | near-side RIGHT (marked ONLY) | car |
| 147 | 84 | 33.3 | 66.7 | near-side LEFT (L1 target) | car |
| 233 | 77 | 0.0 | 100.0 | far-side carriageway | car |
| 429 | 77 | 32.5 | 67.5 | near-side LEFT (L1 target) | car |
| 249 | 76 | 0.0 | 100.0 | far-side carriageway | car |
| 341 | 76 | 0.0 | 100.0 | static/background | car |
| 5 | 75 | 46.7 | 53.3 | near-side LEFT (L1 target) | car |
| 543 | 71 | 31.0 | 69.0 | near-side LEFT (L1 target) | truck |
| 190 | 69 | 40.6 | 59.4 | near-side RIGHT (marked ONLY) | car |
| 10 | 66 | 50.0 | 50.0 | other/ambiguous/partial | truck |
| 209 | 65 | 0.0 | 100.0 | far-side carriageway | car |
| 473 | 64 | 37.5 | 62.5 | near-side RIGHT (marked ONLY) | car |
| 128 | 63 | 0.0 | 100.0 | far-side carriageway | car |
| 614 | 58 | 0.0 | 100.0 | far-side carriageway | car |
| 3 | 57 | 8.8 | 91.2 | other/ambiguous/partial | car |
| 271 | 57 | 52.6 | 47.4 | near-side LEFT (L1 target) | truck |
| 424 | 57 | 0.0 | 100.0 | far-side carriageway | car |
| 278 | 56 | 50.0 | 50.0 | near-side RIGHT (marked ONLY) | truck |
| 285 | 56 | 46.4 | 53.6 | near-side RIGHT (marked ONLY) | truck |
| 101 | 55 | 0.0 | 100.0 | far-side carriageway | car |
| 437 | 55 | 43.6 | 56.4 | near-side RIGHT (marked ONLY) | car |
| 241 | 54 | 0.0 | 100.0 | static/background | car |
| 488 | 54 | 0.0 | 100.0 | far-side carriageway | car |
| 155 | 53 | 0.0 | 100.0 | far-side carriageway | car |
| 571 | 53 | 0.0 | 100.0 | other/ambiguous/partial | car |
| 1 | 52 | 0.0 | 100.0 | far-side carriageway | truck |
| 13 | 50 | 22.0 | 78.0 | other/ambiguous/partial | car |
| 180 | 49 | 10.2 | 89.8 | near-side LEFT (L1 target) | car |
| 518 | 49 | 0.0 | 100.0 | static/background | car |
| 388 | 48 | 56.2 | 43.8 | near-side RIGHT (marked ONLY) | car |
| 527 | 48 | 0.0 | 100.0 | far-side carriageway | car |
| 633 | 47 | 53.2 | 46.8 | near-side LEFT (L1 target) | car |
| 310 | 45 | 46.7 | 53.3 | near-side LEFT (L1 target) | bus |
| 333 | 45 | 60.0 | 40.0 | near-side RIGHT (marked ONLY) | car |
| 493 | 42 | 0.0 | 100.0 | far-side carriageway | car |
| 254 | 39 | 30.8 | 69.2 | near-side LEFT (L1 target) | bus |
| 298 | 39 | 53.8 | 46.2 | near-side RIGHT (marked ONLY) | car |
| 221 | 37 | 0.0 | 100.0 | far-side carriageway | car |
| 18 | 33 | 0.0 | 100.0 | other/ambiguous/partial | car |
| 428 | 32 | 0.0 | 100.0 | far-side carriageway | car |
| 572 | 30 | 0.0 | 100.0 | static/background | car |
| 29 | 26 | 0.0 | 100.0 | static/background | car |
| 272 | 26 | 0.0 | 100.0 | static/background | car |
| 466 | 25 | 0.0 | 100.0 | far-side carriageway | car |
| 17 | 22 | 0.0 | 100.0 | static/background | car |
| 319 | 22 | 0.0 | 100.0 | other/ambiguous/partial | car |
| 6 | 21 | 0.0 | 100.0 | other/ambiguous/partial | car |
| 684 | 21 | 28.6 | 71.4 | other/ambiguous/partial | truck |
| 58 | 18 | 0.0 | 100.0 | other/ambiguous/partial | car |
| 152 | 16 | 0.0 | 100.0 | static/background | car |
| 351 | 15 | 0.0 | 100.0 | static/background | car |
| 467 | 15 | 0.0 | 100.0 | static/background | car |
| 694 | 15 | 0.0 | 100.0 | far-side carriageway | car |
| 4 | 12 | 0.0 | 100.0 | other/ambiguous/partial | car |
| 15 | 12 | 0.0 | 100.0 | static/background | car |
| 185 | 12 | 0.0 | 100.0 | static/background | car |
| 454 | 12 | 0.0 | 100.0 | static/background | car |
| 390 | 11 | 0.0 | 100.0 | static/background | car |
| 123 | 10 | 0.0 | 100.0 | static/background | car |
| 535 | 10 | 0.0 | 100.0 | static/background | car |
| 469 | 9 | 0.0 | 100.0 | other/ambiguous/partial | truck |
| 43 | 8 | 0.0 | 100.0 | other/ambiguous/partial | car |
| 519 | 8 | 0.0 | 100.0 | other/ambiguous/partial | car |
| 659 | 8 | 0.0 | 100.0 | static/background | car |
| 9 | 7 | 0.0 | 100.0 | static/background | car |
| 74 | 6 | 0.0 | 100.0 | static/background | car |
| 8 | 5 | 0.0 | 100.0 | static/background | car |
| 364 | 5 | 0.0 | 100.0 | static/background | truck |
| 477 | 5 | 0.0 | 100.0 | other/ambiguous/partial | truck |
| 676 | 5 | 60.0 | 40.0 | other/ambiguous/partial | car |
| 72 | 4 | 0.0 | 100.0 | static/background | car |
| 609 | 4 | 0.0 | 100.0 | static/background | car |
| 195 | 3 | 0.0 | 100.0 | far-side carriageway | car |
| 320 | 3 | 0.0 | 100.0 | static/background | car |
| 433 | 3 | 0.0 | 100.0 | static/background | car |
| 556 | 3 | 0.0 | 100.0 | other/ambiguous/partial | truck |
| 627 | 3 | 100.0 | 0.0 | static/background | bus |
| 2 | 2 | 0.0 | 100.0 | static/background | car |
| 25 | 2 | 0.0 | 100.0 | other/ambiguous/partial | truck |
| 159 | 2 | 0.0 | 100.0 | static/background | car |
| 175 | 2 | 0.0 | 100.0 | static/background | truck |
| 246 | 2 | 0.0 | 100.0 | static/background | car |
| 568 | 2 | 0.0 | 100.0 | static/background | car |
| 578 | 2 | 0.0 | 100.0 | static/background | bus |
| 592 | 2 | 0.0 | 100.0 | static/background | car |
| 610 | 2 | 0.0 | 100.0 | static/background | car |
| 57 | 1 | 0.0 | 100.0 | static/background | car |
| 169 | 1 | 0.0 | 100.0 | static/background | car |
| 196 | 1 | 0.0 | 100.0 | static/background | car |
| 242 | 1 | 0.0 | 100.0 | static/background | car |
| 282 | 1 | 0.0 | 100.0 | static/background | car |
| 468 | 1 | 100.0 | 0.0 | static/background | bus |
| 628 | 1 | 0.0 | 100.0 | static/background | truck |
| 726 | 1 | 0.0 | 100.0 | static/background | car |

## Appendix — images

All under `docs/demo_reference/s02_c006_lane_coverage_analysis/`:

1. `01_current_L1_overlay.jpg` — current `L1` polygon + pixel grid, for
   reference.
2. `02_real_frame_two_lanes_confirmed.jpg` — **real frame 0 of the
   actual analyzed video**, real YOLO bounding boxes, showing a
   six-vehicle queue in the true through lane alongside track 7
   independently occupying the marked "ONLY" lane.
3. `03_two_empirical_streams.jpg` — every LEFT/RIGHT-stream ground point
   for the full 30s sample, overlaid on the reference still.
4. `04_candidate_polygons_illustrative.jpg` — current `L1` (green) vs.
   illustrative, non-binding `L1'`/`L2` envelopes (yellow/magenta) drawn
   around the two empirical streams.
5. `05_farside_scurve_confirmed.jpg` — all far-side carriageway
   observations, confirming the existing exclusion rationale.

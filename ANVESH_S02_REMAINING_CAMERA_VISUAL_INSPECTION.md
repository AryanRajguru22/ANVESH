# ANVESH — S02 Remaining Camera Visual Inspection (c006, c008 vs c009)

**Scope:** Pure visual inspection only, same methodology as the earlier c007/c009
pass. No ANVESH pipeline, YOLO, or tracking was run. Frames were extracted
directly from the raw `.avi` files with OpenCV (`cv2.VideoCapture` read +
`cv2.imwrite`) at 0%, 25%, 50%, 75%, and ~97% of each video's timeline. No
detection, fusion, calibration, or lane-assignment logic from the ANVESH
codebase was invoked, and no source/dataset files were modified.

- `E:\traffic dataset\validation\S02\c006\vdo.avi`
- `E:\traffic dataset\validation\S02\c008\vdo.avi`
- `E:\traffic dataset\validation\S02\c009\vdo.avi` (re-referenced from the prior
  inspection; not re-extracted)

Frames and contact sheets are saved under:
- `docs/demo_reference/s02_c006_c008_inspection/` (new, this pass)
- `docs/demo_reference/s02_c007_c009_inspection/` (from the prior pass, c009 frames reused)

**Context carried over:** c007 was already rejected as the main showcase camera
(weak lane markings, unmarked intersection apron, turning/merge geometry). This
pass evaluates whether c006 or c008 pairs better with c009.

## Video properties (OpenCV metadata only)

| Camera | Resolution | FPS | Frames | Duration |
|---|---|---|---|---|
| c006 | 1920×1080 | 10.0 | 2110 | 211.0 s |
| c008 | 1920×1080 | 10.0 | 1924 | 192.4 s |
| c009 | 1920×1080 | 10.0 | 2110 | 211.0 s |

## Supporting GT/metadata evidence (from `ANVESH_S02_CAMERA_PAIR_ANALYSIS.md`)

This is reused as **supporting context only** — a high GT/coverage score is not
treated as proof of visual suitability, per the task instructions.

| Pair | Shared IDs | Cov A→B | Cov B→A | Dominant dir. | Direction consistency | Non-overlap (real handoff) | Median transit | Score /90 |
|---|---|---|---|---|---|---|---|---|
| c006–c009 | 116 (highest) | 93.5% | 84.7% | c006→c009 | 55% (near coin-flip) | 98% | 0.96 s | 76.3 |
| c008–c009 | 95 | 96.0% | 69.3% | c008→c009 | 67% | 97% | 8.49 s | 73.7 |
| c006–c008 | 84 | 67.7% | 84.8% | c008→c006 | 86% | 100% | 9.83 s | 76.1 |

Reprojection error (calibration confidence, lower is better): c006 = 10.25
(worst of all four cameras), c008 = 5.83 (best of all four), c009 = 7.11.

---

## Observations — c006

Frames reviewed: `c006_start_f0.jpg`, `c006_50pct_f1054.jpg`,
`c006_75pct_f1581.jpg`, `c006_end_f2045.jpg` (contact sheet:
`c006_contact_sheet.jpg`).

c006 looks down a **long, straight, divided arterial** toward a busy retail
intersection (Target, Best Buy, Dick's Sporting Goods, JCPenney visible in the
background of every sampled frame) — structurally the same *type* of road as
c009 (dashed white lane lines, a painted left-turn-only pocket lane with "ONLY"
text and turn arrows, a grass median with "DO NOT ENTER" signs on diamond
islands, matching sign/signal hardware), but pointed at a different retail
node than c009's Ford-dealership/Dodge-St intersection.

1. **Lane markings:** Clear and consistent across all sampled frames — dashed
   white lines, solid turn-pocket boundary, painted turn arrows, yellow median
   edge line.
2. **Usable lanes:** Near-side carriageway has **3 lanes** (2 through + 1
   left-turn-only pocket); far-side carriageway has **2 lanes**, visible
   across the median. Same lane count/layout as c009.
3. **Stable geometry:** Straight for the full visible length, no merges/splits
   in the through lanes; geometry is stable across all 5 sampled timestamps.
4. **Vehicle contact-point visibility:** Good in the two near lanes (elevated,
   steep-enough viewing angle); more compressed/less precise for the far
   carriageway, same limitation as c009.
5. **Traffic density:** **Higher than c009** — every sampled frame shows a
   longer queue (6–9+ vehicles bunched at a red light) than c009's
   corresponding samples. This is a busier retail-corridor approach.
6. **Occlusion:** Moderate — in the queued frames, adjacent vehicles in the
   two near lanes sit close together (e.g., a green taxi and a dark SUV nearly
   touching in `c006_start_f0.jpg`); still visually distinguishable, but tighter
   than c009's typical spacing.
7. **Road direction:** Traffic in the near carriageway moves toward the camera
   (downstream toward this retail intersection); opposing carriageway moves
   away, across the median — same directional pattern as c009.
8. **Speed-estimation suitability:** Good — long straight measured corridor,
   consistent single-direction motion per carriageway, comparable to c009.
9. **Lane-assignment suitability:** Good — same clear multi-lane structure as
   c009, straightforward to define stable polygonal lane regions.
10. **Corridor fit with c009:** Visually consistent with being a second point
    along the same style of divided arterial (same lane paint conventions,
    median design, sign hardware), but pointed at a **different** retail node
    than c009 — plausible as a genuine "further up/down the same road" pairing.

---

## Observations — c008

Frames reviewed: `c008_start_f0.jpg`, `c008_50pct_f961.jpg`,
`c008_75pct_f1442.jpg`, `c008_end_f1865.jpg` (contact sheet:
`c008_contact_sheet.jpg`).

c008 looks across a **HyVee grocery-store parking-lot exit lane** onto the same
"Dodge St" signalized intersection visible in the (already-rejected) c007
footage — same green "20 Dodge St" street sign and "LEFT TURN YIELD ON GREEN"
signal head. This is effectively a low-volume lot-exit approach, not a
multi-lane arterial.

1. **Lane markings:** Minimal — only a short white lane-divider stub and a
   painted raised median nose near the stop line; the rest of the approach is
   an unmarked, fan-shaped concrete apron, structurally similar to the apron
   that got c007 rejected.
2. **Usable lanes:** Effectively **1 exit lane** (plus a small right-side lane
   near the stop bar); not a real multi-lane corridor.
3. **Stable geometry:** The short marked section is stable, but it covers only
   a few car-lengths before the apron; not enough marked distance to define a
   meaningful multi-lane polygon set.
4. **Vehicle contact-point visibility:** Fine when a vehicle is present (steep
   angle, sharp image) — but there are very few vehicles to observe.
5. **Traffic density: Low.** Every sampled frame (start/50%/75%/end) shows at
   most 1–2 moving vehicles on the approach itself; most visible vehicles are
   parked in the HyVee lot, not driving.
6. **Occlusion:** Not a meaningful concern — too little traffic for occlusion
   to arise, but this is a sign of low value for "candidate vehicle ranking"
   (there is rarely more than one candidate on screen).
7. **Road direction:** Single direction — vehicles exit the lot and proceed
   toward the Dodge St intersection; no opposing-flow lane is visible.
8. **Speed-estimation suitability:** Poor — very short travel distance before
   the intersection, low and inconsistent traffic (vehicles are often braking/
   stopping for the light rather than in steady-state motion).
9. **Lane-assignment suitability:** Poor — same core issue as rejected c007:
   most of the visible pavement has no lane paint to assign vehicles to.
10. **Corridor fit with c009:** Weak. c008 reads as a minor lot-access
    apron next to the c007 intersection, not a comparable arterial segment to
    c009's clean multi-lane corridor — pairing it with c009 would visually mix
    a "clean multi-lane highway" scene with a "sparse parking-lot exit" scene.

---

## c006 → c009 vs c008 → c009

| Camera/pair | Lane quality | Speed quality | Vehicle visibility | Corridor quality | Occlusion | Overall showcase suitability |
|---|---|---|---|---|---|---|
| **c006** (solo) | High — clear multi-lane markings, stable straight geometry | High — long straight measured corridor | Good contact points in near lanes; more vehicles per frame than c009 | — | Moderate — denser queuing than c009, still distinguishable | Strong |
| **c008** (solo) | Low — unmarked apron, only a stub of paint | Low — short lot-exit approach, inconsistent motion | Good contact points but very few vehicles present | — | Low (too little traffic to occlude) | Weak (mirrors c007's rejection reasons) |
| **c009** (solo, reference) | High — clear multi-lane markings, stable straight geometry | High — long straight measured corridor | Good contact points in near lanes | — | Low–moderate | Strong |
| **c006 → c009** | Both cameras high, visually matched | Both cameras high, visually matched | Good in both; c006 has slightly more traffic to rank from | Same road *style*, different retail node; 116 shared GT IDs (highest of all pairs), 93.5%/84.7% coverage; dominant-direction subset is 98% genuine (non-overlapping) with a tight ~1 s median transit | Moderate in c006, low–moderate in c009 — symmetric, manageable challenge | **Strong** — full pipeline (detect→lane→speed→rank→cross-camera) supportable on both ends |
| **c008 → c009** | c009 high, c008 low — mismatched | c009 high, c008 low — mismatched | c009 good; c008 has too few candidate vehicles | GT-cleaner by direction ratio (67% vs 55%) and by raw non-overlap test, with a longer, more visually separable 8.49 s median transit | Low in c008 (too sparse), low–moderate in c009 | **Weak overall** — cross-camera GT evidence is fine, but c008 cannot itself support a convincing detection→lane→speed segment |

**On the GT/metadata caveat:** c008–c009's higher direction consistency (67%
vs. 55%) and cleaner non-overlap timing look better on paper, but that score
only certifies that *ID matching* is unambiguous — it says nothing about
whether c008 can independently demonstrate lane assignment or speed
estimation, and visually it cannot (same failure mode as the already-rejected
c007). c006–c009's weaker 55% direction consistency is a real caveat, but the
dominant-direction subset is still 98% genuinely sequential with a short,
crisp ~1 s handoff and the largest pool of shared vehicles (116) of any pair
to cherry-pick a clean example from.

## Limitations

- Only 5 sparse timestamps per video were sampled; conditions elsewhere in
  each ~190–211 s clip were not surveyed.
- No metric speed, lane width, or physical geometry is claimed — calibration
  files exist for all cameras but were not applied or validated visually here.
- Occlusion and lane-assignment suitability are visual impressions from single
  frames, not measurements of ANVESH's actual detector/tracker behavior.
- The "same road, different retail node" read on c006/c009 is a visual
  impression from shared paint conventions and sign hardware, not confirmed
  geographic adjacency or distance.

## Final recommendation

**c006 → c009.**

Rationale: this is the only remaining pair where **both** cameras
independently support the full showcase chain — clear multi-lane markings,
stable straight geometry, good vehicle contact-point visibility, and enough
traffic density to make "candidate vehicle ranking" and a "red highlight"
moment meaningful — while also carrying the strongest raw cross-camera GT
evidence (116 shared vehicles, best coverage) with a genuinely clean,
fast (~1 s), non-overlapping handoff available in the dominant travel
direction. c008–c009 is rejected despite a cleaner GT direction-consistency
ratio, because c008 fails on the same visual grounds that already disqualified
c007: an unmarked, low-traffic apron that cannot itself demonstrate lane
assignment or speed estimation. Reusing c008 would produce a demo that is
visually convincing on one end (c009) and unconvincing on the other (c008).

**Caveat carried forward:** c006 has the worst reprojection error of the four
cameras (10.25) — a calibration-confidence note to keep in mind if/when actual
metric speed values are computed later, though it does not affect this visual
assessment.

## Confirmations

- No ANVESH source code was modified.
- No dataset files were modified.
- No video frames were decoded/processed beyond representative-timestamp
  extraction and metadata queries — no YOLO/tracking/full-video processing
  was run.

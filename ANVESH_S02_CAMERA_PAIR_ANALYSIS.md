# ANVESH — S02 Camera Pair Analysis (Metadata/GT Only)

**Dataset:** AI City 2022 / CityFlowV2, `E:\traffic dataset\validation\S02\`
**Cameras analyzed:** c006, c007, c008, c009
**Method:** Ground-truth ID matching, calibration parsing, and OpenCV video-property queries only. No frame decoding, no YOLO/tracking was run. No dataset or ANVESH source files were modified.

## Governing assumption

All shared-vehicle analysis assumes CityFlowV2 ground-truth vehicle IDs are **global across cameras within a scenario** (i.e., the same numeric ID in `c006/gt/gt.txt` and `c009/gt/gt.txt` refers to the same physical vehicle). This is correct for this dataset's GT format and is the load-bearing premise behind sections A–C and G below.

## Per-camera baseline (from GT + video properties)

| Camera | Unique GT IDs | Resolution | FPS | Frames | Duration | Timestamp offset | Homography | Reprojection error |
|---|---|---|---|---|---|---|---|---|
| c006 | 124 | 1920×1080 | 10.0 | 2110 | 211.0 s | 0.000 s | valid 3×3 | 10.25 (highest / worst of the 4) |
| c007 | 90 | 1920×1080 | 10.0 | 1965 | 196.5 s | 0.061 s | valid 3×3 | 7.25 |
| c008 | 99 | 1920×1080 | 10.0 | 1924 | 192.4 s | 0.421 s | valid 3×3 | 5.83 (best) |
| c009 | 137 | 1920×1080 | 10.0 | 2110 | 211.0 s | 0.660 s | valid 3×3 | 7.11 |

All four cameras share identical resolution/FPS/codec (MP42), so video quality itself does not discriminate between pairs — only **coverage** and **corridor behavior** do. GT max-frame values match `cam_framenum/S02.txt` exactly for all four cameras (no truncation/misalignment detected). Track annotations are dense (95–98% frame-fill within each ID's span), i.e. GT tracks are not fragmented.

## Methodology notes (important for interpreting the table)

- **First-appearance separation** (`t_B_first − t_A_first`) tells you about *ID direction consistency* but cannot by itself distinguish a genuine sequential handoff from two cameras with overlapping fields of view (a vehicle visible in both at once also produces a small, consistent separation).
- To resolve this, each pair was additionally checked for **FOV-overlap contamination**: for the dominant travel direction, we computed `transit = enter_B_time − exit_A_time`. A **negative** transit time means the vehicle was still visible in camera A when it appeared in camera B (same-scene/overlapping view, not a real corridor). A pair is only a credible "upstream → downstream" corridor if most dominant-direction transit times are non-negative.
- **Direction consistency** = fraction of shared vehicles moving in the pair's majority direction. Near 50% indicates two comparably-sized opposing flows (e.g., a two-way street or intersection approach), not a single tellable corridor.
- Two of the 90 "confirmed" score components — Calibration Availability (15 pts) and Temporal Synchronization (10 pts) — are **constant across all six pairs** (every camera has a valid homography and a documented `cam_timestamp` offset). They don't discriminate the ranking; the real spread is in the remaining 65 points.
- **Lane visibility / road geometry (10 pts)** is left unscored per instructions — it requires opening the actual video frames, which this pass deliberately did not do.

## A–F: Per-pair findings

| Pair | Shared IDs | Cov A→B | Cov B→A | Dominant direction | Direction consistency | Non-overlap % (real handoff) | Median transit (exit A → enter B) | Calibration | FPS/Res |
|---|---|---|---|---|---|---|---|---|---|
| c006–c007 | 76 | 61.3% | 84.4% | c006→c007 | 75% | **12%** (mostly simultaneous) | −0.44 s | both valid | 10fps/1080p |
| c006–c008 | 84 | 67.7% | 84.8% | c008→c006 | 86% | **100%** | 9.83 s | both valid | 10fps/1080p |
| c006–c009 | 116 | 93.5% | 84.7% | c006→c009 | 55% (near coin-flip) | 98% | 0.96 s | both valid | 10fps/1080p |
| c007–c008 | 46 | 51.1% | 46.5% | c008→c007 | 87% | 82% | 6.74 s | both valid | 10fps/1080p |
| c007–c009 | 82 | 91.1% | 59.9% | c007→c009 | **94% (best)** | 87% | 0.70 s | both valid | 10fps/1080p |
| c008–c009 | 95 | 96.0% | 69.3% | c008→c009 | 67% | 97% | 8.49 s | both valid | 10fps/1080p |

**Reading the table:** `c006–c007`'s apparent tightness (median −0.44 s) is an artifact of overlapping fields of view (only 12% of dominant-direction vehicles show a genuine after-the-fact handoff) — it is **not** a clean sequential corridor despite decent direction consistency. `c006–c009` has the most raw shared vehicles and best coverage, but only 55% of them travel the same way, meaning this pair likely observes two roughly balanced opposing flows rather than one demonstrable A→B story.

## G: Showcase Suitability Score (transparent, component-scored)

Scoring is **out of 90 confirmed points**; the remaining **10 points (lane visibility/road geometry) are provisional**, pending visual inspection, and are not included in the ranking below.

Components:
- Shared vehicle evidence (30): scaled linearly against the max observed (116, c006–c009)
- Temporal/corridor relationship (20): `20 × (0.5 × direction_consistency_norm + 0.5 × non_overlap_fraction)` — rewards both a dominant direction and genuine (non-simultaneous) handoffs
- Calibration availability (15): constant — both cameras have a valid 3×3 homography in every pair
- Temporal synchronization (10): constant — both cameras have a documented `cam_timestamp` offset in every pair
- Video quality/coverage (10): scaled against average bidirectional coverage (max 89.1%, c006–c009); resolution/FPS identical across all cameras so this component is coverage-only
- Lane visibility / road geometry (10): **unscored — requires visual inspection**
- Low ambiguity/occlusion risk from GT evidence only (5): `5 × direction_consistency_norm` (a proxy: the more one-sided the ID traffic, the less an automated match could be confusing two flows — this is **not** a visual occlusion assessment)

| Rank | Pair | Shared (30) | Corridor (20) | Calib (15) | Sync (10) | Coverage (10) | Ambiguity (5) | **Score /90** | Lane visibility |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **c007–c009** | 21.2 | 17.5 | 15 | 10 | 8.5 | 4.4 | **76.6** | requires visual inspection |
| 2 | c006–c009 | 30.0 | 10.8 | 15 | 10 | 10.0 | 0.5 | 76.3 | requires visual inspection |
| 3 | c006–c008 | 21.7 | 17.2 | 15 | 10 | 8.6 | 3.6 | 76.1 | requires visual inspection |
| 4 | c008–c009 | 24.6 | 13.1 | 15 | 10 | 9.3 | 1.7 | 73.7 | requires visual inspection |
| 5 | c007–c008 | 11.9 | 15.6 | 15 | 10 | 5.5 | 3.7 | 61.7 | requires visual inspection |
| 6 | c006–c007 | 19.7 | 6.2 | 15 | 10 | 8.2 | 2.5 | 61.6 | requires visual inspection |

The top three are numerically close (76.1–76.6/90); the tie is broken narratively below, since raw score alone hides the direction-consistency/overlap distinction that matters most for a demo.

## Top 3 pairs — detail

### 1. c007–c009 (score 76.6/90)
**Why promising:** Highest direction consistency of any pair (94% of shared vehicles move c007→c009), meaning almost any shared vehicle ID gives a clean, unambiguous "upstream → downstream" story with minimal cherry-picking. 87% of dominant-direction transits are genuinely sequential (not simultaneous FOV overlap). Both cameras have among the best reprojection errors (7.25, 7.11). Shared count (82) and coverage (91.1%/59.9%) are solid.
**Limitations:** Median transit time is short (~0.7 s, mean ~2.7 s due to a slower tail) — a real but tight handoff window, so a demo clip needs to be trimmed carefully around the transition. 13% of dominant-direction vehicles still show negative transit (brief FOV overlap or annotation imprecision) — worth a visual check. Lane geometry/visibility unverified.
**Video paths:**
- `E:\traffic dataset\validation\S02\c007\vdo.avi`
- `E:\traffic dataset\validation\S02\c009\vdo.avi`

### 2. c006–c009 (score 76.3/90)
**Why promising:** Largest shared-vehicle population by far (116 IDs) and best bidirectional coverage (93.5% / 84.7%) — the strongest raw evidence that these two cameras observe the same stretch of road.
**Limitations:** Direction consistency is only 55% — essentially a coin flip between c006→c009 and c009→c006, which is the signature of two comparably-sized flows in opposite directions (e.g., both directions of a two-way street, or two approaches to an intersection) rather than one clean single-direction corridor. This makes it harder to build one unambiguous "vehicle travels A→B" showcase narrative without picking a subset of IDs. Lane geometry/visibility unverified.
**Video paths:**
- `E:\traffic dataset\validation\S02\c006\vdo.avi`
- `E:\traffic dataset\validation\S02\c009\vdo.avi`

### 3. c006–c008 (score 76.1/90)
**Why promising:** The cleanest true corridor by the overlap test — 100% of dominant-direction (c008→c006) transits are non-negative, meaning zero detected FOV-overlap contamination. Direction consistency is strong (86%).
**Limitations:** Transit time is loose relative to its median (9.83 s median, ~11.2 s IQR — roughly ±50% spread), so timing is less predictable for a scripted demo. c006 has the worst reprojection error of the four cameras (10.25), a mild calibration-confidence caveat for any pair involving it. Coverage is moderate (67.7%/84.8%). Lane geometry/visibility unverified.
**Video paths:**
- `E:\traffic dataset\validation\S02\c006\vdo.avi`
- `E:\traffic dataset\validation\S02\c008\vdo.avi`

## Data quality notes (all pairs)

- No missing/irregular timestamps detected — `cam_timestamp/S02.txt` and `cam_framenum/S02.txt` values are consistent with each video's actual frame count.
- GT track annotations are dense (95–98% frame-fill), so fragmentation within a camera is not a concern.
- Every pair has outlier shared-vehicle separations reaching 60–100+ seconds (close to full video duration), which is why median/IQR (not mean/stdev) was used throughout — the raw mean/stdev of first-appearance separation is inflated by these outliers (e.g., c007–c008 raw stdev was 29.1 s against a 6.7 s dominant-direction transit median).
- c006's reprojection error (10.25) is noticeably worse than the other three cameras (5.8–7.3) — a caveat on any pair involving c006 (c006–c007, c006–c008, c006–c009), though calibration is still nominally present/valid for scoring purposes.
- c006–c007 initially looked like the "tightest" pair by raw separation numbers; the transit/overlap check revealed this is because the two cameras mostly see vehicles **simultaneously** (12% non-overlap), not because of a precise, short corridor. This is flagged as a likely overlapping-FOV pair, not a demo-suitable corridor.

## Recommendation

**Preliminary — requires visual inspection**

Based on metadata and ground-truth evidence only, **c007–c009** is the recommended pair for the ANVESH showcase: it has the highest, least-ambiguous direction consistency (94%), a majority of genuinely sequential (non-overlapping) handoffs (87%), solid shared-vehicle coverage, and the best calibration reprojection errors among available options. **c006–c008** is a strong alternative if a longer, more clearly separated handoff (~10 s) is preferred for the demo narrative, at the cost of looser timing predictability and a worse calibration reprojection error on c006.

This recommendation must be confirmed by opening a short segment of both `vdo.avi` files to visually verify lane geometry, camera framing, occlusion levels, and that the recommended vehicle IDs are visually legible in both views before committing to a final demo pair.

## Confirmations

- No ANVESH source code was modified.
- No dataset files were modified.
- No video frames were decoded/processed beyond OpenCV's metadata properties (FPS, frame count, resolution, codec tag) — no YOLO/tracking was run.

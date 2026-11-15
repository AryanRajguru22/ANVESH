# Speed estimation audit and correction (15 Nov 2026)

Method: the REAL detector + tracker on the first 7 min (10,500 frames) of `demo.mp4`, the same `LiveSpeed` and calibration as the app. Tools:
`demo/tools/speed_audit_collect.py`, `speed_audit_collect_raw.py`, `speed_audit_analyze.py` (`--mode legacy|app`), `dash_detect.py`.
Raw track files are in `demo/runs/` (git-ignored); summaries are `docs/benchmarks/speed_audit_legacy.json` and `speed_audit_app.json`.

## 1. What was verified correct
- **Timing**: constant 25.000 fps; container timestamps equal `index/fps` (constant one-frame read offset only). Speed uses video timestamps - never wall-clock, queue time or processing FPS (so slow processing or dropped frames cannot change it).
- **Arithmetic**: least-squares slope of world position in m/s, x3.6 = km/h; reference point = bottom-centre of the box (the box centre reads ~1.2x higher because it is off the road plane).
- **Tracker smoothing is not the cause**: raw detector boxes and the tracker's Kalman boxes give the same speed profile (1.047 -> 0.958 vs 1.047 -> 0.957 from far to near rows).
- **Random error is small**: displayed band median +/-2.2 km/h (corrected estimator).

## 2. The defect and its evidence
`LiveSpeed` tested only the NEWEST sample against the speed zone but fitted a 1.6 s window. At motorway speed 1.6 s is ~48 m; the calibrated region (fit rows 227-337, hold-out dash near rows 206-210) is only ~20 m. Most of the window therefore lay in the EXTRAPOLATED far field.
- Local (0.5 s) speed of the same vehicles relative to their own speed in calibrated rows 225-300: rows 225-250 **1.03**, 250-275 **1.00**, 275-300 **0.98**, but rows 195-225 **1.10** and rows 170-195 **1.22**. A vehicle cannot change speed 22 % within ~10 m of road, so this is calibration extrapolation.
- The same vehicles measured near the camera read ~7 % slower than measured far (median ratio 0.929, 10-90 %: 0.907-0.947), identically in all three lanes.
- Far dashes beyond row ~200 are 3-4 px blobs (1 px > 1 m), so the calibration cannot be extended from this video.

## 3. Correction (anvesh/app/livespeed.py, runner, officer tile)
- Fit ONLY samples inside the calibrated zone, unclipped, with no time gap > 0.3 s and no persistent impossible jump (id swap); an isolated spike is removed by a median-absolute-deviation rule and the fit repeated.
- App settings: min 8 samples spanning >= 0.4 s, no extra smoothing (`ema=1.0`: the least-squares window already averages), precision gate `max_error_kmh=10` (status `insufficient_precision`, no number shown), newest box clipped by the frame edge -> no number.
- Officer Speed tile: median of the vehicles measured now plus the range (was the FASTEST vehicle, a biased maximum): "110 km/h - median of 4 measured now, range 80-140 - approximate, scale not surveyed".

## 4. Results on the same 7 minutes (every 0.25 s per tracked vehicle in the zone)
| | before (legacy) | after (app) |
|---|---|---|
| estimates | 1,539 | 689 (shorter, honest windows) |
| speed p10 / p50 / p90 | 94.6 / 111.4 / 131.9 | 83.3 / 94.9 / 109.3 |
| share above 112.7 km/h | 47.4 % | 5.8 % |
| median by lane (near / middle / offside) | 102.4 / 113.3 / 120.0 | 88.1 / 96.6 / 102.2 |
| median `truck` / `bus` / `car` | 101.5 / 101.7 / 112.6 | 88.1 / 87.0 / 95.8 |
| officer tile (median of per-second maxima) | 122.5 | 101.7 |
| displayed random band median / p90 | 2.5 / 3.2 | 2.2 / 3.0 |
So the readings the dashboard used to show (e.g. 114 km/h) were inflated by about 13-17 % at the median.

## 5. What can and cannot be claimed
- Established: the formula, units and timing are correct; the geometry is self-consistent inside the calibrated zone; the inflation came from fitting outside it and is removed.
- Plausibility only (not validation): HGVs in the UK are limited to ~90-97 km/h; the corrected `truck` median is 88 km/h (the legacy value 101.5 exceeded the legal HGV limit). The `truck` class also contains vans, so this is soft evidence.
- **NOT verifiable from this footage: the absolute scale.** It rests on the ASSUMED 9 m lane-marking pitch (UK 2 m + 7 m) and 3.65 m lane width; the road is unidentified (a different country's marking pitch would rescale every speed) and there is no radar, GPS or surveyed distance. The +/- band is random error only; scale uncertainty (plausibly +/-10 % or more) is NOT in it. Speeds stay labelled approximate.
- Needed to validate: one surveyed distance along the road (or the road's identity and its marking drawing), or a few passes at known speed (radar/GPS); then fit/replace the validation record with `survey_holdout`.
- Consequence of the short calibrated zone (~20 m, ~0.7 s at 30 m/s): a vehicle is measurable only for a fraction of a second, and the over-speed detector (needs 2 s inside the zone) cannot fire at motorway speed on this camera; it never did on this footage.

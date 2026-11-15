# Speed estimation audit - findings so far (15 Nov 2026, INVESTIGATION ONLY: the estimator is NOT yet corrected)

Method: real detector + tracker on the first 7 min (10,500 frames) of `demo.mp4`, same `LiveSpeed` and calibration as the app (`demo/tools/speed_audit_collect.py`,
`speed_audit_collect_raw.py`, `speed_audit_analyze.py`, `dash_detect.py`). Numbers are from those runs; raw files are in `demo/runs/` (git-ignored).

## Verified correct
- **Timing**: constant 25.000 fps; container timestamps equal `index/fps` (offset is a constant one-frame read convention). Speed uses video timestamps, never wall-clock or processing FPS.
- **Arithmetic**: least-squares slope in metres per second, x3.6 for km/h; ground point = bottom-centre of the box.
- **Tracker smoothing is not the cause**: speeds from the raw detector box and from the Kalman-filtered box give the same profile (1.047 -> 0.958 vs 1.047 -> 0.957).
- Random error is small: displayed band median +/-2.5 km/h.

## Defect found (cause of inflated speeds)
`LiveSpeed` gates only the NEWEST sample by the speed zone, but fits a 1.6 s window of samples. At motorway speed 1.6 s is about 48 m, while the calibrated region
(fit rows 227-337, hold-out dash near rows 206-210) is only about 20 m long. So most of the window lies in the EXTRAPOLATED far field.
- Local (0.5 s) speed of the same vehicles vs their own speed in calibrated rows 225-300: rows 225-250 1.03, 250-275 1.00, 275-300 0.98 (flat), but rows 195-225 **1.10** and rows 170-195 **1.22**.
  A vehicle cannot change speed 22 % within ~10 m, so this is calibration extrapolation, not driving.
- Same vehicles measured near the camera read ~7 % slower than measured far (median ratio 0.929, 10-90 %: 0.907-0.947), identical in all three lanes.
- Prototype fix (only in-zone, unclipped samples, min 8 samples / 0.4 s): median **93.0 km/h vs 106.6 current**, p90 **106.6 vs 125.8**, band median +/-2.2 km/h,
  one vehicle's own estimates spread 2.7 % (median), 337 of 399 zone-entering vehicles still get an estimate. Far dashes beyond row ~200 are 4-px blobs (1 px > 1 m), so the calibration cannot be extended from this video.

## Still NOT verifiable
- The absolute scale: it rests on the ASSUMED 9 m lane-marking pitch (UK 2 m + 7 m) and 3.65 m lane width; the road is unidentified. No radar/GPS/surveyed distance exists.
  Needed: one surveyed distance on the road (or the road's identity and its marking drawing), or a few vehicles with known speeds.
- Plausibility only (not validation): current estimator puts 45 % of estimates above 112.7 km/h and 46 % of the `truck` class above 100 km/h; the prototype fix lowers all of these.

## Also found (UI)
The officer "Speed" tile shows the FASTEST measured vehicle (max of N), biased upward: median of per-second maxima 122.8 vs median of per-second medians 113.0 km/h. It should show a typical value and the range.
The over-speed detector needs 2 s continuously inside a ~20 m zone, which no vehicle at motorway speed can satisfy (it never fires on this camera).

## TO DO (not done - usage limit reached)
1. Implement in `anvesh/app/livespeed.py`: fit only in-zone unclipped samples; stop at time gaps and implausible jumps; robust outlier rejection; precision gate (status `insufficient_precision`); time-based or no EMA; unavailable when the newest box is clipped.
2. Officer tile: show median + range, not max. Runner: pass the new parameters.
3. Tests: known distance/time, unit conversion, irregular and dropped frames, jitter, track discontinuity, perspective flatness, invalid calibration, outliers, zone-restriction regression, unreliable states.
4. Rerun the full suite; update handbook and viva guide (speed sections, validation results); commit and push.

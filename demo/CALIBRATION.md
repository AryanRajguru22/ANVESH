# Speed calibration for the demonstration camera

Every km/h figure ANVESH shows on `demo.mp4` comes from this calibration. It is an **approximate** scale taken from a published road-marking
standard. It is **not** a site survey, and speeds from it are not speed-camera readings.

## Method (reproduce with `demo/tools/build_demo_config.py`)

1. **Reference.** UK *Traffic Signs Manual* Chapter 5 (DfT, 2018, GOV.UK), Table 2-2 and Figure 2-3: the lane line on 70 mph dual carriageways
   and motorways is diagram **1005.1** = **2000 mm marking + 7000 mm gap** (9 m pitch; the manual's stud spacing of 18 m = two pitches agrees).
   The text of the figure was read from the PDF; the lane width of **3.65 m** across the pair of dashed lines is the standard motorway lane width
   and is an assumption.
2. **Measurement.** A temporal-median background of the video (vehicles removed) was thresholded; each bright dash was fitted with a principal axis to
   get sub-pixel end points. The two dashed lines of the toward-camera carriageway give consecutive dashes 0..3 (dash 0 is clipped by the bottom edge,
   so only its far end is used).
3. **Fit.** A planar homography (image -> road plane, metres) from 10 points on dashes 0-2. Fit RMSE 0.094 m.
4. **Hold-out.** The 4 end points of dash 3 were *not* used in the fit; they land within **0.43 m RMSE** of where the standard says they should be
   (27 m and 29 m along the road). This checks that the geometry is self-consistent (consecutive dashes, planar road, perspective); it does **not**
   verify the assumed 9 m pitch, because a homography can absorb any uniform scale.
5. **Recorded as** `ValidationEvidenceKind.MARKING_STANDARD_HOLDOUT` (a new kind; it is deliberately *not* `SURVEY_HOLDOUT`). Dashboard and API call it
   "APPROXIMATE metric - scaled from a published lane-marking module ... site NOT surveyed"; overspeed events carry confidence tier `low` and say so.
6. **Speed zone.** The trusted region is the near field of the toward-camera carriageway (image rows >= 225, about 0.35 m per pixel or better). The
   *displayed* speed, the over-speed detector and the braking detector are limited to it. **Caveat:** the stopped-vehicle and congestion logic read the
   same homography over the whole frame (the camera is "metric" for the engine), so stops in the far field and on the away-from-camera carriageway - where
   all 14 real stops occur - are judged on an extrapolated, approximate scale (0.75 m radius; far-field pixels are several decimetres). Each stopped-vehicle
   incident says so in its "Uncertainty and limits".

## What this does and does not establish

* Plausibility check (`demo/tools/speed_plausibility.py`, first 300 s): median per-track speed 97 km/h, p10 87, p90 107, max 116 km/h. That is the
  right order of magnitude for a UK motorway. Had the markings been the shorter diagram 1005 (1 m + 5 m), the same vehicles would read about 65 km/h.
  The plausibility check cannot tell these apart with certainty; it only makes the 9 m assumption reasonable.
* **Unverified assumptions:** (a) the road is built to diagram 1005.1, (b) lane width 3.65 m, (c) the road surface is planar across both
  carriageways (only the toward carriageway is used), (d) the video is not time-lapsed or speed-altered (its content looks real-time).
* Estimation: the least-squares slope of the calibrated ground-point track over 1.6 s; the displayed `+/-` is about twice the standard error of that slope
  plus 1 km/h. It does not include the scale uncertainty above (plausibly +/-10 % or more). Boxes clipped by the frame edge are discarded.
* Over-speed uses 112.7 km/h (70 mph) as a **demonstration threshold**; the real signed limit of the site is unknown.

## If a more defensible number is needed

Survey two known distances on the road (or find the road's identity and its published marking drawing) and replace the validation record with a
`survey_holdout` one; no other change is needed.

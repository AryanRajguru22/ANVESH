"""Evidence package boundary.

`reliability.py` (M3, blueprint Part 3 module 9) is a per-camera,
per-window heuristic quality score -- NOT cross-camera fusion.
`evidence_builder.py` (M4, module 8) turns per-camera `TrafficState` into
explicit, traceable `Evidence` records against H1-H6. Neither module
combines or ranks anything -- that is `fusion/ds_fusion.py` and
`hypotheses/ranking.py`.
"""

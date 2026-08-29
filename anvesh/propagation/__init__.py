"""Propagation package boundary.

`shockwave.py` (M5, blueprint Part 3 module 13 / Part 8): classical,
rule-based Rankine-Hugoniot shockwave-speed PREDICTION from a corridor's
per-camera `TrafficState`s. No ML/learned propagation model (blueprint
Part 2, LATER). Comparing a prediction against what was later observed is
`feedback/update_engine.py`'s job, not this package's.
"""

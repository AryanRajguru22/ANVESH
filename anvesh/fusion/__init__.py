"""Fusion package boundary.

`alignment.py` (M2, blueprint Part 3 module 6) synchronizes two cameras'
observation streams onto a shared timeline -- no evidence fusion.
`ds_fusion.py` (M4, blueprint Part 6) is the reliability-discounted
Dempster-Shafer cross-camera *evidence* fusion: combining two cameras'
candidate-cause `Evidence` into one belief-mass state. No propagation
feedback lives here -- that is `feedback/`, still unimplemented
(blueprint Part 14, milestone M5+).
"""

"""Evaluation package boundary.

`baselines.py` (M4, blueprint Part 10) is a thin, uniform dispatch that
runs Baseline A/B/C/ANVESH on the SAME evidence. M5 adds
`compare_anvesh_vs_baseline_c`, which finally makes ANVESH differ from
Baseline C by running propagation prediction + feedback on top of C's own
ranking. Still not the full evaluation/metrics/ablation framework, which
stays unimplemented until milestone M8 (`metrics.py`, `experiment_run.py`).
"""

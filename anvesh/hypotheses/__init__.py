"""Hypotheses package boundary.

`cause_model.py` (M4) holds the static H1-H7 definitions (blueprint
Part 7). `ranking.py` (M4) produces the ranked `CandidateCauseHypothesis`
from a fused belief state (`fusion/ds_fusion.py`). Neither module performs
propagation prediction or feedback -- those are `propagation/` and
`feedback/`, still unimplemented (blueprint Part 14, milestone M5+).
"""

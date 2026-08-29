"""Feedback package boundary.

`update_engine.py` (M5, blueprint Part 3 module 14 / Part 9): compares a
`propagation.shockwave` PREDICTION against what was subsequently OBSERVED
and, per the four explicit branches (CONFIRMED/PARTIAL/CONTRADICTED/
EVIDENCE_MISSING), revises the prior M4 `CandidateCauseHypothesis`
ranking -- never overwriting history, never forcing a revision when
evidence is missing.
"""

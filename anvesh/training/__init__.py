"""Detector-training support: dataset conversion, leakage-aware splitting, and a single shared
detection-metric implementation used to compare ANY two checkpoints on the same labelled images.

Nothing here is imported by the runtime perception pipeline; it exists for the offline training
milestone (see docs/ANVESH_MODEL_TRAINING_REPORT.md).
"""

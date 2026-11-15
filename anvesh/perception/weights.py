"""Detector-weights resolution and provenance.

Two rules, both deliberate:

1. Weights are NEVER downloaded implicitly. Ultralytics will fetch a recognised
   name such as `yolov8n.pt` from the internet when the file is absent; ANVESH
   must instead fail closed with a clear message.
2. An EXPLICITLY configured fine-tuned checkpoint that is missing is an error. It
   never silently falls back to the pretrained model -- that would let an
   operator believe they are running the fine-tuned detector when they are not.
   Only an UNSET fine-tuned path falls back to the pretrained checkpoint.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class WeightsNotFoundError(FileNotFoundError):
    """Configured detector weights do not exist on disk."""


def require_weights_file(path) -> Path:
    p = Path(path)
    if not p.is_file():
        raise WeightsNotFoundError(
            f"detector weights not found at {p}: vehicle detection/tracking cannot run. "
            "ANVESH never downloads model weights on its own; place the file there or fix the configured path."
        )
    return p


def resolve_detector_weights(finetuned_path, pretrained_path) -> tuple:
    """-> (path, source) where source is "finetuned" or "pretrained".

    `finetuned_path` unset/empty -> the pretrained checkpoint (must exist).
    `finetuned_path` set -> that file (must exist; no fallback).
    """
    if finetuned_path:
        return require_weights_file(finetuned_path), "finetuned"
    return require_weights_file(pretrained_path), "pretrained"


def checkpoint_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_version_label(path, source: str = None) -> str:
    """Short provenance label for evidence records, e.g. `finetuned:sha256-1a2b3c4d5e6f`.
    The hash is of the actual file, so a swapped checkpoint is visible in the record."""
    try:
        digest = checkpoint_sha256(path)[:12]
    except OSError:
        digest = "unreadable"
    return f"{source}:sha256-{digest}" if source else f"sha256-{digest}"

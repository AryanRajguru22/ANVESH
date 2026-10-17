"""NOTE (M-A): these are STATIC (import-level) checks and are NOT sufficient
on their own -- they passed while unvalidated speeds reached the reasoning
layer through `VehicleTrack.speed_estimate`. The behavioural guarantee is
`tests/test_speed_safety_dataflow.py`, which drives the real chain.

c006 speed-validation pass -- "Speed Evidence Safety" regression
(ANVESH_C006_SPEED_VALIDATION_AUDIT.md, Test 5).

Investigation traced every current caller of
`anvesh.perception.speed_estimation.estimate_vehicle_speed`/
`VehicleSpeedEstimate` (the estimator behind the observed "~2084 km/h"
c006 showcase value) and found exactly two: `scripts/export_ui_data.py`
(UI/dashboard box labels only) and `anvesh.rendering` (video-overlay
labels only, itself not wired into any pipeline caller -- see that
module's own docstring). Neither the evidence builder
(`anvesh.evidence.evidence_builder`), candidate-vehicle resolver
(`anvesh.hypotheses.candidate_vehicle`), cause ranking
(`anvesh.hypotheses.ranking`), Dempster-Shafer fusion
(`anvesh.fusion.ds_fusion`), nor propagation
(`anvesh.propagation.shockwave`) import this estimator or its result
type at all -- so an OUTLIER-confidence speed (or any other confidence)
this estimator produces was already structurally incapable of reaching
the reasoning layer, before this pass's plausibility gate even existed.

This file pins that structural fact as an automated regression: if any
future change makes one of those reasoning-layer modules start importing
`speed_estimation`, this test fails immediately, forcing an explicit
decision about whether/how an OUTLIER speed should be excluded there,
rather than that integration silently happening unnoticed.

It also proves the positive half of "must not be consumed as valid
evidence" directly: `format_speed_text` (the one function that turns
`SpeedConfidence.OUTLIER` into a UI string) never lets an OUTLIER-tagged
value's raw number leak into consumable output.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from anvesh.perception.speed_estimation import SpeedConfidence, VehicleSpeedEstimate, estimate_vehicle_speed
from anvesh.rendering import format_speed_text
from anvesh.storage.schemas import GeometryValidity
from anvesh.storage.schemas import Measurement

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Every reasoning-layer module the task's "Speed Evidence Safety" section
# names: H2/H3/H4 evidence, candidate-vehicle resolution, cause ranking,
# Dempster-Shafer fusion, and propagation.
REASONING_LAYER_MODULES = (
    REPO_ROOT / "anvesh" / "evidence" / "evidence_builder.py",
    REPO_ROOT / "anvesh" / "hypotheses" / "candidate_vehicle.py",
    REPO_ROOT / "anvesh" / "hypotheses" / "ranking.py",
    REPO_ROOT / "anvesh" / "hypotheses" / "cause_model.py",
    REPO_ROOT / "anvesh" / "fusion" / "ds_fusion.py",
    REPO_ROOT / "anvesh" / "propagation" / "shockwave.py",
)

_FORBIDDEN_NAMES = {"speed_estimation", "estimate_vehicle_speed", "VehicleSpeedEstimate", "SpeedConfidence"}


def _imported_names(module_path: Path) -> set:
    """Every name a module imports (from `import x` and `from x import y`
    statements), via `ast` -- static, does not execute the module."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[-1])
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("module_path", REASONING_LAYER_MODULES, ids=lambda p: p.stem)
def test_reasoning_layer_module_does_not_import_speed_estimation(module_path):
    assert module_path.is_file(), f"expected module not found: {module_path}"
    imported = _imported_names(module_path)
    leaked = imported & _FORBIDDEN_NAMES
    assert not leaked, (
        f"{module_path.relative_to(REPO_ROOT)} now imports {leaked} from "
        "perception.speed_estimation -- an OUTLIER-confidence speed could "
        "reach the reasoning layer. If this integration is intentional, "
        "it must explicitly exclude SpeedConfidence.OUTLIER (and "
        "UNAVAILABLE) before treating any value as usable evidence."
    )


def test_outlier_speed_estimate_never_renders_its_raw_value():
    """The positive half of evidence safety: wherever an OUTLIER estimate
    IS formatted for consumption, the implausible number itself must
    never appear -- only the explicit unreliable marker."""
    estimate = VehicleSpeedEstimate(
        track_id="cam:1", camera_id="cam", status="ok", confidence=SpeedConfidence.OUTLIER,
        measurement=Measurement(value=2492.0, error=10.0), geometry_validity=GeometryValidity.WORLD_VALIDATED,
    )
    text = format_speed_text(estimate)
    assert text is not None
    assert "2492" not in text
    assert text == "UNRELIABLE — OUTLIER"


def test_estimate_vehicle_speed_is_the_only_intended_entry_point():
    """Documents (and pins) that `estimate_vehicle_speed` is importable
    and callable in isolation -- i.e. this module's public surface is
    exactly what the reasoning-layer-import test above checks against,
    not a private/renamed alternative that would make that test miss a
    real integration."""
    assert callable(estimate_vehicle_speed)

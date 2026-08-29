import importlib

import anvesh

EXPECTED_MODULE_BOUNDARIES = [
    "anvesh.perception",
    "anvesh.world",
    "anvesh.evidence",
    "anvesh.fusion",
    "anvesh.hypotheses",
    "anvesh.propagation",
    "anvesh.feedback",
    "anvesh.api",
    "anvesh.evaluation",
]


def test_top_level_package_has_version():
    assert anvesh.__version__ == "0.1.0"


def test_all_module_boundaries_import_cleanly():
    for module_name in EXPECTED_MODULE_BOUNDARIES:
        importlib.import_module(module_name)

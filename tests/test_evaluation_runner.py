"""Smoke test for scripts/run_evaluation.py: it must run end-to-end and
print the expected disclaimer/terminology without raising."""

import runpy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_run_evaluation_script_executes_and_prints_disclaimer(capsys):
    script_path = REPO_ROOT / "scripts" / "run_evaluation.py"
    sys.argv = ["run_evaluation.py"]
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(script_path), run_name="__main__")
    assert exc_info.value.code == 0

    output = capsys.readouterr().out
    assert "CONTROLLED SYNTHETIC MECHANISM EVALUATION" in output
    assert "candidate-cause" in output.lower()
    assert "causal inference" in output.lower()  # must appear in the disclaimer explicitly denying it
    assert "real-world accuracy" in output.lower()
    assert "N/A" in output  # the evidence-missing propagation-error row must show N/A, not a fake 0.0

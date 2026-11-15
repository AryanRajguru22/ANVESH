#!/bin/bash
# Runs the full test suite as 9 sequential pytest processes (a single invocation was killed by the memory guard in earlier sessions).
# Usage: bash demo/tools/run_test_groups.sh demo/runs/test_groups.log
cd "$(dirname "$0")/../.." || exit 1
LOG="${1:-demo/runs/test_groups.log}"
PY=.venv/Scripts/python.exe
: > "$LOG"
run() {
  name="$1"; shift
  echo "=== GROUP: $name :: $*" | tee -a "$LOG"
  start=$(date +%s)
  $PY -m pytest "$@" -q -p no:cacheprovider --basetemp="D:/ANVESH/outputs/pytest_tmp/s6_$(echo "$name" | tr ' /:' '___')" 2>&1 | grep -v -E "openh264|cap_ffmpeg|h264" | tail -6 | tee -a "$LOG"
  echo "exit-code-of-pytest: ${PIPESTATUS[0]}  seconds: $(( $(date +%s) - start ))" | tee -a "$LOG"
}
run "app-1" tests/app/test_alerts.py tests/app/test_api.py tests/app/test_foundation.py tests/app/test_incidents.py
run "app-2" tests/app/test_cases.py tests/app/test_registry_handoff.py tests/app/test_registry_policy.py tests/app/test_plates.py tests/app/test_plates_robustness.py
run "app-3" tests/app/test_detectors.py tests/app/test_event_logic.py tests/app/test_session_synthetic.py tests/app/test_ingestion.py tests/app/test_live_speed.py tests/app/test_speed_accuracy.py tests/app/test_event_lifecycle.py tests/app/test_tracker_config.py
run "app-4" tests/app/test_officer_layer.py tests/app/test_mjpeg_stream.py tests/app/test_retention.py tests/app/test_security_hardening.py tests/app/test_account_security.py
run "streaming+training" tests/streaming tests/training
run "perception" tests/perception
run "world+safety+storage" tests/world tests/safety tests/storage
run "evidence..evaluation" tests/evidence tests/hypotheses tests/propagation tests/feedback tests/fusion tests/evaluation
run "top-level" tests/test_*.py
echo "=== ALL GROUPS DONE" | tee -a "$LOG"

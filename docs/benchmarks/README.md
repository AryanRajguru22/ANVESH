# Benchmarks

| File | What it is |
|---|---|
| `s9_realtime_samples.csv`, `s9_realtime_cpu_samples.csv`, `s9_realtime_summary.json` | 20-minute real-time run, Session 9 (10 s samples; `summarize_realtime.py` output incl. the final session record). Machine not isolated (see the finalization report, section 5.1) |
| `s8_fast_summary.json` | 150 s fast-mode run, Session 8 (transcribed from a terminal printout) |
| `s9_fast_full_ABORTED_contended.csv` | a full-length fast-mode run aborted at ~580 s because the machine was contended; no figure is derived from it |

Reproduce: start `scripts/run_demo.py --reset --no-simulated --autostart --port N`, then `demo/tools/monitor_run.py`, `demo/tools/sample_cpu.py` and finally `demo/tools/summarize_realtime.py` (see their docstrings).

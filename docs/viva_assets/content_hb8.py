import json
from pathlib import Path
from docbuilder import *

blocks = []
A = blocks.append
BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
S = json.loads((BENCH / "s9_realtime_summary.json").read_text(encoding="utf-8"))
FAST = json.loads((BENCH / "s8_fast_summary.json").read_text(encoding="utf-8"))
rt = S["final_session_record"]
lat = rt["latency"]
f2 = lambda x: f"{x:.1f}"
cpu = S.get("cpu") or {}

A(H1("Chapter 22 - Performance Measurements and Testing (Session 9)"))
A(CALLOUT("note", "Every figure below was read from a stored file (`docs/benchmarks/`). Real-time and fast-mode results are kept apart, and the Session 7 figures are labelled **historical**.", "How to read this chapter"))
A(H2("22.1 The 20-minute real-time benchmark"))
A(P("Setup: `scripts/run_demo.py --reset --no-simulated --autostart` on the first 1,200 s of `demo.mp4` (640 x 360, 25 fps, 30,000 source frames), `realtime_factor` 1.0 (the file is read at video speed, so throughput cannot exceed about 25 fps; "
    "the question is whether the system **keeps up**). Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU, 16 logical CPU cores, ultralytics 8.4.132, torch 2.13.0+cu130, default tracker (`bytetrack.yaml`, conf 0.25). One run, one video: no repeat, so no confidence interval."))
A(CALLOUT("limit", "**The machine was not isolated.** It is a laptop that was running the Windows *Balanced* power plan, and `Win32_Battery` reported it **discharging (on battery, 58 %)** when checked right after the runs; the user's browser, Docker and a vendor control agent were also running. "
        "A fast-mode re-run started afterwards was about 2.5x slower than the same code had been earlier the same day (17 fps vs 41 fps in the first minute), so the machine's speed was **not constant**. Treat the result as one measurement on a shared, possibly power-limited laptop; it is **not** a clean capacity figure. "
        "A clean run needs the charger connected, a high-performance power plan and nothing else running.", "Conditions"))
A(TABLE(["Measure", "Value", "Source / definition"], [
    ["Final state", rt["state"], "`/sessions/{id}` runtime record"],
    ["Wall time of the session", f"{rt['wall_elapsed_s']:.1f} s", "session clock"],
    ["Stream (video) time", f"{rt['stream_time_s']:.1f} s", "source timestamps"],
    ["Frames received (put in the frame queue)", str(rt["queues"]["frames"]["put"]), "queue counter"],
    ["Frames processed", str(rt["frames_processed"]), "runtime record"],
    ["Frames dropped", str(rt["frames_dropped"]), "runtime record; frame-queue drops: " + str(rt["queues"]["frames"]["dropped"])],
    ["Average processing FPS", f"{rt['processing_fps']:.2f}", "frames_processed / wall_elapsed_s (source rate 25.00)"],
    ["Frame processing latency p50 / p95 / p99", f"{f2(lat['frame_processing']['p50_ms'])} / {f2(lat['frame_processing']['p95_ms'])} / {f2(lat['frame_processing']['p99_ms'])} ms",
     "engine start of the frame to its end: window closing, detect+track, store update and the per-frame callback (which includes the officer overlay drawing and JPEG encoding); exact percentiles over all frames"],
    ["detect + track p50 / p95", f"{f2(lat['detect_track']['p50_ms'])} / {f2(lat['detect_track']['p95_ms'])} ms", "`model.track()` only (GPU YOLO + NMS + ByteTrack)"],
    ["End-to-end latency p50 / p95", f"{f2(lat['frame_end_to_end']['p50_ms'])} / {f2(lat['frame_end_to_end']['p95_ms'])} ms",
     "from the reader thread producing the decoded frame (`capture_wall_s`, set after decoding) to the engine finishing it, **including time waiting in the frame queue**; the browser display and the 80 ms stream poll are not included"],
    ["Frame queue high-water / capacity", f"{rt['queues']['frames']['high_water']} / {rt['queues']['frames']['capacity']}", "policy `block`; depth sampled every 10 s: max " + str(S['frame_queue_depth_max_sampled'])],
    ["Server memory (RSS incl. children)", f"{S['rss_mb']['min']:.0f}-{S['rss_mb']['max']:.0f} MB; first quarter mean {S['rss_mb']['first_quarter_mean']:.0f}, last quarter mean {S['rss_mb']['last_quarter_mean']:.0f}", "psutil, every 10 s; **peak sampled** " + f"{S['rss_mb']['max']:.0f} MB"],
    ["GPU utilisation (nvidia-smi)", f"mean {S['gpu_util_pct']['mean']:.0f} %, max sampled {S['gpu_util_pct']['max']:.0f} %", "instantaneous reading every 10 s (not an average over the interval)"],
    ["GPU memory (nvidia-smi, total used)", f"{S['gpu_mem_mb_nvidia_smi_total_used']['min']:.0f}-{S['gpu_mem_mb_nvidia_smi_total_used']['max']:.0f} MiB", "looks low for a CUDA process and was **not cross-checked** (Windows WDDM reporting)"],
    ["CPU, server process tree", (f"mean {cpu['server_cpu_pct_of_one_core']['mean']:.0f} %, max sampled {cpu['server_cpu_pct_of_one_core']['max']:.0f} % of one core ({cpu['logical_cores']} logical cores); whole system mean {cpu['system_cpu_pct_all_cores']['mean']:.0f} %" if cpu else "not measured"),
     "`demo/tools/sample_cpu.py`"],
    ["Stalls / instability", f"{S['live_samples_with_no_frame_progress']} live samples without frame progress; {S['sample_errors']} sampler errors; states seen: {', '.join(S['states_seen'])}", "10 s sampling cannot see a stall shorter than that"],
    ["Incidents raised", str(S["incidents_total"]), "real footage only; includes stopped-vehicle and congestion candidates"],
], [2.1, 2.2, 2.5], "Real-time run (measured, this session)"))
import csv as _csv
_rows = [r for r in _csv.DictReader(open(BENCH / "s9_realtime_samples.csv", newline="")) if r["state"] == "live"]
_seg = []
for _lo, _hi in ((0, 300), (300, 600), (600, 800), (800, 1000), (1000, 1200)):
    _s = [r for r in _rows if _lo <= float(r["stream_s"]) < _hi]
    if len(_s) > 1:
        _dw = float(_s[-1]["wall_s"]) - float(_s[0]["wall_s"]); _df = float(_s[-1]["frames"]) - float(_s[0]["frames"])
        _seg.append([f"{_lo}-{_hi} s", f"{_df / _dw:.1f}", f"{max(float(r['frames_q_depth']) for r in _s):.0f} / 8", f"{sum(float(r['gpu_util_pct']) for r in _s) / len(_s):.0f} %"])
A(H2("22.1b Where real time was kept and where it was lost"))
A(TABLE(["Video segment", "Frames per wall second", "Max sampled frame-queue depth", "Mean GPU util (instantaneous samples)"], _seg, [1.6, 1.8, 1.9, 1.5], "Marginal throughput by segment (from the 10 s samples)"))
A(P("Up to about 740 s of video the system ran at 25.0 fps with an empty queue (it was keeping pace). From about 800 s the rate fell to 15-21 fps and the frame queue stayed full (capacity 8, policy `block`): the reader was held back, so **no frame was dropped**, but the session lagged the video and finished "
    f"{rt['wall_elapsed_s']:.0f} s of wall time for 1,200 s of video. The cause is **not established**. Candidates, none proven: power/thermal limiting while on battery; background applications (system CPU use rose from about 28 % in the first quarter to about 43 % in the last, and the server's own CPU use from about 84 % to about 122 % of one core); "
    "heavier per-frame work in the denser later footage (more boxes for ByteTrack, the jam-cause analysis and the overlay). The Session 7 run on the same footage kept 24.89 fps throughout, so something differs between the runs, and the code has changed since (officer-screen layer and plate-reader thread)."))
A(H2("22.2 The fast-mode runs (kept separate)"))
f = FAST
A(P(f"`run_demo.py --fast --seconds 150` on the same footage with pacing removed: {f['frames_processed']} frames, {f['frames_dropped']} dropped, **{f['processing_fps']:.1f} fps** in {f['wall_elapsed_s']:.1f} s; frame processing p50 {f['frame_processing_p50_ms']:.1f} ms, p95 {f['frame_processing_p95_ms']:.1f} ms; detect+track p50 {f['detect_track_p50_ms']:.1f} ms. "
    f"Its end-to-end p50 of {f['end_to_end_p50_ms']:.0f} ms is **not** a live latency: the reader runs ahead and frames wait in a full queue. This run shows that the pipeline was faster than real time on this laptop at that moment (about 41 fps for 150 s). "
    "A later full-length fast-mode run (`s9_fast_full_ABORTED_contended.csv`) was **aborted after about 580 s** because it ran at only about 17 fps (frame-processing p50 about 54 ms) while the machine was shared and on battery; it is kept only as evidence that machine speed varied, and no capacity figure is derived from it."))
A(H2("22.3 Comparison with the historical Session 7 run"))
A(TABLE(["", "Session 7 (historical, 31 Oct 2026)", "Session 9 (this run)"], [
    ["Frames processed / dropped", "30,001 / 0", f"{rt['frames_processed']} / {rt['frames_dropped']}"],
    ["Average FPS", "24.89 (monitor)", f"{rt['processing_fps']:.2f}"],
    ["Frame latency p50 / p95", "27.4 / 38.9 ms (means of the 10 s monitor samples)", f"{f2(lat['frame_processing']['p50_ms'])} / {f2(lat['frame_processing']['p95_ms'])} ms (exact over all frames)"],
    ["RSS", "1,365-1,573 MB, +1.8 % first to last quarter", f"{S['rss_mb']['min']:.0f}-{S['rss_mb']['max']:.0f} MB, first-to-last-quarter change {100 * (S['rss_mb']['last_quarter_mean'] / S['rss_mb']['first_quarter_mean'] - 1):+.1f} %"],
    ["Incidents", "18 (14 stopped, 4 congestion)", str(S["incidents_total"])],
], [1.6, 2.6, 2.6], "Historical vs new"))
A(P("The two runs used different percentile definitions (Session 7 reports the mean of periodic samples of a running percentile; this chapter uses the final exact percentile over every frame), the machine was in a different state, and the code has since changed (officer-screen layer, plate reader thread). "
    "The comparison shows the same no-drop behaviour and similar memory, **but this run did not keep real time for the last third of the video whereas the Session 7 run did**. Session 7's figures were **not reproduced** in this environment; they remain historical."))
A(H2("22.4 Testing"))
A(P("The suite runs as nine sequential pytest processes (`bash demo/tools/run_test_groups.sh <log>`) because one process over the whole suite was once killed by the memory guard. The final counts are in `docs/ANVESH_FINALIZATION_REPORT.md`. "
    "Layers: unit tests of every module (perception, world / calibration, safety, evidence, fusion, hypotheses, propagation, storage, evaluation); streaming tests (queues, windows, identity, soak, idle polling); "
    "application tests (API, authentication and account security, cases, incidents, evidence verification, audit chain, retention, registry hand-off, alerts, plates); scripted-scene end-to-end tests through the real streaming engine "
    "(`test_session_synthetic.py`, `test_officer_layer.py`); the MJPEG stream tests over a real socket (`test_mjpeg_stream.py`). "
    "Outside pytest: `demo/tools/ui_lifecycle.py` (real Chrome, real pipeline), `demo/tools/ui_viewports.py` (layout) and `demo/tools/presenter_walkthrough.py` (the full officer workflow). "
    "Tests use scripted trackers and synthetic scenes; the real detector is exercised by the benchmark runs and the browser checks, not by unit tests."))

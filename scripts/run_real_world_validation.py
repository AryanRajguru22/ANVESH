"""M6 real-world validation runner.

Usage (from the repository root, with the venv active):

    python scripts/run_real_world_validation.py

Pipeline per sequence: REAL VIDEO -> M1 (detection+tracking) -> WINDOWING
-> M3 (traffic-state aggregation) -> M4 (evidence + single-camera
candidate-cause ranking) -> report.

*** EXPERIMENTAL HONESTY -- READ BEFORE INTERPRETING OUTPUT ***

The two configured sequences are NOT two views of the same physical
corridor. They are independent real-video sequences used for
single-camera validation, plus a mechanically labeled non-corridor
stand-in for exercising multi-camera interfaces. Any "two-sequence"
section below is explicitly labeled NON-CORRIDOR TWO-SEQUENCE
INTEGRATION TEST and must never be read as cross-camera agreement,
propagation validation, or real corridor fusion accuracy.

No calibration data exists for either sequence (no real reference
points, no surveyed segment length, no inter-camera distance). Per M6's
explicit instruction, this script does NOT fabricate any of those to
make propagation run -- it demonstrates the existing safety behavior
(a clear `ValueError` from `propagation.shockwave.
flow_density_from_traffic_state`) and reports propagation as
UNRESOLVABLE, which is the CORRECT outcome here, not a bug.

Every reported quantity is labeled OBSERVED, REFERENCE LABEL, INFERRED,
or UNRESOLVABLE (see the module docstring of `evaluation/scenarios.py`
for the analogous, but NOT interchangeable, "synthetic design label"
concept from the evaluation phase -- these are different labels for a
different, real-data context). Belief/plausibility values are
Dempster-Shafer quantities, never probabilities. ANVESH's output here is
a candidate-cause ranking, never a causal claim, never validated
real-world accuracy.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`

from anvesh.evaluation.baselines import run_baseline
from anvesh.evidence.reliability import compute_reliability
from anvesh.evidence.evidence_builder import EvidenceThresholds, build_evidence_for_camera
from anvesh.fusion.alignment import align_observations
from anvesh.perception.ingestion import VideoFileFrameSource
from anvesh.perception.pipeline import run_m1_pipeline
from anvesh.perception.traffic_state import CongestionStateTracker, CongestionThresholds
from anvesh.perception.tracking import ByteTrackYoloTracker
from anvesh.perception.windowing import build_traffic_state_timeline, generate_windows, tracks_in_window
from anvesh.propagation.shockwave import flow_density_from_traffic_state
from anvesh.storage.schemas import BaselineType, CameraRole, CandidateOutcome, MotionSpace

CONFIG_PATH = REPO_ROOT / "configs" / "experiment" / "m6_validation.toml"

_EVIDENCE_THRESHOLD_FIELDS = {f.name for f in dataclasses.fields(EvidenceThresholds)}
_CONGESTION_THRESHOLD_FIELDS = {f.name for f in dataclasses.fields(CongestionThresholds)}


def _load_config() -> dict:
    with CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def _load_real_camera_thresholds(path: Path) -> dict:
    """Parse an M6.5 real-camera threshold config into
    {sequence_id: {"max_capacity_vehicles": int|None, "evidence_thresholds": dict, "congestion_thresholds": dict}}.
    Missing keys per sequence simply mean "no override for that piece" --
    the caller falls back to the exact same defaults as the original M6
    (no-real-camera-config) run.
    """
    with path.open("rb") as f:
        data = tomllib.load(f)
    return {
        entry["sequence_id"]: {
            "max_capacity_vehicles": entry.get("max_capacity_vehicles"),
            "evidence_thresholds": entry.get("evidence_thresholds", {}),
            "congestion_thresholds": entry.get("congestion_thresholds", {}),
        }
        for entry in data.get("sequence_threshold", [])
    }


def _build_evidence_thresholds(overrides: dict):
    """None (no override -- falls back to EvidenceThresholds()'s own
    defaults, identical to every other caller in the project) unless
    `overrides` is non-empty. Raises cleanly on any field name that isn't
    a real EvidenceThresholds field, rather than silently ignoring a typo."""
    if not overrides:
        return None
    unknown = set(overrides) - _EVIDENCE_THRESHOLD_FIELDS
    if unknown:
        raise ValueError(f"Unknown EvidenceThresholds field(s) in real-camera config: {sorted(unknown)}")
    return EvidenceThresholds(**overrides)


def _build_congestion_thresholds(overrides: dict):
    """Same contract as `_build_evidence_thresholds`, for CongestionThresholds."""
    if not overrides:
        return None
    unknown = set(overrides) - _CONGESTION_THRESHOLD_FIELDS
    if unknown:
        raise ValueError(f"Unknown CongestionThresholds field(s) in real-camera config: {sorted(unknown)}")
    return CongestionThresholds(**overrides)


class SequenceResult:
    def __init__(self, sequence_id, camera_id, camera_role, video_path, config_label="default"):
        self.sequence_id = sequence_id
        self.camera_id = camera_id
        self.camera_role = camera_role
        self.video_path = video_path
        self.config_label = config_label  # "default" or "real-camera" -- for report labeling only
        self.max_capacity_vehicles_used = None
        self.fps = None
        self.frame_count = None
        self.total_duration = None
        self.m1_result = None
        self.timeline = None
        self.windows = None
        self.per_window_evidence = None
        self.per_window_rankings = None
        self.whole_video_aggregation = None
        self.whole_video_evidence = None
        self.whole_video_reliability = None


def _run_sequence(
    seq_cfg: dict,
    validation_cfg: dict,
    max_capacity_vehicles: int = None,
    evidence_thresholds=None,
    congestion_thresholds=None,
    config_label: str = "default",
) -> SequenceResult:
    """Run one sequence through M1 -> windowing/M3 -> M4.

    `max_capacity_vehicles`/`evidence_thresholds`/`congestion_thresholds`
    are all optional overrides (M6.5 Phase 2). Leaving all three at their
    default `None` reproduces the original M6 behavior EXACTLY -- the same
    `validation_cfg["max_capacity_vehicles"]` and bare `EvidenceThresholds()`/
    `CongestionThresholds()` defaults used everywhere else in the project.
    """
    video_path = REPO_ROOT / seq_cfg["video_path"]
    result = SequenceResult(
        seq_cfg["sequence_id"], seq_cfg["camera_id"], seq_cfg["camera_role"], video_path, config_label=config_label
    )
    effective_max_capacity = (
        max_capacity_vehicles if max_capacity_vehicles is not None else validation_cfg["max_capacity_vehicles"]
    )
    result.max_capacity_vehicles_used = effective_max_capacity

    result.fps = VideoFileFrameSource(video_path).fps()

    tracker = ByteTrackYoloTracker(
        model_name=str(REPO_ROOT / validation_cfg["model_name"]),
        confidence_threshold=validation_cfg["confidence_threshold"],
        tracker_config=validation_cfg["tracker_config"],
    )
    m1_result = run_m1_pipeline(
        video_path=video_path,
        camera_id=result.camera_id,
        tracker=tracker,
        model_id=validation_cfg["model_name"],
        model_version="pretrained",
        max_missed_frames=validation_cfg["max_missed_frames"],
        partial_after_missed_frames=validation_cfg["partial_after_missed_frames"],
    )
    result.m1_result = m1_result
    result.frame_count = m1_result.frame_count
    result.total_duration = m1_result.frame_count / result.fps if result.fps else 0.0

    # --- windowing + M3 ---
    window_seconds = validation_cfg["window_seconds"]
    tracker_state = CongestionStateTracker(thresholds=congestion_thresholds)
    result.windows = generate_windows(result.total_duration, window_seconds)
    result.timeline = build_traffic_state_timeline(
        result.camera_id,
        m1_result.vehicle_tracks,
        MotionSpace.IMAGE,  # no real calibration -- see module docstring
        window_seconds,
        tracker_state,
        total_duration=result.total_duration,
        max_capacity_vehicles=effective_max_capacity,
    )

    # --- M4 per window: reliability + evidence + single-camera ranking ---
    per_window_evidence = []
    per_window_rankings = []
    role = CameraRole.UPSTREAM if result.camera_role == "upstream" else CameraRole.DOWNSTREAM
    for window, aggregation in zip(result.windows, result.timeline):
        window_tracks = tracks_in_window(m1_result.vehicle_tracks, window)
        reliability = compute_reliability(result.camera_id, window.start, window.end, window_tracks, aggregation)
        evidence = build_evidence_for_camera(
            result.camera_id, (window.start, window.end), role, aggregation, window_tracks, reliability.score,
            thresholds=evidence_thresholds,
        )
        ranking = run_baseline(
            BaselineType.A,
            f"real-{result.sequence_id}-w{window.index}",
            f"cs-real-{result.sequence_id}-w{window.index}",
            (window.start, window.end),
            {result.camera_id: evidence},
            single_camera_id=result.camera_id,
        )
        per_window_evidence.append((window, aggregation, reliability, evidence))
        per_window_rankings.append((window, ranking))
    result.per_window_evidence = per_window_evidence
    result.per_window_rankings = per_window_rankings

    # --- whole-video aggregate (used only for the non-corridor integration test) ---
    whole_tracker = CongestionStateTracker(thresholds=congestion_thresholds)
    whole_agg = build_traffic_state_timeline(
        result.camera_id, m1_result.vehicle_tracks, MotionSpace.IMAGE, result.total_duration, whole_tracker,
        total_duration=result.total_duration, max_capacity_vehicles=effective_max_capacity,
    )
    result.whole_video_aggregation = whole_agg[0] if whole_agg else None
    if result.whole_video_aggregation is not None:
        result.whole_video_reliability = compute_reliability(
            result.camera_id, 0.0, result.total_duration, m1_result.vehicle_tracks, result.whole_video_aggregation
        )
        result.whole_video_evidence = build_evidence_for_camera(
            result.camera_id, (0.0, result.total_duration), role, result.whole_video_aggregation,
            m1_result.vehicle_tracks, result.whole_video_reliability.score,
            thresholds=evidence_thresholds,
        )

    return result


def _print_sequence_report(r: SequenceResult) -> None:
    print(
        f"\n{'=' * 78}\nSEQUENCE: {r.sequence_id}  (camera_id={r.camera_id}, video={r.video_path.name}, "
        f"config={r.config_label}, max_capacity_vehicles={r.max_capacity_vehicles_used})\n{'=' * 78}"
    )

    # --- OBSERVED ---
    total_detections = sum(len(obs.detections) for obs in r.m1_result.camera_observations)
    track_durations = [t.last_seen - t.first_seen for t in r.m1_result.vehicle_tracks]
    mean_track_duration = sum(track_durations) / len(track_durations) if track_durations else None
    class_counts = {}
    for t in r.m1_result.vehicle_tracks:
        class_counts[t.vehicle_class.value] = class_counts.get(t.vehicle_class.value, 0) + 1

    print("--- OBSERVED ---")
    print(f"  frame_count={r.frame_count}  fps={r.fps:.2f}  duration={r.total_duration:.1f}s")
    print(f"  total raw detections (all frames, all classes kept): {total_detections}")
    print(f"  finished/active tracks: {len(r.m1_result.vehicle_tracks)}  class counts: {class_counts or 'NONE'}")
    if mean_track_duration is None:
        print("  track duration: N/A (zero tracks)")
    else:
        print(
            f"  track duration: mean={mean_track_duration:.2f}s min={min(track_durations):.2f}s max={max(track_durations):.2f}s"
        )
    print(f"  windows: {len(r.windows)} x {r.windows[0].end - r.windows[0].start if r.windows else 0:.0f}s"
          if r.windows else "  windows: 0")

    print(f"  {'window':>14} {'veh_count':>10} {'mean_speed(px/s)':>18} {'congestion_level':>18}")
    for window, agg in zip(r.windows, r.timeline):
        ts = agg.traffic_state
        print(f"  [{window.start:>5.1f},{window.end:>5.1f})  {ts.vehicle_count:>10} {ts.mean_speed.value:>18.2f} {ts.congestion_level.value:>18}")

    print(f"  {'window':>14} {'outcome':>20} {'top-1':>8} {'belief':>8}")
    for window, ranking in r.per_window_rankings:
        top = ranking.ranked_list[0] if ranking.ranked_list else None
        top_str = top.hypothesis_id if top else "N/A"
        belief_str = f"{top.belief:.3f}" if top else "N/A"
        print(f"  [{window.start:>5.1f},{window.end:>5.1f})  {ranking.outcome.value:>20} {top_str:>8} {belief_str:>8}")

    # --- REFERENCE LABEL ---
    print("--- REFERENCE LABEL ---")
    print("  None recorded automatically -- this script does not fabricate human annotations.")
    print("  A manual spot-check would require visually reviewing frames (not performed here).")

    # --- INFERRED ---
    ranked_outcomes = [rk.outcome for _, rk in r.per_window_rankings]
    top1_changes = sum(
        1 for i in range(1, len(r.per_window_rankings))
        if r.per_window_rankings[i][1].ranked_list and r.per_window_rankings[i - 1][1].ranked_list
        and r.per_window_rankings[i][1].ranked_list[0].hypothesis_id != r.per_window_rankings[i - 1][1].ranked_list[0].hypothesis_id
    )
    print("--- INFERRED (candidate-cause ranking, per window, single-camera Baseline A) ---")
    print(f"  outcome distribution: {[(o.value, ranked_outcomes.count(o)) for o in set(ranked_outcomes)]}")
    print(f"  top-1 candidate changes between consecutive windows: {top1_changes} / {max(0, len(r.per_window_rankings) - 1)} transitions")

    # --- UNRESOLVABLE ---
    print("--- UNRESOLVABLE ---")
    print("  real (metric) speed: UNRESOLVABLE -- no calibration; speed above is PIXELS/SECOND, not km/h or m/s")
    print("  true congestion cause: UNRESOLVABLE -- no labeled real incident exists in this footage")
    print("  detection/tracking precision/recall: UNRESOLVABLE -- no frame-level human annotation performed")

    # --- SANITY CHECKS ---
    print("--- SANITY CHECKS ---")
    if total_detections == 0:
        print("  [FINDING] ZERO raw detections in the entire sequence.")
    if not r.m1_result.vehicle_tracks:
        print("  [FINDING] ZERO tracks produced.")
    elif mean_track_duration is not None and mean_track_duration < 1.0:
        print(f"  [FINDING] mean track duration is very short ({mean_track_duration:.2f}s) -- possible ID churn/fragmentation.")
    empty_windows = sum(1 for agg in r.timeline if agg.traffic_state.vehicle_count == 0)
    if empty_windows:
        print(f"  [FINDING] {empty_windows}/{len(r.timeline)} windows had zero vehicles.")
    congestion_levels_seen = {agg.traffic_state.congestion_level.value for agg in r.timeline}
    if len(congestion_levels_seen) == 1 and r.timeline:
        print(f"  [FINDING] every window classified as the SAME congestion level: {congestion_levels_seen}")
    insufficient_count = sum(1 for o in ranked_outcomes if o == CandidateOutcome.INSUFFICIENT_EVIDENCE)
    if insufficient_count:
        print(f"  [FINDING] {insufficient_count}/{len(ranked_outcomes)} windows produced no useful evidence (insufficient_evidence).")
    if r.per_window_rankings and top1_changes >= max(1, len(r.per_window_rankings) - 1) // 2 and len(r.per_window_rankings) > 1:
        print(f"  [FINDING] ranking is unstable: top-1 changed on {top1_changes}/{len(r.per_window_rankings) - 1} window transitions.")
    if not any([total_detections == 0, not r.m1_result.vehicle_tracks, empty_windows, len(congestion_levels_seen) == 1, insufficient_count]):
        print("  none of the checked failure patterns were observed.")


def _print_two_sequence_integration(results) -> None:
    print(f"\n{'=' * 78}\nNON-CORRIDOR TWO-SEQUENCE INTEGRATION TEST\n{'=' * 78}")
    print("These two sequences are NOT two views of the same physical corridor.")
    print("This section is a mechanical exercise of the M2/M4 interfaces on real")
    print("evidence from two unrelated real videos -- NOT cross-camera agreement,")
    print("NOT propagation validation, NOT real corridor fusion accuracy.\n")

    a, b = results[0], results[1]
    if a.whole_video_evidence is None or b.whole_video_evidence is None:
        print("  N/A -- at least one sequence produced no evidence at all.")
        return

    evidence_by_camera = {a.camera_id: a.whole_video_evidence, b.camera_id: b.whole_video_evidence}
    reliability_by_camera = {a.camera_id: a.whole_video_reliability, b.camera_id: b.whole_video_reliability}

    for baseline in (BaselineType.B, BaselineType.C):
        ranking = run_baseline(
            baseline, f"real-nc-{baseline.value}", "cs-real-noncorridor", (0.0, max(a.total_duration, b.total_duration)),
            evidence_by_camera, reliability_by_camera=reliability_by_camera,
        )
        top = ranking.ranked_list[0] if ranking.ranked_list else None
        print(f"  Baseline {baseline.value}: outcome={ranking.outcome.value}  top-1={top.hypothesis_id if top else 'N/A'}  belief={top.belief:.3f}" if top else f"  Baseline {baseline.value}: outcome={ranking.outcome.value}")

    print("\n  Mechanical M2 alignment check (real per-frame timestamps, unrelated clocks):")
    pairs = align_observations(a.m1_result.camera_observations, b.m1_result.camera_observations, max_offset_seconds=0.1)
    complete = sum(1 for p in pairs if p.is_complete)
    print(f"    {len(pairs)} aligned entries, {complete} 'matched' by coincidental timestamp proximity (physically meaningless here)")


def _print_propagation_section(results) -> None:
    print(f"\n{'=' * 78}\nM5 PROPAGATION\n{'=' * 78}")
    for r in results:
        if r.whole_video_aggregation is None:
            print(f"  {r.sequence_id}: N/A -- no traffic-state aggregation available.")
            continue
        try:
            # segment_length_m is a required argument but is provably never
            # used: flow_density_from_traffic_state raises on the
            # motion_space check before any division by it occurs. No real
            # or invented distance is used anywhere in this call.
            flow_density_from_traffic_state(r.whole_video_aggregation.traffic_state, segment_length_m=1.0)
            print(f"  {r.sequence_id}: UNEXPECTED -- calibration check did not raise (should not happen for MotionSpace.IMAGE).")
        except ValueError as exc:
            print(f"  {r.sequence_id}: UNRESOLVABLE -- required real calibration/corridor geometry unavailable.")
            print(f"    (safety behavior confirmed: {exc})")
    print("\n  No CorridorTopology or inter-camera distance was constructed for this check -- both")
    print("  sequences are uncalibrated (motion_space=IMAGE), so propagation cannot be attempted")
    print("  regardless of any distance value. This is the correct, safe outcome, not a bug.")


def _sequence_summary_row(r: SequenceResult) -> tuple:
    """(vehicle_count total, congestion levels seen, evidence-available window count,
    top-1 candidates seen, abstention outcome distribution) -- one row of the
    DEFAULT vs REAL-CAMERA comparison table."""
    total_vehicles = sum(agg.traffic_state.vehicle_count for agg in r.timeline)
    congestion_levels = sorted({agg.traffic_state.congestion_level.value for agg in r.timeline})
    outcomes = [rk.outcome for _, rk in r.per_window_rankings]
    ranked_count = sum(1 for o in outcomes if o == CandidateOutcome.RANKED)
    top1_ids = sorted({rk.ranked_list[0].hypothesis_id for _, rk in r.per_window_rankings if rk.ranked_list})
    outcome_dist = {o.value: outcomes.count(o) for o in set(outcomes)}
    return total_vehicles, congestion_levels, ranked_count, len(outcomes), top1_ids, outcome_dist


def _print_default_vs_real_camera_comparison(default_results, real_camera_results) -> None:
    print(f"\n{'=' * 78}\nDEFAULT vs. REAL-CAMERA CONFIGURATION COMPARISON\n{'=' * 78}")
    print("Both configurations were run against the exact same M1 perception output")
    print("(tracking/detection are OUT OF SCOPE for this comparison -- only")
    print("max_capacity_vehicles differs, per configs/experiment/m6_real_camera_thresholds.toml).\n")

    for d, rc in zip(default_results, real_camera_results):
        assert d.sequence_id == rc.sequence_id
        d_veh, d_cong, d_ranked, d_windows, d_top1, d_dist = _sequence_summary_row(d)
        rc_veh, rc_cong, rc_ranked, rc_windows, rc_top1, rc_dist = _sequence_summary_row(rc)
        print(f"  sequence: {d.sequence_id}")
        print(f"    max_capacity_vehicles:  default={d.max_capacity_vehicles_used:>3}   real-camera={rc.max_capacity_vehicles_used:>3}")
        print(f"    total vehicle_count (sum over windows, UNCHANGED by config): default={d_veh:>4}   real-camera={rc_veh:>4}")
        print(f"    congestion levels observed:  default={d_cong}   real-camera={rc_cong}")
        print(f"    windows with a RANKED outcome:  default={d_ranked}/{d_windows}   real-camera={rc_ranked}/{rc_windows}")
        print(f"    outcome distribution:  default={d_dist}   real-camera={rc_dist}")
        print(f"    distinct top-1 candidates seen:  default={d_top1 or 'N/A'}   real-camera={rc_top1 or 'N/A'}")
        if rc_ranked > d_ranked:
            print(
                "    [NOTE] the real-camera config produced MORE ranked windows than default. This is an\n"
                "    INCREASE IN EVIDENCE AVAILABILITY (occupancy proxy crossed a threshold more often\n"
                "    because the capacity denominator changed), NOT a validated accuracy improvement --\n"
                "    no reference label exists for either sequence, so neither configuration's ranking\n"
                "    has been checked against a real cause. Do not read more rankings as 'more correct.'"
            )
        elif rc_ranked == d_ranked:
            print("    [NOTE] no change in ranked-window count between configurations.")
        else:
            print("    [NOTE] the real-camera config produced FEWER ranked windows than default.")
        print()


def _run_all_sequences(config: dict, validation_cfg: dict, real_camera_thresholds: dict, config_label: str) -> list:
    results = []
    for seq_cfg in config["sequence"]:
        seq_id = seq_cfg["sequence_id"]
        overrides = real_camera_thresholds.get(seq_id, {}) if real_camera_thresholds else {}
        max_capacity = overrides.get("max_capacity_vehicles") if overrides else None
        evidence_thresholds = _build_evidence_thresholds(overrides.get("evidence_thresholds", {}) if overrides else {})
        congestion_thresholds = _build_congestion_thresholds(overrides.get("congestion_thresholds", {}) if overrides else {})
        results.append(
            _run_sequence(
                seq_cfg, validation_cfg,
                max_capacity_vehicles=max_capacity,
                evidence_thresholds=evidence_thresholds,
                congestion_thresholds=congestion_thresholds,
                config_label=config_label,
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-camera-config",
        type=Path,
        default=None,
        help=(
            "Optional path to an M6.5 real-camera threshold TOML (e.g. "
            "configs/experiment/m6_real_camera_thresholds.toml). When omitted, "
            "behavior is byte-identical to the original M6 default run."
        ),
    )
    args = parser.parse_args()

    config = _load_config()
    validation_cfg = config["validation"]

    print("ANVESH M6 -- real-world validation (independent real-video sequences)")
    print("This is NOT a production-readiness, causal-inference, or real-corridor-accuracy claim.")

    print(f"\n{'#' * 78}\n# DEFAULT CONFIGURATION (frozen M6 baseline)\n{'#' * 78}")
    default_results = _run_all_sequences(config, validation_cfg, real_camera_thresholds=None, config_label="default")
    for r in default_results:
        _print_sequence_report(r)
    if len(default_results) >= 2:
        _print_two_sequence_integration(default_results[:2])
        _print_propagation_section(default_results[:2])

    real_camera_results = None
    if args.real_camera_config is not None:
        real_camera_thresholds = _load_real_camera_thresholds(args.real_camera_config)
        print(f"\n{'#' * 78}\n# REAL-CAMERA CONFIGURATION ({args.real_camera_config})\n{'#' * 78}")
        real_camera_results = _run_all_sequences(
            config, validation_cfg, real_camera_thresholds=real_camera_thresholds, config_label="real-camera"
        )
        for r in real_camera_results:
            _print_sequence_report(r)
        if len(real_camera_results) >= 2:
            _print_two_sequence_integration(real_camera_results[:2])
            _print_propagation_section(real_camera_results[:2])

        _print_default_vs_real_camera_comparison(default_results, real_camera_results)
    else:
        print("\n(no --real-camera-config supplied -- only the default configuration was run)")

    print(f"\n{'=' * 78}")
    print("M6 SUMMARY: real footage was consumed by M1-M4 end to end. Propagation (M5)")
    print("is correctly UNRESOLVABLE without real calibration/corridor geometry, which")
    print("this run did not fabricate. Any 'two-sequence' results above are a mechanical")
    print("interface exercise only, not a validated multi-camera result.")
    if real_camera_results is not None:
        print("The DEFAULT vs. REAL-CAMERA comparison above shows evidence-availability effects")
        print("only -- it is NOT a validated accuracy comparison (no reference labels exist).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

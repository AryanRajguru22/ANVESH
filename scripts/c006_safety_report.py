"""c006 speed-safety before/after report, computed from the exported dashboard
JSONs (never hand-entered):

  BEFORE       outputs/m_a/s02_c006_data_BEFORE_fix.json      (the repo's prior export)
  SPEEDFIX     outputs/m_a/s02_c006_data_AFTER_speedfix_only_30s.json
               (WORLD only from validated geometry; no abstention rule yet)
  AFTER        outputs/m_a/s02_c006_data_AFTER_fix_30s.json   (+ evaluable-hypothesis abstention)
  MA1          outputs/m_a1/s02_c006_data_AFTER_ma1_30s.json  (M-A.1 hardening: evidence-derived
               eligibility, stop episodes across windows, run-scoped identities; written to
               outputs/m_a1/c006_safety_report.json, never over M-A's report)

Per window it reports the motion space and unit of the track speeds the
reasoning layer received, how many tracks exceeded 55.6 "m/s" (200 km/h) when
the values were labelled m/s, the number of duplicate track ids inside the
window, and the H1 ranking outcome / tier / belief.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "m_a"
OUT_A1 = REPO / "outputs" / "m_a1"

FILES = {
    "BEFORE": OUT / "s02_c006_data_BEFORE_fix.json",
    "SPEEDFIX_ONLY": OUT / "s02_c006_data_AFTER_speedfix_only_30s.json",
    "AFTER": OUT / "s02_c006_data_AFTER_fix_30s.json",
    "MA1": OUT_A1 / "s02_c006_data_AFTER_ma1_30s.json",
}


def window_row(w: dict) -> dict:
    tracks = next(iter(w["tracks_by_camera"].values()))
    ids = [t["track_id"] for t in tracks]
    # M-A.1: a track without a real measurement exports speed_estimate=null -- never a number
    speeds = [t["speed_estimate"]["value"] for t in tracks if t.get("speed_estimate") is not None]
    spaces = sorted({t["motion_space"] for t in tracks})
    units = sorted({t.get("speed_unit", "m/s (legacy export: unit implied by motion_space=world)" if t["motion_space"] == "world" else "px/s") for t in tracks})
    rl = {e["hypothesis_id"]: e for e in w["candidate_ranking"]["ranked_list"]}
    h1 = rl["H1"]
    evidence = {i["hypothesis_id"]: i for i in next(iter(w["evidence_by_camera"].values()))}
    return {
        "window": w["window"],
        "tracks": len(tracks),
        "duplicate_ids_in_window": len(ids) - len(set(ids)),
        "motion_spaces": spaces,
        "speed_units": units,
        "median_speed": statistics.median(speeds) if speeds else None,
        "tracks_over_55.6_labelled_m_per_s": sum(
            1 for t in tracks if t["motion_space"] == "world" and t.get("speed_estimate") is not None and t["speed_estimate"]["value"] > 55.6
        ),
        "tracks_with_withheld_speed": sum(1 for t in tracks if t.get("speed_estimate") is None),
        "H1_H2_H3_completeness": [evidence.get(h, {}).get("evidence_completeness") or evidence.get(h, {}).get("completeness") for h in ("H1", "H2", "H3")],
        "safety_events": [(e["event_id"], e.get("motion_space")) for e in w.get("safety_events", [])],
        "stop_episode_phases": [(e["phase"], e["speed_basis"]) for e in w.get("stop_episodes", [])],
        "outcome": w["candidate_ranking"]["outcome"],
        "H1_belief": h1["belief"],
        "H1_tier": h1["confidence_tier"],
        "H2_completeness": evidence.get("H2", {}).get("evidence_completeness") or evidence.get("H2", {}).get("completeness"),
        "candidate_vehicle": w["candidate_vehicle"]["status"],
    }


def main() -> int:
    report = {}
    for name, path in FILES.items():
        if not path.is_file():
            print(f"missing {path}", file=sys.stderr)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rv = data["real_video"]
        report[name] = {"total_tracks": rv["total_tracks"], "windows": [window_row(w) for w in rv["windows"]]}

    for name, r in report.items():
        print(f"\n=== {name}  (total_tracks={r['total_tracks']})")
        for row in r["windows"]:
            print(
                f"  win {row['window'][0]:>4.0f}-{row['window'][1]:<4.0f} tracks={row['tracks']:<3} dupIDs={row['duplicate_ids_in_window']:<2} "
                f"space={','.join(row['motion_spaces'])} median_speed={(row['median_speed'] if row['median_speed'] is not None else float('nan')):.1f} "
                f"over55.6(as m/s)={row['tracks_over_55.6_labelled_m_per_s']} | outcome={row['outcome']} "
                f"H1 belief={row['H1_belief']:.2f} tier={row['H1_tier']} | candidate_vehicle={row['candidate_vehicle']} "
                f"| H1/H2/H3 completeness={row['H1_H2_H3_completeness']} stops={row['stop_episode_phases']}"
            )
    # M-A's own report (outputs/m_a/) is NEVER overwritten: when the M-A.1 export is present
    # the combined report goes beside it, in outputs/m_a1/.
    target = OUT_A1 / "c006_safety_report.json" if "MA1" in report else OUT / "c006_safety_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

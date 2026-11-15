"""Temporal-event dataset tooling: validate a manifest, split it by source group, evaluate predictions.

    .venv\\Scripts\\python.exe -I scripts\\event_manifest.py validate --manifest m.json
    .venv\\Scripts\\python.exe -I scripts\\event_manifest.py split    --manifest m.json --train 0.7 --val 0.15 --test 0.15 --seed 0 --out m_split.json
    .venv\\Scripts\\python.exe -I scripts\\event_manifest.py evaluate --manifest m_split.json --predictions p.json --split test --out report.json

No model is involved: `evaluate` scores alarms that some detector produced. `split` never writes over an existing file. The evaluator
prints NOT_EVALUABLE_FOR_CLAIMS unless the evaluated split has enough events and verified-negative hours (see anvesh.training.temporal_events).
Exit codes: 0 ok, 1 invalid manifest / refused, 2 usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anvesh.training import temporal_events as te  # noqa: E402


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--manifest", required=True)
    s = sub.add_parser("split")
    s.add_argument("--manifest", required=True)
    s.add_argument("--train", type=float, default=0.7)
    s.add_argument("--val", type=float, default=0.15)
    s.add_argument("--test", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--manifest", required=True)
    e.add_argument("--predictions", required=True)
    e.add_argument("--split")
    e.add_argument("--early-tol-s", type=float, default=2.0)
    e.add_argument("--late-tol-s", type=float, default=2.0)
    e.add_argument("--cooldown-s", type=float, default=5.0)
    e.add_argument("--out")
    args = ap.parse_args(argv)

    manifest = _load(args.manifest)
    if args.cmd == "validate":
        r = te.validate_manifest(manifest)
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1
    if args.cmd == "split":
        if Path(args.out).exists():
            print(f"refusing to overwrite {args.out}", file=sys.stderr)
            return 1
        r = te.validate_manifest(manifest)
        if not r["ok"] or any(vid.get("split") for vid in manifest["videos"]):
            print("manifest must be valid and not already split:", r["errors"][:3] or "already has splits", file=sys.stderr)
            return 1
        try:
            assign = te.split_by_source_group(manifest, {"train": args.train, "val": args.val, "test": args.test}, args.seed)
        except ValueError as exc:
            print("split refused:", exc, file=sys.stderr)
            return 1
        for vid in manifest["videos"]:
            vid["split"] = assign[vid["video_id"]]
        Path(args.out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(te.validate_manifest(manifest)["stats"], indent=2))
        return 0
    try:
        report = te.evaluate_event_detector(manifest, _load(args.predictions), args.split, args.early_tol_s, args.late_tol_s, args.cooldown_s)
    except ValueError as exc:
        print("refused:", exc, file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Temporal event-detection research tooling (accident / incident detection from video): annotation schema, manifest validation,
source-grouped splitting, and EVENT-level evaluation.

NOTHING HERE IS A MODEL. It is the scaffolding any future accident detector must be built and judged with. No dataset has been
downloaded, no detector trained, and any numbers produced from synthetic manifests in the tests are software checks only.

SCHEMA (JSON, `schema_version` 1)
    {"schema_version": 1, "dataset": "<name>",
     "videos": [{"video_id", "source_group", "camera_id" (optional), "fps", "duration_s", "license", "fully_annotated", "split" (optional)}],
     "events": [{"event_id", "video_id", "type", "start_s", "end_s", "annotator", "certainty": "certain"|"probable"}]}

  * `source_group` is the unit that must never straddle a split: all clips cut from the same source video / recording / camera
    session share one group. Adjacent frames and overlapping windows of one recording are near-identical, so splitting by frame or
    by window leaks and inflates every held-out number.
  * `fully_annotated` is the "missing vs empty" flag: True means a person reviewed the WHOLE video and annotated every event in it,
    so time outside the events is a verified negative and an alarm there is a false alarm. False means unlabelled: alarms there are
    neither hits nor false alarms and the video contributes nothing to the false-alarm rate.
  * `license` is required and free text; the validator refuses an empty one so provenance cannot be dropped.

EVALUATION (`evaluate_event_detector`) reports, per split: event recall, detection delay (alarm time minus event start; early alarms
are reported as such, not clipped), false alarms per hour of verified-negative video, with video-level bootstrap intervals and an
explicit `claims_allowed` flag that is False until the evaluation set is large enough to say anything.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
CERTAINTY = ("certain", "probable")
MIN_EVENTS_FOR_CLAIMS = 30
MIN_NEGATIVE_HOURS_FOR_CLAIMS = 10.0


def validate_manifest(manifest: dict) -> dict:
    """{ok, errors, warnings, stats}. Never raises on malformed input - malformed input IS the finding."""
    errors, warnings = [], []
    if not isinstance(manifest, dict):
        return {"ok": False, "errors": ["manifest must be an object"], "warnings": [], "stats": {}}
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    videos, events = manifest.get("videos"), manifest.get("events")
    if not isinstance(videos, list) or not videos:
        errors.append("`videos` must be a non-empty list")
        videos = []
    if not isinstance(events, list):
        errors.append("`events` must be a list")
        events = []
    vids = {}
    for i, v in enumerate(videos):
        where = f"videos[{i}]"
        if not isinstance(v, dict):
            errors.append(f"{where}: must be an object")
            continue
        vid = v.get("video_id")
        if not isinstance(vid, str) or not vid:
            errors.append(f"{where}: video_id must be a non-empty string")
            continue
        if vid in vids:
            errors.append(f"{where}: duplicate video_id {vid!r}")
            continue
        vids[vid] = v
        for key in ("source_group", "license"):
            if not isinstance(v.get(key), str) or not v[key].strip():
                errors.append(f"{vid}: `{key}` is required and must be a non-empty string")
        fps, dur = v.get("fps"), v.get("duration_s")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 1 <= fps <= 240:
            errors.append(f"{vid}: fps must be a number in [1, 240]")
        if isinstance(dur, bool) or not isinstance(dur, (int, float)) or not math.isfinite(dur) or dur <= 0:
            errors.append(f"{vid}: duration_s must be a positive finite number")
        if not isinstance(v.get("fully_annotated"), bool):
            errors.append(f"{vid}: fully_annotated must be true or false (missing is not 'empty')")
        if v.get("split") is not None and v["split"] not in SPLITS:
            errors.append(f"{vid}: split must be one of {SPLITS}")
    seen_ev, by_video = set(), defaultdict(list)
    for i, e in enumerate(events):
        where = f"events[{i}]"
        if not isinstance(e, dict):
            errors.append(f"{where}: must be an object")
            continue
        eid = e.get("event_id")
        if not isinstance(eid, str) or not eid or eid in seen_ev:
            errors.append(f"{where}: event_id must be a unique non-empty string")
            continue
        seen_ev.add(eid)
        v = vids.get(e.get("video_id"))
        if v is None:
            errors.append(f"{eid}: unknown video_id {e.get('video_id')!r}")
            continue
        s, t = e.get("start_s"), e.get("end_s")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (s, t)):
            errors.append(f"{eid}: start_s/end_s must be finite numbers")
            continue
        if not 0 <= s < t:
            errors.append(f"{eid}: need 0 <= start_s < end_s (got {s}, {t})")
            continue
        if isinstance(v.get("duration_s"), (int, float)) and t > v["duration_s"] + 1e-6:
            errors.append(f"{eid}: end_s {t} is beyond the video duration {v['duration_s']}")
            continue
        if e.get("certainty") not in CERTAINTY:
            errors.append(f"{eid}: certainty must be one of {CERTAINTY}")
        if not isinstance(e.get("type"), str) or not e["type"]:
            errors.append(f"{eid}: type must be a non-empty string")
        if not isinstance(e.get("annotator"), str) or not e["annotator"]:
            errors.append(f"{eid}: annotator is required")
        by_video[e["video_id"]].append((s, t, eid))
    for vid, evs in by_video.items():
        evs.sort()
        for (s1, t1, id1), (s2, t2, id2) in zip(evs, evs[1:]):
            if s2 < t1:
                errors.append(f"{vid}: events {id1} and {id2} overlap (merge them or fix the times)")
    errors += _split_errors(vids)
    stats = {"videos": len(vids), "events": len(seen_ev), "fully_annotated_videos": sum(1 for v in vids.values() if v.get("fully_annotated") is True),
             "negative_hours_available": round(sum(v["duration_s"] for v in vids.values() if v.get("fully_annotated") is True and isinstance(v.get("duration_s"), (int, float))) / 3600.0, 3)}
    if videos and not stats["fully_annotated_videos"]:
        warnings.append("no video is fully annotated: false alarms per hour cannot be measured on this dataset")
    split_events = defaultdict(int)
    for e in events:
        if isinstance(e, dict) and e.get("video_id") in vids and vids[e["video_id"]].get("split"):
            split_events[vids[e["video_id"]]["split"]] += 1
    stats["events_per_split"] = dict(split_events)
    if any(v.get("split") for v in vids.values()) and any(split_events.get(s, 0) == 0 for s in SPLITS):
        warnings.append(f"a split has zero events: {dict(split_events)}")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "stats": stats}


def _split_errors(vids: dict) -> list:
    """No source group in two splits; if any video has a split, all must."""
    errs, group_splits = [], defaultdict(set)
    assigned = [v for v in vids.values() if v.get("split")]
    if assigned and len(assigned) != len(vids):
        errs.append(f"{len(vids) - len(assigned)} videos have no split while others do")
    for v in assigned:
        if isinstance(v.get("source_group"), str):
            group_splits[v["source_group"]].add(v["split"])
    for g, s in sorted(group_splits.items()):
        if len(s) > 1:
            errs.append(f"source_group {g!r} appears in splits {sorted(s)}: adjacent frames of one recording would leak across the split")
    return errs


def split_by_source_group(manifest: dict, fractions: dict, seed: int = 0) -> dict:
    """Return {video_id: split}, assigning WHOLE source groups. Groups containing events are placed first (largest first, seeded
    tie-break) into the split furthest below its event target, so a small dataset still gets events in every split; event-free groups
    then balance duration. Raises when a split cannot receive an event (too few groups) instead of silently returning an empty split."""
    if abs(sum(fractions.values()) - 1.0) > 1e-9 or set(fractions) - set(SPLITS):
        raise ValueError(f"fractions must sum to 1 over {SPLITS}")
    groups = defaultdict(lambda: {"videos": [], "events": 0, "seconds": 0.0})
    ev_count = defaultdict(int)
    for e in manifest["events"]:
        ev_count[e["video_id"]] += 1
    for v in manifest["videos"]:
        g = groups[v["source_group"]]
        g["videos"].append(v["video_id"])
        g["events"] += ev_count[v["video_id"]]
        g["seconds"] += v["duration_s"]
    rng = random.Random(seed)
    order = sorted(groups, key=lambda k: (-groups[k]["events"], rng.random()))
    with_events = [k for k in order if groups[k]["events"]]
    without = [k for k in order if not groups[k]["events"]]
    total_ev = sum(g["events"] for g in groups.values())
    total_s = sum(g["seconds"] for g in groups.values())
    ev_fill, s_fill, out = {k: 0 for k in fractions}, {k: 0.0 for k in fractions}, {}
    for k in with_events:
        pick = max(fractions, key=lambda s: (fractions[s] * total_ev - ev_fill[s]) / max(fractions[s] * total_ev, 1e-9))
        ev_fill[pick] += groups[k]["events"]
        s_fill[pick] += groups[k]["seconds"]
        out.update({vid: pick for vid in groups[k]["videos"]})
    for k in without:
        pick = max(fractions, key=lambda s: (fractions[s] * total_s - s_fill[s]) / max(fractions[s] * total_s, 1e-9))
        s_fill[pick] += groups[k]["seconds"]
        out.update({vid: pick for vid in groups[k]["videos"]})
    empty = [s for s in fractions if fractions[s] > 0 and ev_fill[s] == 0 and total_ev > 0]
    if empty:
        raise ValueError(f"split(s) {empty} received no events: {len(with_events)} event-bearing source groups cannot be spread over {len(fractions)} splits")
    return out


# -- evaluation -----------------------------------------------------------------------------------------------------------


def _merge_alarms(alarms: list, cooldown_s: float) -> list:
    """Alarms (time, score) of ONE video; those within cooldown_s of the previously KEPT alarm are repeats, not new alarms."""
    kept = []
    for t, sc in sorted(alarms):
        if not kept or t - kept[-1][0] >= cooldown_s:
            kept.append((t, sc))
    return kept


def _score_split(videos: dict, events: list, alarms: dict, early_tol_s: float, late_tol_s: float, cooldown_s: float, min_score) -> dict:
    ev_by_video = defaultdict(list)
    for e in events:
        ev_by_video[e["video_id"]].append(e)
    hits, delays, false_alarms, repeat_alarms, ignored = 0, [], 0, 0, 0
    per_video = {}
    for vid, v in videos.items():
        raw = [(t, s) for t, s in alarms.get(vid, []) if min_score is None or s is None or s >= min_score]
        kept = _merge_alarms(raw, cooldown_s)
        repeat_alarms += len(raw) - len(kept)
        evs = sorted(ev_by_video.get(vid, []), key=lambda e: e["start_s"])
        matched, fa, v_delays = set(), 0, []
        for t, _ in kept:
            hit = next((e for e in evs if e["start_s"] - early_tol_s <= t <= e["end_s"] + late_tol_s), None)
            if hit is not None:
                if hit["event_id"] not in matched:
                    matched.add(hit["event_id"])
                    v_delays.append(t - hit["start_s"])
                # a later alarm inside an already-matched event is a repeat of the same detection, not a false alarm
            elif v["fully_annotated"]:
                fa += 1
            else:
                ignored += 1
        hits += len(matched)
        delays += v_delays
        false_alarms += fa
        per_video[vid] = {"events": len(evs), "hits": len(matched), "false_alarms": fa, "hours": v["duration_s"] / 3600.0 if v["fully_annotated"] else 0.0,
                          "delays": v_delays}
    total_events = sum(p["events"] for p in per_video.values())
    hours = sum(p["hours"] for p in per_video.values())
    return {"events": total_events, "hits": hits, "recall": (hits / total_events) if total_events else None, "false_alarms": false_alarms,
            "negative_hours": hours, "false_alarms_per_hour": (false_alarms / hours) if hours > 0 else None,
            "delays_s": delays, "repeat_alarms_merged": repeat_alarms, "alarms_in_unannotated_videos_ignored": ignored, "per_video": per_video}


def _delay_summary(delays: list) -> dict:
    if not delays:
        return {"n": 0, "median_s": None, "mean_s": None, "p90_s": None, "early_alarms": 0}
    d = sorted(delays)
    p90 = d[min(len(d) - 1, math.ceil(0.9 * len(d)) - 1)]
    return {"n": len(d), "median_s": statistics.median(d), "mean_s": statistics.fmean(d), "p90_s": p90, "early_alarms": sum(1 for x in d if x < 0),
            "note": "negative = the alarm preceded the annotated start (anticipation or annotation lag), reported as such, not clipped"}


def _bootstrap(per_video: dict, n_boot: int, seed: int) -> dict:
    rng = random.Random(seed)
    ids = list(per_video)
    if len(ids) < 2:
        return {"recall_ci95": None, "false_alarms_per_hour_ci95": None, "note": "fewer than 2 videos: no interval"}
    rec, fah = [], []
    for _ in range(n_boot):
        sample = [per_video[rng.choice(ids)] for _ in ids]
        ev, hit = sum(p["events"] for p in sample), sum(p["hits"] for p in sample)
        hrs, fa = sum(p["hours"] for p in sample), sum(p["false_alarms"] for p in sample)
        if ev:
            rec.append(hit / ev)
        if hrs > 0:
            fah.append(fa / hrs)

    def ci(x):
        if len(x) < 20:
            return None
        x = sorted(x)
        return [x[int(0.025 * (len(x) - 1))], x[int(0.975 * (len(x) - 1))]]

    return {"recall_ci95": ci(rec), "false_alarms_per_hour_ci95": ci(fah), "resampled": "videos (with replacement)"}


def evaluate_event_detector(manifest: dict, predictions: list, split: str = None, early_tol_s: float = 2.0, late_tol_s: float = 2.0,
                            cooldown_s: float = 5.0, min_score: float = None, n_boot: int = 1000, seed: int = 0) -> dict:
    """predictions: [{"video_id", "time_s", "score" (optional)}]. Evaluates the videos of `split` (all videos if None)."""
    check = validate_manifest(manifest)
    if not check["ok"]:
        raise ValueError("invalid manifest: " + "; ".join(check["errors"][:5]))
    videos = {v["video_id"]: v for v in manifest["videos"] if split is None or v.get("split") == split}
    if split is not None and not videos:
        raise ValueError(f"no videos in split {split!r}")
    unknown = {p["video_id"] for p in predictions} - {v["video_id"] for v in manifest["videos"]}
    if unknown:
        raise ValueError(f"predictions reference unknown video ids: {sorted(unknown)[:5]}")
    alarms = defaultdict(list)
    for p in predictions:
        if p["video_id"] in videos:
            t = p["time_s"]
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
                raise ValueError(f"invalid prediction time {t!r}")
            alarms[p["video_id"]].append((float(t), p.get("score")))
    events = [e for e in manifest["events"] if e["video_id"] in videos]
    r = _score_split(videos, events, alarms, early_tol_s, late_tol_s, cooldown_s, min_score)
    out = {"split": split, "settings": {"early_tol_s": early_tol_s, "late_tol_s": late_tol_s, "cooldown_s": cooldown_s, "min_score": min_score},
           "videos": len(videos), "events": r["events"], "hits": r["hits"], "recall": r["recall"], "false_alarms": r["false_alarms"],
           "negative_hours": round(r["negative_hours"], 4), "false_alarms_per_hour": r["false_alarms_per_hour"],
           "detection_delay": _delay_summary(r["delays_s"]), "repeat_alarms_merged": r["repeat_alarms_merged"],
           "alarms_in_unannotated_videos_ignored": r["alarms_in_unannotated_videos_ignored"],
           "uncertainty": _bootstrap(r["per_video"], n_boot, seed)}
    reasons = []
    if r["events"] < MIN_EVENTS_FOR_CLAIMS:
        reasons.append(f"only {r['events']} events (< {MIN_EVENTS_FOR_CLAIMS})")
    if r["negative_hours"] < MIN_NEGATIVE_HOURS_FOR_CLAIMS:
        reasons.append(f"only {r['negative_hours']:.2f} h of verified-negative video (< {MIN_NEGATIVE_HOURS_FOR_CLAIMS} h)")
    if split not in (None, "test"):
        reasons.append(f"split is {split!r}: only a held-out `test` split supports a claim")
    out["claims_allowed"] = not reasons
    out["verdict"] = "evaluable" if not reasons else "NOT_EVALUABLE_FOR_CLAIMS: " + "; ".join(reasons)
    out["scope_note"] = "Numbers describe THIS annotated sample under these tolerances. They are not a general accuracy claim."
    return out


def operating_curve(manifest: dict, predictions: list, split: str = None, thresholds: int = 25, **kw) -> list:
    """Recall / false alarms per hour / median delay at score thresholds taken from the observed scores (quantiles)."""
    scores = sorted({p["score"] for p in predictions if isinstance(p.get("score"), (int, float))})
    if not scores:
        raise ValueError("predictions carry no scores")
    picks = sorted({scores[min(len(scores) - 1, int(i * (len(scores) - 1) / max(1, thresholds - 1)))] for i in range(thresholds)})
    rows = []
    for th in picks:
        r = evaluate_event_detector(manifest, predictions, split=split, min_score=th, n_boot=0, **kw)
        rows.append({"threshold": th, "recall": r["recall"], "false_alarms_per_hour": r["false_alarms_per_hour"],
                     "median_delay_s": r["detection_delay"]["median_s"]})
    return rows

"""Ground-truth scoring primitives: detection precision/recall and
multi-object-tracking identity metrics (IDF1, ID switches, fragmentation),
plus the duplicate-identity statistic. Pure functions, no model/video access,
so every metric here is unit-tested against toy cases with known answers
(`tests/evaluation/test_mot_metrics.py`).

THREE TIERS, NEVER MIXED (M-A requirement): the same functions score three
different things, and the caller must say which one it is scoring --
  1. detector output (class-agnostic boxes per frame, no identity);
  2. tracker output (raw ByteTrack IDs per box);
  3. TrackManager/store output (session-scoped identity keys per box).
Nothing in this module knows which tier it is given.

Box convention: `(x, y, w, h)` in pixels (MOTChallenge). A "frame table" is a
dict `frame -> list of (obj_id, x, y, w, h)`; for detections `obj_id` is
ignored (use any value).

MATCHING: IoU >= `iou_threshold` (default 0.5), one-to-one, optimal per frame
(Hungarian via `lap`, already an Ultralytics dependency).

METRIC DEFINITIONS (so the numbers are reproducible):
  - detection: per-frame optimal one-to-one matching; precision = TP/(TP+FP),
    recall = TP/(TP+FN).
  - IDF1/IDP/IDR: Ristani et al. 2016 -- ONE global bipartite assignment of
    whole GT trajectories to whole predicted trajectories; a (gt, pred) pair's
    overlap is the number of frames where both exist with IoU >= threshold.
    IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN).
  - ID switches: CLEAR-MOT style frame-by-frame matching that keeps the
    previous GT->pred correspondence while it still overlaps; a switch is
    counted each time a GT trajectory is matched to a different predicted ID
    than the last one it was matched to.
  - fragmentations: number of times a GT trajectory goes tracked -> untracked
    -> tracked again (counted on re-acquisition).
  - duplicate identity: see `duplicate_identity_stats`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # `lap` ships with Ultralytics' tracker stack
    import lap as _lap
except ImportError:  # pragma: no cover - fall back to scipy if lap is absent
    _lap = None

_BIG = 1e9


def _linear_assignment(cost: np.ndarray) -> list:
    """Optimal (row, col) pairs minimising total cost for a rectangular
    matrix. Gating (forbidden pairs) is expressed as cost >= _BIG/2 and such
    pairs are dropped from the result."""
    if cost.size == 0:
        return []
    if _lap is not None:
        _, x, _y = _lap.lapjv(cost, extend_cost=True)
        pairs = [(i, int(j)) for i, j in enumerate(x) if j >= 0]
    else:  # pragma: no cover
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        pairs = list(zip(rows.tolist(), cols.tolist()))
    return [(i, j) for i, j in pairs if cost[i, j] < _BIG / 2]


def iou_matrix(a: list, b: list) -> np.ndarray:
    """IoU between two lists of (x, y, w, h) boxes -> (len(a), len(b))."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    ax1, ay1, ax2, ay2 = A[:, 0], A[:, 1], A[:, 0] + A[:, 2], A[:, 1] + A[:, 3]
    bx1, by1, bx2, by2 = B[:, 0], B[:, 1], B[:, 0] + B[:, 2], B[:, 1] + B[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    union = (A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def _match_frame(gt_boxes: list, pred_boxes: list, iou_threshold: float) -> list:
    """Optimal one-to-one matching on IoU >= threshold -> [(gi, pj, iou)]."""
    iou = iou_matrix(gt_boxes, pred_boxes)
    if iou.size == 0:
        return []
    cost = np.where(iou >= iou_threshold, 1.0 - iou, _BIG)
    return [(i, j, float(iou[i, j])) for i, j in _linear_assignment(cost)]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionScore:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    frames: int

    @property
    def f1(self) -> float:
        d = self.precision + self.recall
        return 2 * self.precision * self.recall / d if d > 0 else 0.0


def evaluate_detection(gt: dict, pred: dict, iou_threshold: float = 0.5, frames=None) -> DetectionScore:
    """Per-frame optimal matching at `iou_threshold`, summed over `frames`
    (default: the union of GT and prediction frames)."""
    frame_ids = sorted(set(gt) | set(pred)) if frames is None else list(frames)
    tp = fp = fn = 0
    for f in frame_ids:
        g = [r[1:5] for r in gt.get(f, [])]
        p = [r[1:5] for r in pred.get(f, [])]
        m = _match_frame(g, p, iou_threshold)
        tp += len(m)
        fp += len(p) - len(m)
        fn += len(g) - len(m)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return DetectionScore(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, frames=len(frame_ids))


# ---------------------------------------------------------------------------
# Tracking identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackingScore:
    idf1: float
    idp: float
    idr: float
    idtp: int
    idfp: int
    idfn: int
    id_switches: int
    fragmentations: int
    gt_tracks: int
    pred_tracks: int
    gt_boxes: int
    pred_boxes: int
    mostly_tracked: int  # GT tracks matched >= 80% of their frames (CLEAR-MOT matching)
    mostly_lost: int  # GT tracks matched <= 20%
    frame_tp: int
    frame_fp: int
    frame_fn: int


def _collect_tracks(table: dict) -> dict:
    """obj_id -> {frame: box}."""
    tracks: dict = {}
    for f, rows in table.items():
        for oid, x, y, w, h in rows:
            tracks.setdefault(oid, {})[f] = (x, y, w, h)
    return tracks


def _idf1(gt_tracks: dict, pred_tracks: dict, iou_threshold: float) -> tuple:
    gids = list(gt_tracks)
    pids = list(pred_tracks)
    G, P = len(gids), len(pids)
    total_gt = sum(len(t) for t in gt_tracks.values())
    total_pred = sum(len(t) for t in pred_tracks.values())
    if G == 0 or P == 0:
        return 0, total_pred, total_gt

    # overlap[i, j] = frames where both exist with IoU >= threshold
    overlap = np.zeros((G, P), dtype=np.int64)
    for i, gid in enumerate(gids):
        gframes = gt_tracks[gid]
        for j, pid in enumerate(pids):
            pframes = pred_tracks[pid]
            common = gframes.keys() & pframes.keys()
            if not common:
                continue
            n = 0
            for f in common:
                if iou_matrix([gframes[f]], [pframes[f]])[0, 0] >= iou_threshold:
                    n += 1
            overlap[i, j] = n

    glen = np.array([len(gt_tracks[g]) for g in gids], dtype=np.float64)
    plen = np.array([len(pred_tracks[p]) for p in pids], dtype=np.float64)
    N = G + P
    cost = np.full((N, N), _BIG, dtype=np.float64)
    cost[:G, :P] = glen[:, None] + plen[None, :] - 2.0 * overlap  # match: leftover FN + FP
    for i in range(G):
        cost[i, P + i] = glen[i]  # GT i unmatched: all FN
    for j in range(P):
        cost[G + j, j] = plen[j]  # pred j unmatched: all FP
    cost[G:, P:] = 0.0
    pairs = _linear_assignment(cost)
    idtp = 0
    for i, j in pairs:
        if i < G and j < P:
            idtp += int(overlap[i, j])
    return idtp, total_pred - idtp, total_gt - idtp


def evaluate_tracking(gt: dict, pred: dict, iou_threshold: float = 0.5, frames=None) -> TrackingScore:
    frame_ids = sorted(set(gt) | set(pred)) if frames is None else sorted(frames)
    gt = {f: gt.get(f, []) for f in frame_ids}
    pred = {f: pred.get(f, []) for f in frame_ids}
    gt_tracks = _collect_tracks(gt)
    pred_tracks = _collect_tracks(pred)

    idtp, idfp, idfn = _idf1(gt_tracks, pred_tracks, iou_threshold)
    denom = 2 * idtp + idfp + idfn
    idf1 = 2 * idtp / denom if denom else 0.0
    idp = idtp / (idtp + idfp) if (idtp + idfp) else 0.0
    idr = idtp / (idtp + idfn) if (idtp + idfn) else 0.0

    # CLEAR-MOT style sequential matching for IDSW / fragmentation.
    prev_match: dict = {}  # gt id -> pred id matched in the previous frame it appeared
    last_pred: dict = {}  # gt id -> last pred id it was ever matched to
    tracked_state: dict = {}  # gt id -> was matched in its previous frame?
    ever_tracked: set = set()
    matched_frames: dict = {}
    gt_frames_count: dict = {}
    id_switches = fragmentations = 0
    frame_tp = frame_fp = frame_fn = 0

    for f in frame_ids:
        grows, prows = gt[f], pred[f]
        gboxes = [r[1:5] for r in grows]
        pboxes = [r[1:5] for r in prows]
        iou = iou_matrix(gboxes, pboxes)
        pred_index = {r[0]: j for j, r in enumerate(prows)}
        matches: dict = {}  # gi -> pj
        used_p: set = set()
        # 1) keep previous correspondences that still overlap
        for gi, row in enumerate(grows):
            pid = prev_match.get(row[0])
            if pid is not None and pid in pred_index:
                pj = pred_index[pid]
                if iou[gi, pj] >= iou_threshold and pj not in used_p:
                    matches[gi] = pj
                    used_p.add(pj)
        # 2) optimal assignment for the rest
        rest_g = [gi for gi in range(len(grows)) if gi not in matches]
        rest_p = [pj for pj in range(len(prows)) if pj not in used_p]
        if rest_g and rest_p:
            sub = iou[np.ix_(rest_g, rest_p)]
            cost = np.where(sub >= iou_threshold, 1.0 - sub, _BIG)
            for a, b in _linear_assignment(cost):
                matches[rest_g[a]] = rest_p[b]
        frame_tp += len(matches)
        frame_fn += len(grows) - len(matches)
        frame_fp += len(prows) - len(matches)

        new_prev: dict = {}
        for gi, row in enumerate(grows):
            gid = row[0]
            gt_frames_count[gid] = gt_frames_count.get(gid, 0) + 1
            if gi in matches:
                pid = prows[matches[gi]][0]
                if gid in last_pred and last_pred[gid] != pid:
                    id_switches += 1
                last_pred[gid] = pid
                new_prev[gid] = pid
                matched_frames[gid] = matched_frames.get(gid, 0) + 1
                if gid in ever_tracked and not tracked_state.get(gid, False):
                    fragmentations += 1
                ever_tracked.add(gid)
                tracked_state[gid] = True
            else:
                tracked_state[gid] = False
        prev_match = new_prev

    mt = sum(1 for g, n in gt_frames_count.items() if matched_frames.get(g, 0) / n >= 0.8)
    ml = sum(1 for g, n in gt_frames_count.items() if matched_frames.get(g, 0) / n <= 0.2)
    return TrackingScore(
        idf1=idf1, idp=idp, idr=idr, idtp=idtp, idfp=idfp, idfn=idfn,
        id_switches=id_switches, fragmentations=fragmentations,
        gt_tracks=len(gt_tracks), pred_tracks=len(pred_tracks),
        gt_boxes=sum(len(t) for t in gt_tracks.values()),
        pred_boxes=sum(len(t) for t in pred_tracks.values()),
        mostly_tracked=mt, mostly_lost=ml,
        frame_tp=frame_tp, frame_fp=frame_fp, frame_fn=frame_fn,
    )


# ---------------------------------------------------------------------------
# Duplicate identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateIdentityStats:
    """Internal-consistency statistic (needs NO ground truth).

    `states`: number of distinct track-state objects produced.
    `identity_keys`: number of distinct identity keys among them.
    `duplicated_keys`: identity keys carried by more than one state.
    `duplicate_rate`: (states - identity_keys) / states -- the fraction of
    states that are "extra" copies under an already-used key; 0.0 iff every
    state owns a unique key. This is the quantity the invariant "never two
    distinct track states sharing a vehicle identity key" requires to be 0.
    """

    states: int
    identity_keys: int
    duplicated_keys: int
    duplicate_rate: float


def duplicate_identity_stats(state_keys: list) -> DuplicateIdentityStats:
    """`state_keys` = one identity key per track-state object (finalized and
    live), in any order."""
    counts: dict = {}
    for k in state_keys:
        counts[k] = counts.get(k, 0) + 1
    states = len(state_keys)
    keys = len(counts)
    return DuplicateIdentityStats(
        states=states,
        identity_keys=keys,
        duplicated_keys=sum(1 for c in counts.values() if c > 1),
        duplicate_rate=(states - keys) / states if states else 0.0,
    )


def duplicate_ids_per_window(window_track_ids: list) -> tuple:
    """(entries, duplicate_entries) for the audit's other framing: how many
    per-window track entries carry an id already present in that window.
    `window_track_ids` is a list (one per window) of id lists."""
    entries = dup = 0
    for ids in window_track_ids:
        entries += len(ids)
        dup += len(ids) - len(set(ids))
    return entries, dup

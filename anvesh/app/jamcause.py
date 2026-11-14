"""Jam-cause candidate: which vehicle is at the origin of a queue of stopped vehicles.

WHAT THIS IS. A transparent, rule-based reading of the tracked boxes, evaluated twice a second while the video plays:

  1. A vehicle is STOPPED when its ground point has moved less than a fraction of its own width over the last
     `STOP_WINDOW_S` seconds (scale-free: a far vehicle and a near one are judged by their own size).
  2. Stopped vehicles that are close to each other (a few vehicle widths) form a QUEUE; at least `min_vehicles`
     stopped vehicles make a JAM.
  3. The vehicle at the downstream front of the queue (furthest along the direction the traffic was travelling
     BEFORE it stopped) is the JAM-CAUSE CANDIDATE. Three observable facts are reported with it:
       - did it stop before the vehicles behind it (`stopped_first`),
       - how many vehicles queued up behind it (`behind`),
       - whether vehicles further upstream stopped later (`order_correlation`, Spearman over the queue) - the signature
         of a queue that grew backwards from the front, as opposed to one that merely stopped together.
     It is rated `likely` when it stopped first and the order agrees, else `possible`.

WHAT THIS IS NOT. It does not know WHY the front vehicle stopped (a breakdown, a crash, a red light, a merge) and it
cannot prove responsibility. It is always worded "candidate". Nothing is learned and nothing is pre-computed:
the verdict is recomputed from the live tracks and can change.

Everything here works in IMAGE space (pixels, relative to each vehicle's own width), so it needs no calibration.
"""

from __future__ import annotations

import math

from anvesh.app.detectors import point_in_polygon

STOP_WINDOW_S = 2.0
MIN_STOPPED_S = 2.0
CLEAR_AFTER_S = 8.0  # a jam is declared over after this long without a qualifying queue
EVAL_EVERY_S = 0.5


def _spearman(xs: list, ys: list):
    n = len(xs)
    if n < 4:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


class JamCauseAnalyzer:
    def __init__(self, history, min_vehicles: int = 3, link_widths: float = 3.2, zones=()) -> None:
        self.history = history
        self.zones = tuple(zones or ())  # configured direction zones: {"polygon", "expected"}; the fallback when no vehicle was seen moving first
        self.min_vehicles = int(min_vehicles)
        self.link_widths = float(link_widths)
        self._trk: dict = {}  # track_id -> {"stop_since", "heading"(unit vec of pre-stop travel), "w"}
        self._last_eval = -1e9
        self._jam = None  # the open jam record
        self.closed: list = []  # last few finished jams (newest last)
        self._seq = 0
        self.state: dict = self._idle()

    @staticmethod
    def _idle() -> dict:
        return {"active": False, "vehicles": 0, "cause_track": None, "members": [], "rating": None, "text": "No jam", "jam_id": None}

    # -- per-track facts -------------------------------------------------------------------------
    def _facts(self, tid: str, t: float):
        samples = self.history.samples(tid)
        if len(samples) < 4:
            return None
        last = samples[-1]
        w = max(8.0, last.bbox[2] - last.bbox[0])
        old = None
        for s in samples:
            if t - s.t <= STOP_WINDOW_S:
                old = s
                break
        if old is None or last.t - old.t < STOP_WINDOW_S * 0.6:
            return None
        disp = math.hypot(last.ground[0] - old.ground[0], last.ground[1] - old.ground[1])
        stopped = disp < max(6.0, 0.30 * w)
        rec = self._trk.setdefault(tid, {"stop_since": None, "heading": None})
        if stopped:
            if rec["stop_since"] is None:
                rec["stop_since"] = old.t
                rec["stop_known"] = bool(rec.get("was_moving"))  # False: it was already standing still when first tracked
        else:
            rec["stop_since"] = None
            rec["was_moving"] = True
            # the direction the vehicle travels while moving; used later to say which stopped vehicle is "in front"
            far = samples[max(0, len(samples) - 12)]
            dx, dy = last.ground[0] - far.ground[0], last.ground[1] - far.ground[1]
            n = math.hypot(dx, dy)
            if n > 0.6 * w:
                h = (dx / n, dy / n)
                prev = rec["heading"]
                rec["heading"] = h if prev is None else _unit((0.7 * prev[0] + 0.3 * h[0], 0.7 * prev[1] + 0.3 * h[1]))
        return {"w": w, "ground": last.ground, "stopped_s": (t - rec["stop_since"]) if rec["stop_since"] is not None else 0.0,
                "stop_since": rec["stop_since"] if rec.get("stop_known") else None, "heading": rec["heading"], "cls": last.vehicle_class}

    # -- main entry ------------------------------------------------------------------------------
    def update(self, t: float) -> dict:
        if t - self._last_eval < EVAL_EVERY_S:
            return self.state
        self._last_eval = t
        active = self.history.active(t, 0.6)
        facts = {}
        for tid in active:
            f = self._facts(tid, t)
            if f is not None:
                facts[tid] = f
        for tid in [k for k in self._trk if k not in active and t - (self._trk[k].get("seen", t)) > 30]:
            self._trk.pop(tid, None)
        for tid in active:
            self._trk.setdefault(tid, {"stop_since": None, "heading": None})["seen"] = t
        stopped = [tid for tid, f in facts.items() if f["stopped_s"] >= MIN_STOPPED_S]
        cluster = self._best_cluster(stopped, facts)
        if cluster and len(cluster) >= self.min_vehicles:
            self.state = self._describe(cluster, facts, t)
        elif self._jam is not None and t - self._jam["last_t"] <= CLEAR_AFTER_S:
            self.state = dict(self.state, holding=True)  # brief gap (a vehicle lost for a moment): keep the verdict
        else:
            if self._jam is not None:
                self._jam["ended_t"] = self._jam["last_t"]
                self.closed = (self.closed + [self._jam_summary(self._jam)])[-5:]
                self._jam = None
            self.state = self._idle()
        return self.state

    def _best_cluster(self, stopped: list, facts: dict):
        if not stopped:
            return None
        parent = {k: k for k in stopped}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, a in enumerate(stopped):
            for b in stopped[i + 1:]:
                ga, gb = facts[a]["ground"], facts[b]["ground"]
                gap = max(facts[a]["w"], facts[b]["w"]) * self.link_widths + 8.0
                if math.hypot(ga[0] - gb[0], ga[1] - gb[1]) <= gap:
                    parent[find(a)] = find(b)
        groups: dict = {}
        for k in stopped:
            groups.setdefault(find(k), []).append(k)
        return max(groups.values(), key=len)

    def _describe(self, members: list, facts: dict, t: float) -> dict:
        heads = [facts[m]["heading"] for m in members if facts[m]["heading"] is not None]
        direction, dir_src = None, None
        if heads:
            direction, dir_src = _unit((sum(h[0] for h in heads), sum(h[1] for h in heads))), "the motion of these vehicles before they stopped"
        else:
            votes = [self._zone_dir(facts[m]["ground"]) for m in members]
            votes = [v for v in votes if v is not None]
            if votes:
                direction, dir_src = _unit((sum(v[0] for v in votes), sum(v[1] for v in votes))), "the configured direction of the lane area"
        if self._jam is None:
            self._seq += 1
            self._jam = {"id": self._seq, "start_t": t, "peak": 0, "cause_log": []}
        jam = self._jam
        jam["last_t"] = t
        jam["peak"] = max(jam["peak"], len(members))
        if direction is None:
            return {"active": True, "vehicles": len(members), "cause_track": None, "members": members, "rating": None, "jam_id": jam["id"],
                    "since_s": round(t - jam["start_t"], 1), "peak": jam["peak"],
                    "text": f"Jam: {len(members)} vehicles stopped together - front vehicle cannot be told (travel direction unknown)",
                    "evidence": []}
        proj = {m: facts[m]["ground"][0] * direction[0] + facts[m]["ground"][1] * direction[1] for m in members}
        head = max(members, key=lambda m: proj[m])
        starts = {m: facts[m]["stop_since"] for m in members if facts[m]["stop_since"] is not None}
        others = [m for m in members if m != head and m in starts]
        stopped_first = bool(head in starts and others and starts[head] <= min(starts[m] for m in others) + 1.0)
        order_known = head in starts and bool(others)
        lead_s = None
        if head in starts and others:
            lead_s = round(sorted(starts[m] for m in others)[len(others) // 2] - starts[head], 1)
        corr = None
        if len(members) >= 4 and len(starts) >= 4:
            ms = [m for m in members if m in starts]
            corr = _spearman([-proj[m] for m in ms], [starts[m] for m in ms])  # further upstream -> stopped later => positive
        ordered = corr is not None and corr >= 0.5
        rating = "likely" if (stopped_first and (ordered or corr is None)) else "possible"
        h = head.rsplit("-t", 1)[-1].lstrip("0") or "0"
        ev = [f"vehicle in front of the queue; {len(members) - 1} stopped behind it"]
        if stopped_first:
            ev.append("it stopped first" + (f" ({lead_s:+.1f} s before the middle of the queue)" if lead_s is not None else ""))
        elif order_known:
            ev.append("it did NOT clearly stop first")
        else:
            ev.append("who stopped first could not be measured (vehicles were already standing still when first tracked)")
        ev.append(f"'front' means furthest along {dir_src}")
        if corr is not None:
            ev.append(f"vehicles further back stopped later (order correlation {corr:+.2f})" if ordered else f"stop order does not follow the queue (correlation {corr:+.2f})")
        jam["cause_log"].append(head)
        text = (f"Jam of {len(members)} stopped vehicles. Candidate cause: {facts[head]['cls']} #{h} at the front of the queue"
                f" ({rating}); " + ("it stopped first." if stopped_first else ("it did not clearly stop first." if order_known else "who stopped first could not be measured.")))
        return {"active": True, "vehicles": len(members), "cause_track": head, "cause_label": f"{facts[head]['cls']} #{h}", "members": members,
                "rating": rating, "stopped_first": stopped_first, "behind": len(members) - 1, "order_correlation": None if corr is None else round(corr, 2),
                "lead_s": lead_s, "since_s": round(t - jam["start_t"], 1), "peak": jam["peak"], "jam_id": jam["id"], "text": text, "evidence": ev,
                "direction": [round(direction[0], 2), round(direction[1], 2)], "direction_source": dir_src,
                "disclaimer": "Candidate only: the system sees that this vehicle is at the front, not why it stopped."}

    def _zone_dir(self, ground):
        for z in self.zones:
            if point_in_polygon(ground, z["polygon"]):
                return _unit(tuple(z["expected"]))
        return None

    def _jam_summary(self, jam: dict) -> dict:
        last = jam["cause_log"][-1] if jam["cause_log"] else None
        return {"jam_id": jam["id"], "start_t": round(jam["start_t"], 1), "end_t": round(jam["ended_t"], 1), "peak_vehicles": jam["peak"],
                "last_cause_track": last}


def _unit(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-9 else (0.0, 0.0)

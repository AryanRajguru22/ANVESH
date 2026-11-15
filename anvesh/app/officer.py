"""The officer layer: everything the simple one-screen view needs, built from what the perception engine already produces.

  * `LaneMap`        - which configured lane a vehicle is in (point-in-polygon on its ground point) and readable lane names.
  * `AlertBoard`     - plain-language alerts (wrong-way, possible accident, stopped, over-speed, jam ...) with a snapshot each.
  * `OfficerLayer`   - ties the lane map, alert board, jam-cause analyzer and plate reader together, draws the large annotated frame and
                       produces the six status tiles (lanes, speed, wrong-way, accident, number plate, jam cause).

NOTHING HERE DETECTS ANYTHING BY ITSELF except the jam-cause candidate (jamcause.py) and the plate reading (plate_reader.py). Wrong-way,
accident, stopped-vehicle and over-speed alerts are the existing detectors' events, shown in words an officer can read. Every statement keeps
the system's standing caveats: candidates for human review, approximate speed, simulated scenes labelled.
"""

from __future__ import annotations

import collections
import math
import re
import threading
import time

import cv2
import numpy as np

from anvesh.app.detectors import point_in_polygon
from anvesh.app.jamcause import JamCauseAnalyzer
from anvesh.app.plate_reader import PlateReader

TARGET_WIDTH = 1280
FONT = cv2.FONT_HERSHEY_DUPLEX

# kind -> (headline, severity colour name, plain explanation)
ALERT_KINDS = {
    "wrong_way": ("Wrong-way vehicle", "bad", "A vehicle is travelling against the direction of traffic in its lane area."),
    "possible_collision": ("Possible accident", "bad", "Two moving vehicles touched and both stopped suddenly. Check the clip."),
    "stopped_vehicle": ("Vehicle stopped", "warn", "A vehicle has stopped outside a queue."),
    "long_duration_stop": ("Vehicle stopped for a long time", "bad", "A vehicle has been stationary for a long time."),
    "overspeed": ("Over the speed limit", "bad", "A vehicle was measured above the speed limit (approximate speed)."),
    "sudden_braking": ("Sudden braking", "warn", "A vehicle slowed very sharply."),
    "wrong_lane": ("Wrong lane", "warn", "A vehicle is in a lane it is not allowed in."),
    "rash_driving": ("Rapid lane changes", "warn", "A vehicle changed lanes several times in quick succession."),
    "red_light_violation": ("Red light crossed", "bad", "A vehicle crossed the stop line while the signal was red."),
    "illegal_turn": ("Illegal turn", "warn", "A vehicle made a turn that is not allowed here."),
    "congestion": ("Heavy traffic", "warn", "Traffic is building up or crawling."),
    "jam": ("Traffic jam", "bad", "Several vehicles are stopped together in a queue."),
    "plate": ("Number plate read", "info", "A plate was read from a vehicle by the plate reader. Check it before relying on it."),
}
ACTIVE_ALERT_S = 8.0  # stream seconds an alert stays highlighted on the video
BGR = {"bad": (40, 40, 235), "warn": (0, 165, 255), "ok": (90, 190, 60), "info": (230, 160, 40)}


def lane_label(lane_id: str) -> str:
    """'TOWARD-3-offside' -> 'TOWARD 3'; a plain id such as 'L2' is kept."""
    m = re.match(r"^([A-Za-z]+)-(\d+)", lane_id or "")
    return f"{m.group(1).upper()} {m.group(2)}" if m else (lane_id or "")


class LaneMap:
    def __init__(self, lanes) -> None:
        self.lanes = [{"id": l["lane_id"], "label": lane_label(l["lane_id"]), "poly": [tuple(p) for p in l["vertices"]]} for l in (lanes or ())]

    def __bool__(self) -> bool:
        return bool(self.lanes)

    def lane_of(self, ground):
        for l in self.lanes:
            if point_in_polygon(ground, l["poly"]):
                return l
        return None


class AlertBoard:
    def __init__(self, keep: int = 80) -> None:
        self._lock = threading.Lock()
        self._items: collections.deque = collections.deque(maxlen=keep)
        self._thumbs: dict = {}
        self._seq = 0
        self.counts: collections.Counter = collections.Counter()

    def add(self, kind: str, detail: str, t: float, *, track_ids=(), incident_id=None, severity=None, simulated=False, thumb=None, extra=None) -> dict:
        head, sev, why = ALERT_KINDS.get(kind, (kind.replace("_", " ").capitalize(), "warn", ""))
        with self._lock:
            self._seq += 1
            a = {"id": self._seq, "kind": kind, "title": head, "detail": detail, "why": why, "severity": severity or sev, "t": round(t, 1),
                 "wall": time.time(), "track_ids": list(track_ids), "incident_id": incident_id, "simulated": bool(simulated),
                 "has_thumb": thumb is not None}
            if extra:
                a["extra"] = extra
            self._items.append(a)
            self.counts[kind] += 1
            if thumb is not None:
                self._thumbs[a["id"]] = thumb
                for k in [k for k in self._thumbs if k < self._seq - 40]:
                    self._thumbs.pop(k, None)
            return a

    def recent(self, n: int = 12) -> list:
        with self._lock:
            return list(self._items)[-n:][::-1]

    def thumb(self, alert_id: int):
        with self._lock:
            return self._thumbs.get(alert_id)

    def active(self, t: float, kinds=None) -> list:
        with self._lock:
            return [a for a in self._items if (kinds is None or a["kind"] in kinds) and 0 <= t - a["t"] <= ACTIVE_ALERT_S]


def _chip(img, text, x, y, fg=(255, 255, 255), bg=(0, 0, 0), scale=0.55, thick=1, pad=4, anchor="bl"):
    """Filled text label. (x,y) is the bottom-left (default) or top-left ('tl') corner; kept inside the image."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    w, h = tw + 2 * pad, th + base + 2 * pad - 2
    x0 = int(max(0, min(x, img.shape[1] - w)))
    y0 = int(y - h if anchor == "bl" else y)
    y0 = max(0, min(y0, img.shape[0] - h))
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), bg, -1)
    cv2.putText(img, text, (x0 + pad, y0 + pad + th - 1), FONT, scale, fg, thick, cv2.LINE_AA)
    return x0, y0, w, h


class OfficerLayer:
    def __init__(self, services_config, camera: dict, cfg, history, simulated: bool, plate_reader: PlateReader = None) -> None:
        self.cfg = cfg
        self.camera = camera
        self.simulated = simulated
        self.lanes = LaneMap(cfg.lanes)
        self.alerts = AlertBoard()
        self.plates_only = bool(cfg.threshold("plates_only", 0))  # a plate-reader test source: no traffic analysis is meaningful on still photos
        self._plates_alerted: set = set()
        self.jam = JamCauseAnalyzer(history, min_vehicles=int(cfg.threshold("jam_min_vehicles", 3)), zones=cfg.direction_zones)
        self.plates = plate_reader
        self._jam_alerted: set = set()
        self._lane_counts: dict = {}
        self._speed_now: list = []
        self.max_speed_seen = None
        self.over_limit_events = 0
        self.last_t = 0.0

    # -- events from the detectors ----------------------------------------------------------------
    def on_event(self, ev, incident: dict, thumb) -> None:
        kind = ev.event_type
        if self.plates_only:
            return
        if kind == "stopped_vehicle" and (ev.severity == "low" or (ev.indicators or {}).get("queue_member")):
            return  # a vehicle stopped inside a queue is the jam tile's business, not an alert of its own
        detail = ev.summary or ""
        if kind == "possible_collision":
            detail = "Two vehicles touched and stopped suddenly: " + ", ".join(f"#{self._short(t)}" for t in ev.track_ids)
        elif kind == "wrong_way":
            detail = f"{self._cls(ev)} #{self._short(ev.track_ids[0])} is moving against the traffic direction" if ev.track_ids else detail
        elif kind in ("stopped_vehicle", "long_duration_stop"):
            detail = f"{self._cls(ev)} #{self._short(ev.track_ids[0])} has stopped" if ev.track_ids else detail
        elif kind == "overspeed":
            detail = ev.summary or detail
        elif kind == "congestion":
            detail = "Traffic is building up or crawling in this view."
        self.alerts.add(kind, detail, ev.detected_offset_s, track_ids=ev.track_ids, incident_id=incident.get("incident_id"),
                        severity=None, simulated=self.simulated, thumb=thumb)

    @staticmethod
    def _short(tid: str) -> str:
        return (str(tid).rsplit("-t", 1)[-1].lstrip("0")) or "0"

    @staticmethod
    def _cls(ev) -> str:
        return (ev.involved[0]["vehicle_class"] if ev.involved else "vehicle").capitalize()

    # -- per frame ---------------------------------------------------------------------------------
    def on_frame(self, t: float, image, obs: list, live: list, thumb_fn) -> dict:
        self.last_t = t
        jam = self.jam.update(t) if not self.plates_only else JamCauseAnalyzer._idle()
        if jam.get("active") and jam.get("jam_id") not in self._jam_alerted and jam.get("cause_track") is not None:
            self._jam_alerted.add(jam["jam_id"])
            self.alerts.add("jam", jam["text"], t, track_ids=[jam["cause_track"]], simulated=self.simulated, thumb=thumb_fn(),
                            extra={"cause_track": jam["cause_track"], "rating": jam.get("rating")})
        if self.plates is not None and image is not None:
            for o in sorted(obs, key=lambda o: -(o.bbox[2] - o.bbox[0]))[:3]:
                self.plates.offer(o.track_id, image, o.bbox, o.vehicle_class)
        if self.plates is not None:
            for tid, res in list(self.plates.results.items()):
                if tid not in self._plates_alerted:
                    self._plates_alerted.add(tid)
                    self.alerts.add("plate", f"Read {res['text']} (reader confidence {res['conf']:.2f}). Check it against the picture.", t, track_ids=[tid],
                                    simulated=False, thumb=thumb_fn(), extra={"plate": res["text"]})
        lane_counts: dict = {}
        speeds = []
        for row in live:
            if row.get("lane"):
                lane_counts[row["lane"]] = lane_counts.get(row["lane"], 0) + 1
            if row.get("speed_kmh") is not None:
                speeds.append(row["speed_kmh"])
        self._lane_counts, self._speed_now = lane_counts, speeds
        if speeds:
            self.max_speed_seen = max(speeds) if self.max_speed_seen is None else max(self.max_speed_seen, max(speeds))
        return jam

    # -- drawing -----------------------------------------------------------------------------------
    def render(self, image, t: float, obs: list, live: list, jam: dict, speed_zone=None):
        h0, w0 = image.shape[:2]
        scale = max(1.0, TARGET_WIDTH / w0)
        out = cv2.resize(image, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR) if scale > 1.0 else image.copy()
        s = scale
        info = {r["track_full"]: r for r in live}
        self._draw_lanes(out, s)
        if speed_zone:
            cv2.polylines(out, [(np.array(speed_zone, np.float32) * s).astype(np.int32)], True, (200, 60, 200), 1, cv2.LINE_AA)
        ww = {tid for a in self.alerts.active(t, ("wrong_way",)) for tid in a["track_ids"]}
        acc = {tid for a in self.alerts.active(t, ("possible_collision",)) for tid in a["track_ids"]}
        stopped = {tid for a in self.alerts.active(t, ("stopped_vehicle", "long_duration_stop")) for tid in a["track_ids"]}
        speeding = {tid for a in self.alerts.active(t, ("overspeed",)) for tid in a["track_ids"]}
        cause = jam.get("cause_track") if jam.get("active") else None
        members = set(jam.get("members") or ()) if jam.get("active") else set()
        if members:
            self._draw_jam_hull(out, obs, members, s)
        for o in obs:
            x1, y1, x2, y2 = (int(v * s) for v in o.bbox)
            tid = o.track_id
            r = info.get(tid, {})
            tag_bits, colour, thick, flag = [], (215, 200, 160), 1, None
            if tid in ww:
                colour, thick, flag = BGR["bad"], 4, "WRONG WAY"
            elif tid in acc:
                colour, thick, flag = BGR["bad"], 4, "ACCIDENT?"
            elif tid == cause:
                colour, thick, flag = (230, 0, 200), 4, "JAM CAUSE?"
            elif tid in speeding:
                colour, thick, flag = BGR["bad"], 3, "TOO FAST"
            elif tid in stopped:
                colour, thick, flag = BGR["warn"], 3, "STOPPED"
            elif tid in members:
                colour, thick = BGR["warn"], 2
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick, cv2.LINE_AA)
            plate = self.plates.get(tid) if self.plates is not None else None
            if flag:
                _chip(out, flag, x1, y1 - 2, fg=(255, 255, 255), bg=colour, scale=0.62, thick=1)
            if r.get("speed_kmh") is not None or flag or plate:
                if r.get("lane_label"):
                    tag_bits.append(r["lane_label"])
                if r.get("speed_kmh") is not None:
                    tag_bits.append(f"{r['speed_kmh']:.0f} km/h~")
                if tag_bits:
                    _chip(out, "#" + self._short(tid) + "  " + "  ".join(tag_bits), x1, y2 + 22, fg=(255, 255, 255), bg=(60, 40, 20), scale=0.5)
            if plate:
                pb = plate.get("box")
                if pb:
                    px1, py1, px2, py2 = (int(v * s) for v in pb)
                    cv2.rectangle(out, (px1, py1), (px2, py2), (0, 230, 255), 3, cv2.LINE_AA)
                    _chip(out, plate["text"], px1, py2 + 30, fg=(0, 0, 0), bg=(0, 230, 255), scale=0.8, thick=2, anchor="tl")
                else:
                    _chip(out, plate["text"], x1, y2 + (46 if tag_bits else 22), fg=(0, 0, 0), bg=(0, 230, 255), scale=0.62, thick=1)
        self._draw_banner(out, t, len(obs), jam, s)
        return out

    def _draw_lanes(self, out, s: float) -> None:
        if not self.lanes:
            return
        shade = out.copy()
        palette = [(255, 170, 60), (255, 120, 40), (255, 200, 110), (90, 200, 120), (60, 170, 90), (120, 220, 150)]
        for i, l in enumerate(self.lanes.lanes):
            poly = (np.array(l["poly"], np.float32) * s).astype(np.int32)
            cv2.fillPoly(shade, [poly], palette[i % len(palette)])
        cv2.addWeighted(shade, 0.16, out, 0.84, 0, out)
        for l in self.lanes.lanes:
            poly = (np.array(l["poly"], np.float32) * s).astype(np.int32)
            cv2.polylines(out, [poly], True, (255, 255, 255), 1, cv2.LINE_AA)
            x, y = self._label_point(l["poly"], s, l["label"])
            _chip(out, l["label"], x, y, fg=(255, 255, 255), bg=(70, 50, 30), scale=0.42, pad=3, anchor="tl")

    @staticmethod
    def _label_point(poly, s: float, label: str):
        (tw, _), _b = cv2.getTextSize(label, FONT, 0.42, 1)
        need = (tw + 10) / s
        ys = [p[1] for p in poly]
        top, bottom = min(ys), max(ys)
        y = top + 0.30 * (bottom - top)
        for frac in (0.30, 0.38, 0.46, 0.54, 0.62, 0.70):
            y = top + frac * (bottom - top)
            xs = []
            n = len(poly)
            for i in range(n):
                (xa, ya), (xb, yb) = poly[i], poly[(i + 1) % n]
                if (ya <= y < yb) or (yb <= y < ya):
                    xs.append(xa + (y - ya) * (xb - xa) / (yb - ya))
            if len(xs) >= 2 and max(xs) - min(xs) >= need:
                return ((min(xs) + max(xs)) / 2 * s - tw / 2 - 3, y * s)
        xs = [p[0] for p in poly]
        return (sum(xs) / len(xs) * s - tw / 2, y * s)

    @staticmethod
    def _draw_jam_hull(out, obs, members, s: float) -> None:
        pts = []
        for o in obs:
            if o.track_id in members:
                x1, y1, x2, y2 = (o.bbox[0] * s, o.bbox[1] * s, o.bbox[2] * s, o.bbox[3] * s)
                pts += [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if len(pts) < 8:
            return
        hull = cv2.convexHull(np.array(pts, np.float32)).astype(np.int32)
        shade = out.copy()
        cv2.fillPoly(shade, [hull], (0, 140, 255))
        cv2.addWeighted(shade, 0.22, out, 0.78, 0, out)
        cv2.polylines(out, [hull], True, (0, 165, 255), 2, cv2.LINE_AA)

    def _draw_banner(self, out, t: float, n: int, jam: dict, s: float) -> None:
        w = out.shape[1]
        mm, ss = int(t // 60), int(t % 60)
        cv2.rectangle(out, (0, 0), (w, 34), (30, 22, 12), -1)
        cv2.putText(out, f"ANVESH LIVE   video {mm:02d}:{ss:02d}   {n} vehicles in view", (12, 24), FONT, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        if self.simulated:
            txt = "SIMULATED TEST SCENE - NOT REAL FOOTAGE"
            tw = cv2.getTextSize(txt, FONT, 0.62, 1)[0][0]
            cv2.putText(out, txt, (w - tw - 12, 24), FONT, 0.62, (0, 200, 255), 1, cv2.LINE_AA)
        # one big alert line, most urgent first
        line = None
        for kind, text, col in (("possible_collision", "POSSIBLE ACCIDENT", BGR["bad"]), ("wrong_way", "WRONG-WAY VEHICLE", BGR["bad"])):
            if self.alerts.active(t, (kind,)):
                line = (text, col)
                break
        if line is None and jam.get("active") and jam.get("cause_track") is not None:
            line = (f"TRAFFIC JAM  -  {jam['vehicles']} vehicles  -  candidate cause: {jam.get('cause_label', 'vehicle')}", (200, 0, 170))
        elif line is None and jam.get("active"):
            line = (f"TRAFFIC JAM  -  {jam['vehicles']} vehicles stopped together", (200, 0, 170))
        if line is not None:
            flash = int(t * 2) % 2 == 0 or line[1] != BGR["bad"]
            if flash:
                (tw, th), _b = cv2.getTextSize(line[0], FONT, 0.85, 2)
                x0 = (w - tw) // 2 - 16
                cv2.rectangle(out, (x0, 42), (x0 + tw + 32, 42 + th + 18), line[1], -1)
                cv2.putText(out, line[0], (x0 + 16, 42 + th + 8), FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # -- tiles --------------------------------------------------------------------------------------
    def _plate_tile(self) -> dict:
        if self.plates is None or not self.plates.available:
            return {"state": "off", "value": "Unavailable", "detail": (self.plates.reason if self.plates is not None else "plate reader not enabled for this source")}
        ps = self.plates.summary()
        if ps["plates_read"]:
            last = ps["recent"][0]
            return {"state": "ok", "value": last["text"], "detail": f"{ps['plates_read']} plate(s) read" + (" - confirmed by a second read" if last.get("confirmed") else " - single read, please check")}
        if ps["largest_vehicle_px_seen"] and ps["largest_vehicle_px_seen"] < ps["min_vehicle_px"]:
            return {"state": "warn", "value": "Too small", "detail": f"vehicles are only {ps['largest_vehicle_px_seen']} px wide; a plate needs about {ps['min_vehicle_px']} px"}
        return {"state": "info", "value": "Looking", "detail": "waiting for a close, clear vehicle"}

    def tiles(self, t: float, jam: dict, speed_limit, calibrated: bool, approximate: bool, live: list) -> list:
        if self.plates_only:
            na = {"state": "off", "value": "n/a", "detail": "This test shows only the plate reader."}
            return [{"key": "lanes", "title": "Lanes", **na}, {"key": "speed", "title": "Speed", **na}, {"key": "wrongway", "title": "Wrong-way", **na},
                    {"key": "accident", "title": "Accident", **na}, {"key": "plate", "title": "Number plate", **self._plate_tile()},
                    {"key": "jam", "title": "Jam cause", **na}]
        ww_n, acc_n = self.alerts.counts.get("wrong_way", 0), self.alerts.counts.get("possible_collision", 0)
        ww_now, acc_now = bool(self.alerts.active(t, ("wrong_way",))), bool(self.alerts.active(t, ("possible_collision",)))
        # lanes
        if self.lanes:
            counts = {k: v for k, v in self._lane_counts.items() if v}
            if counts:
                busiest = max(sorted(counts), key=counts.get)
                lane_detail = f"{sum(counts.values())} vehicles in lanes now; busiest {busiest} ({counts[busiest]})"
            else:
                lane_detail = "no vehicle in a lane right now"
            lanes_t = {"state": "ok", "value": f"{len(self.lanes.lanes)} lanes", "detail": lane_detail}
        else:
            lanes_t = {"state": "off", "value": "Not set up", "detail": "Lane areas are not drawn for this camera."}
        # speed
        if calibrated:
            sp = self._speed_now
            over = self.alerts.counts.get("overspeed", 0)
            if sp:
                value = f"{max(sp):.0f} km/h"
                detail = f"fastest of {len(sp)} measured now" + (" (approximate)" if approximate else "")
            else:
                value, detail = "Measuring", "speed is read inside the magenta zone on the video"
            speed_t = {"state": "bad" if over else "ok", "value": value, "detail": detail + (f" - limit {speed_limit:.0f}" if speed_limit else "") + (f" - {over} over limit" if over else "")}
        else:
            speed_t = {"state": "off", "value": "Not calibrated", "detail": "Speed needs a calibrated camera; not shown instead of guessing."}
        ww_t = {"state": "bad" if ww_now else "ok", "value": "WRONG WAY" if ww_now else "None", "detail": f"{ww_n} detected so far" if ww_n else "No vehicle against traffic"}
        acc_t = {"state": "bad" if acc_now else "ok", "value": "POSSIBLE ACCIDENT" if acc_now else "None", "detail": f"{acc_n} detected so far" if acc_n else "No collision pattern seen"}
        plate_t = self._plate_tile()
        # jam
        if jam.get("active"):
            jam_t = {"state": "bad", "value": f"JAM - {jam['vehicles']} vehicles", "detail": jam.get("text", "")}
        else:
            last = self.jam.closed[-1] if self.jam.closed else None
            jam_t = {"state": "ok", "value": "No jam", "detail": (f"last jam peaked at {last['peak_vehicles']} vehicles" if last else "traffic is moving")}
        return [{"key": "lanes", "title": "Lanes", **lanes_t}, {"key": "speed", "title": "Speed", **speed_t},
                {"key": "wrongway", "title": "Wrong-way", **ww_t}, {"key": "accident", "title": "Accident", **acc_t},
                {"key": "plate", "title": "Number plate", **plate_t}, {"key": "jam", "title": "Jam cause", **jam_t}]

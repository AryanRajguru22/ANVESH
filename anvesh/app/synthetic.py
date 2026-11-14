"""Scripted SIMULATED traffic for workflow tests and the clearly-labelled simulated demo.

Everything produced here is synthetic: the frames are drawn rectangles with a
"SIMULATED" banner, and `ScriptedTracker` returns the scripted boxes as "detections".
It exists so the incident -> evidence -> case -> handoff workflow can be exercised for
event types the available real footage does not contain (a collision, a wrong-way
vehicle) WITHOUT pretending a real detector found them. A camera using it must have
data class `simulated`; the API refuses anything else (`cameras.validate_source`).

It is never used when a real source is configured, and real inference on real footage is
demonstrated separately.
"""

from __future__ import annotations

import numpy as np

from anvesh.perception.detection import COCO_CLASS_NAMES
from anvesh.perception.ingestion import SourceHealth
from anvesh.perception.types import Frame, RawDetection, TrackedDetection
from anvesh.storage.schemas import VehicleClass

WIDTH, HEIGHT, FPS = 640, 360, 10.0
_CLASS_BOX = {"car": (60, 36, 2), "truck": (90, 44, 7), "bus": (100, 46, 5), "motorcycle": (26, 24, 3)}


def _obj(cls, waypoints):
    w, h, coco = _CLASS_BOX[cls]
    return {"class": cls, "w": w, "h": h, "coco": coco, "path": [(float(t), float(x), float(y)) for t, x, y in waypoints]}


def _scenario(name: str) -> dict:
    cross = [_obj("car", [(0, -60, 200 + 40 * i), (14, 700, 200 + 40 * i)]) for i in range(3)]
    if name == "stopped_car":
        # one car drives in and stops for the rest of the clip; two others keep moving
        stopper = _obj("car", [(0, -60, 250), (4, 260, 250), (30, 260, 250)])
        return {"duration": 30.0, "objects": cross[:2] + [stopper]}
    if name == "collision":
        a = _obj("car", [(0, -60, 250), (4, 300, 250), (5, 305, 250), (30, 305, 250)])
        b = _obj("car", [(0, 700, 250), (4, 340, 250), (5, 338, 250), (30, 338, 250)])
        return {"duration": 30.0, "objects": [a, b] + cross[:1]}
    if name == "wrong_way":
        norm = [_obj("car", [(0, -60, 200 + 40 * i), (14, 700, 200 + 40 * i)]) for i in range(2)]
        wrong = _obj("car", [(0, 700, 280), (14, -60, 280)])
        return {"duration": 14.0, "objects": norm + [wrong]}
    if name == "red_light":
        runner = _obj("car", [(0, 320, 40), (10, 320, 340)])
        return {"duration": 10.0, "objects": [runner]}
    if name == "queue":
        objs = [_obj("car", [(0, 80 + 70 * i, 250), (40, 83 + 70 * i, 250)]) for i in range(8)]
        return {"duration": 40.0, "objects": objs}
    if name.startswith("show_"):
        return _show_scenario(name)
    raise ValueError(f"unknown synthetic scenario {name!r}")


LANE_Y = (200.0, 250.0, 300.0)  # lane centres of the demonstration scenes (road band y 175..325)


def _stream(lane_y: float, first: float, last: float, gap: float, cls: str = "car", speed: float = 110.0) -> list:
    """Free-flowing cars entering from the left every `gap` seconds."""
    out, t = [], first
    travel = (700 + 60) / speed
    while t <= last:
        out.append(_obj(cls, [(t, -60, lane_y), (t + travel, 700, lane_y)]))
        t += gap
    return out


def _show_scenario(name: str) -> dict:
    """Scenes for the officer demonstration. Scripted boxes, always labelled SIMULATED on screen and in the data."""
    if name == "show_wrong_way":
        flow = _stream(LANE_Y[0], 0.0, 30.0, 3.1) + _stream(LANE_Y[1], 1.2, 30.0, 3.7, "truck", 90.0)
        wrong = _obj("car", [(7.0, 700, LANE_Y[2]), (7.0 + 760 / 105.0, -60, LANE_Y[2])])
        return {"duration": 24.0, "objects": flow + [wrong]}
    if name == "show_accident":
        flow = _stream(LANE_Y[0], 0.0, 30.0, 3.3) + _stream(LANE_Y[2], 0.7, 30.0, 4.0, "car", 100.0)
        a = _obj("car", [(4.0, -60, LANE_Y[1]), (8.0, 300, LANE_Y[1]), (8.8, 306, LANE_Y[1]), (40.0, 306, LANE_Y[1])])
        b = _obj("car", [(4.0, 700, LANE_Y[1]), (8.0, 340, LANE_Y[1]), (8.8, 338, LANE_Y[1]), (40.0, 338, LANE_Y[1])])
        return {"duration": 30.0, "objects": flow + [a, b]}
    if name == "show_jam":
        flow = _stream(LANE_Y[0], 0.0, 40.0, 3.2) + _stream(LANE_Y[2], 1.0, 40.0, 3.9, "car", 100.0)
        stop_x = 470.0
        truck = _obj("truck", [(0.0, -60, LANE_Y[1]), (5.3, stop_x, LANE_Y[1]), (60.0, stop_x, LANE_Y[1])])
        followers = []
        for i, enter in enumerate((2.5, 5.0, 7.5, 10.0), start=1):
            x = stop_x - 85.0 * i
            followers.append(_obj("car", [(enter, -60, LANE_Y[1]), (enter + (x + 60) / 100.0, x, LANE_Y[1]), (60.0, x, LANE_Y[1])]))
        return {"duration": 44.0, "objects": flow + [truck] + followers}
    raise ValueError(f"unknown synthetic scenario {name!r}")


SCENARIOS = ("stopped_car", "collision", "wrong_way", "red_light", "queue", "show_wrong_way", "show_accident", "show_jam")


def _position(obj: dict, t: float):
    path = obj["path"]
    if t < path[0][0] or t > path[-1][0]:
        return None
    for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:]):
        if t0 <= t <= t1:
            a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
    return path[-1][1], path[-1][2]


def boxes_at(scenario: dict, t: float) -> list:
    """[(object_index, class_name, coco_id, (x1,y1,x2,y2))] for the objects on screen at `t`."""
    out = []
    for i, obj in enumerate(scenario["objects"]):
        pos = _position(obj, t)
        if pos is None:
            continue
        cx, cy = pos
        x1, y1, x2, y2 = cx - obj["w"] / 2, cy - obj["h"] / 2, cx + obj["w"] / 2, cy + obj["h"] / 2
        if x2 < 0 or x1 > WIDTH or y2 < 0 or y1 > HEIGHT:
            continue
        out.append((i, obj["class"], obj["coco"], (max(0.0, x1), max(0.0, y1), min(float(WIDTH), x2), min(float(HEIGHT), y2))))
    return out


_COLOURS = [(190, 70, 50), (60, 90, 200), (200, 200, 205), (70, 160, 90), (40, 170, 220), (120, 60, 150), (150, 150, 155), (60, 60, 70)]
_ROAD_CACHE = {}


def _road():
    """Static asphalt scene with three dashed lanes (drawn once, copied per frame)."""
    import cv2

    if "img" not in _ROAD_CACHE:
        img = np.full((HEIGHT, WIDTH, 3), (95, 120, 85), np.uint8)  # verge
        cv2.rectangle(img, (0, 168), (WIDTH, 332), (58, 58, 62), -1)
        cv2.rectangle(img, (0, 168), (WIDTH, 172), (210, 210, 210), -1)
        cv2.rectangle(img, (0, 328), (WIDTH, 332), (210, 210, 210), -1)
        for y in (225, 275):
            for x in range(0, WIDTH, 40):
                cv2.line(img, (x, y), (x + 22, y), (225, 225, 225), 2)
        _ROAD_CACHE["img"] = img
    return _ROAD_CACHE["img"].copy()


def _draw_vehicle(img, cls, idx, x1, y1, x2, y2, right=True):
    import cv2

    col = _COLOURS[idx % len(_COLOURS)]
    if cls in ("truck", "bus"):
        col = (200, 200, 205) if cls == "truck" else (60, 90, 200)
    cv2.rectangle(img, (x1 + 1, y1 + 3), (x2 + 1, y2 + 3), (25, 25, 28), -1)  # shadow
    cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 22), 1)
    wx = (x2 - x1)
    if cls in ("car", "motorcycle"):
        a, b = (x1 + int(0.55 * wx), x1 + int(0.85 * wx)) if right else (x1 + int(0.15 * wx), x1 + int(0.45 * wx))
        cv2.rectangle(img, (a, y1 + 4), (b, y2 - 4), (40, 45, 55), -1)  # windscreen
        cv2.rectangle(img, (x1 + int(0.3 * wx), y1 + 4), (x1 + int(0.55 * wx), y2 - 4), tuple(int(c * 0.8) for c in col), -1)  # roof
    else:
        cv2.rectangle(img, (x1 + int(0.78 * wx) if right else x1 + 3, y1 + 3), (x2 - 3 if right else x1 + int(0.22 * wx), y2 - 3), (40, 45, 55), -1)
        cv2.line(img, (x1 + int(0.7 * wx), y1), (x1 + int(0.7 * wx), y2), (90, 90, 95), 1)


class SyntheticFrameSource:
    """FrameSource yielding drawn frames of a named scenario at 10 fps. Finite."""

    live = False

    def __init__(self, scenario: str = "stopped_car") -> None:
        self.scenario_name = scenario
        self._scenario = _scenario(scenario)
        self._frames = 0
        self._state = "idle"
        self._stop = False

    @property
    def scenario(self) -> dict:
        return self._scenario

    def fps(self) -> float:
        return FPS

    def _moving_right(self, idx: int, t: float) -> bool:
        path = self._scenario["objects"][idx]["path"]
        return path[-1][1] >= path[0][1]

    def health(self) -> SourceHealth:
        return SourceHealth(state=self._state, epoch=0, frames_read=self._frames, reconnects=0, source_gap_frames=0,
                            measured_fps=0.0, interval_jitter_s=0.0)

    def close(self) -> None:
        self._stop = True

    def __iter__(self):
        import cv2

        self._state, self._stop = "connected", False
        n = int(self._scenario["duration"] * FPS)
        for index in range(n):
            if self._stop:
                self._state = "stopped"
                return
            t = index / FPS
            img = _road()
            for (i, cls, _coco, (x1, y1, x2, y2)) in boxes_at(self._scenario, t):
                _draw_vehicle(img, cls, i, int(x1), int(y1), int(x2), int(y2), self._moving_right(i, t))
            cv2.putText(img, "SIMULATED - NOT REAL FOOTAGE", (12, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)
            self._frames += 1
            yield Frame(index=index, timestamp=t, image=img)
        self._state = "ended"


_COCO_VEHICLE = {2: VehicleClass.CAR, 3: VehicleClass.MOTORCYCLE, 5: VehicleClass.BUS, 7: VehicleClass.TRUCK}


class ScriptedTracker:
    """A Tracker that returns the scenario's scripted boxes with stable raw ids. SIMULATION ONLY."""

    lost_buffer_frames = 30
    simulated = True

    def __init__(self, scenario: str = "stopped_car") -> None:
        self._scenario = _scenario(scenario)

    def reset(self) -> None:
        pass

    def update(self, image, frame_index: int, timestamp: float):
        raw, tracked = [], []
        for i, cls, coco, box in boxes_at(self._scenario, timestamp):
            det = RawDetection(frame_index=frame_index, timestamp=timestamp, class_name=COCO_CLASS_NAMES.get(coco, cls),
                               vehicle_class=_COCO_VEHICLE[coco], confidence=0.9, bbox_xyxy=box)
            raw.append(det)
            tracked.append(TrackedDetection(detection=det, track_id=str(i + 1)))
        return raw, tracked

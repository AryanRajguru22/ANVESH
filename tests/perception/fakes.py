"""Lightweight fakes mimicking Ultralytics' `Results.boxes` shape, so
detection/tracking conversion logic is testable without loading a real
YOLO model or ByteTrack.
"""


class FakeTensor:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class FakeBoxes:
    def __init__(self, xyxy, cls, conf, ids=None):
        self.xyxy = FakeTensor(xyxy)
        self.cls = FakeTensor(cls)
        self.conf = FakeTensor(conf)
        self.id = FakeTensor(ids) if ids is not None else None

    def __len__(self):
        return len(self.xyxy.tolist())


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeEmptyBoxes:
    def __len__(self):
        return 0

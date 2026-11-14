"""Warm the heavy parts once at server start, in the background, so the first click does not wait.

Without this the first session spends ~15-20 s initialising CUDA and loading the detector, which looks like a hang to someone who has just pressed a
button. Everything here is best-effort: a failure only means the first session is slower; it never prevents the server from starting.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


def _run(model_path: str) -> None:
    t0 = time.time()
    try:
        import numpy as np
        from ultralytics import YOLO

        model = YOLO(model_path)
        blank = np.full((360, 640, 3), 100, np.uint8)
        for _ in range(2):
            model.predict(blank, imgsz=640, verbose=False)
        log.info("detector warm-up finished in %.1f s", time.time() - t0)
    except Exception:  # noqa: BLE001
        log.exception("detector warm-up failed (the first session will simply be slower)")
    try:
        from anvesh.app.plate_reader import PlateReader

        PlateReader().warm_up()
        log.info("plate reader warm-up finished at %.1f s", time.time() - t0)
    except Exception:  # noqa: BLE001
        log.exception("plate reader warm-up failed")


def start(model_path: str) -> threading.Thread:
    t = threading.Thread(target=_run, args=(str(model_path),), name="anvesh-warmup", daemon=True)
    t.start()
    return t

"""Threaded streaming pipeline around `StreamingPerceptionEngine`:

    reader thread (decode) --[BoundedQueue frames]--> worker thread
        (detect+track -> store -> windows) --[BoundedQueue windows]--> consumer thread

Every queue is bounded and every drop is counted:
  - FILE (lossless) mode: `frame_policy=BLOCK` -- the reader waits for the
    worker, nothing is dropped, so the engine sees exactly the frames a
    synchronous replay sees (the basis of batch/stream parity).
  - LIVE mode: `frame_policy=DROP_OLDEST` -- if the worker is slower than the
    source the OLDEST queued frame is discarded and counted; the engine then
    sees an index gap and flags the affected windows `degraded`.
The worker never blocks the reader in live mode, and the window queue
likewise never grows (BLOCK for files, DROP_OLDEST for live: a stale window
result is worth less than a fresh one, and the loss is counted).

Errors in any thread are captured and re-raised from `run()` in the caller's
thread; `stop()` unblocks every stage.
"""

from __future__ import annotations

import dataclasses
import threading
import time

from anvesh.streaming.engine import StreamingPerceptionEngine
from anvesh.streaming.queues import BoundedQueue, DropPolicy, QueueClosed

_SENTINEL = object()


class StreamingPipeline:
    def __init__(
        self,
        source,
        engine: StreamingPerceptionEngine,
        on_window=None,
        frame_queue_size: int = 8,
        window_queue_size: int = 16,
        frame_policy: DropPolicy = DropPolicy.BLOCK,
        window_policy: DropPolicy = None,
        max_frames: int = None,
        idle_poll_s: float = None,
    ) -> None:
        if idle_poll_s is not None and idle_poll_s <= 0:
            raise ValueError("idle_poll_s must be > 0 when given")
        # LIVENESS (M-A.1): with `idle_poll_s` set, a worker that receives no frame for
        # that long calls `engine.tick(...)` with the source's stream clock/state, so a
        # silent camera still closes windows (as UNAVAILABLE) and changes `engine.health`
        # instead of the pipeline waiting forever for a frame that never comes. Leave it
        # None for recorded files (the reader simply ends).
        self._idle_poll_s = idle_poll_s
        self._source = source
        self._engine = engine
        self._on_window = on_window
        self._max_frames = max_frames
        self.frame_queue = BoundedQueue(frame_queue_size, frame_policy, name="frames")
        wp = window_policy or (DropPolicy.BLOCK if DropPolicy(frame_policy) == DropPolicy.BLOCK else DropPolicy.DROP_OLDEST)
        self.window_queue = BoundedQueue(window_queue_size, wp, name="windows")
        self._errors: list = []
        self._stop = threading.Event()
        self.reader_frames = 0
        self.wall_seconds = 0.0

    def stop(self) -> None:
        self._stop.set()
        try:
            self._source.close()
        except Exception:  # noqa: BLE001
            pass
        self.frame_queue.close()
        self.window_queue.close()

    # -- threads -----------------------------------------------------------

    def _reader(self) -> None:
        try:
            for frame in self._source:
                if self._stop.is_set():
                    break
                self.frame_queue.put(frame)
                self.reader_frames += 1
                if self._max_frames is not None and self.reader_frames >= self._max_frames:
                    break
        except QueueClosed:
            pass
        except Exception as exc:  # noqa: BLE001
            self._errors.append(exc)
        finally:
            try:
                self._source.close()
            except Exception:  # noqa: BLE001
                pass
            self.frame_queue.close()

    def _worker(self) -> None:
        try:
            while True:
                try:
                    frame = self.frame_queue.get(timeout=self._idle_poll_s)
                except QueueClosed:
                    break
                if frame is None:  # idle: no frame within idle_poll_s
                    for snap in self._idle_tick():
                        self.window_queue.put(snap)
                    continue
                for snap in self._engine.process_frame(frame):
                    self.window_queue.put(snap)
            for snap in self._engine.finish(duration=self._source_stream_time()):
                self.window_queue.put(snap)
        except QueueClosed:
            pass
        except Exception as exc:  # noqa: BLE001
            self._errors.append(exc)
            self.stop()
        finally:
            self.window_queue.close()

    def _source_stream_time(self):
        clock = getattr(self._source, "stream_time", None)
        return clock() if callable(clock) else None

    def _idle_tick(self) -> list:
        health = getattr(self._source, "health", None)
        state = health().state if callable(health) else None
        now = self._source_stream_time()
        if now is None:
            now = self._engine.stream_time if self._engine.stream_time is not None else 0.0
        return self._engine.tick(now, state)

    def _consumer(self) -> None:
        try:
            while True:
                try:
                    snap = self.window_queue.get()
                except QueueClosed:
                    break
                if self._on_window is not None:
                    self._on_window(snap)
        except Exception as exc:  # noqa: BLE001
            self._errors.append(exc)
            self.stop()

    # -- run ---------------------------------------------------------------

    def run(self) -> dict:
        """Run until the source ends (or `max_frames`/`stop()`); returns a
        stats dict. Re-raises the first error from any stage."""
        threads = [
            threading.Thread(target=self._reader, name="anvesh-reader", daemon=True),
            threading.Thread(target=self._worker, name="anvesh-worker", daemon=True),
            threading.Thread(target=self._consumer, name="anvesh-consumer", daemon=True),
        ]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.wall_seconds = time.perf_counter() - t0
        if self._errors:
            raise self._errors[0]
        return self.stats()

    def stats(self) -> dict:
        eng = self._engine.stats()
        return {
            "engine": eng,
            "frame_queue": self.frame_queue.stats(),
            "window_queue": self.window_queue.stats(),
            "reader_frames": self.reader_frames,
            "wall_seconds": self.wall_seconds,
            "processing_fps": eng["frames_processed"] / self.wall_seconds if self.wall_seconds else 0.0,
            "source_health": dataclasses.asdict(self._source.health()) if hasattr(self._source, "health") else None,
        }

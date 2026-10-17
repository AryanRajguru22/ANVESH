"""Bounded queue between pipeline stages with an explicit, ACCOUNTED drop
policy. Nothing in the streaming pipeline is allowed to grow without bound:
when a producer outruns a consumer, either the producer waits (BLOCK --
lossless, for recorded files, so batch and stream see identical frames) or
the oldest item is discarded and COUNTED (DROP_OLDEST -- for live sources,
where a fresh frame is worth more than a stale one)."""

from __future__ import annotations

import collections
import enum
import threading


class DropPolicy(str, enum.Enum):
    BLOCK = "block"  # producer waits; never drops
    DROP_OLDEST = "drop_oldest"  # evict the oldest queued item, count it
    DROP_NEWEST = "drop_newest"  # refuse the incoming item, count it


class QueueClosed(Exception):
    pass


class BoundedQueue:
    def __init__(self, capacity: int, policy: DropPolicy = DropPolicy.BLOCK, name: str = "queue") -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.policy = DropPolicy(policy)
        self.name = name
        self._items: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self._closed = False
        self.put_count = 0
        self.dropped = 0
        self.high_water = 0

    def __len__(self) -> int:
        with self._cv:
            return len(self._items)

    def put(self, item, timeout: float = None) -> bool:
        """Enqueue. Returns True if `item` was enqueued, False if it was
        dropped (DROP_NEWEST) or the wait timed out (BLOCK). Under
        DROP_OLDEST the incoming item is always enqueued and the evicted one
        is returned via `dropped` accounting only."""
        with self._cv:
            if self._closed:
                raise QueueClosed(self.name)
            if self.policy == DropPolicy.BLOCK:
                ok = self._cv.wait_for(lambda: len(self._items) < self.capacity or self._closed, timeout)
                if self._closed:
                    raise QueueClosed(self.name)
                if not ok:
                    return False
            elif len(self._items) >= self.capacity:
                if self.policy == DropPolicy.DROP_OLDEST:
                    self._items.popleft()
                    self.dropped += 1
                else:
                    self.dropped += 1
                    return False
            self._items.append(item)
            self.put_count += 1
            self.high_water = max(self.high_water, len(self._items))
            self._cv.notify_all()
            return True

    def get(self, timeout: float = None):
        """Dequeue the oldest item; returns None on timeout; raises
        QueueClosed once the queue is closed AND drained."""
        with self._cv:
            ok = self._cv.wait_for(lambda: self._items or self._closed, timeout)
            if self._items:
                item = self._items.popleft()
                self._cv.notify_all()
                return item
            if self._closed:
                raise QueueClosed(self.name)
            return None

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def stats(self) -> dict:
        with self._cv:
            return {
                "name": self.name, "capacity": self.capacity, "policy": self.policy.value,
                "put": self.put_count, "dropped": self.dropped, "high_water": self.high_water,
                "depth": len(self._items),
            }

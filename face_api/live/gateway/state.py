from __future__ import annotations

import threading
import time
from typing import Callable


Clock = Callable[[], float]


class LiveRecognitionState:
    def __init__(
        self,
        default_threshold: float = 0.3,
        heartbeat_timeout_seconds: float = 10.0,
        clock: Clock | None = None,
    ) -> None:
        self._clock = clock or time.monotonic
        self.heartbeat_timeout_seconds = float(heartbeat_timeout_seconds)
        self.threshold = 0.3
        self.status = "resume"
        self.closed = False
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self._close_lock = threading.Lock()
        self.update_threshold(default_threshold)
        self.last_heartbeat_at = self._clock()

    @property
    def is_paused(self) -> bool:
        return self.status == "pause"

    def update_threshold(self, threshold: float) -> None:
        value = float(threshold)
        if value < 0.0 or value > 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = value

    def update_status(self, status: str) -> None:
        if status not in {"pause", "resume"}:
            raise ValueError("status must be pause or resume")
        self.status = status

    def mark_heartbeat(self) -> None:
        self.last_heartbeat_at = self._clock()

    def is_heartbeat_timed_out(self) -> bool:
        return self._clock() - self.last_heartbeat_at >= self.heartbeat_timeout_seconds

    def close(self) -> None:
        with self._close_lock:
            self.closed = True

    def request_close(self, *, code: int, reason: str) -> bool:
        with self._close_lock:
            if self.closed:
                return False
            self.closed = True
            self.close_code = int(code)
            self.close_reason = str(reason)
            return True

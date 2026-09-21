from __future__ import annotations
"""
实时视频最新帧缓存
不保存所有视频帧，只保存最新的一帧，让慢消费者永远处理最新画面，而不是处理过期画面。
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


Clock = Callable[[], float]

# 一帧视频的信息。
@dataclass(frozen=True)
class LatestFrame:
    seq: int
    frame: np.ndarray
    captured_at: float
    width: int
    height: int


class LatestFrameBuffer:
    """Thread-safe latest-value frame cache.

    This intentionally does not queue frames. Each update overwrites the previous
    frame so slow consumers always resume from the newest available frame.
    """

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._seq = 0
        self._latest: LatestFrame | None = None

    def update(self, frame: np.ndarray) -> LatestFrame:
        height, width = frame.shape[:2]
        with self._lock:
            self._seq += 1
            self._latest = LatestFrame(
                seq=self._seq,
                frame=frame,
                captured_at=float(self._clock()),
                width=int(width),
                height=int(height),
            )
            return self._latest

    def get_latest(self) -> LatestFrame | None:
        with self._lock:
            return self._latest

    def clear(self) -> None:
        with self._lock:
            self._latest = None

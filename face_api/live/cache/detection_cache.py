from __future__ import annotations
"""
    线程安全的人脸检测结果缓存器（DetectionCache）
    视频采集线程不断产生检测结果 → 保存最新一次检测 → 其他线程（API、推流、业务逻辑）随时读取最近有效结果。
"""
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


Clock = Callable[[], float]

# 人脸检测结果快照对象
@dataclass(frozen=True)
class DetectionSnapshot:
    frame_seq: int
    frame: np.ndarray
    detection: Any
    detected_at: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self.detection.box)

    @property
    def score(self) -> float:
        return float(getattr(self.detection, "score", 0.0))

# 线程安全的人脸检测结果缓存
# 保存最近一次有效的人脸检测结果，并允许其他线程安全读取；同时支持检测结果过期自动失效。
class DetectionCache:
    """Thread-safe cache for the most recent face detection snapshot."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.Lock() # 创建线程锁, 保证读取和写入新结果隔离
        self._latest: DetectionSnapshot | None = None
    # 更新最新检测结果
    def update(self, *, frame_seq: int, frame: np.ndarray, detection: Any) -> DetectionSnapshot:
        snapshot = DetectionSnapshot(
            frame_seq=int(frame_seq),
            frame=frame,
            detection=detection,
            detected_at=float(self._clock()),
        )
        with self._lock:
            self._latest = snapshot
        return snapshot

    # 获取最新检测结果
    def get_latest(self) -> DetectionSnapshot | None:
        with self._lock:
            return self._latest
    #  没过期的人脸检测结果。 只接受ttl_seconds秒内的人脸检测
    def get_valid(self, *, ttl_seconds: float) -> DetectionSnapshot | None:
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        if float(self._clock()) - latest.detected_at > float(ttl_seconds):
            return None
        return latest
    # 清空缓存。
    def clear(self) -> None:
        with self._lock:
            self._latest = None

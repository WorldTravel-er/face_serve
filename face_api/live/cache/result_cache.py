from __future__ import annotations
"""
    保存最近一次人脸识别结果，让预览线程（LivePreviewWorker）可以快速读取并显示，而不用每一帧重新执行识别。
"""
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


Clock = Callable[[], float]


@dataclass(frozen=True)
class CachedRecognitionResult:
    result: dict[str, Any]
    updated_at: float


class RecognitionResultCache:
    """Stores the latest recognition result for preview overlays."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._latest: CachedRecognitionResult | None = None

    def update(self, result: dict[str, Any] | None) -> None:
        with self._lock:
            if result is None:
                self._latest = None
                return
            self._latest = CachedRecognitionResult(result=result, updated_at=float(self._clock()))

    def get_valid(self, *, ttl_seconds: float) -> dict[str, Any] | None:
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        if float(self._clock()) - latest.updated_at > float(ttl_seconds):
            return None
        return latest.result

    def clear(self) -> None:
        with self._lock:
            self._latest = None

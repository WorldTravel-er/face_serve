from __future__ import annotations
# 实时系统性能监控模块（FPS Metrics）
# 在实时视频、人脸检测、人脸识别系统中，实时统计每个环节当前处理速度。
import threading
import time
from collections import deque
from collections.abc import Callable


Clock = Callable[[], float]

# 统计某一种事件流的实时 FPS。
class FpsMeter:
    """Sliding-window FPS meter for event streams."""

    def __init__(self, *, window_seconds: float = 2.0, clock: Clock | None = None) -> None:
        self.window_seconds = max(float(window_seconds), 0.1) # 只看最近window_seconds秒的性能
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._events: deque[float] = deque()# 根据这些时间计算fps
    # 记录一次事件。
    # 每发生一次事件（比如摄像头收到一帧、人脸检测完成一次），
    # 调用 mark()，它会记录当前时间，并删除太旧的记录。
    def mark(self) -> None:
        now = float(self._clock())
        with self._lock:
            self._events.append(now)
            self._prune_locked(now)
    # 计算时间fps
    def fps(self) -> float:
        now = float(self._clock())
        with self._lock:
            self._prune_locked(now)
            if len(self._events) < 2:
                return 0.0
            elapsed = self._events[-1] - self._events[0]
            if elapsed <= 1e-9:
                return 0.0
            return float((len(self._events) - 1) / elapsed)

    # 删除 FPS 统计窗口之外的旧事件时间，只保留最近 window_seconds 秒内的事件
    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()


class LiveMetrics:
    """Runtime FPS metrics shared by capture, detection, recognition and preview."""

    def __init__(self, *, window_seconds: float = 2.0, clock: Clock | None = None) -> None:
        self.stream = FpsMeter(window_seconds=window_seconds, clock=clock)
        self.detection = FpsMeter(window_seconds=window_seconds, clock=clock)
        self.recognition = FpsMeter(window_seconds=window_seconds, clock=clock)

    def mark_stream_frame(self) -> None:
        self.stream.mark()

    def mark_detection(self) -> None:
        self.detection.mark()

    def mark_recognition(self) -> None:
        self.recognition.mark()

    def snapshot(self) -> dict[str, float]:
        return {
            "stream_fps": self.stream.fps(),
            "detection_fps": self.detection.fps(),
            "recognition_fps": self.recognition.fps(),
        }

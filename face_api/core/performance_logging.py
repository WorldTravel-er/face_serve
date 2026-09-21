from __future__ import annotations

import contextlib
import contextvars
import json
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any


MonotonicClock = Callable[[], float]
DateTimeClock = Callable[[], datetime]

_CURRENT_LOGGER: contextvars.ContextVar["FacePerformanceLogger | None"] = contextvars.ContextVar(
    "face_performance_logger",
    default=None,
)
_CURRENT_FIELDS: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "face_performance_fields",
    default={},
)


class _FpsMeter:
    def __init__(self, *, window_seconds: float = 2.0, clock: MonotonicClock | None = None) -> None:
        self.window_seconds = max(float(window_seconds or 2.0), 0.1)
        self.clock = clock or time.monotonic
        self.events: deque[float] = deque()
        self.last_seen = 0.0

    def mark(self) -> float:
        now = float(self.clock())
        self.last_seen = now
        self.events.append(now)
        self._prune(now)
        if len(self.events) < 2:
            return 0.0
        elapsed = self.events[-1] - self.events[0]
        if elapsed <= 1e-9:
            return 0.0
        return float((len(self.events) - 1) / elapsed)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.events and self.events[0] < cutoff:
            self.events.popleft()


class FacePerformanceLogger:
    """JSONL logger for per-stage face inference latency and recognition FPS."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        stdout: bool = False,
        log_file: str | Path | None = None,
        fps_window_seconds: float = 2.0,
        max_fps_meters: int = 1024,
        monotonic_clock: MonotonicClock | None = None,
        datetime_clock: DateTimeClock | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.stdout = bool(stdout)
        self.log_file = Path(log_file) if log_file is not None else None
        self.fps_window_seconds = max(float(fps_window_seconds or 2.0), 0.1)
        self.max_fps_meters = max(int(max_fps_meters or 1024), 1)
        self.monotonic_clock = monotonic_clock or time.perf_counter
        self.datetime_clock = datetime_clock or datetime.now
        self._lock = threading.Lock()
        self._fps_meters: dict[str, _FpsMeter] = {}

    @classmethod
    def from_config(cls, config: Any) -> "FacePerformanceLogger":
        return cls(
            enabled=bool(getattr(config, "performance_log_enabled", False)),
            stdout=bool(getattr(config, "performance_log_stdout", False)),
            log_file=getattr(config, "performance_log_file", None),
            fps_window_seconds=float(getattr(config, "performance_log_fps_window_seconds", 2.0)),
            max_fps_meters=int(getattr(config, "performance_log_max_fps_meters", 1024)),
        )

    def is_enabled(self) -> bool:
        return self.enabled and (self.log_file is not None or self.stdout)

    @contextlib.contextmanager
    def context(self, **fields: Any) -> Iterator[None]:
        with performance_context(self, **fields):
            yield

    @contextlib.contextmanager
    def measure(self, stage: str, **fields: Any) -> Iterator[None]:
        with measure_face_stage(stage, logger=self, **fields):
            yield

    def log_stage(self, stage: str, elapsed_ms: float, fields: dict[str, Any]) -> None:
        if not self.is_enabled():
            return
        clean_stage = str(stage or "unknown")
        payload = {
            "event": "face_perf",
            "timestamp": self.datetime_clock().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "stage": clean_stage,
        }
        payload.update(_clean_fields(fields))
        payload[f"{clean_stage}_latency_ms"] = round(float(elapsed_ms), 3)
        if clean_stage == "recognition":
            payload["recognition_fps"] = round(self._mark_recognition_fps(payload), 3)
        self._write_json(payload)

    def _mark_recognition_fps(self, fields: dict[str, Any]) -> float:
        key = _fps_key(fields)
        with self._lock:
            meter = self._fps_meters.get(key)
            if meter is None:
                self._evict_fps_meter_if_needed()
                meter = _FpsMeter(window_seconds=self.fps_window_seconds, clock=self.monotonic_clock)
                self._fps_meters[key] = meter
            return meter.mark()

    def _evict_fps_meter_if_needed(self) -> None:
        if len(self._fps_meters) < self.max_fps_meters:
            return
        oldest_key = min(self._fps_meters, key=lambda key: self._fps_meters[key].last_seen)
        self._fps_meters.pop(oldest_key, None)

    def _write_json(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if self.stdout:
                try:
                    print(line, flush=True)
                except (OSError, ValueError):
                    self.stdout = False
            if self.log_file is not None:
                try:
                    self.log_file.parent.mkdir(parents=True, exist_ok=True)
                    with self.log_file.open("a", encoding="utf-8") as handle:
                        handle.write(line + "\n")
                except OSError:
                    self.log_file = None


@contextlib.contextmanager
def performance_context(logger: FacePerformanceLogger | None, **fields: Any) -> Iterator[None]:
    logger_token = _CURRENT_LOGGER.set(logger)
    previous = dict(_CURRENT_FIELDS.get() or {})
    previous.update(_clean_fields(fields))
    fields_token = _CURRENT_FIELDS.set(previous)
    try:
        yield
    finally:
        _CURRENT_FIELDS.reset(fields_token)
        _CURRENT_LOGGER.reset(logger_token)


@contextlib.contextmanager
def measure_face_stage(
    stage: str,
    *,
    logger: FacePerformanceLogger | None = None,
    **fields: Any,
) -> Iterator[None]:
    active_logger = logger if logger is not None else _CURRENT_LOGGER.get()
    if active_logger is None or not active_logger.is_enabled():
        yield
        return
    started_at = active_logger.monotonic_clock()
    try:
        yield
    finally:
        elapsed_ms = (active_logger.monotonic_clock() - started_at) * 1000.0
        merged = dict(_CURRENT_FIELDS.get() or {})
        merged.update(_clean_fields(fields))
        active_logger.log_stage(stage, elapsed_ms, merged)


def record_face_stage(
    stage: str,
    elapsed_ms: float,
    *,
    logger: FacePerformanceLogger | None = None,
    **fields: Any,
) -> None:
    active_logger = logger if logger is not None else _CURRENT_LOGGER.get()
    if active_logger is None or not active_logger.is_enabled():
        return
    merged = dict(_CURRENT_FIELDS.get() or {})
    merged.update(_clean_fields(fields))
    active_logger.log_stage(stage, elapsed_ms, merged)


def current_performance_logger() -> FacePerformanceLogger | None:
    return _CURRENT_LOGGER.get()


def _clean_fields(fields: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[str(key)] = value
        elif isinstance(value, Path):
            clean[str(key)] = str(value)
        else:
            clean[str(key)] = str(value)
    return clean


def _fps_key(fields: dict[str, Any]) -> str:
    channel = str(fields.get("channel", "unknown"))
    if fields.get("task_name"):
        return f"{channel}:task:{fields['task_name']}"
    if fields.get("connection_id"):
        return f"{channel}:connection:{fields['connection_id']}"
    return f"{channel}:process"

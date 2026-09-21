from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator


PerfCounter = Callable[[], float]
MonotonicClock = Callable[[], float]
DateTimeClock = Callable[[], datetime]


class LatencyTrace:
    """Collects model-stage timings for one inference operation."""

    def __init__(self, *, perf_counter: PerfCounter | None = None) -> None:
        self.perf_counter = perf_counter or time.perf_counter
        self.started_at = float(self.perf_counter())
        self._stages: dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started_at = float(self.perf_counter())
        try:
            yield
        finally:
            elapsed_ms = (float(self.perf_counter()) - started_at) * 1000.0
            self.add_ms(stage, elapsed_ms)

    def add_ms(self, stage: str, elapsed_ms: float | int | None) -> None:
        if elapsed_ms is None:
            return
        try:
            value = float(elapsed_ms)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        self._stages[str(stage)] = max(value, 0.0)

    def snapshot(self) -> dict[str, float]:
        data = dict(self._stages)
        data["total_ms"] = max((float(self.perf_counter()) - self.started_at) * 1000.0, 0.0)
        return data


class InferenceLatencyLogger:
    """Best-effort JSONL logger for inference latency events."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        log_file: str | Path | None = None,
        stdout: bool = True,
        throttle_seconds: float = 0.0,
        monotonic_clock: MonotonicClock | None = None,
        datetime_clock: DateTimeClock | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_file = Path(log_file) if log_file is not None else None
        self.stdout = bool(stdout)
        self.throttle_seconds = max(float(throttle_seconds or 0.0), 0.0)
        self.monotonic_clock = monotonic_clock or time.monotonic
        self.datetime_clock = datetime_clock or datetime.now
        self._last_written_at: dict[str, float] = {}
        self._file_enabled = self.log_file is not None
        self._stdout_enabled = self.stdout

    @classmethod
    def from_config(cls, config: Any) -> "InferenceLatencyLogger":
        return cls(
            enabled=bool(getattr(config, "latency_logging_enabled", True)),
            log_file=getattr(config, "latency_log_file", Path("logs/inference_latency.log")),
            stdout=bool(getattr(config, "latency_log_stdout", True)),
            throttle_seconds=float(getattr(config, "latency_log_throttle_seconds", 0.0)),
        )

    def emit(self, event: dict[str, Any], *, event_key: str | None = None) -> None:
        if not self.enabled:
            return
        key = event_key or _default_event_key(event)
        if not self._should_write(key):
            return
        payload = self._normalize_event(event)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if self._file_enabled and self.log_file is not None:
            self._write_file(line)
        if self._stdout_enabled:
            self._write_stdout(line)

    def _should_write(self, event_key: str) -> bool:
        if self.throttle_seconds <= 0.0:
            return True
        now = float(self.monotonic_clock())
        last = self._last_written_at.get(event_key)
        if last is not None and now - last < self.throttle_seconds:
            return False
        self._last_written_at[event_key] = now
        return True

    def _normalize_event(self, event: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ts": self.datetime_clock().isoformat(timespec="milliseconds"),
        }
        for key, value in event.items():
            normalized = _json_safe(value)
            if key.endswith("_ms") and isinstance(normalized, (int, float)) and not isinstance(normalized, bool):
                normalized = round(float(normalized), 2)
            payload[str(key)] = normalized
        return payload

    def _write_file(self, line: str) -> None:
        try:
            assert self.log_file is not None
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
        except Exception:
            self._file_enabled = False

    def _write_stdout(self, line: str) -> None:
        try:
            print(line, flush=True)
        except Exception:
            self._stdout_enabled = False


def _default_event_key(event: dict[str, Any]) -> str:
    source = str(event.get("source", "unknown"))
    operation = str(event.get("operation", "unknown"))
    return f"{source}:{operation}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(timespec="milliseconds")
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)

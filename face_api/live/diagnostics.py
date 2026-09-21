from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


MonotonicClock = Callable[[], float]
DateTimeClock = Callable[[], datetime]


class LiveDiagnosticsLogger:
    """Optional file-based diagnostics logger for live detection and recognition."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        log_file: str | Path | None = None,
        throttle_seconds: float = 1.0,
        monotonic_clock: MonotonicClock | None = None,
        datetime_clock: DateTimeClock | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_file = Path(log_file) if log_file is not None else None
        self.throttle_seconds = max(float(throttle_seconds or 0.0), 0.0)
        self.monotonic_clock = monotonic_clock or time.monotonic
        self.datetime_clock = datetime_clock or datetime.now
        self._last_written_at: dict[str, float] = {}

    @classmethod
    def from_config(cls, config: Any) -> "LiveDiagnosticsLogger":
        return cls(
            enabled=bool(getattr(config, "live_diagnostics_enabled", False)),
            log_file=getattr(config, "live_diagnostics_log_file", None),
            throttle_seconds=float(getattr(config, "live_diagnostics_throttle_seconds", 1.0)),
        )

    def log_websocket_connected(self, *, connection_id: str, elapsed_ms: float) -> None:
        if not self._can_write():
            return
        self._write_line(
            "[live-websocket] connected "
            f"connection_id={str(connection_id or 'unknown')} "
            f"elapsed_ms={float(elapsed_ms):.2f}"
        )

    def log_first_detection_result(
        self,
        *,
        connection_id: str,
        elapsed_ms: float,
        detection_elapsed_ms: float,
        frame_seq: int,
        detection: Any,
    ) -> None:
        if not self._can_write():
            return
        box = _format_box(_extract_box(detection))
        score = _extract_score(detection)
        fields = [
            "[live-first-detection]",
            f"connection_id={str(connection_id or 'unknown')}",
            f"elapsed_ms={float(elapsed_ms):.2f}",
            f"detection_elapsed_ms={float(detection_elapsed_ms):.2f}",
            f"frame_seq={int(frame_seq)}",
            f"box={box}",
        ]
        if score is not None:
            fields.append(f"score={float(score):.3f}")
        keypoints = _extract_keypoints(detection)
        if keypoints is not None:
            fields.append(f"keypoints={keypoints}")
        self._write_line(" ".join(fields))


    def log_runner_stage(self, *, connection_id: str, stage: str, elapsed_ms: float | None = None) -> None:
        if not self._can_write():
            return
        fields = [
            "[live-runner]",
            f"stage={str(stage or 'unknown')}",
            f"connection_id={str(connection_id or 'unknown')}",
        ]
        if elapsed_ms is not None:
            fields.append(f"elapsed_ms={float(elapsed_ms):.2f}")
        self._write_line(" ".join(fields))

    def log_capture_stage(
        self,
        *,
        connection_id: str,
        stage: str,
        stream_url: str | None = None,
        elapsed_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        if not self._can_write():
            return
        fields = [
            "[live-capture]",
            f"stage={str(stage or 'unknown')}",
            f"connection_id={str(connection_id or 'unknown')}",
        ]
        if stream_url is not None:
            fields.append(f"stream_url={stream_url}")
        if elapsed_ms is not None:
            fields.append(f"elapsed_ms={float(elapsed_ms):.2f}")
        if error:
            fields.append(f"error={_sanitize_log_value(error)}")
        self._write_line(" ".join(fields))

    def log_capture_first_frame(
        self,
        *,
        connection_id: str,
        elapsed_ms: float,
        open_elapsed_ms: float | None,
        frame_seq: int,
        width: int,
        height: int,
    ) -> None:
        if not self._can_write():
            return
        fields = [
            "[live-capture]",
            "stage=first_frame",
            f"connection_id={str(connection_id or 'unknown')}",
            f"elapsed_ms={float(elapsed_ms):.2f}",
        ]
        if open_elapsed_ms is not None:
            fields.append(f"open_elapsed_ms={float(open_elapsed_ms):.2f}")
        fields.extend([
            f"frame_seq={int(frame_seq)}",
            f"width={int(width)}",
            f"height={int(height)}",
        ])
        self._write_line(" ".join(fields))

    def log_detection_stage(self, *, connection_id: str, stage: str, elapsed_ms: float | None = None, frame_seq: int | None = None) -> None:
        if not self._can_write():
            return
        fields = [
            "[live-detection-worker]",
            f"stage={str(stage or 'unknown')}",
            f"connection_id={str(connection_id or 'unknown')}",
        ]
        if elapsed_ms is not None:
            fields.append(f"elapsed_ms={float(elapsed_ms):.2f}")
        if frame_seq is not None:
            fields.append(f"frame_seq={int(frame_seq)}")
        self._write_line(" ".join(fields))

    def log_detection(self, *, frame_seq: int, detection: Any, elapsed_ms: float | None = None) -> None:
        if not self._should_write("detection"):
            return
        box = _format_box(_extract_box(detection))
        score = _extract_score(detection)
        fields = [
            "[live-detection]",
            f"frame_seq={int(frame_seq)}",
        ]
        if elapsed_ms is not None:
            fields.append(f"elapsed_ms={float(elapsed_ms):.2f}")
        fields.append(f"box={box}")
        if score is not None:
            fields.append(f"score={float(score):.3f}")
        keypoints = _extract_keypoints(detection)
        if keypoints is not None:
            fields.append(f"keypoints={keypoints}")
        self._write_line(" ".join(fields))

    def log_recognition(self, live_result: dict[str, Any] | None, *, candidates_count: int, threshold: float, elapsed_ms: float | None = None) -> None:
        if not self._should_write("recognition"):
            return
        result = live_result or {}
        box = _format_box(result.get("box"))
        matches = result.get("matches") or []
        if matches:
            best = matches[0]
            elapsed_text = _format_elapsed(elapsed_ms)
            line = (
                "[live-recognition] matched "
                f"{elapsed_text}"
                f"subject_id={best.get('subject_id')} "
                f"name={best.get('name')} "
                f"similarity={float(best.get('similarity', 0.0)):.3f} "
                f"threshold={float(threshold):.3f} "
                f"candidates={int(candidates_count)} "
                f"box={box}"
            )
        else:
            elapsed_text = _format_elapsed(elapsed_ms)
            line = (
                "[live-recognition] no_match "
                f"{elapsed_text}"
                f"candidates={int(candidates_count)} "
                f"threshold={float(threshold):.3f} "
                f"box={box}"
            )
        self._write_line(line)

    def _can_write(self) -> bool:
        return self.enabled and self.log_file is not None

    def _should_write(self, event_key: str) -> bool:
        if not self._can_write():
            return False
        now = self.monotonic_clock()
        last = self._last_written_at.get(event_key)
        if last is not None and now - last < self.throttle_seconds:
            return False
        self._last_written_at[event_key] = now
        return True

    def _write_line(self, message: str) -> None:
        if self.log_file is None:
            return
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            timestamp = self.datetime_clock().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"{timestamp} {message}\n")
        except Exception:
            self.enabled = False


def _format_elapsed(elapsed_ms: float | None) -> str:
    if elapsed_ms is None:
        return ""
    return f"elapsed_ms={float(elapsed_ms):.2f} "


def _extract_box(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("box") or value.get("bbox")
    return getattr(value, "box", getattr(value, "bbox", None))


def _extract_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get("score", value.get("confidence", value.get("conf")))
    else:
        raw = getattr(value, "score", getattr(value, "confidence", getattr(value, "conf", None)))
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_keypoints(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get("keypoints") or value.get("kps")
    else:
        raw = getattr(value, "keypoints", getattr(value, "kps", None))
    if raw is None:
        return None
    return str(raw)


def _format_box(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return "[" + ",".join(str(int(item)) for item in value) + "]"
    except TypeError:
        return str(value)

def _sanitize_log_value(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ")

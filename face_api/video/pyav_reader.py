from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import av


@dataclass(frozen=True)
class DecodedVideoFrame:
    image: Any
    source_index: int
    timestamp_seconds: float
    key_frame: bool


@dataclass
class VideoDecodeStats:
    packets_seen: int = 0
    corrupt_packets: int = 0
    decode_errors: int = 0
    decoded_frames: int = 0
    corrupt_frames: int = 0
    usable_frames: int = 0


class PyAVVideoReader:
    def __init__(self, source: str | Path, *, max_consecutive_errors: int = 100) -> None:
        self.source = str(source)
        self.max_consecutive_errors = max(int(max_consecutive_errors), 1)
        self.stats = VideoDecodeStats()
        self.container: Any = None
        self.stream: Any = None

    def __enter__(self) -> "PyAVVideoReader":
        # PyAV still raises decode exceptions at this level, but repeated
        # recoverable FFmpeg diagnostics no longer flood service stderr.
        av.logging.set_level(av.logging.FATAL)
        try:
            self.container = av.open(self.source, mode="r")
        except av.FFmpegError as exc:
            raise RuntimeError(f"Cannot open video source: {exc}") from exc

        self.stream = next(
            (stream for stream in self.container.streams if stream.type == "video"),
            None,
        )
        if self.stream is None:
            self.container.close()
            self.container = None
            raise RuntimeError("Video source contains no video stream")

        options = dict(getattr(self.stream.codec_context, "options", {}) or {})
        options["err_detect"] = "explode"
        self.stream.codec_context.options = options
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.container is not None:
            self.container.close()
            self.container = None

    @property
    def fps(self) -> float:
        rate = getattr(self.stream, "average_rate", None)
        try:
            value = float(rate) if rate is not None else 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            value = 0.0
        return value if math.isfinite(value) and value > 0.0 else 25.0

    @property
    def total_frames(self) -> int:
        return max(int(getattr(self.stream, "frames", 0) or 0), 0)

    @property
    def duration_seconds(self) -> float:
        duration = getattr(self.stream, "duration", None)
        time_base = getattr(self.stream, "time_base", None)
        if duration is not None and time_base is not None:
            value = float(duration * time_base)
            if math.isfinite(value) and value > 0.0:
                return value
        total_frames = self.total_frames
        return total_frames / self.fps if total_frames > 0 else 0.0

    def frames(self) -> Iterator[DecodedVideoFrame]:
        source_index = 0
        consecutive_errors = 0

        for packet in self.container.demux(self.stream):
            self.stats.packets_seen += 1
            if bool(getattr(packet, "is_corrupt", False)):
                self.stats.corrupt_packets += 1
                consecutive_errors = self._record_error(consecutive_errors)
                continue

            try:
                decoded_frames = packet.decode()
            except av.FFmpegError:
                self.stats.decode_errors += 1
                consecutive_errors = self._record_error(consecutive_errors)
                continue

            for frame in decoded_frames:
                current_index = source_index
                source_index += 1
                self.stats.decoded_frames += 1

                if bool(getattr(frame, "is_corrupt", False)):
                    self.stats.corrupt_frames += 1
                    consecutive_errors = self._record_error(consecutive_errors)
                    continue

                try:
                    image = frame.to_ndarray(format="bgr24")
                except (av.FFmpegError, ValueError):
                    self.stats.decode_errors += 1
                    consecutive_errors = self._record_error(consecutive_errors)
                    continue
                if image is None or getattr(image, "size", 0) <= 0:
                    self.stats.corrupt_frames += 1
                    consecutive_errors = self._record_error(consecutive_errors)
                    continue

                consecutive_errors = 0
                self.stats.usable_frames += 1
                yield DecodedVideoFrame(
                    image=image,
                    source_index=current_index,
                    timestamp_seconds=self._timestamp_seconds(frame, current_index),
                    key_frame=bool(getattr(frame, "key_frame", False)),
                )

    def _record_error(self, consecutive_errors: int) -> int:
        consecutive_errors += 1
        if consecutive_errors > self.max_consecutive_errors:
            raise RuntimeError(
                f"Too many consecutive video decode errors: {consecutive_errors}"
            )
        return consecutive_errors

    def _timestamp_seconds(self, frame: Any, source_index: int) -> float:
        timestamp = getattr(frame, "time", None)
        if timestamp is None:
            pts = getattr(frame, "pts", None)
            time_base = getattr(frame, "time_base", None)
            if pts is not None and time_base is not None:
                timestamp = pts * time_base
        try:
            value = float(timestamp) if timestamp is not None else source_index / self.fps
        except (TypeError, ValueError, ZeroDivisionError):
            value = source_index / self.fps
        return max(value, 0.0) if math.isfinite(value) else source_index / self.fps

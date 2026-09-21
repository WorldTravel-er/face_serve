from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import cv2

from face_api.live.capture.frame_buffer import LatestFrameBuffer


FrameReaderFactory = Callable[[str], Any]
ErrorCallback = Callable[[str], None]

STREAM_OFFLINE_MESSAGE = "视频源打开或读取失败，摄像头离线"


class OpenCvFrameReader:
    def __init__(self, stream_url: str, *, backend: Any | None = None) -> None:
        self.stream_url = stream_url
        self.backend = backend or cv2
        self._capture: Any | None = None

    def open(self) -> None:
        if _is_v4l2_video_device(self.stream_url):
            self._capture = self.backend.VideoCapture(self.stream_url, self.backend.CAP_V4L2)
        else:
            if isinstance(self.stream_url, str) and self.stream_url.isdigit():
                self.stream_url = int(self.stream_url)
            self._capture = self.backend.VideoCapture(self.stream_url)
        try:
            self._capture.set(self.backend.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self._capture.isOpened():
            self.close()
            raise RuntimeError(STREAM_OFFLINE_MESSAGE)

    def read(self):
        if self._capture is None:
            return False, None
        return self._capture.read()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def _is_v4l2_video_device(source: str) -> bool:
    return str(source or "").startswith("/dev/video")


class CaptureWorker:
    """Continuously consumes a video stream and keeps only the newest frame."""

    def __init__(
        self,
        *,
        stream_url: str,
        frame_buffer: LatestFrameBuffer,
        frame_reader_factory: FrameReaderFactory | None = None,
        error_callback: ErrorCallback | None = None,
        metrics: Any | None = None,
        reconnect_enabled: bool = True,
        reconnect_interval: float = 3.0,
        max_consecutive_failures: int = 10,
        failure_sleep_seconds: float = 0.01,
        diagnostics: Any | None = None,
        perf_counter: Any | None = None,
        connection_id: str | None = None,
        websocket_accepted_at: float | None = None,
    ) -> None:
        self.stream_url = stream_url
        self.frame_buffer = frame_buffer
        self.frame_reader_factory = frame_reader_factory or OpenCvFrameReader
        self.error_callback = error_callback
        self.metrics = metrics
        self.reconnect_enabled = bool(reconnect_enabled)
        self.reconnect_interval = max(float(reconnect_interval), 0.0)
        self.max_consecutive_failures = max(int(max_consecutive_failures), 1)
        self.failure_sleep_seconds = max(float(failure_sleep_seconds), 0.0)
        self.diagnostics = diagnostics
        self.perf_counter = perf_counter or time.perf_counter
        self.connection_id = connection_id
        self.websocket_accepted_at = websocket_accepted_at
        self._first_frame_logged = False
        self._last_open_finished_at: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="live-capture-worker", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)

    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def _run(self) -> None:
        while not self._stop_event.is_set():
            reader = self.frame_reader_factory(self.stream_url)
            open_started_at = self.perf_counter()
            self._log_capture_stage("open_start", stream_url=str(self.stream_url))
            try:
                reader.open()
            except Exception as exc:
                open_failed_at = self.perf_counter()
                self._log_capture_stage(
                    "open_failed",
                    stream_url=str(self.stream_url),
                    elapsed_ms=(open_failed_at - open_started_at) * 1000.0,
                    error=str(exc) or STREAM_OFFLINE_MESSAGE,
                )
                self._report_error(str(exc) or STREAM_OFFLINE_MESSAGE)
                if not self.reconnect_enabled:
                    self._close_reader(reader)
                    return
                self._log_capture_stage("reconnect_sleep", stream_url=str(self.stream_url), elapsed_ms=self.reconnect_interval * 1000.0)
                self._sleep_until_reconnect()
                continue

            open_finished_at = self.perf_counter()
            self._last_open_finished_at = open_finished_at
            self._log_capture_stage(
                "open_success",
                stream_url=str(self.stream_url),
                elapsed_ms=(open_finished_at - open_started_at) * 1000.0,
            )

            failures = 0
            try:
                while not self._stop_event.is_set():
                    ok, frame = reader.read()
                    if ok and frame is not None:
                        failures = 0
                        latest = self.frame_buffer.update(frame)
                        self._log_first_frame(latest)
                        self._mark_stream_frame()
                        continue

                    failures += 1
                    if failures >= self.max_consecutive_failures:
                        self._log_capture_stage("read_failed", stream_url=str(self.stream_url), elapsed_ms=None)
                        self._report_error(STREAM_OFFLINE_MESSAGE)
                        break
                    if self.failure_sleep_seconds:
                        time.sleep(self.failure_sleep_seconds)
            finally:
                self._close_reader(reader)

            if not self.reconnect_enabled:
                return
            self._log_capture_stage("reconnect_sleep", stream_url=str(self.stream_url), elapsed_ms=self.reconnect_interval * 1000.0)
            self._sleep_until_reconnect()

    def _log_capture_stage(
        self,
        stage: str,
        *,
        stream_url: str | None = None,
        elapsed_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        logger = getattr(self.diagnostics, "log_capture_stage", None)
        if callable(logger):
            logger(
                connection_id=str(self.connection_id or "unknown"),
                stage=stage,
                stream_url=stream_url,
                elapsed_ms=elapsed_ms,
                error=error,
            )

    def _log_first_frame(self, latest: Any) -> None:
        if self._first_frame_logged:
            return
        self._first_frame_logged = True
        if self.websocket_accepted_at is None:
            return
        now = self.perf_counter()
        open_elapsed_ms = None
        if self._last_open_finished_at is not None:
            open_elapsed_ms = (now - self._last_open_finished_at) * 1000.0
        logger = getattr(self.diagnostics, "log_capture_first_frame", None)
        if callable(logger):
            logger(
                connection_id=str(self.connection_id or "unknown"),
                elapsed_ms=(now - self.websocket_accepted_at) * 1000.0,
                open_elapsed_ms=open_elapsed_ms,
                frame_seq=int(getattr(latest, "seq", 0)),
                width=int(getattr(latest, "width", 0)),
                height=int(getattr(latest, "height", 0)),
            )

    def _sleep_until_reconnect(self) -> None:
        if self.reconnect_interval <= 0:
            return
        self._stop_event.wait(self.reconnect_interval)

    def _report_error(self, message: str) -> None:
        if self.error_callback is not None:
            self.error_callback(message or STREAM_OFFLINE_MESSAGE)

    def _mark_stream_frame(self) -> None:
        marker = getattr(self.metrics, "mark_stream_frame", None)
        if callable(marker):
            marker()

    @staticmethod
    def _close_reader(reader: Any) -> None:
        try:
            reader.close()
        except Exception:
            pass
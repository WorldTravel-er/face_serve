from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from typing import Any

from face_api.live.cache.detection_cache import DetectionCache
from face_api.live.capture.frame_buffer import LatestFrameBuffer


class DetectionWorker:
    """Runs face detection at a higher cadence than identity recognition."""

    def __init__(
        self,
        *,
        frame_buffer: LatestFrameBuffer,
        detection_cache: DetectionCache,
        recognition_service: Any,
        state: Any,
        fps: float = 12.0,
        error_callback: Any | None = None,
        metrics: Any | None = None,
        diagnostics: Any | None = None,
        perf_counter: Any | None = None,
        connection_id: str | None = None,
        websocket_accepted_at: float | None = None,
    ) -> None:
        self.frame_buffer = frame_buffer
        self.detection_cache = detection_cache
        self.recognition_service = recognition_service
        self.state = state
        self.fps = max(float(fps or 12.0), 0.1)
        self.error_callback = error_callback
        self.metrics = metrics
        self.diagnostics = diagnostics
        self.perf_counter = perf_counter or time.perf_counter
        self.connection_id = connection_id
        self.websocket_accepted_at = websocket_accepted_at
        self._first_detection_logged = False
        self._first_frame_seen_logged = False
        self._first_detect_start_logged = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_detected_seq = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="live-detection-worker", daemon=True)
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
        period = 1.0 / self.fps
        while not self._stop_event.is_set():
            if self.state.is_paused:
                self._stop_event.wait(period)
                continue

            latest = self.frame_buffer.get_latest()
            if latest is None or latest.seq == self._last_detected_seq:
                self._stop_event.wait(period)
                continue
            self._last_detected_seq = latest.seq
            if self._should_log_detection_stage("first_frame_seen"):
                self._log_detection_stage("first_frame_seen", latest.seq, event_at=self.perf_counter())

            try:
                started_at = self.perf_counter()
                if self._should_log_detection_stage("first_detect_start"):
                    self._log_detection_stage("first_detect_start", latest.seq, event_at=started_at)
                context = getattr(self.recognition_service, "performance_context", None)
                manager = (
                    context(
                        channel="websocket",
                        connection_id=str(self.connection_id or "unknown"),
                        frame_seq=latest.seq,
                    )
                    if callable(context)
                    else nullcontext()
                )
                with manager:
                    detection = self.recognition_service.detect_frame(latest.frame)
                finished_at = self.perf_counter()
                elapsed_ms = (finished_at - started_at) * 1000.0
                self._log_first_detection(
                    latest.seq,
                    detection,
                    detected_at=finished_at,
                    detection_elapsed_ms=elapsed_ms,
                )
                self._mark_detection()
                if detection is None:
                    self.detection_cache.clear()
                else:
                    self._log_detection(latest.seq, detection, elapsed_ms=elapsed_ms)
                    self.detection_cache.update(frame_seq=latest.seq, frame=latest.frame, detection=detection)
            except Exception as exc:
                self.detection_cache.clear()
                if self.error_callback is not None:
                    self.error_callback(str(exc) or "live face detection service error")

            self._stop_event.wait(period)

    def _log_first_detection(
        self,
        frame_seq: int,
        detection: Any,
        *,
        detected_at: float,
        detection_elapsed_ms: float,
    ) -> None:
        if self._first_detection_logged:
            return
        self._first_detection_logged = True
        if self.websocket_accepted_at is None:
            return
        logger = getattr(self.diagnostics, "log_first_detection_result", None)
        if callable(logger):
            logger(
                connection_id=str(self.connection_id or "unknown"),
                elapsed_ms=(detected_at - self.websocket_accepted_at) * 1000.0,
                detection_elapsed_ms=detection_elapsed_ms,
                frame_seq=frame_seq,
                detection=detection,
            )

    def _should_log_detection_stage(self, stage: str) -> bool:
        if self.websocket_accepted_at is None:
            return False
        logger = getattr(self.diagnostics, "log_detection_stage", None)
        if not callable(logger):
            return False
        if stage == "first_frame_seen":
            return not self._first_frame_seen_logged
        if stage == "first_detect_start":
            return not self._first_detect_start_logged
        return True

    def _log_detection_stage(self, stage: str, frame_seq: int, *, event_at: float) -> None:
        if stage == "first_frame_seen":
            if self._first_frame_seen_logged:
                return
            self._first_frame_seen_logged = True
        elif stage == "first_detect_start":
            if self._first_detect_start_logged:
                return
            self._first_detect_start_logged = True
        if self.websocket_accepted_at is None:
            return
        logger = getattr(self.diagnostics, "log_detection_stage", None)
        if callable(logger):
            logger(
                connection_id=str(self.connection_id or "unknown"),
                stage=stage,
                elapsed_ms=(event_at - self.websocket_accepted_at) * 1000.0,
                frame_seq=frame_seq,
            )

    def _log_detection(self, frame_seq: int, detection: Any, *, elapsed_ms: float | None = None) -> None:
        logger = getattr(self.diagnostics, "log_detection", None)
        if callable(logger):
            logger(frame_seq=frame_seq, detection=detection, elapsed_ms=elapsed_ms)

    def _mark_detection(self) -> None:
        marker = getattr(self.metrics, "mark_detection", None)
        if callable(marker):
            marker()

"""RKNN detector adapter with third-party trackers face ID association."""

from __future__ import annotations

from typing import Any

import numpy as np

from face_api.core.performance_logging import measure_face_stage
from face_core.face_recognition_engine.face_detection.base import FaceDetection
from face_core.face_recognition_engine.face_detection.trackers_adapter import TrackersFaceAssociator
from face_core.face_recognition_engine.face_detection.yolo_face import YoloTrackResult, _choose_nearest


class RknnByteTrackFaceTracker:
    """Track the nearest face using detections from the configured NPU adapter.

    The detector owns the RKNN model session.  This adapter contains no model
    loader or heavyweight neural-network framework dependency.
    """

    def __init__(
        self,
        detector: Any,
        track_buffer: int = 30,
        match_thresh: float = 0.3,
    ) -> None:
        if not callable(getattr(detector, "detect_all", None)):
            raise TypeError("RKNN video tracker requires a detector with detect_all(frame)")
        self.detector = detector
        self.tracker = TrackersFaceAssociator(
            track_activation_threshold=float(getattr(detector, "conf", 0.25)),
            lost_track_buffer=track_buffer,
            minimum_iou_threshold=match_thresh,
        )
        self.last_box: tuple[int, int, int, int] | None = None
        self.lost_frames = 0

    def track_nearest(self, frame: np.ndarray) -> YoloTrackResult | None:
        with measure_face_stage("detect"):
            detections = self.detector.detect_all(frame)
        tracked = self.tracker.update(detections)
        if not tracked:
            self.lost_frames += 1
            return None

        # _choose_nearest adjusts ``score`` for target selection.  Copy each
        # detection so that its tracking confidence remains intact in state.
        candidates = [
            (FaceDetection(tuple(detection.box), float(detection.score), detection.index), track_id)
            for detection, track_id in tracked
        ]
        height, width = _frame_height_width(frame, candidates)
        detection = _choose_nearest(
            [candidate for candidate, _ in candidates],
            width,
            height,
            self.last_box,
        )
        if detection is None:  # defensive: non-empty candidates always select.
            self.lost_frames += 1
            return None
        track_id = next(track_id for candidate, track_id in candidates if candidate is detection)
        self.last_box = detection.box
        self.lost_frames = 0
        return YoloTrackResult(
            detection=detection,
            track_id=track_id,
            source="rknn_bytetrack",
            lost_frames=0,
        )


def _frame_height_width(
    frame: Any,
    candidates: list[tuple[FaceDetection, int]],
) -> tuple[int, int]:
    """Use normal image dimensions, with a minimal fallback for test fakes."""
    shape = getattr(frame, "shape", ())
    if len(shape) >= 2 and int(shape[0]) > 0 and int(shape[1]) > 0:
        return int(shape[0]), int(shape[1])
    right = max(detection.box[0] + detection.box[2] for detection, _ in candidates)
    bottom = max(detection.box[1] + detection.box[3] for detection, _ in candidates)
    return max(bottom, 1), max(right, 1)

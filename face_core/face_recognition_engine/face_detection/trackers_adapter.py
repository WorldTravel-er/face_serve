from __future__ import annotations
# FaceDetection 列表
#       ↓
# 过滤无效框
#       ↓
# 转换为 supervision.Detections
#       ↓
# ByteTrackTracker.update()
#       ↓
# 获得 tracker_id
#       ↓
# 重新转换成 FaceDetection
#       ↓
# [(FaceDetection, track_id), ...]
from typing import Any, Iterable

import numpy as np

from .base import FaceDetection


class TrackersFaceAssociator:
    """Adapter from project face detections to Roboflow trackers."""

    def __init__(
        self,
        *,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_consecutive_frames: int = 1,
        minimum_iou_threshold: float = 0.3,
        frame_rate: int = 30,
    ) -> None:
        try:
            from trackers import ByteTrackTracker
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "The 'trackers' package is required for ONNX/RKNN face tracking. "
                "Install it with `pip install trackers`."
            ) from exc

        self._sv = sv
        self._tracker = ByteTrackTracker(
            track_activation_threshold=float(track_activation_threshold),
            lost_track_buffer=int(lost_track_buffer),
            minimum_consecutive_frames=int(minimum_consecutive_frames),
            minimum_iou_threshold=float(minimum_iou_threshold),
            frame_rate=int(frame_rate),
        )

    def update(self, detections: Iterable[FaceDetection]) -> list[tuple[FaceDetection, int]]:
        original = [detection for detection in detections if _is_valid_detection(detection)]
        if not original:
            self._tracker.update(self._sv.Detections.empty())
            return []

        tracked = self._tracker.update(_to_sv_detections(self._sv, original))
        tracker_ids = getattr(tracked, "tracker_id", None)
        if tracker_ids is None:
            return []
        return _from_sv_detections(tracked, tracker_ids)


def _is_valid_detection(detection: FaceDetection) -> bool:
    x, y, w, h = detection.box
    values = np.asarray([x, y, w, h, detection.score], dtype=np.float64)
    return bool(np.all(np.isfinite(values)) and w > 0 and h > 0)


def _to_sv_detections(sv: Any, detections: list[FaceDetection]) -> Any:
    xyxy = np.asarray(
        [[x, y, x + w, y + h] for x, y, w, h in (detection.box for detection in detections)],
        dtype=np.float32,
    )
    confidence = np.asarray([detection.score for detection in detections], dtype=np.float32)
    class_id = np.zeros(len(detections), dtype=np.int32)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def _from_sv_detections(tracked: Any, tracker_ids: Any) -> list[tuple[FaceDetection, int]]:
    xyxy = np.asarray(tracked.xyxy, dtype=np.float32)
    confidence_value = getattr(tracked, "confidence", None)
    confidence = (
        np.ones(len(xyxy), dtype=np.float32)
        if confidence_value is None
        else np.asarray(confidence_value, dtype=np.float32)
    )
    ids = np.asarray(tracker_ids)
    results: list[tuple[FaceDetection, int]] = []
    for index, raw_track_id in enumerate(ids):
        track_id = _safe_track_id(raw_track_id)
        if track_id is None:
            continue
        x1, y1, x2, y2 = [float(value) for value in xyxy[index]]
        if not np.all(np.isfinite([x1, y1, x2, y2])):
            continue
        width = max(1, int(np.ceil(x2 - x1)))
        height = max(1, int(np.ceil(y2 - y1)))
        if width <= 0 or height <= 0:
            continue
        score = float(confidence[index]) if index < len(confidence) else 0.0
        detection = FaceDetection((int(np.floor(x1)), int(np.floor(y1)), width, height), score, None)
        results.append((detection, track_id))
    return results


def _safe_track_id(value: Any) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(numeric) or numeric < 0:
        return None
    return int(numeric)

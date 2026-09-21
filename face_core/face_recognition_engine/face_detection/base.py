from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class FaceDetection:
    """A detected face whose box uses OpenCV's ``(x, y, width, height)`` form."""

    box: tuple[int, int, int, int]
    score: float
    index: int | None = None
    landmarks: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass
class FaceTrackResult:
    """The selected face and its optional temporal identity for one frame."""

    detection: FaceDetection
    track_id: int | None
    source: str
    lost_frames: int = 0


class FaceDetector(Protocol):
    """Interface shared by still-image, live and video-analysis callers."""

    def detect_all(self, image: np.ndarray) -> list[FaceDetection]: ...

    def detect_largest(self, image: np.ndarray) -> FaceDetection | None: ...

    def describe(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _choose_nearest(
    detections: list[FaceDetection],
    width: int,
    height: int,
    prev_box: tuple[int, int, int, int] | None,
) -> FaceDetection | None:
    if not detections:
        return None
    if prev_box is None:
        return max(detections, key=lambda detection: float(detection.score))
    return max(
        detections,
        key=lambda detection: _iou(detection.box, prev_box) * 3.0 + float(detection.score),
    )


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int] | None) -> float:
    if right is None:
        return 0.0
    left_x, left_y, left_width, left_height = left
    right_x, right_y, right_width, right_height = right
    intersection_x1 = max(left_x, right_x)
    intersection_y1 = max(left_y, right_y)
    intersection_x2 = min(left_x + left_width, right_x + right_width)
    intersection_y2 = min(left_y + left_height, right_y + right_height)
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    union = left_width * left_height + right_width * right_height - intersection
    return intersection / union if union else 0.0


__all__ = ["FaceDetection", "FaceDetector", "FaceTrackResult"]

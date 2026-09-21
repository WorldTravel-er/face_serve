"""Detection quality policy and per-frame face selection for video analysis.

``face_core`` analysis and the ``face_api`` video tasks both run the same
"detect one subject and follow it" pipeline.  This module keeps that policy in
one place so both entry points behave identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .base import FaceDetection, _choose_nearest


FaceSelection = Literal["largest", "confidence", "continuity"]

FACE_SELECTION_STRATEGIES: tuple[FaceSelection, ...] = ("largest", "confidence", "continuity")


@dataclass(frozen=True)
class DetectionPolicy:
    """Quality bar a detection must clear before it is treated as a face.

    Detections below :attr:`minimum_score` are discarded, so a frame whose only
    candidates are noise yields no face at all instead of a junk crop.
    """

    minimum_score: float = 0.05
    min_face_area_ratio: float = 0.001
    max_face_area_ratio: float = 0.30
    min_face_aspect_ratio: float = 0.30
    max_face_aspect_ratio: float = 1.35
    max_faces_per_frame: int = 1
    #: How the single subject is chosen when several faces are visible.
    #: ``largest`` follows the biggest face, i.e. the person closest to the camera.
    face_selection: FaceSelection = "largest"

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_face_area_ratio <= self.max_face_area_ratio <= 1.0:
            raise ValueError("face area ratios must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.min_face_aspect_ratio <= self.max_face_aspect_ratio:
            raise ValueError("face aspect ratios must satisfy 0 <= min <= max")
        if self.max_faces_per_frame < 1:
            raise ValueError("max_faces_per_frame must be at least one")
        if self.face_selection not in FACE_SELECTION_STRATEGIES:
            raise ValueError(
                f"face_selection must be one of {list(FACE_SELECTION_STRATEGIES)}, got {self.face_selection!r}"
            )

    def accepts(self, detection: FaceDetection, width: int, height: int) -> bool:
        """Report whether one detection is a usable face in a ``width`` x ``height`` frame."""
        if float(detection.score) < self.minimum_score:
            return False
        _, _, box_width, box_height = detection.box
        if box_width <= 0 or box_height <= 0:
            return False
        area_ratio = box_width * box_height / max(1, width * height)
        if not self.min_face_area_ratio <= area_ratio <= self.max_face_area_ratio:
            return False
        return self.min_face_aspect_ratio <= box_width / box_height <= self.max_face_aspect_ratio

    def filter(self, detections: Iterable[FaceDetection], width: int, height: int) -> list[FaceDetection]:
        accepted = [item for item in detections if self.accepts(item, width, height)]
        accepted.sort(key=_face_order_key(self.face_selection), reverse=True)
        return accepted

    def select(self, detections: list[FaceDetection], previous_box: tuple[int, int, int, int] | None) -> FaceDetection | None:
        return primary_face(detections, previous_box=previous_box, strategy=self.face_selection)

    def select_largest(self, detections: list[FaceDetection]) -> FaceDetection | None:
        """Pick the biggest face, ignoring any temporal strategy."""
        return primary_face(detections, strategy="largest")


def _face_area(detection: FaceDetection) -> float:
    _, _, box_width, box_height = detection.box
    return float(max(0, box_width) * max(0, box_height))


def _face_order_key(strategy: FaceSelection):
    """Ranking key: best first when sorted descending."""
    if strategy == "largest":
        # Area decides; score and position only break exact ties, so the ranking
        # stays deterministic for equal-sized boxes.
        return lambda detection: (_face_area(detection), float(detection.score), -detection.box[0], -detection.box[1])
    return lambda detection: float(detection.score)


def primary_face(
    detections: list[FaceDetection],
    previous_box: tuple[int, int, int, int] | None = None,
    strategy: FaceSelection = "largest",
) -> FaceDetection | None:
    """Pick the one face to follow, or ``None`` when nothing qualifies.

    ``largest`` tracks the biggest face, which in practice is the person closest
    to the camera; its apparent size barely changes as the subject moves
    sideways, so the selection is stable.  ``continuity`` instead favours the
    face overlapping ``previous_box``, keeping an identity through crossings;
    without a previous box it bootstraps from the largest face.
    """
    if not detections:
        return None
    if strategy == "continuity":
        if previous_box is None:
            return max(detections, key=_face_order_key("largest"))
        return _choose_nearest(detections, 0, 0, previous_box)
    return max(detections, key=_face_order_key(strategy))


__all__ = ["DetectionPolicy", "FACE_SELECTION_STRATEGIES", "FaceSelection", "primary_face"]

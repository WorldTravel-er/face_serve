"""Small, dependency-light ByteTrack-style association for face detections.

This module deliberately uses only NumPy and ``lap``.  RKNN deployments must
not bring a heavyweight neural-network framework into video analysis merely
to retain stable detection IDs.
"""

from __future__ import annotations

from dataclasses import dataclass

import lap
import numpy as np

from face_core.face_recognition_engine.face_detection.base import FaceDetection


@dataclass
class _Track:
    track_id: int
    box: tuple[int, int, int, int]
    detection: FaceDetection
    misses: int = 0

    def update(self, detection: FaceDetection) -> None:
        self.box = detection.box
        self.detection = detection
        self.misses = 0


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    left_x, left_y, left_w, left_h = left
    right_x, right_y, right_w, right_h = right
    left_x2, left_y2 = left_x + max(left_w, 0), left_y + max(left_h, 0)
    right_x2, right_y2 = right_x + max(right_w, 0), right_y + max(right_h, 0)
    overlap_w = max(0, min(left_x2, right_x2) - max(left_x, right_x))
    overlap_h = max(0, min(left_y2, right_y2) - max(left_y, right_y))
    intersection = overlap_w * overlap_h
    union = max(left_w, 0) * max(left_h, 0) + max(right_w, 0) * max(right_h, 0) - intersection
    return float(intersection / union) if union > 0 else 0.0


def _associate(
    tracks: list[_Track],
    detections: list[FaceDetection],
    minimum_iou: float,
) -> list[tuple[int, int]]:
    """Return local ``(track_index, detection_index)`` assignment pairs."""
    if not tracks or not detections:
        return []
    cost = 1.0 - np.asarray(
        [[_iou(track.box, detection.box) for detection in detections] for track in tracks],
        dtype=np.float64,
    )
    _, row_to_column, _ = lap.lapjv(
        cost,
        extend_cost=True,
        cost_limit=1.0 - float(minimum_iou),
    )
    return [(row, int(column)) for row, column in enumerate(row_to_column) if column >= 0]


class ByteTrackAssociator:
    """Two-stage IoU associator retaining IDs across short detection gaps."""

    def __init__(
        self,
        track_buffer: int = 30,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_iou: float = 0.5,
    ) -> None:
        if track_buffer < 0:
            raise ValueError("track_buffer must be non-negative")
        if not 0.0 <= low_threshold <= high_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= low <= high <= 1")
        if not 0.0 <= match_iou <= 1.0:
            raise ValueError("match_iou must be in [0, 1]")
        self.track_buffer = int(track_buffer)
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.match_iou = float(match_iou)
        self._tracks: list[_Track] = []
        self._next_track_id = 1

    def update(self, detections: list[FaceDetection]) -> list[tuple[FaceDetection, int]]:
        """Associate one frame and return only tracks observed in this frame.

        High-confidence detections can reactivate tracks lost during the
        configured buffer.  Low-confidence detections only rescue tracks that
        were active in the immediately preceding frame, matching ByteTrack's
        conservative second stage and avoiding spurious lost-track revival.
        """
        high = [detection for detection in detections if detection.score >= self.high_threshold]
        low = [
            detection
            for detection in detections
            if self.low_threshold <= detection.score < self.high_threshold
        ]
        for track in self._tracks:
            track.misses += 1

        high_matched = self._match_and_update(high, range(len(self._tracks)))
        active_unmatched = [
            index
            for index, track in enumerate(self._tracks)
            if index not in high_matched and track.misses == 1
        ]
        self._match_and_update(low, active_unmatched)

        for detection_index, detection in enumerate(high):
            if detection_index not in high_matched.values():
                self._tracks.append(
                    _Track(self._next_track_id, detection.box, detection))
                self._next_track_id += 1

        self._tracks = [track for track in self._tracks if track.misses <= self.track_buffer]
        return [
            (track.detection, track.track_id)
            for track in self._tracks
            if track.misses == 0
        ]

    def _match_and_update(
        self,
        detections: list[FaceDetection],
        track_indices: range | list[int],
    ) -> dict[int, int]:
        selected_track_indices = list(track_indices)
        selected_tracks = [self._tracks[index] for index in selected_track_indices]
        matched: dict[int, int] = {}
        for local_track_index, detection_index in _associate(selected_tracks, detections, self.match_iou):
            track_index = selected_track_indices[local_track_index]
            self._tracks[track_index].update(detections[detection_index])
            matched[track_index] = detection_index
        return matched

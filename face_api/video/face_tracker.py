"""Runtime-independent ByteTrack adapter for face detectors."""

from __future__ import annotations

from typing import Any

import numpy as np

from face_api.core.performance_logging import measure_face_stage
from face_api.video.bytetrack import ByteTrackAssociator
from face_core.face_recognition_engine.face_detection.base import FaceDetection, FaceTrackResult
from face_core.face_recognition_engine.face_detection.policy import DetectionPolicy


class ByteTrackFaceTracker:
    """Own tracking state while sharing a stateless detector.

    Only the primary face is handed to callers.  Weak candidates are dropped by
    the :class:`DetectionPolicy`, so a frame whose only detections are noise
    reports no face at all instead of a junk crop.

    The default ``largest`` selection follows the biggest face on screen, which
    is the person closest to the camera.  Its apparent size stays stable while
    the subject moves sideways, so the chosen box does not jump between people.
    Note that when the biggest face changes between frames, the reported
    ``track_id`` follows the new face rather than the previous subject.
    """

    def __init__(
        self,
        detector: Any,
        track_buffer: int = 30,
        match_thresh: float = 0.3,
        frame_rate: int = 30,
        high_threshold: float | None = None,
        min_face_area_ratio: float = 0.0,
        max_face_area_ratio: float = 1.0,
        min_face_aspect_ratio: float = 0.0,
        max_face_aspect_ratio: float = float("inf"),
        max_candidates: int | None = None,
        min_face_score: float | None = None,
        max_faces_per_frame: int | None = None,
        face_selection: str | None = None,
    ) -> None:
        if not callable(getattr(detector, "detect_all", None)):
            raise TypeError("Face tracker requires a detector with detect_all(frame)")
        self.detector = detector

        candidate_threshold = float(getattr(detector, "conf", 0.25))
        # ``high_threshold`` predates the policy, where it only cut the weak
        # detections fed to the associator.  It is now the score a detection must
        # reach to be treated as a face at all.
        quality_floor = max(0.05, candidate_threshold) if high_threshold is None else float(high_threshold)
        if min_face_score is not None:
            quality_floor = max(quality_floor, float(min_face_score))
        self.policy = DetectionPolicy(
            minimum_score=quality_floor,
            min_face_area_ratio=min_face_area_ratio,
            max_face_area_ratio=max_face_area_ratio,
            min_face_aspect_ratio=min_face_aspect_ratio,
            max_face_aspect_ratio=max_face_aspect_ratio,
            max_faces_per_frame=1 if max_faces_per_frame is None else int(max_faces_per_frame),
            face_selection="largest" if face_selection is None else str(face_selection),
        )
        self.max_candidates = None if max_candidates is None else int(max_candidates)

        self.tracker = ByteTrackAssociator(
            track_buffer=int(track_buffer),
            high_threshold=quality_floor,
            low_threshold=min(0.1, candidate_threshold, quality_floor),
            match_iou=float(match_thresh),
        )
        self.last_box: tuple[int, int, int, int] | None = None
        self.lost_frames = 0
    def detect_all(self, frame: np.ndarray) -> list[FaceDetection]:
        """Return the faces reported for one frame, best-for-selection first."""
        return self._accepted(frame)[: self.policy.max_faces_per_frame]

    def _accepted(self, frame: np.ndarray) -> list[FaceDetection]:
        """Every accepted face, before the per-frame report limit is applied.

        Selection and association need the full candidate list: capping it first
        would hide the alternative faces that ``face_selection`` chooses between.
        """
        height, width = frame.shape[:2]
        accepted = self.policy.filter(self.detector.detect_all(frame), width, height)
        if self.max_candidates is not None:
            accepted = accepted[: self.max_candidates]
        return accepted

    def detect_largest(self, frame: np.ndarray) -> FaceDetection | None:
        """Return the biggest face in a single frame (no temporal context)."""
        return self.policy.select_largest(self.detect_all(frame))

    def track_nearest(self, frame: np.ndarray) -> FaceTrackResult | None:
        """Track the primary face, keeping its identity across frames."""
        with measure_face_stage("detect"):
            detections = self._accepted(frame)
        tracked = self.tracker.update(detections)
        if not tracked:
            self.lost_frames += 1
            return None
        candidates = [
            (FaceDetection(tuple(detection.box), float(detection.score), detection.index), track_id)
            for detection, track_id in tracked
        ]
        selected = self.policy.select(
            [detection for detection, _ in candidates],
            self.last_box,
        )
        if selected is None:
            self.lost_frames += 1
            return None
        track_id = next(track_id for detection, track_id in candidates if detection is selected)
        self.last_box = selected.box
        self.lost_frames = 0
        return FaceTrackResult(selected, int(track_id), "retinaface_bytetrack", 0)

    def describe(self) -> dict[str, Any]:
        inner = getattr(self.detector, "describe", None)
        return {
            "tracker": "ByteTrackFaceTracker",
            "policy": {
                "minimum_score": self.policy.minimum_score,
                "min_face_area_ratio": self.policy.min_face_area_ratio,
                "max_face_area_ratio": self.policy.max_face_area_ratio,
                "min_face_aspect_ratio": self.policy.min_face_aspect_ratio,
                "max_face_aspect_ratio": self.policy.max_face_aspect_ratio,
                "max_faces_per_frame": self.policy.max_faces_per_frame,
                "face_selection": self.policy.face_selection,
            },
            "detector": inner() if callable(inner) else {},
        }


__all__ = ["ByteTrackFaceTracker"]

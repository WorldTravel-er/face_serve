from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


from .base import FaceDetection, YoloTrackResult, _choose_nearest, LetterboxMeta, letterbox_rgb_uint8, decode_yolo_output
from .trackers_adapter import TrackersFaceAssociator
from ..onnx_runtime import create_session, input_names, session_provider_names
from face_api.core.performance_logging import measure_face_stage







# 把图片变成ONNX输入。
def preprocess_image(image: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, LetterboxMeta]:
    rgb, meta = letterbox_rgb_uint8(image, imgsz)
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), meta





class YoloFaceOnnxDetector:
    """ONNX Runtime detector with the same high-level methods as YoloFaceTracker."""

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        face_class: int | None = None,
        session: Any | None = None,
        provider_mode: str | None = None,
        track_buffer: int = 30,
        track_iou: float = 0.3,
        track_frame_rate: int = 30,
    ) -> None:
        self.session = session if session is not None else create_session(model_path, provider_mode=provider_mode)
        names = input_names(self.session)
        if len(names) != 1:
            raise ValueError(f"YOLO ONNX model must have one image input, found: {names}")
        self.input_name = names[0]
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.face_class = face_class
        self.locked_track_id: int | None = None
        self.last_box: tuple[int, int, int, int] | None = None
        self.lost_frames = 0
        self.tracker = TrackersFaceAssociator(
            track_activation_threshold=self.conf,
            lost_track_buffer=track_buffer,
            minimum_iou_threshold=track_iou,
            frame_rate=track_frame_rate,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "detector": "YOLO_ONNXRuntime",
            "providers": session_provider_names(self.session) or [],
        }

    def detect_largest(self, image: np.ndarray) -> FaceDetection | None:
        detections = self.detect_all(image)
        if not detections:
            return None
        return _choose_nearest(detections, image.shape[1], image.shape[0], None)

    def detect_all(self, image: np.ndarray) -> list[FaceDetection]:
        return self._detect(image)

    def track_nearest(self, frame: np.ndarray) -> YoloTrackResult | None:
        with measure_face_stage("detect"):
            detections = self.detect_all(frame)
        tracked = self.tracker.update(detections)
        if tracked:
            selected = self._select_tracked_target(tracked, frame.shape[1], frame.shape[0])
            if selected is not None:
                detection, track_id = selected
                self.locked_track_id = track_id
                self.last_box = detection.box
                self.lost_frames = 0
                return YoloTrackResult(
                    detection=detection,
                    track_id=track_id,
                    source="yolo_onnx_track",
                    lost_frames=0,
                )

        detection = _choose_nearest(detections, frame.shape[1], frame.shape[0], self.last_box)
        if detection is None:
            self.lost_frames += 1
            return None
        self.last_box = detection.box
        self.lost_frames = 0
        return YoloTrackResult(detection=detection, track_id=None, source="yolo_onnx_detect", lost_frames=0)

    def _select_tracked_target(
        self,
        candidates: list[tuple[FaceDetection, int]],
        width: int,
        height: int,
    ) -> tuple[FaceDetection, int] | None:
        if self.locked_track_id is not None:
            for detection, track_id in candidates:
                if track_id == self.locked_track_id:
                    return detection, track_id

        detections = [detection for detection, _ in candidates]
        selected = _choose_nearest(detections, width, height, self.last_box)
        if selected is None:
            return None
        for detection, track_id in candidates:
            if detection is selected:
                return detection, track_id
        return None

    def _detect(self, image: np.ndarray) -> list[FaceDetection]:
        tensor, meta = preprocess_image(image, self.imgsz)
        outputs = self.session.run(None, {self.input_name: tensor})
        return decode_yolo_output(outputs[0] if len(outputs) == 1 else outputs, meta, self.conf, self.iou, self.face_class)



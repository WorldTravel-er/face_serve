from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .base import FaceDetection, YoloTrackResult, _choose_nearest
import numpy as np



class YoloFaceTracker:
    """Nearest-target face selector based on YOLO and Ultralytics tracking."""
    def __init__(
        self,
        model_path: str | Path,
        tracker: str = "bytetrack.yaml",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
        face_class: int | None = None,
        task: str = "detect",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency-gated.
            raise RuntimeError(
                "Ultralytics is not installed. Run `pip install -r requirements.txt` "
                "and provide a YOLO face model with `--yolo-face-model`."
            ) from exc

        self.model = YOLO(str(model_path), task=task) # YOLO模型路径
        self.tracker = tracker # 跟踪算法配置
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.face_class = face_class
        self.task = task
        self.locked_track_id: int | None = None
        self.last_box: tuple[int, int, int, int] | None = None
        self.lost_frames = 0

    def describe(self) -> dict[str, Any]:
        return {
            "detector": "YOLO",
            "device": str(self.device or "auto"),
            "task": self.task,
        }

    def detect_largest(self, image: np.ndarray) -> FaceDetection | None:
        candidates = self._predict_candidates(image)
        if not candidates:
            return None
        h, w = image.shape[:2]
        detections = [detection for detection, _ in candidates]
        return _choose_nearest(detections, w, h, None)

    def track_nearest(self, frame: np.ndarray) -> YoloTrackResult | None:
        candidates = self._track_candidates(frame)
        source = "yolo_track"
        if not candidates:
            candidates = self._predict_candidates(frame)
            source = "yolo_detect"
        if not candidates:
            self.lost_frames += 1
            return None

        selected = self._select_target(candidates, frame.shape[1], frame.shape[0])
        if selected is None:
            self.lost_frames += 1
            return None

        detection, track_id = selected
        self.locked_track_id = track_id
        self.last_box = detection.box
        self.lost_frames = 0
        return YoloTrackResult(
            detection=detection,
            track_id=track_id,
            source="yolo_track" if source == "yolo_track" and track_id is not None else "yolo_detect",
            lost_frames=0,
        )

    def _predict_candidates(self, image: np.ndarray) -> list[tuple[FaceDetection, int | None]]:
        results = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._parse_detections(results[0] if results else None)

    def _track_candidates(self, frame: np.ndarray) -> list[tuple[FaceDetection, int | None]]:
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except (RuntimeError, ValueError, OverflowError, FloatingPointError):
            return []
        return self._parse_detections(results[0] if results else None)

    def _select_target(
        self,
        candidates: list[tuple[FaceDetection, int | None]],
        width: int,
        height: int,
    ) -> tuple[FaceDetection, int | None] | None:
        # 选择顺序是：
        if self.locked_track_id is not None:
            for detection, track_id in candidates:
                if track_id == self.locked_track_id:
                    return detection, track_id

        detections = [detection for detection, _ in candidates]
        target = _choose_nearest(detections, width, height, self.last_box)
        if target is None:
            return None
        for detection, track_id in candidates:
            if detection is target:
                return detection, track_id
        return None

    def _parse_detections(self, result: Any | None) -> list[tuple[FaceDetection, int | None]]:
        if result is None or getattr(result, "boxes", None) is None:
            return []
        boxes = result.boxes
        xyxy = _to_numpy(getattr(boxes, "xyxy", None))
        confs = _to_numpy(getattr(boxes, "conf", None))
        classes = _to_numpy(getattr(boxes, "cls", None))
        ids = _to_numpy(getattr(boxes, "id", None))
        if xyxy is None:
            return []

        parsed: list[tuple[FaceDetection, int | None]] = []
        for idx, row in enumerate(xyxy):
            class_id = _safe_int(classes[idx]) if classes is not None and idx < len(classes) else None
            if self.face_class is not None and class_id != self.face_class:
                continue
            if len(row) < 4:
                continue
            coords = np.asarray(row[:4], dtype=np.float32)
            if not np.all(np.isfinite(coords)):
                continue
            x1, y1, x2, y2 = [float(v) for v in coords]
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 0.0 or box_h <= 0.0:
                continue
            x = int(np.floor(x1))
            y = int(np.floor(y1))
            w = max(1, int(np.ceil(box_w)))
            h = max(1, int(np.ceil(box_h)))
            score = _safe_float(confs[idx]) if confs is not None and idx < len(confs) else 0.0
            if score is None:
                continue
            track_id = _safe_int(ids[idx]) if ids is not None and idx < len(ids) else None
            parsed.append((FaceDetection((x, y, w, h), score, None), track_id))
        return parsed


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _safe_int(value: Any) -> int | None:
    result = _safe_float(value)
    return int(result) if result is not None else None


def _to_numpy(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)







"""RKNN adapter for the project's shared YOLO face post-processing."""
# 把摄像头 BGR 图像预处理成 RKNN 模型需要的 uint8 Tensor，
# 送入 RK3588 NPU 推理，再复用 ONNX 版本的 YOLO 后处理和 ByteTrack 跟踪逻辑，
# 最终输出项目统一的 FaceDetection / YoloTrackResult。
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import FaceDetection, YoloTrackResult, _choose_nearest, LetterboxMeta, letterbox_rgb_uint8, decode_yolo_output
from .trackers_adapter import TrackersFaceAssociator
from ..rknn_runtime import RknnModelSession, coerce_rknn_input_dtype, load_model_specs
from face_api.core.performance_logging import current_performance_logger, measure_face_stage, record_face_stage


def decode_yolo_outputs(
    outputs: Sequence[np.ndarray],
    meta: LetterboxMeta,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    face_class: int | None = None,
) -> list[FaceDetection]:
    """Decode the currently supported one-output RKNN YOLO contract."""
    if len(outputs) == 1:
        return decode_yolo_output(outputs[0], meta, conf_threshold, iou_threshold, face_class)
    shapes = [tuple(np.asarray(output).shape) for output in outputs]
    raise ValueError(
        "RKNN YOLO output contract must contain exactly one output; "
        f"received {len(outputs)} outputs with shapes {shapes}"
    )


class YoloFaceRknnDetector:
    """Face detector that supplies manifest-defined uint8 tensors to RKNN Lite2."""

    def __init__(
        self,
        model_path: str | Path,
        manifest_path: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        face_class: int | None = None,
        session: Any | None = None,
        core_mask: str = "auto",
        track_buffer: int = 30,
        track_iou: float = 0.3,
        track_frame_rate: int = 30,
    ) -> None:
        if session is None:
            if manifest_path is None:
                raise ValueError("manifest_path is required when constructing an RKNN model session")
            model = Path(model_path)
            input_spec, output_specs = load_model_specs(Path(manifest_path), model)
            session = RknnModelSession(
                model,
                input_spec,
                output_specs=output_specs,
                core_mask=core_mask,
            )
        self.session = session
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
        description = dict(self.session.describe())
        description.update({"detector": "YOLO_RKNNLite", "runtime": "rknn"})
        return description

    def close(self) -> None:
        self.session.close()

    def detect_all(self, image: np.ndarray) -> list[FaceDetection]:
        logger = current_performance_logger()
        if logger is None or not logger.is_enabled():
            tensor, meta = self._input_tensor(image)
            outputs = self.session.infer(tensor)
            return decode_yolo_outputs(outputs, meta, self.conf, self.iou, self.face_class)

        clock = logger.monotonic_clock
        started_at = clock()
        tensor, meta = self._input_tensor(image)
        preprocessed_at = clock()
        outputs = self.session.infer(tensor)
        inferred_at = clock()
        detections = decode_yolo_outputs(outputs, meta, self.conf, self.iou, self.face_class)
        finished_at = clock()

        preprocess_ms = (preprocessed_at - started_at) * 1000.0
        rknn_infer_ms = (inferred_at - preprocessed_at) * 1000.0
        postprocess_ms = (finished_at - inferred_at) * 1000.0
        detect_latency_ms = (finished_at - started_at) * 1000.0
        record_face_stage(
            "detect_detail",
            detect_latency_ms,
            logger=logger,
            preprocess_ms=round(preprocess_ms, 3),
            rknn_infer_ms=round(rknn_infer_ms, 3),
            postprocess_ms=round(postprocess_ms, 3),
            detect_latency_ms=round(detect_latency_ms, 3),
        )
        return detections

    def detect_largest(self, image: np.ndarray) -> FaceDetection | None:
        return _choose_nearest(self.detect_all(image), image.shape[1], image.shape[0], None)

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
                return YoloTrackResult(detection, track_id, "yolo_rknn_track", 0)

        detection = _choose_nearest(detections, frame.shape[1], frame.shape[0], self.last_box)
        if detection is None:
            self.lost_frames += 1
            return None
        self.last_box = detection.box
        self.lost_frames = 0
        return YoloTrackResult(detection, None, "yolo_rknn_detect", 0)

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

    def _input_tensor(self, image: np.ndarray) -> tuple[np.ndarray, LetterboxMeta]:
        rgb, meta = letterbox_rgb_uint8(image, self.imgsz)
        input_dtype = self.session.input_spec.dtype
        if input_dtype in (np.dtype("float16"), np.dtype("float32")):
            pixels = rgb.astype(input_dtype, copy=False)
        else:
            pixels = rgb
        if self.session.input_spec.layout == "nchw":
            tensor = np.ascontiguousarray(np.transpose(pixels, (2, 0, 1))[None, :, :, :])
        else:
            tensor = np.ascontiguousarray(pixels[None, :, :, :])
        return coerce_rknn_input_dtype(tensor, self.session.input_spec), meta

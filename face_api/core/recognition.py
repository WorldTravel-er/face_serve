"""
    人脸识别业务层（Recognition Service）
    负责：
        图片 → 人脸检测
        人脸 → 特征向量
        特征向量归一化
        两张脸特征相似度计算
        Top-K 排序匹配
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Protocol

import cv2
import numpy as np

from face_api.core.config import ApiConfig, default_config
from face_api.core.errors import ErrorCode, FaceApiError
from face_api.core.performance_logging import FacePerformanceLogger, measure_face_stage
from face_api.core.subject_store import EmbeddingIdentity


class FeatureCandidate(Protocol):
    subject_id: str
    name: str
    feature: np.ndarray


class RecognitionService:
    def __init__(
        self,
        config: ApiConfig | None = None,
        detector: object | None = None,
        engine: object | None = None,
        provider_mode: str | None = None,
        performance_logger: FacePerformanceLogger | None = None,
    ) -> None:
        self.config = config or default_config()
        self.runtime = self.config.runtime
        self.provider_mode = provider_mode if provider_mode is not None else self.config.provider
        self.detector = detector if detector is not None else self._create_detector()
        try:
            self.engine = engine if engine is not None else self._create_engine()
        except Exception:
            if detector is None:
                close = getattr(self.detector, "close", None)
                if callable(close):
                    close()
            raise
        self.performance_logger = (
            performance_logger if performance_logger is not None else FacePerformanceLogger.from_config(self.config)
        )

    def extract_feature(self, image_bytes: bytes) -> np.ndarray:
        image = decode_image_bytes(image_bytes)
        detection = self.detect_frame(image)
        if detection is None:
            raise FaceApiError(ErrorCode.FACE_NOT_FOUND, "Face not found", http_status=404)
        return self.extract_feature_from_detection(image, detection)

    def extract_feature_from_frame(self, image: np.ndarray) -> np.ndarray:
        detection = self.detect_frame(image)
        if detection is None:
            raise FaceApiError(ErrorCode.FACE_NOT_FOUND, "Face not found", http_status=404)
        return self.extract_feature_from_detection(image, detection)

    def detect_frame(self, frame: np.ndarray) -> Any | None:
        with measure_face_stage("detect", logger=self.performance_logger, runtime=self.runtime):
            return self.detector.detect_largest(frame)

    def extract_feature_from_detection(self, frame: np.ndarray, detection: Any) -> np.ndarray:
        if hasattr(self.engine, "feature_from_detection_with_info"):
            info = self.engine.feature_from_detection_with_info(frame, detection)
            if not getattr(info, "feature_valid", True):
                raise FaceApiError(ErrorCode.INFERENCE_FAILED, "Invalid face feature", http_status=500)
            feature = np.asarray(info, dtype=np.float32)
        else:
            feature = np.asarray(self.engine.feature_from_detection(frame, detection), dtype=np.float32)
        return l2_normalize(feature)

    def match_feature(
        self,
        feature: np.ndarray,
        candidates: Iterable[FeatureCandidate],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        query = l2_normalize(np.asarray(feature, dtype=np.float32))
        results: list[dict] = []
        for candidate in candidates:
            candidate_feature = l2_normalize(np.asarray(candidate.feature, dtype=np.float32))
            similarity = cosine_similarity(query, candidate_feature)
            if threshold is not None and similarity < float(threshold):
                continue
            results.append(
                {
                    "subject_id": candidate.subject_id,
                    "name": candidate.name,
                    "similarity": float(similarity),
                }
            )
        results.sort(key=lambda item: item["similarity"], reverse=True)
        if top_k is not None:
            results = results[: int(top_k)]
        return results

    def recognize_detection(
        self,
        frame: np.ndarray,
        detection: Any,
        candidates: Iterable[FeatureCandidate],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> dict:
        query = self.extract_feature_from_detection(frame, detection)
        return {
            "box": tuple(int(value) for value in detection.box),
            "matches": self.match_feature(query, candidates, top_k=top_k, threshold=threshold),
        }

    def match_frame_with_detection(
        self,
        frame: np.ndarray,
        candidates: Iterable[FeatureCandidate],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> dict:
        detection = self.detect_frame(frame)
        if detection is None:
            raise FaceApiError(ErrorCode.FACE_NOT_FOUND, "Face not found", http_status=404)
        return self.recognize_detection(frame, detection, candidates, top_k=top_k, threshold=threshold)

    def match_frame(
        self,
        frame: np.ndarray,
        candidates: Iterable[FeatureCandidate],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        query = self.extract_feature_from_frame(frame)
        return self.match_feature(query, candidates, top_k=top_k, threshold=threshold)

    def match(
        self,
        image_bytes: bytes,
        candidates: Iterable[FeatureCandidate],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        query = self.extract_feature(image_bytes)
        return self.match_feature(query, candidates, top_k=top_k, threshold=threshold)

    def describe(self) -> dict:
        engine_describe = getattr(self.engine, "describe", None)
        detector_describe = getattr(self.detector, "describe", None)
        engine_info = engine_describe() if callable(engine_describe) else {}
        detector_info = detector_describe() if callable(detector_describe) else {}
        return {
            "runtime": self.runtime,
            "provider_mode": self.provider_mode or "auto",
            "detector": detector_info,
            "engine": engine_info,
        }

    def embedding_identity(self) -> EmbeddingIdentity:
        model_path = self._active_recognition_model_path()
        try:
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(f"Active recognition model is unavailable: {model_path}") from exc
        pipeline_hash = getattr(self.engine, "embedding_pipeline_sha256", None)
        return EmbeddingIdentity(runtime=self.runtime, model_sha256=digest,
                                 pipeline_sha256=pipeline_hash() if callable(pipeline_hash) else "legacy")

    def _active_recognition_model_path(self) -> Path:
        if self.runtime == "onnx":
            return Path(self.config.recognition_model_path)
        if self.runtime == "rknn":
            return Path(self.config.recognition_rknn_model_path)
        raise RuntimeError(f"Unknown inference runtime for embedding identity: {self.runtime!r}")

    def _create_detector(self):
        return self._create_retinaface_detector()

    def _create_engine(self):
        if self.runtime == "onnx":
            return self._create_onnx_engine()
        if self.runtime == "rknn":
            return self._create_rknn_engine()
        raise ValueError(f"Unknown inference runtime: {self.runtime!r}")

    def _create_retinaface_detector(self):
        from face_core.face_recognition_engine.face_detection.factory import create_retinaface_detector
        return create_retinaface_detector(self.config)

    def _create_onnx_engine(self):
        from face_core.face_recognition_engine import FaceOnnxEngine

        return FaceOnnxEngine(
            recognition_model_path=self.config.recognition_model_path,
            aligner_model_path=self.config.aligner_model_path,
            align_score_threshold=self.config.align_score_threshold,
            provider_mode=self.provider_mode,
        )

    def _create_rknn_engine(self):
        from face_core.face_recognition_engine import FaceRknnEngine
        from face_core.face_recognition_engine.rknn_runtime import validate_aligner_runtime_contract

        aligner_runtime = "onnx-cpu" if self.config.rknn_aligner_fallback == "onnx-cpu" else "rknn"
        validate_aligner_runtime_contract(
            self.config.rknn_manifest_path,
            aligner_runtime=aligner_runtime,
            aligner_onnx_path=self.config.aligner_model_path,
        )
        return FaceRknnEngine(
            self.config.recognition_rknn_model_path,
            self.config.aligner_rknn_model_path,
            manifest_path=self.config.rknn_manifest_path,
            aligner_runtime=aligner_runtime,
            aligner_onnx_model_path=self.config.aligner_model_path,
            align_score_threshold=self.config.align_score_threshold,
            core_mask=self.config.rknn_recognition_core_mask,
        )

    def close(self) -> None:
        for adapter in (self.detector, self.engine):
            close = getattr(adapter, "close", None)
            if callable(close):
                close()

    def performance_context(self, **fields: Any):
        return self.performance_logger.context(runtime=self.runtime, **fields)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise FaceApiError(ErrorCode.INVALID_BASE64_IMAGE, "Image content is empty")
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FaceApiError(ErrorCode.INVALID_BASE64_IMAGE, "Cannot decode image")
    return image


def l2_normalize(feature: np.ndarray) -> np.ndarray:
    value = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise FaceApiError(ErrorCode.INFERENCE_FAILED, "Face feature is empty", http_status=500)
    return (value / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

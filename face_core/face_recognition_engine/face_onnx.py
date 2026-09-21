from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from .base import FaceEngine 
from .face_detection import FaceDetection
from .onnx_runtime import create_session, input_names, output_names, session_provider_names
from face_api.core.performance_logging import measure_face_stage

# 人脸特征提取ONNX模型
class FaceRecognitionOnnxSession:
    """ONNX Runtime wrapper for Face recognition models."""
    def __init__(self, session_or_path: Any, provider_mode: str | None = None) -> None:
        self.session = create_session(session_or_path, provider_mode=provider_mode) if isinstance(session_or_path, (str, Path)) else session_or_path
        names = input_names(self.session)
        if "image" not in names:
            raise ValueError(f"Face ONNX model is missing required image input. Found: {names}")
        self.image_input = "image"
        self.keypoints_input = "keypoints" if "keypoints" in names else None
        self.uses_keypoints = self.keypoints_input is not None

    # 输入对其后的图片和关键点坐标（可选），输出人脸512embedding
    def extract(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        image = _ensure_float32_batch(image, expected_ndim=4, name="image")
        keypoints = _ensure_float32_batch(keypoints, expected_ndim=3, name="keypoints")
        feeds = {self.image_input: image}
        if self.keypoints_input is not None:
            feeds[self.keypoints_input] = keypoints
        outputs = self.session.run(None, feeds)
        if not outputs:
            return np.zeros(0, dtype=np.float32)
        feature = np.asarray(outputs[0], dtype=np.float32)
        if feature.ndim > 1:
            feature = feature[0]
        return l2_normalize(feature.reshape(-1))


class FaceOnnxEngine(FaceEngine):
    """ONNX Runtime Face engine.
    The exported aligner keeps the neural network outputs accurate, while its
    original skimage/cv2 geometry path is not safely traceable. This engine uses
    ONNX Runtime for the neural models and recomputes the affine alignment with
    NumPy/OpenCV before calling the recognition ONNX model with image + keypoints.
    图片 - 检测框 - crop - align - recognition - embedding
    """

    def __init__(
        self,
        recognition_model_path: str | Path = Path("models/onnx/cvlface_adaface_ir18_webface4m.onnx"),
        aligner_model_path: str | Path = Path("models/onnx/cvlface_dfa_mobilenet.onnx"),
        face_margin: float = 0.55, # 扩大裁剪区域，保留更多脸部上下文信息。
        align_score_threshold: float = 0.0, # 对齐质量阈值。
        input_size: int = 112,
        aligner_input_size: int = 160,
        aligner_output_size: int = 112,
        recognition_session: Any | None = None,
        aligner_session: Any | None = None,
        provider_mode: str | None = None,
    ) -> None:
        self.recognition_model_path = Path(recognition_model_path)
        self.aligner_model_path = Path(aligner_model_path)
        self.face_margin = float(face_margin)
        self.align_score_threshold = float(align_score_threshold)
        self.input_size = int(input_size)
        self.aligner_input_size = int(aligner_input_size)
        self.aligner_output_size = int(aligner_output_size)
        self.feature_dim = 512
        self.provider_mode = provider_mode

        self.recognition = FaceRecognitionOnnxSession(
            recognition_session or self.recognition_model_path,
            provider_mode=self.provider_mode,
        )
        self.aligner_session = aligner_session if aligner_session is not None else create_session(self.aligner_model_path, provider_mode=self.provider_mode)

        aligner_inputs = input_names(self.aligner_session)
        if len(aligner_inputs) != 1:
            raise ValueError(f"Face aligner ONNX model must have one image input, found: {aligner_inputs}")
        self.aligner_input = aligner_inputs[0]
        self.aligner_outputs = list(output_names(self.aligner_session) or [])

    def crop_face(self, frame: np.ndarray, box: tuple[int, int, int, int], margin: float | None = None) -> np.ndarray:
        h, w = frame.shape[:2]
        x, y, bw, bh = box
        margin = self.face_margin if margin is None else margin
        pad_x = int(bw * margin)
        pad_y = int(bh * margin)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        return frame[y1:y2, x1:x2].copy()

    def feature_from_detection(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        return self.feature(self.crop_face(frame, detection.box))


    def feature(self, face: np.ndarray) -> np.ndarray:
        if face.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
        image = self._bgr_to_input(face)
        with measure_face_stage("align"):
            aligned_image, aligned_keypoints, align_score = self._align(image)
        if align_score is not None and align_score < self.align_score_threshold:
            return np.zeros(self.feature_dim, dtype=np.float32)
        with measure_face_stage("recognition"):
            feature = self.recognition.extract(aligned_image, aligned_keypoints)
        if feature.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
        return feature



    def embedding_pipeline_sha256(self) -> str:
        from .base import alignment_fingerprint
        return alignment_fingerprint(self.aligner_model_path, runtime="onnx",
                                     face_margin=self.face_margin, geometry_size=self.aligner_input_size)

    def describe(self) -> dict[str, Any]:
        return {
            "feature_engine": "face_ONNXRuntime",
            "recognition_model_path": str(self.recognition_model_path),
            "aligner_model_path": str(self.aligner_model_path),
            "provider_mode": self.provider_mode,
            "providers": session_provider_names(self.aligner_session) or session_provider_names(self.recognition.session) or [],
            "face_margin": self.face_margin,
            "align_score_threshold": self.align_score_threshold,
            "uses_keypoints": self.recognition.uses_keypoints,
        }
    def _align(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
        outputs = self.aligner_session.run(None, {self.aligner_input: image})
        original_keypoints = self._named_or_indexed_output(outputs, "original_keypoints", 1)
        align_score = self._score_to_float(self._named_or_indexed_output(outputs, "align_score", 3, required=False))
        aligned_image, aligned_keypoints = align_image_and_keypoints(
            image,
            original_keypoints,
            input_size=self.aligner_input_size,
            output_size=self.aligner_output_size,
        )
        return aligned_image, aligned_keypoints, align_score

    # 从 ONNX 模型推理输出结果 outputs 中，按照“输出名字优先、输出序号兜底”的方式，获取指定输出数据。
    def _named_or_indexed_output(
        self,
        outputs: Sequence[np.ndarray],
        name: str,
        index: int,
        required: bool = True,
    ) -> np.ndarray | None:
        if name in self.aligner_outputs:
            value = outputs[self.aligner_outputs.index(name)]
        elif index < len(outputs):
            value = outputs[index]
        else:
            value = None
        if value is None and required:
            raise ValueError(f"Face aligner ONNX output {name!r} is missing.")
        return None if value is None else np.asarray(value, dtype=np.float32)

    def _bgr_to_input(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.asarray(pil).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(arr.astype(np.float32))

    def _tensor_to_bgr(self, tensor: np.ndarray | None) -> np.ndarray | None:
        if tensor is None:
            return None
        image = np.asarray(tensor, dtype=np.float32)[0]
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image * 0.5 + 0.5, 0.0, 1.0)
        rgb = (image * 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _score_to_float(self, score: np.ndarray | None) -> float | None:
        if score is None:
            return None
        flat = np.asarray(score, dtype=np.float32).reshape(-1)
        return float(flat[0]) if flat.size else None

# 根据人脸检测网络预测出的 5 个关键点，
# 通过相似性变换（Similarity Transform），
# 把不同姿态的人脸旋转、缩放、平移到统一标准位置，
# 输出标准化的人脸图像和对应的新关键点
def align_image_and_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    input_size: int = 160,
    output_size: int = 112,
) -> tuple[np.ndarray, np.ndarray]:
    image = _ensure_float32_batch(image, expected_ndim=4, name="image")
    keypoints = _ensure_float32_batch(keypoints, expected_ndim=3, name="keypoints")
    resized = resize_nchw_align_corners(image, input_size, input_size)
    reference = reference_landmark()
    aligned_images: list[np.ndarray] = []
    aligned_keypoints: list[np.ndarray] = []

    for batch_index in range(resized.shape[0]):
        src_points = keypoints[batch_index].reshape(5, 2) * np.array([input_size, input_size], dtype=np.float32)
        matrix = estimate_similarity_transform(src_points, reference)
        hwc = np.transpose(resized[batch_index], (1, 2, 0))
        warped = cv2.warpAffine(
            hwc,
            matrix.astype(np.float32),
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(-1.0, -1.0, -1.0),
        )
        aligned = np.transpose(warped, (2, 0, 1))
        dst_points = src_points @ matrix[:, :2].T + matrix[:, 2]
        dst_points = dst_points / np.array([output_size, output_size], dtype=np.float32)
        aligned_images.append(aligned.astype(np.float32))
        aligned_keypoints.append(dst_points.astype(np.float32))

    return np.ascontiguousarray(np.stack(aligned_images, axis=0)), np.ascontiguousarray(np.stack(aligned_keypoints, axis=0))


def reference_landmark() -> np.ndarray:
    return np.array(
        [
            [38.29459953, 51.69630051],
            [73.53179932, 51.50139999],
            [56.02519989, 71.73660278],
            [41.54930115, 92.36550140],
            [70.72990036, 92.20410156],
        ],
        dtype=np.float32,
    )


def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    try:
        from skimage import transform as trans

        tform = trans.SimilarityTransform()
        if tform.estimate(src, dst):
            return tform.params[:2, :].astype(np.float32)
    except Exception:
        pass
    return _estimate_similarity_umeyama(src, dst).astype(np.float32)


def _estimate_similarity_umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = dst_centered.T @ src_centered / float(src.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(2, dtype=np.float32)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = np.sum(src_centered * src_centered) / float(src.shape[0])
    scale = float(np.sum(singular_values * np.diag(correction))) / max(variance, 1e-12)
    affine = scale * rotation
    translation = dst_mean - affine @ src_mean
    return np.concatenate([affine, translation[:, None]], axis=1)


def resize_nchw_align_corners(image: np.ndarray, height: int, width: int) -> np.ndarray:
    image = _ensure_float32_batch(image, expected_ndim=4, name="image")
    batch, channels, in_h, in_w = image.shape
    if in_h == height and in_w == width:
        return np.ascontiguousarray(image.copy())

    y = np.linspace(0, in_h - 1, height, dtype=np.float32)
    x = np.linspace(0, in_w - 1, width, dtype=np.float32)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (y - y0).reshape(1, 1, height, 1)
    wx = (x - x0).reshape(1, 1, 1, width)

    top_left = image[:, :, y0[:, None], x0[None, :]]
    top_right = image[:, :, y0[:, None], x1[None, :]]
    bottom_left = image[:, :, y1[:, None], x0[None, :]]
    bottom_right = image[:, :, y1[:, None], x1[None, :]]
    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    resized = top * (1.0 - wy) + bottom * wy
    return np.ascontiguousarray(resized.astype(np.float32).reshape(batch, channels, height, width))


def l2_normalize(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    norm = float(np.linalg.norm(feature))
    if norm < 1e-8:
        return feature.astype(np.float32)
    return (feature / norm).astype(np.float32)


def _ensure_float32_batch(array: np.ndarray, expected_ndim: int, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != expected_ndim:
        raise ValueError(f"{name} must have {expected_ndim} dimensions, got shape {value.shape}")
    return value


__all__ = [
    "FaceOnnxEngine",
    "FaceRecognitionOnnxSession",
    "align_image_and_keypoints",
    "estimate_similarity_transform",
    "l2_normalize",
    "reference_landmark",
    "resize_nchw_align_corners",
]

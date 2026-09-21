"""RKNN Lite2 CVLFace feature extraction with an explicitly selected aligner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np
from PIL import Image

from .onnx_runtime import create_session, input_names, output_names
from .rknn_runtime import RknnInputSpec, RknnModelSession, coerce_rknn_input_dtype, load_model_specs

from .base import FaceEngine
from .face_detection.base import FaceDetection
from face_api.core.performance_logging import measure_face_stage



# 维度检查
def _ensure_float32_batch(array: np.ndarray, expected_ndim: int, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != expected_ndim:
        raise ValueError(f"{name} must have {expected_ndim} dimensions, got shape {value.shape}")
    return value

# 人脸特征归一化
def l2_normalize(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    norm = float(np.linalg.norm(feature))
    if norm < 1e-8:
        return feature.astype(np.float32)
    return (feature / norm).astype(np.float32)

# 返回aling后的图片和keypoints
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

# 返回标准五点
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

# src五点 → dst标准五点之间的相似变换矩阵。
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



def bgr_to_rgb_uint8(image: np.ndarray, input_spec: RknnInputSpec) -> np.ndarray:
    """Convert BGR pixels to the manifest-declared RKNN tensor layout."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected BGR HxWx3 image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 BGR image, got dtype {image.dtype}")
    if input_spec.dtype not in (np.dtype("uint8"), np.dtype("int8"), np.dtype("float16"), np.dtype("float32")):
        raise ValueError(f"RKNN CVLFace input must be uint8, int8, float16, or float32, got {input_spec.dtype}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if input_spec.layout == "nhwc":
        _, height, width, _ = input_spec.shape
    else:
        _, _, height, width = input_spec.shape
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    if input_spec.dtype in (np.dtype("float16"), np.dtype("float32")):
        pixels = resized.astype(input_spec.dtype, copy=False)
    else:
        pixels = resized
    if input_spec.layout == "nhwc":
        tensor = np.ascontiguousarray(pixels[None, :, :, :])
    else:
        tensor = np.ascontiguousarray(np.transpose(pixels, (2, 0, 1))[None, :, :, :])
    return coerce_rknn_input_dtype(tensor, input_spec)


class RknnFaceRecognitionSession:
    """Extract normalized AdaFace embeddings from a manifest-configured session."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def extract(self, aligned_rgb_uint8: np.ndarray) -> np.ndarray:
        outputs = self.session.infer(aligned_rgb_uint8)
        if not outputs:
            return np.zeros(0, dtype=np.float32)
        return l2_normalize(np.asarray(outputs[0], dtype=np.float32).reshape(-1))


class FaceRknnEngine(FaceEngine):
    """AdaFace IR50 on RKNN, with either RKNN or explicit CPU ONNX alignment."""

    def __init__(
        self,
        recognition_model_path: str | Path,
        aligner_model_path: str | Path,
        manifest_path: str | Path | None = None,
        *,
        aligner_runtime: Literal["rknn", "onnx-cpu"] = "rknn",
        aligner_onnx_model_path: str | Path | None = None,
        align_score_threshold: float = 0.0,
        core_mask: str = "auto",
        recognition_session: Any | None = None,
        aligner_session: Any | None = None,
    ) -> None:
        self.recognition_model_path = Path(recognition_model_path)
        self.aligner_model_path = Path(aligner_model_path)
        self.aligner_runtime = aligner_runtime
        self.aligner_onnx_model_path = Path(aligner_onnx_model_path) if aligner_onnx_model_path is not None else None
        self.align_score_threshold = float(align_score_threshold)
        self.core_mask = core_mask
        self.face_margin = 0.55
        self.feature_dim = 512
        self._rknn_aligner: Any | None = None
        self._owns_recognition_session = False
        self._owns_rknn_aligner = False
        self._onnx_aligner: Any | None = None
        self._onnx_aligner_input: str | None = None
        self._onnx_aligner_image_size: tuple[int, int] | None = None
        self._onnx_aligner_outputs: list[str] = []

        if recognition_session is None:
            if manifest_path is None:
                raise ValueError("manifest_path is required when constructing an RKNN recognition session")
            recognition_input, recognition_outputs = load_model_specs(
                Path(manifest_path), self.recognition_model_path
            )
            recognition_session = RknnModelSession(
                self.recognition_model_path,
                recognition_input,
                output_specs=recognition_outputs,
                core_mask=core_mask,
            )
            self._owns_recognition_session = True
        self.recognition = RknnFaceRecognitionSession(recognition_session)

        try:
            if aligner_runtime == "onnx-cpu":
                if aligner_session is not None:
                    self._onnx_aligner = aligner_session
                else:
                    if self.aligner_onnx_model_path is None:
                        raise ValueError("aligner_onnx_model_path is required for onnx-cpu fallback")
                    self._onnx_aligner = create_session(self.aligner_onnx_model_path, provider_mode="cpu")
                aligner_inputs = input_names(self._onnx_aligner)
                if len(aligner_inputs) != 1:
                    raise ValueError(f"CVLFace aligner ONNX model must have one image input, found: {aligner_inputs}")
                self._onnx_aligner_input = aligner_inputs[0]
                self._onnx_aligner_image_size = self._onnx_session_spatial_dimensions(self._onnx_aligner)
                self._onnx_aligner_outputs = list(output_names(self._onnx_aligner) or [])
            elif aligner_runtime == "rknn":
                if aligner_session is None:
                    if manifest_path is None:
                        raise ValueError("manifest_path is required when constructing an RKNN aligner session")
                    aligner_input, aligner_outputs = load_model_specs(
                        Path(manifest_path), self.aligner_model_path
                    )
                    aligner_session = RknnModelSession(
                        self.aligner_model_path,
                        aligner_input,
                        output_specs=aligner_outputs,
                        core_mask=core_mask,
                    )
                    self._owns_rknn_aligner = True
                self._rknn_aligner = aligner_session
            else:
                raise ValueError(f"Unknown aligner_runtime: {aligner_runtime!r}")
        except Exception:
            self.close()
            raise

    def feature_from_detection(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        return self.feature_from_detection_with_info(frame, detection)

    def feature_from_detection_with_info(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        return self.feature_with_info(self.crop_face(frame, detection.box))

    def feature(self, face: np.ndarray) -> np.ndarray:
        return self.feature_with_info(face)

    def feature_with_info(self, face: np.ndarray) -> np.ndarray:
        if face.size == 0:
            return self._empty_feature()
        with measure_face_stage("align"):
            aligned_image, align_score = self._align(face)
        if align_score is not None and align_score < self.align_score_threshold:
            return self._empty_feature(align_score=align_score, aligned_image=aligned_image)
        aligned_bgr = self._aligned_tensor_to_bgr(aligned_image)
        tensor = bgr_to_rgb_uint8(aligned_bgr, self.recognition.session.input_spec)
        with measure_face_stage("recognition"):
            feature = self.recognition.extract(tensor)
        if feature.size == 0:
            return self._empty_feature(align_score=align_score, aligned_image=aligned_image)
        self.feature_dim = int(feature.size)
        return feature


    def embedding_pipeline_sha256(self) -> str:
        from .base import alignment_fingerprint
        path = self.aligner_onnx_model_path if self.aligner_runtime == "onnx-cpu" else self.aligner_model_path
        return alignment_fingerprint(path, runtime=self.aligner_runtime, face_margin=self.face_margin)

    def describe(self) -> dict[str, Any]:
        return {
            "feature_engine": "CVLFace_RKNNLite",
            "recognition_runtime": "rknn",
            "aligner_runtime": self.aligner_runtime,
            "recognition_model_path": str(self.recognition_model_path),
            "aligner_model_path": str(self.aligner_onnx_model_path if self.aligner_runtime == "onnx-cpu" else self.aligner_model_path),
            "device": "RK3588 NPU",
            "align_score_threshold": self.align_score_threshold,
            "recognition": self.recognition.session.describe(),
            "aligner": self._aligner_description(),
        }

    def close(self) -> None:
        if self._owns_recognition_session:
            self.recognition.session.close()
            self._owns_recognition_session = False
        if self._owns_rknn_aligner and self._rknn_aligner is not None:
            self._rknn_aligner.close()
            self._owns_rknn_aligner = False

    def _align(self, face: np.ndarray) -> tuple[np.ndarray, float | None]:
        if self._rknn_aligner is not None:
            input_tensor = bgr_to_rgb_uint8(face, self._rknn_aligner.input_spec)
            outputs = self._rknn_aligner.infer(input_tensor)
            original_keypoints = self._output_at(outputs, 1, "original_keypoints")
            align_score = self._score_to_float(self._output_at(outputs, 3, "align_score", required=False))
            self._spatial_dimensions(self._rknn_aligner.input_spec)
            height = width = 160
            image = self._rknn_geometry_image(face, height, width)
        else:
            assert self._onnx_aligner is not None and self._onnx_aligner_input is not None
            height, width = self._onnx_aligner_image_size or self._onnx_session_spatial_dimensions(self._onnx_aligner)
            image = self._onnx_geometry_image(face, height, width)
            outputs = self._onnx_aligner.run(None, {self._onnx_aligner_input: image})
            original_keypoints = self._onnx_named_or_indexed_output(outputs, "original_keypoints", 1)
            align_score = self._score_to_float(self._onnx_named_or_indexed_output(outputs, "align_score", 3, required=False))
            _, _, height, width = image.shape
        if height != width:
            raise ValueError(f"CVLFace aligner input must be square, got {(height, width)}")
        aligned_image, _ = align_image_and_keypoints(
            image,
            original_keypoints,
            input_size=160,
            output_size=112,
        )
        return aligned_image, align_score

    @staticmethod
    def _spatial_dimensions(spec: RknnInputSpec) -> tuple[int, int, int]:
        if spec.layout == "nhwc":
            _, height, width, channels = spec.shape
        else:
            _, channels, height, width = spec.shape
        if channels != 3:
            raise ValueError(f"CVLFace aligner input must have three channels, got {spec.shape}")
        return channels, height, width

    @staticmethod
    def _rknn_geometry_image(face: np.ndarray, height: int, width: int) -> np.ndarray:
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        return np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[None, :, :, :])

    @staticmethod
    def _onnx_session_spatial_dimensions(session: object) -> tuple[int, int]:
        try:
            input_info = session.get_inputs()[0]
            raw_shape = getattr(input_info, "shape")
        except Exception as exc:
            raise ValueError("CVLFace aligner ONNX input shape could not be read") from exc
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 4:
            raise ValueError(f"CVLFace aligner ONNX input must be four-dimensional, got {raw_shape!r}")
        if raw_shape[1] == 3:
            height, width = raw_shape[2], raw_shape[3]
        elif raw_shape[3] == 3:
            height, width = raw_shape[1], raw_shape[2]
        else:
            raise ValueError(f"CVLFace aligner ONNX input must have three channels, got {raw_shape!r}")
        if (
            not isinstance(height, int)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or isinstance(width, bool)
            or height < 1
            or width < 1
        ):
            raise ValueError(f"CVLFace aligner ONNX input must have static spatial dimensions, got {raw_shape!r}")
        return int(height), int(width)

    @staticmethod
    def _onnx_geometry_image(face: np.ndarray, height: int, width: int) -> np.ndarray:
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        resized = Image.fromarray(rgb).resize((width, height), Image.BILINEAR)
        normalized = (np.asarray(resized).astype(np.float32) / 255.0 - 0.5) / 0.5
        return np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[None, :, :, :])

    @staticmethod
    def _aligned_tensor_to_bgr(tensor: np.ndarray) -> np.ndarray:
        image = np.asarray(tensor, dtype=np.float32)[0]
        rgb = np.clip(np.transpose(image, (1, 2, 0)) * 0.5 + 0.5, 0.0, 1.0)
        return cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _output_at(outputs: Sequence[np.ndarray], index: int, name: str, required: bool = True) -> np.ndarray | None:
        if index >= len(outputs):
            if required:
                shapes = [tuple(np.asarray(output).shape) for output in outputs]
                raise ValueError(f"CVLFace RKNN aligner output {name!r} is missing; outputs have shapes {shapes}")
            return None
        return np.asarray(outputs[index], dtype=np.float32)

    def _onnx_named_or_indexed_output(
        self, outputs: Sequence[np.ndarray], name: str, index: int, required: bool = True
    ) -> np.ndarray | None:
        if name in self._onnx_aligner_outputs:
            value = outputs[self._onnx_aligner_outputs.index(name)]
        elif index < len(outputs):
            value = outputs[index]
        else:
            value = None
        if value is None and required:
            raise ValueError(f"CVLFace aligner ONNX output {name!r} is missing.")
        return None if value is None else np.asarray(value, dtype=np.float32)

    def _empty_feature(self, align_score: float | None = None, aligned_image: np.ndarray | None = None) -> np.ndarray:
        return np.zeros(self.feature_dim, dtype=np.float32)


    @staticmethod
    def _score_to_float(score: np.ndarray | None) -> float | None:
        if score is None:
            return None
        flat = np.asarray(score, dtype=np.float32).reshape(-1)
        return float(flat[0]) if flat.size else None

    def _aligner_description(self) -> dict[str, Any]:
        if self._rknn_aligner is not None:
            return self._rknn_aligner.describe()
        return {"runtime": "onnx-cpu", "model_path": str(self.aligner_onnx_model_path or self.aligner_model_path)}


__all__ = [
    "FaceRknnEngine",
    "RknnFaceRecognitionSession",
    "align_image_and_keypoints",
    "bgr_to_rgb_uint8",
    "estimate_similarity_transform",
    "l2_normalize",
    "reference_landmark",
    "resize_nchw_align_corners",
]

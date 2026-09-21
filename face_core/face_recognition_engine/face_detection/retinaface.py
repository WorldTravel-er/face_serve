from __future__ import annotations

import hashlib
import math
import threading
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .base import FaceDetection
from ..onnx_runtime import create_session, output_names, session_provider_names


MIN_SIZES = ((16, 32), (64, 128), (256, 512))
STEPS = (8, 16, 32)
VARIANCES = (0.1, 0.2)
BGR_MEAN = np.asarray((104.0, 117.0, 123.0), dtype=np.float32)
_RETINAFACE_ONNX_OUTPUTS = ("boxes", "scores", "landmarks")


class _RetinaFaceDetectorBase:
    """Shared RetinaFace detection policy: preprocessing, rescaling and decode.

    RetinaFace is a single-shot detector: one forward pass, one confidence
    threshold, one NMS. Subclasses only supply the concrete inference backend
    through :meth:`_infer` and report their runtime through :meth:`_runtime_fields`.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.4,
        target_size: int = 640,
        max_size: int = 960,
        top_k: int = 5000,
        keep_top_k: int = 750,
        **legacy_kwargs: Any,
    ) -> None:
        if legacy_kwargs:
            # The removed two-pass "rescue" detection used these names. They are
            # accepted and ignored so existing callers such as ApiConfig keep working.
            unknown = {key for key in legacy_kwargs if not key.startswith("rescue_")}
            if unknown:
                raise TypeError(f"Unexpected RetinaFace detector arguments: {sorted(unknown)}")
        self.model_path = Path(model_path).expanduser().resolve()
        self.conf = _unit_interval(confidence_threshold, "confidence_threshold")
        self.nms_threshold = _unit_interval(nms_threshold, "nms_threshold")
        self.target_size = _positive_int(target_size, "target_size")
        self.max_size = _positive_int(max_size, "max_size")
        self.top_k = _positive_int(top_k, "top_k")
        self.keep_top_k = _positive_int(keep_top_k, "keep_top_k")
        self.backbone = "custom"
        self._model_sha256 = _sha256_file(self.model_path)
        self._load_backend()
        self._info = self._build_info()

    def _load_backend(self) -> None:
        """Load the detection backend. Subclasses must implement this."""
        raise NotImplementedError

    def _runtime_fields(self) -> dict[str, Any]:
        """Backend-specific entries for :meth:`describe`."""
        return {}

    def _build_info(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "model_sha256": self._model_sha256,
            "backbone": self.backbone,
            "confidence_threshold": self.conf,
            "nms_threshold": self.nms_threshold,
            "target_size": self.target_size,
            "max_size": self.max_size,
            "top_k": self.top_k,
            "keep_top_k": self.keep_top_k,
            **self._runtime_fields(),
        }

    def detect_all(self, image: np.ndarray) -> list[FaceDetection]:
        """Run one single-shot pass over the full frame."""
        return self._detect_once(
            image,
            target_size=self.target_size,
            max_size=self.max_size,
            confidence_threshold=self.conf,
        )

    def _detect_once(
        self,
        image: np.ndarray,
        *,
        target_size: int,
        max_size: int,
        confidence_threshold: float,
    ) -> list[FaceDetection]:
        tensor, scale, resized_shape = preprocess_image(image, target_size, max_size)
        locations, scores, landmarks = self._infer(tensor)
        return decode_retinaface(
            locations,
            scores,
            landmarks,
            resized_shape=resized_shape,
            original_shape=image.shape[:2],
            resize_scale=scale,
            confidence_threshold=confidence_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
            keep_top_k=self.keep_top_k,
        )

    def _infer(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one forward pass on an ``NCHW`` float32 batch. Subclasses must implement this."""
        raise NotImplementedError

    def detect_largest(self, image: np.ndarray) -> FaceDetection | None:
        detections = self.detect_all(image)
        return select_primary_face(detections, image.shape[1], image.shape[0])

    def describe(self) -> dict[str, Any]:
        return dict(self._info)

    def close(self) -> None:
        return None


class RetinaFacePytorchDetector(_RetinaFaceDetectorBase):
    """Full-frame RetinaFace detector backed by PyTorch."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        self.device = _resolve_device(device)
        self._inference_lock = threading.Lock()
        self._provided_model = model
        super().__init__(model_path, **kwargs)

    def _load_backend(self) -> None:
        import torch
        from .retinaface_model import RetinaFace

        if self._provided_model is not None:
            self.model = self._provided_model
        else:
            state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
            if not isinstance(state_dict, dict):
                raise TypeError(f"RetinaFace checkpoint must be a state dict, got {type(state_dict).__name__}")
            normalized = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
            self.backbone = _checkpoint_backbone(normalized)
            self.model = RetinaFace(self.backbone)
            self.model.load_state_dict(normalized, strict=True)
        self.model.eval().to(self.device)

    def _infer(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        with self._inference_lock, torch.inference_mode():
            locations, scores, landmarks = self.model(torch.from_numpy(tensor).to(self.device))
        return (
            locations[0].detach().cpu().numpy(),
            scores[0].detach().cpu().numpy(),
            landmarks[0].detach().cpu().numpy(),
        )

    def _build_info(self) -> dict[str, Any]:
        return {"detector": "RetinaFace_PyTorch", **super()._build_info(), "device": str(self.device)}


class RetinaFaceOnnxDetector(_RetinaFaceDetectorBase):
    """Full-frame RetinaFace detector backed by ONNX Runtime.

    The exported graphs take a dynamic ``float32[N, 3, H, W]`` BGR-minus-mean input
    and return ``boxes``, ``scores`` and ``landmarks``, so the rescale-and-decode
    stage is identical to the PyTorch detector.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        provider_mode: str | None = None,
        session: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.provider_mode = provider_mode
        self._provided_session = session
        super().__init__(model_path, **kwargs)

    def _load_backend(self) -> None:
        self.session = self._provided_session if self._provided_session is not None else create_session(
            self.model_path, provider_mode=self.provider_mode
        )
        inputs = list(self.session.get_inputs())
        if len(inputs) != 1:
            raise ValueError(f"RetinaFace ONNX model must have exactly one input, found: {[item.name for item in inputs]}")
        outputs = list(output_names(self.session) or [])
        missing = [name for name in _RETINAFACE_ONNX_OUTPUTS if name not in outputs]
        if missing:
            raise ValueError(f"RetinaFace ONNX model is missing outputs {missing}. Found: {outputs}")
        self.input_name = inputs[0].name
        self.output_names = [*_RETINAFACE_ONNX_OUTPUTS]
        self.backbone = _onnx_backbone(self.session) or "custom"
        self._input_hw = _onnx_input_hw(inputs[0])
        shape = getattr(inputs[0], "shape", None)
        if isinstance(shape, (list, tuple)) and len(shape) != 4:
            raise ValueError(f"RetinaFace ONNX input must be NCHW, found shape {shape}")

    def _infer(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        boxes, scores, landmarks = (np.asarray(value, dtype=np.float32) for value in outputs)
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError(f"RetinaFace ONNX boxes output has unexpected shape {boxes.shape}")
        if scores.ndim != 3 or scores.shape[-1] != 2:
            raise ValueError(f"RetinaFace ONNX scores output has unexpected shape {scores.shape}")
        if landmarks.ndim != 3 or landmarks.shape[-1] != 10:
            raise ValueError(f"RetinaFace ONNX landmarks output has unexpected shape {landmarks.shape}")
        return boxes[0], scores[0], landmarks[0]

    def _runtime_fields(self) -> dict[str, Any]:
        providers = session_provider_names(self.session) or []
        return {
            "detector": "RetinaFace_ONNXRuntime",
            "provider_mode": self.provider_mode or "auto",
            "providers": providers,
            "device": _effective_device(providers),
            "input_name": self.input_name,
            "input_hw": list(self._input_hw) if self._input_hw else None,
        }



class RetinaFaceRknnDetector(_RetinaFaceDetectorBase):
    """Fixed-shape RetinaFace with raw BGR input and compiled mean subtraction."""

    def __init__(
        self, model_path: str | Path, *, manifest_path: str | Path | None = None,
        core_mask: str = "0", session: Any | None = None, **kwargs: Any,
    ) -> None:
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.core_mask = core_mask
        self._provided_session = session
        self._owns_session = False
        self._output_order = (0, 1, 2)
        super().__init__(model_path, **kwargs)

    def _load_backend(self) -> None:
        from ..rknn_runtime import RknnModelSession, load_model_specs, load_model_metadata

        if self._provided_session is not None:
            self.session = self._provided_session
        else:
            if self.manifest_path is None:
                raise ValueError("manifest_path is required for RKNN RetinaFace")
            metadata = load_model_metadata(self.manifest_path, self.model_path)
            # The architecture is recorded by the manifest generator; never infer
            # it from the artifact or assume the previous default backbone.
            backbone = metadata.get("backbone")
            self.backbone = str(backbone) if isinstance(backbone, str) and backbone else "custom"
            preprocessing = metadata.get("preprocessing", {})
            expected = {
                "color_order": "bgr", "input_range": [0, 255],
                "mean": [104, 117, 123], "std": [1, 1, 1],
                "normalization": "compiled", "resize": "letterbox",
            }
            if any(preprocessing.get(key) != value for key, value in expected.items()):
                raise ValueError("RKNN RetinaFace requires verified raw BGR / compiled mean / letterbox preprocessing")
            roles = metadata.get("output_roles", {})
            if set(roles) != {"boxes", "scores", "landmarks"} or sorted(roles.values()) != [0, 1, 2]:
                raise ValueError("RKNN RetinaFace manifest must map boxes, scores, landmarks to three outputs")
            self._output_order = tuple(roles[name] for name in ("boxes", "scores", "landmarks"))
            input_spec, outputs = load_model_specs(self.manifest_path, self.model_path)
            self.session = RknnModelSession(self.model_path, input_spec, core_mask=self.core_mask, output_specs=outputs)
            self._owns_session = True
        try:
            spec = self.session.input_spec
            if spec.layout == "nchw":
                batch, channels, height, width = spec.shape
            elif spec.layout == "nhwc":
                batch, height, width, channels = spec.shape
            else:
                raise ValueError(f"Unsupported RetinaFace input layout: {spec.layout}")
            if batch != 1 or channels != 3 or min(height, width) < 1:
                raise ValueError(f"Invalid fixed RetinaFace input shape: {spec.shape}")
            if spec.dtype not in (np.dtype("int8"), np.dtype("uint8"), np.dtype("float16"), np.dtype("float32")):
                raise ValueError(f"Unsupported RetinaFace input dtype: {spec.dtype}")
            self._input_hw = (height, width)
        except Exception:
            self.close()
            raise

    def _input_tensor(self, image: np.ndarray):
        import cv2
        from ..rknn_runtime import coerce_rknn_input_dtype

        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("RetinaFace input must be a uint8 BGR HxWx3 image")
        height, width = image.shape[:2]
        if min(height, width) < 1:
            raise ValueError("RetinaFace input image must not be empty")
        target_h, target_w = self._input_hw
        scale = min(target_w / width, target_h / height)
        new_w = min(target_w, max(1, round(width * scale)))
        new_h = min(target_h, max(1, round(height * scale)))
        left, top = (target_w - new_w) // 2, (target_h - new_h) // 2
        pixels = np.empty((target_h, target_w, 3), dtype=np.uint8)
        pixels[:] = BGR_MEAN
        pixels[top:top + new_h, left:left + new_w] = cv2.resize(image, (new_w, new_h))
        spec = self.session.input_spec
        # Do not subtract the mean here: it is already compiled into this RKNN.
        if spec.dtype.kind == "f":
            pixels = pixels.astype(spec.dtype)
        tensor = pixels[None] if spec.layout == "nhwc" else pixels.transpose(2, 0, 1)[None]
        return coerce_rknn_input_dtype(tensor, spec), (new_w / width, new_h / height), (left, top)

    def _infer(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = self.session.infer(tensor)
        if outputs is None or len(outputs) != 3:
            raise ValueError("RKNN RetinaFace must return three outputs")
        count = len(generate_priors(self._input_hw))
        values = []
        for index, columns in zip(self._output_order, (4, 2, 10)):
            raw = np.asarray(outputs[index])
            if raw.dtype.kind != "f":
                raise ValueError("RKNN RetinaFace outputs must be dequantized floating-point tensors")
            value = np.asarray(raw, dtype=np.float32)
            if value.shape == (1, count, columns):
                value = value[0]
            elif value.shape == (1, columns, count):
                value = value[0].T
            if value.shape != (count, columns):
                raise ValueError(f"RKNN RetinaFace output {index} shape {raw.shape}; expected (1, {count}, {columns})")
            values.append(value)
        return tuple(values)

    def detect_all(self, image: np.ndarray) -> list[FaceDetection]:
        tensor, scale, padding = self._input_tensor(image)
        locations, scores, landmarks = self._infer(tensor)
        return decode_retinaface(
            locations, scores, landmarks, resized_shape=self._input_hw,
            original_shape=image.shape[:2], resize_scale=scale, padding=padding,
            confidence_threshold=self.conf, nms_threshold=self.nms_threshold,
            top_k=self.top_k, keep_top_k=self.keep_top_k,
        )

    def _runtime_fields(self) -> dict[str, Any]:
        return {
            "detector": "RetinaFace_RKNNLite", "runtime": "rknn", "device": "RK3588 NPU",
            "input_hw": list(self._input_hw), "core_mask": self.core_mask,
            "preprocessing": "raw BGR, compiled mean subtraction, centered letterbox",
            "session": self.session.describe(),
        }

    def close(self) -> None:
        if self._owns_session:
            self.session.close()
            self._owns_session = False


def _onnx_input_hw(model_input: Any) -> tuple[int, int] | None:
    """Return the fixed ``(height, width)`` of an ONNX input, or ``None`` when dynamic."""
    shape = getattr(model_input, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return None
    height, width = shape[2], shape[3]
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        return height, width
    return None


def _onnx_backbone(session: Any) -> str | None:
    """Read the backbone recorded in the ONNX metadata by the conversion script."""
    try:
        metadata = session.get_modelmeta().custom_metadata_map or {}
    except Exception:
        return None
    value = metadata.get("backbone")
    return str(value) if value else None


def _effective_device(providers: list[str]) -> str:
    if "CUDAExecutionProvider" in providers:
        return "CUDA"
    if "CPUExecutionProvider" in providers:
        return "CPU"
    return "unknown"


def preprocess_image(image: np.ndarray, target_size: int, max_size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    import cv2

    if not isinstance(image, np.ndarray):
        raise ValueError("RetinaFace input must be a NumPy image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RetinaFace input must be a BGR image with shape HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"RetinaFace input must have dtype uint8, got {image.dtype}")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("RetinaFace input image must not be empty")
    scale = float(target_size) / float(min(height, width))
    if round(scale * max(height, width)) > max_size:
        scale = float(max_size) / float(max(height, width))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pixels = resized.astype(np.float32) - BGR_MEAN
    tensor = np.transpose(pixels, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor, dtype=np.float32), scale, (resized_height, resized_width)


def generate_priors(image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    anchors: list[float] = []
    for min_sizes, step in zip(MIN_SIZES, STEPS):
        feature_height = math.ceil(height / step)
        feature_width = math.ceil(width / step)
        for row, column in product(range(feature_height), range(feature_width)):
            for min_size in min_sizes:
                anchors.extend(
                    (
                        (column + 0.5) * step / width,
                        (row + 0.5) * step / height,
                        min_size / width,
                        min_size / height,
                    )
                )
    return np.asarray(anchors, dtype=np.float32).reshape(-1, 4)


def decode_retinaface(
    locations: np.ndarray,
    scores: np.ndarray,
    landmarks: np.ndarray,
    *,
    resized_shape: tuple[int, int],
    original_shape: tuple[int, int],
    resize_scale: float | tuple[float, float],
    confidence_threshold: float,
    nms_threshold: float,
    top_k: int,
    keep_top_k: int,
    padding: tuple[int, int] = (0, 0),
) -> list[FaceDetection]:
    priors = generate_priors(resized_shape)
    locations = np.asarray(locations, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if locations.shape != (len(priors), 4):
        raise ValueError(f"RetinaFace location output {locations.shape} does not match {len(priors)} priors")
    if scores.shape != (len(priors), 2):
        raise ValueError(f"RetinaFace score output {scores.shape} does not match {len(priors)} priors")
    if landmarks.shape != (len(priors), 10):
        raise ValueError(f"RetinaFace landmark output {landmarks.shape} does not match {len(priors)} priors")

    boxes = np.concatenate(
        (
            priors[:, :2] + locations[:, :2] * VARIANCES[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(locations[:, 2:] * VARIANCES[1]),
        ),
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2.0
    boxes[:, 2:] += boxes[:, :2]
    resized_height, resized_width = resized_shape
    boxes *= np.asarray((resized_width, resized_height, resized_width, resized_height), dtype=np.float32)
    scale_x, scale_y = (resize_scale, resize_scale) if np.isscalar(resize_scale) else resize_scale
    offset = np.asarray(padding, dtype=np.float32)
    scale = np.asarray((scale_x, scale_y), dtype=np.float32)
    boxes = (boxes - np.tile(offset, 2)) / np.tile(scale, 2)
    points = priors[:, None, :2] + landmarks.reshape(-1, 5, 2) * VARIANCES[0] * priors[:, None, 2:]
    points = (points * [resized_width, resized_height] - offset) / scale
    confidence = scores[:, 1]
    valid = np.isfinite(confidence) & (confidence >= float(confidence_threshold))
    valid &= np.all(np.isfinite(boxes), axis=1)
    points = points[valid]
    boxes = boxes[valid]
    confidence = confidence[valid]
    if not len(boxes):
        return []
    order = confidence.argsort()[::-1][: int(top_k)]
    points = points[order]
    boxes = boxes[order]
    confidence = confidence[order]
    keep = _nms(boxes, confidence, float(nms_threshold))[: int(keep_top_k)]
    height, width = original_shape
    detections: list[FaceDetection] = []
    for index in keep:
        x1, y1, x2, y2 = boxes[index]
        x1 = float(np.clip(x1, 0.0, float(width)))
        y1 = float(np.clip(y1, 0.0, float(height)))
        x2 = float(np.clip(x2, 0.0, float(width)))
        y2 = float(np.clip(y2, 0.0, float(height)))
        if x2 <= x1 or y2 <= y1:
            continue
        x = int(math.floor(x1))
        y = int(math.floor(y1))
        box_width = max(1, int(math.ceil(x2 - x1)))
        box_height = max(1, int(math.ceil(y2 - y1)))
        detections.append(FaceDetection((x, y, box_width, box_height), float(confidence[index]), None, points[index].astype(np.float32)))
    return detections


def select_primary_face(detections: list[FaceDetection], width: int, height: int) -> FaceDetection | None:
    if not detections:
        return None
    center_x = width / 2.0
    center_y = height / 2.0
    diagonal = max(1.0, math.hypot(width, height))

    def rank(detection: FaceDetection) -> float:
        x, y, box_width, box_height = detection.box
        area_score = (box_width * box_height) / max(1, width * height)
        distance = math.hypot(x + box_width / 2.0 - center_x, y + box_height / 2.0 - center_y)
        center_score = 1.0 - min(1.0, distance / diagonal)
        return area_score * 4.0 + center_score * 0.55 + float(detection.score) * 0.25

    return max(detections, key=rank)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
        y1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
        x2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
        y2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= threshold]
    return keep


def _resolve_device(value: str):
    import torch

    normalized = str(value or "auto").lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported RetinaFace device: {value!r}")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RetinaFace CUDA was requested but CUDA is not available")
    return torch.device(normalized)


def _checkpoint_backbone(state_dict: dict[str, Any]) -> str:
    keys = set(state_dict)
    if "body.stage1.0.0.weight" in keys:
        return "mobilenet0.25"
    if "body.conv1.weight" in keys and any(key.startswith("body.layer4.") for key in keys):
        return "resnet50"
    raise ValueError("Unsupported RetinaFace checkpoint backbone")


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"RetinaFace model not found: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"RetinaFace model is not readable: {path}") from exc
    return digest.hexdigest()


def _positive_int(value: int, name: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least one")
    return result


def _unit_interval(value: float, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


__all__ = [
    "RetinaFaceRknnDetector",
    "RetinaFaceOnnxDetector",
    "RetinaFacePytorchDetector",
    "decode_retinaface",
    "generate_priors",
    "preprocess_image",
    "select_primary_face",
]

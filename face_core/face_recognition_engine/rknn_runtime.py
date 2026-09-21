"""Thread-safe, dependency-gated runtime support for RKNN Lite2 models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Callable, Literal

import numpy as np

from face_api.core.performance_logging import measure_face_stage

# auto 系统决定
# 0 只用Core0
# 0_1_2 全部使用
CORE_MASK_NAMES = {"auto", "0", "1", "2", "0_1", "0_1_2"}

# 允许的Tensor类型
MANIFEST_TENSOR_DTYPES = {
    "float32", "float16", "int8", "uint8", "int16", "uint16", "int32", "uint32"
}

# RKNN输入信息
@dataclass(frozen=True)
class RknnInputSpec:
    name: str # 输入节点名字
    layout: Literal["nchw", "nhwc"] # 数据排列
    shape: tuple[int, int, int, int] # 输入尺寸
    dtype: np.dtype # 数据类型

# 描述输出Tensor
@dataclass(frozen=True)
class RknnOutputSpec:
    # 这里只用于：诊断。
    """Manifest-probed output tensor metadata used only for runtime diagnostics."""

    name: str
    layout: Literal["nchw", "nhwc", "nc1hwc2", "undefined"]
    shape: tuple[int, ...]
    dtype: np.dtype

# 创建RKNN运行对象
def _create_rknn() -> Any:
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError(
            "RKNN runtime requires rknn-toolkit-lite2==2.3.2"
        ) from exc
    return RKNNLite()


# 核心分配


def is_compatible_rknn_input_dtype(value_dtype: np.dtype, input_spec: RknnInputSpec) -> bool:
    """Return whether a NumPy input dtype is acceptable for the probed RKNN input."""
    dtype = np.dtype(value_dtype)
    target_dtype = input_spec.dtype
    return dtype == target_dtype or (dtype == np.dtype("uint8") and target_dtype == np.dtype("int8"))


def coerce_rknn_input_dtype(array: np.ndarray, input_spec: RknnInputSpec) -> np.ndarray:
    """Return a contiguous image tensor acceptable to RKNNLite for this input."""
    value = np.ascontiguousarray(array)
    if is_compatible_rknn_input_dtype(value.dtype, input_spec):
        return value
    raise ValueError(f"RKNN input dtype {value.dtype} cannot be used with {input_spec.dtype}")

def _core_mask(rknn: Any, requested: str) -> int | None:
    if requested not in CORE_MASK_NAMES:
        raise ValueError(f"Unknown RKNN core_mask: {requested!r}")
    if requested == "0":
        return rknn.NPU_CORE_0
    if requested == "1":
        return rknn.NPU_CORE_1
    if requested == "2":
        return rknn.NPU_CORE_2
    if requested == "0_1":
        return rknn.NPU_CORE_0_1
    if requested == "0_1_2":
        return rknn.NPU_CORE_0_1_2
    return None

# 封装加载模型，初始化NPU，推理，释放资源
class RknnModelSession:
    def __init__(
        self,
        model_path: Path,
        input_spec: RknnInputSpec,
        core_mask: str = "auto",
        rknn_factory: Callable[[], Any] | None = None,
        *,
        output_specs: tuple[RknnOutputSpec, ...] = (),
    ) -> None:
        self.model_path = Path(model_path) # # 模型路径
        self.input_spec = input_spec # # 输入规格
        self.output_specs = tuple(output_specs) # # 输出规格
        self.core_mask = core_mask
        self._lock = threading.RLock()# 创建锁，保证infer()不会被多个线程同时执行。
        self._closed = False # 模型是否释放。

        # 检查core参数
        if core_mask not in CORE_MASK_NAMES:
            raise ValueError(f"Unknown RKNN core_mask: {core_mask!r}")
        # 检查模型文件
        if self.model_path.suffix != ".rknn" or not self.model_path.is_file():
            raise ValueError(f"RKNN model file does not exist: {self.model_path}")

        # 创建RKNN
        self._rknn = rknn_factory() if rknn_factory is not None else _create_rknn()
        try:
            selected_core_mask = _core_mask(self._rknn, core_mask)
            load_code = self._rknn.load_rknn(str(self.model_path))
            if load_code != 0:
                raise RuntimeError(
                    f"RKNN model load failed for {self.model_path}: return code {load_code}"
                )
            init_code = (
                self._rknn.init_runtime()
                if selected_core_mask is None
                else self._rknn.init_runtime(core_mask=selected_core_mask)
            )
            if init_code != 0:
                raise RuntimeError(
                    f"RKNN runtime initialization failed for {self.model_path}: return code {init_code}"
                )
        except Exception:
            self.close()
            raise

    def infer(self, array: np.ndarray) -> list[np.ndarray]:
        # 保证连续内存
        value = np.ascontiguousarray(array)
        # 检查输入格式是否匹配
        if tuple(value.shape) != self.input_spec.shape:
            raise ValueError(
                f"RKNN input shape {tuple(value.shape)} does not match {self.input_spec.shape}"
            )
        # 检查dtype是否兼容。RKNN probe 可能报告 int8 内部量化输入，
        # 但 RKNNLite 图像推理仍可以接收 uint8 原始像素并按模型 mean/std 处理。
        if not is_compatible_rknn_input_dtype(value.dtype, self.input_spec):
            raise ValueError(
                f"RKNN input dtype {value.dtype} does not match {self.input_spec.dtype}"
            )
        # Keep the exported manifest contract at the public boundary. Lite2
        # expects 4-D image buffers in NHWC on RK3588; transpose the actual
        # pixels as well as the format tag, preserving dtype and normalization.
        if self.input_spec.layout == "nchw":
            value = np.ascontiguousarray(value.transpose(0, 2, 3, 1))
        # 一个时间只有一个线程调用NPU。
        with self._lock:
            if self._closed:
                raise RuntimeError(f"RKNN model session is closed: {self.model_path}")
            with measure_face_stage("rknn_inference", model=self.model_path.name, core_mask=self.core_mask):
                outputs = self._rknn.inference(inputs=[value], data_format=["nhwc"])
            if not isinstance(outputs, (list, tuple)) or not outputs:
                raise RuntimeError(f"RKNN returned no output tensors: {self.model_path}")
            result = []
            for index, output in enumerate(outputs):
                value = np.asarray(output)
                if value.dtype.kind != "f" or not value.size or not np.isfinite(value).all():
                    raise RuntimeError(
                        f"RKNN output {index} must be non-empty, finite, dequantized floating point: {self.model_path}"
                    )
                # Own the result before another inference or close can use the session.
                result.append(np.array(value, dtype=np.float32, copy=True))
            return result

    # 返回模型信息
    def describe(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "input": {
                "name": self.input_spec.name,
                "layout": self.input_spec.layout,
                "shape": self.input_spec.shape,
                "dtype": self.input_spec.dtype.name,
            },
            "outputs": [
                {
                    "name": output.name,
                    "layout": output.layout,
                    "shape": list(output.shape),
                    "dtype": output.dtype.name,
                }
                for output in self.output_specs
            ],
            "core_mask": self.core_mask,
            "runtime_input_layout": "nhwc",
        }

    # 释放NPU
    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._rknn.release()
                self._closed = True


# 读取：manifest。
# 返回：输入信息。
def load_model_metadata(manifest_path: Path, model_path: Path) -> dict[str, Any]:
    """Load a model entry only after verifying its artifact hash."""
    return _load_verified_manifest_model(manifest_path, model_path)


def load_input_spec(manifest_path: Path, model_path: Path) -> RknnInputSpec:
    """Load a single RKNN input specification, preserving the legacy API."""
    return _parse_input_spec(_load_verified_manifest_model(manifest_path, model_path))

# 读取：manifest。
# 返回：输入信息，输出信息
def load_model_specs(
    manifest_path: Path, model_path: Path
) -> tuple[RknnInputSpec, tuple[RknnOutputSpec, ...]]:
    """Load manifest-probed input and output metadata for one verified artifact."""
    model = _load_verified_manifest_model(manifest_path, model_path)
    input_spec = _parse_input_spec(model)
    outputs = model.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("RKNN manifest model must contain at least one output")
    output_specs = tuple(_parse_output_spec(output) for output in outputs)
    return input_spec, output_specs



def _parse_input_spec(model: dict[str, Any]) -> RknnInputSpec:
    inputs = model.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], dict):
        raise ValueError("RKNN manifest model must contain exactly one input")
    input_data = inputs[0]
    try:
        name = input_data["name"]
        layout = input_data["layout"]
        raw_shape = input_data["shape"]
        dtype = np.dtype(input_data["dtype"])
    except (KeyError, TypeError) as exc:
        raise ValueError("RKNN manifest input is missing required fields") from exc
    if not isinstance(name, str):
        raise ValueError("RKNN manifest input name must be a string")
    if layout not in {"nchw", "nhwc"}:
        raise ValueError(f"Unsupported RKNN input layout: {layout!r}")
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 4
        or any(not isinstance(dimension, int) for dimension in raw_shape)
    ):
        raise ValueError("RKNN input shape must be a four-integer list")
    shape = tuple(raw_shape)
    if shape[0] != 1:
        raise ValueError(f"RKNN input shape must have batch size one: {shape}")
    return RknnInputSpec(name, layout, shape, dtype)


def _load_verified_manifest_model(manifest_path: Path, model_path: Path) -> dict[str, Any]:
    with Path(manifest_path).open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    try:
        models = manifest["models"]
    except (KeyError, TypeError) as exc:
        raise ValueError("RKNN manifest must contain a models list") from exc
    if not isinstance(models, list):
        raise ValueError("RKNN manifest models must be a list")

    target_path = Path(model_path).resolve()
    manifest_dir = Path(manifest_path).resolve().parent
    matches = []
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("rknn_path"), str):
            continue
        if (manifest_dir / model["rknn_path"]).resolve() == target_path:
            matches.append(model)
    if len(matches) != 1:
        raise ValueError(
            f"RKNN manifest must contain exactly one entry for model: {target_path}"
        )

    recorded_sha256 = matches[0].get("rknn_sha256")
    if not isinstance(recorded_sha256, str) or _sha256_file(target_path) != recorded_sha256.lower():
        raise ValueError(f"RKNN manifest SHA-256 does not match selected model: {target_path}")
    return matches[0]


def _parse_output_spec(output: Any) -> RknnOutputSpec:
    if not isinstance(output, dict):
        raise ValueError("RKNN manifest output must be an object")
    try:
        name = output["name"]
        layout = output["layout"]
        raw_shape = output["shape"]
        dtype = np.dtype(output["dtype"])
    except (KeyError, TypeError) as exc:
        raise ValueError("RKNN manifest output is missing required fields") from exc
    if not isinstance(name, str) or not name:
        raise ValueError("RKNN manifest output name must be a non-empty string")
    if layout not in {"nchw", "nhwc", "nc1hwc2", "undefined"}:
        raise ValueError(f"Unsupported RKNN output layout: {layout!r}")
    if dtype.name not in MANIFEST_TENSOR_DTYPES:
        raise ValueError(f"Unsupported RKNN output dtype: {dtype.name!r}")
    if (
        not isinstance(raw_shape, list)
        or not raw_shape
        or any(
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1
            for dimension in raw_shape
        )
    ):
        raise ValueError("RKNN manifest output shape must be a non-empty positive-integer list")
    return RknnOutputSpec(name, layout, tuple(raw_shape), dtype)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 模型启动之前检查配置是否正确
def validate_aligner_runtime_contract(
    manifest_path: Path,
    *,
    aligner_runtime: Literal["rknn", "onnx-cpu"],
    aligner_onnx_path: Path | None,
) -> None:
    """Reject an implicit/incorrect alignment fallback before model startup."""

    with Path(manifest_path).open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    runtime = manifest.get("runtime") if isinstance(manifest, dict) else None
    if not isinstance(runtime, dict) or runtime.get("aligner_runtime") != aligner_runtime:
        raise ValueError("RKNN manifest runtime.aligner_runtime does not match configured runtime")
    if aligner_runtime == "rknn":
        return
    if aligner_onnx_path is None:
        raise ValueError("ONNX aligner path is required for onnx-cpu runtime contract")
    configured = Path(aligner_onnx_path).resolve()
    recorded_path = runtime.get("aligner_onnx_path")
    if not isinstance(recorded_path, str) or (Path(manifest_path).resolve().parent / recorded_path).resolve() != configured:
        raise ValueError("RKNN manifest CPU aligner path does not match configured path")
    expected_hash = runtime.get("aligner_onnx_sha256")
    if not isinstance(expected_hash, str) or expected_hash.lower() != _sha256_file(configured):
        raise ValueError("RKNN manifest CPU aligner SHA-256 does not match configured model")

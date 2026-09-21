from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
PROVIDER_ENV = "FACEPROJECT_ONNX_PROVIDER"
_VALID_PROVIDER_MODES = {"auto", "cpu", "cuda"}

# 决定 ONNX Runtime 使用哪个执行设备。
def _execution_providers(mode: str | None = None, available: Iterable[str] | None = None) -> list[str]:
    """Return ONNX Runtime providers for auto/cuda/cpu deployment modes."""
    if available is None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency-gated.
            raise RuntimeError("onnxruntime is required for ONNX deployment.") from exc
        available = ort.get_available_providers()
    mode = str(mode or os.environ.get(PROVIDER_ENV, "auto")).lower()
    installed = list(available)

    if mode == "cpu":
        if CPU_PROVIDER not in installed:
            raise RuntimeError(f"{CPU_PROVIDER} is not available. Installed providers: {installed}")
        return [CPU_PROVIDER]

    if mode == "cuda":
        if CUDA_PROVIDER not in installed:
            raise RuntimeError(f"{CUDA_PROVIDER} is not available. Installed providers: {installed}")
        return [CUDA_PROVIDER, CPU_PROVIDER] if CPU_PROVIDER in installed else [CUDA_PROVIDER]

    if mode == "auto":
        if CUDA_PROVIDER in installed:
            return [CUDA_PROVIDER, CPU_PROVIDER] if CPU_PROVIDER in installed else [CUDA_PROVIDER]
        if CPU_PROVIDER in installed:
            return [CPU_PROVIDER]

    raise RuntimeError(f"No supported ONNX Runtime provider is available. Installed providers: {installed}")



# 提前加载CUDA库。
def _preload_cuda_dependencies(providers: Sequence[str], ort_module: object | None = None) -> bool:
    """Preload CUDA/cuDNN libraries installed by onnxruntime-gpu[cuda,cudnn]."""
    # 不支持gpu不加载
    if CUDA_PROVIDER not in providers:
        return False

    if ort_module is None:
        try:
            import onnxruntime as ort_module
        except ImportError as exc:  # pragma: no cover - dependency-gated.
            raise RuntimeError("onnxruntime is required for ONNX deployment.") from exc


    preload = getattr(ort_module, "preload_dlls", None)
    if not callable(preload):
        return False
    try:
        preload(directory="")
    except TypeError:
        preload()
    except Exception:
        return False
    return True


# 创建推理对象
def create_session(
    model_path: str | Path,
    provider_mode: str | None = None,
    session_options: object | None = None,
):
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - dependency-gated.
        raise RuntimeError("onnxruntime is required for ONNX deployment.") from exc

    providers = _execution_providers(provider_mode, ort.get_available_providers())
    _preload_cuda_dependencies(providers, ort)
    path = str(Path(model_path))
    if session_options is None:
        return ort.InferenceSession(path, providers=providers)
    return ort.InferenceSession(path, sess_options=session_options, providers=providers)



# 查看当前Session运行设备
def session_provider_names(session: object) -> list[str] | None:
    try:
        return list(session.get_providers())
    except Exception:
        return None

# 获取输入节点信息
def input_names(session: object) -> list[str]:
    return [item.name for item in session.get_inputs()]

# 获取输出节点信息
def output_names(session: object) -> Sequence[str] | None:
    try:
        return [item.name for item in session.get_outputs()]
    except Exception:
        return None



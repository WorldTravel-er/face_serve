from __future__ import annotations
"""
    根据 AI 推理框架提供的 Execution Provider 信息，判断当前模型运行在 GPU、CPU 还是未知设备
"""

from collections.abc import Iterable
from typing import Any

# GPU执行提供者列表
GPU_EXECUTION_PROVIDERS = {
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
}
# CPU推理
CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"

# 根据 provider 信息判断设备
def device_from_providers(providers: object) -> str:
    values = _provider_values(providers)
    if any(provider in GPU_EXECUTION_PROVIDERS for provider in values):
        return "GPU"
    if CPU_EXECUTION_PROVIDER in values:
        return "CPU"
    return "unknown"

# 生成显示字符串。
# 输出：det:GPU rec:CPU
def preview_device_label(runtime_info: dict[str, Any] | None) -> str:
    runtime = runtime_info or {}
    detector_device = device_from_providers(_providers_from(runtime.get("detector")))
    recognition_device = device_from_providers(_providers_from(runtime.get("engine")))
    return f"det:{detector_device} rec:{recognition_device}"


def _providers_from(value: object) -> object:
    if isinstance(value, dict):
        return value.get("providers", [])
    return []


def _provider_values(providers: object) -> list[str]:
    if isinstance(providers, str):
        return [providers]
    if isinstance(providers, Iterable):
        return [str(provider) for provider in providers]
    return []
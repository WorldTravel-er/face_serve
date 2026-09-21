"""Select the same detector backend for API, video and CLI entry points."""
from pathlib import Path
from typing import Any


def create_retinaface_detector(config: Any):
    from . import retinaface

    kwargs = dict(
        confidence_threshold=config.retinaface_conf, nms_threshold=config.retinaface_nms,
        target_size=config.retinaface_target_size, max_size=config.retinaface_max_size,
        top_k=config.retinaface_top_k, keep_top_k=config.retinaface_keep_top_k,
    )
    if config.runtime == "rknn":
        path = config.retinaface_model_path or config.retinaface_rknn_model_path
        if Path(path).suffix.lower() != ".rknn":
            raise ValueError("runtime=rknn requires a .rknn RetinaFace model")
        return retinaface.RetinaFaceRknnDetector(
            model_path=path, manifest_path=config.rknn_manifest_path,
            core_mask=config.rknn_detector_core_mask, **kwargs,
        )
    if config.runtime == "onnx":
        path = config.retinaface_model_path or config.retinaface_onnx_model_path
        if Path(path).suffix.lower() == ".pth":
            return retinaface.RetinaFacePytorchDetector(
                model_path=path, device=config.retinaface_device, **kwargs,
            )
        if Path(path).suffix.lower() != ".onnx":
            raise ValueError("runtime=onnx requires an ONNX detector or an explicit .pth checkpoint")
        return retinaface.RetinaFaceOnnxDetector(
            model_path=path, provider_mode=config.retinaface_provider or config.provider, **kwargs,
        )
    raise ValueError(f"Unknown inference runtime: {config.runtime!r}")

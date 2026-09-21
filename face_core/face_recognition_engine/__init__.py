from __future__ import annotations

# Unified face-engine package exports. The concrete production backend is CVLFace.
from .base import FaceEngine
from .face_detection import FaceDetection

__all__ = [
    "FaceEngine",
    "FaceOnnxEngine",
    "FaceRknnEngine",
]


def __getattr__(name: str):
    if name == "FaceOnnxEngine":
        from .face_onnx import FaceOnnxEngine
        globals()["FaceOnnxEngine"] = FaceOnnxEngine
        return FaceOnnxEngine


    if name == "FaceRknnEngine":
        from .face_rknn import FaceRknnEngine
        globals()["FaceRknnEngine"] = FaceRknnEngine
        return FaceRknnEngine


    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

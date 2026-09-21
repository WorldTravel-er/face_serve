from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from .face_detection import FaceDetection
import numpy as np


class FaceEngine(ABC):
    """人脸特征引擎抽象基类。
    上层流程只依赖这个抽象，不关心底层的人脸特征提取算法。
    只要具体实现能完成“裁剪、人脸特征提取、相似度计算、状态描述”这几件事，
    上层的视频处理、底库匹配和报告生成就可以复用同一套逻辑。
    """
    feature_dim = 512
    face_margin = 0.25

    def detect_faces(self, frame: np.ndarray) -> list[FaceDetection]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement face detection.")

    # 这里统一把检测框扩大一点再裁剪，尽量保留额头、下巴和侧边轮廓，
    # 让后续对齐和识别模型有更完整的人脸上下文。
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

    def feature_from_detection(self, frame: np.ndarray, detection: FaceDetection) ->np.ndarray:
        # 对外最常用的快捷路径：先按检测框裁剪，再进入具体后端做特征提取。
        return self.feature(self.crop_face(frame, detection.box))



    @abstractmethod
    def feature(self, face: np.ndarray) -> np.ndarray:
        # 子类只需要实现“给定一张人脸图，返回特征向量”。
        raise NotImplementedError


    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # 默认相似度采用余弦相似度的向量点积形式，并限制在 [-1, 1]。
        if a.size == 0 or b.size == 0 or a.shape != b.shape:
            return 0.0
        # 特征已经归一化，dot等同于计算余弦相似度，clip 限制范围（-1，1）
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    def describe(self) -> dict[str, Any]:
        # 给报告层返回一份可序列化的引擎元信息，便于验收时追踪当前用的到底是哪套模型。
        return {
            "feature_engine": self.__class__.__name__,
            "feature_dim": self.feature_dim,
        }


__all__ = ["FaceEngine"]

def alignment_fingerprint(model_path, *, runtime, face_margin, geometry_size=160):
    """Version the alignment model and geometry independently of recognition."""
    import hashlib
    import json
    from pathlib import Path
    with Path(model_path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    payload = {
        "aligner_sha256": digest, "aligner_runtime": runtime, "face_margin": float(face_margin),
        "geometry_size": geometry_size, "output_size": 112,
        "implementation": "cvlface_pil_input_align_corners_similarity_v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

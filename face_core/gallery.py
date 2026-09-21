from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .face_recognition_engine import FaceEngine
from .utils import read_json
from face_api.core.performance_logging import measure_face_stage
from .face_recognition_engine.face_detection import FaceDetector


@dataclass
class Person:
    """底库里的一名候选人员。"""

    person_id: str
    name: str
    id_card: str
    feature: np.ndarray
    image_count: int


class Gallery:
    """人员底库：把目录里的多张人脸图整理成可比对的特征集合。"""

    def __init__(self, people: list[Person]) -> None:
        if not people:
            raise ValueError("Gallery is empty. Add at least one person folder with face images.")
        self.people = people

    @classmethod
    def from_dir(cls, root: Path, engine: FaceEngine, detector: FaceDetector) -> "Gallery":
        # `gallery` 目录通常按人员分文件夹；如果根目录下直接放图片，也兼容把根目录当成一个人员目录。
        if not root.exists():
            raise FileNotFoundError(f"Gallery directory not found: {root}")

        people: list[Person] = []
        person_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
        if not person_dirs:
            person_dirs = [root]
        person_images = [(person_dir, list(_iter_images(person_dir))) for person_dir in person_dirs]
        total_images = sum(len(image_paths) for _, image_paths in person_images)
        print(
            "[gallery-build] start: "
            f"root={root}, person_dirs={len(person_images)}, images={total_images}",
            flush=True,
        )
        started_at = time.monotonic()
        progress_interval = 5.0
        next_progress_at = started_at + progress_interval
        processed_images = 0
        valid_features = 0

        def maybe_log_progress() -> None:
            nonlocal next_progress_at
            now = time.monotonic()
            if now >= next_progress_at:
                _print_progress(
                    "gallery-build",
                    processed_images,
                    total_images,
                    started_at,
                    extra=f"people={len(people)} valid_features={valid_features}",
                )
                next_progress_at = now + progress_interval

        for person_dir, image_paths in person_images:
            info_path = person_dir / "info.json"
            info = read_json(info_path) if info_path.exists() else {}
            person_id = str(info.get("person_id") or person_dir.name)
            name = str(info.get("name") or person_id)
            id_card = str(info.get("id_card") or "")
            features: list[np.ndarray] = []

            # 同一个人可能有多张照片；每张图片先检测人脸，再提特征，最后汇总成一个稳定向量。
            for image_path in image_paths:
                processed_images += 1
                image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"[gallery-build] skip: image={image_path} reason=decode_failed", flush=True)
                    maybe_log_progress()
                    continue
                feature = _extract_gallery_feature(image, engine, detector)
                if feature is not None and np.isfinite(feature).all() and np.linalg.norm(feature) > 1e-8:
                    features.append(feature)
                    valid_features += 1
                else:
                    reason = "face_not_detected" if feature is None else "invalid_feature"
                    print(f"[gallery-build] skip: image={image_path} reason={reason}", flush=True)
                maybe_log_progress()

            if features:
                feature = np.mean(np.vstack(features), axis=0)
                feature = feature / max(1e-8, np.linalg.norm(feature))
                people.append(Person(person_id, name, id_card, feature.astype(np.float32), len(features)))

        _print_progress(
            "gallery-build",
            processed_images,
            total_images,
            started_at,
            extra=f"people={len(people)} valid_features={valid_features}",
            done=True,
        )
        return cls(people)

    def match(self, feature: np.ndarray) -> tuple[Person, float]:
        # 视频帧特征和底库特征都已经归一化，直接点积等价于余弦相似度。
        scored = [(person, float(np.dot(person.feature, feature))) for person in self.people]
        return max(scored, key=lambda item: item[1])


def _extract_gallery_feature(image: np.ndarray, engine: FaceEngine, detector: FaceDetector) -> np.ndarray | None:
    with measure_face_stage("detect", phase="gallery"):
        detection = detector.detect_largest(image)
    if detection is not None:
        return engine.feature_from_detection(image, detection)
    return None


def _iter_images(path: Path):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix.lower() in suffixes and child.name != "info.json":
            yield child


def _print_progress(
    stage: str,
    current: int,
    total: int,
    started_at: float,
    extra: str = "",
    done: bool = False,
) -> None:
    elapsed = max(0.0, time.monotonic() - started_at)
    rate = current / elapsed if elapsed > 0 and current > 0 else 0.0
    label = "done" if done else "progress"
    extra_text = f", {extra}" if extra else ""
    if total > 0:
        safe_current = min(current, total)
        percent = safe_current / total * 100.0
        remaining = max(0, total - safe_current)
        eta = remaining / rate if rate > 0 and remaining > 0 else 0.0
        print(
            f"[{stage}] {label}: {safe_current}/{total} images ({percent:.1f}%), "
            f"elapsed {_format_duration(elapsed)}, eta {_format_duration(eta)}, "
            f"speed {rate:.2f} images/s{extra_text}",
            flush=True,
        )
    else:
        print(
            f"[{stage}] {label}: {current} images, elapsed {_format_duration(elapsed)}, "
            f"speed {rate:.2f} images/s{extra_text}",
            flush=True,
        )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

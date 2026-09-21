#!/usr/bin/env python3
"""Batch test: cropped face -> CVLFace DFA alignment -> AdaFace embedding."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_core.face_recognition_engine.face_onnx import FaceOnnxEngine, align_image_and_keypoints

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LANDMARK_NAMES = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")
LANDMARK_COLORS = ((0, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 0, 255))


def arguments() -> argparse.Namespace:
    models = ROOT / "models" / "onnx"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "test_data")
    parser.add_argument("--aligner", type=Path, default=models / "cvlface_dfa_mobilenet.onnx")
    parser.add_argument("--recognizer", type=Path, default=models / "cvlface_adaface_ir50_webface4m.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "onnx_align_recognize")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--align-score-threshold", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all images")
    parser.add_argument("--no-save-images", action="store_true")
    return parser.parse_args()


def model_info(session: Any) -> dict[str, Any]:
    def node(item: Any) -> dict[str, Any]:
        return {"name": item.name, "shape": list(item.shape), "type": item.type}
    return {"providers": list(session.get_providers()),
            "inputs": [node(x) for x in session.get_inputs()],
            "outputs": [node(x) for x in session.get_outputs()]}


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"cannot encode image: {path}")
    encoded.tofile(path)


def safe_name(path: Path, data: Path) -> str:
    base = data if data.is_dir() else data.parent
    try:
        relative = path.relative_to(base)
    except ValueError:
        relative = Path(path.name)
    return "__".join(relative.with_suffix("").parts)


def mark_keypoints(image: np.ndarray, normalized_points: np.ndarray, score: float | None) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    points = np.asarray(normalized_points, dtype=np.float32).reshape(5, 2)
    pixel_points = points * np.array([width, height], dtype=np.float32)
    pixel_points[:, 0] = np.clip(pixel_points[:, 0], 0, width - 1)
    pixel_points[:, 1] = np.clip(pixel_points[:, 1], 0, height - 1)
    marked = image.copy()
    scale = max(0.5, min(width, height) / 500.0)
    radius = max(3, round(min(width, height) / 140.0))
    thickness = max(1, radius // 2)
    for index, (point, color) in enumerate(zip(pixel_points, LANDMARK_COLORS)):
        x, y = np.rint(point).astype(int)
        cv2.circle(marked, (x, y), radius + thickness, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(marked, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.putText(marked, str(index), (x + radius + 2, y - radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    label = "DFA keypoints" if score is None else f"DFA score: {score:.4f}"
    cv2.putText(marked, label, (12, max(28, round(30 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return marked, pixel_points


def main() -> int:
    args = arguments()
    for key in ("aligner", "recognizer"):
        path = getattr(args, key).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{key} model not found: {path}")
        setattr(args, key, path)
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if not args.data.exists() or args.limit < 0:
        raise ValueError("invalid data path or limit")
    paths = ([args.data] if args.data.is_file() else
             sorted(p for p in args.data.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES))
    paths = paths[:args.limit] if args.limit else paths
    if not paths:
        raise FileNotFoundError(f"no images found under {args.data}")

    outputs = {name: args.output / name for name in ("keypoints", "aligned", "embeddings")}
    for directory in [args.output, *outputs.values()]:
        directory.mkdir(parents=True, exist_ok=True)
    engine = FaceOnnxEngine(recognition_model_path=args.recognizer,
        aligner_model_path=args.aligner, align_score_threshold=args.align_score_threshold,
        provider_mode=args.provider)
    signatures = {"aligner": model_info(engine.aligner_session),
                  "recognizer": model_info(engine.recognition.session)}
    for name, value in signatures.items():
        print(f"[{name}] {value}")

    records: list[dict[str, Any]] = []
    vectors: list[tuple[str, np.ndarray]] = []
    timings: dict[str, list[float]] = {"alignment": [], "recognition": []}
    succeeded = failures = 0
    for number, path in enumerate(paths, 1):
        record: dict[str, Any] = {"image": str(path), "error": None}
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            record["error"] = "image_decode_failed"
            failures += 1
            records.append(record)
            continue
        stem = safe_name(path, args.data)
        try:
            tensor = engine._bgr_to_input(image)
            started = time.perf_counter()
            raw_outputs = engine.aligner_session.run(None, {engine.aligner_input: tensor})
            original_points = engine._named_or_indexed_output(raw_outputs, "original_keypoints", 1)
            align_score = engine._score_to_float(
                engine._named_or_indexed_output(raw_outputs, "align_score", 3, required=False))
            aligned, aligned_points = align_image_and_keypoints(
                tensor, original_points, input_size=engine.aligner_input_size,
                output_size=engine.aligner_output_size)
            timings["alignment"].append((time.perf_counter() - started) * 1000)
            if align_score is not None and align_score < args.align_score_threshold:
                raise ValueError(f"align score {align_score:.6f} below threshold {args.align_score_threshold:.6f}")

            started = time.perf_counter()
            vector = engine.recognition.extract(aligned, aligned_points)
            timings["recognition"].append((time.perf_counter() - started) * 1000)
            norm = float(np.linalg.norm(vector))
            if vector.shape != (512,) or not np.isfinite(vector).all() or not np.isclose(norm, 1.0, atol=1e-4):
                raise ValueError(f"invalid embedding: shape={vector.shape}, norm={norm}")
            embedding_path = outputs["embeddings"] / f"{stem}.npy"
            np.save(embedding_path, vector)

            marked, pixel_points = mark_keypoints(image, original_points[0], align_score)
            keypoint_path = aligned_path = None
            if not args.no_save_images:
                keypoint_path = outputs["keypoints"] / f"{stem}.jpg"
                aligned_path = outputs["aligned"] / f"{stem}.jpg"
                save_image(keypoint_path, marked)
                save_image(aligned_path, engine._tensor_to_bgr(aligned))
            record.update({"align_score": align_score,
                "original_keypoints_normalized": original_points[0],
                "original_keypoints_pixels": pixel_points,
                "landmark_names": LANDMARK_NAMES,
                "aligned_keypoints_normalized": aligned_points[0],
                "keypoint_image": keypoint_path, "aligned_image": aligned_path,
                "embedding": embedding_path, "embedding_dimension": 512, "embedding_norm": norm})
            vectors.append((stem, vector))
            succeeded += 1
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            failures += 1
        records.append(record)
        print(f"[{number}/{len(paths)}] {path}: {'OK' if record['error'] is None else record['error']}")

    with (args.output / "similarities.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left", "right", "cosine_similarity"])
        for index, (left_name, left) in enumerate(vectors):
            for right_name, right in vectors[index + 1:]:
                writer.writerow([left_name, right_name, f"{float(left @ right):.8f}"])
    summary = {"images": len(paths), "succeeded": succeeded, "failures": failures,
        "average_ms": {name: (float(np.mean(values)) if values else None)
                       for name, values in timings.items()}}
    report = {"models": {"aligner": args.aligner, "recognizer": args.recognizer},
        "model_signatures": signatures, "parameters": vars(args), "summary": summary, "results": records}
    report_path = args.output / "results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_value) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report: {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

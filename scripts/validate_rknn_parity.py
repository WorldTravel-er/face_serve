"""Compare the explicit ONNX and RKNN face-recognition pipelines.

This command is deliberately import-safe on a conversion host: it does not
load a model or import RKNN Lite2 until ``main`` has parsed its arguments and
started a validation run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


THRESHOLDS = {"detection_iou_rate": 0.98, "feature_cosine_rate": 0.95, "top1_match_rate": 0.98}


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Return IoU for ``(x, y, width, height)`` boxes."""

    ax, ay, aw, ah = (float(value) for value in a)
    bx, by, bw, bh = (float(value) for value in b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def require_thresholds(metrics: dict[str, float]) -> None:
    for key, threshold in THRESHOLDS.items():
        if metrics[key] < threshold:
            raise SystemExit(f"{key}={metrics[key]:.4f} is below {threshold:.4f}")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _failure(record: Mapping[str, Any], reason: str) -> dict[str, Any]:
    keys = (
        "image_path", "onnx_box", "rknn_box", "iou", "feature_cosine",
        "onnx_top1_subject_id", "rknn_top1_subject_id", "error",
    )
    result = {"reason": reason}
    result.update({key: _json_value(record[key]) for key in keys if key in record and record[key] is not None})
    return result


def summarize_comparisons(comparisons: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate rates with distinct denominators and preserve every failure."""

    records = list(comparisons)
    criteria = (
        ("detection", "detection_iou_rate", "detections"),
        ("feature", "feature_cosine_rate", "features"),
        ("top1", "top1_match_rate", "top1"),
    )
    metrics: dict[str, float] = {}
    denominators: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    for prefix, metric_name, denominator_name in criteria:
        eligible = [record for record in records if record.get(f"{prefix}_eligible", False)]
        denominators[denominator_name] = len(eligible)
        passes = sum(bool(record.get(f"{prefix}_pass", False)) for record in eligible)
        metrics[metric_name] = passes / len(eligible) if eligible else 0.0
        failures.extend(_failure(record, f"{prefix}_mismatch") for record in eligible if not record.get(f"{prefix}_pass", False))
        if not eligible:
            failures.append({"reason": f"no_eligible_{denominator_name}"})
    inference_failures = [record for record in records if record.get("error")]
    failures.extend(_failure(record, "inference_failure") for record in inference_failures)
    return {
        "metrics": metrics,
        "denominators": denominators,
        "inference_failure_count": len(inference_failures),
        "failures": failures,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_model(manifest: Mapping[str, Any], expected_name: str, selected_path: Path, manifest_dir: Path) -> Mapping[str, Any]:
    matches = [
        item for item in manifest.get("models", [])
        if isinstance(item, Mapping)
        and item.get("name") == expected_name
        and isinstance(item.get("rknn_path"), str)
        and (manifest_dir / item["rknn_path"]).resolve() == selected_path.resolve()
    ]
    if len(matches) != 1:
        raise ValueError(f"Selected {expected_name} RKNN model is not exactly one manifest artifact: {selected_path}")
    recorded_hash = matches[0].get("rknn_sha256")
    actual_hash = _sha256(selected_path)
    if not isinstance(recorded_hash, str) or recorded_hash.lower() != actual_hash:
        raise ValueError(f"Selected {expected_name} RKNN model SHA-256 does not match manifest: {selected_path}")
    return matches[0]


def model_context_from_manifest(
    manifest_path: Path,
    selected_rknn_paths: Mapping[str, Path],
    aligner_runtime: str,
    aligner_onnx_path: Path | None,
) -> dict[str, Any]:
    """Verify selected artifacts and return only models active in this run."""

    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("models"), list):
        raise ValueError("RKNN manifest must contain a models list")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("aligner_runtime") not in {"rknn", "onnx-cpu"}:
        raise ValueError("RKNN manifest must declare runtime.aligner_runtime")
    if runtime["aligner_runtime"] != aligner_runtime:
        raise ValueError(
            f"RKNN manifest aligner_runtime={runtime['aligner_runtime']!r} does not match selected {aligner_runtime!r}"
        )
    expected = {"rknn_yolo": "yolo", "rknn_recognition": "recognition"}
    if aligner_runtime == "rknn":
        expected["rknn_aligner"] = "aligner"
    if set(selected_rknn_paths) != set(expected):
        raise ValueError(f"Selected RKNN paths must be exactly {sorted(expected)}")
    hashes: dict[str, str] = {}
    active_models: dict[str, dict[str, str]] = {}
    role_names = {"rknn_yolo": "detector", "rknn_recognition": "recognition", "rknn_aligner": "aligner"}
    for key, name in expected.items():
        entry = _manifest_model(manifest, name, Path(selected_rknn_paths[key]), path.parent)
        hashes[key] = str(entry["rknn_sha256"])
        if key != "rknn_aligner":
            active_models[role_names[key]] = {
                "runtime": "rknn",
                "model_path": str(Path(selected_rknn_paths[key])),
                "model_sha256": hashes[key],
            }
    if aligner_runtime == "rknn":
        active_aligner = {
            "runtime": "rknn",
            "model_path": str(Path(selected_rknn_paths["rknn_aligner"])),
            "model_sha256": hashes["rknn_aligner"],
        }
    else:
        if aligner_onnx_path is None:
            raise ValueError("--aligner-onnx is required for onnx-cpu fallback")
        selected = Path(aligner_onnx_path).resolve()
        recorded_path = runtime.get("aligner_onnx_path")
        recorded_hash = runtime.get("aligner_onnx_sha256")
        if not isinstance(recorded_path, str) or Path(recorded_path).resolve() != selected:
            raise ValueError("Selected CPU aligner path does not match manifest runtime contract")
        actual_hash = _sha256(selected)
        if not isinstance(recorded_hash, str) or recorded_hash.lower() != actual_hash:
            raise ValueError("Selected CPU aligner SHA-256 does not match manifest runtime contract")
        active_aligner = {"runtime": "onnx-cpu", "model_path": str(selected), "model_sha256": actual_hash}
    active_models["aligner"] = active_aligner
    return {"model_sha256": hashes, "active_aligner": active_aligner, "active_models": active_models}


def _load_image_paths(list_path: Path) -> list[Path]:
    paths = [Path(line.strip()) for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not paths:
        raise ValueError(f"Image list is empty: {list_path}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Image list has missing files: {missing[:5]}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    from face_api.core.config import ApiConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Text file containing absolute validation image paths.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--onnx-gallery-data-dir", type=Path, required=True)
    parser.add_argument("--rknn-gallery-data-dir", type=Path, required=True)
    parser.add_argument("--recognition-onnx", type=Path, default=ApiConfig().recognition_model_path)
    parser.add_argument("--aligner-onnx", type=Path, default=ApiConfig().aligner_model_path)
    parser.add_argument("--yolo-onnx", type=Path, default=ApiConfig().yolo_model_path)
    parser.add_argument("--recognition-rknn", type=Path, default=ApiConfig().recognition_rknn_model_path)
    parser.add_argument("--aligner-rknn", type=Path, default=ApiConfig().aligner_rknn_model_path)
    parser.add_argument("--yolo-rknn", type=Path, default=ApiConfig().yolo_rknn_model_path)
    parser.add_argument("--rknn-manifest", type=Path, default=ApiConfig().rknn_manifest_path)
    parser.add_argument("--rknn-core-mask", choices=("auto", "0", "1", "2", "0_1_2"), default="auto")
    parser.add_argument("--rknn-aligner-fallback", choices=("error", "onnx-cpu"), default="error")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="cpu")
    return parser


def _config(args: argparse.Namespace, runtime: str):
    from face_api.core.config import ApiConfig

    return ApiConfig(
        runtime=runtime, recognition_model_path=args.recognition_onnx, aligner_model_path=args.aligner_onnx,
        yolo_model_path=args.yolo_onnx, recognition_rknn_model_path=args.recognition_rknn,
        aligner_rknn_model_path=args.aligner_rknn, yolo_rknn_model_path=args.yolo_rknn,
        rknn_manifest_path=args.rknn_manifest, rknn_core_mask=args.rknn_core_mask,
        rknn_aligner_fallback=args.rknn_aligner_fallback, provider=args.provider,
    )


def _top1(service: Any, feature: np.ndarray, candidates: list[Any]) -> str | None:
    matches = service.match_feature(feature, candidates, top_k=1)
    return matches[0]["subject_id"] if matches else None


def _compare_one(image_path: Path, onnx: Any, rknn: Any, onnx_candidates: list[Any], rknn_candidates: list[Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"image_path": str(image_path), "detection_eligible": False, "feature_eligible": False, "top1_eligible": False}
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        record["error"] = "image_decode_failed"
        return record
    try:
        onnx_detection = onnx.detect_frame(frame)
        rknn_detection = rknn.detect_frame(frame)
    except Exception as exc:
        record["error"] = f"detection_inference_failed: {type(exc).__name__}: {exc}"
        return record
    if onnx_detection is None or float(onnx_detection.score) < 0.30:
        return record
    record["detection_eligible"] = True
    record["onnx_box"] = tuple(int(value) for value in onnx_detection.box)
    if rknn_detection is None:
        record["detection_pass"] = False
        return record
    record["rknn_box"] = tuple(int(value) for value in rknn_detection.box)
    record["iou"] = box_iou(record["onnx_box"], record["rknn_box"])
    record["detection_pass"] = record["iou"] >= 0.90
    try:
        onnx_feature = onnx.extract_feature_from_detection(frame, onnx_detection)
        rknn_feature = rknn.extract_feature_from_detection(frame, rknn_detection)
    except Exception as exc:
        record["feature_eligible"] = True
        record["feature_pass"] = False
        record["error"] = f"feature_inference_failed: {type(exc).__name__}: {exc}"
        return record
    record["feature_eligible"] = True
    record["feature_cosine"] = float(np.clip(np.dot(onnx_feature, rknn_feature), -1.0, 1.0))
    record["feature_pass"] = record["feature_cosine"] >= 0.98
    record["top1_eligible"] = True
    try:
        record["onnx_top1_subject_id"] = _top1(onnx, onnx_feature, onnx_candidates)
        record["rknn_top1_subject_id"] = _top1(rknn, rknn_feature, rknn_candidates)
        record["top1_pass"] = (
            record["onnx_top1_subject_id"] is not None
            and record["onnx_top1_subject_id"] == record["rknn_top1_subject_id"]
        )
    except Exception as exc:
        record["top1_pass"] = False
        record["error"] = f"top1_inference_failed: {type(exc).__name__}: {exc}"
    return record


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    from face_api.core.recognition import RecognitionService
    from face_api.core.subject_store import SubjectStore

    image_paths = _load_image_paths(args.images)
    fallback = args.rknn_aligner_fallback
    selected_rknn = {"rknn_yolo": args.yolo_rknn, "rknn_recognition": args.recognition_rknn}
    if fallback == "error":
        selected_rknn["rknn_aligner"] = args.aligner_rknn
    rknn_context = model_context_from_manifest(
        args.rknn_manifest,
        selected_rknn,
        "onnx-cpu" if fallback == "onnx-cpu" else "rknn",
        args.aligner_onnx,
    )
    model_hashes = {
        "onnx_yolo": _sha256(args.yolo_onnx),
        "onnx_recognition": _sha256(args.recognition_onnx),
        "onnx_aligner": _sha256(args.aligner_onnx),
        **rknn_context["model_sha256"],
    }
    onnx, rknn = RecognitionService(_config(args, "onnx")), RecognitionService(_config(args, "rknn"))
    try:
        onnx_candidates = SubjectStore(args.onnx_gallery_data_dir).list_feature_records(onnx.embedding_identity())
        rknn_candidates = SubjectStore(args.rknn_gallery_data_dir).list_feature_records(rknn.embedding_identity())
        comparisons = [_compare_one(path, onnx, rknn, onnx_candidates, rknn_candidates) for path in image_paths]
    finally:
        onnx.close()
        rknn.close()
    summary = summarize_comparisons(comparisons)
    # Keep hashes beside every failure as well as at report level.  Reports are
    # often split by a log collector, so a detached failure must remain
    # attributable to the exact six-model set that produced it.
    for failure in summary["failures"]:
        failure["model_sha256"] = model_hashes
    return {
        "runtime": "onnx_vs_rknn", "samples": len(comparisons), "model_sha256": model_hashes,
        "active_aligner": rknn_context["active_aligner"], **summary, "comparisons": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_paths = {
        "onnx_yolo": str(args.yolo_onnx), "onnx_recognition": str(args.recognition_onnx),
        "onnx_aligner": str(args.aligner_onnx), "rknn_yolo": str(args.yolo_rknn),
        "rknn_recognition": str(args.recognition_rknn),
    }
    if args.rknn_aligner_fallback != "onnx-cpu":
        selected_paths["rknn_aligner"] = str(args.aligner_rknn)
    report: dict[str, Any]
    try:
        report = run_validation(args)
    except Exception as exc:
        startup_hashes = {}
        for name, raw_path in selected_paths.items():
            candidate = Path(raw_path)
            try:
                startup_hashes[name] = _sha256(candidate)
            except OSError:
                startup_hashes[name] = None
        report = {
            "runtime": "onnx_vs_rknn", "samples": 0, "selected_model_paths": selected_paths,
            "model_sha256": startup_hashes, "active_aligner_runtime": "onnx-cpu" if args.rknn_aligner_fallback == "onnx-cpu" else "rknn",
            "failures": [{"reason": "startup_failure", "error": f"{type(exc).__name__}: {exc}", "model_sha256": startup_hashes}],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"validation failed before comparisons: {exc}", file=sys.stderr)
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_value) + "\n", encoding="utf-8")
    try:
        require_thresholds(report["metrics"])
        if report["inference_failure_count"]:
            raise SystemExit(f"inference_failure_count={report['inference_failure_count']} must be zero")
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

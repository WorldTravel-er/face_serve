"""Measure end-to-end face-service latency with an explicit runtime and gallery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_rknn_parity import model_context_from_manifest


WARMUP_SAMPLES = 10
MIN_MEASURED_SAMPLES = 300


class BenchmarkInferenceError(RuntimeError):
    """A failed sample run with the exact stage-level evidence retained."""

    def __init__(self, failures: list[dict[str, Any]], *, npu_error_count: int) -> None:
        self.failures = failures
        self.npu_error_count = npu_error_count
        super().__init__(f"benchmark inference failures: {failures}")


def _required_number(report: dict[str, Any], field: str) -> float:
    """Read a finite numeric evidence field or stop at its actionable name."""
    value: Any = report
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            raise SystemExit(f"release evidence is missing {field}")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise SystemExit(f"release evidence has invalid {field}")
    return float(value)


def require_release_evidence(report: dict[str, Any]) -> None:
    """Refuse promotion unless a complete RKNN 10-minute benchmark passed.

    This is intentionally a pure report validator: collection, 24-hour canary,
    and traffic changes remain explicit operator actions on the target device.
    """
    if report.get("runtime") != "rknn":
        raise SystemExit("release evidence requires runtime == rknn")
    # End-to-end latency is the primary promotion measure, so its absence is
    # reported before ancillary evidence (and keeps the CLI diagnostic useful).
    for field in ("end_to_end_ms.p95", "detector_ms.p95", "recognition_ms.p95"):
        _required_number(report, field)
    samples = _required_number(report, "samples")
    if samples < MIN_MEASURED_SAMPLES:
        raise SystemExit(f"release evidence requires samples >= {MIN_MEASURED_SAMPLES}")
    npu_error_count = _required_number(report, "npu_error_count")
    if npu_error_count != 0:
        raise SystemExit("release evidence requires npu_error_count == 0")
    improvement = _required_number(report, "end_to_end_p95_improvement")
    if improvement < 0.50:
        raise SystemExit("release evidence requires end_to_end_p95_improvement >= 0.50")


def summarize_latency_ns(samples_ns: np.ndarray) -> dict[str, float]:
    values = np.asarray(samples_ns, dtype=np.int64)
    if values.size == 0 or np.any(values < 0):
        raise ValueError("Latency samples must be non-empty non-negative nanoseconds")
    milliseconds = values.astype(np.float64) / 1_000_000.0
    p50 = float(np.percentile(milliseconds, 50))
    p95 = float(np.percentile(milliseconds, 95))
    return {"p50": round(p50, 4), "p95": round(p95, 4), "fps": round(1000.0 / p50, 4) if p50 else 0.0}


def build_parser() -> argparse.ArgumentParser:
    from face_api.core.config import ApiConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gallery-data-dir", type=Path, required=True, help="Gallery rebuilt for --runtime.")
    parser.add_argument("--runtime", choices=("onnx", "rknn"), required=True)
    parser.add_argument("--samples", type=int, default=MIN_MEASURED_SAMPLES)
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
    parser.add_argument(
        "--onnx-baseline-report", type=Path,
        help="Completed ONNX-CPU benchmark report used to calculate RKNN p95 improvement.",
    )
    parser.add_argument(
        "--require-release-evidence", action="store_true",
        help="Exit nonzero unless the written RKNN report meets the promotion benchmark gate.",
    )
    return parser


def _config(args: argparse.Namespace):
    from face_api.core.config import ApiConfig

    return ApiConfig(
        runtime=args.runtime, recognition_model_path=args.recognition_onnx, aligner_model_path=args.aligner_onnx,
        yolo_model_path=args.yolo_onnx, recognition_rknn_model_path=args.recognition_rknn,
        aligner_rknn_model_path=args.aligner_rknn, yolo_rknn_model_path=args.yolo_rknn,
        rknn_manifest_path=args.rknn_manifest, rknn_core_mask=args.rknn_core_mask,
        rknn_aligner_fallback=args.rknn_aligner_fallback, provider=args.provider,
    )


def _load_frames(image_list: Path) -> list[np.ndarray]:
    paths = [Path(line.strip()) for line in image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not paths:
        raise ValueError("Benchmark image list is empty")
    frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if any(frame is None or frame.size == 0 for frame in frames):
        raise ValueError("Benchmark image list contains unreadable images")
    return frames


def _elapsed_ns(callable_: Any) -> tuple[Any, int]:
    started = time.perf_counter_ns()
    result = callable_()
    return result, time.perf_counter_ns() - started


def benchmark_model_context(args: argparse.Namespace) -> dict[str, Any]:
    """Return every model active in the benchmark, including CPU fallback."""
    if args.runtime == "rknn":
        selected_rknn = {"rknn_yolo": args.yolo_rknn, "rknn_recognition": args.recognition_rknn}
        if args.rknn_aligner_fallback == "error":
            selected_rknn["rknn_aligner"] = args.aligner_rknn
        return model_context_from_manifest(
            args.rknn_manifest,
            selected_rknn,
            "onnx-cpu" if args.rknn_aligner_fallback == "onnx-cpu" else "rknn",
            args.aligner_onnx,
        )
    import hashlib
    digest = hashlib.sha256(args.aligner_onnx.read_bytes()).hexdigest()
    return {
        "model_sha256": {},
        "active_aligner": {"runtime": "onnx", "model_path": str(args.aligner_onnx), "model_sha256": digest},
        "active_models": {
            "detector": {"runtime": "onnx", "model_path": str(args.yolo_onnx), "model_sha256": hashlib.sha256(args.yolo_onnx.read_bytes()).hexdigest()},
            "recognition": {"runtime": "onnx", "model_path": str(args.recognition_onnx), "model_sha256": hashlib.sha256(args.recognition_onnx.read_bytes()).hexdigest()},
            "aligner": {"runtime": "onnx", "model_path": str(args.aligner_onnx), "model_sha256": digest},
        },
    }


def _baseline_end_to_end_p95(path: Path) -> float:
    return onnx_baseline_provenance(path)["end_to_end_ms_p95"]


def onnx_baseline_provenance(path: Path) -> dict[str, Any]:
    """Return the resolved, content-addressed ONNX baseline used for promotion."""
    resolved_path = path.resolve()
    try:
        report_bytes = resolved_path.read_bytes()
        report = json.loads(report_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read ONNX baseline report {resolved_path}: {exc}") from exc
    if report.get("runtime") != "onnx":
        raise ValueError("ONNX baseline report must have runtime == onnx")
    p95 = _required_number(report, "end_to_end_ms.p95")
    if p95 <= 0:
        raise ValueError("ONNX baseline end_to_end_ms.p95 must be greater than zero")
    return {
        "report_path": str(resolved_path),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "end_to_end_ms_p95": p95,
    }


def benchmark_failure_report(runtime: str, error: Exception) -> dict[str, Any]:
    """Serialize an unsuccessful benchmark without discarding sample evidence."""
    if isinstance(error, BenchmarkInferenceError):
        return {
            "runtime": runtime,
            "npu_error_count": error.npu_error_count,
            "failures": error.failures,
        }
    return {
        "runtime": runtime,
        "failures": [{"reason": "benchmark_failed", "error": f"{type(error).__name__}: {error}"}],
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from face_api.core.recognition import RecognitionService
    from face_api.core.subject_store import SubjectStore

    if args.samples < MIN_MEASURED_SAMPLES:
        raise ValueError(f"--samples must be at least {MIN_MEASURED_SAMPLES}")
    frames = _load_frames(args.images)
    model_context = benchmark_model_context(args)
    service = RecognitionService(_config(args))
    try:
        # Candidates are deliberately loaded only for the active embedding
        # identity. A gallery from the other runtime is not comparable.
        candidates = SubjectStore(args.gallery_data_dir).list_feature_records(service.embedding_identity())
        detector, feature, recognition, end_to_end = [], [], [], []
        failures: list[dict[str, Any]] = []
        npu_error_count = 0
        total = WARMUP_SAMPLES + args.samples
        for index in range(total):
            frame = frames[index % len(frames)]
            try:
                detection, detect_ns = _elapsed_ns(lambda: service.detect_frame(frame))
            except Exception as exc:
                failures.append({"sample_index": index, "stage": "detector", "error": f"{type(exc).__name__}: {exc}"})
                npu_error_count += int(args.runtime == "rknn")
                continue
            if detection is None:
                failures.append({"sample_index": index, "stage": "detector", "error": "no_face_detected"})
                continue
            try:
                _, feature_ns = _elapsed_ns(lambda: service.extract_feature_from_detection(frame, detection))
            except Exception as exc:
                failures.append({"sample_index": index, "stage": "feature_extraction", "error": f"{type(exc).__name__}: {exc}"})
                npu_error_count += int(args.runtime == "rknn")
                continue
            try:
                _, recognize_ns = _elapsed_ns(lambda: service.recognize_detection(frame, detection, candidates, top_k=1))
            except Exception as exc:
                failures.append({"sample_index": index, "stage": "recognition", "error": f"{type(exc).__name__}: {exc}"})
                npu_error_count += int(args.runtime == "rknn")
                continue
            try:
                # End-to-end is measured separately (detect + recognize) rather
                # than summed, so recognition's internal feature extraction is
                # never double-counted in the reported pipeline latency.
                _, end_ns = _elapsed_ns(lambda: service.match_frame_with_detection(frame, candidates, top_k=1))
            except Exception as exc:
                failures.append({"sample_index": index, "stage": "end_to_end", "error": f"{type(exc).__name__}: {exc}"})
                npu_error_count += int(args.runtime == "rknn")
                continue
            if index >= WARMUP_SAMPLES:
                detector.append(detect_ns)
                feature.append(feature_ns)
                recognition.append(recognize_ns)
                end_to_end.append(end_ns)
        if failures:
            raise BenchmarkInferenceError(failures, npu_error_count=npu_error_count)
        if len(end_to_end) != args.samples:
            raise RuntimeError(f"only collected {len(end_to_end)} post-warmup samples")
        detector_stats, feature_stats = summarize_latency_ns(np.asarray(detector)), summarize_latency_ns(np.asarray(feature))
        recognition_stats, end_stats = summarize_latency_ns(np.asarray(recognition)), summarize_latency_ns(np.asarray(end_to_end))
        report = {
            "runtime": args.runtime, "samples": len(end_to_end),
            "detector_ms": {key: detector_stats[key] for key in ("p50", "p95")},
            "feature_extraction_ms": {key: feature_stats[key] for key in ("p50", "p95")},
            "recognition_ms": {key: recognition_stats[key] for key in ("p50", "p95")},
            "end_to_end_ms": {key: end_stats[key] for key in ("p50", "p95")},
            "fps": end_stats["fps"],
            "active_aligner": model_context["active_aligner"],
            "active_models": model_context["active_models"],
            "model_sha256": model_context["model_sha256"],
            "npu_error_count": npu_error_count,
        }
        if args.runtime == "rknn" and args.onnx_baseline_report is not None:
            baseline = onnx_baseline_provenance(args.onnx_baseline_report)
            baseline_p95 = baseline["end_to_end_ms_p95"]
            report["onnx_baseline"] = baseline
            report["end_to_end_p95_improvement"] = round(1.0 - (end_stats["p95"] / baseline_p95), 6)
        return report
    finally:
        service.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_benchmark(args)
    except Exception as exc:
        report = benchmark_failure_report(args.runtime, exc)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(str(exc), file=sys.stderr)
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.require_release_evidence:
        try:
            require_release_evidence(report)
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

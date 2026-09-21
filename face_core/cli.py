from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from pathlib import Path

from .gallery import Gallery
from .live_camera import run_live_camera
from .report import build_reports
from .utils import ensure_dir
from .video_processor import analyze_video, annotate_video


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def _env_value(name: str, default: str) -> str:
    return os.environ.get(name, default)

def _format_providers(providers: object) -> str:
    if isinstance(providers, (list, tuple)):
        return ", ".join(str(provider) for provider in providers)
    return str(providers or "unknown")


def _effective_device_from_providers(providers: object) -> str:
    values = providers if isinstance(providers, (list, tuple)) else []
    if "CUDAExecutionProvider" in values:
        return "CUDA"
    if "CPUExecutionProvider" in values:
        return "CPU"
    return "unknown"


def _print_runtime_summary(runtime: str, engine: object, detector: object) -> None:
    engine_info = engine.describe() if hasattr(engine, "describe") else {}
    detector_info = detector.describe() if hasattr(detector, "describe") else {}
    # The ByteTrack wrapper reports the tracker policy plus a nested detector entry.
    inner_detector = detector_info.get("detector")
    if isinstance(inner_detector, dict):
        detector_info = {**detector_info, **inner_detector}
    print(f"Runtime: {runtime}", flush=True)
    if runtime == "onnx":
        engine_providers = engine_info.get("providers", [])
        detector_providers = detector_info.get("providers", [])
        effective_providers = engine_providers or detector_providers
        print(f"ONNX provider mode: {engine_info.get('provider_mode', 'auto')}", flush=True)
        print(f"Face engine providers: {_format_providers(engine_providers)}", flush=True)
        print(f"Face detector providers: {_format_providers(detector_providers)}", flush=True)
        print(f"Effective device: {_effective_device_from_providers(effective_providers)}", flush=True)
        return

    print(f"Face engine device: {engine_info.get('device', 'unknown')}", flush=True)
    print(f"Face detector device: {detector_info.get('device', 'unknown')}", flush=True)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face recognition V2 with RetinaFace detection and CVLFace features.")
    parser.add_argument("--video", type=Path, default="data/videos/input_m.mp4", help="Input video path.")


    # parser.add_argument("--camera", default=0, help="OpenCV camera index or stream URL for live recognition mode.")
    parser.add_argument("--camera", default=None, help="OpenCV camera index or stream URL for live recognition mode.")
    parser.add_argument("--camera-width", type=int, default=None, help="Requested live camera capture width.")
    parser.add_argument("--camera-height", type=int, default=None, help="Requested live camera capture height.")
    parser.add_argument("--camera-fps", type=float, default=30, help="Requested live camera FPS and clip writer fallback FPS.")
    parser.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="auto", help="OpenCV camera backend. On Windows, auto tries DirectShow before MSMF for local cameras.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional maximum live camera runtime in seconds.")


    parser.add_argument("--record-pre-roll", type=float, default=1.5, help="Seconds of annotated frames to keep before a live face event.")
    parser.add_argument("--record-post-roll", type=float, default=2.0, help="Seconds to keep recording after a live face disappears.")
    parser.add_argument("--preview", default=True, help="Show annotated live camera preview; press q to exit.")
    # parser.add_argument("--preview", action="store_true", help="Show annotated live camera preview; press q to exit.")


    parser.add_argument("--gallery", type=Path, default="data/gallery", help="Gallery root with person folders/images.")
    parser.add_argument("--output", type=Path, default="outputs/input_m", help="Output directory.")
    parser.add_argument("--sample-fps", type=float, default=0, help="Frames sampled per second. Use 0 (default) to detect every source frame.")
    parser.add_argument("--min-similarity", type=float, default=0.1, help="Similarity threshold for segments.")
    parser.add_argument("--save-faces", type=int, default=80, help="Maximum low-confidence face crops to save.")
    parser.add_argument("--runtime", choices=("rknn", "onnx"), default=_env_value("FACE_API_RUNTIME", "rknn"), help="Inference runtime. Default rknn targets RK3588 RKNN Lite2; use onnx for Windows/local development without NPU.")
    parser.add_argument("--retinaface-model", type=Path, default=Path(os.environ["FACE_API_RETINAFACE_MODEL"]) if "FACE_API_RETINAFACE_MODEL" in os.environ else None, help="Explicit detector override; otherwise select the runtime-specific model.")
    parser.add_argument("--retinaface-rknn", type=Path, default=_env_path("FACE_API_RETINAFACE_RKNN", "models/rknn/retinaface-mobilenet0.25-480x720-int8.rknn"), help="Fixed-shape RKNN RetinaFace detector.")
    parser.add_argument("--retinaface-onnx", type=Path, default=_env_path("FACE_API_RETINAFACE_ONNX", "models/onnx/Retinaface_mobilenet0.25.onnx"), help="ONNX detector used with --runtime onnx.")
    parser.add_argument("--rknn-detector-core-mask", choices=("auto", "0", "1", "2", "0_1", "0_1_2"), default=_env_value("FACE_API_RKNN_DETECTOR_CORE_MASK", "1"), help="Detector NPU core mask, default Core0.")
    parser.add_argument("--retinaface-provider", choices=("auto", "cuda", "cpu"), default=_env_value("FACE_API_RETINAFACE_PROVIDER", "auto"), help="ONNX Runtime provider mode for RetinaFace.")
    parser.add_argument("--retinaface-conf", type=float, default=float(os.environ.get("FACE_API_RETINAFACE_CONF", "0.1")), help="RetinaFace candidate confidence threshold. 0.01 is the validated value: the primary pass already accepts weak faces, so no second pass is needed.")
    parser.add_argument("--retinaface-nms", type=float, default=float(os.environ.get("FACE_API_RETINAFACE_NMS", "0.40")), help="RetinaFace NMS threshold.")
    parser.add_argument("--retinaface-target-size", type=int, default=int(os.environ.get("FACE_API_RETINAFACE_TARGET_SIZE", "640")), help="RetinaFace shorter-side target size. 640 validated for 1280x720 video; raise for smaller distant faces.")
    parser.add_argument("--retinaface-max-size", type=int, default=int(os.environ.get("FACE_API_RETINAFACE_MAX_SIZE", "640")), help="RetinaFace longer-side size cap. At 640 a 1280x720 frame is fed as 640x360 and a 1920x1080 frame as 640x360.")
    parser.add_argument("--retinaface-top-k", type=int, default=int(os.environ.get("FACE_API_RETINAFACE_TOP_K", "5000")), help="Maximum candidates retained before RetinaFace NMS.")
    parser.add_argument("--retinaface-keep-top-k", type=int, default=int(os.environ.get("FACE_API_RETINAFACE_KEEP_TOP_K", "750")), help="Maximum RetinaFace detections retained after NMS.")
    parser.add_argument("--retinaface-max-faces", type=int, default=int(os.environ.get("FACE_API_RETINAFACE_MAX_FACES", "1")), help="Faces processed per frame. 1 keeps only the primary face; raise it to recognize several people per frame.")
    parser.add_argument("--retinaface-min-face-score", type=float, default=float(os.environ.get("FACE_API_RETINAFACE_MIN_FACE_SCORE", "0.05")), help="Score a candidate needs before it counts as a face. Frames whose only candidates fall below it are skipped.")
    parser.add_argument("--face-selection", choices=("largest", "confidence", "continuity"), default=_env_value("FACE_API_FACE_SELECTION", "largest"), help="Which face to follow per frame: largest = closest to camera (default), confidence = strongest detection, continuity = keep the previous subject.")

    parser.add_argument("--onnx-provider", choices=("auto", "cuda", "cpu"), default='auto', help="ONNX Runtime provider mode. Default auto prefers CUDAExecutionProvider and falls back to CPUExecutionProvider.")

    parser.add_argument("--cvlface-path", type=Path, default=Path("models/cvlface"), help="Local CVLface model root directory.")

    parser.add_argument("--cvlface-device", default="auto", help="PyTorch runtime CVLface device. ONNX runtime ignores this option.")
    parser.add_argument("--cvlface-face-margin", type=float, default=0.55, help="Margin added around the detected face before CVLFace alignment.")
    parser.add_argument("--cvlface-align-score-threshold", type=float, default=0.0, help="Minimum aligner score. Use 0 to disable filtering.")
    parser.add_argument("--cvlface-recognition-onnx", type=Path, default=_env_path("FACE_API_CVLFACE_RECOGNITION_ONNX", "models/onnx/cvlface_adaface_ir50_webface4m.onnx"), help="CVLFace recognition ONNX model path for ONNX Runtime. Can be overridden with FACE_API_CVLFACE_RECOGNITION_ONNX.")
    parser.add_argument("--cvlface-aligner-onnx", type=Path, default=_env_path("FACE_API_CVLFACE_ALIGNER_ONNX", "models/onnx/cvlface_dfa_mobilenet.onnx"), help="CVLFace aligner ONNX model path for ONNX Runtime. Can be overridden with FACE_API_CVLFACE_ALIGNER_ONNX.")
    parser.add_argument("--cvlface-recognition-rknn", type=Path, default=_env_path("FACE_API_CVLFACE_RECOGNITION_RKNN", "models/rknn/recognition-int8.rknn"), help="CVLFace recognition RKNN model path. Can be overridden with FACE_API_CVLFACE_RECOGNITION_RKNN.")
    parser.add_argument("--cvlface-aligner-rknn", type=Path, default=_env_path("FACE_API_CVLFACE_ALIGNER_RKNN", "models/rknn/aligner-int8.rknn"), help="CVLFace aligner RKNN model path. Can be overridden with FACE_API_CVLFACE_ALIGNER_RKNN.")
    parser.add_argument("--rknn-manifest", type=Path, default=_env_path("FACE_API_RKNN_MANIFEST", "models/rknn/manifest.json"), help="RKNN model manifest path. Can be overridden with FACE_API_RKNN_MANIFEST.")
    parser.add_argument("--rknn-core-mask", choices=("auto", "0", "1", "2", "0_1", "0_1_2"), default=_env_value("FACE_API_RKNN_CORE_MASK", "auto"), help="RKNN NPU core mask.")
    parser.add_argument("--rknn-recognition-core-mask", choices=("auto", "0", "1", "2", "0_1", "0_1_2"), default=os.environ.get("FACE_API_RKNN_RECOGNITION_CORE_MASK"), help="Recognition NPU core mask; default Core1.")
    parser.add_argument("--rknn-aligner-fallback", choices=("error", "onnx-cpu"), default=_env_value("FACE_API_RKNN_ALIGNER_FALLBACK", "onnx-cpu"), help="RKNN aligner fallback mode. Use onnx-cpu only with a manifest that records the CPU aligner contract.")
    parser.add_argument("--performance-log-enabled", action="store_true", help="Log detect, align and recognition stage latency.")
    parser.add_argument("--performance-log-file", type=Path, default=Path("logs/demo_performance.jsonl"))
    parser.add_argument("--performance-log-stdout", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _create_detector(args):
    from face_api.core.config import ApiConfig
    from .face_recognition_engine.face_detection.factory import create_retinaface_detector
    return create_retinaface_detector(ApiConfig(
        runtime=args.runtime, retinaface_model_path=args.retinaface_model,
        retinaface_rknn_model_path=args.retinaface_rknn,
        retinaface_onnx_model_path=args.retinaface_onnx,
        retinaface_provider=args.retinaface_provider,
        rknn_manifest_path=args.rknn_manifest, rknn_detector_core_mask=args.rknn_detector_core_mask,
        retinaface_conf=args.retinaface_conf, retinaface_nms=args.retinaface_nms,
        retinaface_target_size=args.retinaface_target_size, retinaface_max_size=args.retinaface_max_size,
        retinaface_top_k=args.retinaface_top_k, retinaface_keep_top_k=args.retinaface_keep_top_k,
    ))


def main() -> None:
    """Parse CLI arguments and run the face recognition demo."""

    args = build_parser().parse_args()

    from face_api.core.performance_logging import FacePerformanceLogger
    logger = FacePerformanceLogger(
        enabled=args.performance_log_enabled,
        stdout=args.performance_log_stdout,
        log_file=args.performance_log_file,
    )
    with logger.context(runtime=args.runtime, source="demo"):
        _run(args)


def _run(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output)
    engine = None
    raw_detector = None
    try:
        if args.runtime == "rknn":
            from .face_recognition_engine import FaceRknnEngine
            from .face_recognition_engine.rknn_runtime import validate_aligner_runtime_contract
    
            aligner_runtime = "onnx-cpu" if args.rknn_aligner_fallback == "onnx-cpu" else "rknn"
            validate_aligner_runtime_contract(
                args.rknn_manifest,
                aligner_runtime=aligner_runtime,
                aligner_onnx_path=args.cvlface_aligner_onnx,
            )
            print("Initializing RKNN Lite2 face engine...", flush=True)
            engine = FaceRknnEngine(
                args.cvlface_recognition_rknn,
                args.cvlface_aligner_rknn,
                manifest_path=args.rknn_manifest,
                aligner_runtime=aligner_runtime,
                aligner_onnx_model_path=args.cvlface_aligner_onnx,
                align_score_threshold=args.cvlface_align_score_threshold,
                core_mask=args.rknn_recognition_core_mask if args.rknn_recognition_core_mask is not None else (args.rknn_core_mask if args.rknn_core_mask != "auto" else "1"),
            )
        elif args.runtime == "onnx":
            from .face_recognition_engine import FaceOnnxEngine
    
            print("Initializing ONNX Runtime face engine...", flush=True)
            engine = FaceOnnxEngine(
                recognition_model_path=args.cvlface_recognition_onnx,
                aligner_model_path=args.cvlface_aligner_onnx,
                face_margin=args.cvlface_face_margin,
                align_score_threshold=args.cvlface_align_score_threshold,
                provider_mode=args.onnx_provider,
            )
        else:
            raise ValueError(f"Unknown inference runtime: {args.runtime!r}")
    
        print("Initializing RetinaFace detector...", flush=True)
        from face_api.video.face_tracker import ByteTrackFaceTracker
    
        raw_detector = _create_detector(args)
        detector = ByteTrackFaceTracker(
            raw_detector,
            high_threshold=max(0.05, args.retinaface_conf),
            min_face_area_ratio=0.001,
            max_face_area_ratio=0.30,
            min_face_aspect_ratio=0.30,
            max_face_aspect_ratio=1.35,
            max_candidates=25,
            min_face_score=args.retinaface_min_face_score,
            max_faces_per_frame=args.retinaface_max_faces,
            face_selection=args.face_selection,
        )
    
        _print_runtime_summary(args.runtime, engine, detector)
    
        print(f"Building gallery features from: {args.gallery}", flush=True)
        gallery = Gallery.from_dir(args.gallery, engine, raw_detector)
        print(f"Gallery loaded: {len(gallery.people)} people.", flush=True)
    
        if args.camera is not None:
            results, events = run_live_camera(
                camera_source=args.camera,
                output_dir=output_dir,
                engine=engine,
                gallery=gallery,
                detector=detector,
                min_similarity=args.min_similarity,
                save_faces=args.save_faces,
                camera_width=args.camera_width,
                camera_height=args.camera_height,
                camera_fps=args.camera_fps,
                camera_backend=args.camera_backend,
                record_pre_roll=args.record_pre_roll,
                record_post_roll=args.record_post_roll,
                preview=args.preview,
                max_seconds=args.max_seconds,
            )
            print("Live face recognition finished.")
            print(f"Output: {output_dir}")
            print(f"Recognized frames: {len(results)}")
            print(f"Captured events: {len(events)}")
            print(f"Events: {output_dir / 'live_events.json'}")
            print(f"Frame results: {output_dir / 'live_results.csv'}")
            return
    
        print(f"Analyzing video: {args.video}", flush=True)
        meta, results, segments = analyze_video(
            video_path=args.video,
            output_dir=output_dir,
            engine=engine,
            gallery=gallery,
            detector=detector,
            sample_fps=args.sample_fps,
            min_similarity=args.min_similarity,
            save_faces=args.save_faces,
        )
        print(len(results))
        print("Writing annotated video...", flush=True)
        annotated_video = annotate_video(args.video, output_dir, results, meta)
    
        report = build_reports(output_dir, args.video, meta, results, segments, annotated_video, engine.describe())
        rec = report["recognition"]
    
        print("Face recognition demo finished.")
        print(f"Output: {output_dir}")
        print(f"Best match: {rec['name']} {rec['id_card']}")
        print(f"Best similarity: {rec['best_raw_similarity']:.4f}")
        print(f"Segments: {len(report['segments'])}")
        print(f"Annotated video: {annotated_video}")
    finally:
        try:
            close = getattr(raw_detector, "close", None)
            if callable(close):
                close()
        finally:
            close = getattr(engine, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    main()

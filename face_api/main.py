from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from face_api.core.config import ApiConfig

RKNN_CORE_MASK_CHOICES = ("auto", "0", "1", "2", "0_1", "0_1_2")


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def _env_value(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_optional_value(name: str) -> str | None:
    return os.environ.get(name)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the face recognition REST API service.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8001, help="Bind port.")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto", help="ONNX Runtime provider mode.")
    parser.add_argument("--runtime", choices=("rknn", "onnx"), default=_env_value("FACE_API_RUNTIME", "rknn"), help="Inference runtime.")
    parser.add_argument("--retinaface-model", type=Path, default=Path(os.environ["FACE_API_RETINAFACE_MODEL"]) if "FACE_API_RETINAFACE_MODEL" in os.environ else None, help="Explicit detector override; otherwise select the runtime-specific model.")
    parser.add_argument("--retinaface-rknn", type=Path, default=_env_path("FACE_API_RETINAFACE_RKNN", "models/rknn/retinaface-mobilenet0.25-480x720-int8.rknn"), help="Fixed-shape RKNN RetinaFace detector.")
    parser.add_argument("--retinaface-onnx", type=Path, default=_env_path("FACE_API_RETINAFACE_ONNX", "models/onnx/Retinaface_mobilenet0.25.onnx"), help="ONNX detector used with --runtime onnx.")
    parser.add_argument("--rknn-detector-core-mask", choices=("auto", "0", "1", "2", "0_1", "0_1_2"), default=_env_value("FACE_API_RKNN_DETECTOR_CORE_MASK", "1"), help="Detector NPU core mask, default Core0.")
    parser.add_argument("--retinaface-provider", choices=("auto", "cuda", "cpu"), default=_env_value("FACE_API_RETINAFACE_PROVIDER", "auto"), help="ONNX RetinaFace provider.")
    parser.add_argument("--retinaface-device", choices=("auto", "cuda", "cpu"), default=_env_value("FACE_API_RETINAFACE_DEVICE", "auto"), help="PyTorch device for RetinaFace.")
    parser.add_argument("--retinaface-conf", type=float, default=float(os.environ.get("FACE_API_RETINAFACE_CONF", "0.2")), help="RetinaFace candidate confidence threshold. Validated default for input_m.mp4: 0.01.")
    parser.add_argument("--retinaface-nms", type=float, default=float(os.environ.get("FACE_API_RETINAFACE_NMS", "0.40")), help="RetinaFace NMS IoU threshold.")
    parser.add_argument("--retinaface-target-size", type=_positive_int, default=int(os.environ.get("FACE_API_RETINAFACE_TARGET_SIZE", "640")), help="RetinaFace shorter-side target size. 640 validated for 1280x720 video.")
    parser.add_argument("--retinaface-max-size", type=_positive_int, default=int(os.environ.get("FACE_API_RETINAFACE_MAX_SIZE", "640")), help="RetinaFace longer-side size cap. 640 feeds a 1280x720 frame as 640x360.")
    parser.add_argument("--retinaface-top-k", type=_positive_int, default=int(os.environ.get("FACE_API_RETINAFACE_TOP_K", "5000")), help="Maximum candidates retained before RetinaFace NMS.")
    parser.add_argument("--retinaface-keep-top-k", type=_positive_int, default=int(os.environ.get("FACE_API_RETINAFACE_KEEP_TOP_K", "750")), help="Maximum RetinaFace detections retained after NMS.")
    parser.add_argument("--retinaface-max-faces", type=_positive_int, default=int(os.environ.get("FACE_API_RETINAFACE_MAX_FACES", "1")), help="Faces processed per frame. 1 keeps only the primary face; raise it to recognize several people per frame.")
    parser.add_argument("--retinaface-min-face-score", type=float, default=float(os.environ.get("FACE_API_RETINAFACE_MIN_FACE_SCORE", "0.2")), help="Score a candidate needs before it counts as a face. Frames whose only candidates fall below it are skipped.")
    parser.add_argument("--face-selection", choices=("largest", "confidence", "continuity"), default=_env_value("FACE_API_FACE_SELECTION", "largest"), help="Which face to follow per frame: largest = closest to camera (default), confidence = strongest detection, continuity = keep the previous subject.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/face_api"), help="Subject database and image directory.")
    parser.add_argument("--cvlface-recognition-onnx", type=Path, default=_env_path("FACE_API_CVLFACE_RECOGNITION_ONNX", "models/onnx/cvlface_adaface_ir50_webface4m.onnx"), help="CVLFace recognition ONNX model path. Can be overridden with FACE_API_CVLFACE_RECOGNITION_ONNX.")
    parser.add_argument("--cvlface-aligner-onnx", type=Path, default=_env_path("FACE_API_CVLFACE_ALIGNER_ONNX", "models/onnx/cvlface_dfa_mobilenet.onnx"), help="CVLFace aligner ONNX model path. Can be overridden with FACE_API_CVLFACE_ALIGNER_ONNX.")
    parser.add_argument("--cvlface-recognition-rknn", type=Path, default=_env_path("FACE_API_CVLFACE_RECOGNITION_RKNN", "models/rknn/recognition-int8.rknn"), help="CVLFace recognition RKNN model path. Can be overridden with FACE_API_CVLFACE_RECOGNITION_RKNN.")
    parser.add_argument("--cvlface-aligner-rknn", type=Path, default=_env_path("FACE_API_CVLFACE_ALIGNER_RKNN", "models/rknn/aligner-int8.rknn"), help="CVLFace aligner RKNN model path. Can be overridden with FACE_API_CVLFACE_ALIGNER_RKNN.")
    parser.add_argument("--rknn-manifest", type=Path, default=_env_path("FACE_API_RKNN_MANIFEST", "models/rknn/manifest.json"), help="RKNN model manifest path. Can be overridden with FACE_API_RKNN_MANIFEST.")
    parser.add_argument("--rknn-core-mask", choices=RKNN_CORE_MASK_CHOICES, default=_env_value("FACE_API_RKNN_CORE_MASK", "auto"), help="Legacy RKNN NPU core mask used when the recognition mask is not set.")
    parser.add_argument("--rknn-recognition-core-mask", choices=RKNN_CORE_MASK_CHOICES, default=_env_optional_value("FACE_API_RKNN_RECOGNITION_CORE_MASK"), help="RKNN NPU core mask for the face recognition model. Defaults to core 1.")
    parser.add_argument("--rknn-aligner-fallback", choices=("onnx-cpu", "error"), default=_env_value("FACE_API_RKNN_ALIGNER_FALLBACK", "onnx-cpu"), help="RKNN aligner fallback mode.")

    parser.add_argument("--video-analysis-max-concurrency", type=_positive_int, default=1, help="Maximum concurrent long-video analysis tasks.")
    parser.add_argument("--video-analysis-recognition-interval", type=_positive_float, default=float(os.environ.get("FACE_API_VIDEO_ANALYSIS_RECOGNITION_INTERVAL", "0.1")), help="Seconds between queued nearest-track recognition attempts during long-video analysis.")
    parser.add_argument("--video-analysis-recognition-queue-size", type=_positive_int, default=int(os.environ.get("FACE_API_VIDEO_ANALYSIS_RECOGNITION_QUEUE_SIZE", "32")), help="Maximum pending nearest-track recognition jobs during long-video analysis.")
    parser.add_argument("--video-analysis-download-source", action="store_true", help="Download HTTP/HTTPS video sources before long-video analysis.")
    parser.add_argument("--live-stream-url", default="rtsp://192.168.3.78:8554/camera", help="Video source consumed by live recognition, such as a v4l2loopback /dev/video device, RTSP URL, video file, or camera index. Default: /dev/video10.")

    parser.add_argument("--live-video-stream-id", default="rtsp://192.168.3.78:8554/camera", help="video_stream_id field pushed to WebSocket clients; defaults to the live video source.")
    parser.add_argument("--live-match-threshold", type=float, default=0.3, help="Default live face match threshold.")
    parser.add_argument("--live-heartbeat-timeout-seconds", type=float, default=30.0, help="Seconds before closing WebSocket when client heartbeat is missing.")
    parser.add_argument("--live-detection-fps", type=float, default=30, help="Maximum live face detection FPS.")
    parser.add_argument("--live-detection-result-ttl", type=float, default=0.8, help="Seconds to keep the latest detection box valid for recognition/preview.")
    parser.add_argument("--live-recognition-fps", type=float, default=30, help="Maximum live identity recognition loop FPS.")
    parser.add_argument("--live-rtsp-transport", choices=("tcp", "udp", "auto"), default="tcp", help="RTSP transport used by OpenCV FFmpeg capture for RTSP video sources. Use auto to leave the environment unchanged.")
    parser.add_argument("--live-capture-reconnect-interval", type=float, default=1.0, help="Seconds between video source reconnect attempts.")
    parser.add_argument("--live-capture-max-consecutive-failures", type=int, default=20, help="Read failures before reporting stream_error and reconnecting.")
    parser.add_argument("--live-diagnostics-enabled", action="store_true", help="Write live detection and recognition diagnostics to a log file. Disabled by default.")
    parser.add_argument("--live-diagnostics-log-file", type=Path, default=Path("logs/live_recognition_diagnostics.log"), help="File path for live diagnostics when --live-diagnostics-enabled is set.")
    parser.add_argument("--live-diagnostics-throttle-seconds", type=float, default=1.0, help="Minimum seconds between live diagnostics log lines per event type.")
    parser.add_argument("--performance-log-enabled", action="store_true", help="Write REST, video analysis, and WebSocket face performance metrics as JSONL.")
    parser.add_argument("--performance-log-stdout", action=argparse.BooleanOptionalAction, default=True, help="Mirror enabled performance logs to stdout.")
    parser.add_argument("--performance-log-file", type=Path, default=Path("logs/rknn_performance.jsonl"), help="JSONL file path for face performance metrics.")
    parser.add_argument("--performance-log-fps-window-seconds", type=float, default=2.0, help="Sliding window used to compute recognition_fps in performance logs.")
    parser.add_argument("--performance-log-max-fps-meters", type=_positive_int, default=1024, help="Maximum active recognition_fps buckets kept for tasks and WebSocket connections.")

    parser.add_argument("--live-preview", action="store_true", help="Show a local OpenCV preview window with live recognition overlays.")
    parser.add_argument("--live-preview-window-name", default="Live Face Recognition", help="OpenCV window title for live preview.")
    parser.add_argument("--live-preview-width", type=int, default=None, help="Optional exact live preview display width. Prefer --live-preview-max-width for aspect-safe scaling.")
    parser.add_argument("--live-preview-height", type=int, default=None, help="Optional exact live preview display height. Prefer --live-preview-max-height for aspect-safe scaling.")
    parser.add_argument("--live-preview-fps", type=float, default=20.0, help="Maximum local preview display FPS.")
    parser.add_argument("--live-preview-max-width", type=int, default=None, help="Optional aspect-safe maximum preview display width.")
    parser.add_argument("--live-preview-max-height", type=int, default=None, help="Optional aspect-safe maximum preview display height.")
    parser.add_argument("--live-preview-result-ttl", type=float, default=1.5, help="Seconds to keep the last recognition overlay visible in preview.")
    return parser


def build_opencv_ffmpeg_capture_options(rtsp_transport: str) -> str | None:
    transport = str(rtsp_transport or "auto").lower()
    if transport == "auto":
        return None
    return f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay|max_delay;500000"


def is_rtsp_video_source(video_source: str | None) -> bool:
    source = str(video_source or "").lower()
    return source.startswith(("rtsp://", "rtsps://"))


def configure_live_capture_environment(rtsp_transport: str, video_source: str | None = None) -> None:
    if video_source is not None and not is_rtsp_video_source(video_source):
        return
    options = build_opencv_ffmpeg_capture_options(rtsp_transport)
    if options:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options


def _resolve_rknn_recognition_core_mask(
    legacy_core_mask: str,
    recognition_core_mask: str | None,
) -> str:
    legacy = legacy_core_mask or "auto"
    if recognition_core_mask is not None:
        return recognition_core_mask
    return legacy if legacy != "auto" else "1"


def _restore_standard_logging_levels() -> None:
    for level, name in (
        (logging.CRITICAL, "CRITICAL"),
        (logging.ERROR, "ERROR"),
        (logging.WARNING, "WARNING"),
        (logging.INFO, "INFO"),
        (logging.DEBUG, "DEBUG"),
        (logging.NOTSET, "NOTSET"),
    ):
        logging.addLevelName(level, name)
    logging.addLevelName(logging.CRITICAL, "FATAL")
    logging.addLevelName(logging.WARNING, "WARN")


def main() -> None:
    args = build_parser().parse_args()

    from face_api.app import create_app

    configure_live_capture_environment(args.live_rtsp_transport, args.live_stream_url)
    rknn_recognition_core_mask = _resolve_rknn_recognition_core_mask(
        args.rknn_core_mask,
        args.rknn_recognition_core_mask,
    )
    config = ApiConfig(
        runtime=args.runtime,
        retinaface_model_path=args.retinaface_model,
        retinaface_rknn_model_path=args.retinaface_rknn,
        retinaface_onnx_model_path=args.retinaface_onnx,
        retinaface_provider=args.retinaface_provider,
        rknn_detector_core_mask=args.rknn_detector_core_mask,
        retinaface_device=args.retinaface_device,
        retinaface_conf=args.retinaface_conf,
        retinaface_nms=args.retinaface_nms,
        retinaface_target_size=args.retinaface_target_size,
        retinaface_max_size=args.retinaface_max_size,
        retinaface_top_k=args.retinaface_top_k,
        retinaface_keep_top_k=args.retinaface_keep_top_k,
        max_faces_per_frame=args.retinaface_max_faces,
        min_face_score=args.retinaface_min_face_score,
        face_selection=args.face_selection,
        data_dir=args.data_dir,
        recognition_model_path=args.cvlface_recognition_onnx,
        aligner_model_path=args.cvlface_aligner_onnx,
        recognition_rknn_model_path=args.cvlface_recognition_rknn,
        aligner_rknn_model_path=args.cvlface_aligner_rknn,
        rknn_manifest_path=args.rknn_manifest,
        rknn_core_mask=args.rknn_core_mask,
        rknn_recognition_core_mask=rknn_recognition_core_mask,
        rknn_aligner_fallback=args.rknn_aligner_fallback,
        provider=args.provider,
        video_analysis_max_concurrency=args.video_analysis_max_concurrency,
        video_analysis_recognition_interval_seconds=args.video_analysis_recognition_interval,
        video_analysis_recognition_queue_size=args.video_analysis_recognition_queue_size,
        video_analysis_download_source_enabled=args.video_analysis_download_source,
        live_stream_url=args.live_stream_url,
        live_video_stream_id=args.live_video_stream_id,
        live_match_threshold=args.live_match_threshold,
        live_heartbeat_timeout_seconds=args.live_heartbeat_timeout_seconds,
        live_detection_fps=args.live_detection_fps,
        live_detection_result_ttl=args.live_detection_result_ttl,
        live_recognition_fps=args.live_recognition_fps,
        live_rtsp_transport=args.live_rtsp_transport,
        live_capture_reconnect_interval=args.live_capture_reconnect_interval,
        live_capture_max_consecutive_failures=args.live_capture_max_consecutive_failures,
        live_diagnostics_enabled=args.live_diagnostics_enabled,
        live_diagnostics_log_file=args.live_diagnostics_log_file,
        live_diagnostics_throttle_seconds=args.live_diagnostics_throttle_seconds,
        performance_log_enabled=args.performance_log_enabled,
        performance_log_stdout=args.performance_log_stdout,
        performance_log_file=args.performance_log_file,
        performance_log_fps_window_seconds=args.performance_log_fps_window_seconds,
        performance_log_max_fps_meters=args.performance_log_max_fps_meters,
        live_preview_enabled=args.live_preview,
        live_preview_window_name=args.live_preview_window_name,
        live_preview_width=args.live_preview_width,
        live_preview_height=args.live_preview_height,
        live_preview_fps=args.live_preview_fps,
        live_preview_max_width=args.live_preview_max_width,
        live_preview_max_height=args.live_preview_max_height,
        live_preview_result_ttl=args.live_preview_result_ttl,
    )
    app = create_app(config=config)
    _restore_standard_logging_levels()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

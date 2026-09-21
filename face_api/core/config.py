from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ApiConfig:
    runtime: Literal["onnx", "rknn"] = "rknn"
    data_dir: Path = Path("data/face_api")
    recognition_model_path: Path = Path("models/onnx/cvlface_adaface_ir50_webface4m.onnx")
    aligner_model_path: Path = Path("models/onnx/cvlface_dfa_mobilenet.onnx")
    retinaface_model_path: Path | None = None
    retinaface_rknn_model_path: Path = Path("models/rknn/retinaface-mobilenet0.25-480x720-int8.rknn")
    retinaface_onnx_model_path: Path = Path("models/onnx/Retinaface_mobilenet0.25.onnx")
    retinaface_provider: str | None = None
    rknn_detector_core_mask: str = "0"
    retinaface_device: str = "auto"
    retinaface_conf: float = 0.01
    retinaface_nms: float = 0.40
    retinaface_target_size: int = 640
    retinaface_max_size: int = 640
    retinaface_top_k: int = 5000
    retinaface_keep_top_k: int = 750
    # Only the primary face is followed per frame; weaker candidates are dropped
    # instead of being fed to recognition as junk crops.
    max_faces_per_frame: int = 1
    min_face_score: float = 0.05
    # largest = follow the person closest to the camera (stable apparent size);
    # confidence = strongest detection; continuity = keep the previous subject.
    face_selection: Literal["largest", "confidence", "continuity"] = "largest"
    recognition_rknn_model_path: Path = Path("models/rknn/recognition-int8.rknn")
    aligner_rknn_model_path: Path = Path("models/rknn/aligner-int8.rknn")
    rknn_manifest_path: Path = Path("models/rknn/manifest.json")
    rknn_core_mask: str = "auto"
    rknn_recognition_core_mask: str = "1"
    rknn_aligner_fallback: str = "onnx-cpu"
    provider: str | None = None
    align_score_threshold: float = 0.0
    video_analysis_max_concurrency: int = 1
    video_analysis_recognition_interval_seconds: float = 1.0
    video_analysis_recognition_queue_size: int = 32
    video_analysis_max_consecutive_decode_errors: int = 100
    video_analysis_download_source_enabled: bool = False
    live_stream_url: str = "/dev/video10"
    live_video_stream_id: str | None = None
    live_match_threshold: float = 0.3
    live_heartbeat_timeout_seconds: float = 30.0
    live_detection_fps: float = 12.0
    live_detection_result_ttl: float = 0.8
    live_recognition_fps: float = 5.0
    live_rtsp_transport: str = "tcp"
    live_capture_reconnect_enabled: bool = True
    live_capture_reconnect_interval: float = 1.0
    live_capture_max_consecutive_failures: int = 30
    live_preview_enabled: bool = False
    live_preview_window_name: str = "Live Face Recognition"
    live_preview_width: int | None = None
    live_preview_height: int | None = None
    live_preview_fps: float = 20.0
    live_preview_max_width: int | None = None
    live_preview_max_height: int | None = None
    live_preview_result_ttl: float = 1.5
    live_diagnostics_enabled: bool = False
    live_diagnostics_log_file: Path = Path("logs/live_recognition_diagnostics.log")
    live_diagnostics_throttle_seconds: float = 1.0
    performance_log_enabled: bool = False
    performance_log_stdout: bool = True
    performance_log_file: Path = Path("logs/rknn_performance.jsonl")
    performance_log_fps_window_seconds: float = 2.0
    performance_log_max_fps_meters: int = 1024


def default_config() -> ApiConfig:
    return ApiConfig()

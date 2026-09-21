from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .face_recognition_engine import FaceEngine
from .gallery import Gallery
from .utils import ensure_dir, find_executable, format_timestamp
from .face_recognition_engine.face_detection import FaceDetector


UNKNOWN_SIMILARITY_THRESHOLD = 0.1
UNKNOWN_NAME = "unknow"


def _sampling_step(source_fps: float, sample_fps: float) -> int:
    """Return one for explicit all-frame mode, otherwise a sampling stride."""
    if sample_fps <= 0:
        return 1
    return max(1, int(round(source_fps / max(0.1, sample_fps))))


@dataclass
class FrameResult:
    """视频采样帧的完整识别结果。"""

    frame_index: int
    timestamp: float
    box: tuple[int, int, int, int]
    raw_similarity: float
    best_name: str
    id_card: str
    raw_face_path: str
    raw_aligned_face_path: str
    target_score: float
    track_source: str
    lost_frames: int
    track_id: int | None
    # raw_align_score: float | None
    # raw_feature_valid: bool


@dataclass
class VideoMeta:
    """视频基础信息，后续抽帧和标注都依赖它。"""

    fps: float
    frame_count: int
    width: int
    height: int
    duration: float


def analyze_video(
    video_path: Path,
    output_dir: Path,
    engine: FaceEngine,
    gallery: Gallery,
    detector: FaceDetector,
    sample_fps: float = 5.0,
    min_similarity: float = 0.35,
    save_faces: int = 80,
) -> tuple[VideoMeta, list[FrameResult], list[dict]]:
    # 核心视频分析链路：打开视频、按频率抽帧、锁定最近目标、提取原图特征并记录结果。
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if frame_count else 0.0
    meta = VideoMeta(fps=fps, frame_count=frame_count, width=width, height=height, duration=duration)

    # 只保留原图裁剪和原图对齐结果，不再生成增强图目录。
    raw_dir = ensure_dir(output_dir / "frames" / "face_raw")
    raw_aligned_dir = ensure_dir(output_dir / "frames" / "face_aligned")
    step = _sampling_step(fps, sample_fps)
    expected_samples = (frame_count + step - 1) // step if frame_count else 0
    print(
        "[video-analysis] start: "
        f"video={video_path}, frames={frame_count}, duration={format_timestamp(duration)}, "
        f"resolution={width}x{height}, source_fps={fps:.2f}, sample_fps={sample_fps:.2f}, "
        f"step={step}, expected_sampled_frames={expected_samples}",
        flush=True,
    )

    results: list[FrameResult] = []
    saved_count = 0
    started_at = time.monotonic()
    progress_interval = 5.0
    next_progress_at = started_at + progress_interval

    def maybe_log_progress(current_frame: int) -> None:
        nonlocal next_progress_at
        now = time.monotonic()
        if now >= next_progress_at:
            _print_progress(
                "video-analysis",
                current_frame,
                frame_count,
                started_at,
                extra=f"sampled_results={len(results)} saved_faces={saved_count}",
            )
            next_progress_at = now + progress_interval

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step != 0:
            frame_index += 1
            maybe_log_progress(frame_index)
            continue

        timestamp = frame_index / fps
        tracked = detector.track_nearest(frame)
        # print(tracked)
        if tracked is None:
            frame_index += 1
            maybe_log_progress(frame_index)
            continue
        target = tracked.detection

        # 对最近目标只走原图路径：裁剪、对齐、提特征、和底库匹配。
        crop = engine.crop_face(frame, target.box)
        raw_info = engine.feature_from_detection(frame, target)
        best_person, raw_sim = gallery.match(raw_info)
        best_name, id_card = _identity_from_match(best_person, raw_sim)

        raw_path = ""
        raw_aligned_path = ""
        if saved_count < save_faces or raw_sim >= min_similarity:
            stem = f"{frame_index:08d}_{format_timestamp(timestamp).replace(':', '-')}"
            raw_path = str(raw_dir / f"{stem}_raw.jpg")
            cv2.imencode(".jpg", crop)[1].tofile(raw_path)
            # if raw_info.aligned_bgr is not None:
            #     raw_aligned_path = str(raw_aligned_dir / f"{stem}_aligned.jpg")
            #     cv2.imencode(".jpg", raw_info.aligned_bgr)[1].tofile(raw_aligned_path)
            saved_count += 1

        results.append(
            FrameResult(
                frame_index=frame_index,
                timestamp=timestamp,
                box=target.box,
                raw_similarity=float(raw_sim),
                best_name=best_name,
                id_card=id_card,
                raw_face_path=raw_path,
                raw_aligned_face_path=raw_aligned_path,
                target_score=target.score,
                track_source=tracked.source,
                lost_frames=tracked.lost_frames,
                track_id=tracked.track_id,
                # raw_align_score=raw_info.align_score,
                # raw_feature_valid=raw_info.feature_valid,
            )
        )
        frame_index += 1
        maybe_log_progress(frame_index)

    cap.release()
    _print_progress(
        "video-analysis",
        frame_index,
        frame_count,
        started_at,
        extra=f"sampled_results={len(results)} saved_faces={saved_count}",
        done=True,
    )
    segments = build_segments(results, min_similarity=min_similarity)
    return meta, results, segments


def build_segments(
    results: list[FrameResult],
    min_similarity: float,
    max_gap: float = 3.0,
    padding: float = 1.5,
) -> list[dict]:
    # 根据原图相似度筛出有效帧，并合并成目标出现时间段。
    accepted = [r for r in results if r.raw_similarity >= min_similarity]
    if not accepted:
        accepted = results
    if not accepted:
        return []

    segments = []
    start = accepted[0].timestamp
    end = accepted[0].timestamp
    for result in accepted[1:]:
        if result.timestamp - end <= max_gap:
            end = result.timestamp
        else:
            segments.append({"start": max(0.0, start - padding), "end": end + padding})
            start = result.timestamp
            end = result.timestamp
    segments.append({"start": max(0.0, start - padding), "end": end + padding})
    return segments


def annotate_video(
    video_path: Path,
    output_dir: Path,
    results: list[FrameResult],
    meta: VideoMeta,
    carry_seconds: float = 1.0,
) -> str:
    # 把采样帧上的识别结果向后延续一小段时间，方便观看标注视频时看清目标。
    output_path = output_dir / "annotated_video.mp4"
    intermediate_path = output_dir / ".annotated_video.mp4v.tmp.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for annotation: {video_path}")

    fps = meta.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = meta.width or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = meta.height or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(intermediate_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create annotated video: {intermediate_path}")

    print(
        "[video-annotation] start: "
        f"output={output_path}, frames={meta.frame_count}, duration={format_timestamp(meta.duration)}, "
        f"resolution={width}x{height}",
        flush=True,
    )
    started_at = time.monotonic()
    progress_interval = 5.0
    next_progress_at = started_at + progress_interval

    sorted_results = sorted(results, key=lambda r: r.frame_index)
    result_idx = 0
    active: FrameResult | None = None
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while result_idx < len(sorted_results) and sorted_results[result_idx].frame_index <= frame_index:
            active = sorted_results[result_idx]
            result_idx += 1

        timestamp = frame_index / fps
        if active is not None and 0 <= timestamp - active.timestamp <= carry_seconds:
            draw_frame_result(frame, active)

        writer.write(frame)
        frame_index += 1
        now = time.monotonic()
        if now >= next_progress_at:
            _print_progress("video-annotation", frame_index, meta.frame_count, started_at)
            next_progress_at = now + progress_interval

    writer.release()
    cap.release()
    try:
        _transcode_for_browser_playback(intermediate_path, output_path)
    finally:
        intermediate_path.unlink(missing_ok=True)
    _print_progress("video-annotation", frame_index, meta.frame_count, started_at, done=True)
    return str(output_path)


def _transcode_for_browser_playback(source_path: Path, output_path: Path) -> None:
    """Encode an OpenCV MP4 as H.264 and move its index to the file start."""

    ffmpeg = find_executable("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create a browser-compatible annotated video")

    encoded_path = output_path.with_name(f".{output_path.stem}.h264.tmp{output_path.suffix}")
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(encoded_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise RuntimeError(f"Failed to encode browser-compatible video: {detail}")
        encoded_path.replace(output_path)
    finally:
        encoded_path.unlink(missing_ok=True)


def _identity_from_match(person: Any, similarity: float) -> tuple[str, str]:
    if similarity < UNKNOWN_SIMILARITY_THRESHOLD:
        return UNKNOWN_NAME, ""
    return str(person.name), str(person.id_card)


def draw_frame_result(frame: np.ndarray, result: FrameResult) -> None:
    """Draw a recognition result on a BGR frame in-place."""

    _draw_frame_result(frame, result)


def _draw_frame_result(frame: np.ndarray, result: FrameResult) -> None:
    x, y, w, h = result.box
    x1, y1 = max(0, x), max(0, y)
    x2 = min(frame.shape[1] - 1, x + w)
    y2 = min(frame.shape[0] - 1, y + h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    label = f"{result.best_name}  sim:{result.raw_similarity:.4f}  id:{result.track_id}  {result.track_source}"
    _draw_label(frame, label, x1, y1)


def _draw_label(frame: np.ndarray, label: str, x: int, y: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = 5
    label_y = y - 8 if y - text_h - baseline - pad * 2 >= 0 else y + text_h + baseline + pad * 2
    top = max(0, label_y - text_h - baseline - pad)
    bottom = min(frame.shape[0] - 1, label_y + pad)
    right = min(frame.shape[1] - 1, x + text_w + pad * 2)
    cv2.rectangle(frame, (x, top), (right, bottom), (0, 150, 0), -1)
    cv2.putText(frame, label, (x + pad, label_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


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
            f"[{stage}] {label}: {safe_current}/{total} frames ({percent:.1f}%), "
            f"elapsed {_format_duration(elapsed)}, eta {_format_duration(eta)}, "
            f"speed {rate:.2f} frames/s{extra_text}",
            flush=True,
        )
    else:
        print(
            f"[{stage}] {label}: {current} frames, elapsed {_format_duration(elapsed)}, "
            f"speed {rate:.2f} frames/s{extra_text}",
            flush=True,
        )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

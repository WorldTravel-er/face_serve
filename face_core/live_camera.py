from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .face_recognition_engine import FaceEngine
from .gallery import Gallery
from .utils import ensure_dir, format_timestamp, write_csv, write_json
from .video_processor import FrameResult, draw_frame_result, _identity_from_match
from .face_recognition_engine.face_detection import FaceDetector


WriterFactory = Callable[[Path, float, tuple[int, int]], Any]


@dataclass
class _BufferedFrame:
    frame_index: int
    timestamp: float
    frame: np.ndarray


@dataclass
class _ActiveClip:
    writer: Any
    clip_path: Path
    start: float
    last_detected_timestamp: float
    best_result: FrameResult
    frame_count: int = 0


class ClipRecorder:
    """Record annotated event clips when live face detections appear."""

    def __init__(
        self,
        output_dir: Path,
        fps: float,
        frame_size: tuple[int, int],
        pre_roll_seconds: float = 1.5,
        post_roll_seconds: float = 2.0,
        camera_source: str = "0",
        writer_factory: WriterFactory | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.clips_dir = ensure_dir(output_dir / "clips")
        self.fps = max(0.1, float(fps))
        self.frame_size = frame_size
        self.pre_roll_seconds = max(0.0, float(pre_roll_seconds))
        self.post_roll_seconds = max(0.0, float(post_roll_seconds))
        self.camera_source = camera_source
        self.writer_factory = writer_factory or _create_mp4_writer
        self.events: list[dict[str, Any]] = []
        pre_roll_frames = int(round(self.fps * self.pre_roll_seconds))
        self._pre_roll: deque[_BufferedFrame] = deque(maxlen=max(0, pre_roll_frames))
        self._active: _ActiveClip | None = None
        self._event_index = 0

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
        result: FrameResult | None,
    ) -> None:
        buffered = _BufferedFrame(frame_index, timestamp, frame.copy())
        detected = result is not None

        if self._active is None:
            if self._pre_roll.maxlen:
                self._pre_roll.append(buffered)
            if detected:
                frames_to_write = list(self._pre_roll)
                if not frames_to_write or frames_to_write[-1].frame_index != frame_index:
                    frames_to_write.append(buffered)
                self._start_clip(frames_to_write, result)
            return

        self._write_frame(buffered.frame)
        if detected:
            self._active.last_detected_timestamp = timestamp
            self._update_best(result)
            return

        if timestamp - self._active.last_detected_timestamp >= self.post_roll_seconds:
            self.close_active(timestamp)

    def close_active(self, end_timestamp: float | None = None) -> None:
        if self._active is None:
            return
        active = self._active
        end = active.last_detected_timestamp if end_timestamp is None else end_timestamp
        active.writer.release()
        best = active.best_result
        self.events.append(
            {
                "camera": self.camera_source,
                "clip_path": str(active.clip_path),
                "start": active.start,
                "end": end,
                "start_text": format_timestamp(active.start),
                "end_text": format_timestamp(end),
                "best_name": best.best_name,
                "id_card": best.id_card,
                "best_similarity": best.raw_similarity,
                "best_frame_index": best.frame_index,
                "frame_count": active.frame_count,
            }
        )
        self._active = None
        self._pre_roll.clear()

    def _start_clip(self, frames: list[_BufferedFrame], result: FrameResult) -> None:
        self._event_index += 1
        clip_path = self.clips_dir / f"event_{self._event_index:04d}.mp4"
        writer = self.writer_factory(clip_path, self.fps, self.frame_size)
        if not writer.isOpened():
            if hasattr(writer, "release"):
                writer.release()
            raise RuntimeError(f"Failed to create live event clip: {clip_path}")
        start = frames[0].timestamp if frames else result.timestamp
        self._active = _ActiveClip(
            writer=writer,
            clip_path=clip_path,
            start=start,
            last_detected_timestamp=result.timestamp,
            best_result=result,
        )
        for item in frames:
            self._write_frame(item.frame)

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._active is None:
            return
        self._active.writer.write(frame)
        self._active.frame_count += 1

    def _update_best(self, result: FrameResult) -> None:
        if self._active is None:
            return
        if result.raw_similarity > self._active.best_result.raw_similarity:
            self._active.best_result = result


def run_live_camera(
    camera_source: str | int,
    output_dir: Path,
    engine: FaceEngine,
    gallery: Gallery,
    detector: FaceDetector,
    min_similarity: float = 0.3,
    save_faces: int = 80,
    camera_width: int | None = None,
    camera_height: int | None = None,
    camera_fps: float = 25.0,
    camera_backend: str = "auto",
    record_pre_roll: float = 1.5,
    record_post_roll: float = 2.0,
    preview: bool = False,
    max_seconds: float | None = None,
) -> tuple[list[FrameResult], list[dict[str, Any]]]:
    """Run real-time recognition from a camera source and save event clips."""

    output_dir = ensure_dir(output_dir)
    source = _coerce_camera_source(camera_source)
    backend_ids = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
    if camera_backend not in backend_ids:
        raise ValueError(f"Unknown camera backend: {camera_backend!r}")
    backends = (
        ("dshow", "msmf")
        if camera_backend == "auto" and os.name == "nt" and isinstance(source, int)
        else (camera_backend,)
    )
    for selected_backend in backends:
        cap = cv2.VideoCapture(source, backend_ids[selected_backend])
        if cap.isOpened():
            break
        cap.release()
    else:
        raise RuntimeError(f"Failed to open camera source {camera_source!r} with backend {camera_backend!r}")
    print(f"Camera backend: {selected_backend}", flush=True)

    if camera_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(camera_width))
    if camera_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(camera_height))
    if camera_fps:
        cap.set(cv2.CAP_PROP_FPS, float(camera_fps))

    fps = float(cap.get(cv2.CAP_PROP_FPS) or camera_fps or 25.0)
    raw_dir = ensure_dir(output_dir / "frames" / "face_raw")
    raw_aligned_dir = ensure_dir(output_dir / "frames" / "face_aligned")
    results: list[FrameResult] = []
    recorder: ClipRecorder | None = None
    saved_count = 0
    frame_index = 0
    last_timestamp = 0.0
    started_at = time.monotonic()

    try:
        while True:
            elapsed = time.monotonic() - started_at
            if max_seconds is not None and elapsed >= max_seconds:
                break

            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Failed to read camera frame from source {camera_source!r} "
                    f"with backend {selected_backend!r} after {frame_index} frames. "
                    "Try --camera-backend dshow or another --camera index."
                )

            timestamp = time.monotonic() - started_at
            last_timestamp = timestamp
            if recorder is None:
                height, width = frame.shape[:2]
                recorder = ClipRecorder(
                    output_dir=output_dir,
                    fps=fps,
                    frame_size=(width, height),
                    pre_roll_seconds=record_pre_roll,
                    post_roll_seconds=record_post_roll,
                    camera_source=str(camera_source),
                )

            result: FrameResult | None = None
            tracked = detector.track_nearest(frame)
            if tracked is not None:
                target = tracked.detection
                crop = engine.crop_face(frame, target.box)
                # raw_info = engine.feature_from_detection_with_info(frame, target)
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

                result = FrameResult(
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
                results.append(result)

            annotated = frame.copy()
            if result is not None:
                draw_frame_result(annotated, result)
            recorder.update(annotated, frame_index=frame_index, timestamp=timestamp, result=result)

            if preview:
                cv2.imshow("Live face recognition", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            recorder.close_active(last_timestamp)
        cap.release()
        if preview:
            cv2.destroyAllWindows()

    events = recorder.events if recorder is not None else []
    write_live_reports(output_dir, str(camera_source), results, events, engine.describe())
    return results, events


def write_live_reports(
    output_dir: Path,
    camera_source: str,
    results: list[FrameResult],
    events: list[dict[str, Any]],
    feature_meta: dict[str, Any] | None = None,
) -> None:
    del camera_source, feature_meta
    write_json(output_dir / "live_events.json", events)
    write_csv(
        output_dir / "live_results.csv",
        [_frame_to_live_row(result) for result in results],
        [
            "frame_index",
            "timestamp",
            "raw_similarity",
            "raw_align_score",
            "raw_feature_valid",
            "best_match_name",
            "id_card",
            "face_box",
            "raw_face_path",
            "raw_aligned_face_path",
            "track_source",
            "track_id",
            "lost_frames",
        ],
    )


def _frame_to_live_row(result: FrameResult) -> dict[str, Any]:
    return {
        "frame_index": result.frame_index,
        "timestamp": format_timestamp(result.timestamp),
        "raw_similarity": f"{result.raw_similarity:.6f}",
        "raw_align_score": "" if result.raw_align_score is None else f"{result.raw_align_score:.6f}",
        "raw_feature_valid": result.raw_feature_valid,
        "best_match_name": result.best_name,
        "id_card": result.id_card,
        "face_box": result.box,
        "raw_face_path": result.raw_face_path,
        "raw_aligned_face_path": result.raw_aligned_face_path,
        "track_source": result.track_source,
        "track_id": result.track_id,
        "lost_frames": result.lost_frames,
    }


def _create_mp4_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> Any:
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)


def _coerce_camera_source(camera_source: str | int) -> str | int:
    if isinstance(camera_source, int):
        return camera_source
    value = str(camera_source)
    return int(value) if value.isdecimal() else value


__all__ = ["ClipRecorder", "run_live_camera", "write_live_reports"]

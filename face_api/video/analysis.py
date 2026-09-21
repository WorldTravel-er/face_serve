from __future__ import annotations

import math
import queue
import threading
import time
import tempfile
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2

from face_api.core.errors import FaceApiError
from face_api.video.store import VideoAnalysisStore, VideoFaceRecord


CaptureFactory = Callable[[str], Any]
TrackerFactory = Callable[[], Any]


@dataclass(frozen=True)
class _RecognitionObservation:
    subject_id: str
    name: str
    similarity: float
    timestamp: float
    sequence: int


@dataclass
class _TrackState:
    track_id: int
    first_seconds: float
    last_seconds: float
    hits: int = 0
    observations: list[_RecognitionObservation] = field(default_factory=list)


@dataclass(frozen=True)
class _RecognitionJob:
    track_id: int
    frame_seq: int
    frame: Any
    detection: Any
    seconds: float
    sequence: int

class VideoAnalysisProcessor:
    def __init__(
        self,
        *,
        video_store: VideoAnalysisStore,
        subject_store: Any,
        recognition_service: Any,
        tracker_factory: TrackerFactory,
        capture_factory: CaptureFactory | None = None,
        sample_interval_seconds: float = 1.0,
        threshold: float = 0.3,
        download_source_enabled: bool = False,
        download_dir: str | Path | None = None,
        recognition_queue_size: int = 32,
    ) -> None:
        self.video_store = video_store
        self.subject_store = subject_store
        self.recognition_service = recognition_service
        if not callable(tracker_factory):
            raise ValueError("tracker_factory must be configured")
        self.tracker_factory = tracker_factory
        self.capture_factory = capture_factory or cv2.VideoCapture
        self.sample_interval_seconds = max(float(sample_interval_seconds or 1.0), 0.001)
        self.threshold = float(threshold)
        self.download_source_enabled = bool(download_source_enabled)
        self.download_dir = Path(download_dir) if download_dir is not None else Path(tempfile.gettempdir()) / "face_api_video_downloads"
        self.recognition_queue_size = max(int(recognition_queue_size or 1), 1)

    def run(self, task_name: str) -> None:
        try:
            task = self.video_store.get_task(task_name)
            with self._prepared_video_source(task.task_name, task.source_url) as analysis_source:
                records = self._analyze_source(task.task_name, analysis_source)
            self.video_store.finish_task(task_name, records)
        except Exception as exc:
            self.video_store.fail_task(task_name, str(exc) or "Video analysis failed")

    @contextmanager
    def _prepared_video_source(self, task_name: str, source_url: str) -> Iterator[str]:
        if not self._should_download_source(source_url):
            yield source_url
            return

        local_path = self._download_source(source_url)
        print(
            f"[video-analysis] source downloaded: task={task_name}, url={source_url}, path={local_path}",
            flush=True,
        )
        try:
            yield str(local_path)
        finally:
            local_path.unlink()

    def _should_download_source(self, source_url: str) -> bool:
        if not self.download_source_enabled:
            return False
        return urlparse(str(source_url)).scheme.lower() in {"http", "https"}

    def _download_source(self, source_url: str) -> Path:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(source_url)
        suffix = Path(parsed.path).suffix or ".video"
        target = self.download_dir / f"{uuid.uuid4().hex}{suffix}"
        downloaded = 0
        try:
            with urlopen(source_url, timeout=60.0) as response:
                expected_length = response.headers.get("Content-Length")
                with target.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)

            if downloaded <= 0:
                raise RuntimeError("Downloaded video source is empty")
            if expected_length is not None and downloaded != int(expected_length):
                raise RuntimeError(
                    f"Downloaded video source is incomplete: expected {expected_length} bytes, got {downloaded}"
                )
            return target
        except Exception:
            if target.exists():
                target.unlink()
            raise

    def _analyze_source(self, task_name: str, source_url: str) -> list[VideoFaceRecord]:
        candidates = list(self.subject_store.list_feature_records(self.recognition_service.embedding_identity()))
        tracker = self.tracker_factory()
        capture = self.capture_factory(source_url)
        tracks: dict[int, _TrackState] = {}
        tracks_lock = threading.Lock()
        recognition_errors: list[Exception] = []
        recognition_queue: queue.Queue[_RecognitionJob | None] | None = None
        recognition_worker: threading.Thread | None = None
        if candidates:
            recognition_queue = queue.Queue(maxsize=self.recognition_queue_size)
            recognition_worker = threading.Thread(
                target=self._recognition_worker,
                name=f"video-recognition-{task_name}",
                args=(task_name, recognition_queue, tracks, tracks_lock, candidates, recognition_errors),
                daemon=True,
            )
            recognition_worker.start()
        job_sequence = 0
        try:
            if not capture.isOpened():
                raise RuntimeError("Cannot open video source")
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if not math.isfinite(fps) or fps <= 0.0:
                fps = 25.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            video_duration = total_frames / fps if total_frames > 0 and fps > 0.0 else 0.0
            self.video_store.update_task_video_duration(task_name, video_duration)
            self.video_store.update_task_progress(task_name, 0, total_frames)
            last_progress_update = time.monotonic()
            progress_update_interval = 1.0
            sample_every = max(int(round(fps * self.sample_interval_seconds)), 1)
            frame_index = 0
            error_count = 0

            def maybe_update_progress(force: bool = False) -> None:
                nonlocal last_progress_update
                now = time.monotonic()
                if force or now - last_progress_update >= progress_update_interval:
                    self.video_store.update_task_progress(task_name, frame_index, total_frames)
                    last_progress_update = now

            while True:
                ok, frame = capture.read()
                if not ok:
                    error_count += 1
                    if error_count > 20:
                        break
                    continue

                tracked = self._track_nearest_with_performance(
                    task_name=task_name,
                    tracker=tracker,
                    frame=frame,
                    frame_seq=frame_index,
                )
                if tracked is not None and tracked.track_id is not None:
                    track_id = int(tracked.track_id)
                    seconds = frame_index / fps
                    with tracks_lock:
                        state = tracks.get(track_id)
                        if state is None:
                            state = _TrackState(
                                track_id=track_id,
                                first_seconds=seconds,
                                last_seconds=seconds,
                            )
                            tracks[track_id] = state
                        else:
                            state.last_seconds = seconds
                        state.hits += 1

                    if recognition_queue is not None and frame_index % sample_every == 0:
                        job = _RecognitionJob(
                            track_id=track_id,
                            frame_seq=frame_index,
                            frame=self._copy_frame_for_recognition(frame),
                            detection=tracked.detection,
                            seconds=seconds,
                            sequence=job_sequence,
                        )
                        if self._enqueue_recognition_job(recognition_queue, job):
                            job_sequence += 1

                frame_index += 1
                maybe_update_progress()
            maybe_update_progress(force=True)
        finally:
            capture.release()
            if recognition_queue is not None and recognition_worker is not None:
                self._stop_recognition_worker(recognition_queue, recognition_worker, recognition_errors)

        if recognition_errors:
            raise recognition_errors[0]

        with tracks_lock:
            states = sorted(tracks.values(), key=lambda value: (value.first_seconds, value.track_id))
        # One subject per video: keep only the track followed for the most frames,
        # so a competing track cannot produce a second identity record.
        if len(states) > 1:
            dominant = max(states, key=lambda value: value.hits)
            states = [dominant]
        records: list[VideoFaceRecord] = []
        for state in states:
            subject_id, name = self._resolve_identity(state.observations)
            records.append(
                VideoFaceRecord(
                    subject_id=subject_id,
                    name=name,
                    first_appear_ts=format_video_timestamp(state.first_seconds),
                    last_disappear_ts=format_video_timestamp(state.last_seconds),
                    track_id=state.track_id,
                )
            )
        return optimize_video_face_records(records)

    def _copy_frame_for_recognition(self, frame: Any) -> Any:
        copy = getattr(frame, "copy", None)
        return copy() if callable(copy) else frame

    def _track_nearest_with_performance(self, *, task_name: str, tracker: Any, frame: Any, frame_seq: int) -> Any:
        context = getattr(self.recognition_service, "performance_context", None)
        manager = (
            context(channel="video_analysis", task_name=task_name, frame_seq=frame_seq)
            if callable(context)
            else nullcontext()
        )
        with manager:
            return tracker.track_nearest(frame)

    def _enqueue_recognition_job(
        self,
        recognition_queue: queue.Queue[_RecognitionJob | None],
        job: _RecognitionJob,
    ) -> bool:
        try:
            recognition_queue.put_nowait(job)
        except queue.Full:
            return False
        return True

    def _recognition_worker(
        self,
        task_name: str,
        recognition_queue: queue.Queue[_RecognitionJob | None],
        tracks: dict[int, _TrackState],
        tracks_lock: threading.Lock,
        candidates: Any,
        recognition_errors: list[Exception],
    ) -> None:
        while True:
            job = recognition_queue.get()
            if job is None:
                return
            try:
                context = getattr(self.recognition_service, "performance_context", None)
                manager = (
                    context(
                        channel="video_analysis",
                        task_name=task_name,
                        track_id=job.track_id,
                        frame_seq=job.frame_seq,
                        sequence=job.sequence,
                    )
                    if callable(context)
                    else nullcontext()
                )
                with manager:
                    observation = self._recognize_track(
                        job.frame,
                        job.detection,
                        candidates,
                        job.seconds,
                        job.sequence,
                    )
                if observation is not None:
                    with tracks_lock:
                        state = tracks.get(job.track_id)
                        if state is not None:
                            state.observations.append(observation)
            except Exception as exc:
                recognition_errors.append(exc)
                return

    def _stop_recognition_worker(
        self,
        recognition_queue: queue.Queue[_RecognitionJob | None],
        recognition_worker: threading.Thread,
        recognition_errors: list[Exception],
    ) -> None:
        while recognition_worker.is_alive():
            try:
                recognition_queue.put(None, timeout=0.1)
                break
            except queue.Full:
                if recognition_errors or not recognition_worker.is_alive():
                    break
        recognition_worker.join()
    def _recognize_track(
        self,
        frame: Any,
        detection: Any,
        candidates: Any,
        seconds: float,
        sequence: int,
    ) -> _RecognitionObservation | None:
        try:
            result = self.recognition_service.recognize_detection(
                frame,
                detection,
                candidates,
                top_k=1,
                threshold=None,
            )
        except FaceApiError:
            return None

        matches = result.get("matches") or []
        if not matches:
            return None
        best = matches[0]
        return _RecognitionObservation(
            subject_id=str(best["subject_id"]),
            name=str(best["name"]),
            similarity=float(best["similarity"]),
            timestamp=seconds,
            sequence=sequence,
        )

    def _resolve_identity(
        self,
        observations: list[_RecognitionObservation],
    ) -> tuple[str, str]:
        high_confidence = [item for item in observations if item.similarity > 0.5]
        if high_confidence:
            winner = max(high_confidence, key=lambda item: (item.similarity, -item.sequence))
            return winner.subject_id, winner.name

        eligible = [
            item
            for item in observations
            if self.threshold <= item.similarity <= 0.5
        ]
        if not eligible:
            return "", "unknown"

        grouped: dict[str, list[_RecognitionObservation]] = {}
        for item in eligible:
            grouped.setdefault(item.subject_id, []).append(item)

        def rank(group: list[_RecognitionObservation]) -> tuple[float, float, int, str]:
            mean_similarity = sum(item.similarity for item in group) / len(group)
            earliest = min(item.sequence for item in group)
            return (-len(group), -mean_similarity, earliest, group[0].subject_id)

        winning_group = min(grouped.values(), key=rank)
        representative = min(winning_group, key=lambda item: item.sequence)
        return representative.subject_id, representative.name


def format_video_timestamp(seconds: float) -> str:
    milliseconds = int(round(max(float(seconds), 0.0) * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def optimize_video_face_records(records: list[VideoFaceRecord]) -> list[VideoFaceRecord]:
    sorted_records = sorted(records, key=lambda record: (_record_start_seconds(record), _record_end_seconds(record)))
    merged = _merge_same_identity_records(sorted_records)
    filtered = _remove_short_isolated_records(merged)
    return _merge_same_identity_records(filtered)


def _merge_same_identity_records(records: list[VideoFaceRecord]) -> list[VideoFaceRecord]:
    optimized: list[VideoFaceRecord] = []
    for record in records:
        if not optimized:
            optimized.append(record)
            continue

        previous = optimized[-1]
        gap = _record_start_seconds(record) - _record_end_seconds(previous)
        if _record_identity(previous) == _record_identity(record) and gap < 2.0:
            optimized[-1] = VideoFaceRecord(
                subject_id=previous.subject_id,
                name=previous.name,
                first_appear_ts=previous.first_appear_ts,
                last_disappear_ts=_later_timestamp(previous.last_disappear_ts, record.last_disappear_ts),
                track_id=previous.track_id,
            )
        else:
            optimized.append(record)
    return optimized


def _remove_short_isolated_records(records: list[VideoFaceRecord]) -> list[VideoFaceRecord]:
    if len(records) < 3:
        return records

    filtered: list[VideoFaceRecord] = []
    for index, record in enumerate(records):
        if index == 0 or index == len(records) - 1:
            filtered.append(record)
            continue

        duration = _record_end_seconds(record) - _record_start_seconds(record)
        current_identity = _record_identity(record)
        previous_identity = _record_identity(records[index - 1])
        next_identity = _record_identity(records[index + 1])
        is_short_isolated = duration < 1.0 and previous_identity != current_identity and next_identity != current_identity
        if not is_short_isolated:
            filtered.append(record)
    return filtered


def _record_identity(record: VideoFaceRecord) -> tuple[str, str]:
    return record.subject_id, record.name


def _record_start_seconds(record: VideoFaceRecord) -> float:
    return _parse_video_timestamp(record.first_appear_ts)


def _record_end_seconds(record: VideoFaceRecord) -> float:
    return _parse_video_timestamp(record.last_disappear_ts)


def _later_timestamp(left: str, right: str) -> str:
    return left if _parse_video_timestamp(left) >= _parse_video_timestamp(right) else right


def _parse_video_timestamp(value: str) -> float:
    hours_text, minutes_text, seconds_text = str(value).split(':', 2)
    return int(hours_text) * 3600.0 + int(minutes_text) * 60.0 + float(seconds_text)

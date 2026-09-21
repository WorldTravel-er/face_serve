from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pytest

from face_api.video.analysis import VideoAnalysisProcessor
from face_api.video.pyav_reader import DecodedVideoFrame, VideoDecodeStats


class EmptyReader:
    def __init__(self) -> None:
        self.stats = VideoDecodeStats()
        self.closed = False

    def __enter__(self) -> "EmptyReader":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.closed = True

    @property
    def total_frames(self) -> int:
        return 0

    @property
    def duration_seconds(self) -> float:
        return 0.0

    def frames(self) -> Iterator[DecodedVideoFrame]:
        return iter(())


class FrameReader(EmptyReader):
    def __init__(self, frames: list[DecodedVideoFrame]) -> None:
        super().__init__()
        self._frames = frames
        self.stats.usable_frames = len(frames)

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    def frames(self) -> Iterator[DecodedVideoFrame]:
        return iter(self._frames)


class StoreStub:
    def update_task_video_duration(self, *args: Any) -> None:
        return

    def update_task_progress(self, *args: Any) -> None:
        return


class SubjectStoreStub:
    def list_feature_records(self, embedding_identity: str) -> list[Any]:
        return []


class RecognitionServiceStub:
    def embedding_identity(self) -> str:
        return "test-embedding"


class TrackerStub:
    def track_nearest(self, frame: Any) -> Any:
        return SimpleNamespace(track_id=7, detection=object())


def test_video_analysis_rejects_source_that_yields_no_usable_frames() -> None:
    reader = EmptyReader()
    processor = VideoAnalysisProcessor(
        video_store=StoreStub(),
        subject_store=SubjectStoreStub(),
        recognition_service=RecognitionServiceStub(),
        tracker_factory=object,
        reader_factory=lambda _: reader,
    )

    with pytest.raises(RuntimeError, match="Video contains no usable frames"):
        processor._analyze_source("empty-video", "http://example.test/empty.mp4")

    assert reader.closed is True


def test_video_analysis_uses_pyav_presentation_timestamps() -> None:
    image = np.ones((2, 2, 3), dtype=np.uint8)
    reader = FrameReader(
        [
            DecodedVideoFrame(image=image, source_index=0, timestamp_seconds=12.3, key_frame=True),
            DecodedVideoFrame(image=image, source_index=1, timestamp_seconds=14.8, key_frame=False),
        ]
    )
    processor = VideoAnalysisProcessor(
        video_store=StoreStub(),
        subject_store=SubjectStoreStub(),
        recognition_service=RecognitionServiceStub(),
        tracker_factory=TrackerStub,
        reader_factory=lambda _: reader,
    )

    records = processor._analyze_source("timestamp-video", "http://example.test/video.mp4")

    assert len(records) == 1
    assert records[0].track_id == 7
    assert records[0].first_appear_ts == "00:00:12.300"
    assert records[0].last_disappear_ts == "00:00:14.800"

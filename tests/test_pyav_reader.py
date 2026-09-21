from __future__ import annotations

from fractions import Fraction

import av
import numpy as np
import pytest

from face_api.video.pyav_reader import PyAVVideoReader


class FakeCodecContext:
    def __init__(self) -> None:
        self.options: dict[str, str] = {}


class FakeStream:
    type = "video"
    average_rate = Fraction(25, 1)
    frames = 2
    duration = 2
    time_base = Fraction(1, 25)

    def __init__(self) -> None:
        self.codec_context = FakeCodecContext()


class FakeFrame:
    def __init__(self, *, corrupt: bool, pts: int) -> None:
        self.is_corrupt = corrupt
        self.pts = pts
        self.time = float(Fraction(pts, 25))
        self.time_base = Fraction(1, 25)
        self.key_frame = pts == 0

    def to_ndarray(self, *, format: str) -> np.ndarray:
        assert format == "bgr24"
        return np.ones((2, 2, 3), dtype=np.uint8)


class FakePacket:
    def __init__(
        self,
        *,
        corrupt: bool = False,
        frames: list[FakeFrame] | None = None,
        decode_error: bool = False,
    ) -> None:
        self.is_corrupt = corrupt
        self._frames = frames or []
        self._decode_error = decode_error

    def decode(self) -> list[FakeFrame]:
        if self._decode_error:
            raise av.InvalidDataError(1, "invalid packet")
        return self._frames


class FakeContainer:
    def __init__(self, packets: list[FakePacket]) -> None:
        self.streams = [FakeStream()]
        self._packets = packets
        self.closed = False

    def demux(self, stream: FakeStream):
        assert stream is self.streams[0]
        yield from self._packets

    def close(self) -> None:
        self.closed = True


def test_reader_skips_corrupt_packets_decode_errors_and_corrupt_frames(monkeypatch) -> None:
    container = FakeContainer(
        [
            FakePacket(corrupt=True),
            FakePacket(decode_error=True),
            FakePacket(frames=[FakeFrame(corrupt=True, pts=1), FakeFrame(corrupt=False, pts=2)]),
        ]
    )
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: container)

    with PyAVVideoReader("sample.mp4", max_consecutive_errors=3) as reader:
        frames = list(reader.frames())

    assert len(frames) == 1
    assert frames[0].source_index == 1
    assert frames[0].timestamp_seconds == pytest.approx(0.08)
    assert reader.stats.packets_seen == 3
    assert reader.stats.corrupt_packets == 1
    assert reader.stats.decode_errors == 1
    assert reader.stats.decoded_frames == 2
    assert reader.stats.corrupt_frames == 1
    assert reader.stats.usable_frames == 1
    assert container.streams[0].codec_context.options["err_detect"] == "explode"
    assert container.closed is True


def test_reader_fails_after_consecutive_decode_errors(monkeypatch) -> None:
    container = FakeContainer(
        [FakePacket(corrupt=True), FakePacket(decode_error=True), FakePacket(corrupt=True)]
    )
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: container)

    with PyAVVideoReader("sample.mp4", max_consecutive_errors=2) as reader:
        with pytest.raises(RuntimeError, match="Too many consecutive video decode errors: 3"):
            list(reader.frames())

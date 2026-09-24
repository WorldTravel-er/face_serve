from __future__ import annotations

from face_api.main import build_parser


def test_live_stream_defaults_are_not_bound_to_a_deployment(monkeypatch) -> None:
    monkeypatch.delenv("FACE_API_LIVE_STREAM_URL", raising=False)
    monkeypatch.delenv("FACE_API_LIVE_VIDEO_STREAM_ID", raising=False)

    args = build_parser().parse_args([])

    assert args.live_stream_url == "/dev/video10"
    assert args.live_video_stream_id is None


def test_live_stream_can_be_configured_with_environment(monkeypatch) -> None:
    monkeypatch.setenv("FACE_API_LIVE_STREAM_URL", "rtsp://camera.example/live")
    monkeypatch.setenv("FACE_API_LIVE_VIDEO_STREAM_ID", "entrance-camera")

    args = build_parser().parse_args([])

    assert args.live_stream_url == "rtsp://camera.example/live"
    assert args.live_video_stream_id == "entrance-camera"


def test_live_stream_command_line_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("FACE_API_LIVE_STREAM_URL", "rtsp://camera.example/from-env")
    monkeypatch.setenv("FACE_API_LIVE_VIDEO_STREAM_ID", "environment-camera")

    args = build_parser().parse_args(
        [
            "--live-stream-url",
            "rtsp://camera.example/from-cli",
            "--live-video-stream-id",
            "cli-camera",
        ]
    )

    assert args.live_stream_url == "rtsp://camera.example/from-cli"
    assert args.live_video_stream_id == "cli-camera"


def test_service_bind_and_storage_can_be_configured_with_environment(monkeypatch) -> None:
    monkeypatch.setenv("FACE_API_HOST", "127.0.0.1")
    monkeypatch.setenv("FACE_API_PORT", "8100")
    monkeypatch.setenv("FACE_API_PROVIDER", "cpu")
    monkeypatch.setenv("FACE_API_DATA_DIR", "/var/lib/face-serve-test")

    args = build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 8100
    assert args.provider == "cpu"
    assert str(args.data_dir) == "/var/lib/face-serve-test"


def test_default_service_port_is_8000(monkeypatch) -> None:
    monkeypatch.delenv("FACE_API_PORT", raising=False)

    args = build_parser().parse_args([])

    assert args.port == 8000

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RTSP_URL = "rtsp://127.0.0.1:8554/camera"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "rtsp_frame.jpg"


class RtspFrameError(RuntimeError):
    """Raised when an RTSP frame cannot be captured or saved."""


def load_cv2_module() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required. Run this inside the project venv, e.g. `uv run python ...`.") from exc
    return cv2


def resolve_output_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def save_frame(
    rtsp_url: str,
    output_path: str | Path,
    *,
    cv2_module: Any | None = None,
    warmup_frames: int = 0,
) -> Path:
    cv2 = cv2_module or load_cv2_module()
    resolved_output = resolve_output_path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(rtsp_url)
    try:
        if not capture.isOpened():
            raise RtspFrameError(f"failed to open RTSP stream: {rtsp_url}")

        frame = None
        ok = False
        read_count = max(int(warmup_frames), 0) + 1
        for _ in range(read_count):
            ok, frame = capture.read()
            if ok and frame is not None:
                continue

        if not ok or frame is None:
            raise RtspFrameError(f"failed to read frame from RTSP stream: {rtsp_url}")

        saved = bool(cv2.imwrite(str(resolved_output), frame))
        if not saved:
            raise RtspFrameError(f"failed to save frame: {resolved_output}")
    finally:
        capture.release()

    return resolved_output


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save one frame from an RTSP video stream.")
    parser.add_argument("--rtsp-url", default=DEFAULT_RTSP_URL, help="RTSP stream URL.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output image path.")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Read and discard this many frames before saving, useful for waiting for an RTSP key frame.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        output = save_frame(args.rtsp_url, args.output, warmup_frames=args.warmup_frames)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print(f"[PASS] saved RTSP frame: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

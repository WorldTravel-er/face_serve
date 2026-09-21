"""Verify that an RK3588 host can load and execute an RKNN Lite2 model.

This is deliberately a host-side preflight.  It checks the mounted runtime and
the NPU DRM render node before it imports or initializes Lite2, so deployment
failures do not get mistaken for a model-conversion problem.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import os
import stat
import sys
from typing import TYPE_CHECKING, Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from face_core.face_recognition_engine.rknn_runtime import RknnInputSpec, RknnModelSession


RUNTIME_LIBRARY = Path("/usr/lib/librknnrt.so")
RENDER_DEVICE = Path("/dev/dri/renderD129")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an RKNN Lite2 device preflight.")
    parser.add_argument("--model", type=Path, required=True, help="RKNN model to load.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("models/rknn/manifest.json"),
        help="Verified RKNN manifest containing the model input metadata.",
    )
    parser.add_argument(
        "--core-mask",
        choices=("auto", "0", "1", "2", "0_1_2"),
        default="auto",
        help="NPU core mask passed to RKNN Lite2.",
    )
    return parser


def check_prerequisites(
    model_path: Path,
    *,
    runtime_library: Path = RUNTIME_LIBRARY,
    render_device: Path = RENDER_DEVICE,
) -> None:
    """Fail with an actionable message before importing/initializing Lite2."""

    if not runtime_library.is_file():
        raise RuntimeError(f"RKNN runtime library is missing: {runtime_library}")
    if not render_device.exists():
        raise RuntimeError(f"RKNN NPU render device is missing: {render_device}")
    if not stat.S_ISCHR(render_device.stat().st_mode):
        raise RuntimeError(f"RKNN NPU render device is not a character device: {render_device}")
    if not os.access(render_device, os.R_OK | os.W_OK):
        raise RuntimeError(
            f"RKNN NPU render device is not readable and writable: {render_device}; "
            "run the service in the render group"
        )
    if not model_path.is_file():
        raise RuntimeError(f"RKNN model file is missing: {model_path}")
    try:
        lite_spec = importlib.util.find_spec("rknnlite.api")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "RKNN Lite2 Python module rknnlite.api is unavailable; install "
            "rknn-toolkit-lite2==2.3.2 for this CPython/aarch64 host"
        ) from exc
    if lite_spec is None:
        raise RuntimeError(
            "RKNN Lite2 Python module rknnlite.api is unavailable; install "
            "rknn-toolkit-lite2==2.3.2 for this CPython/aarch64 host"
        )


def main(
    argv: list[str] | None = None,
    session_factory: Callable[..., Any] | None = None,
    *,
    prerequisite_checker: Callable[[Path], None] = check_prerequisites,
    input_spec_loader: Callable[[Path, Path], Any] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        prerequisite_checker(args.model)
        # Do not import the Lite2 wrapper until every host prerequisite has
        # passed.  This makes a missing device/library diagnostic deterministic.
        if input_spec_loader is None or session_factory is None:
            from face_core.face_recognition_engine.rknn_runtime import RknnModelSession, load_input_spec

            input_spec_loader = input_spec_loader or load_input_spec
            session_factory = session_factory or RknnModelSession
        input_spec = input_spec_loader(args.manifest, args.model)
        session = session_factory(args.model, input_spec, core_mask=args.core_mask)
        try:
            # Deployment models use a uint8 image input.  The session validates
            # the manifest-derived shape and dtype before reaching the NPU.
            session.infer(np.zeros(session.input_spec.shape, dtype=np.uint8))
        finally:
            session.close()
        print(f"runtime=rknn model={args.model}")
        return 0
    except Exception as exc:
        print(f"RKNN device check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

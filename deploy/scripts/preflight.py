#!/usr/bin/env python3
"""Validate host, model, storage, and inference prerequisites before startup."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"{path}:{number}: invalid environment key")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def resolve_path(value: str, app_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else app_dir / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(name: str, path: Path) -> Check:
    if not path.is_file():
        return Check(name, False, f"missing file: {path}")
    if not os.access(path, os.R_OK):
        return Check(name, False, f"file is not readable: {path}")
    return Check(name, True, str(path))


def check_directory(name: str, path: Path) -> Check:
    if not path.is_dir():
        return Check(name, False, f"missing directory: {path}")
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        return Check(name, False, f"directory is not readable/writable: {path}")
    return Check(name, True, str(path))


def verify_rknn_manifest(manifest_path: Path, model_paths: list[Path]) -> list[Check]:
    checks: list[Check] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["models"]
        if not isinstance(entries, list):
            raise TypeError("models is not a list")
    except Exception as exc:
        return [Check("rknn-manifest-content", False, str(exc))]

    manifest_dir = manifest_path.resolve().parent
    by_path: dict[Path, dict] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("rknn_path"), str):
            by_path[(manifest_dir / entry["rknn_path"]).resolve()] = entry

    for model_path in model_paths:
        resolved = model_path.resolve()
        entry = by_path.get(resolved)
        name = f"rknn-model-sha256:{model_path.name}"
        if entry is None:
            checks.append(Check(name, False, "model is absent from manifest"))
            continue
        expected = entry.get("rknn_sha256")
        if not isinstance(expected, str):
            checks.append(Check(name, False, "manifest SHA-256 is missing"))
            continue
        actual = sha256_file(model_path) if model_path.is_file() else ""
        checks.append(Check(name, actual == expected.lower(), f"expected={expected.lower()} actual={actual}"))

    runtime = manifest.get("runtime")
    if isinstance(runtime, dict) and runtime.get("aligner_runtime") == "onnx-cpu":
        aligner_rel = runtime.get("aligner_onnx_path")
        expected = runtime.get("aligner_onnx_sha256")
        if isinstance(aligner_rel, str) and isinstance(expected, str):
            aligner = (manifest_dir / aligner_rel).resolve()
            actual = sha256_file(aligner) if aligner.is_file() else ""
            checks.append(
                Check(
                    "rknn-aligner-sha256",
                    actual == expected.lower(),
                    f"path={aligner} expected={expected.lower()} actual={actual}",
                )
            )
        else:
            checks.append(Check("rknn-aligner-sha256", False, "aligner contract is incomplete"))
    return checks


def check_onnx_runtime(provider: str) -> Check:
    try:
        import onnxruntime as ort
        providers = list(ort.get_available_providers())
    except Exception as exc:
        return Check("onnxruntime", False, f"import failed: {exc}")
    required = "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    return Check("onnxruntime", required in providers, f"required={required} available={providers}")


def check_rknn_runtime(library_path: Path, device_path: Path) -> list[Check]:
    checks: list[Check] = [check_file("rknn-runtime-library", library_path)]
    if library_path.is_file():
        try:
            ctypes.CDLL(str(library_path))
            checks.append(Check("rknn-runtime-load", True, str(library_path)))
        except OSError as exc:
            checks.append(Check("rknn-runtime-load", False, str(exc)))

    if not device_path.exists():
        checks.append(Check("rknn-device", False, f"missing device: {device_path}"))
    else:
        mode_ok = stat.S_ISCHR(device_path.stat().st_mode)
        access_ok = os.access(device_path, os.R_OK | os.W_OK)
        checks.append(
            Check(
                "rknn-device",
                mode_ok and access_ok,
                f"path={device_path} character_device={mode_ok} read_write={access_ok}",
            )
        )

    try:
        spec = importlib.util.find_spec("rknnlite.api")
    except (ImportError, ModuleNotFoundError):
        spec = None
    checks.append(Check("rknnlite-module", spec is not None, "rknnlite.api"))
    if spec is not None:
        try:
            version = importlib.metadata.version("rknn-toolkit-lite2")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        checks.append(Check("rknnlite-version", version == "2.3.2", f"installed={version} required=2.3.2"))
    return checks


def run_deep_rknn_checks(app_dir: Path, python: Path, manifest: Path, models: list[Path]) -> list[Check]:
    checker = app_dir / "scripts" / "check_rknn_device.py"
    if not checker.is_file():
        return [Check("rknn-deep-check", False, f"missing checker: {checker}")]
    checks = []
    for model in models:
        completed = subprocess.run(
            [str(python), str(checker), "--model", str(model), "--manifest", str(manifest)],
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        detail = (completed.stdout + completed.stderr).strip()
        checks.append(Check(f"rknn-inference:{model.name}", completed.returncode == 0, detail))
    return checks


def collect_checks(args: argparse.Namespace, environ: Mapping[str, str] | None = None) -> list[Check]:
    file_env = load_env_file(args.env_file)
    effective = dict(file_env)
    effective.update(dict(environ or os.environ))
    app_dir = args.app_dir.resolve()
    runtime = args.runtime or effective.get("FACE_API_RUNTIME", "rknn").lower()
    checks = [
        Check("python-version", sys.version_info[:2] in {(3, 11), (3, 12)}, platform.python_version()),
        Check("runtime", runtime in {"onnx", "rknn"}, runtime),
        Check("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not found"),
        check_directory("data-directory", args.data_dir),
        check_directory("log-directory", args.log_dir),
    ]
    if runtime not in {"onnx", "rknn"}:
        return checks

    defaults = {
        "FACE_API_RETINAFACE_ONNX": "models/onnx/Retinaface_mobilenet0.25.onnx",
        "FACE_API_CVLFACE_RECOGNITION_ONNX": "models/onnx/cvlface_adaface_ir50_webface4m.onnx",
        "FACE_API_CVLFACE_ALIGNER_ONNX": "models/onnx/cvlface_dfa_mobilenet.onnx",
        "FACE_API_RETINAFACE_RKNN": "models/rknn/retinaface-mobilenet0.25-480x720-int8.rknn",
        "FACE_API_CVLFACE_RECOGNITION_RKNN": "models/rknn/recognition-int8.rknn",
        "FACE_API_RKNN_MANIFEST": "models/rknn/manifest.json",
    }
    path_for = lambda key: resolve_path(effective.get(key, defaults[key]), app_dir)

    if runtime == "onnx":
        model_paths = [
            path_for("FACE_API_RETINAFACE_ONNX"),
            path_for("FACE_API_CVLFACE_RECOGNITION_ONNX"),
            path_for("FACE_API_CVLFACE_ALIGNER_ONNX"),
        ]
        checks.extend(check_file(f"model:{path.name}", path) for path in model_paths)
        if not args.skip_runtime_import:
            provider = effective.get("FACE_API_PROVIDER", "auto").lower()
            provider = "cuda" if provider == "cuda" else "cpu"
            checks.append(check_onnx_runtime(provider))
        return checks

    checks.append(Check("architecture", platform.machine().lower() in {"aarch64", "arm64"}, platform.machine()))
    detector = path_for("FACE_API_RETINAFACE_RKNN")
    recognition = path_for("FACE_API_CVLFACE_RECOGNITION_RKNN")
    manifest = path_for("FACE_API_RKNN_MANIFEST")
    aligner = path_for("FACE_API_CVLFACE_ALIGNER_ONNX")
    for path in (detector, recognition, manifest, aligner):
        checks.append(check_file(f"model:{path.name}", path))
    if manifest.is_file():
        checks.extend(verify_rknn_manifest(manifest, [detector, recognition]))
    if not args.skip_runtime_import:
        library = Path(effective.get("FACE_API_RKNN_RUNTIME_LIBRARY", "/usr/lib/librknnrt.so"))
        device = Path(effective.get("FACE_API_RKNN_DEVICE", "/dev/dri/renderD129"))
        checks.extend(check_rknn_runtime(library, device))
    if args.deep_rknn_check:
        checks.extend(run_deep_rknn_checks(app_dir, Path(sys.executable), manifest, [detector, recognition]))
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path("/etc/face-serve/face-serve.env"))
    parser.add_argument("--app-dir", type=Path, default=Path("/opt/face-serve/current"))
    parser.add_argument("--data-dir", type=Path, default=Path("/var/lib/face-serve"))
    parser.add_argument("--log-dir", type=Path, default=Path("/var/log/face-serve"))
    parser.add_argument("--runtime", choices=("onnx", "rknn"))
    parser.add_argument("--skip-runtime-import", action="store_true")
    parser.add_argument("--deep-rknn-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        checks = collect_checks(args)
    except Exception as exc:
        checks = [Check("preflight", False, str(exc))]
    failed = [check for check in checks if check.required and not check.ok]
    if args.json:
        print(json.dumps({"ok": not failed, "checks": [asdict(check) for check in checks]}, ensure_ascii=False))
    else:
        for check in checks:
            print(f"{'OK' if check.ok else 'FAIL':4} {check.name}: {check.detail}")
        print(f"preflight={'passed' if not failed else 'failed'} failures={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Convert the project's ONNX models to auditable RKNN artifacts.

This module deliberately keeps its metadata and manifest helpers independent of
``rknn-toolkit2``.  That lets CI validate the manifest contract without a
vendor wheel, while the command line entry point imports Toolkit2 only when an
actual conversion has been requested.  Tensor contracts come from the
target-side ``rknn_query`` probe (not undocumented Toolkit2 Python methods).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "rknn"
MANIFEST_NAME = "manifest.json"


# Store source paths relative to the repository so a manifest can be moved
# together with a checked-out source tree.  RKNN artifact paths are relative to
# the manifest itself; this is also what ``load_input_spec`` expects at runtime.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "yolo": {
        "onnx": Path("models/onnx/yolov12n-face.onnx"),
        "mean": [0, 0, 0],
        "std": [255, 255, 255],
    },
    "recognition": {
        "onnx": Path("models/onnx/cvlface_adaface_ir50_webface4m.onnx"),
        "mean": [127.5, 127.5, 127.5],
        "std": [127.5, 127.5, 127.5],
    },
    "aligner": {
        "onnx": Path("models/onnx/cvlface_dfa_mobilenet.onnx"),
        "mean": [127.5, 127.5, 127.5],
        "std": [127.5, 127.5, 127.5],
    },
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_primitive(value: Any) -> Any:
    """Convert NumPy/vendor scalar values to JSON-compatible primitives."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Tensor metadata value is not JSON serializable: {value!r}")


def _normalise_layout(raw_layout: Any, shape: list[int]) -> str:
    layout = str(_json_primitive(raw_layout)).lower()
    aliases = {
        "nchw": "nchw",
        "nhwc": "nhwc",
        "nc1hwc2": "nc1hwc2",
        "undefined": "undefined",
    }
    if layout in aliases:
        return aliases[layout]
    # Toolkit2 versions sometimes expose a numeric fmt enum.  Do not guess:
    # requiring a known layout avoids emitting an unusable runtime manifest.
    raise ValueError(f"Unsupported RKNN tensor layout {raw_layout!r} for shape {shape}")


def _normalise_dtype(raw_dtype: Any) -> str:
    dtype = str(_json_primitive(raw_dtype)).lower()
    aliases = {
        "float": "float32",
        "fp32": "float32",
        "float32": "float32",
        "float16": "float16",
        "fp16": "float16",
        "int8": "int8",
        "uint8": "uint8",
        "int16": "int16",
        "uint16": "uint16",
        "int32": "int32",
        "uint32": "uint32",
    }
    if dtype in aliases:
        return aliases[dtype]
    raise ValueError(f"Unsupported RKNN tensor dtype {raw_dtype!r}")


def serialise_tensor_metadata(tensors: Iterable[Any]) -> list[dict[str, Any]]:
    """Translate Toolkit2 tensor metadata to the portable manifest contract.

    Toolkit2 may provide attributes on an object or mapping keys, so accepting
    both avoids hard-coding a particular release's wrapper representation.
    """

    entries: list[dict[str, Any]] = []
    for tensor in tensors:
        if isinstance(tensor, Mapping):
            def get(name: str, _tensor: Mapping[str, Any] = tensor) -> Any:
                if name in _tensor:
                    return _tensor[name]
                # Older Toolkit2 metadata uses the C API field names.
                aliases = {"layout": "fmt", "dtype": "type"}
                return _tensor[aliases.get(name, name)]
        else:
            def get(name: str, _tensor: Any = tensor) -> Any:
                if hasattr(_tensor, name):
                    return getattr(_tensor, name)
                aliases = {"layout": "fmt", "dtype": "type"}
                return getattr(_tensor, aliases.get(name, name))
        try:
            name = _json_primitive(get("name"))
            raw_shape = get("shape")
            raw_dtype = get("dtype")
            raw_layout = get("layout")
        except (AttributeError, KeyError) as exc:
            raise ValueError("RKNN tensor metadata requires name, shape, dtype, and layout") from exc
        if not isinstance(name, str) or not name:
            raise ValueError("RKNN tensor metadata name must be a non-empty string")
        if not isinstance(raw_shape, (list, tuple)):
            raise ValueError(f"RKNN tensor {name!r} has no sequence shape")
        shape = [_json_primitive(item) for item in raw_shape]
        if not shape or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in shape):
            raise ValueError(f"RKNN tensor {name!r} has invalid shape: {shape!r}")
        entries.append(
            {
                "name": name,
                "layout": _normalise_layout(raw_layout, shape),
                "shape": shape,
                "dtype": _normalise_dtype(raw_dtype),
            }
        )
    if not entries:
        raise ValueError("RKNN tensor metadata must not be empty")
    return entries


def build_manifest_entry(
    name: str,
    source: Path,
    output: Path,
    *,
    precision: str,
    toolkit_version: str,
    inputs: Iterable[Any] | None = None,
    outputs: Iterable[Any] | None = None,
    conversion_log: Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a manifest record only after successful conversion artifacts exist."""

    source = Path(source)
    output = Path(output)
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model name: {name!r}")
    if precision not in {"fp", "int8"}:
        raise ValueError(f"Unsupported precision: {precision!r}")
    if not source.is_file() or not output.is_file():
        raise FileNotFoundError("Cannot build manifest entry before source and RKNN artifacts exist")

    entry: dict[str, Any] = {
        "name": name,
        "precision": precision,
        "target_platform": "rk3588",
        "toolkit_version": toolkit_version,
        "onnx_path": _project_relative_path(source),
        "onnx_sha256": sha256_file(source),
        # Output is intentionally relative to manifest.json, not the invoking
        # working directory; the Lite2 runtime resolves it from manifest_dir.
        "rknn_path": output.name,
        "rknn_sha256": sha256_file(output),
        "inputs": serialise_tensor_metadata(inputs) if inputs is not None else [],
        "outputs": serialise_tensor_metadata(outputs) if outputs is not None else [],
        "conversion_log": conversion_log.name if conversion_log is not None else "",
        "created_at": created_at or datetime.now(UTC).isoformat(),
    }
    return entry


def _project_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # Caller-selected files outside the source tree need a stable absolute
        # path rather than an ambiguous cwd-relative path.
        return str(resolved)


def _return_code(operation: str, result: Any) -> None:
    if result != 0:
        raise RuntimeError(f"RKNN {operation} failed: return code {result!r}")


def _release_checked(rknn: Any) -> None:
    release = getattr(rknn, "release", None)
    if callable(release):
        # Most Toolkit2 releases return None here; versions that return a code
        # are still checked just like config/build/export/init.
        result = release()
        if result not in (None, 0):
            raise RuntimeError(f"RKNN release failed: return code {result!r}")


def configure_and_build(
    rknn: Any,
    spec: Mapping[str, Any],
    *,
    precision: str,
    calibration_list: Path | None,
) -> dict[str, str]:
    """Run the code-returning Toolkit2 conversion calls with strict checking."""

    _return_code(
        "config",
        rknn.config(
            target_platform="rk3588",
            mean_values=[spec["mean"]],
            std_values=[spec["std"]],
        ),
    )
    _return_code("load_onnx", rknn.load_onnx(model=str(spec["onnx"])))
    _return_code(
        "build",
        rknn.build(
            do_quantization=precision == "int8",
            dataset=str(calibration_list) if precision == "int8" else None,
        ),
    )


def _validate_calibration_list(path: Path | None, precision: str) -> Path | None:
    if precision == "fp":
        if path is not None:
            raise ValueError("--calibration-list is valid only with --precision int8")
        return None
    if path is None:
        raise ValueError("--calibration-list is required with --precision int8")
    supplied_path = Path(path).expanduser()
    if not supplied_path.is_absolute():
        raise ValueError("Calibration list must be an existing absolute file path")
    path = supplied_path.resolve()
    if not path.is_file():
        raise ValueError("Calibration list must be an existing absolute file path")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Calibration list must contain at least one image path")
    supported_extensions = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    non_absolute = [line for line in lines if not Path(line).is_absolute()]
    invalid_images = []
    for line in lines:
        image = Path(line)
        if not image.is_file() or not os.access(image, os.R_OK) or image.suffix.lower() not in supported_extensions:
            invalid_images.append(line)
            continue
        try:
            with image.open("rb") as handle:
                handle.read(1)
        except OSError:
            invalid_images.append(line)
    if non_absolute or invalid_images:
        raise ValueError(
            "Calibration list entries must be readable, existing absolute image files"
            f" (non-absolute={len(non_absolute)}, invalid={len(invalid_images)})"
        )
    return path


def load_target_probe(path: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the JSON emitted by ``probe_rknn_tensor_metadata`` on RK3588.

    The vendor-supported way to obtain tensor attributes is C Runtime
    ``rknn_query`` after ``rknn_init``.  The converter therefore refuses to
    invent a Python Toolkit2 metadata API.  The probe is deliberately an
    explicit handoff because the x86_64 build host cannot query the target
    RKNN runtime ABI.
    """

    probe_path = Path(path)
    if not probe_path.is_file():
        raise FileNotFoundError(f"RKNN target metadata probe does not exist: {probe_path}")
    try:
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        probed_path = Path(probe["rknn_path"])
        probe_sha256 = probe["rknn_sha256"]
        inputs = serialise_tensor_metadata(probe["inputs"])
        outputs = serialise_tensor_metadata(probe["outputs"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid RKNN target metadata probe: {probe_path}") from exc
    if probed_path.name != output.name:
        raise ValueError(
            "RKNN target metadata probe is for a different artifact: "
            f"{probed_path.name!r} != {output.name!r}"
        )
    if not isinstance(probe_sha256, str) or len(probe_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in probe_sha256.lower()
    ):
        raise ValueError("RKNN target metadata probe has an invalid rknn_sha256")
    if probe_sha256.lower() != sha256_file(output):
        raise ValueError("RKNN target metadata probe SHA-256 does not match the retained artifact")
    unsupported_inputs = [item["layout"] for item in inputs if item["layout"] not in {"nchw", "nhwc"}]
    if unsupported_inputs:
        raise ValueError(
            "RKNN target metadata probe inputs must use NCHW or NHWC layout, got "
            f"{unsupported_inputs}"
        )
    return inputs, outputs


def _onnx_preflight_input(source: Path) -> np.ndarray:
    """Build a deterministic simulator input from the ONNX graph only.

    This input is for the Toolkit2 simulator gate, never for manifest tensor
    metadata.  The runtime manifest is populated exclusively by the target C
    query probe above.
    """

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX is required to construct the Toolkit2 preflight input") from exc
    model = onnx.load(str(source))
    initializers = {item.name for item in model.graph.initializer}
    graph_inputs = [item for item in model.graph.input if item.name not in initializers]
    if len(graph_inputs) != 1:
        raise RuntimeError(f"ONNX preflight requires exactly one graph input, got {len(graph_inputs)}")
    tensor_type = graph_inputs[0].type.tensor_type
    if not tensor_type.HasField("shape"):
        raise RuntimeError("ONNX preflight input has no tensor shape")
    dimensions = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise RuntimeError("ONNX preflight input shape must be static and positive")
        dimensions.append(dimension.dim_value)
    dtypes = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.INT8: np.int8,
        onnx.TensorProto.UINT8: np.uint8,
        onnx.TensorProto.INT16: np.int16,
        onnx.TensorProto.UINT16: np.uint16,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.UINT32: np.uint32,
    }
    try:
        dtype = dtypes[tensor_type.elem_type]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported ONNX preflight input element type: {tensor_type.elem_type}") from exc
    return np.zeros(tuple(dimensions), dtype=dtype)


def _toolkit_version(rknn: Any) -> str:
    for attr in ("version", "__version__"):
        value = getattr(rknn, attr, None)
        if value:
            return str(value)
    try:
        from importlib.metadata import version

        return version("rknn-toolkit2")
    except Exception:
        return "unknown"


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {"models": []}
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict) or not isinstance(existing.get("models"), list):
        raise ValueError("Existing RKNN manifest must contain a models list")
    return existing


def _runtime_section(aligner_runtime: str, aligner_onnx_path: Path | None) -> dict[str, str]:
    if aligner_runtime == "rknn":
        if aligner_onnx_path is not None:
            raise ValueError("--aligner-onnx-path is only valid with --aligner-runtime onnx-cpu")
        return {"aligner_runtime": "rknn"}
    if aligner_runtime != "onnx-cpu":
        raise ValueError(f"Unsupported aligner runtime: {aligner_runtime!r}")
    if aligner_onnx_path is None:
        raise ValueError("--aligner-onnx-path is required with --aligner-runtime onnx-cpu")
    path = Path(aligner_onnx_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CPU fallback aligner does not exist: {path}")
    return {
        "aligner_runtime": "onnx-cpu",
        "aligner_onnx_path": str(path),
        "aligner_onnx_sha256": sha256_file(path),
    }


def update_manifest_runtime(
    manifest_path: Path,
    *,
    aligner_runtime: str,
    aligner_onnx_path: Path | None = None,
) -> None:
    """Atomically record the active alignment implementation for deployment."""

    if not Path(manifest_path).is_file():
        raise ValueError("Cannot set manifest runtime without non-empty models")
    existing = _read_manifest(manifest_path)
    if not existing["models"]:
        raise ValueError("Cannot set manifest runtime without non-empty models")
    existing["runtime"] = _runtime_section(aligner_runtime, aligner_onnx_path)
    _atomic_write_json(manifest_path, existing)


def update_manifest(
    manifest_path: Path,
    entry: Mapping[str, Any],
    *,
    runtime: Mapping[str, str] | None = None,
) -> None:
    """Replace one model and its runtime declaration in one atomic write."""

    if not entry.get("inputs") or not entry.get("outputs"):
        raise ValueError("A successful manifest entry must include input and output metadata")
    existing = _read_manifest(manifest_path)
    models = existing["models"]
    replacement_key = (entry["name"], entry["precision"])
    models = [
        model
        for model in models
        if (model.get("name"), model.get("precision")) != replacement_key
    ]
    models.append(dict(entry))
    existing["models"] = models
    if runtime is not None:
        existing["runtime"] = dict(runtime)
    _atomic_write_json(manifest_path, existing)


def _write_success_log(path: Path, *, source: Path, output: Path, precision: str, toolkit_version: str) -> None:
    path.write_text(
        "\n".join(
            (
                "status=success",
                f"source={_project_relative_path(source)}",
                f"output={output.name}",
                f"precision={precision}",
                f"toolkit_version={toolkit_version}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def convert_model(
    *,
    rknn: Any,
    model_name: str,
    precision: str,
    calibration_list: Path | None,
    output_dir: Path,
) -> None:
    """Build and simulator-preflight an artifact for a later target query."""

    spec = MODEL_SPECS[model_name]
    source = PROJECT_ROOT / spec["onnx"]
    if not source.is_file():
        raise FileNotFoundError(f"ONNX source model does not exist: {source}")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{model_name}-{precision}.rknn"
    log_path = output_dir / f"{model_name}-{precision}.log"
    manifest_path = output_dir / MANIFEST_NAME
    # A stale previous binary is never evidence of a new successful build.
    for stale in (output, log_path):
        stale.unlink(missing_ok=True)

    release_attempted = False
    try:
        configure_and_build(rknn, {**spec, "onnx": source}, precision=precision, calibration_list=calibration_list)
        _return_code("export_rknn", rknn.export_rknn(str(output)))
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"RKNN export did not create a non-empty artifact: {output}")
        _return_code("init_runtime", rknn.init_runtime(target="rk3588"))
        inference = rknn.inference(inputs=[_onnx_preflight_input(source)])
        if inference is None or not isinstance(inference, (list, tuple)) or not inference:
            raise RuntimeError("RKNN simulator inference did not return output tensors")
        release_attempted = True
        _release_checked(rknn)
        # The artifact is intentionally retained after a successful host
        # build/simulator gate so it can be copied to RK3588 for the C runtime
        # query. No log or manifest is written in this stage.
        return {
            "status": "awaiting_target_probe",
            "artifact": str(output),
            "next_step": "copy artifact to RK3588, run scripts/probe_rknn_tensor_metadata, then finalize with --finalize --metadata-probe <probe.json>",
        }
    except Exception:
        # Export/build output must never survive a failed preflight and be
        # mistaken for deployable output.  Keep prior manifest entries intact.
        output.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)
        release = getattr(rknn, "release", None)
        if callable(release) and not release_attempted:
            # Keep the original conversion failure as the actionable error;
            # artifact cleanup is already enforced above.
            release()
        raise


def finalize_manifest(
    *,
    model_name: str,
    precision: str,
    output_dir: Path,
    metadata_probe: Path,
    toolkit_version: str,
    aligner_runtime: str = "rknn",
    aligner_onnx_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically record an existing, target-probed artifact without rebuilding it."""

    spec = MODEL_SPECS[model_name]
    source = PROJECT_ROOT / spec["onnx"]
    output_dir = Path(output_dir).resolve()
    output = output_dir / f"{model_name}-{precision}.rknn"
    log_path = output_dir / f"{model_name}-{precision}.log"
    if not source.is_file() or not output.is_file() or output.stat().st_size == 0:
        raise FileNotFoundError("Finalization requires the retained ONNX source and non-empty RKNN artifact")
    # Validate the deployment runtime contract before changing either manifest
    # entry or success log; a bad CPU fallback declaration must be atomic too.
    runtime = _runtime_section(aligner_runtime, aligner_onnx_path)
    inputs, outputs = load_target_probe(metadata_probe, output)
    entry = build_manifest_entry(
        model_name,
        source,
        output,
        precision=precision,
        toolkit_version=toolkit_version,
        inputs=inputs,
        outputs=outputs,
        conversion_log=log_path,
    )
    _write_success_log(log_path, source=source, output=output, precision=precision, toolkit_version=toolkit_version)
    try:
        update_manifest(output_dir / MANIFEST_NAME, entry, runtime=runtime)
    except Exception:
        log_path.unlink(missing_ok=True)
        raise
    return entry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--precision", choices=("fp", "int8"), required=True)
    parser.add_argument("--calibration-list", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metadata-probe",
        type=Path,
        help="JSON emitted by target-side scripts/probe_rknn_tensor_metadata",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="write a manifest for an existing, target-probed artifact without rebuilding it",
    )
    parser.add_argument("--toolkit-version", default="2.3.2")
    parser.add_argument("--aligner-runtime", choices=("rknn", "onnx-cpu"), default="rknn")
    parser.add_argument("--aligner-onnx-path", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.finalize:
            if args.calibration_list is not None:
                raise ValueError("--calibration-list is not used with --finalize")
            if args.metadata_probe is None:
                raise ValueError("--finalize requires --metadata-probe")
            entry = finalize_manifest(
                model_name=args.model,
                precision=args.precision,
                output_dir=args.output_dir,
                metadata_probe=args.metadata_probe,
                toolkit_version=args.toolkit_version,
                aligner_runtime=args.aligner_runtime,
                aligner_onnx_path=args.aligner_onnx_path,
            )
            print(json.dumps(entry, indent=2, sort_keys=True))
            return 0
        if args.metadata_probe is not None:
            raise ValueError("--metadata-probe is only valid with --finalize")
        if args.aligner_runtime != "rknn" or args.aligner_onnx_path is not None:
            raise ValueError("--aligner-runtime and --aligner-onnx-path are only valid with --finalize")
        calibration_list = _validate_calibration_list(args.calibration_list, args.precision)
        # Keep this import here: unit tests and host-only tooling must not need
        # the x86_64 vendor Toolkit2 wheel.
        from rknnlite.api import RKNNLite

        from rknn.api import RKNN

        rknn = RKNN(verbose=True)
        staged = convert_model(
            rknn=rknn,
            model_name=args.model,
            precision=args.precision,
            calibration_list=calibration_list,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"RKNN conversion failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(staged, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

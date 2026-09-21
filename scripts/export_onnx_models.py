from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OnnxOutputPaths:
    recognition: Path
    aligner: Path
    yolo: Path


@dataclass(frozen=True)
class RecognitionExportSpec:
    model_dir: Path
    output_path: Path
    input_names: tuple[str, ...]
    output_names: tuple[str, ...] = ("embedding",)
    uses_keypoints: bool = False


def default_output_paths(models_root: Path) -> OnnxOutputPaths:
    output_dir = models_root / "onnx"
    return OnnxOutputPaths(
        recognition=output_dir / "cvlface_adaface_vit_base_kprpe_webface4m.onnx",
        aligner=output_dir / "cvlface_dfa_mobilenet.onnx",
        yolo=output_dir / "yolov12l-face.onnx",
    )


def build_cvlface_dummy_inputs(batch_size: int = 1):
    import torch

    image = torch.zeros(batch_size, 3, 112, 112, dtype=torch.float32)
    keypoints = torch.tensor(
        [[[0.32, 0.38], [0.68, 0.38], [0.50, 0.55], [0.38, 0.72], [0.62, 0.72]]],
        dtype=torch.float32,
    ).repeat(batch_size, 1, 1)
    return image, keypoints


def build_aligner_dummy_input(batch_size: int = 1):
    import torch

    return torch.zeros(batch_size, 3, 112, 112, dtype=torch.float32)


def build_ir_recognition_dummy_input(batch_size: int = 1):
    import torch

    return torch.zeros(batch_size, 3, 112, 112, dtype=torch.float32)


def default_ir_recognition_specs(models_root: Path, output_dir: Path | None = None) -> list[RecognitionExportSpec]:
    models_root = Path(models_root)
    output_dir = Path(output_dir) if output_dir is not None else models_root / "onnx"
    model_names = (
        "cvlface_adaface_ir18_webface4m",
        "cvlface_adaface_ir50_webface4m",
        "cvlface_adaface_ir101_webface12m",
    )
    return [
        RecognitionExportSpec(
            model_dir=models_root / "cvlface" / model_name,
            output_path=output_dir / f"{model_name}.onnx",
            input_names=("image",),
        )
        for model_name in model_names
    ]


def make_traceable_rel_keypoints(keypoints: Any, query: Any):
    import math
    import torch

    seq_length = query.shape[1]
    side = int(math.sqrt(seq_length))
    coord = torch.linspace(0, 1, side + 1, device=query.device, dtype=query.dtype)
    coord = (coord[:-1] + coord[1:]) / 2
    x, y = torch.meshgrid(coord, coord, indexing="ij")
    grid = torch.stack([y, x], dim=-1).reshape(-1, 2).unsqueeze(0).unsqueeze(-2)
    diff = grid - keypoints.unsqueeze(-3)
    return diff.flatten(2)


def patch_traceable_keypoints(model: Any) -> None:
    import sys

    inner = getattr(model, "model", model)
    net = getattr(inner, "net", inner)
    forward_features = getattr(net, "forward_features", None)
    if forward_features is None:
        return
    make_kprpe_input = forward_features.__func__.__globals__.get("make_kprpe_input")
    if make_kprpe_input is None:
        return
    relative_keypoints = make_kprpe_input.__globals__.get("relative_keypoints")
    if relative_keypoints is not None:
        relative_keypoints.make_rel_keypoints = make_traceable_rel_keypoints
    for module_name, module in sys.modules.items():
        if module_name.endswith("RPE.KPRPE.kprpe_shared") and hasattr(module, "RPEIndexFunction"):
            module.RPEIndexFunction = None
    for block in getattr(net, "blocks", []):
        rpe_k = getattr(getattr(block, "attn", None), "rpe_k", None)
        if rpe_k is not None and hasattr(rpe_k, "_rp_bucket_buf"):
            rpe_k._rp_bucket_buf = (None, None, None)


class RecognitionExportWrapper:
    def __init__(self, model: Any, uses_keypoints: bool = True) -> None:
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, inner: Any, inner_uses_keypoints: bool) -> None:
                super().__init__()
                self.inner = inner
                self.inner_uses_keypoints = inner_uses_keypoints

            def forward(self, image, keypoints=None):
                if self.inner_uses_keypoints:
                    output = self.inner(image, keypoints)
                else:
                    output = self.inner(image)
                if isinstance(output, dict):
                    for key in ("feature", "features", "embedding", "embeddings", "logits"):
                        value = output.get(key)
                        if value is not None:
                            return value
                if isinstance(output, (list, tuple)):
                    return output[0]
                return output

        self.module = _Wrapper(model, uses_keypoints)


class AlignerExportWrapper:
    def __init__(self, model: Any) -> None:
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, image):
                output = self.inner(image)
                if isinstance(output, dict):
                    aligned_x = output.get("aligned_x", output.get("image", image))
                    aligned_ldmks = output.get("aligned_ldmks", output.get("keypoints"))
                    score = output.get("score")
                    return aligned_x, aligned_ldmks, aligned_ldmks, score
                if isinstance(output, (list, tuple)):
                    return tuple(output[:6])
                return output

        self.module = _Wrapper(model)


def export_cvlface_recognition(model_dir: Path, output_path: Path, opset: int = 17, dynamic_batch: bool = False) -> None:
    import torch

    model = load_local_cvlface_model(model_dir).eval()
    patch_traceable_keypoints(model)
    wrapper = RecognitionExportWrapper(model).module.eval()
    image, keypoints = build_cvlface_dummy_inputs()
    image.requires_grad_(True)
    keypoints.requires_grad_(True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "keypoints": {0: "batch"},
            "embedding": {0: "batch"},
        }
    torch.onnx.export(
        wrapper,
            (image, keypoints),
            str(output_path),
            input_names=["image", "keypoints"],
            output_names=["embedding"],
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            dynamo=False,
        )


def export_ir_cvlface_recognition(model_dir: Path, output_path: Path, opset: int = 17, dynamic_batch: bool = False) -> None:
    import torch

    model = load_local_cvlface_model(model_dir).eval()
    wrapper = RecognitionExportWrapper(model, uses_keypoints=False).module.eval()
    image = build_ir_recognition_dummy_input()
    image.requires_grad_(True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "embedding": {0: "batch"},
        }
    torch.onnx.export(
        wrapper,
        image,
        str(output_path),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=False,
        dynamo=False,
    )

def export_cvlface_aligner(model_dir: Path, output_path: Path, opset: int = 17, dynamic_batch: bool = False) -> None:
    import torch

    model = load_local_cvlface_model(model_dir).eval()
    wrapper = AlignerExportWrapper(model).module.eval()
    image = build_aligner_dummy_input()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_names = ["aligned_image", "original_keypoints", "aligned_keypoints", "align_score", "theta", "bbox"]
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"image": {0: "batch"}}
        dynamic_axes.update({name: {0: "batch"} for name in output_names})
    torch.onnx.export(
        wrapper,
            image,
            str(output_path),
            input_names=["image"],
            output_names=output_names,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            dynamo=False,
        )


def export_yolo(model_path: Path, output_path: Path, imgsz: int = 640, opset: int = 17, dynamic_batch: bool = False) -> None:
    from ultralytics import YOLO

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    exported = model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=dynamic_batch, simplify=False, device="cpu")
    exported_path = Path(exported)
    if exported_path.resolve() != output_path.resolve():
        shutil.copy2(exported_path, output_path)


def load_local_cvlface_model(model_dir: Path):
    from transformers import AutoModel, PreTrainedModel

    model_dir = model_dir.resolve()
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    previous_model_dir = os.environ.get("CVLFACE_LOCAL_MODEL_DIR")
    cwd = Path.cwd()
    os.environ["CVLFACE_LOCAL_MODEL_DIR"] = str(model_dir)
    sys.path.insert(0, str(model_dir))
    try:
        os.chdir(model_dir)
        return AutoModel.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    finally:
        os.chdir(cwd)
        if previous_model_dir is None:
            os.environ.pop("CVLFACE_LOCAL_MODEL_DIR", None)
        else:
            os.environ["CVLFACE_LOCAL_MODEL_DIR"] = previous_model_dir
        try:
            sys.path.remove(str(model_dir))
        except ValueError:
            pass


def write_manifest(paths: OnnxOutputPaths, manifest_path: Path) -> None:
    manifest = {
        "runtime": "onnxruntime",
        "providers": ["CPUExecutionProvider"],
        "models": {key: str(value) for key, value in asdict(paths).items()},
        "inputs": {
            "cvlface_recognition": {
                "image": "float32[N,3,112,112]",
                "keypoints": "float32[N,5,2]",
            },
            "cvlface_aligner": {
                "image": "float32[N,3,112,112]",
            },
            "yolo_face": {
                "image": "float32[N,3,640,640]",
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export local PyTorch face models to ONNX Runtime-compatible models.")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--recognition-model",
        type=Path,
        default=Path("models/cvlface/minchul__cvlface_adaface_vit_base_kprpe_webface4m"),
    )
    parser.add_argument("--aligner-model", type=Path, default=Path("models/cvlface/minchul__cvlface_DFA_mobilenet"))
    parser.add_argument("--yolo-model", type=Path, default=Path("models/yolov8n-face.pt"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--skip-aligner", action="store_true")
    parser.add_argument("--skip-recognition", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--export-ir-models", action="store_true", help="Export bundled CVLFace IR18/IR50/IR101 recognition models.")
    parser.add_argument("--only-ir-models", action="store_true", help="Export only bundled CVLFace IR18/IR50/IR101 recognition models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = default_output_paths(args.models_root)
    if args.output_dir is not None:
        paths = OnnxOutputPaths(
            recognition=args.output_dir / paths.recognition.name,
            aligner=args.output_dir / paths.aligner.name,
            yolo=args.output_dir / paths.yolo.name,
        )
    paths.recognition.parent.mkdir(parents=True, exist_ok=True)

    if args.only_ir_models:
        args.skip_recognition = True
        args.skip_aligner = True
        args.skip_yolo = True
        args.export_ir_models = True

    # if not args.skip_recognition:
    #     print(f"Exporting CVLFace recognition -> {paths.recognition}", flush=True)
    #     export_cvlface_recognition(args.recognition_model, paths.recognition, args.opset, args.dynamic_batch)
    # if not args.skip_aligner:
    #     print(f"Exporting CVLFace aligner -> {paths.aligner}", flush=True)
    #     export_cvlface_aligner(args.aligner_model, paths.aligner, args.opset, args.dynamic_batch)
    if not args.skip_yolo or True:
        print(f"Exporting YOLO face detector -> {paths.yolo}", flush=True)
        export_yolo(args.yolo_model, paths.yolo, args.imgsz, args.opset, args.dynamic_batch)
    # if args.export_ir_models:
    #     for spec in default_ir_recognition_specs(args.models_root, paths.recognition.parent):
    #         print(f"Exporting CVLFace IR recognition -> {spec.output_path}", flush=True)
    #         export_ir_cvlface_recognition(spec.model_dir, spec.output_path, args.opset, args.dynamic_batch)

    write_manifest(paths, paths.recognition.parent / "onnx_models.json")
    print(f"Wrote manifest -> {paths.recognition.parent / 'onnx_models.json'}", flush=True)


if __name__ == "__main__":
    main()




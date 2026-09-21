"""Rebuild a gallery into a new database for one explicit embedding runtime."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_api.core.config import ApiConfig
from face_api.core.recognition import RecognitionService
from face_api.core.subject_store import SubjectStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, required=True, help="Existing gallery data directory.")
    parser.add_argument("--target-data-dir", type=Path, required=True, help="Empty destination gallery data directory.")
    parser.add_argument("--runtime", choices=("onnx", "rknn"), required=True, help="Runtime used to regenerate embeddings.")
    parser.add_argument("--recognition-onnx", type=Path, default=ApiConfig().recognition_model_path)
    parser.add_argument("--aligner-onnx", type=Path, default=ApiConfig().aligner_model_path)
    parser.add_argument("--yolo-onnx", type=Path, default=ApiConfig().yolo_model_path)
    parser.add_argument("--recognition-rknn", type=Path, default=ApiConfig().recognition_rknn_model_path)
    parser.add_argument("--aligner-rknn", type=Path, default=ApiConfig().aligner_rknn_model_path)
    parser.add_argument("--yolo-rknn", type=Path, default=ApiConfig().yolo_rknn_model_path)
    parser.add_argument("--rknn-manifest", type=Path, default=ApiConfig().rknn_manifest_path)
    parser.add_argument("--rknn-core-mask", default="auto", choices=("auto", "0", "1", "2", "0_1_2"))
    parser.add_argument("--rknn-aligner-fallback", default="error", choices=("error", "onnx-cpu"))
    parser.add_argument("--provider", default=None, choices=("auto", "cuda", "cpu"))
    return parser


def build_config(args: argparse.Namespace) -> ApiConfig:
    """Construct every runtime-relevant field explicitly; never infer a runtime."""
    return ApiConfig(
        runtime=args.runtime,
        data_dir=args.target_data_dir,
        recognition_model_path=args.recognition_onnx,
        aligner_model_path=args.aligner_onnx,
        yolo_model_path=args.yolo_onnx,
        recognition_rknn_model_path=args.recognition_rknn,
        aligner_rknn_model_path=args.aligner_rknn,
        yolo_rknn_model_path=args.yolo_rknn,
        rknn_manifest_path=args.rknn_manifest,
        rknn_core_mask=args.rknn_core_mask,
        rknn_aligner_fallback=args.rknn_aligner_fallback,
        provider=args.provider,
    )


def rebuild(args: argparse.Namespace) -> int:
    target_db = args.target_data_dir / "face_api.db"
    if target_db.exists():
        raise FileExistsError(f"Refusing to overwrite existing target gallery database: {target_db}")

    source_store = SubjectStore(args.source_data_dir)
    target_store = SubjectStore(args.target_data_dir)
    service = RecognitionService(build_config(args))
    try:
        identity = service.embedding_identity()
        rebuilt = 0
        for record in source_store.list_subjects():
            image_bytes = Path(record.image_path).read_bytes()
            feature = service.extract_feature(image_bytes)
            target_store.create_subject(
                record.subject_id,
                record.name,
                image_bytes,
                feature,
                embedding_identity=identity,
            )
            rebuilt += 1
    finally:
        service.close()
    print(f"rebuilt={rebuilt} runtime={identity.runtime} model_sha256={identity.model_sha256}")
    return 0


def main() -> int:
    return rebuild(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

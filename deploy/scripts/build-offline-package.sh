#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE=""
VERSION=""
WHEELHOUSE=""
DEB_DIR=""
REQUIREMENTS=""
OUTPUT_DIR="${PROJECT_ROOT}/dist"

usage() {
    printf '%s\n' "Usage: $0 --profile PROFILE --version VERSION --wheelhouse DIR [--deb-dir DIR] [--requirements FILE] [--output-dir DIR]"
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --wheelhouse) WHEELHOUSE="$2"; shift 2 ;;
        --deb-dir) DEB_DIR="$2"; shift 2 ;;
        --requirements) REQUIREMENTS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) usage; exit 2 ;;
    esac
done

[[ "${VERSION}" =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'invalid or missing version\n' >&2; exit 2; }
[[ -d "${WHEELHOUSE}" ]] || { printf 'wheelhouse is required: %s\n' "${WHEELHOUSE}" >&2; exit 2; }
compgen -G "${WHEELHOUSE}/*.whl" >/dev/null || { printf 'wheelhouse contains no .whl files\n' >&2; exit 2; }

case "${PROFILE}" in
    rk3588-rknn)
        ARCHITECTURE="aarch64"
        CONFIG_SAMPLE="face-serve-rknn.env.example"
        MODEL_FILES=(
            "models/rknn/retinaface-mobilenet0.25-480x720-int8.rknn"
            "models/rknn/recognition-int8.rknn"
            "models/rknn/manifest.json"
            "models/onnx/cvlface_dfa_mobilenet.onnx"
        )
        ;;
    linux-x86_64-onnx-cpu)
        ARCHITECTURE="x86_64"
        CONFIG_SAMPLE="face-serve-onnx-cpu.env.example"
        MODEL_FILES=(
            "models/onnx/Retinaface_mobilenet0.25.onnx"
            "models/onnx/cvlface_adaface_ir50_webface4m.onnx"
            "models/onnx/cvlface_dfa_mobilenet.onnx"
        )
        ;;
    linux-x86_64-onnx-cuda)
        ARCHITECTURE="x86_64"
        CONFIG_SAMPLE="face-serve-onnx-cuda.env.example"
        MODEL_FILES=(
            "models/onnx/Retinaface_mobilenet0.25.onnx"
            "models/onnx/cvlface_adaface_ir50_webface4m.onnx"
            "models/onnx/cvlface_dfa_mobilenet.onnx"
        )
        [[ -n "${REQUIREMENTS}" ]] || { printf 'CUDA profile requires --requirements with onnxruntime-gpu\n' >&2; exit 2; }
        ;;
    *)
        printf 'unsupported profile: %s\n' "${PROFILE}" >&2
        exit 2
        ;;
esac

for model in "${MODEL_FILES[@]}"; do
    [[ -f "${PROJECT_ROOT}/${model}" ]] || { printf 'missing model: %s\n' "${model}" >&2; exit 1; }
done

command -v zstd >/dev/null || { printf 'zstd is required to build .tar.zst packages\n' >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${BUILD_DIR}"' EXIT
PACKAGE_NAME="face-serve-${VERSION}-${PROFILE}"
STAGE="${BUILD_DIR}/${PACKAGE_NAME}"
mkdir -p "${STAGE}/app" "${STAGE}/models" "${STAGE}/packages/wheelhouse"

cp -a "${PROJECT_ROOT}/face_api" "${PROJECT_ROOT}/face_core" "${PROJECT_ROOT}/scripts" "${STAGE}/app/"
cp -a "${PROJECT_ROOT}/pyproject.toml" "${PROJECT_ROOT}/uv.lock" "${STAGE}/app/"
cp -a "${PROJECT_ROOT}/deploy" "${STAGE}/app/"
cp -a "${PROJECT_ROOT}/deploy" "${STAGE}/"
cp -a "${WHEELHOUSE}/." "${STAGE}/packages/wheelhouse/"
find "${STAGE}/app" "${STAGE}/deploy" -type d -name __pycache__ -prune -exec rm -rf {} +
find "${STAGE}/app" "${STAGE}/deploy" -type f \( -name '*.pyc' -o -name '*.orig' -o -name '*.rej' \) -delete

for model in "${MODEL_FILES[@]}"; do
    destination="${STAGE}/$(dirname "${model}")"
    mkdir -p "${destination}"
    cp -a "${PROJECT_ROOT}/${model}" "${destination}/"
done

if [[ -n "${DEB_DIR}" ]]; then
    [[ -d "${DEB_DIR}" ]] || { printf 'deb directory does not exist: %s\n' "${DEB_DIR}" >&2; exit 2; }
    mkdir -p "${STAGE}/packages/debs"
    cp -a "${DEB_DIR}/." "${STAGE}/packages/debs/"
fi

if [[ -n "${REQUIREMENTS}" ]]; then
    cp "${REQUIREMENTS}" "${STAGE}/app/requirements.lock"
elif command -v uv >/dev/null; then
    UV_ARGS=(export --frozen --no-dev --no-emit-project)
    [[ "${PROFILE}" == "rk3588-rknn" ]] && UV_ARGS+=(--extra rknn)
    (cd "${PROJECT_ROOT}" && uv "${UV_ARGS[@]}" --output-file "${STAGE}/app/requirements.lock")
elif [[ -x "${PROJECT_ROOT}/.venv/bin/uv" ]]; then
    UV_ARGS=(export --frozen --no-dev --no-emit-project)
    [[ "${PROFILE}" == "rk3588-rknn" ]] && UV_ARGS+=(--extra rknn)
    (cd "${PROJECT_ROOT}" && .venv/bin/uv "${UV_ARGS[@]}" --output-file "${STAGE}/app/requirements.lock")
else
    printf 'uv is required to export requirements; pass --requirements instead\n' >&2
    exit 1
fi

SOURCE_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || true)"
SOURCE_COMMIT="${SOURCE_COMMIT:-unknown}"
python3 - "${STAGE}/manifest.json" "${VERSION}" "${PROFILE}" "${ARCHITECTURE}" "${CONFIG_SAMPLE}" "${SOURCE_COMMIT}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, version, profile, architecture, config_sample, source_commit = sys.argv[1:]
manifest = {
    "schema_version": 1,
    "name": "face-serve",
    "version": version,
    "source_commit": source_commit,
    "profile": profile,
    "architecture": architecture,
    "python": ">=3.11,<3.13",
    "config_sample": config_sample,
    "system_packages": ["ffmpeg", "libgl1", "libglib2.0-0", "libgomp1", "python3.11", "python3.11-venv"],
    "created_at": datetime.now(timezone.utc).isoformat(),
}
Path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
python3 "${STAGE}/deploy/scripts/generate_sbom.py" --package-dir "${STAGE}" --output-dir "${STAGE}/sbom"


(
    cd "${STAGE}"
    find app deploy models packages sbom -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    VERIFY_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
    VERIFY_PYTHON="$(command -v python3.11 || command -v python3)"
fi
FACE_SERVE_VERIFY_PYTHON="${VERIFY_PYTHON}" "${PROJECT_ROOT}/deploy/scripts/verify-package.sh" "${STAGE}"
ARCHIVE="${OUTPUT_DIR}/${PACKAGE_NAME}.tar.zst"
tar --zstd -C "${BUILD_DIR}" -cf "${ARCHIVE}" "${PACKAGE_NAME}"
(
    cd "${OUTPUT_DIR}"
    sha256sum "${PACKAGE_NAME}.tar.zst" > "${PACKAGE_NAME}.tar.zst.sha256"
)
printf '%s\n' "${ARCHIVE}"

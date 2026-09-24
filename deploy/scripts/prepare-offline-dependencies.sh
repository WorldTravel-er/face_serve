#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE=""
OUTPUT_DIR=""
REQUIREMENTS=""
WITH_DEBS=false

usage() {
    printf '%s\n' "Usage: $0 --profile PROFILE --output-dir DIR [--requirements FILE] [--with-debs]"
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --requirements) REQUIREMENTS="$2"; shift 2 ;;
        --with-debs) WITH_DEBS=true; shift ;;
        *) usage; exit 2 ;;
    esac
done

[[ -n "${OUTPUT_DIR}" ]] || { usage; exit 2; }
case "${PROFILE}" in
    rk3588-rknn)
        EXPECTED_ARCH="aarch64"
        ;;
    linux-x86_64-onnx-cpu)
        EXPECTED_ARCH="x86_64"
        ;;
    linux-x86_64-onnx-cuda)
        EXPECTED_ARCH="x86_64"
        [[ -n "${REQUIREMENTS}" ]] || { printf 'CUDA profile requires --requirements using onnxruntime-gpu\n' >&2; exit 2; }
        ;;
    *) printf 'unsupported profile: %s\n' "${PROFILE}" >&2; exit 2 ;;
esac
HOST_ARCH="$(uname -m)"
[[ "${HOST_ARCH}" == "arm64" ]] && HOST_ARCH="aarch64"
[[ "${HOST_ARCH}" == "amd64" ]] && HOST_ARCH="x86_64"
[[ "${HOST_ARCH}" == "${EXPECTED_ARCH}" ]] || { printf 'build host architecture mismatch: profile=%s expected=%s actual=%s\n' "${PROFILE}" "${EXPECTED_ARCH}" "${HOST_ARCH}" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}/wheelhouse"
if [[ -z "${REQUIREMENTS}" ]]; then
    REQUIREMENTS="${OUTPUT_DIR}/requirements.lock"
    UV_BIN="$(command -v uv || true)"
    [[ -n "${UV_BIN}" ]] || UV_BIN="${PROJECT_ROOT}/.venv/bin/uv"
    [[ -x "${UV_BIN}" ]] || { printf 'uv is required\n' >&2; exit 1; }
    UV_ARGS=(export --frozen --no-dev --no-emit-project)
    [[ "${PROFILE}" == "rk3588-rknn" ]] && UV_ARGS+=(--extra rknn)
    (cd "${PROJECT_ROOT}" && "${UV_BIN}" "${UV_ARGS[@]}" --output-file "${REQUIREMENTS}")
fi

PYTHON_BIN="${FACE_SERVE_DOWNLOAD_PYTHON:-$(command -v python3 || command -v python)}"
"${PYTHON_BIN}" -m pip --version >/dev/null || { printf 'the download Python has no pip: %s\n' "${PYTHON_BIN}" >&2; exit 1; }
if [[ "${PROFILE}" == "rk3588-rknn" ]]; then
    PIP_PLATFORMS=(
        --platform manylinux_2_28_aarch64
        --platform manylinux_2_27_aarch64
        --platform manylinux2014_aarch64
        --platform manylinux_2_17_aarch64
    )
else
    PIP_PLATFORMS=(
        --platform manylinux_2_28_x86_64
        --platform manylinux_2_27_x86_64
        --platform manylinux2014_x86_64
        --platform manylinux_2_17_x86_64
    )
fi
"${PYTHON_BIN}" -m pip download --requirement "${REQUIREMENTS}" --dest "${OUTPUT_DIR}/wheelhouse" --only-binary=:all: --python-version 311 --implementation cp --abi cp311 "${PIP_PLATFORMS[@]}"

if "${WITH_DEBS}"; then
    [[ "${EUID}" -eq 0 ]] || { printf '--with-debs requires root in a clean matching build host/container\n' >&2; exit 1; }
    mkdir -p "${OUTPUT_DIR}/debs/partial"
    apt-get update
    apt-get install --yes --download-only --reinstall         -o "Dir::Cache::archives=${OUTPUT_DIR}/debs"         ffmpeg libgl1 libglib2.0-0 libgomp1 python3.11 python3.11-venv
    rm -rf "${OUTPUT_DIR}/debs/partial" "${OUTPUT_DIR}/debs/lock"
fi

printf 'dependencies prepared at %s\n' "${OUTPUT_DIR}"

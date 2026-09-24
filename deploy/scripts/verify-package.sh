#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_DIR="${1:-.}"
MANIFEST="${PACKAGE_DIR}/manifest.json"
CHECKSUMS="${PACKAGE_DIR}/SHA256SUMS"

[[ -f "${MANIFEST}" ]] || { printf 'missing %s\n' "${MANIFEST}" >&2; exit 1; }
[[ -f "${CHECKSUMS}" ]] || { printf 'missing %s\n' "${CHECKSUMS}" >&2; exit 1; }
[[ -d "${PACKAGE_DIR}/app" ]] || { printf 'missing app directory\n' >&2; exit 1; }
[[ -d "${PACKAGE_DIR}/models" ]] || { printf 'missing models directory\n' >&2; exit 1; }
[[ -d "${PACKAGE_DIR}/deploy" ]] || { printf 'missing deploy directory\n' >&2; exit 1; }
[[ -f "${PACKAGE_DIR}/sbom/face-serve.spdx.json" ]] || { printf 'missing SPDX SBOM\n' >&2; exit 1; }
[[ -f "${PACKAGE_DIR}/sbom/third-party-licenses.json" ]] || { printf 'missing license inventory\n' >&2; exit 1; }

(
    cd "${PACKAGE_DIR}"
    sha256sum --check --strict SHA256SUMS
)

VERIFY_PYTHON="${FACE_SERVE_VERIFY_PYTHON:-$(command -v python3.11 || command -v python3)}"
"${VERIFY_PYTHON}" - "${MANIFEST}" <<'PY'
import json
import platform
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = {"schema_version", "name", "version", "source_commit", "profile", "architecture", "python"}
missing = sorted(required - manifest.keys())
if missing:
    raise SystemExit(f"manifest is missing keys: {missing}")
source_commit = str(manifest["source_commit"])
if source_commit != "unknown" and not (
    len(source_commit) in {40, 64} and all(character in "0123456789abcdef" for character in source_commit.lower())
):
    raise SystemExit(f"invalid source commit: {source_commit}")
if manifest["profile"] not in {"rk3588-rknn", "linux-x86_64-onnx-cpu", "linux-x86_64-onnx-cuda"}:
    raise SystemExit(f"unsupported profile: {manifest['profile']}")
machine = platform.machine().lower()
expected = str(manifest["architecture"]).lower()
aliases = {"arm64": "aarch64", "amd64": "x86_64"}
if aliases.get(machine, machine) != aliases.get(expected, expected):
    raise SystemExit(f"architecture mismatch: package={expected} host={machine}")
if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit(f"unsupported installer Python: {platform.python_version()}")
print(f"verified face-serve {manifest['version']} profile={manifest['profile']} source_commit={source_commit}")
PY

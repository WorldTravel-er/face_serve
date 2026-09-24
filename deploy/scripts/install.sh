#!/usr/bin/env bash
set -Eeuo pipefail

[[ "${EUID}" -eq 0 ]] || { printf 'install.sh must run as root\n' >&2; exit 1; }
[[ "$#" -eq 1 ]] || { printf 'Usage: %s PACKAGE_DIR_OR_ARCHIVE\n' "$0" >&2; exit 2; }

SOURCE="$1"
TEMP_DIR=""
if [[ -d "${SOURCE}" ]]; then
    PACKAGE_DIR="$(cd "${SOURCE}" && pwd)"
elif [[ -f "${SOURCE}" ]]; then
    TEMP_DIR="$(mktemp -d)"
    trap '[[ -z "${TEMP_DIR}" ]] || rm -rf "${TEMP_DIR}"' EXIT
    tar -xf "${SOURCE}" -C "${TEMP_DIR}"
    PACKAGE_DIR="$(find "${TEMP_DIR}" -mindepth 1 -maxdepth 1 -type d -print -quit)"
    [[ -n "${PACKAGE_DIR}" ]] || { printf 'archive does not contain a package directory\n' >&2; exit 1; }
else
    printf 'package not found: %s\n' "${SOURCE}" >&2
    exit 2
fi

"${PACKAGE_DIR}/deploy/scripts/verify-package.sh" "${PACKAGE_DIR}"
mapfile -t META < <(python3 - "${PACKAGE_DIR}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(data["version"])
print(data["profile"])
print(data["config_sample"])
PY
)
VERSION="${META[0]}"
PROFILE="${META[1]}"
CONFIG_SAMPLE="${META[2]}"

APP_ROOT="/opt/face-serve"
RELEASES_DIR="${APP_ROOT}/releases"
RELEASE_DIR="${RELEASES_DIR}/${VERSION}"
CURRENT_LINK="${APP_ROOT}/current"
CONFIG_DIR="/etc/face-serve"
DATA_DIR="/var/lib/face-serve"
LOG_DIR="/var/log/face-serve"
BACKUP_DIR="/var/backups/face-serve"
LIBEXEC_DIR="/usr/libexec/face-serve"
SYSTEMD_DIR="/etc/systemd/system"
LOCK_FILE="/run/lock/face-serve-install.lock"

exec 9>"${LOCK_FILE}"
flock -n 9 || { printf 'another face-serve install is running\n' >&2; exit 1; }

if ! id face-serve >/dev/null 2>&1; then
    useradd --system --home-dir "${DATA_DIR}" --shell /usr/sbin/nologin face-serve
fi
if [[ "${PROFILE}" == "rk3588-rknn" ]]; then
    usermod -a -G render,video face-serve
fi

install -d -o root -g root -m 0755 "${APP_ROOT}" "${RELEASES_DIR}" "${LIBEXEC_DIR}"
install -d -o face-serve -g face-serve -m 0750 "${DATA_DIR}" "${LOG_DIR}" "${BACKUP_DIR}"
install -d -o root -g face-serve -m 0750 "${CONFIG_DIR}"

if compgen -G "${PACKAGE_DIR}/packages/debs/*.deb" >/dev/null; then
    dpkg -i "${PACKAGE_DIR}"/packages/debs/*.deb || dpkg -i "${PACKAGE_DIR}"/packages/debs/*.deb
fi

PYTHON_BIN="$(command -v python3.11 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi
"${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] in {(3, 11), (3, 12)} else 1)'     || { printf 'Python 3.11 or 3.12 is required\n' >&2; exit 1; }

[[ ! -e "${RELEASE_DIR}" ]] || { printf 'release already installed: %s\n' "${RELEASE_DIR}" >&2; exit 1; }
STAGED_RELEASE="${RELEASES_DIR}/.${VERSION}.$$.staging"
trap 'rm -rf "${STAGED_RELEASE}"; [[ -z "${TEMP_DIR}" ]] || rm -rf "${TEMP_DIR}"' EXIT
mkdir -p "${STAGED_RELEASE}"
cp -a "${PACKAGE_DIR}/app/." "${STAGED_RELEASE}/"
cp -a "${PACKAGE_DIR}/models" "${STAGED_RELEASE}/models"
"${PYTHON_BIN}" -m venv "${STAGED_RELEASE}/.venv"
"${STAGED_RELEASE}/.venv/bin/python" -m pip install     --no-index --find-links "${PACKAGE_DIR}/packages/wheelhouse"     --requirement "${STAGED_RELEASE}/requirements.lock"
"${STAGED_RELEASE}/.venv/bin/python" -c 'import av, cv2, fastapi, numpy, onnxruntime, uvicorn'
printf '%s\n' "${VERSION}" > "${STAGED_RELEASE}/VERSION"
chmod -R a-w "${STAGED_RELEASE}"
mv "${STAGED_RELEASE}" "${RELEASE_DIR}"

for script in preflight.py healthcheck.py backup.py install.sh uninstall.sh; do
    install -o root -g root -m 0755 "${PACKAGE_DIR}/deploy/scripts/${script}" "${LIBEXEC_DIR}/${script}"
done
install -o root -g root -m 0755 "${PACKAGE_DIR}/deploy/scripts/facectl" /usr/sbin/facectl
install -o root -g root -m 0644 "${PACKAGE_DIR}/deploy/systemd/face-serve.service" "${SYSTEMD_DIR}/face-serve.service"
install -o root -g root -m 0644 "${PACKAGE_DIR}/deploy/systemd/face-serve-healthcheck.service" "${SYSTEMD_DIR}/face-serve-healthcheck.service"
install -o root -g root -m 0644 "${PACKAGE_DIR}/deploy/systemd/face-serve-healthcheck.timer" "${SYSTEMD_DIR}/face-serve-healthcheck.timer"

install -d -o root -g root -m 0755 "${SYSTEMD_DIR}/face-serve.service.d"
rm -f "${SYSTEMD_DIR}/face-serve.service.d/20-rknn.conf"
if [[ "${PROFILE}" == "rk3588-rknn" ]]; then
    install -o root -g root -m 0644 "${PACKAGE_DIR}/deploy/systemd/face-serve-rknn.conf" "${SYSTEMD_DIR}/face-serve.service.d/20-rknn.conf"
fi

if [[ ! -f "${CONFIG_DIR}/face-serve.env" ]]; then
    install -o root -g face-serve -m 0640 "${PACKAGE_DIR}/deploy/config/${CONFIG_SAMPLE}" "${CONFIG_DIR}/face-serve.env"
fi
if [[ ! -f "${CONFIG_DIR}/dependencies.conf" ]]; then
    install -o root -g root -m 0644 "${PACKAGE_DIR}/deploy/config/dependencies.conf.example" "${CONFIG_DIR}/dependencies.conf"
fi

PREVIOUS="$(readlink -f "${CURRENT_LINK}" || true)"
if [[ -n "${PREVIOUS}" && -d "${PREVIOUS}" ]]; then
    /usr/bin/python3 "${LIBEXEC_DIR}/backup.py" --data-dir "${DATA_DIR}" --backup-root "${BACKUP_DIR}" --label "before-${VERSION}"
fi

systemctl stop face-serve-healthcheck.timer face-serve.service 2>/dev/null || true
TEMP_LINK="${APP_ROOT}/.current.$$.tmp"
ln -s "${RELEASE_DIR}" "${TEMP_LINK}"
mv -Tf "${TEMP_LINK}" "${CURRENT_LINK}"
systemctl daemon-reload

if ! "${RELEASE_DIR}/.venv/bin/python" "${LIBEXEC_DIR}/preflight.py"     --env-file "${CONFIG_DIR}/face-serve.env" --app-dir "${CURRENT_LINK}"     --data-dir "${DATA_DIR}" --log-dir "${LOG_DIR}"; then
    if [[ -n "${PREVIOUS}" && -d "${PREVIOUS}" ]]; then
        TEMP_LINK="${APP_ROOT}/.current.$$.restore"
        ln -s "${PREVIOUS}" "${TEMP_LINK}"
        mv -Tf "${TEMP_LINK}" "${CURRENT_LINK}"
    fi
    printf 'preflight failed; previous release restored\n' >&2
    exit 1
fi

if ! systemctl enable --now face-serve.service face-serve-healthcheck.timer; then
    if [[ -n "${PREVIOUS}" && -d "${PREVIOUS}" ]]; then
        TEMP_LINK="${APP_ROOT}/.current.$$.restore"
        ln -s "${PREVIOUS}" "${TEMP_LINK}"
        mv -Tf "${TEMP_LINK}" "${CURRENT_LINK}"
        systemctl restart face-serve.service || true
    fi
    printf 'service start failed; previous release restored\n' >&2
    exit 1
fi

API_PORT="$(sed -n 's/^[[:space:]]*FACE_API_PORT[[:space:]]*=[[:space:]]*//p' "${CONFIG_DIR}/face-serve.env" | tail -n 1 | tr -d "'\"")"
API_PORT="${API_PORT:-8000}"
[[ "${API_PORT}" =~ ^[0-9]{1,5}$ ]] && ((API_PORT >= 1 && API_PORT <= 65535)) || { printf 'invalid FACE_API_PORT: %s\n' "${API_PORT}" >&2; exit 1; }
for attempt in $(seq 1 60); do
    if /usr/bin/python3 "${LIBEXEC_DIR}/healthcheck.py" --url "http://127.0.0.1:${API_PORT}/readyz" --state-file /run/face-serve-install-health >/dev/null 2>&1; then
        printf 'face-serve %s installed and ready\n' "${VERSION}"
        exit 0
    fi
    sleep 2
done

journalctl -u face-serve.service -n 100 --no-pager >&2 || true
if [[ -n "${PREVIOUS}" && -d "${PREVIOUS}" ]]; then
    systemctl stop face-serve.service
    TEMP_LINK="${APP_ROOT}/.current.$$.restore"
    ln -s "${PREVIOUS}" "${TEMP_LINK}"
    mv -Tf "${TEMP_LINK}" "${CURRENT_LINK}"
    systemctl start face-serve.service || true
fi
printf 'readiness timeout; previous release restored\n' >&2
exit 1

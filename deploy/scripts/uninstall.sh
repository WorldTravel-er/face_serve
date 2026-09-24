#!/usr/bin/env bash
set -Eeuo pipefail

[[ "${EUID}" -eq 0 ]] || { printf 'uninstall.sh must run as root\n' >&2; exit 1; }
PURGE_DATA=false
PURGE_RELEASES=false
for argument in "$@"; do
    case "${argument}" in
        --purge-data) PURGE_DATA=true ;;
        --purge-releases) PURGE_RELEASES=true ;;
        *) printf 'unknown option: %s\n' "${argument}" >&2; exit 2 ;;
    esac
done

systemctl disable --now face-serve-healthcheck.timer face-serve.service 2>/dev/null || true
rm -f     /etc/systemd/system/face-serve.service     /etc/systemd/system/face-serve-healthcheck.service     /etc/systemd/system/face-serve-healthcheck.timer     /etc/systemd/system/face-serve.service.d/20-rknn.conf     /usr/sbin/facectl
rm -rf /usr/libexec/face-serve
systemctl daemon-reload

if "${PURGE_RELEASES}"; then
    rm -rf /opt/face-serve/releases /opt/face-serve/current
fi
if "${PURGE_DATA}"; then
    rm -rf /var/lib/face-serve /var/log/face-serve /var/backups/face-serve /etc/face-serve
    userdel face-serve 2>/dev/null || true
fi

printf 'face-serve units removed; data/config and releases are preserved unless purge flags were supplied\n'

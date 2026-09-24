#!/usr/bin/env python3
"""Probe readiness and restart an active service only after repeated failures."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable


SERVICE_RE = re.compile(r"^[A-Za-z0-9_.@-]+\.service$")


def probe(url: str, timeout: float) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            payload = json.load(exc)
        except Exception:
            return False, f"HTTP {exc.code}"
        return False, f"HTTP {exc.code}: {payload}"
    except Exception as exc:
        return False, str(exc)
    try:
        ready = payload["code"] == 0 and payload["data"]["ready"] is True
    except (KeyError, TypeError):
        return False, f"unexpected readiness payload: {payload}"
    return ready, "ready" if ready else f"not ready: {payload}"


def read_failures(path: Path) -> int:
    try:
        return max(int(path.read_text(encoding="ascii").strip()), 0)
    except (FileNotFoundError, ValueError, OSError):
        return 0


def write_failures(path: Path, failures: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(f"{max(failures, 0)}\n")
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def service_is_active(service: str, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> bool:
    result = runner(
        ["systemctl", "is-active", "--quiet", service],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def restart_service(service: str, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> tuple[bool, str]:
    if not SERVICE_RE.fullmatch(service):
        return False, f"refusing invalid service name: {service}"
    result = runner(["systemctl", "try-restart", service], check=False, capture_output=True, text=True)
    detail = (result.stdout + result.stderr).strip()
    return result.returncode == 0, detail or f"try-restart exit={result.returncode}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    port = os.environ.get("FACE_API_PORT", "8000")
    parser.add_argument("--url", default=f"http://127.0.0.1:{port}/readyz")
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--state-file", type=Path, default=Path("/run/face-serve/health-failures"))
    parser.add_argument("--failure-threshold", type=int, default=3)
    parser.add_argument("--restart-service")
    parser.add_argument("--supervisor", action="store_true", help="Always exit zero so a systemd timer stays healthy.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    threshold = max(args.failure_threshold, 1)
    healthy, detail = probe(args.url, args.timeout)
    if healthy:
        write_failures(args.state_file, 0)
        print(f"READY {detail}")
        return 0

    failures = read_failures(args.state_file) + 1
    write_failures(args.state_file, failures)
    print(f"UNHEALTHY failures={failures}/{threshold} detail={detail}")

    if args.restart_service and failures >= threshold:
        if not service_is_active(args.restart_service):
            print(f"SKIP service={args.restart_service} is not active")
            write_failures(args.state_file, 0)
        else:
            restarted, restart_detail = restart_service(args.restart_service)
            print(f"{'RESTARTED' if restarted else 'RESTART_FAILED'} service={args.restart_service} {restart_detail}")
            if restarted:
                write_failures(args.state_file, 0)

    return 0 if args.supervisor else 1


if __name__ == "__main__":
    raise SystemExit(main())

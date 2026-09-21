from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen

from interface_test_logger import InterfacePacketLogger, default_log_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE_1 = Path(r"data/imgs/1.png")
DEFAULT_IMAGE_2 = Path(r"data/imgs/2.png")
DEFAULT_IMAGE_3 = Path(r"data/imgs/3.png")


@dataclass(frozen=True)
class SubjectSpec:
    subject_id: str
    name: str
    image_path: Path


@dataclass(frozen=True)
class ApiResult:
    status: int
    reason: str
    headers: dict[str, str]
    body: Any


def encode_image_base64(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"image file not found: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"image path is not a file: {resolved}")
    return base64.b64encode(resolved.read_bytes()).decode("ascii")


def build_subject_specs(args: argparse.Namespace) -> list[SubjectSpec]:
    return [
        SubjectSpec(args.subject1_id, args.subject1_name, Path(args.image1)),
        SubjectSpec(args.subject2_id, args.subject2_name, Path(args.image2)),
        SubjectSpec(args.subject3_id, args.subject3_name, Path(args.image3)),
    ]


def build_create_payload(subject: SubjectSpec) -> dict[str, Any]:
    return {
        "subject_id": subject.subject_id,
        "name": subject.name,
        "image_base64": encode_image_base64(subject.image_path),
    }


def build_update_payload(subject: SubjectSpec) -> dict[str, Any]:
    return {
        "name": subject.name,
        "image_base64": encode_image_base64(subject.image_path),
    }


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None,
    *,
    timeout: float,
    logger: InterfacePacketLogger | None = None,
) -> ApiResult:
    base = base_url.rstrip("/") + "/"
    url = urljoin(base, path.lstrip("/"))
    body_bytes: bytes | None = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    if logger is not None:
        logger.log_http_request(method=method, url=url, headers=headers, body=payload)

    request = Request(url, data=body_bytes, headers=headers, method=method.upper())
    try:
        with urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            reason = str(response.reason)
            response_headers = dict(response.headers.items())
            response_text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        status = int(exc.code)
        reason = str(exc.reason)
        response_headers = dict(exc.headers.items())
        response_text = exc.read().decode("utf-8", errors="replace")
    except URLError as exc:
        raise RuntimeError(f"failed to connect to {url}: {exc}") from exc

    try:
        response_body: Any = json.loads(response_text) if response_text else None
    except json.JSONDecodeError:
        response_body = response_text

    if logger is not None:
        logger.log_http_response(status=status, reason=reason, headers=response_headers, body=response_body)

    return ApiResult(status=status, reason=reason, headers=response_headers, body=response_body)


def is_success(result: ApiResult) -> bool:
    if not (200 <= result.status < 300):
        return False
    if isinstance(result.body, dict) and result.body.get("code") not in (None, 0):
        return False
    return True


def api_message(result: ApiResult) -> str:
    if isinstance(result.body, dict):
        return f"HTTP {result.status}, code={result.body.get('code')}, msg={result.body.get('msg')}"
    return f"HTTP {result.status}, body={result.body!r}"


def print_pass(name: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[PASS] {name}{suffix}")


def print_warn(name: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[WARN] {name}{suffix}")


def create_or_update_subject(subject: SubjectSpec, args: argparse.Namespace) -> None:
    create_payload = build_create_payload(subject)
    create_result = request_json(
        "POST",
        args.base_url,
        "/api/recognition/subjects",
        create_payload,
        timeout=args.timeout,
        logger=args.packet_logger,
    )
    if is_success(create_result):
        print_pass(f"registered {subject.subject_id}", api_message(create_result))
        return

    if not args.update_existing:
        raise AssertionError(f"register {subject.subject_id} failed and update-existing is disabled: {api_message(create_result)}")

    print_warn(f"register {subject.subject_id} failed, trying update", api_message(create_result))
    update_payload = build_update_payload(subject)
    update_result = request_json(
        "PUT",
        args.base_url,
        f"/api/recognition/subjects/{quote(subject.subject_id, safe='')}",
        update_payload,
        timeout=args.timeout,
        logger=args.packet_logger,
    )
    if not is_success(update_result):
        raise AssertionError(f"update {subject.subject_id} failed: {api_message(update_result)}")
    print_pass(f"updated existing {subject.subject_id}", api_message(update_result))


def verify_subject(subject: SubjectSpec, args: argparse.Namespace) -> None:
    result = request_json(
        "GET",
        args.base_url,
        f"/api/recognition/subjects/{quote(subject.subject_id, safe='')}",
        None,
        timeout=args.timeout,
        logger=args.packet_logger,
    )
    if not is_success(result):
        raise AssertionError(f"verify {subject.subject_id} failed: {api_message(result)}")
    data = result.body.get("data", {}) if isinstance(result.body, dict) else {}
    if data.get("subject_id") != subject.subject_id:
        raise AssertionError(f"verify {subject.subject_id} returned wrong subject_id: {result.body}")
    if data.get("name") != subject.name:
        raise AssertionError(f"verify {subject.subject_id} returned wrong name: {result.body}")
    print_pass(f"verified {subject.subject_id}", api_message(result))


def validate_base_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("base URL must look like http://127.0.0.1:8003")
    return value.rstrip("/")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register the three attached face images through the REST subject API.")
    parser.add_argument("--base-url", type=validate_base_url, default="http://127.0.0.1:8003", help="Face API base URL.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--log-dir", default="logs/interface_tests", help="Directory for packet log files.")
    parser.add_argument("--log-file", default=None, help="Optional exact packet log file path.")
    parser.add_argument("--no-update-existing", action="store_false", dest="update_existing", help="Fail instead of PUT updating when a subject already exists.")

    parser.add_argument("--subject1-id", default="qq_face_001", help="Subject ID for image 1.")
    parser.add_argument("--subject1-name", default="qq-face-1", help="Display name for image 1.")
    parser.add_argument("--image1", default=str(DEFAULT_IMAGE_1), help="Path to face image 1.")

    parser.add_argument("--subject2-id", default="qq_face_002", help="Subject ID for image 2.")
    parser.add_argument("--subject2-name", default="qq-face-2", help="Display name for image 2.")
    parser.add_argument("--image2", default=str(DEFAULT_IMAGE_2), help="Path to face image 2.")

    parser.add_argument("--subject3-id", default="qq_face_003", help="Subject ID for image 3.")
    parser.add_argument("--subject3-name", default="qq-face-3", help="Display name for image 3.")
    parser.add_argument("--image3", default=str(DEFAULT_IMAGE_3), help="Path to face image 3.")

    parser.set_defaults(update_existing=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    log_path = Path(args.log_file).resolve() if args.log_file else default_log_path("register_three_faces", args.log_dir, base_dir=PROJECT_ROOT)
    args.packet_logger = InterfacePacketLogger(log_path)
    subjects = build_subject_specs(args)

    print(f"[INFO] base_url: {args.base_url}")
    print(f"[INFO] packet log: {args.packet_logger.log_path}")
    started_at = time.time()
    try:
        for subject in subjects:
            print(f"[INFO] registering {subject.subject_id} name={subject.name} image={subject.image_path}")
            create_or_update_subject(subject, args)
            verify_subject(subject, args)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        print(f"[INFO] elapsed: {time.time() - started_at:.2f}s")

    print("[PASS] registered all three face images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

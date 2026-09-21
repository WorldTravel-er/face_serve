from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

from interface_test_logger import InterfacePacketLogger, default_log_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBJECT1_IMAGE = PROJECT_ROOT / "data/gallery/person_001/face_01.jpg"
DEFAULT_SUBJECT2_IMAGE = PROJECT_ROOT / "data/gallery/person_002/face_01.jpg"
DEFAULT_SUBJECT1_MATCH_IMAGE = PROJECT_ROOT / "data/gallery/person_001/face_02.jpg"


@dataclass(frozen=True)
class ApiResult:
    status: int
    body: dict[str, Any]


def encode_image_base64(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"image file not found: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
    logger: InterfacePacketLogger | None = None,
) -> ApiResult:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    headers = {"Content-Type": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers=headers, method=method)

    if logger is not None:
        logger.log_http_request(method=method, url=url, headers=headers, body=payload)

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            body = json.loads(raw) if raw else {}
            if logger is not None:
                logger.log_http_response(
                    status=response.status,
                    reason=getattr(response, "reason", ""),
                    headers={"Content-Type": response.headers.get("Content-Type", "")},
                    body=body,
                )
            return ApiResult(status=response.status, body=body)
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"code": -1, "msg": raw, "data": {}}
        if logger is not None:
            logger.log_http_response(
                status=exc.code,
                reason=getattr(exc, "reason", ""),
                headers={"Content-Type": exc.headers.get("Content-Type", "")},
                body=body,
            )
        return ApiResult(status=exc.code, body=body)
    except URLError as exc:
        raise RuntimeError(f"cannot connect to API service at {base_url}: {exc.reason}") from exc


def assert_envelope(step: str, body: dict[str, Any]) -> None:
    if not isinstance(body, dict):
        raise AssertionError(f"{step} response body must be an object, got: {body!r}")
    for field in ("code", "msg", "data"):
        if field not in body:
            raise AssertionError(f"{step} response missing {field!r}: {body}")


def assert_success(step: str, result: ApiResult) -> None:
    assert_envelope(step, result.body)
    if result.body.get("code") != 0:
        raise AssertionError(f"{step} expected code=0, got HTTP {result.status}, body={result.body}")


def assert_business_error(step: str, result: ApiResult) -> None:
    assert_envelope(step, result.body)
    if result.body.get("code") == 0:
        raise AssertionError(f"{step} expected business error, got HTTP {result.status}, body={result.body}")


def assert_fields(step: str, data: dict[str, Any], required_fields: set[str]) -> None:
    missing = sorted(required_fields - set(data))
    if missing:
        raise AssertionError(f"{step} missing fields: {missing}, data={data}")


def print_pass(name: str, result: ApiResult, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[PASS] {name}: HTTP {result.status}, code={result.body.get('code')}, msg={result.body.get('msg')}{suffix}")


def assert_subject_shape(step: str, data: dict[str, Any], expected_subject_id: str | None = None) -> None:
    assert_fields(step, data, {"subject_id", "name", "image_base64", "create_time", "update_time"})
    if expected_subject_id is not None and data.get("subject_id") != expected_subject_id:
        raise AssertionError(f"{step} returned wrong subject_id: {data}")
    if not isinstance(data.get("image_base64"), str) or not data["image_base64"]:
        raise AssertionError(f"{step} image_base64 must be a non-empty string: {data}")


def run_case(args: argparse.Namespace) -> None:
    unique = uuid.uuid4().hex[:8]
    subject1_id = args.subject1_id or f"rest_e2e_{unique}_001"
    subject2_id = args.subject2_id or f"rest_e2e_{unique}_002"
    subject1_image = Path(args.subject1_image).resolve()
    subject2_image = Path(args.subject2_image).resolve()
    subject1_match_image = Path(args.subject1_match_image).resolve() if args.subject1_match_image else subject1_image

    subject1_b64 = encode_image_base64(subject1_image)
    subject2_b64 = encode_image_base64(subject2_image)
    subject1_match_b64 = encode_image_base64(subject1_match_image)

    def api(method: str, path: str, payload: dict[str, Any] | None = None) -> ApiResult:
        return request_json(args.base_url, method, path, payload, timeout=args.timeout, logger=args.packet_logger)

    created_subjects: list[str] = []
    try:
        health = api("GET", "/healthz")
        assert_success("GET /healthz", health)
        if health.body.get("data", {}).get("status") != "ok":
            raise AssertionError(f"GET /healthz returned unexpected data: {health.body}")
        print_pass("GET /healthz", health)

        ready = api("GET", "/readyz")
        assert_success("GET /readyz", ready)
        ready_data = ready.body.get("data", {})
        if not isinstance(ready_data.get("ready"), bool):
            raise AssertionError(f"GET /readyz data.ready must be bool: {ready.body}")
        if not args.allow_not_ready and ready_data.get("ready") is not True:
            raise AssertionError(f"GET /readyz reports service not ready: {ready.body}")
        runtime = ready_data.get("runtime", {})
        print_pass("GET /readyz", ready, detail=f"ready={ready_data.get('ready')} runtime={json.dumps(runtime, ensure_ascii=False)}")

        create1 = api(
            "POST",
            "/api/recognition/subjects",
            {"subject_id": subject1_id, "name": args.subject1_name, "image_base64": subject1_b64},
        )
        assert_success("POST /api/recognition/subjects subject1", create1)
        if create1.status != 201:
            raise AssertionError(f"create subject1 expected HTTP 201, got HTTP {create1.status}, body={create1.body}")
        if create1.body.get("data", {}).get("subject_id") != subject1_id:
            raise AssertionError(f"create subject1 returned wrong subject_id: {create1.body}")
        created_subjects.append(subject1_id)
        print_pass("POST /api/recognition/subjects subject1", create1)

        duplicate = api(
            "POST",
            "/api/recognition/subjects",
            {"subject_id": subject1_id, "name": args.subject1_name, "image_base64": subject1_b64},
        )
        assert_business_error("duplicate POST /api/recognition/subjects", duplicate)
        print_pass("duplicate subject business error", duplicate)

        create2 = api(
            "POST",
            "/api/recognition/subjects",
            {"subject_id": subject2_id, "name": args.subject2_name, "image_base64": subject2_b64},
        )
        assert_success("POST /api/recognition/subjects subject2", create2)
        if create2.status != 201:
            raise AssertionError(f"create subject2 expected HTTP 201, got HTTP {create2.status}, body={create2.body}")
        created_subjects.append(subject2_id)
        print_pass("POST /api/recognition/subjects subject2", create2)

        get1 = api("GET", f"/api/recognition/subjects/{quote(subject1_id, safe='')}")
        assert_success("GET /api/recognition/subjects/{subject_id}", get1)
        assert_subject_shape("get subject1", get1.body.get("data", {}), expected_subject_id=subject1_id)
        print_pass("GET /api/recognition/subjects/{subject_id}", get1)

        update1 = api(
            "PUT",
            f"/api/recognition/subjects/{quote(subject1_id, safe='')}",
            {"name": args.subject1_updated_name},
        )
        assert_success("PUT /api/recognition/subjects/{subject_id}", update1)
        if update1.body.get("data", {}).get("subject_id") != subject1_id:
            raise AssertionError(f"update subject1 returned wrong subject_id: {update1.body}")
        print_pass("PUT /api/recognition/subjects/{subject_id}", update1)

        get_updated = api("GET", f"/api/recognition/subjects/{quote(subject1_id, safe='')}")
        assert_success("GET updated subject", get_updated)
        updated_data = get_updated.body.get("data", {})
        assert_subject_shape("get updated subject1", updated_data, expected_subject_id=subject1_id)
        if updated_data.get("name") != args.subject1_updated_name:
            raise AssertionError(f"subject1 name was not updated: {get_updated.body}")
        print_pass("GET updated subject", get_updated)

        match = api(
            "POST",
            "/api/recognition/match",
            {"image_base64": subject1_match_b64, "top_k": 2, "threshold": args.threshold},
        )
        assert_success("POST /api/recognition/match", match)
        results = match.body.get("data", {}).get("result")
        if not isinstance(results, list):
            raise AssertionError(f"match data.result must be a list: {match.body}")
        if not results:
            raise AssertionError(f"match returned an empty result: {match.body}")
        first = results[0]
        assert_fields("match result[0]", first, {"subject_id", "name", "similarity"})
        if args.expect_top_subject and first.get("subject_id") != subject1_id:
            raise AssertionError(f"top match is not subject1: {match.body}")
        print_pass("POST /api/recognition/match", match, detail=f"top={first.get('subject_id')} similarity={first.get('similarity')}")

        invalid_b64 = api("POST", "/api/recognition/match", {"image_base64": "not-a-valid-base64"})
        assert_business_error("invalid base64 POST /api/recognition/match", invalid_b64)
        print_pass("invalid base64 business error", invalid_b64)

        missing = api("GET", "/api/recognition/subjects/not_existing_rest_e2e_subject")
        assert_business_error("missing subject GET /api/recognition/subjects/{subject_id}", missing)
        print_pass("missing subject business error", missing)

    finally:
        if not args.keep_data:
            for subject_id in reversed(created_subjects):
                delete = api("DELETE", f"/api/recognition/subjects/{quote(subject_id, safe='')}")
                if delete.body.get("code") == 0:
                    if delete.body.get("data", {}).get("subject_id") != subject_id:
                        print(f"[WARN] cleanup returned unexpected subject_id for {subject_id}: {delete.body}")
                    print_pass(f"DELETE cleanup {subject_id}", delete)
                else:
                    print(f"[WARN] cleanup {subject_id} failed: HTTP {delete.status}, body={delete.body}")

    print("[PASS] REST API interface test completed")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test all REST interfaces in the face recognition service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="HTTP API base URL.")
    parser.add_argument("--subject1-id", default=None, help="Optional fixed subject_id for the first test subject.")
    parser.add_argument("--subject1-name", default="rest-test-subject-1", help="Display name for subject1.")
    parser.add_argument("--subject1-updated-name", default="rest-test-subject-1-updated", help="Updated name for subject1.")
    parser.add_argument("--subject1-image", default=str(DEFAULT_SUBJECT1_IMAGE), help="Image used to enroll subject1.")
    parser.add_argument("--subject1-match-image", default=str(DEFAULT_SUBJECT1_MATCH_IMAGE), help="Image used to match subject1.")
    parser.add_argument("--subject2-id", default=None, help="Optional fixed subject_id for the second test subject.")
    parser.add_argument("--subject2-name", default="rest-test-subject-2", help="Display name for subject2.")
    parser.add_argument("--subject2-image", default=str(DEFAULT_SUBJECT2_IMAGE), help="Image used to enroll subject2.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Match threshold used by /api/recognition/match.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--keep-data", action="store_true", help="Keep created test subjects after the test.")
    parser.add_argument("--allow-not-ready", action="store_true", help="Do not fail when /readyz reports data.ready=false.")
    parser.add_argument("--log-dir", default="logs/interface_tests", help="Directory for packet log files.")
    parser.add_argument("--log-file", default=None, help="Optional exact packet log file path.")
    parser.add_argument(
        "--no-expect-top-subject",
        action="store_false",
        dest="expect_top_subject",
        help="Do not require subject1 to be the first match result.",
    )
    parser.set_defaults(expect_top_subject=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    log_path = Path(args.log_file).resolve() if args.log_file else default_log_path("rest_api", args.log_dir, base_dir=PROJECT_ROOT)
    args.packet_logger = InterfacePacketLogger(log_path)
    print(f"[INFO] packet log: {args.packet_logger.log_path}")
    started_at = time.time()
    try:
        run_case(args)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        print(f"[INFO] elapsed: {time.time() - started_at:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from interface_test_logger import InterfacePacketLogger, default_log_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "cyy.mp4"
VIDEO_TERMINAL_STATUSES = {"finished", "failed", "stopped"}
VIDEO_ALLOWED_STATUSES = {"running", *VIDEO_TERMINAL_STATUSES}


@dataclass(frozen=True)
class ApiResult:
    status: int
    body: dict[str, Any]


def open_url(request: Request, timeout: float) -> Any:
    hostname = urlparse(request.full_url).hostname
    is_loopback = hostname == "localhost"
    if hostname:
        try:
            is_loopback = is_loopback or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            pass
    if is_loopback:
        return build_opener(ProxyHandler({})).open(request, timeout=timeout)
    return urlopen(request, timeout=timeout)


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
        with open_url(request, timeout=timeout) as response:
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
    if not 200 <= result.status < 300:
        raise AssertionError(f"{step} expected HTTP 2xx, got HTTP {result.status}, body={result.body}")
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


def print_pass(name: str, result: ApiResult | None = None, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    if result is None:
        print(f"[PASS] {name}{suffix}")
        return
    print(f"[PASS] {name}: HTTP {result.status}, code={result.body.get('code')}, msg={result.body.get('msg')}{suffix}")


def validate_video_source_url(source_url: str) -> None:
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--video-source-url must be an absolute http or https URL")


class QuietVideoRequestHandler(SimpleHTTPRequestHandler):
    _range: tuple[int, int] | None = None

    def log_message(self, format: str, *args: Any) -> None:
        return

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self) -> Any:
        self._range = None
        range_header = self.headers.get("Range")
        if range_header is None:
            return super().send_head()

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        file_size = os.path.getsize(path)
        byte_range = self._parse_range_header(range_header, file_size)
        if byte_range is None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        start, end = byte_range
        handle = open(path, "rb")
        try:
            stat = os.fstat(handle.fileno())
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.end_headers()
            handle.seek(start)
            self._range = (start, end)
            return handle
        except Exception:
            handle.close()
            raise

    def copyfile(self, source: Any, outputfile: Any) -> None:
        if self._range is None:
            super().copyfile(source, outputfile)
            return

        start, end = self._range
        remaining = end - start + 1
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break
            remaining -= len(chunk)

    @staticmethod
    def _parse_range_header(value: str, file_size: int) -> tuple[int, int] | None:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
        if match is None or file_size <= 0:
            return None

        start_text, end_text = match.groups()
        if not start_text and not end_text:
            return None
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                return None
            start = max(file_size - suffix_length, 0)
            return start, file_size - 1

        start = int(start_text)
        if start >= file_size:
            return None
        end = int(end_text) if end_text else file_size - 1
        if end < start:
            return None
        return start, min(end, file_size - 1)


@contextmanager
def prepared_video_source(
    source_url: str | None,
    *,
    bind_host: str,
    advertised_host: str,
) -> Iterator[str]:
    if source_url is not None:
        validate_video_source_url(source_url)
        yield source_url
        return

    video_path = DEFAULT_VIDEO_PATH.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"default video file does not exist: {video_path}")

    handler = partial(QuietVideoRequestHandler, directory=str(video_path.parent))
    server = ThreadingHTTPServer((bind_host, 0), handler)
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="video-analysis-test-http-server",
        daemon=True,
    )
    server_thread.start()
    local_url = f"http://{advertised_host}:{server.server_port}/{quote(video_path.name)}"
    print(f"[INFO] serving default video: {video_path} -> {local_url}")
    try:
        yield local_url
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)


def assert_video_task_shape(
    step: str,
    data: dict[str, Any],
    *,
    require_track_ids: bool = False,
) -> None:
    assert_fields(step, data, {"task_id", "task_name", "source_url", "task_status", "face_records", "error_info"})
    if data.get("task_status") not in VIDEO_ALLOWED_STATUSES:
        raise AssertionError(f"{step} returned invalid task_status: {data}")
    if not isinstance(data.get("face_records"), list):
        raise AssertionError(f"{step} face_records must be a list: {data}")
    for index, record in enumerate(data["face_records"]):
        if not isinstance(record, dict):
            raise AssertionError(f"{step} face_records[{index}] must be an object: {record!r}")
        assert_fields(
            f"{step} face_records[{index}]",
            record,
            {"subject_id", "name", "first_appear_ts", "last_disappear_ts"},
        )
        # track_id = record.get("track_id")
        # if track_id is not None and (not isinstance(track_id, int) or isinstance(track_id, bool)):
        #     raise AssertionError(f"{step} face_records[{index}].track_id must be an integer or null: {record}")
        # if require_track_ids and track_id is None:
        #     raise AssertionError(f"{step} new finished record must have an integer track_id: {record}")


def wait_video_analysis_task(args: argparse.Namespace, task_name: str) -> ApiResult:
    path = f"/api/recognition/video/analyze/{quote(task_name, safe='')}"

    while True:
        result = request_json(args.base_url, "GET", path, timeout=args.timeout, logger=args.packet_logger)
        assert_success("poll GET /api/recognition/video/analyze/{task_name}", result)
        data = result.body.get("data", {})
        assert_video_task_shape("poll video analysis task", data, require_track_ids=False)
        status = data.get("task_status")
        print(f"[INFO] polling video task {task_name}: status={status}")
        if status in VIDEO_TERMINAL_STATUSES:
            return result
        time.sleep(max(float(args.poll_interval), 0.1))


def run_case(args: argparse.Namespace) -> None:
    validate_video_source_url(args.video_source_url)
    task_name = args.task_name or f"video_e2e_{uuid.uuid4().hex[:12]}"

    def api(method: str, path: str, payload: dict[str, Any] | None = None) -> ApiResult:
        return request_json(args.base_url, method, path, payload, timeout=args.timeout, logger=args.packet_logger)

    health = api("GET", "/healthz")
    assert_success("GET /healthz", health)
    print_pass("GET /healthz", health)

    invalid = api(
        "POST",
        "/api/recognition/video/analyze",
        {"task_name": f"{task_name}_invalid", "source_url": "not-a-valid-url"},
    )
    assert_business_error("invalid source_url POST /api/recognition/video/analyze", invalid)
    print_pass("invalid source_url business error", invalid)

    create = api(
        "POST",
        "/api/recognition/video/analyze",
        {"task_name": task_name, "source_url": args.video_source_url},
    )
    assert_success("POST /api/recognition/video/analyze", create)
    if create.status != 201:
        raise AssertionError(f"create video analysis task expected HTTP 201, got HTTP {create.status}, body={create.body}")
    created_data = create.body.get("data", {})
    assert_fields("create video task data", created_data, {"task_id", "task_name", "create_time", "source_url", "task_status"})
    if created_data.get("task_name") != task_name:
        raise AssertionError(f"created task has wrong task_name: {create.body}")
    if created_data.get("source_url") != args.video_source_url:
        raise AssertionError(f"created task has wrong source_url: {create.body}")
    if created_data.get("task_status") != "running":
        raise AssertionError(f"created task should start as running: {create.body}")
    print_pass("POST /api/recognition/video/analyze", create, detail=f"task_name={task_name}")

    result = wait_video_analysis_task(args, task_name)
    assert_success("GET /api/recognition/video/analyze/{task_name}", result)
    data = result.body.get("data", {})
    final_status = data.get("task_status")
    assert_video_task_shape("query video analysis task", data, require_track_ids=final_status == "finished")
    if data.get("task_name") != task_name:
        raise AssertionError(f"queried task has wrong task_name: {result.body}")
    if data.get("source_url") != args.video_source_url:
        raise AssertionError(f"queried task has wrong source_url: {result.body}")

    if final_status != "finished" and not args.allow_failed:
        raise AssertionError(f"video task did not finish successfully: {result.body}")
    if final_status != "finished":
        print(f"[WARN] video task ended with status={final_status}; error_info={data.get('error_info')}")
    print_pass("GET /api/recognition/video/analyze/{task_name}", result, detail=f"status={final_status} records={len(data.get('face_records', []))}")

    missing = api("GET", "/api/recognition/video/analyze/not_existing_video_e2e_task")
    assert_business_error("missing video task GET /api/recognition/video/analyze/{task_name}", missing)
    if missing.status != 404:
        raise AssertionError(f"missing video task expected HTTP 404, got HTTP {missing.status}, body={missing.body}")
    print_pass("missing video task business error", missing)

    print("[PASS] Video analysis interface test completed")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test long-video analysis REST interfaces.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="HTTP API base URL.")
    parser.add_argument(
        "--video-source-url",
        default=None,
        help=f"Optional HTTP/HTTPS video URL. By default, serve and analyze {DEFAULT_VIDEO_PATH.relative_to(PROJECT_ROOT)}.",
    )
    parser.add_argument(
        "--video-server-bind",
        default="127.0.0.1",
        help="Address used by the temporary local video server. Default: 127.0.0.1.",
    )
    parser.add_argument(
        "--video-server-host",
        default="host.docker.internal",
        help="Video-server hostname sent to the API. Default targets an API running in Docker; use 127.0.0.1 for a host API.",
    )
    parser.add_argument("--task-name", default=None, help="Optional fixed video analysis task_name.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--log-dir", default="logs/interface_tests", help="Directory for packet log files.")
    parser.add_argument("--log-file", default=None, help="Optional exact packet log file path.")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between task result polls.")
    parser.add_argument("--allow-failed", action="store_true", help="Do not fail if the task ends as failed or stopped.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    log_path = Path(args.log_file).resolve() if args.log_file else default_log_path("video_analysis", args.log_dir, base_dir=PROJECT_ROOT)
    args.packet_logger = InterfacePacketLogger(log_path)
    print(f"[INFO] packet log: {args.packet_logger.log_path}")
    started_at = time.time()
    try:
        with prepared_video_source(
            args.video_source_url,
            bind_host=args.video_server_bind,
            advertised_host=args.video_server_host,
        ) as source_url:
            args.video_source_url = source_url
            run_case(args)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        print(f"[INFO] elapsed: {time.time() - started_at:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

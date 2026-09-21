from __future__ import annotations

import sys
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from test_video_analysis_interfaces import QuietVideoRequestHandler  # noqa: E402


def test_video_handler_serves_byte_ranges(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"0123456789")
    handler = partial(QuietVideoRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    opener = build_opener(ProxyHandler({}))
    url = f"http://127.0.0.1:{server.server_port}/{video_path.name}"

    try:
        response = opener.open(Request(url, headers={"Range": "bytes=2-5"}), timeout=5)
        assert response.status == 206
        assert response.headers["Accept-Ranges"] == "bytes"
        assert response.headers["Content-Range"] == "bytes 2-5/10"
        assert response.headers["Content-Length"] == "4"
        assert response.read() == b"2345"

        suffix_response = opener.open(Request(url, headers={"Range": "bytes=-3"}), timeout=5)
        assert suffix_response.status == 206
        assert suffix_response.headers["Content-Range"] == "bytes 7-9/10"
        assert suffix_response.read() == b"789"

        try:
            opener.open(Request(url, headers={"Range": "bytes=20-30"}), timeout=5)
        except HTTPError as exc:
            assert exc.code == 416
            assert exc.headers["Content-Range"] == "bytes */10"
        else:
            raise AssertionError("unsatisfiable byte range should return HTTP 416")
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

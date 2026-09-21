from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class InterfacePacketLogger:
    """Write transmitted packets in the document-style structure used by the API spec."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def log_http_request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: Any = None,
    ) -> None:
        parsed = urlparse(url)
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        lines = [
            "========== REQUEST ==========",
            f"{method.upper()} {target} HTTP/1.1",
            f"Host: {parsed.netloc}",
        ]
        lines.extend(self._header_lines(headers))
        lines.append("")
        if body is not None:
            lines.append(self._json(body))
        self._write_block(lines)

    def log_http_response(
        self,
        *,
        status: int,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        body: Any = None,
    ) -> None:
        status_line = f"HTTP/1.1 {int(status)}"
        if reason:
            status_line = f"{status_line} {reason}"
        lines = ["========== RESPONSE ==========", status_line]
        lines.extend(self._header_lines(headers))
        lines.append("")
        if body is not None:
            lines.append(self._json(body))
        self._write_block(lines)

    def log_websocket_send(self, url: str, body: Any) -> None:
        self._write_block(
            [
                "========== WEBSOCKET SEND ==========",
                url,
                "",
                self._json(body),
            ]
        )

    def log_websocket_receive(self, body: Any) -> None:
        self._write_block(
            [
                "========== WEBSOCKET RECEIVE ==========",
                self._json(body),
            ]
        )

    def log_websocket_close(self, body: Any) -> None:
        self._write_block(
            [
                "========== WEBSOCKET CLOSE ==========",
                self._json(body),
            ]
        )

    def _write_block(self, lines: list[str]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines).rstrip())
            handle.write("\n\n")

    def _header_lines(self, headers: dict[str, str] | None) -> list[str]:
        if not headers:
            return []
        return [f"{key}: {value}" for key, value in headers.items()]

    def _json(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2)


def default_log_path(script_name: str, log_dir: str | Path, *, base_dir: str | Path | None = None) -> Path:
    root = Path(log_dir)
    if not root.is_absolute() and base_dir is not None:
        root = Path(base_dir) / root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return root / f"{script_name}_{timestamp}_{suffix}.log"

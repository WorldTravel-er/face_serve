from __future__ import annotations

import inspect
from typing import Any


async def run_live_runner(
    runner: object,
    state: object,
    send_json: object,
    request_close: object,
    *,
    connection_id: str | None = None,
    websocket_accepted_at: float | None = None,
) -> None:
    run = getattr(runner, "run")
    try:
        signature = inspect.signature(run)
    except (TypeError, ValueError):
        result = run(state, send_json, request_close=request_close)
    else:
        kwargs: dict[str, object] = {}
        if "request_close" in signature.parameters:
            kwargs["request_close"] = request_close
        if connection_id is not None and "connection_id" in signature.parameters:
            kwargs["connection_id"] = connection_id
        if websocket_accepted_at is not None and "websocket_accepted_at" in signature.parameters:
            kwargs["websocket_accepted_at"] = websocket_accepted_at
        result = run(state, send_json, **kwargs)
    if inspect.isawaitable(result):
        await result


def log_websocket_connected(runner: object, *, connection_id: str, elapsed_ms: float) -> None:
    diagnostics = getattr(runner, "diagnostics", None)
    logger = getattr(diagnostics, "log_websocket_connected", None)
    if callable(logger):
        logger(connection_id=connection_id, elapsed_ms=elapsed_ms)
from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from face_api.live.gateway.protocol import (
    CLIENT_HEARTBEAT,
    CONTROL_RECOGNITION,
    SET_MATCH_THRESHOLD,
    LiveProtocolError,
    build_control_ack,
    build_server_heartbeat_ack,
    build_stream_error,
    build_threshold_ack,
    parse_client_message,
    build_close_ack,
)
from face_api.live.gateway.state import LiveRecognitionState
from face_api.live.pipeline.runner import LiveRecognitionRunner
from face_api.live.gateway.runner_dispatch import log_websocket_connected, run_live_runner


router = APIRouter(tags=["live-recognition"])


@router.websocket("/ws/recognition/live")
async def live_recognition_websocket(websocket: WebSocket) -> None:
    accept_started_at = time.perf_counter()
    await websocket.accept()
    websocket_accepted_at = time.perf_counter()
    websocket_accept_elapsed_ms = (websocket_accepted_at - accept_started_at) * 1000.0
    connection_id = uuid.uuid4().hex[:12]

    config = websocket.app.state.config
    state = LiveRecognitionState(
        default_threshold=getattr(config, "live_match_threshold", 0.3),
        heartbeat_timeout_seconds=getattr(config, "live_heartbeat_timeout_seconds", 10.0),
    )

    runner = getattr(websocket.app.state, "live_runner", None)
    if runner is None:
        runner = LiveRecognitionRunner(
            config=config,
            store=websocket.app.state.store,
            recognition_service=websocket.app.state.recognition_service,
        )

    log_websocket_connected(runner, connection_id=connection_id, elapsed_ms=websocket_accept_elapsed_ms)

    async def request_close(*, code: int, reason: str) -> None:
        await _close_websocket(websocket, state, code=code, reason=reason)

    runner_task = asyncio.create_task(
        run_live_runner(
            runner,
            state,
            websocket.send_json,
            request_close,
            connection_id=connection_id,
            websocket_accepted_at=websocket_accepted_at,
        )
    )
    heartbeat_task = asyncio.create_task(_watch_heartbeat(websocket, state))

    try:
        while not state.closed:
            payload = await websocket.receive_json()
            await _handle_client_message(websocket, state, payload)
    except WebSocketDisconnect:
        state.close()
    finally:
        state.close()
        runner_task.cancel()
        heartbeat_task.cancel()
        await _cancel_quietly(runner_task)
        await _cancel_quietly(heartbeat_task)



async def _close_websocket(websocket: WebSocket, state: LiveRecognitionState, *, code: int, reason: str) -> bool:
    if not state.request_close(code=code, reason=reason):
        return False
    try:
        print("reason: ", reason)
        # await websocket.send_json(build_close_ack(reason))
        await websocket.close(code=code, reason=reason)
    except RuntimeError:
        return False
    return True


async def _handle_client_message(websocket: WebSocket, state: LiveRecognitionState, payload: object) -> None:
    try:
        message = parse_client_message(payload)
        if message.msg_type == CLIENT_HEARTBEAT:
            state.mark_heartbeat()
            await websocket.send_json(build_server_heartbeat_ack())
            return
        if message.msg_type == SET_MATCH_THRESHOLD:
            state.update_threshold(message.data.get("threshold"))
            await websocket.send_json(build_threshold_ack(state.threshold))
            return
        if message.msg_type == CONTROL_RECOGNITION:
            status = message.data.get("status")
            state.update_status(status)
            await websocket.send_json(build_control_ack(state.status))
            return
        await websocket.send_json(build_stream_error(f"Unsupported msg_type: {message.msg_type}"))
    except (LiveProtocolError, TypeError, ValueError) as exc:
        await websocket.send_json(build_stream_error(str(exc)))


async def _watch_heartbeat(websocket: WebSocket, state: LiveRecognitionState) -> None:
    try:
        while not state.closed:
            await asyncio.sleep(1.0)
            if state.is_heartbeat_timed_out():
                await _close_websocket(websocket, state, code=1000, reason="heartbeat timeout")
                return
    except RuntimeError:
        state.close()


async def _cancel_quietly(task: asyncio.Task) -> None:
    try:
        await task
    except asyncio.CancelledError:
        return

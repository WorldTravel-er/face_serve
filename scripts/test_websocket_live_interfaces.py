from __future__ import annotations
"""
连接实时识别 WebSocket → 接收识别结果 → 暂停/恢复识别 → 修改匹配阈值 → 测试非法消息 → 测试心跳保活 → 测试心跳超时 → 测试断线重连。
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from interface_test_logger import InterfacePacketLogger, default_log_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASYNC_MESSAGE_TYPES = {"face_match_result", "stream_error"}

DEFAULT_OBSERVE_SECONDS = 10.0 # 每次观察实时识别结果默认 10 秒
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0 # 心跳间隔默认 10 秒
DEFAULT_HEARTBEAT_COUNT = 4 # 正常心跳默认发 4 次
DEFAULT_HEARTBEAT_TIMEOUT_WAIT_SECONDS = 40.0 # 心跳超时测试最多等 30 秒
DEFAULT_HEARTBEAT_TIMEOUT_REASON = "heartbeat timeout" # 预期服务器关闭连接的原因是 "heartbeat timeout"
DEFAULT_RECONNECT_INTERVAL_SECONDS = 3.0 # 重连间隔 3 秒
DEFAULT_RECONNECT_MAX_ATTEMPTS = 10 # 最多重连 10 次

LIVE_CONTROL_FLOW_STEPS = (
    "connect",
    "observe_before_first_pause",
    "pause",
    "resume",
    "observe_after_resume",
    "pause_again",
    "set_match_threshold",
    "invalid_msg_type",
    "heartbeat_keepalive",
    "heartbeat_timeout",
    "reconnect",
)


def load_websockets_module():
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("websockets is required. Install it with `uv add websockets` or run inside the project venv.") from exc
    return websockets
def load_connection_closed_exception():
    try:
        from websockets.exceptions import ConnectionClosed
    except ImportError as exc:
        raise RuntimeError("websockets.exceptions.ConnectionClosed is required by the heartbeat-timeout test.") from exc
    return ConnectionClosed

# 检查 WebSocket 消息格式
# {
#   "msg_type": "control_ack",
#   "data": {
#     ...
#   }
# }
def assert_message_object(step: str, message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise AssertionError(f"{step} expected a JSON object, got: {message!r}")
    if not isinstance(message.get("msg_type"), str) or not message["msg_type"]:
        raise AssertionError(f"{step} response missing non-empty msg_type: {message}")
    if "data" not in message or not isinstance(message["data"], dict):
        raise AssertionError(f"{step} response missing object data: {message}")
    return message


# 检查字段是否缺失
def assert_fields(step: str, data: dict[str, Any], required_fields: set[str]) -> None:
    missing = sorted(required_fields - set(data))
    if missing:
        raise AssertionError(f"{step} missing fields: {missing}, data={data}")

# 记录日志，发送 WebSocket JSON
async def send_json(websocket: Any, payload: dict[str, Any], *, url: str, logger: InterfacePacketLogger) -> None:
    logger.log_websocket_send(url, payload)
    await websocket.send(json.dumps(payload, ensure_ascii=False))

# 接收并解析 JSON
async def receive_json(websocket: Any, timeout: float, *, logger: InterfacePacketLogger) -> dict[str, Any]:
    raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"received non-JSON WebSocket message: {raw!r}") from exc
    message = assert_message_object("receive websocket message", message)
    logger.log_websocket_receive(message)
    return message

# 等待指定类型的消息
async def wait_for_message(
    websocket: Any,
    expected_types: set[str],
    *,
    timeout: float,
    step: str,
    logger: InterfacePacketLogger,
    error_contains: str | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    ignored: list[str] = []

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"{step} timed out waiting for {sorted(expected_types)}; ignored={ignored}")

        message = await receive_json(websocket, remaining, logger=logger)
        msg_type = message.get("msg_type")

        if msg_type in expected_types:
            if error_contains is not None:
                error_msg = str(message.get("data", {}).get("error_msg", ""))
                if error_contains not in error_msg:
                    ignored.append(f"{msg_type}:{error_msg}")
                    print(f"[INFO] ignored non-matching stream_error while waiting for {step}: {error_msg}")
                    continue
            return message

        if msg_type in ASYNC_MESSAGE_TYPES:
            ignored.append(str(msg_type))
            detail = message.get("data", {}).get("error_msg") if msg_type == "stream_error" else ""
            suffix = f" ({detail})" if detail else ""
            print(f"[INFO] ignored async {msg_type} while waiting for {step}{suffix}")
            continue

        raise AssertionError(f"{step} received unexpected message: {message}")

#
def print_pass(name: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[PASS] {name}{suffix}")

# 检查暂停/恢复操作的响应
def assert_control_ack(step: str, message: dict[str, Any], expected_status: str) -> None:
    if message.get("msg_type") != "control_ack":
        raise AssertionError(f"{step} expected control_ack, got: {message}")
    data = message["data"]
    if data.get("code") != 0 or data.get("status") != expected_status:
        raise AssertionError(f"{step} returned unexpected data: {message}")

# 心跳协议
def assert_heartbeat_ack(step: str, message: dict[str, Any]) -> None:
    if message.get("msg_type") != "server_heartbeat_ack":
        raise AssertionError(f"{step} expected server_heartbeat_ack, got: {message}")
    assert_fields("server_heartbeat_ack.data", message["data"], {"server_time"})

# 人脸识别结果协议的校验
def assert_face_match_result(message: dict[str, Any]) -> None:
    if message.get("msg_type") != "face_match_result":
        raise AssertionError(f"expected face_match_result, got: {message}")
    live_data = message["data"]
    assert_fields("face_match_result.data", live_data, {"frame_time", "video_stream_id", "match_info"})
    match_info = live_data["match_info"]
    if not isinstance(match_info, dict):
        raise AssertionError(f"face_match_result.data.match_info must be an object: {message}")
    assert_fields("face_match_result.data.match_info", match_info, {"subject_id", "name", "similarity", "threshold"})

# 发送暂停 / 恢复，等待ack, 验证
async def send_control_and_expect_ack(websocket: Any, args: argparse.Namespace, status: str) -> None:
    step = f"control_recognition {status}"
    await send_json(
        websocket,
        {"msg_type": "control_recognition", "data": {"status": status}},
        url=args.ws_url,
        logger=args.packet_logger,
    )
    ack = await wait_for_message(
        websocket,
        {"control_ack"},
        timeout=args.timeout,
        step=step,
        logger=args.packet_logger,
    )
    assert_control_ack(step, ack, status)
    print_pass(f"{step} -> control_ack")

# 发送心跳，等待ack, 验证
async def send_client_heartbeat(websocket: Any, args: argparse.Namespace, *, label: str) -> dict[str, Any]:
    await send_json(websocket, {"msg_type": "client_heartbeat", "data": {}}, url=args.ws_url, logger=args.packet_logger)
    heartbeat_ack = await wait_for_message(
        websocket,
        {"server_heartbeat_ack"},
        timeout=args.timeout,
        step=label,
        logger=args.packet_logger,
    )
    assert_heartbeat_ack(label, heartbeat_ack)
    print_pass(f"{label} -> server_heartbeat_ack", detail=f"server_time={heartbeat_ack['data'].get('server_time')}")
    return heartbeat_ack

# 持续接受10s的websocket信息，观察人脸识别服务信息推送是否正常
async def observe_recognition_window(websocket: Any, args: argparse.Namespace, *, label: str) -> dict[str, int]:
    deadline = time.monotonic() + args.observe_seconds # 观察截止时间
    counts: dict[str, int] = {} # 消息计数字典

    print(f"[INFO] observing {label} for {args.observe_seconds:.1f}s")
    while True:
        remaining = deadline - time.monotonic() # 每次循环重新计算剩余时间
        if remaining <= 0:
            break

        try: # 接收 WebSocket 消息
            message = await receive_json(websocket, min(args.timeout, remaining), logger=args.packet_logger)
        except asyncio.TimeoutError:
            continue
        # 获取消息类型
        msg_type = str(message.get("msg_type"))
        counts[msg_type] = counts.get(msg_type, 0) + 1
        # 处理 face_match_result
        if msg_type == "face_match_result":
            assert_face_match_result(message)
            match_info = message["data"].get("match_info", {})
            print(
                "[INFO] face_match_result "
                f"subject={match_info.get('subject_id')} similarity={match_info.get('similarity')}"
            )
        elif msg_type == "stream_error":
            print(f"[INFO] stream_error during {label}: {message['data'].get('error_msg', '')}")
        elif msg_type == "server_heartbeat_ack":
            print(f"[INFO] heartbeat ack observed during {label}: {message['data'].get('server_time')}")
        else:
            raise AssertionError(f"{label} received unexpected message: {message}")

    face_count = counts.get("face_match_result", 0)
    stream_error_count = counts.get("stream_error", 0)
    if args.expect_live_result and face_count <= 0:
        raise AssertionError(f"{label} expected at least one face_match_result in {args.observe_seconds:.1f}s")
    print_pass(
        label,
        detail=f"observed face_match_result={face_count}, stream_error={stream_error_count}",
    )
    return counts

# 主测试流程建立 WebSocket->观察 10 秒 ->  pause -> 观察10s -> resume -> 观察 10 秒
#  -> pause -> 设置 threshold -> 发送非法 msg_type -> 测试心跳保活
async def run_primary_live_control_flow(args: argparse.Namespace) -> int:
    websockets = load_websockets_module()
    heartbeat_sent = 0

    async with websockets.connect(
        args.ws_url,
        open_timeout=args.timeout,
        close_timeout=args.timeout,
        ping_interval=None,
    ) as websocket: # 建立 WebSocket
        print_pass("websocket connected", detail=args.ws_url)
        # 观察10s识别情况，服务端能否读取并正确识别视频流并推送结果
        await observe_recognition_window(websocket, args, label="observe_before_first_pause")
        # 发送暂停信号，服务端停止读取视频流，识别和推送的过程
        await send_control_and_expect_ack(websocket, args, "pause")
        # 再观察10s识别情况，看看是否已经暂停
        await observe_recognition_window(websocket, args, label="observe_after_first_pause")
        # 恢复识别
        await send_control_and_expect_ack(websocket, args, "resume")
        # 再观察10s识别情况，看看是否已经恢复识别
        await observe_recognition_window(websocket, args, label="observe_after_resume")
        # 测试发送心跳，并观察服务端ack
        await send_client_heartbeat(websocket, args, label="client_heartbeat after first 10s observe")
        heartbeat_sent += 1
        # 再次暂停，测试其他功能
        await send_control_and_expect_ack(websocket, args, "pause")
        # 测试修改识别阈值
        await send_json(
            websocket,
            {"msg_type": "set_match_threshold", "data": {"threshold": args.threshold}},
            url=args.ws_url,
            logger=args.packet_logger,
        )
        threshold_ack = await wait_for_message(
            websocket,
            {"threshold_ack"},
            timeout=args.timeout,
            step="set_match_threshold",
            logger=args.packet_logger,
        )
        data = threshold_ack["data"]
        if data.get("code") != 0:
            raise AssertionError(f"set_match_threshold returned non-zero code: {threshold_ack}")
        actual_threshold = float(data.get("new_threshold"))
        if abs(actual_threshold - args.threshold) > 1e-9:
            raise AssertionError(f"set_match_threshold returned wrong threshold: {threshold_ack}")
        print_pass("set_match_threshold -> threshold_ack", detail=f"new_threshold={actual_threshold}")

        # 测试错误websocket信息
        await send_json(websocket, {"msg_type": "unknown_message_for_e2e", "data": {}}, url=args.ws_url, logger=args.packet_logger)
        protocol_error = await wait_for_message(
            websocket,
            {"stream_error"},
            timeout=args.timeout,
            step="invalid msg_type",
            logger=args.packet_logger,
            error_contains="Unsupported msg_type",
        )
        print_pass("invalid msg_type -> stream_error", detail=str(protocol_error["data"].get("error_msg", "")))
        # 做 heartbeat keepalive，按指定时间间隔发送count次，heartbeat
        remaining_heartbeats = max(args.heartbeat_count - heartbeat_sent, 0)

        await run_heartbeat_keepalive_on_existing_connection(websocket, args, remaining_heartbeats)
        heartbeat_sent += remaining_heartbeats


    return heartbeat_sent


async def run_heartbeat_keepalive_on_existing_connection(websocket: Any, args: argparse.Namespace, count: int) -> None:
    if count <= 0:
        print_pass("heartbeat_keepalive", detail="already satisfied by primary live-control flow")
        return

    for index in range(1, count + 1):
        await asyncio.sleep(args.heartbeat_interval)
        await send_client_heartbeat(websocket, args, label=f"heartbeat_keepalive #{index}")
    print_pass("heartbeat_keepalive", detail=f"sent remaining={count}, total_default={args.heartbeat_count}")



# 客户端故意不发送 heartbeat，观察服务器是否会因为 heartbeat 超时主动关闭 WebSocket，并且关闭原因是否符合预期。
async def run_heartbeat_timeout_case(args: argparse.Namespace) -> None:
    websockets = load_websockets_module()
    # 加载连接关闭异常类型,用于try_catch捕获
    # WebSocket 被服务器关闭时，通常会抛：websockets.exceptions.ConnectionClosed
    connection_closed_cls = load_connection_closed_exception()
    # 设置超时检测截止时间
    deadline = time.monotonic() + args.heartbeat_timeout_wait # 最多等服务器 30 秒关闭连接

    async with websockets.connect(
        args.ws_url,
        open_timeout=args.timeout,
        close_timeout=args.timeout,
        ping_interval=None,
    ) as websocket:
        print("[INFO] heartbeat_timeout: connected and intentionally sending no client_heartbeat")
        await send_control_and_expect_ack(websocket, args, "pause")
        # 故意不发送 heartbeat。
        while True:
            remaining = deadline - time.monotonic()
            # 等了这么久，服务器还没关闭连接，测试失败
            if remaining <= 0:
                raise AssertionError(
                    f"heartbeat_timeout did not close within {args.heartbeat_timeout_wait:.1f}s "
                    f"with reason={args.heartbeat_timeout_reason!r}"
                )

            try:

                message = await receive_json(websocket, min(args.timeout, remaining), logger=args.packet_logger)
            except asyncio.TimeoutError:
                print("[INFO] heartbeat_timeout: timed out")
                continue
            except connection_closed_cls as exc:
                print("[INFO] heartbeat_timeout: connection closed")
                close_info = {"code": exc.code, "reason": exc.reason}
                args.packet_logger.log_websocket_close(close_info)
                reason = exc.reason or ""
                if args.heartbeat_timeout_reason not in reason:
                    raise AssertionError(f"heartbeat_timeout close reason mismatch: {close_info}")
                print_pass("heartbeat_timeout close", detail=f"code={exc.code}, reason={reason}")
                return

            msg_type = message.get("msg_type")
            if msg_type in ASYNC_MESSAGE_TYPES:
                print(f"[INFO] ignored async {msg_type} while waiting for heartbeat timeout")
                continue
            print(f"[INFO] ignored message while waiting for heartbeat timeout: {message}")

# WebSocket 断线重连策略
# 尝试最多连接 reconnect_max_attempts 次，每次成功连接后发送一个 heartbeat 验证连接有效；
# 如果失败，则等待 reconnect_interval 秒后重试；所有尝试失败则测试失败
async def run_reconnect_case(args: argparse.Namespace) -> None:
    websockets = load_websockets_module()
    last_error: Exception | None = None

    for attempt in range(1, args.reconnect_max_attempts + 1):
        try:
            print(f"[INFO] reconnect attempt {attempt}/{args.reconnect_max_attempts}")
            async with websockets.connect(
                args.ws_url,
                open_timeout=args.timeout,
                close_timeout=args.timeout,
                ping_interval=None,
            ) as websocket:
                await send_control_and_expect_ack(websocket, args, "pause")
                await send_client_heartbeat(websocket, args, label=f"reconnect heartbeat attempt {attempt}")
                print_pass("reconnect strategy", detail=f"connected on attempt={attempt}")
                return
        except Exception as exc:
            last_error = exc
            args.packet_logger.log_websocket_close({"reconnect_attempt": attempt, "error": str(exc)})
            print(f"[WARN] reconnect attempt {attempt} failed: {exc}")
            if attempt < args.reconnect_max_attempts:
                await asyncio.sleep(args.reconnect_interval)

    raise AssertionError(
        f"reconnect strategy failed after {args.reconnect_max_attempts} attempts "
        f"with {args.reconnect_interval:.1f}s interval; last_error={last_error}"
    )


async def run_case(args: argparse.Namespace) -> None:
    # 主测试流程建立 WebSocket->观察 10 秒 ->  pause -> 观察10s -> resume -> 观察 10 秒
    #  -> pause -> 设置 threshold -> 发送非法 msg_type -> 测试心跳保活
    await run_primary_live_control_flow(args)

    await run_heartbeat_timeout_case(args)

    await run_reconnect_case(args)
    print("[PASS] WebSocket live interface test completed")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test WebSocket live-recognition control protocol interfaces.")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8003/ws/recognition/live", help="Live-recognition WebSocket URL.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout in seconds for each WebSocket step.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold sent by set_match_threshold.")
    parser.add_argument("--log-dir", default="logs/interface_tests", help="Directory for packet log files.")
    parser.add_argument("--log-file", default=None, help="Optional exact packet log file path.")
    parser.add_argument(
        "--expect-live-result",
        action="store_true",
        help="Require at least one face_match_result in each live observation window.",
    )
    parser.add_argument(
        "--live-result-timeout",
        type=float,
        default=30.0,
        help="Deprecated compatibility option. Live result waiting is now controlled by --observe-seconds.",
    )
    parser.add_argument("--observe-seconds", type=float, default=DEFAULT_OBSERVE_SECONDS, help="Seconds to observe live pushed messages before pause/resume checks.")
    parser.add_argument("--heartbeat-interval", type=float, default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS, help="Seconds between client_heartbeat packets in keepalive checks.")
    parser.add_argument("--heartbeat-count", type=int, default=DEFAULT_HEARTBEAT_COUNT, help="Total client_heartbeat packets expected in the normal keepalive path.")
    parser.add_argument("--heartbeat-timeout-wait", type=float, default=DEFAULT_HEARTBEAT_TIMEOUT_WAIT_SECONDS, help="Max seconds to wait for server heartbeat-timeout close.")
    parser.add_argument("--heartbeat-timeout-reason", default=DEFAULT_HEARTBEAT_TIMEOUT_REASON, help="Expected server close reason when heartbeat times out.")
    parser.add_argument("--reconnect-interval", type=float, default=DEFAULT_RECONNECT_INTERVAL_SECONDS, help="Seconds between reconnect attempts.")
    parser.add_argument("--reconnect-max-attempts", type=int, default=DEFAULT_RECONNECT_MAX_ATTEMPTS, help="Maximum reconnect attempts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    log_path = Path(args.log_file).resolve() if args.log_file else default_log_path("websocket_live", args.log_dir, base_dir=PROJECT_ROOT)
    args.packet_logger = InterfacePacketLogger(log_path)
    print(f"[INFO] packet log: {args.packet_logger.log_path}")
    print(f"[INFO] live control flow: {' -> '.join(LIVE_CONTROL_FLOW_STEPS)}")
    started_at = time.time()
    try:
        asyncio.run(run_case(args))
    except Exception as exc:
        args.packet_logger.log_websocket_close({"error": str(exc)})
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        print(f"[INFO] elapsed: {time.time() - started_at:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
from __future__ import annotations
"""
    定义消息格式
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# {
#     "msg_type":"face_match_result"
#     "data": {}
# }
# 定义msg_type
FACE_MATCH_RESULT = "face_match_result" # 人脸实时识别结果
SET_MATCH_THRESHOLD = "set_match_threshold" # 修改人脸匹配阈值
THRESHOLD_ACK = "threshold_ack"# 修改人脸匹配阈值应答
CONTROL_RECOGNITION = "control_recognition" # 暂停/恢复实时识别
CONTROL_ACK = "control_ack" # 暂停/恢复实时识别应答
CLIENT_HEARTBEAT = "client_heartbeat" # 客户端心跳
SERVER_HEARTBEAT_ACK = "server_heartbeat_ack" # 人脸识别服务端心跳应答
STREAM_ERROR = "stream_error" # 异常推送消息

# 自定义异常类，表示协议数据不合法。
# 当 WebSocket JSON 不符合协议格式时抛出。
class LiveProtocolError(ValueError):
    """Raised when a WebSocket JSON message does not match the live protocol envelope."""

# 数据类
# {
#     "msg_type":"face_match_result"
#     "data": {}
# }
@dataclass(frozen=True)
class ClientMessage:
    msg_type: str
    data: dict[str, Any]


# 把 datetime 对象转换成协议规定的时间字符串（精确到毫秒）
def format_live_time(value: datetime | None = None) -> str:
    current = value or datetime.now()
    return current.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

# 对客户端发送过来的 JSON 消息进行解析和合法性校验，并转换成 ClientMessage 对象。
def parse_client_message(payload: Any) -> ClientMessage:
    if not isinstance(payload, dict):
        raise LiveProtocolError("WebSocket message must be a JSON object")
    msg_type = payload.get("msg_type")
    if not isinstance(msg_type, str) or not msg_type.strip():
        raise LiveProtocolError("WebSocket message must include msg_type")
    data = payload.get("data", {})
    if not isinstance(data, dict):
        raise LiveProtocolError("WebSocket message data must be a JSON object")
    return ClientMessage(msg_type=msg_type.strip(), data=data)

# 根据识别结果，构造一个符合协议格式的"人脸匹配结果"消息（JSON 字典），用于服务器通过 WebSocket 发送给客户端。
def build_face_match_result(
    *,
    frame_time: str,
    video_stream_id: str,
    subject_id: str,
    name: str,
    similarity: float,
    threshold: float,
) -> dict:
    return {
        "msg_type": FACE_MATCH_RESULT,
        "data": {
            "frame_time": frame_time,
            "video_stream_id": video_stream_id,
            "match_info": {
                "subject_id": subject_id,
                "name": name,
                "similarity": float(similarity),
                "threshold": float(threshold),
            },
        },
    }


def build_threshold_ack(threshold: float) -> dict:
    return {
        "msg_type": THRESHOLD_ACK,
        "data": {
            "code": 0,
            "msg": "匹配阈值更新成功",
            "new_threshold": float(threshold),
        },
    }

def build_close_ack(reason: str) -> dict:
    return {
        "msg_type": reason,
        "data": {},
    }

def build_control_ack(status: str) -> dict:
    if status == "pause":
        msg = "已暂停实时人脸识别"
    elif status == "resume":
        msg = "已恢复实时人脸识别"
    else:
        raise LiveProtocolError("Invalid recognition control status")
    return {
        "msg_type": CONTROL_ACK,
        "data": {"code": 0, "msg": msg, "status": status},
    }


def build_server_heartbeat_ack(server_time: datetime | None = None) -> dict:
    return {
        "msg_type": SERVER_HEARTBEAT_ACK,
        "data": {"server_time": format_live_time(server_time)},
    }


def build_stream_error(error_msg: str) -> dict:
    return {
        "msg_type": STREAM_ERROR,
        "data": {"error_msg": str(error_msg)},
    }


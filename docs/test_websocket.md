## 测试websocket

```bash
# 首先注册人脸
uv run python scripts/register_three_faces.py
uv run python scripts/register_four_faces.py

uv run python scripts/test_websocket_live_interfaces.py # 测试websocket
```

客户端控制台日志

![image-20260807174012628](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260807174012628.png)

服务端控制台日志

![image-20260807174035317](C:\Users\ChenHM\AppData\Roaming\Typora\typora-user-images\image-20260807174035317.png)

心跳机制

```
========== WEBSOCKET SEND ==========
ws://127.0.0.1:8000/ws/recognition/live

{
  "msg_type": "client_heartbeat",
  "data": {}
}

========== WEBSOCKET RECEIVE ==========
{
  "msg_type": "server_heartbeat_ack",
  "data": {
    "server_time": "2026-08-07 08:15:30.138"
  }
}
```



更新匹配阈值

```
========== WEBSOCKET SEND ==========
ws://127.0.0.1:8000/ws/recognition/live

{
  "msg_type": "set_match_threshold",
  "data": {
    "threshold": 0.85
  }
}

========== WEBSOCKET RECEIVE ==========
{
  "msg_type": "threshold_ack",
  "data": {
    "code": 0,
    "msg": "匹配阈值更新成功",
    "new_threshold": 0.85
  }
}
```



人脸识别服务暂停与恢复

```
========== WEBSOCKET SEND ==========
ws://127.0.0.1:8000/ws/recognition/live

{
  "msg_type": "control_recognition",
  "data": {
    "status": "pause"
  }
}

========== WEBSOCKET RECEIVE ==========
{
  "msg_type": "control_ack",
  "data": {
    "code": 0,
    "msg": "已暂停实时人脸识别",
    "status": "pause"
  }
}
========== WEBSOCKET SEND ==========
ws://127.0.0.1:8000/ws/recognition/live

{
  "msg_type": "control_recognition",
  "data": {
    "status": "resume"
  }
}

========== WEBSOCKET RECEIVE ==========
{
  "msg_type": "control_ack",
  "data": {
    "code": 0,
    "msg": "已恢复实时人脸识别",
    "status": "resume"
  }
}
```



错误msg

```
========== WEBSOCKET SEND ==========
ws://127.0.0.1:8000/ws/recognition/live

{
  "msg_type": "unknown_message_for_e2e",
  "data": {}
}

========== WEBSOCKET RECEIVE ==========
{
  "msg_type": "stream_error",
  "data": {
    "error_msg": "Unsupported msg_type: unknown_message_for_e2e"
  }
}
```




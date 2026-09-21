"""
    API 成功响应统一封装工具
    将接口返回的数据统一包装成固定格式：
    {
        "code":0,
        "msg":"success",
        "data":{}
    }
"""
from __future__ import annotations

from face_api.core.errors import ErrorCode


def api_response(data: dict | list | None = None, msg: str = "success", code: int | ErrorCode = ErrorCode.SUCCESS) -> dict:
    return {"code": int(code), "msg": msg, "data": data if data is not None else {}}
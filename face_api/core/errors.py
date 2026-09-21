"""
    Face Recognition API 的统一错误定义模块
    定义所有业务错误码
    定义统一业务异常类型
    定义统一错误返回 JSON 格式
"""
from __future__ import annotations

from enum import IntEnum


class ErrorCode(IntEnum):
    SUCCESS = 0  # 成功状态
    INVALID_ARGUMENT = 1001 # 参数错误
    INVALID_BASE64_IMAGE = 1002 # Base64 图片错误，上传的人脸图片不是合法 Base64。
    FACE_NOT_FOUND = 1003 # 没检测到人脸
    MULTIPLE_FACES_UNSUPPORTED = 1004
    SUBJECT_ALREADY_EXISTS = 1005 # 用户已经存在
    SUBJECT_NOT_FOUND = 1006 # 用户不存在
    GALLERY_EMPTY = 1007 # 人脸库为空
    MODEL_NOT_READY = 2001 # 模型未准备
    INFERENCE_FAILED = 2002 # 推理失败 模型异常。
    STORAGE_ERROR = 3001 # 存储错误

# 自定义异常类
class FaceApiError(Exception):
    # 保存异常信息
    def __init__(self, code: ErrorCode, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status

# 统一错误响应函数
def business_error_response(code: ErrorCode, message: str, data: dict | None = None) -> dict:
    return {"code": int(code), "msg": message, "data": data or {}}
from __future__ import annotations

import base64
import binascii

from .errors import ErrorCode, FaceApiError

# 去掉 Data URL 前缀
def _strip_data_url(value: str) -> str:
    # 是否以data：开始且是否包含逗号
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value

# Base64 字符串 → 图片二进制 bytes
def decode_base64_image(value: str) -> bytes:
    # 是不是字符串，去掉空格后为空
    if not isinstance(value, str) or not value.strip():
        raise FaceApiError(ErrorCode.INVALID_BASE64_IMAGE, "图片 Base64 不能为空")
    # 去掉 Data URL 前缀
    normalized = _strip_data_url(value.strip())
    try:
        return base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FaceApiError(ErrorCode.INVALID_BASE64_IMAGE, "图片 Base64 格式错误") from exc

# 图片 bytes → Base64 字符串
def encode_base64_image(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
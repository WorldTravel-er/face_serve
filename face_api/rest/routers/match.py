from __future__ import annotations

from contextlib import nullcontext

from fastapi import APIRouter, Request

from face_api.core.errors import ErrorCode, FaceApiError
from face_api.core.image_codec import decode_base64_image
from face_api.rest.responses import api_response
from face_api.rest.schemas import MatchRequest

# tags=["recognition-match"]用于 Swagger 文档分类。
# 访问：/docs
# 显示：
# recognition-match
#     POST /api/recognition/match
router = APIRouter(prefix="/api/recognition", tags=["recognition-match"])


@router.post("/match")
def match_subjects(payload: MatchRequest, request: Request) -> dict:
    store = request.app.state.store
    recognition = request.app.state.recognition_service
    # 获取所有注册人脸。
    candidates = store.list_feature_records(recognition.embedding_identity())
    if not candidates:
        raise FaceApiError(ErrorCode.GALLERY_EMPTY, "Gallery is empty", http_status=400)
    # 解码图片
    image_bytes = decode_base64_image(payload.image_base64)
    # 调用识别服务
    context = getattr(recognition, "performance_context", None)
    manager = context(channel="rest_api", route="/api/recognition/match") if callable(context) else nullcontext()
    with manager:
        result = recognition.match(image_bytes, candidates, top_k=payload.top_k, threshold=payload.threshold)
    # 返回结果
    return api_response({"result": result}, msg="\u6bd4\u5bf9\u5b8c\u6210")

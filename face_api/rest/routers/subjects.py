"""
    人脸库管理接口（Subject Management API）。
    负责：
        注册人脸（新增 Subject）
        查询人员信息
        更新人员信息
        删除人员信息
"""
from __future__ import annotations

from contextlib import nullcontext

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from face_api.core.image_codec import decode_base64_image
from face_api.rest.responses import api_response
from face_api.rest.schemas import SubjectCreateRequest, SubjectUpdateRequest

router = APIRouter(prefix="/api/recognition/subjects", tags=["recognition-subjects"])


def _subject_data(record) -> dict:
    return {
        "subject_id": record.subject_id,
        "name": record.name,
        "image_base64": record.image_base64,
        "create_time": record.create_time,
        "update_time": record.update_time,
    }


def _performance_context(recognition, *, route: str):
    context = getattr(recognition, "performance_context", None)
    return context(channel="rest_api", route=route) if callable(context) else nullcontext()


@router.post("")
def create_subject(payload: SubjectCreateRequest, request: Request) -> JSONResponse:
    store = request.app.state.store
    recognition = request.app.state.recognition_service
    image_bytes = decode_base64_image(payload.image_base64)
    with _performance_context(recognition, route="/api/recognition/subjects"):
        feature = recognition.extract_feature(image_bytes)
    record = store.create_subject(
        payload.subject_id,
        payload.name,
        image_bytes,
        feature,
        embedding_identity=recognition.embedding_identity(),
    )
    return JSONResponse(
        status_code=201,
        content=api_response(
            {"subject_id": record.subject_id, "create_time": record.create_time},
            msg="\u4eba\u8138\u5f55\u5165\u6210\u529f",
        ),
    )


@router.get("/{subject_id}")
def get_subject(subject_id: str, request: Request) -> dict:
    record = request.app.state.store.get_subject(subject_id)
    return api_response(_subject_data(record))


@router.put("/{subject_id}")
def update_subject(subject_id: str, payload: SubjectUpdateRequest, request: Request) -> dict:
    store = request.app.state.store
    recognition = request.app.state.recognition_service
    image_bytes = None
    feature = None
    if payload.image_base64 is not None:
        image_bytes = decode_base64_image(payload.image_base64)
        with _performance_context(recognition, route="/api/recognition/subjects/{subject_id}"):
            feature = recognition.extract_feature(image_bytes)
    embedding_identity = recognition.embedding_identity() if feature is not None else None
    record = store.update_subject(
        subject_id,
        name=payload.name,
        image_bytes=image_bytes,
        feature=feature,
        embedding_identity=embedding_identity,
    )
    return api_response(
        {"subject_id": record.subject_id, "update_time": record.update_time},
        msg="\u4e3b\u4f53\u4fe1\u606f\u66f4\u65b0\u6210\u529f",
    )


@router.delete("/{subject_id}")
def delete_subject(subject_id: str, request: Request) -> dict:
    record = request.app.state.store.delete_subject(subject_id)
    return api_response({"subject_id": record.subject_id}, msg="\u4eba\u8138\u6570\u636e\u5220\u9664\u6210\u529f")

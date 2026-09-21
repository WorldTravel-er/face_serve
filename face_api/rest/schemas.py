from __future__ import annotations
"""
整体作用就是：
前端发送的数据 → Pydantic 自动校验 → 校验通过才进入接口，否则自动返回 422 错误。
"""
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


class SubjectCreateRequest(BaseModel):
    subject_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    image_base64: str = Field(..., min_length=1)

    @field_validator("subject_id", "name", "image_base64")
    def _strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class SubjectUpdateRequest(BaseModel):
    name: Optional[str] = None
    image_base64: Optional[str] = None

    @field_validator("name", "image_base64")
    def _strip_optional(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class MatchRequest(BaseModel):
    image_base64: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=3, gt=0)
    threshold: Optional[float] = Field(default=0.25, ge=0.0, le=1.0)

    @field_validator("image_base64")
    def _strip_image(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class SubjectResponse(BaseModel):
    subject_id: str
    name: str
    image_base64: str
    create_time: str
    update_time: str


class MatchResult(BaseModel):
    subject_id: str
    name: str
    similarity: float


class VideoAnalyzeRequest(BaseModel):
    task_name: str = Field(..., min_length=1)
    source_url: str = Field(..., min_length=1)

    @field_validator("task_name", "source_url")
    def _strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("source_url")
    def _validate_source_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source_url must be an http or https URL")
        return value

from __future__ import annotations

import re
import sqlite3
import threading
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from face_api.core.errors import ErrorCode, FaceApiError
from face_api.core.image_codec import encode_base64_image

# 定义人员完整信息
# frozen=True
# 表示：不可修改。
# 类似：只读对象。
@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    name: str
    image_path: str
    image_base64: str
    feature: np.ndarray
    create_time: str
    update_time: str


@dataclass(frozen=True)
class SubjectFeatureRecord:
    subject_id: str
    name: str
    feature: np.ndarray


@dataclass(frozen=True)
class EmbeddingIdentity:
    """The exact runtime/model pair that produced a face embedding."""

    runtime: str
    model_sha256: str
    pipeline_sha256: str = "legacy"


class SubjectStore:
    def __init__(self, root_dir: str | Path, clock: Callable[[], str] | None = None) -> None:
        # 数据目录
        self.root_dir = Path(root_dir)
        # 数据库路径
        self.db_path = self.root_dir / "face_api.db"
        # 图片目录
        self.subjects_dir = self.root_dir / "subjects"
        # 时间函数
        self.clock = clock or (lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        #RLock：可重复进入锁。防止：多线程数据库冲突。
        self._lock = threading.RLock()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.subjects_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # 数据库连接
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            self._commit(conn)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _commit(self, conn: sqlite3.Connection) -> None:
        """Separate hook keeps transaction-finalization failures testable."""
        conn.commit()

    # 初始化数据库表
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS subjects (
                    subject_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    feature BLOB NOT NULL,
                    feature_dim INTEGER NOT NULL,
                    create_time TEXT NOT NULL,
                    update_time TEXT NOT NULL
                )
                """
            )
            self._ensure_embedding_columns(conn)

    def _ensure_embedding_columns(self, conn: sqlite3.Connection) -> None:
        names = {row["name"] for row in conn.execute("PRAGMA table_info(subjects)")}
        if "embedding_runtime" not in names:
            conn.execute("ALTER TABLE subjects ADD COLUMN embedding_runtime TEXT NOT NULL DEFAULT 'legacy'")
        if "embedding_model_sha256" not in names:
            conn.execute("ALTER TABLE subjects ADD COLUMN embedding_model_sha256 TEXT NOT NULL DEFAULT 'legacy'")
        if "embedding_pipeline_sha256" not in names:
            conn.execute("ALTER TABLE subjects ADD COLUMN embedding_pipeline_sha256 TEXT NOT NULL DEFAULT 'legacy'")
    # 创建人员
    def create_subject(
        self,
        subject_id: str,
        name: str,
        image_bytes: bytes,
        feature: np.ndarray,
        *,
        embedding_identity: EmbeddingIdentity,
    ) -> SubjectRecord:
        self._validate_subject_id(subject_id)
        self._validate_name(name)
        self._validate_embedding_identity(embedding_identity)
        feature_array = self._normalize_feature(feature)
        now = self.clock()
        image_path = self._image_path(subject_id)
        with self._lock:
            temp_image_path: Path | None = None
            image_installed = False
            try:
                with self._connect() as conn:
                    if self._subject_exists(conn, subject_id):
                        raise FaceApiError(ErrorCode.SUBJECT_ALREADY_EXISTS, "Subject already exists", http_status=409)
                    temp_image_path = self._write_temp_image(image_path, image_bytes)
                    os.replace(temp_image_path, image_path)
                    image_installed = True
                    conn.execute(
                        """
                        INSERT INTO subjects(
                            subject_id, name, image_path, feature, feature_dim,
                            create_time, update_time, embedding_runtime, embedding_model_sha256, embedding_pipeline_sha256
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            subject_id,
                            name,
                            str(image_path),
                            feature_array.tobytes(),
                            int(feature_array.size),
                            now,
                            now,
                            embedding_identity.runtime,
                            embedding_identity.model_sha256,
                            embedding_identity.pipeline_sha256,
                        ),
                    )
            except Exception:
                if image_installed:
                    image_path.unlink(missing_ok=True)
                if temp_image_path is not None:
                    temp_image_path.unlink(missing_ok=True)
                raise
        return self.get_subject(subject_id)
    # 根据ID查询
    def get_subject(self, subject_id: str) -> SubjectRecord:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM subjects WHERE subject_id = ?", (subject_id,)).fetchone()
        if row is None:
            raise FaceApiError(ErrorCode.SUBJECT_NOT_FOUND, "Subject not found", http_status=404)
        return self._record_from_row(row)
    # 更新信息
    def update_subject(
        self,
        subject_id: str,
        name: str | None = None,
        image_bytes: bytes | None = None,
        feature: np.ndarray | None = None,
        embedding_identity: EmbeddingIdentity | None = None,
    ) -> SubjectRecord:
        if name is None and image_bytes is None and feature is None:
            raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Invalid argument")
        if (image_bytes is None) != (feature is None):
            raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Invalid argument")
        if feature is not None and embedding_identity is None:
            raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Embedding identity is required when replacing a feature")
        if embedding_identity is not None:
            self._validate_embedding_identity(embedding_identity)
        if name is not None:
            self._validate_name(name)

        with self._lock:
            current = self.get_subject(subject_id)
            new_name = name if name is not None else current.name
            new_image_path = Path(current.image_path)
            new_feature = current.feature if feature is None else self._normalize_feature(feature)
            now = self.clock()
            temp_image_path: Path | None = None
            old_image_bytes: bytes | None = None
            image_replaced = False
            try:
                if image_bytes is not None:
                    old_image_bytes = new_image_path.read_bytes()
                    temp_image_path = self._write_temp_image(new_image_path, image_bytes)
                with self._connect() as conn:
                    if temp_image_path is not None:
                        os.replace(temp_image_path, new_image_path)
                        image_replaced = True
                    conn.execute(
                        """
                        UPDATE subjects
                        SET name = ?, feature = ?, feature_dim = ?, update_time = ?,
                            embedding_runtime = CASE WHEN ? THEN ? ELSE embedding_runtime END,
                            embedding_model_sha256 = CASE WHEN ? THEN ? ELSE embedding_model_sha256 END,
                            embedding_pipeline_sha256 = CASE WHEN ? THEN ? ELSE embedding_pipeline_sha256 END
                        WHERE subject_id = ?
                        """,
                        (
                            new_name,
                            new_feature.tobytes(),
                            int(new_feature.size),
                            now,
                            feature is not None,
                            embedding_identity.runtime if embedding_identity is not None else None,
                            feature is not None,
                            embedding_identity.model_sha256 if embedding_identity is not None else None,
                            feature is not None,
                            embedding_identity.pipeline_sha256 if embedding_identity is not None else None,
                            subject_id,
                        ),
                    )
            except Exception as database_error:
                if image_replaced and old_image_bytes is not None:
                    try:
                        self._replace_image(new_image_path, old_image_bytes)
                    except Exception as restore_error:
                        # Both errors are material: the DB change did not commit and the
                        # on-disk image may no longer correspond to it.  Do not hide either.
                        raise ExceptionGroup(
                            "subject update failed and image rollback failed",
                            [database_error, restore_error],
                        ) from None
                if temp_image_path is not None:
                    temp_image_path.unlink(missing_ok=True)
                raise
        return self.get_subject(subject_id)
    # 删除注册人脸信息
    def delete_subject(self, subject_id: str) -> SubjectRecord:
        with self._lock:
            current = self.get_subject(subject_id)
            with self._connect() as conn:
                conn.execute("DELETE FROM subjects WHERE subject_id = ?", (subject_id,))
            Path(current.image_path).unlink(missing_ok=True)
        return current
    # 获取匹配库,列出所有注册信息
    def list_feature_records(self, embedding_identity: EmbeddingIdentity) -> list[SubjectFeatureRecord]:
        self._validate_embedding_identity(embedding_identity)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT subject_id, name, feature, feature_dim FROM subjects
                    WHERE embedding_runtime = ? AND embedding_model_sha256 = ? AND embedding_pipeline_sha256 = ?
                    ORDER BY create_time, subject_id
                    """,
                    (embedding_identity.runtime, embedding_identity.model_sha256, embedding_identity.pipeline_sha256),
                ).fetchall()
        return [
            SubjectFeatureRecord(
                subject_id=row["subject_id"],
                name=row["name"],
                feature=self._feature_from_row(row),
            )
            for row in rows
        ]
    def list_subjects(self) -> list[SubjectRecord]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("SELECT * FROM subjects ORDER BY create_time, subject_id").fetchall()
        return [self._record_from_row(row) for row in rows]
    # 检查ID是否存在
    def _subject_exists(self, conn: sqlite3.Connection, subject_id: str) -> bool:
        return conn.execute("SELECT 1 FROM subjects WHERE subject_id = ?", (subject_id,)).fetchone() is not None
    # 数据库记录
    def _record_from_row(self, row: sqlite3.Row) -> SubjectRecord:
        image_path = Path(row["image_path"])
        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            raise FaceApiError(ErrorCode.STORAGE_ERROR, "Failed to read subject image", http_status=500) from exc
        return SubjectRecord(
            subject_id=row["subject_id"],
            name=row["name"],
            image_path=str(image_path),
            image_base64=encode_base64_image(image_bytes),
            feature=self._feature_from_row(row),
            create_time=row["create_time"],
            update_time=row["update_time"],
        )
    #
    def _feature_from_row(self, row: sqlite3.Row) -> np.ndarray:
        dim = int(row["feature_dim"])
        return np.frombuffer(row["feature"], dtype=np.float32, count=dim).copy()
    # 生成图片路径
    def _image_path(self, subject_id: str) -> Path:
        return self.subjects_dir / f"{self._safe_filename(subject_id)}.jpg"
    # 保存图片
    def _write_image(self, image_path: Path, image_bytes: bytes) -> None:
        if not image_bytes:
            raise FaceApiError(ErrorCode.INVALID_BASE64_IMAGE, "Image content is empty")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)

    def _write_temp_image(self, image_path: Path, image_bytes: bytes) -> Path:
        if not image_bytes:
            raise FaceApiError(ErrorCode.INVALID_BASE64_IMAGE, "Image content is empty")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=image_path.parent, prefix=f".{image_path.name}.", delete=False) as handle:
                temporary_path = Path(handle.name)
                handle.write(image_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            return temporary_path
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    def _replace_image(self, image_path: Path, image_bytes: bytes) -> None:
        temporary_path = self._write_temp_image(image_path, image_bytes)
        try:
            os.replace(temporary_path, image_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    # 特征标准化
    def _normalize_feature(self, feature: np.ndarray | Iterable[float]) -> np.ndarray:
        array = np.asarray(feature, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise FaceApiError(ErrorCode.INFERENCE_FAILED, "Face feature is empty", http_status=500)
        return array
    # 检查ID。
    def _validate_subject_id(self, subject_id: str) -> None:
        if not subject_id or not subject_id.strip():
            raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Invalid argument")
    # 检查姓名。
    def _validate_name(self, name: str) -> None:
        if not name or not name.strip():
            raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Invalid argument")

    def _validate_embedding_identity(self, identity: EmbeddingIdentity) -> None:
        if identity.runtime not in {"onnx", "rknn"}:
            raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Unsupported embedding runtime")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", identity.model_sha256):
            raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Invalid embedding model SHA-256")
        if identity.pipeline_sha256 != "legacy" and not re.fullmatch(r"[0-9a-fA-F]{64}", identity.pipeline_sha256):
            raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Invalid embedding pipeline SHA-256")
    # 生成安全文件名
    def _safe_filename(self, subject_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", subject_id.strip())
        return safe or "subject"

from __future__ import annotations

import math
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from face_api.core.errors import ErrorCode, FaceApiError


@dataclass(frozen=True)
class VideoFaceRecord:
    subject_id: str
    name: str
    first_appear_ts: str
    last_disappear_ts: str
    track_id: int | None = None


@dataclass(frozen=True)
class VideoAnalysisTask:
    task_id: str
    task_name: str
    source_url: str
    task_status: str
    create_time: str
    start_time: str
    end_time: str
    error_info: str
    processed_frames: int
    total_frames: int
    video_duration: float
    progress_percent: float
    progress_updated_time: str
    face_records: list[VideoFaceRecord]


class VideoAnalysisStore:
    def __init__(self, root_dir: str | Path, clock: Callable[[], str] | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.db_path = self.root_dir / "face_api.db"
        self.clock = clock or (lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._lock = threading.RLock()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            # Serialize schema inspection and migration across app instances/processes.
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS video_analysis_tasks (
                    task_name TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE,
                    source_url TEXT NOT NULL,
                    task_status TEXT NOT NULL,
                    create_time TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    error_info TEXT NOT NULL,
                    processed_frames INTEGER NOT NULL DEFAULT 0,
                    total_frames INTEGER NOT NULL DEFAULT 0,
                    video_duration REAL NOT NULL DEFAULT 0.0,
                    progress_percent REAL NOT NULL DEFAULT 0.0,
                    progress_updated_time TEXT NOT NULL DEFAULT ''
                )
                """
            )
            task_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(video_analysis_tasks)").fetchall()
            }
            if "processed_frames" not in task_columns:
                conn.execute("ALTER TABLE video_analysis_tasks ADD COLUMN processed_frames INTEGER NOT NULL DEFAULT 0")
            if "total_frames" not in task_columns:
                conn.execute("ALTER TABLE video_analysis_tasks ADD COLUMN total_frames INTEGER NOT NULL DEFAULT 0")
            if "video_duration" not in task_columns:
                conn.execute("ALTER TABLE video_analysis_tasks ADD COLUMN video_duration REAL NOT NULL DEFAULT 0.0")
            if "progress_percent" not in task_columns:
                conn.execute("ALTER TABLE video_analysis_tasks ADD COLUMN progress_percent REAL NOT NULL DEFAULT 0.0")
            if "progress_updated_time" not in task_columns:
                conn.execute("ALTER TABLE video_analysis_tasks ADD COLUMN progress_updated_time TEXT NOT NULL DEFAULT ''")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS video_face_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT NOT NULL,
                    track_id INTEGER,
                    subject_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    first_appear_ts TEXT NOT NULL,
                    last_disappear_ts TEXT NOT NULL,
                    FOREIGN KEY(task_name) REFERENCES video_analysis_tasks(task_name) ON DELETE CASCADE
                )
                """
            )
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(video_face_records)").fetchall()
            }
            if "track_id" not in columns:
                conn.execute("ALTER TABLE video_face_records ADD COLUMN track_id INTEGER")
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_video_face_records_task_track
                ON video_face_records(task_name, track_id)
                """
            )

    def create_task(self, task_name: str, source_url: str) -> VideoAnalysisTask:
        task_name = self._required(task_name)
        source_url = self._required(source_url)
        now = self.clock()
        with self._lock:
            with self._connect() as conn:
                if self._task_exists(conn, task_name):
                    raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Video analysis task already exists", http_status=409)
                task_id = self._next_task_id(conn, now)
                conn.execute(
                    """
                    INSERT INTO video_analysis_tasks(
                        task_name, task_id, source_url, task_status, create_time, start_time, end_time, error_info
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (task_name, task_id, source_url, "running", now, now, "", ""),
                )
        return self.get_task(task_name)

    def get_task(self, task_name: str) -> VideoAnalysisTask:
        task_name = self._required(task_name)
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM video_analysis_tasks WHERE task_name = ?", (task_name,)).fetchone()
                if row is None:
                    raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Video analysis task not found", http_status=404)
                records = conn.execute(
                    """
                    SELECT track_id, subject_id, name, first_appear_ts, last_disappear_ts
                    FROM video_face_records
                    WHERE task_name = ?
                    ORDER BY first_appear_ts, track_id, subject_id
                    """,
                    (task_name,),
                ).fetchall()
        return self._task_from_row(row, records)

    def finish_task(self, task_name: str, face_records: list[VideoFaceRecord]) -> VideoAnalysisTask:
        self._validate_new_records(face_records)
        now = self.clock()
        with self._lock:
            with self._connect() as conn:
                if not self._task_exists(conn, task_name):
                    raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Video analysis task not found", http_status=404)
                conn.execute("DELETE FROM video_face_records WHERE task_name = ?", (task_name,))
                conn.executemany(
                    """
                    INSERT INTO video_face_records(
                        task_name, track_id, subject_id, name, first_appear_ts, last_disappear_ts
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            task_name,
                            record.track_id,
                            record.subject_id,
                            record.name,
                            record.first_appear_ts,
                            record.last_disappear_ts,
                        )
                        for record in face_records
                    ],
                )
                conn.execute(
                    """
                    UPDATE video_analysis_tasks
                    SET task_status = ?, end_time = ?, error_info = ?
                    WHERE task_name = ?
                    """,
                    ("finished", now, "", task_name),
                )
        return self.get_task(task_name)

    def fail_task(self, task_name: str, error_info: str) -> VideoAnalysisTask:
        now = self.clock()
        with self._lock:
            with self._connect() as conn:
                if not self._task_exists(conn, task_name):
                    raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Video analysis task not found", http_status=404)
                conn.execute(
                    """
                    UPDATE video_analysis_tasks
                    SET task_status = ?, end_time = ?, error_info = ?
                    WHERE task_name = ?
                    """,
                    ("failed", now, error_info or "Video analysis failed", task_name),
                )
        return self.get_task(task_name)

    def update_task_progress(self, task_name: str, processed_frames: int, total_frames: int) -> VideoAnalysisTask:
        task_name = self._required(task_name)
        processed = max(int(processed_frames or 0), 0)
        total = max(int(total_frames or 0), 0)
        if total > 0:
            processed = min(processed, total)
            progress_percent = processed / total * 100.0
        else:
            progress_percent = 0.0
        now = self.clock()
        with self._lock:
            with self._connect() as conn:
                if not self._task_exists(conn, task_name):
                    raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Video analysis task not found", http_status=404)
                conn.execute(
                    """
                    UPDATE video_analysis_tasks
                    SET processed_frames = ?, total_frames = ?, progress_percent = ?, progress_updated_time = ?
                    WHERE task_name = ?
                    """,
                    (processed, total, progress_percent, now, task_name),
                )
        return self.get_task(task_name)

    def update_task_video_duration(self, task_name: str, video_duration: float) -> VideoAnalysisTask:
        task_name = self._required(task_name)
        try:
            duration = float(video_duration or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if not math.isfinite(duration):
            duration = 0.0
        if duration < 0.0:
            duration = 0.0
        with self._lock:
            with self._connect() as conn:
                if not self._task_exists(conn, task_name):
                    raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Video analysis task not found", http_status=404)
                conn.execute(
                    """
                    UPDATE video_analysis_tasks
                    SET video_duration = ?
                    WHERE task_name = ?
                    """,
                    (duration, task_name),
                )
        return self.get_task(task_name)

    def _task_exists(self, conn: sqlite3.Connection, task_name: str) -> bool:
        return conn.execute("SELECT 1 FROM video_analysis_tasks WHERE task_name = ?", (task_name,)).fetchone() is not None

    def _next_task_id(self, conn: sqlite3.Connection, now: str) -> str:
        date_part = now[:10].replace("-", "")
        count = conn.execute(
            "SELECT COUNT(*) FROM video_analysis_tasks WHERE task_id LIKE ?",
            (f"video_task_{date_part}_%",),
        ).fetchone()[0]
        return f"video_task_{date_part}_{int(count) + 1:03d}"

    def _task_from_row(self, row: sqlite3.Row, records: list[sqlite3.Row]) -> VideoAnalysisTask:
        return VideoAnalysisTask(
            task_id=row["task_id"],
            task_name=row["task_name"],
            source_url=row["source_url"],
            task_status=row["task_status"],
            create_time=row["create_time"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            error_info=row["error_info"],
            processed_frames=int(row["processed_frames"] or 0),
            total_frames=int(row["total_frames"] or 0),
            video_duration=float(row["video_duration"] or 0.0),
            progress_percent=float(row["progress_percent"] or 0.0),
            progress_updated_time=row["progress_updated_time"],
            face_records=[
                VideoFaceRecord(
                    subject_id=record["subject_id"],
                    name=record["name"],
                    first_appear_ts=record["first_appear_ts"],
                    last_disappear_ts=record["last_disappear_ts"],
                    track_id=record["track_id"],
                )
                for record in records
            ],
        )

    @staticmethod
    def _validate_new_records(face_records: list[VideoFaceRecord]) -> None:
        for record in face_records:
            if isinstance(record.track_id, bool) or not isinstance(record.track_id, int):
                raise ValueError("track_id must be an integer for new video face records")

    def _required(self, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise FaceApiError(ErrorCode.INVALID_ARGUMENT, "Invalid argument")
        return value

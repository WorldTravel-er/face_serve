from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from face_api.rest.responses import api_response
from face_api.rest.schemas import VideoAnalyzeRequest

router = APIRouter(prefix="/api/recognition/video", tags=["recognition-video"])


def _task_data(task) -> dict:
    return {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "source_url": task.source_url,
        "task_status": task.task_status,
        "video_duration": task.video_duration,
        "create_time": task.create_time,
        "start_time": task.start_time,
        "end_time": task.end_time,
        "face_records": [
            {
                # "track_id": record.track_id,
                "subject_id": record.subject_id,
                "name": record.name,
                "first_appear_ts": record.first_appear_ts,
                "last_disappear_ts": record.last_disappear_ts,
            }
            for record in task.face_records
        ],
        "error_info": task.error_info,
    }


def _created_task_data(task) -> dict:
    return {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "create_time": task.create_time,
        "source_url": task.source_url,
        "task_status": task.task_status,
    }


def _format_task_progress(task) -> str:
    processed = int(getattr(task, "processed_frames", 0) or 0)
    total = int(getattr(task, "total_frames", 0) or 0)
    status = getattr(task, "task_status", "")
    if total > 0:
        safe_processed = min(processed, total)
        percent = float(getattr(task, "progress_percent", 0.0) or 0.0)
        progress = f"{safe_processed}/{total} frames ({percent:.1f}%)"
    else:
        progress = f"{processed} frames"
    return f"[video-analysis] progress: task={task.task_name}, status={status}, {progress}"


def _print_task_progress(task) -> None:
    print(_format_task_progress(task), flush=True)


def _print_created_source_if_not_downloading(task, request: Request) -> None:
    config = getattr(request.app.state, "config", None)
    download_enabled = bool(getattr(config, "video_analysis_download_source_enabled", False))
    if download_enabled:
        return
    print(f"[video-analysis] source: task={task.task_name}, url={task.source_url}", flush=True)


@router.post("/analyze")
def create_video_analysis_task(payload: VideoAnalyzeRequest, request: Request) -> JSONResponse:
    task = request.app.state.video_analysis_store.create_task(payload.task_name, payload.source_url)
    request.app.state.video_analysis_runner.submit_task(task.task_name)
    _print_created_source_if_not_downloading(task, request)
    return JSONResponse(
        status_code=201,
        content=api_response(_created_task_data(task), msg="\u957f\u89c6\u9891\u5206\u6790\u4efb\u52a1\u521b\u5efa\u6210\u529f"),
    )


@router.get("/analyze/{task_name}")
def get_video_analysis_result(task_name: str, request: Request) -> dict:
    task = request.app.state.video_analysis_store.get_task(task_name)
    _print_task_progress(task)
    return api_response(_task_data(task), msg="\u67e5\u8be2\u6210\u529f")

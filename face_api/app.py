from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from face_api.core.config import ApiConfig, default_config
from face_api.core.errors import ErrorCode, FaceApiError, business_error_response
from face_api.core.recognition import RecognitionService
from face_api.core.subject_store import SubjectStore
from face_api.live.gateway.router import router as live_router
from face_api.rest.routers.health import router as health_router
from face_api.rest.routers.match import router as match_router
from face_api.rest.routers.subjects import router as subjects_router
from face_api.rest.routers.video import router as video_router
from face_api.video.analysis import VideoAnalysisProcessor
from face_api.video.runner import VideoAnalysisRunner
from face_api.video.store import VideoAnalysisStore
from fastapi.middleware.cors import CORSMiddleware

def _build_video_tracker_factory(config: ApiConfig, recognition_service: RecognitionService):
    from face_api.video.face_tracker import ByteTrackFaceTracker

    return lambda: ByteTrackFaceTracker(
        recognition_service.detector,
        high_threshold=max(0.05, config.retinaface_conf),
        min_face_area_ratio=0.001,
        max_face_area_ratio=0.30,
        min_face_aspect_ratio=0.30,
        max_face_aspect_ratio=1.35,
        max_candidates=25,
        min_face_score=config.min_face_score,
        max_faces_per_frame=config.max_faces_per_frame,
        face_selection=config.face_selection,
    )


def create_app(
    config: ApiConfig | None = None,
    store: SubjectStore | None = None,
    recognition_service: RecognitionService | None = None,
    live_runner: object | None = None,
    video_analysis_runner: object | None = None,
) -> FastAPI:
    resolved_config = config or default_config()
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            try:
                if video_analysis_runner is None:
                    # Drain queued jobs before releasing the sessions they use.
                    await run_in_threadpool(app.state.video_analysis_runner.shutdown, wait=True)
            finally:
                if recognition_service is None:
                    await run_in_threadpool(app.state.recognition_service.close)

    app = FastAPI(title="Face Recognition REST API", version="1.0.0", lifespan=lifespan)
    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源，生产环境建议指定具体域名
        allow_credentials=True,
        allow_methods=["*"],  # 允许所有 HTTP 方法
        allow_headers=["*"],  # 允许所有请求头
    )
    app.state.config = resolved_config
    app.state.store = store if store is not None else SubjectStore(resolved_config.data_dir)
    app.state.recognition_service = recognition_service if recognition_service is not None else RecognitionService(resolved_config)
    app.state.live_runner = live_runner

    video_data_dir = getattr(app.state.store, "root_dir", resolved_config.data_dir)
    app.state.video_analysis_store = VideoAnalysisStore(video_data_dir)
    video_processor = VideoAnalysisProcessor(
        video_store=app.state.video_analysis_store,
        subject_store=app.state.store,
        recognition_service=app.state.recognition_service,
        tracker_factory=_build_video_tracker_factory(resolved_config, app.state.recognition_service),
        # threshold=resolved_config.live_match_threshold,
        threshold=0.3,
        sample_interval_seconds=resolved_config.video_analysis_recognition_interval_seconds,
        recognition_queue_size=resolved_config.video_analysis_recognition_queue_size,
        download_source_enabled=resolved_config.video_analysis_download_source_enabled,
    )
    app.state.video_analysis_runner = (
        video_analysis_runner
        if video_analysis_runner is not None
        else VideoAnalysisRunner(
            video_processor,
            max_concurrency=resolved_config.video_analysis_max_concurrency,
        )
    )

    app.include_router(health_router)
    app.include_router(subjects_router)
    app.include_router(match_router)
    app.include_router(live_router)
    app.include_router(video_router)

    @app.exception_handler(FaceApiError)
    async def handle_face_api_error(request: Request, exc: FaceApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=business_error_response(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=business_error_response(ErrorCode.INVALID_ARGUMENT, "Invalid request argument"),
        )

    return app

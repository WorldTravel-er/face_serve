from __future__ import annotations
"""
    娑撴艾濮熺拫鍐ㄥ閺嶇绺鹃敍宀€顓搁悶鍡樺閺堝orker, 閹貉冨煑鐠囧棗鍩嗗ù浣衡柤閿涘矁鐨熸惔锔剧处鐎涙﹫绱濋崥鎴炲复閸欙絾甯归柅浣虹波閺?
"""
import asyncio
import inspect
import queue
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from face_api.core.config import ApiConfig
from face_api.live.capture.worker import CaptureWorker, OpenCvFrameReader, STREAM_OFFLINE_MESSAGE
from face_api.live.cache.detection_cache import DetectionCache, DetectionSnapshot
from face_api.live.pipeline.detection_worker import DetectionWorker
from face_api.live.monitoring.device_info import preview_device_label
from face_api.live.capture.frame_buffer import LatestFrameBuffer
from face_api.live.monitoring.metrics import LiveMetrics
from face_api.live.preview.renderer import LivePreviewRenderer, LivePreviewWorker
from face_api.live.gateway.protocol import build_face_match_result, build_stream_error, format_live_time
from face_api.live.cache.result_cache import RecognitionResultCache
from face_api.live.gateway.state import LiveRecognitionState
from face_api.live.diagnostics import LiveDiagnosticsLogger


Sender = Callable[[dict], Any]
FrameReaderFactory = Callable[[str], Any]
CaptureWorkerFactory = Callable[..., Any]
DetectionWorkerFactory = Callable[..., Any]
DateTimeClock = Callable[[], datetime]
CloseRequester = Callable[..., Any]

STREAM_ERROR_CLOSE_CODE = 1011
STREAM_ERROR_CLOSE_REASON = "stream error: video source offline"


@dataclass(frozen=True)
class WorkerError:
    source: str
    message: str
    close_connection: bool = False
# 娑撯偓娑撶尯unner鐎电懓绨叉稉鈧稉顏囶潒妫版垶绨崪灞肩娑擃亣鐦戦崚顐ｇウ缁?
class LiveRecognitionRunner:
    def __init__(
        self,
        *,
        config: ApiConfig,
        store: Any,
        recognition_service: Any,
        frame_reader_factory: FrameReaderFactory | None = None,
        capture_worker_factory: CaptureWorkerFactory | None = None,
        detection_worker_factory: DetectionWorkerFactory | None = None,
        metrics: Any | None = None,
        clock: DateTimeClock | None = None,
        max_frames: int | None = None,
        preview_renderer: Any | None = None,
        device_label: str | None = None,
    ) -> None:
        self.config = config# 閸忋劌鐪柊宥囩枂
        self.store = store# 娴滈缚鍔弫鐗堝祦鎼?
        self.recognition_service = recognition_service # 娴滈缚鍔Λ鈧ù瀣槕閸掝偅婀囬崝?

        self.frame_reader_factory = frame_reader_factory or OpenCvFrameReader
        self.capture_worker_factory = capture_worker_factory or CaptureWorker
        self.detection_worker_factory = detection_worker_factory or DetectionWorker
        self.metrics = metrics or LiveMetrics()
        self.clock = clock or datetime.now
        self.max_frames = max_frames
        self.preview_renderer = preview_renderer
        self.device_label = device_label or preview_device_label(self._describe_runtime())
        self.diagnostics = LiveDiagnosticsLogger.from_config(config)
        self.perf_counter = time.perf_counter

    async def run(
        self,
        state: LiveRecognitionState,
        send_json: Sender,
        request_close: CloseRequester | None = None,
        *,
        connection_id: str | None = None,
        websocket_accepted_at: float | None = None,
    ) -> None:
        # 閸掋倖鏌囪ぐ鎾冲娴滈缚鍔拠鍡楀焼閺堝秴濮熼弰顖氭儊閺€顖涘瘮閳ユ粍濯堕崚鍡楃础鐎圭偞妞傚ù浣规寜缁惧尅绱橲plit Pipeline閿涘鈧縿鈧?
        # 婵″倹鐏夐弨顖涘瘮閿涘矁铔嬫妯烩偓褑鍏樺ù浣衡柤閿涙稑顩ч弸婊€绗夐弨顖涘瘮閿涘矂鈧偓閸ョ偞妫悧鍫濆礋鐢嗙槕閸掝偅绁︾粙瀣ㄢ偓?
        self._log_runner_stage("started", connection_id=connection_id, websocket_accepted_at=websocket_accepted_at)
        if not self._supports_split_pipeline():
            await self._run_legacy_latest_frame(
                state,
                send_json,
                request_close=request_close,
                connection_id=connection_id,
                websocket_accepted_at=websocket_accepted_at,
            )
            return
        # 閼惧嘲褰囩憴鍡涱暥濞翠礁婀撮崸鈧?
        stream_url = str(self.config.live_stream_url)
        # 閸掓稑缂撶憴鍡涱暥鐢呯处鐎?
        frame_buffer = LatestFrameBuffer()
        # 閸掓稑缂撳Λ鈧ù瀣处鐎?
        detection_cache = DetectionCache()
        # 閸掓稑缂撶拠鍡楀焼缂佹挻鐏夌紓鎾崇摠
        result_cache = RecognitionResultCache()
        # 閸掓稑缂?Worker 闁挎瑨顕ら梼鐔峰灙
        worker_errors: queue.Queue[WorkerError] = queue.Queue()
        # 閸戝棗顦總鏂ょ窗
        # 鐟欏棝顣堕柌鍥肠缁捐法鈻?
        # 娴滈缚鍔Λ鈧ù瀣殠缁?
        # 妫板嫯顫嶉弰鍓с仛缁捐法鈻?
        capture_worker = self._create_capture_worker(
            stream_url,
            frame_buffer,
            worker_errors,
            connection_id=connection_id,
            websocket_accepted_at=websocket_accepted_at,
        )
        detection_worker = self._create_detection_worker(
            frame_buffer,
            detection_cache,
            state,
            worker_errors,
            connection_id=connection_id,
            websocket_accepted_at=websocket_accepted_at,
        )
        preview_worker = self._create_preview_worker(frame_buffer, result_cache, state, detection_cache=detection_cache)

        self._log_runner_stage("workers_created", connection_id=connection_id, websocket_accepted_at=websocket_accepted_at)
        capture_worker.start()
        self._log_runner_stage("capture_worker_started", connection_id=connection_id, websocket_accepted_at=websocket_accepted_at)
        detection_worker.start()
        self._log_runner_stage("detection_worker_started", connection_id=connection_id, websocket_accepted_at=websocket_accepted_at)
        if preview_worker is not None:
            preview_worker.start()
        # 娑撳﹣绔村▎鈥冲嚒缂佸繐浠涙禍楦垮姱鐠囧棗鍩嗛惃鍕梾濞村鎶氱紓鏍у娇
        last_recognized_detection_seq = 0
        # 瀹歌尙绮＄€瑰本鍨氭径姘毌濞喡ょ槕閸?
        recognized_frames = 0
        try:
            while not state.closed:
                # 婢跺嫮鎮婇崥搴″酱缁捐法鈻奸柨娆掝嚖閿涘苯鑻熼崥鎴炲复閸欙絽褰傞柅涔痵on娣団剝浼?
                sent_error_count = await self._drain_worker_errors(
                    worker_errors,
                    send_json,
                    state=state,
                    request_close=request_close,
                )
                if state.closed:
                    break
                #
                if state.is_paused:
                    # 婵″倹鐏夐幗鍕剼婢跺鍤庣粙瀣嚒缂佸繒绮ㄩ弶?
                    if self.max_frames is not None and not capture_worker.is_alive():
                        break
                    # 閹藉嫬鍎氭径瀵稿殠缁嬪鐥呯紒鎾存将閿涘瞼鎴风紒顓㈠櫚闂?
                    await asyncio.sleep(0.05)
                    continue
                # 閼惧嘲褰囬張鈧弬鐗堫梾濞村绮ㄩ弸?
                detection = detection_cache.get_valid(
                    ttl_seconds=getattr(self.config, "live_detection_result_ttl", 0.5)
                )
                if detection is None:
                    if sent_error_count and self.max_frames is not None:
                        break
                    # 婵″倹鐏夐柌鍥肠缁捐法鈻煎缁樺竴閿?闁偓閸?
                    if not capture_worker.is_alive() and self.max_frames is not None:
                        break
                    # 缁涘绶?0ms闁插秵鏌婂Λ鈧弻?
                    await asyncio.sleep(0.02)
                    continue
                # 闂冨弶顒涢柌宥咁槻濡偓濞?
                if detection.frame_seq == last_recognized_detection_seq:
                    if not capture_worker.is_alive() and self.max_frames is not None:
                        break
                    await asyncio.sleep(0.02)
                    continue
                last_recognized_detection_seq = detection.frame_seq
                # 閹笛嗩攽鐠囧棗鍩?
                live_result = await self._recognize_detection_and_push(
                    detection,
                    state,
                    send_json,
                    connection_id=connection_id,
                )
                result_cache.update(live_result)
                recognized_frames += 1

                if self.max_frames is not None and recognized_frames >= self.max_frames:
                    break

                fps = max(float(getattr(self.config, "live_recognition_fps", 5.0) or 5.0), 0.1)
                await asyncio.sleep(1.0 / fps)
        finally:
            if preview_worker is not None:
                preview_worker.stop()
            detection_worker.stop()
            capture_worker.stop()

    async def _run_legacy_latest_frame(
        self,
        state: LiveRecognitionState,
        send_json: Sender,
        request_close: CloseRequester | None = None,
        *,
        connection_id: str | None = None,
        websocket_accepted_at: float | None = None,
    ) -> None:
        stream_url = str(self.config.live_stream_url)
        frame_buffer = LatestFrameBuffer()
        result_cache = RecognitionResultCache()
        worker_errors: queue.Queue[WorkerError] = queue.Queue()
        capture_worker = self._create_capture_worker(
            stream_url,
            frame_buffer,
            worker_errors,
            connection_id=connection_id,
            websocket_accepted_at=websocket_accepted_at,
        )
        preview_worker = self._create_preview_worker(frame_buffer, result_cache, state, detection_cache=None)

        capture_worker.start()
        if preview_worker is not None:
            preview_worker.start()

        last_recognized_seq = 0
        last_seen_seq = 0
        observed_frames = 0
        recognized_frames = 0
        try:
            while not state.closed:
                sent_error_count = await self._drain_worker_errors(
                    worker_errors,
                    send_json,
                    state=state,
                    request_close=request_close,
                )
                if state.closed:
                    break
                latest = frame_buffer.get_latest()
                if latest is None:
                    if sent_error_count and self.max_frames is not None:
                        break
                    if not capture_worker.is_alive() and self.max_frames is not None:
                        break
                    await asyncio.sleep(0.02)
                    continue

                if latest.seq != last_seen_seq:
                    last_seen_seq = latest.seq
                    observed_frames += 1

                if state.is_paused:
                    if self.max_frames is not None and observed_frames >= self.max_frames:
                        break
                    await asyncio.sleep(0.05)
                    continue

                if latest.seq == last_recognized_seq:
                    if not capture_worker.is_alive() and self.max_frames is not None:
                        break
                    await asyncio.sleep(0.02)
                    continue

                last_recognized_seq = latest.seq
                live_result = await self._recognize_frame_and_push(
                    latest.frame,
                    state,
                    send_json,
                    connection_id=connection_id,
                    frame_seq=latest.seq,
                )
                result_cache.update(live_result)
                recognized_frames += 1

                if self.max_frames is not None and recognized_frames >= self.max_frames:
                    break

                fps = max(float(getattr(self.config, "live_recognition_fps", 5.0) or 5.0), 0.1)
                await asyncio.sleep(1.0 / fps)
        finally:
            if preview_worker is not None:
                preview_worker.stop()
            capture_worker.stop()

    async def _recognize_detection_and_push(
        self,
        detection: DetectionSnapshot,
        state: LiveRecognitionState,
        send_json: Sender,
        *,
        connection_id: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            candidates = self.store.list_feature_records(self.recognition_service.embedding_identity())
            if not candidates:
                self.diagnostics.log_recognition(
                    {"box": getattr(detection.detection, "box", None), "matches": []},
                    candidates_count=0,
                    threshold=state.threshold,
                )
                await self._send(send_json, build_stream_error("No gallery feature records available"))
                return None
            started_at = self.perf_counter()
            context = getattr(self.recognition_service, "performance_context", None)
            manager = (
                context(
                    channel="websocket",
                    connection_id=str(connection_id or "unknown"),
                    frame_seq=detection.frame_seq,
                )
                if callable(context)
                else nullcontext()
            )
            with manager:
                live_result = self.recognition_service.recognize_detection(
                    detection.frame,
                    detection.detection,
                    candidates,
                    top_k=1,
                    threshold=state.threshold,
                )
            elapsed_ms = (self.perf_counter() - started_at) * 1000.0
            self._mark_recognition()
            self.diagnostics.log_recognition(
                live_result,
                candidates_count=len(candidates),
                threshold=state.threshold,
                elapsed_ms=elapsed_ms,
            )
            await self._push_match_if_present(live_result, state, send_json)
            return live_result
        except Exception as exc:
            await self._send(send_json, build_stream_error(str(exc) or "live face recognition service error"))
            return None

    async def _recognize_frame_and_push(
        self,
        frame: Any,
        state: LiveRecognitionState,
        send_json: Sender,
        *,
        connection_id: str | None = None,
        frame_seq: int | None = None,
    ) -> dict[str, Any] | None:
        try:
            candidates = self.store.list_feature_records(self.recognition_service.embedding_identity())
            if not candidates:
                self.diagnostics.log_recognition({"box": None, "matches": []}, candidates_count=0, threshold=state.threshold)
                await self._send(send_json, build_stream_error("No gallery feature records available"))
                return None
            started_at = self.perf_counter()
            context = getattr(self.recognition_service, "performance_context", None)
            manager = (
                context(
                    channel="websocket",
                    connection_id=str(connection_id or "unknown"),
                    frame_seq=frame_seq,
                )
                if callable(context)
                else nullcontext()
            )
            with manager:
                live_result = self._match_frame_for_live(frame, candidates, threshold=state.threshold)
            elapsed_ms = (self.perf_counter() - started_at) * 1000.0
            self._mark_recognition()
            self.diagnostics.log_recognition(
                live_result,
                candidates_count=len(candidates),
                threshold=state.threshold,
                elapsed_ms=elapsed_ms,
            )
            await self._push_match_if_present(live_result, state, send_json)
            return live_result
        except Exception as exc:
            await self._send(send_json, build_stream_error(str(exc) or "live face recognition service error"))
            return None

    # 婵″倹鐏夋禍楦垮姱鐠囧棗鍩嗙紒鎾寸亯娑擃厼鐡ㄩ崷銊ュ爱闁板秳姹夐崨姗堢礉閸掓瑧绮嶇憗?WebSocket 濞戝牊浼呴獮鑸靛腹闁胶绮扮€广垺鍩涚粩顖ょ幢婵″倹鐏夊▽鈩冩箒閸栧綊鍘ら敍灞藉灟娴犫偓娑斿牓鍏樻稉宥呬粵閵?
    async def _push_match_if_present(self, live_result: dict[str, Any], state: LiveRecognitionState, send_json: Sender) -> None:
        matches = live_result.get("matches") or []
        if not matches:
            return
        best = matches[0]
        await self._send(
            send_json,
            build_face_match_result(
                frame_time=format_live_time(self.clock()),
                video_stream_id=str(getattr(self.config, "live_video_stream_id", "") or self.config.live_stream_url),
                subject_id=best["subject_id"],
                name=best["name"],
                similarity=float(best["similarity"]),
                threshold=state.threshold,
            ),
        )

    def _match_frame_for_live(self, frame: Any, candidates: Any, threshold: float) -> dict[str, Any]:
        matcher = getattr(self.recognition_service, "match_frame_with_detection", None)
        if callable(matcher):
            return matcher(frame, candidates, top_k=1, threshold=threshold)
        matches = self.recognition_service.match_frame(frame, candidates, top_k=1, threshold=threshold)
        return {"box": None, "matches": matches}

    def _create_capture_worker(
        self,
        stream_url: str,
        frame_buffer: LatestFrameBuffer,
        worker_errors: queue.Queue[WorkerError],
        *,
        connection_id: str | None = None,
        websocket_accepted_at: float | None = None,
    ) -> Any:
        return self.capture_worker_factory(
            stream_url=stream_url,
            frame_buffer=frame_buffer,
            frame_reader_factory=self.frame_reader_factory,
            error_callback=self._worker_error_callback(worker_errors, source="capture", close_connection=True),
            metrics=self.metrics,
            reconnect_enabled=getattr(self.config, "live_capture_reconnect_enabled", True),
            reconnect_interval=getattr(self.config, "live_capture_reconnect_interval", 1.0),
            max_consecutive_failures=getattr(self.config, "live_capture_max_consecutive_failures", 30),
            diagnostics=self.diagnostics,
            perf_counter=self.perf_counter,
            connection_id=connection_id,
            websocket_accepted_at=websocket_accepted_at,
        )

    def _create_detection_worker(
        self,
        frame_buffer: LatestFrameBuffer,
        detection_cache: DetectionCache,
        state: LiveRecognitionState,
        worker_errors: queue.Queue[WorkerError],
        *,
        connection_id: str | None = None,
        websocket_accepted_at: float | None = None,
    ) -> Any:
        return self.detection_worker_factory(
            frame_buffer=frame_buffer,
            detection_cache=detection_cache,
            recognition_service=self.recognition_service,
            state=state,
            fps=getattr(self.config, "live_detection_fps", 12.0),
            error_callback=self._worker_error_callback(worker_errors, source="detection", close_connection=False),
            metrics=self.metrics,
            diagnostics=self.diagnostics,
            perf_counter=self.perf_counter,
            connection_id=connection_id,
            websocket_accepted_at=websocket_accepted_at,
        )

    def _create_preview_worker(
        self,
        frame_buffer: LatestFrameBuffer,
        result_cache: RecognitionResultCache,
        state: LiveRecognitionState,
        *,
        detection_cache: DetectionCache | None,
    ) -> Any | None:
        if not getattr(self.config, "live_preview_enabled", False):
            return None
        renderer = self.preview_renderer
        if renderer is None:
            renderer = LivePreviewRenderer(
                enabled=True,
                window_name=getattr(self.config, "live_preview_window_name", "Live Face Recognition"),
                width=getattr(self.config, "live_preview_width", None),
                height=getattr(self.config, "live_preview_height", None),
                max_width=getattr(self.config, "live_preview_max_width", None),
                max_height=getattr(self.config, "live_preview_max_height", None),
            )
        return LivePreviewWorker(
            renderer=renderer,
            frame_buffer=frame_buffer,
            result_cache=result_cache,
            detection_cache=detection_cache,
            metrics=self.metrics,
            device_label=self.device_label,
            state=state,
            fps=getattr(self.config, "live_preview_fps", 20.0),
            result_ttl_seconds=getattr(self.config, "live_preview_result_ttl", 1.5),
            detection_ttl_seconds=getattr(self.config, "live_detection_result_ttl", 0.8),
        )


    def _log_runner_stage(
        self,
        stage: str,
        *,
        connection_id: str | None,
        websocket_accepted_at: float | None,
    ) -> None:
        logger = getattr(self.diagnostics, "log_runner_stage", None)
        if not callable(logger):
            return
        elapsed_ms = None
        if websocket_accepted_at is not None:
            elapsed_ms = (self.perf_counter() - websocket_accepted_at) * 1000.0
        logger(connection_id=str(connection_id or "unknown"), stage=stage, elapsed_ms=elapsed_ms)

    def _supports_split_pipeline(self) -> bool:
        return callable(getattr(self.recognition_service, "detect_frame", None)) and callable(
            getattr(self.recognition_service, "recognize_detection", None)
        )

    def _mark_recognition(self) -> None:
        marker = getattr(self.metrics, "mark_recognition", None)
        if callable(marker):
            marker()

    def _describe_runtime(self) -> dict[str, Any]:
        describe = getattr(self.recognition_service, "describe", None)
        if not callable(describe):
            return {}
        try:
            runtime_info = describe()
        except Exception:
            return {}
        return runtime_info if isinstance(runtime_info, dict) else {}

    @staticmethod
    def _worker_error_callback(
        worker_errors: queue.Queue[WorkerError],
        *,
        source: str,
        close_connection: bool,
    ) -> Callable[[str], None]:
        def callback(message: str) -> None:
            worker_errors.put(
                WorkerError(
                    source=source,
                    message=str(message or STREAM_OFFLINE_MESSAGE),
                    close_connection=close_connection,
                )
            )

        return callback

    @staticmethod
    def _normalize_worker_error(error: WorkerError | str) -> WorkerError:
        if isinstance(error, WorkerError):
            return error
        return WorkerError(source="unknown", message=str(error or STREAM_OFFLINE_MESSAGE), close_connection=False)

    async def _drain_worker_errors(
        self,
        worker_errors: queue.Queue[WorkerError | str],
        send_json: Sender,
        *,
        state: LiveRecognitionState,
        request_close: CloseRequester | None,
    ) -> int:
        sent = 0
        while True:
            try:
                worker_error = self._normalize_worker_error(worker_errors.get_nowait())
            except queue.Empty:
                return sent
            try:
                await self._send(send_json, build_stream_error(worker_error.message or STREAM_OFFLINE_MESSAGE))
            except RuntimeError:
                state.close()
                return sent
            sent += 1
            if worker_error.close_connection:
                await self._request_close(
                    state,
                    request_close,
                    code=STREAM_ERROR_CLOSE_CODE,
                    reason=STREAM_ERROR_CLOSE_REASON,
                )
                return sent

    async def _request_close(
        self,
        state: LiveRecognitionState,
        request_close: CloseRequester | None,
        *,
        code: int,
        reason: str,
    ) -> None:
        if request_close is None:
            state.request_close(code=code, reason=reason)
            return
        result = request_close(code=code, reason=reason)
        if inspect.isawaitable(result):
            await result
        if not state.closed:
            state.request_close(code=code, reason=reason)
    async def _send(self, send_json: Sender, message: dict) -> None:
        result = send_json(message)
        if inspect.isawaitable(result):
            await result




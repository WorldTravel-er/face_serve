from __future__ import annotations
"""
    实时人脸识别系统的预览显示模块 
    把摄像头帧复制出来,画状态信息, 画 FPS,画人脸框,画识别结果,控制 OpenCV 窗口显示
    通过线程持续刷新画面
"""
import inspect
import threading
from typing import Any

import cv2
import numpy as np

# 输入原始视频帧，输出标注好的预览帧
def build_preview_frame(
    frame: np.ndarray,
    *,
    result: dict[str, Any] | None,
    detection: Any | None = None,
    metrics: dict[str, float] | None = None,
    device_label: str | None = None,
    threshold: float,
    status: str,
) -> np.ndarray:
    preview = frame.copy()
    # 绘制状态
    _draw_status(preview, status=status, threshold=threshold, device_label=device_label)
    # 绘制FPS
    has_metrics = bool(metrics)
    if metrics:
        _draw_metrics(preview, metrics)
    #
    overlay_result = result
    # 有人脸检测，没进行识别，只显示检测框
    if overlay_result is None and detection is not None:
        box = _extract_detection_box(detection)
        if box is not None:
            overlay_result = {"box": box, "matches": []}

    # 没有检测结果，显示No matched face
    if overlay_result is None:
        if status != "pause":
            _draw_text(preview, "No matched face", 10, 88 if has_metrics else 58, color=(0, 220, 255))
        return preview


    box = overlay_result.get("box")
    matches = overlay_result.get("matches") or []
    # 绘制人脸框
    if box is not None:
        x, y, w, h = [int(value) for value in box]
        x1, y1 = max(0, x), max(0, y)
        x2 = min(preview.shape[1] - 1, x + w)
        y2 = min(preview.shape[0] - 1, y + h)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label_y = y1 - 8 if y1 > (58 if has_metrics else 28) else y2 + 24
    else:
        x1, label_y = 10, 114 if has_metrics else 84
    # 显示识别结果
    if matches:
        best = matches[0]
        label = (
            f"{best.get('subject_id', '')} {best.get('name', '')} "
            f"sim:{float(best.get('similarity', 0.0)):.3f}"
        )
    else:
        label = "Face detected, no matched subject"
    _draw_text(preview, label, x1, label_y, color=(255, 255, 255), background=(0, 150, 0))
    return preview


# 按照最大宽度/最大高度限制缩放图片，同时保持原始宽高比例不变。
def resize_keep_aspect(
    frame: np.ndarray,
    *,
    max_width: int | None = None,
    max_height: int | None = None,
    backend: Any = cv2,
) -> np.ndarray:
    if max_width is None and max_height is None:
        return frame
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame

    width_scale = float(max_width) / width if max_width else float("inf")
    height_scale = float(max_height) / height if max_height else float("inf")
    scale = min(width_scale, height_scale)
    if not np.isfinite(scale) or scale <= 0:
        return frame
    if scale >= 1.0:
        return frame

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = getattr(backend, "INTER_AREA", cv2.INTER_AREA)
    return backend.resize(frame, (new_width, new_height), interpolation=interpolation)

# 接收一帧视频 → 加上检测/识别信息 → 调整尺寸 → 显示到 OpenCV 窗口 → 处理退出。
class LivePreviewRenderer:
    def __init__(
        self,
        *,
        enabled: bool, # 控制是否显示窗口。
        window_name: str = "Live Face Recognition",
        width: int | None = None,
        height: int | None = None,
        max_width: int | None = None,
        max_height: int | None = None,
        backend: Any = cv2,
    ) -> None:
        self.enabled = bool(enabled)
        self.window_name = window_name
        self.width = width
        self.height = height
        self.max_width = max_width
        self.max_height = max_height
        self.backend = backend
        self._closed = False

    def show(
        self,
        frame: np.ndarray,
        *,
        result: dict[str, Any] | None,
        threshold: float,
        status: str,
        detection: Any | None = None,
        metrics: dict[str, float] | None = None,
        device_label: str | None = None,
    ) -> None:
        if not self.enabled or self._closed:
            return
        preview = build_preview_frame(
            frame,
            result=result,
            detection=detection,
            metrics=metrics,
            device_label=device_label,
            threshold=threshold,
            status=status,
        )
        if self.width and self.height:
            preview = cv2.resize(preview, (int(self.width), int(self.height)))
        else:
            preview = resize_keep_aspect(
                preview,
                max_width=self.max_width,
                max_height=self.max_height,
                backend=self.backend,
            )
        self.backend.imshow(self.window_name, preview)
        key = self.backend.waitKey(1)
        if key & 0xFF == ord("q"):
            self.close()

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        self._closed = True
        try:
            self.backend.destroyWindow(self.window_name)
        except Exception:
            pass

# 从最新视频帧缓存中取画面 → 获取最新检测/识别结果 → 获取性能指标 → 调用 LivePreviewRenderer 显示。
class LivePreviewWorker:
    def __init__(
        self,
        *,
        renderer: Any,
        frame_buffer: Any, # 视频帧缓存
        result_cache: Any, # 识别结果缓存
        state: Any, # 系统状态
        detection_cache: Any | None = None, # 检测缓存
        metrics: Any | None = None, # 性能指标
        device_label: str | None = None,# det:GPU rec:GPU
        fps: float = 20.0,
        result_ttl_seconds: float = 1, # 缓存识别结果有效时间
        detection_ttl_seconds: float = 0.5, # 缓存检测结果有效时间
    ) -> None:
        self.renderer = renderer
        self.frame_buffer = frame_buffer
        self.result_cache = result_cache
        self.detection_cache = detection_cache
        self.metrics = metrics
        self.device_label = device_label
        self.state = state
        self.fps = max(float(fps or 20.0), 0.1)
        self.result_ttl_seconds = max(float(result_ttl_seconds), 0.0)
        self.detection_ttl_seconds = max(float(detection_ttl_seconds), 0.0)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # 检查 Renderer 支持哪些参数
        self._renderer_accepts_detection = _accepts_kwarg(renderer, "detection")
        self._renderer_accepts_metrics = _accepts_kwarg(renderer, "metrics")
        self._renderer_accepts_device_label = _accepts_kwarg(renderer, "device_label")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="live-preview-worker", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
        self.renderer.close()

    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def _run(self) -> None:
        period = 1.0 / self.fps
        try:
            while not self._stop_event.is_set():
                latest = self.frame_buffer.get_latest()
                if latest is not None:
                    result = self.result_cache.get_valid(ttl_seconds=self.result_ttl_seconds)
                    detection = None
                    if self.detection_cache is not None:
                        detection = self.detection_cache.get_valid(ttl_seconds=self.detection_ttl_seconds)
                    metrics_snapshot = self._metrics_snapshot()
                    self._show(latest.frame, result=result, detection=detection, metrics=metrics_snapshot)
                self._stop_event.wait(period)
        finally:
            self.renderer.close()

    def _show(
        self,
        frame: np.ndarray,
        *,
        result: dict[str, Any] | None,
        detection: Any | None,
        metrics: dict[str, float] | None,
    ) -> None:
        kwargs = {
            "result": result,
            "threshold": self.state.threshold,
            "status": self.state.status,
        }
        if self._renderer_accepts_detection:
            kwargs["detection"] = detection
        if self._renderer_accepts_metrics:
            kwargs["metrics"] = metrics
        if self._renderer_accepts_device_label:
            kwargs["device_label"] = self.device_label
        self.renderer.show(frame, **kwargs)

    def _metrics_snapshot(self) -> dict[str, float] | None:
        snapshot = getattr(self.metrics, "snapshot", None)
        if callable(snapshot):
            return snapshot()
        return None


def _extract_detection_box(detection: Any) -> tuple[int, int, int, int] | None:
    if detection is None:
        return None
    if isinstance(detection, dict):
        box = detection.get("box")
        if box is None and detection.get("detection") is not None:
            box = getattr(detection["detection"], "box", None)
    else:
        box = getattr(detection, "box", None)
        if box is None and getattr(detection, "detection", None) is not None:
            box = getattr(detection.detection, "box", None)
    if box is None:
        return None
    return tuple(int(value) for value in box)


def _accepts_kwarg(renderer: Any, name: str) -> bool:
    try:
        signature = inspect.signature(renderer.show)
    except (TypeError, ValueError, AttributeError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters


def _draw_status(frame: np.ndarray, *, status: str, threshold: float, device_label: str | None = None) -> None:
    if status == "pause":
        text = f"PAUSED  threshold:{threshold:.3f}"
        color = (0, 180, 255)
    else:
        text = f"LIVE  threshold:{threshold:.3f}"
        color = (0, 220, 0)
    if device_label:
        text = f"{text}  device {device_label}"
    _draw_text(frame, text, 10, 28, color=(255, 255, 255), background=color)


def _draw_metrics(frame: np.ndarray, metrics: dict[str, float]) -> None:
    text = (
        f"FPS live:{float(metrics.get('stream_fps', 0.0)):.1f} "
        f"detect:{float(metrics.get('detection_fps', 0.0)):.1f} "
        f"recog:{float(metrics.get('recognition_fps', 0.0)):.1f}"
    )
    _draw_text(frame, text, 10, 58, color=(255, 255, 255), background=(60, 60, 60))


def _draw_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    color: tuple[int, int, int],
    background: tuple[int, int, int] | None = None,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    text = str(text)
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 5
    left = max(0, int(x))
    baseline_y = min(frame.shape[0] - 1, max(text_h + pad, int(y)))
    if background is not None:
        top = max(0, baseline_y - text_h - baseline - pad)
        bottom = min(frame.shape[0] - 1, baseline_y + pad)
        right = min(frame.shape[1] - 1, left + text_w + pad * 2)
        cv2.rectangle(frame, (left, top), (right, bottom), background, -1)
    cv2.putText(frame, text, (left + pad, baseline_y), font, scale, color, thickness, cv2.LINE_AA)
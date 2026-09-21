# RK3588 NPU 人脸服务迁移 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 REST、WebSocket 和视频分析响应 schema 的前提下，为服务增加可验证、可回退的 RK3588 RKNN/NPU 推理运行时。

**Architecture:** 保留现有 ONNX 适配器，并通过 `runtime=onnx|rknn` 显式选择推理实现。RKNN 模型会话将负责 `.rknn` 加载、NPU core 选择、线程串行化和输入布局适配；检测/对齐/特征引擎仅替换神经网络调用，OpenCV 图像处理、YOLO 输出解码/NMS、仿射变换及向量检索仍在 CPU 完成。模型构建在 x86_64 Linux 上用 RKNN-Toolkit2 2.3.2 执行，部署侧以 Lite2 2.3.2 调用盒子已经验证的 `librknnrt.so 2.3.2`。

**Tech Stack:** Python 3.11, FastAPI, NumPy, OpenCV, SQLite, ONNX Runtime（保留）, RKNN-Toolkit2 2.3.2（构建机）, RKNN-Toolkit-Lite2 2.3.2（RK3588）, Docker Compose。

**Spec:** `docs/superpowers/specs/2026-08-20-rk3588-npu-deployment-design.md`

## Global Constraints

* 目标设备是 RK3588 / Ubuntu 22.04.5 / aarch64 / Linux 6.1.141 / RKNPU driver 0.9.8 / `rknnrt 2.3.2`；目标平台固定为 `rk3588`。
* NPU 生产服务只能使用 `rknn-toolkit-lite2 2.3.2` 与设备 `librknnrt.so`；完整 Toolkit2 只安装在 x86_64 Linux 转换环境。
* API 路径、请求字段和成功/错误响应 schema 均不能改变；默认 `runtime=onnx`，不得改变既有部署行为。
* `.rknn` 模型固定 batch=1；检测、对齐和识别模型输入/输出布局由构建 manifest 决定，不能在 Python 中假定 NCHW 或 NHWC。
* RKNN context 不能被并发调用；一个模型会话一次只能执行一条 `inference()`。
* `runtime=rknn` 任何必需模型加载或自检失败时，进程以非零状态退出；不得静默改用 CPU。
* 不提交校准人脸图片、验收人脸图片、厂商 Lite2 wheel、`.rknn` 二进制或任何包含生物信息的数据库备份；只提交脚本、测试、manifest schema 和文档。
* 历史 ONNX 特征和新 INT8-RKNN 特征不能混合匹配；切换前必须备份数据库并基于原始注册图重建特征库。
* 容器只映射 `/dev/dri/renderD129`，以 `render` 组权限运行；不能使用 `privileged: true`，不能在容器中启动第二个 `rknn_server`。

---

## 文件结构与职责

| 路径 | 职责 |
| --- | --- |
| `tests/conftest.py` | 为所有单元测试提供仓库根目录导入路径和临时数据目录 fixture。 |
| `tests/test_rknn_runtime.py` | 用 fake Lite2 实例验证 model load、core mask、输入 layout、互斥锁和释放。 |
| `face_demo/rknn_runtime.py` | 设备侧 RKNN Lite2 的单模型线程安全会话、输入规格和环境诊断。 |
| `tests/test_yolo_face_rknn.py` | 验证 RKNN YOLO 预处理、输出到现有解码器、最大脸选择与描述信息。 |
| `face_demo/yolo_face_rknn.py` | 复用现有 YOLO 几何和后处理的 RKNN 检测器。 |
| `tests/test_cvlface_rknn.py` | 验证 RKNN 对齐/IR50 特征路径、CPU 对齐回退和无效特征处理。 |
| `face_demo/face_engine/cvlface_rknn.py` | 与 `CVLFaceOnnxEngine` 兼容的 RKNN 人脸特征引擎。 |
| `tests/test_runtime_configuration.py` | 验证 CLI、`ApiConfig`、工厂选择、ready 状态和启动期错误。 |
| `face_api/core/config.py` | 增加 `runtime`、三个 RKNN 模型路径、core mask 和对齐回退配置。 |
| `face_api/main.py` | 增加 RKNN CLI/environment 选项并构造完整 `ApiConfig`。 |
| `face_api/core/recognition.py` | 根据 runtime 实例化 ONNX 或 RKNN detector/engine，输出统一 runtime 描述。 |
| `face_api/rest/routers/health.py` | 将模型加载状态纳入 `/readyz`，且报告 runtime 诊断。 |
| `tests/test_subject_store_embedding_metadata.py` | 验证 SQLite schema migration、特征运行时元数据和不兼容特征拒绝。 |
| `face_api/core/subject_store.py` | 保存 `embedding_runtime`、`embedding_model_sha256` 并按兼容性筛选 gallery。 |
| `scripts/rebuild_gallery_features.py` | 从备份数据库中的原始注册图片，用当前 runtime 重建特征库。 |
| `tests/test_rknn_video_tracker.py` | 验证 NPU 全量检测输出能够转换为稳定 track ID，且不导入 PyTorch detector。 |
| `face_api/video/bytetrack.py` | 仅依赖 NumPy 与已存在 `lap` 的两阶段 ByteTrack 数据关联，不导入 PyTorch 或 Ultralytics。 |
| `face_api/video/rknn_tracker.py` | 消费 `YoloFaceRknnDetector.detect_all()` 的轻量 ByteTrack 适配器。 |
| `face_api/app.py` | 视频任务按 runtime 选择 NPU tracker 或既有 ONNX/PyTorch tracker。 |
| `scripts/convert_rknn_models.py` | 以 Toolkit2 2.3.2 从现有 ONNX 生成 FP/INT8 RKNN、记录元数据和 digest。 |
| `scripts/validate_rknn_parity.py` | 生成检测/特征/身份一致性 JSON 报告并以门槛决定退出码。 |
| `scripts/benchmark_rknn_service.py` | 统计模型推理和端到端延迟的 p50/p95/FPS。 |
| `models/rknn/manifest.schema.json` | 描述不可提交模型产物所需的版本、输入输出、量化和 hash 字段。 |
| `requirements-rknn.txt` | 仅说明设备 Python 运行时所需 Lite2 wheel 与版本；wheel 路径由部署方提供。 |
| `Dockerfile.rknn` | ARM64 NPU 服务镜像，安装 Lite2 wheel，不包含 CUDA、PyTorch 或完整 Toolkit2。 |
| `docker-compose.rknn.yml` | 安全映射 NPU render 节点和模型/数据卷的生产服务定义。 |
| `docs/rk3588-deployment.md` | 开发机转换、宿主机验收、容器启动、回滚和排障命令。 |
| `tests/test_rknn_device_check.py` | 验证部署前设备检查成功和失败时的退出码。 |

## Task 1: 建立可重复的单元测试入口与运行时值对象

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_rknn_runtime.py`
- Create: `face_demo/rknn_runtime.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `RknnInputSpec(name: str, layout: Literal["nchw", "nhwc"], shape: tuple[int, int, int, int], dtype: np.dtype)`.
- Produces: `RknnModelSession(model_path: Path, input_spec: RknnInputSpec, core_mask: str = "auto", lite_factory: Callable[[], Any] | None = None)` with `infer(array: np.ndarray) -> list[np.ndarray]`, `describe() -> dict[str, Any]`, and `close() -> None`.
- Produces: `load_input_spec(manifest_path: Path, model_path: Path) -> RknnInputSpec`, which selects the manifest entry whose resolved `rknn_path` equals `model_path`.
- Consumes: a Lite2-compatible object exposing `load_rknn()`, `init_runtime()`, `inference()` and `release()`.

- [ ] **Step 1: Add pytest as a development dependency and create a test fixture that does not load real models**

Modify the root `pyproject.toml` by adding a dev dependency group and create `tests/conftest.py`:

```toml
[dependency-groups]
dev = ["pytest==8.3.5"]
```

```python
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.rknn"
    path.write_bytes(b"rknn-test-model")
    return path
```

- [ ] **Step 2: Write failing unit tests for model load, RK3588 core mapping, input conversion, lock usage and release**

Create a deterministic fake Lite2 class in `tests/test_rknn_runtime.py` and assert the public contract:

```python
import numpy as np
import pytest

from face_demo.rknn_runtime import RknnInputSpec, RknnModelSession


class FakeLite:
    NPU_CORE_0 = 1
    NPU_CORE_1 = 2
    NPU_CORE_2 = 4
    NPU_CORE_0_1_2 = 7

    def __init__(self):
        self.loaded = None
        self.init_kwargs = None
        self.inputs = []
        self.released = False

    def load_rknn(self, path):
        self.loaded = path
        return 0

    def init_runtime(self, **kwargs):
        self.init_kwargs = kwargs
        return 0

    def inference(self, *, inputs, data_format):
        self.inputs.append((inputs, data_format))
        return [np.asarray([[0.25, 0.75]], dtype=np.float16)]

    def release(self):
        self.released = True


def test_load_initializes_all_npu_cores_and_converts_output_to_float32(model_file):
    fake = FakeLite()
    spec = RknnInputSpec("input", "nchw", (1, 3, 112, 112), np.dtype("uint8"))
    session = RknnModelSession(model_file, spec, core_mask="0_1_2", lite_factory=lambda: fake)
    assert fake.loaded == str(model_file)
    assert fake.init_kwargs == {"core_mask": FakeLite.NPU_CORE_0_1_2}
    result = session.infer(np.zeros((1, 3, 112, 112), dtype=np.uint8))
    assert result[0].dtype == np.float32
    assert fake.inputs[0][1] == ["nchw"]


def test_rejects_unknown_core_mask_before_loading(model_file):
    with pytest.raises(ValueError, match="core_mask"):
        RknnModelSession(model_file, RknnInputSpec("input", "nchw", (1, 3, 112, 112), np.dtype("uint8")), core_mask="3", lite_factory=FakeLite)


def test_close_releases_runtime_once(model_file):
    fake = FakeLite()
    session = RknnModelSession(model_file, RknnInputSpec("input", "nchw", (1, 3, 112, 112), np.dtype("uint8")), lite_factory=lambda: fake)
    session.close()
    session.close()
    assert fake.released is True
```

- [ ] **Step 3: Run the test to confirm it fails because the module does not exist**

Run: `uv sync --group dev && uv run pytest tests/test_rknn_runtime.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'face_demo.rknn_runtime'`.

- [ ] **Step 4: Implement the dependency-gated, thread-safe Lite2 session**

Create `face_demo/rknn_runtime.py`. The implementation must import `RKNNLite` only when no `lite_factory` is supplied, keep `self._lock = threading.RLock()`, reject a missing `.rknn` file before initializing Lite2, map core masks exactly, and release exactly once:

```python
CORE_MASK_NAMES = {"auto", "0", "1", "2", "0_1_2"}


def _create_lite() -> Any:
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError(
            "RKNN runtime requires rknn-toolkit-lite2==2.3.2 on the RK3588 device"
        ) from exc
    return RKNNLite()


def _core_mask(lite: Any, requested: str) -> int | None:
    masks = {
        "0": lite.NPU_CORE_0,
        "1": lite.NPU_CORE_1,
        "2": lite.NPU_CORE_2,
        "0_1_2": lite.NPU_CORE_0_1_2,
    }
    if requested not in CORE_MASK_NAMES:
        raise ValueError(f"Unknown RKNN core_mask: {requested!r}")
    return masks.get(requested)


def infer(self, array: np.ndarray) -> list[np.ndarray]:
    if self._closed:
        raise RuntimeError(f"RKNN model session is closed: {self.model_path}")
    value = np.ascontiguousarray(array)
    if tuple(value.shape) != self.input_spec.shape:
        raise ValueError(f"RKNN input shape {tuple(value.shape)} does not match {self.input_spec.shape}")
    if value.dtype != self.input_spec.dtype:
        raise ValueError(f"RKNN input dtype {value.dtype} does not match {self.input_spec.dtype}")
    with self._lock:
        outputs = self._lite.inference(inputs=[value], data_format=[self.input_spec.layout])
    return [np.asarray(output, dtype=np.float32) for output in outputs]
```

Call `init_runtime()` without keyword arguments for `auto`; call it with `core_mask=<constant>` for every explicit mask. If `load_rknn()` or `init_runtime()` returns nonzero, call `release()` and raise `RuntimeError` containing the model path and return code.

Implement `load_input_spec()` as strict JSON parsing: it reads `manifest["models"]`, selects exactly one entry matching the resolved model path, requires exactly one input, converts its `shape` list to a four-integer tuple and `dtype` string through `np.dtype()`, and raises `ValueError` if there is no match, multiple matches, a non-batch-one shape or a layout other than `nchw`/`nhwc`. Add a unit test that writes a one-entry temporary manifest and asserts the returned spec equals `RknnInputSpec("images", "nchw", (1, 3, 640, 640), np.dtype("uint8"))`.

- [ ] **Step 5: Run the runtime test suite**

Run: `uv run pytest tests/test_rknn_runtime.py -q`

Expected: all tests pass without an actual RKNN wheel or NPU device.

- [ ] **Step 6: Commit the independently tested runtime session**

```bash
git add pyproject.toml uv.lock tests/conftest.py tests/test_rknn_runtime.py face_demo/rknn_runtime.py
git commit -m "feat: add thread-safe RKNN Lite runtime session"
```

## Task 2: 实现 RKNN YOLO 检测器且复用现有后处理

**Files:**
- Create: `tests/test_yolo_face_rknn.py`
- Create: `face_demo/yolo_face_rknn.py`
- Modify: `face_demo/yolo_face_onnx.py`

**Interfaces:**
- Consumes: `RknnModelSession.infer()` from Task 1 and `LetterboxMeta`, `decode_yolo_output()`, `_choose_nearest()` from `face_demo.yolo_face_onnx`/`face_demo.yolo_face`.
- Produces: `YoloFaceRknnDetector(model_path, manifest_path, conf=0.25, iou=0.45, imgsz=640, face_class=None, session=None, core_mask="auto")` with `detect_all(image) -> list[FaceDetection]`, `detect_largest(image) -> FaceDetection | None`, `track_nearest(frame) -> YoloTrackResult | None`, `describe() -> dict[str, Any]`, `close() -> None`.
- Produces: `decode_yolo_outputs(outputs: Sequence[np.ndarray], meta: LetterboxMeta, conf_threshold: float, iou_threshold: float, face_class: int | None) -> list[FaceDetection]` for multiple RKNN outputs.

- [ ] **Step 1: Write failing detector tests using a fake session**

Create `tests/test_yolo_face_rknn.py` with a fake session recording its input and returning the present ONNX-style single output:

```python
import numpy as np

from face_demo.rknn_runtime import RknnInputSpec
from face_demo.yolo_face_rknn import YoloFaceRknnDetector


class FakeSession:
    input_spec = RknnInputSpec("images", "nchw", (1, 3, 640, 640), np.dtype("uint8"))

    def __init__(self, output):
        self.output = output
        self.calls = []

    def infer(self, array):
        self.calls.append((array, self.input_spec.layout))
        return [self.output]

    def describe(self):
        return {"runtime": "rknn", "model_path": "test.rknn", "core_mask": "auto"}

    def close(self):
        return None


def test_detect_all_uses_rgb_uint8_nchw_and_existing_box_geometry():
    # cx=320, cy=320, w=160, h=200, score=0.95
    output = np.array([[[320.0], [320.0], [160.0], [200.0], [0.95]]], dtype=np.float32)
    session = FakeSession(output)
    detector = YoloFaceRknnDetector("model.rknn", imgsz=640, session=session)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect_all(image)
    assert len(detections) == 1
    assert detections[0].box == (240, 220, 160, 200)
    tensor, layout = session.calls[0]
    assert layout == "nchw"
    assert tensor.dtype == np.uint8
    assert tensor.shape == (1, 3, 640, 640)


def test_describe_identifies_rknn_runtime():
    detector = YoloFaceRknnDetector("model.rknn", session=FakeSession(np.zeros((1, 5, 0), dtype=np.float32)))
    assert detector.describe()["detector"] == "YOLO_RKNNLite"
    assert detector.describe()["runtime"] == "rknn"
```

- [ ] **Step 2: Run the detector test to confirm it fails**

Run: `uv run pytest tests/test_yolo_face_rknn.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'face_demo.yolo_face_rknn'`.

- [ ] **Step 3: Extract only the shared preprocessing needed for an unsigned-byte RKNN input**

In `face_demo/yolo_face_onnx.py`, split image preparation into a color/geometry helper without changing `preprocess_image()` behavior:

```python
def letterbox_rgb_uint8(image: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, LetterboxMeta]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"YOLO input must be a BGR image with shape HxWx3, got {image.shape}")
    height, width = image.shape[:2]
    ratio = min(imgsz / height, imgsz / width)
    resized_w, resized_h = int(round(width * ratio)), int(round(height * ratio))
    pad_w, pad_h = float(imgsz - resized_w) / 2.0, float(imgsz - resized_h) / 2.0
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    left, top = int(round(pad_w - 0.1)), int(round(pad_h - 0.1))
    padded = cv2.copyMakeBorder(resized, top, int(round(pad_h + 0.1)), left, int(round(pad_w + 0.1)), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    meta = LetterboxMeta(ratio, (float(left), float(top)), (height, width), (imgsz, imgsz))
    return np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)), meta


def preprocess_image(image: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, LetterboxMeta]:
    rgb, meta = letterbox_rgb_uint8(image, imgsz)
    tensor = np.transpose(rgb, (2, 0, 1))[None, :, :, :].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), meta
```

Add assertions to an existing or new test that `preprocess_image()` still returns `float32`, NCHW and values in `[0, 1]`.

- [ ] **Step 4: Implement `YoloFaceRknnDetector` with one inference path and shared decode**

Create `face_demo/yolo_face_rknn.py` with this essential flow:

```python
def _input_tensor(self, image: np.ndarray) -> tuple[np.ndarray, LetterboxMeta]:
    rgb, meta = letterbox_rgb_uint8(image, self.imgsz)
    if self.session.input_spec.layout == "nchw":
        return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, :, :, :]), meta
    return np.ascontiguousarray(rgb[None, :, :, :]), meta


def detect_all(self, image: np.ndarray) -> list[FaceDetection]:
    tensor, meta = self._input_tensor(image)
    outputs = self.session.infer(tensor)
    return decode_yolo_outputs(outputs, meta, self.conf, self.iou, self.face_class)


def detect_largest(self, image: np.ndarray) -> FaceDetection | None:
    return _choose_nearest(self.detect_all(image), image.shape[1], image.shape[0], None)
```

When no test session is injected, construct it with `RknnModelSession(Path(model_path), load_input_spec(Path(manifest_path), Path(model_path)), core_mask=core_mask)`. The detector accepts a fake session in unit tests, but its production constructor must require the manifest path so layout is not inferred from the model filename.

`decode_yolo_outputs()` must retain the current single-output behavior by calling `decode_yolo_output(outputs[0], meta, conf_threshold, iou_threshold, face_class)`. It must otherwise raise a clear `ValueError` that includes each output shape; multi-head RKNN output support may only be added after `scripts/convert_rknn_models.py` writes the actual output contract to the manifest.

- [ ] **Step 5: Run the detector and pre-existing ONNX detector tests**

Run: `uv run pytest tests/test_yolo_face_rknn.py -q`

Run: `uv run pytest -q -k 'yolo or detector'`

Expected: new RKNN tests pass; any discovered existing test failure is fixed only if it is caused by the shared preprocessing extraction.

- [ ] **Step 6: Commit the NPU detector adapter**

```bash
git add face_demo/yolo_face_onnx.py face_demo/yolo_face_rknn.py tests/test_yolo_face_rknn.py
git commit -m "feat: add RKNN face detector adapter"
```

## Task 3: 实现 RKNN 对齐与 AdaFace 特征引擎，并保留明确的 CPU 回退

**Files:**
- Create: `tests/test_cvlface_rknn.py`
- Create: `face_demo/face_engine/cvlface_rknn.py`
- Modify: `face_demo/face_engine/__init__.py`

**Interfaces:**
- Consumes: `FaceEngine`, `FaceFeature`, `FaceDetection`, `align_image_and_keypoints()`, `reference_landmark()` and `l2_normalize()` from the existing face engine; `RknnModelSession` from Task 1.
- Produces: `CVLFaceRknnEngine(recognition_model_path, aligner_model_path, manifest_path: Path | None = None, *, aligner_runtime: Literal["rknn", "onnx-cpu"], aligner_onnx_model_path: Path | None, align_score_threshold=0.0, core_mask="auto", recognition_session=None, aligner_session=None)` with the existing `FaceEngine` feature methods and `close()`.
- Produces: `RknnFaceRecognitionSession(session).extract(aligned_rgb_uint8: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write failing tests for a 512-dimensional normalized RKNN feature and alignment fallback selection**

Create `tests/test_cvlface_rknn.py` with fake RKNN sessions. Test the engine at the public `feature_from_detection_with_info()` boundary:

```python
import numpy as np

from face_demo.face_engine import FaceDetection
from face_demo.face_engine.cvlface_rknn import CVLFaceRknnEngine


class FakeSession:
    input_spec = RknnInputSpec("image", "nhwc", (1, 112, 112, 3), np.dtype("uint8"))

    def __init__(self, outputs):
        self.outputs = outputs

    def infer(self, value):
        return [np.asarray(item, dtype=np.float32) for item in self.outputs]
    def describe(self):
        return {"runtime": "rknn"}
    def close(self):
        return None


def test_feature_engine_returns_l2_normalized_embedding_for_detection():
    keypoints = np.array([[[0.3, 0.35], [0.7, 0.35], [0.5, 0.55], [0.35, 0.75], [0.65, 0.75]]])
    aligner = FakeSession([np.zeros((1, 3, 112, 112)), keypoints, keypoints, np.array([[1.0]])])
    recognizer = FakeSession([np.ones((1, 512), dtype=np.float32)])
    engine = CVLFaceRknnEngine("recognition.rknn", "aligner.rknn", recognition_session=recognizer, aligner_session=aligner)
    info = engine.feature_from_detection_with_info(np.zeros((200, 200, 3), dtype=np.uint8), FaceDetection((20, 20, 100, 100), 0.9, None))
    assert info.feature.shape == (512,)
    assert np.isclose(np.linalg.norm(info.feature), 1.0)
    assert info.feature_valid is True
```

- [ ] **Step 2: Run the engine tests and confirm they fail**

Run: `uv run pytest tests/test_cvlface_rknn.py -q`

Expected: collection fails because `cvlface_rknn` does not exist.

- [ ] **Step 3: Implement RGB uint8 conversion and RKNN recognition extraction**

Create a helper that preserves existing color semantics while shifting normalization into RKNN conversion:

```python
def bgr_to_rgb_uint8(image: np.ndarray, input_spec: RknnInputSpec) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if input_spec.layout == "nhwc":
        _, height, width, _ = input_spec.shape
    else:
        _, _, height, width = input_spec.shape
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    if input_spec.layout == "nhwc":
        return np.ascontiguousarray(resized[None, :, :, :])
    return np.ascontiguousarray(np.transpose(resized, (2, 0, 1))[None, :, :, :])


def extract(self, aligned_rgb_uint8: np.ndarray) -> np.ndarray:
    outputs = self.session.infer(aligned_rgb_uint8)
    if not outputs:
        return np.zeros(0, dtype=np.float32)
    return l2_normalize(np.asarray(outputs[0], dtype=np.float32).reshape(-1))
```

Pass the manifest-derived `self.session.input_spec` into `bgr_to_rgb_uint8()`; the test uses the expected IR50 NHWC layout but production code accepts either declared layout. Do not use `PIL.Image` in the RKNN path because OpenCV resize makes the resize method explicit and testable.

- [ ] **Step 4: Implement both aligner modes with one canonical OpenCV affine stage**

For `aligner_runtime="rknn"`, consume `original_keypoints` and `align_score` from the RKNN outputs, then call the existing `align_image_and_keypoints()` exactly once. For `aligner_runtime="onnx-cpu"`, construct the existing ONNX aligner session only for `cvlface_dfa_mobilenet.onnx`; keep IR50 on RKNN.

```python
if aligner_runtime == "onnx-cpu":
    if aligner_onnx_model_path is None:
        raise ValueError("aligner_onnx_model_path is required for onnx-cpu fallback")
    self._onnx_aligner = create_session(aligner_onnx_model_path, provider_mode="cpu")
elif aligner_runtime == "rknn":
    aligner_spec = load_input_spec(manifest_path, aligner_model_path)
    self._rknn_aligner = aligner_session or RknnModelSession(aligner_model_path, aligner_spec, core_mask=core_mask)
else:
    raise ValueError(f"Unknown aligner_runtime: {aligner_runtime!r}")
```

When `align_score < align_score_threshold`, return the existing zero-vector `FaceFeature` with `feature_valid=False`. `describe()` must name each sub-runtime separately: `recognition_runtime`, `aligner_runtime`, `recognition_model_path`, and `aligner_model_path`.

- [ ] **Step 5: Run the face engine tests and one existing ONNX face-engine regression subset**

Run: `uv run pytest tests/test_cvlface_rknn.py -q`

Run: `uv run pytest -q -k 'cvlface or recognition'`

Expected: the fake-RKNN tests pass and existing ONNX results retain their prior behavior.

- [ ] **Step 6: Commit the RKNN feature engine**

```bash
git add face_demo/face_engine/__init__.py face_demo/face_engine/cvlface_rknn.py tests/test_cvlface_rknn.py
git commit -m "feat: add RKNN face feature engine"
```

## Task 4: 增加显式 runtime 配置、工厂选择与就绪状态

**Files:**
- Create: `tests/test_runtime_configuration.py`
- Modify: `face_api/core/config.py`
- Modify: `face_api/main.py`
- Modify: `face_api/core/recognition.py`
- Modify: `face_api/rest/routers/health.py`

**Interfaces:**
- Consumes: Task 2 `YoloFaceRknnDetector` and Task 3 `CVLFaceRknnEngine`.
- Produces: `ApiConfig.runtime: Literal["onnx", "rknn"]`, RKNn path/config fields, and `RecognitionService.describe()` with `runtime` key.
- Produces: `RecognitionService.close() -> None` which closes detector and engine sessions if they define `close()`.

- [ ] **Step 1: Write failing tests for parser defaults, RKNN construction and strict failure behavior**

Create tests that inject fakes rather than load models:

```python
import pytest

from face_api.core.config import ApiConfig
from face_api.core.recognition import RecognitionService
from face_api.main import build_parser


def test_runtime_defaults_to_onnx():
    assert build_parser().parse_args([]).runtime == "onnx"
    assert ApiConfig().runtime == "onnx"


def test_rknn_runtime_load_failure_is_not_silently_replaced(monkeypatch):
    config = ApiConfig(runtime="rknn")
    monkeypatch.setattr(RecognitionService, "_create_rknn_detector", lambda self: (_ for _ in ()).throw(RuntimeError("load failed")))
    with pytest.raises(RuntimeError, match="load failed"):
        RecognitionService(config)
```

- [ ] **Step 2: Run the configuration tests and confirm they fail**

Run: `uv run pytest tests/test_runtime_configuration.py -q`

Expected: `ApiConfig` has no `runtime` field and the parser does not accept `--runtime`.

- [ ] **Step 3: Add precise configuration fields and CLI/env bindings**

Extend `ApiConfig` with these defaults:

```python
runtime: str = "onnx"
yolo_rknn_model_path: Path = Path("models/rknn/yolov12n-face.rknn")
recognition_rknn_model_path: Path = Path("models/rknn/cvlface_adaface_ir50_webface4m.rknn")
aligner_rknn_model_path: Path = Path("models/rknn/cvlface_dfa_mobilenet.rknn")
rknn_manifest_path: Path = Path("models/rknn/manifest.json")
rknn_core_mask: str = "auto"
rknn_aligner_fallback: str = "error"
```

In `build_parser()`, add `--runtime`, three `--*-rknn` paths, `--rknn-manifest`, `--rknn-core-mask`, and `--rknn-aligner-fallback` with the exact choices in the spec. Use a new `_env_value(name, default)` for strings and existing `_env_path()` for paths. Pass every value to `ApiConfig` in `main()`.

- [ ] **Step 4: Route construction only by runtime and preserve ONNX construction verbatim**

Replace the implicit ONNX-only factory with explicit methods:

```python
def _create_detector(self):
    if self.runtime == "onnx":
        return self._create_onnx_detector()
    if self.runtime == "rknn":
        return self._create_rknn_detector()
    raise ValueError(f"Unknown inference runtime: {self.runtime!r}")

def _create_engine(self):
    if self.runtime == "onnx":
        return self._create_onnx_engine()
    if self.runtime == "rknn":
        return self._create_rknn_engine()
    raise ValueError(f"Unknown inference runtime: {self.runtime!r}")
```

`_create_rknn_detector()` and `_create_rknn_engine()` both pass `config.rknn_manifest_path`; `_create_rknn_engine()` passes `aligner_runtime="onnx-cpu"` only if `rknn_aligner_fallback == "onnx-cpu"`; otherwise it passes `"rknn"`. Include `runtime` in `describe()` at the top level. `/readyz` must call `recognition.describe()` and return `ready=false` if construction or description throws; keep its current outer response envelope unchanged.

- [ ] **Step 5: Run configuration and API interface tests**

Run: `uv run pytest tests/test_runtime_configuration.py -q`

Run: `uv run pytest scripts/test_rest_api_interfaces.py -q`

Expected: parser/configuration tests pass, existing REST schema tests pass, and `runtime=rknn` loading errors are observable.

- [ ] **Step 6: Commit runtime selection**

```bash
git add face_api/core/config.py face_api/main.py face_api/core/recognition.py face_api/rest/routers/health.py tests/test_runtime_configuration.py
git commit -m "feat: make inference runtime configurable"
```

## Task 5: 防止 ONNX/RKNN 特征混用并提供可审计的图库重建

**Files:**
- Create: `tests/test_subject_store_embedding_metadata.py`
- Create: `scripts/rebuild_gallery_features.py`
- Modify: `face_api/core/subject_store.py`
- Modify: `face_api/rest/routers/subjects.py`

**Interfaces:**
- Consumes: `RecognitionService.describe()` with `runtime` and recognition model path; `SubjectStore` image paths.
- Produces: `EmbeddingIdentity(runtime: str, model_sha256: str)` and `SubjectStore.list_feature_records(identity: EmbeddingIdentity) -> list[SubjectFeatureRecord]`.
- Produces: `SubjectStore.create_subject(subject_id: str, name: str, image_bytes: bytes, feature: np.ndarray, embedding_identity: EmbeddingIdentity) -> SubjectRecord` and `update_subject(subject_id: str, name: str | None = None, image_bytes: bytes | None = None, feature: np.ndarray | None = None, embedding_identity: EmbeddingIdentity | None = None) -> SubjectRecord`.
- Produces: `python scripts/rebuild_gallery_features.py --source-data-dir <backup> --target-data-dir <new> --runtime rknn`.

- [ ] **Step 1: Write failing migration and compatibility tests against a temporary SQLite database**

Create a test verifying automatic migration and compatibility filtering:

```python
from face_api.core.subject_store import EmbeddingIdentity, SubjectStore


def test_list_feature_records_rejects_mixed_embedding_identity(tmp_path):
    store = SubjectStore(tmp_path)
    rknn = EmbeddingIdentity(runtime="rknn", model_sha256="a" * 64)
    onnx = EmbeddingIdentity(runtime="onnx", model_sha256="b" * 64)
    store.create_subject("alice", "Alice", b"image", [1.0, 2.0], embedding_identity=rknn)
    assert [item.subject_id for item in store.list_feature_records(rknn)] == ["alice"]
    assert store.list_feature_records(onnx) == []
```

Add a separate test that creates a legacy `subjects` table without the new columns, instantiates `SubjectStore`, then asserts `PRAGMA table_info(subjects)` contains `embedding_runtime` and `embedding_model_sha256`.

- [ ] **Step 2: Run the storage tests to confirm they fail**

Run: `uv run pytest tests/test_subject_store_embedding_metadata.py -q`

Expected: import fails because `EmbeddingIdentity` does not exist.

- [ ] **Step 3: Make the SQLite migration idempotent and attach identity to every new feature**

Add this frozen data class and migration logic:

```python
@dataclass(frozen=True)
class EmbeddingIdentity:
    runtime: str
    model_sha256: str


def _ensure_embedding_columns(self, conn: sqlite3.Connection) -> None:
    names = {row["name"] for row in conn.execute("PRAGMA table_info(subjects)")}
    if "embedding_runtime" not in names:
        conn.execute("ALTER TABLE subjects ADD COLUMN embedding_runtime TEXT NOT NULL DEFAULT 'legacy'")
    if "embedding_model_sha256" not in names:
        conn.execute("ALTER TABLE subjects ADD COLUMN embedding_model_sha256 TEXT NOT NULL DEFAULT 'legacy'")
```

Call `_ensure_embedding_columns()` immediately after `CREATE TABLE IF NOT EXISTS`. Persist the identity fields in both `INSERT` and any update that replaces an image/feature. `list_feature_records(identity)` must use `WHERE embedding_runtime = ? AND embedding_model_sha256 = ?`; it must never return `legacy` data for an ONNX or RKNN query.

- [ ] **Step 4: Propagate identity from the recognition service and add a rebuild command**

Add `RecognitionService.embedding_identity()` that SHA-256 hashes the active recognition model file and returns its runtime/model hash. Subject router create/update calls pass it into `SubjectStore`. The rebuild script must:

```python
for record in source_store.list_subjects():
    image_bytes = Path(record.image_path).read_bytes()
    feature = service.extract_feature(image_bytes)
    target_store.create_subject(record.subject_id, record.name, image_bytes, feature, embedding_identity=identity)
```

Before the loop, reject a target directory whose `face_api.db` already exists; this avoids overwriting production identities. At the end print `rebuilt=<count> runtime=<runtime> model_sha256=<digest>`.

- [ ] **Step 5: Run storage and subject API regression tests**

Run: `uv run pytest tests/test_subject_store_embedding_metadata.py -q`

Run: `uv run pytest scripts/test_rest_api_interfaces.py -q`

Expected: schema migration/identity tests pass and registered feature behavior remains compatible for a single runtime.

- [ ] **Step 6: Commit feature compatibility protection**

```bash
git add face_api/core/subject_store.py face_api/rest/routers/subjects.py scripts/rebuild_gallery_features.py tests/test_subject_store_embedding_metadata.py
git commit -m "feat: track gallery embedding runtime identity"
```

## Task 6: 将长视频流程从 PyTorch 检测替换为 RKNN 检测加 ByteTrack

**Files:**
- Create: `tests/test_rknn_video_tracker.py`
- Create: `face_api/video/bytetrack.py`
- Create: `face_api/video/rknn_tracker.py`
- Modify: `face_api/app.py`
- Modify: `face_api/video/analysis.py`

**Interfaces:**
- Consumes: `YoloFaceRknnDetector.detect_all(frame) -> list[FaceDetection]` from Task 2.
- Produces: `ByteTrackAssociator(track_buffer=30, high_threshold=0.5, low_threshold=0.1, match_iou=0.5)` with `update(detections: list[FaceDetection]) -> list[tuple[FaceDetection, int]]`.
- Produces: `RknnByteTrackFaceTracker(detector, track_buffer=30, match_thresh=0.5)` with `track_nearest(frame) -> YoloTrackResult | None`.
- Produces: `VideoAnalysisProcessor` tracker construction that chooses `RknnByteTrackFaceTracker` for `runtime="rknn"`.

- [ ] **Step 1: Write failing tests for stable IDs and a no-PyTorch construction path**

Use a detector fake which produces slightly moving boxes in two frames:

```python
from face_api.video.rknn_tracker import RknnByteTrackFaceTracker
from face_demo.face_engine import FaceDetection


class SequenceDetector:
    def __init__(self):
        self.frames = [
            [FaceDetection((10, 10, 80, 80), 0.95, None)],
            [FaceDetection((14, 12, 80, 80), 0.94, None)],
        ]
    def detect_all(self, frame):
        return self.frames.pop(0)


def test_tracker_preserves_track_id_for_overlapping_npu_detections():
    tracker = RknnByteTrackFaceTracker(SequenceDetector(), match_thresh=0.5)
    first = tracker.track_nearest(object())
    second = tracker.track_nearest(object())
    assert first is not None and second is not None
    assert first.track_id == second.track_id
    assert first.source == "rknn_bytetrack"
```

Add a test that monkeypatches `sys.modules["torch"] = None`, constructs the NPU tracker, and verifies a frame can be processed. This asserts the tracker itself never imports Ultralytics.

- [ ] **Step 2: Run the video tracker test to confirm it fails**

Run: `uv run pytest tests/test_rknn_video_tracker.py -q`

Expected: collection fails because `face_api.video.rknn_tracker` does not exist.

- [ ] **Step 3: Implement a local two-stage ByteTrack association class with no Torch import**

Create `face_api/video/bytetrack.py`. It uses only NumPy and the already pinned `lap` package: high-confidence detections (`score >= 0.5`) first match active tracks; unmatched tracks then match low-confidence detections (`0.1 <= score < 0.5`); unmatched high-confidence detections create IDs; tracks with `misses > 30` are removed. This is the ByteTrack high/low association needed by the application and avoids carrying a PyTorch or Ultralytics runtime into the RKNN image.

```python
def _associate(tracks: list[_Track], detections: list[FaceDetection], minimum_iou: float) -> list[tuple[int, int]]:
    if not tracks or not detections:
        return []
    cost = 1.0 - np.asarray([[_iou(track.box, det.box) for det in detections] for track in tracks], dtype=np.float64)
    row_to_column, _, _ = lap.lapjv(cost, extend_cost=True, cost_limit=1.0 - minimum_iou)
    return [(row, int(column)) for row, column in enumerate(row_to_column) if column >= 0]

def update(self, detections: list[FaceDetection]) -> list[tuple[FaceDetection, int]]:
    high = [det for det in detections if det.score >= self.high_threshold]
    low = [det for det in detections if self.low_threshold <= det.score < self.high_threshold]
    self._match_and_update(high)
    self._match_and_update(low, only_unmatched_tracks=True)
    self._expire_lost_tracks()
    return [(track.detection(), track.track_id) for track in self._tracks if track.misses == 0]
```

`RknnByteTrackFaceTracker.track_nearest()` calls `self.detector.detect_all(frame)`, passes the list to this associator, converts each returned track to `YoloTrackResult`, then invokes the existing `_choose_nearest()` policy. It has no `torch`, `ultralytics`, `.pt`, or `YOLO` import.

- [ ] **Step 4: Wire the tracker factory based on `ApiConfig.runtime`**

In `face_api/app.py`, replace the unconditional `YoloFaceTracker` constructor with:

```python
def _build_video_tracker_factory(config: ApiConfig, recognition_service: RecognitionService):
    if config.runtime == "rknn":
        from face_api.video.rknn_tracker import RknnByteTrackFaceTracker
        return lambda: RknnByteTrackFaceTracker(recognition_service.detector)
    return _build_onnx_video_tracker_factory(config)
```

Pass `app.state.recognition_service` at app construction. `VideoAnalysisProcessor` remains responsible for sampling and calling `recognize_detection()`; no result schema changes.

- [ ] **Step 5: Run video tracker and video analysis interface tests**

Run: `uv run pytest tests/test_rknn_video_tracker.py -q`

Run: `uv run pytest scripts/test_video_analysis_interfaces.py -q`

Expected: the two-frame ID test and current video-analysis interface tests pass; no `.pt` model is loaded for runtime `rknn`.

- [ ] **Step 6: Commit video tracker integration**

```bash
git add face_api/app.py face_api/video/analysis.py face_api/video/bytetrack.py face_api/video/rknn_tracker.py tests/test_rknn_video_tracker.py
git commit -m "feat: use RKNN detections for video tracking"
```

## Task 7: 创建可审计的 ONNX 到 RKNN 转换与模型 manifest

**Files:**
- Create: `tests/test_rknn_conversion_manifest.py`
- Create: `scripts/convert_rknn_models.py`
- Create: `models/rknn/manifest.schema.json`
- Modify: `.gitignore`
- Modify: `.dockerignore`

**Interfaces:**
- Produces: `python scripts/convert_rknn_models.py --model yolo|recognition|aligner --precision fp|int8 --calibration-list <path> --output-dir <path>`.
- Produces: `models/rknn/manifest.json` conforming to `manifest.schema.json`; actual model binaries remain ignored.
- Consumes: existing ONNX source models and a newline-delimited absolute image calibration list.

- [ ] **Step 1: Write failing manifest serialization tests without importing RKNN Toolkit2**

Test only pure metadata helpers:

```python
from pathlib import Path

from scripts.convert_rknn_models import build_manifest_entry, sha256_file


def test_manifest_entry_records_source_and_output_hashes(tmp_path: Path):
    source = tmp_path / "source.onnx"; source.write_bytes(b"onnx")
    output = tmp_path / "model.rknn"; output.write_bytes(b"rknn")
    entry = build_manifest_entry("yolo", source, output, precision="int8", toolkit_version="2.3.2")
    assert entry["target_platform"] == "rk3588"
    assert entry["onnx_sha256"] == sha256_file(source)
    assert entry["rknn_sha256"] == sha256_file(output)
```

- [ ] **Step 2: Run the manifest test to confirm it fails**

Run: `uv run pytest tests/test_rknn_conversion_manifest.py -q`

Expected: collection fails because the conversion script does not exist.

- [ ] **Step 3: Implement model specifications, FP preflight and INT8 build**

In `scripts/convert_rknn_models.py`, define model specs with the exact preprocessing values:

```python
MODEL_SPECS = {
    "yolo": {"onnx": Path("models/onnx/yolov12n-face.onnx"), "mean": [0, 0, 0], "std": [255, 255, 255]},
    "recognition": {"onnx": Path("models/onnx/cvlface_adaface_ir50_webface4m.onnx"), "mean": [127.5] * 3, "std": [127.5] * 3},
    "aligner": {"onnx": Path("models/onnx/cvlface_dfa_mobilenet.onnx"), "mean": [127.5] * 3, "std": [127.5] * 3},
}

def configure_and_build(rknn, spec, *, precision: str, calibration_list: Path | None) -> None:
    rknn.config(target_platform="rk3588", mean_values=[spec["mean"]], std_values=[spec["std"]])
    assert rknn.load_onnx(model=str(spec["onnx"])) == 0
    dataset = str(calibration_list) if precision == "int8" else None
    assert rknn.build(do_quantization=precision == "int8", dataset=dataset) == 0
```

Import `RKNN` inside `main()` so unit tests do not require Toolkit2. Run FP first (`--precision fp`); write no INT8 output when FP build fails. At the end call `export_rknn()` and run Toolkit2 simulator `init_runtime(target="rk3588")` plus one deterministic test input before writing a successful manifest entry.

- [ ] **Step 4: Define manifest schema and ignore only restricted binary artifacts**

Create `models/rknn/manifest.schema.json` requiring a root object with `models` as a nonempty array. Every `models` item requires `name`, `precision`, `target_platform`, `toolkit_version`, `onnx_path`, `onnx_sha256`, `rknn_path`, `rknn_sha256`, `inputs`, `outputs`, `conversion_log`, and `created_at`. Each `inputs` entry contains `name`, `layout`, `shape`, and `dtype`; the script must populate these from Toolkit2 tensor metadata, not static constants. The model script replaces an item only when its `name` and `precision` match, then atomically writes `manifest.json` through `manifest.json.tmp` and `Path.replace()`.

Add to `.gitignore`:

```gitignore
models/rknn/*.rknn
models/rknn/*.log
models/rknn/manifest.json
calibration/
validation/
third_party/rknn/*.whl
```

Do not ignore `models/rknn/manifest.schema.json` or conversion scripts. Exclude `calibration/`, `validation/`, `.rknn`, `.log`, and local Lite2 wheels from Docker build context in `.dockerignore`.

- [ ] **Step 5: Run unit tests and perform the x86_64 FP preflight for YOLO**

Run: `uv run pytest tests/test_rknn_conversion_manifest.py -q`

Run on the dedicated x86_64 Toolkit2 2.3.2 environment: `python scripts/convert_rknn_models.py --model yolo --precision fp --output-dir /secure/build/rknn`

Expected: unit tests pass; preflight produces a FP RKNN artifact, conversion log and manifest entry, or exits nonzero with an unsupported operator error that identifies the source model and node.

- [ ] **Step 6: Commit conversion source and schema, excluding binary artifacts**

```bash
git add scripts/convert_rknn_models.py models/rknn/manifest.schema.json tests/test_rknn_conversion_manifest.py .gitignore .dockerignore
git commit -m "feat: add reproducible RKNN model conversion"
```

## Task 8: 量化、模型一致性和性能门槛

**Files:**
- Create: `tests/test_rknn_validation.py`
- Create: `scripts/validate_rknn_parity.py`
- Create: `scripts/benchmark_rknn_service.py`
- Create: `docs/rknn-validation.md`

**Interfaces:**
- Consumes: `RecognitionService` configured in `onnx` and `rknn` modes, 500-image calibration list and separate 200-image validation list.
- Produces: `parity-report.json` with `detection_iou_rate`, `feature_cosine_rate`, `top1_match_rate`, failures and all model SHA-256 values.
- Produces: `benchmark-report.json` with per-stage latency and end-to-end p50/p95/FPS.

- [ ] **Step 1: Write failing pure-function tests for metrics and threshold exit behavior**

Create `tests/test_rknn_validation.py`:

```python
import numpy as np
import pytest

from scripts.validate_rknn_parity import box_iou, require_thresholds


def test_box_iou_is_one_for_identical_xywh_boxes():
    assert box_iou((10, 20, 30, 40), (10, 20, 30, 40)) == 1.0


def test_require_thresholds_rejects_feature_regression():
    with pytest.raises(SystemExit, match="feature_cosine_rate"):
        require_thresholds({"detection_iou_rate": 1.0, "feature_cosine_rate": 0.94, "top1_match_rate": 1.0})
```

- [ ] **Step 2: Run the metric test to confirm it fails**

Run: `uv run pytest tests/test_rknn_validation.py -q`

Expected: collection fails because `validate_rknn_parity.py` does not exist.

- [ ] **Step 3: Implement parity reporting with exact acceptance thresholds**

`validate_rknn_parity.py` must load the same image once through ONNX and RKNN configurations, calculate largest-face IoU, cosine similarity for successfully aligned faces and top-1 subject-id equality, and write all failure records as JSON. Its threshold function must be:

```python
THRESHOLDS = {"detection_iou_rate": 0.98, "feature_cosine_rate": 0.95, "top1_match_rate": 0.98}

def require_thresholds(metrics: dict[str, float]) -> None:
    for key, threshold in THRESHOLDS.items():
        if metrics[key] < threshold:
            raise SystemExit(f"{key}={metrics[key]:.4f} is below {threshold:.4f}")
```

Count a detection pass only when the ONNX detection score is at least `0.30` and the RKNN/ONNX box IoU is at least `0.90`; count a feature pass only when the feature cosine is at least `0.98`. Store image path, ONNX/RKNN boxes, similarities and model SHA values for every failed item.

- [ ] **Step 4: Implement a benchmark that measures the whole pipeline rather than raw NPU only**

`benchmark_rknn_service.py` must time `detect_frame`, `extract_feature_from_detection`, and `recognize_detection` independently with `time.perf_counter_ns()`, discard ten warm-up frames, calculate p50/p95 from at least 300 post-warmup samples, and emit:

```json
{
  "runtime": "rknn",
  "samples": 300,
  "detector_ms": {"p50": 0.0, "p95": 0.0},
  "recognition_ms": {"p50": 0.0, "p95": 0.0},
  "end_to_end_ms": {"p50": 0.0, "p95": 0.0},
  "fps": 0.0
}
```

Document that an RK3588 ONNX-CPU baseline is recorded first, then `rknn` must reduce end-to-end p95 by at least 50% for a 10-minute run and run error-free for 24 hours.

- [ ] **Step 5: Run validation unit tests, then build and validate all models**

Run: `uv run pytest tests/test_rknn_validation.py -q`

Run on the conversion host in order:

```bash
python scripts/convert_rknn_models.py --model yolo --precision int8 --calibration-list /secure/calibration/yolo.txt --output-dir /secure/build/rknn
python scripts/convert_rknn_models.py --model recognition --precision int8 --calibration-list /secure/calibration/face.txt --output-dir /secure/build/rknn
python scripts/convert_rknn_models.py --model aligner --precision fp --output-dir /secure/build/rknn
python scripts/convert_rknn_models.py --model aligner --precision int8 --calibration-list /secure/calibration/face.txt --output-dir /secure/build/rknn
```

Then deploy the secure artifacts to the box, run `python scripts/validate_rknn_parity.py --images /secure/validation/list.txt --report /secure/reports/parity-report.json`, and require exit code 0. If the aligner FP/INT8 path fails a build or parity threshold, record `aligner_runtime: onnx-cpu` in the production manifest and rerun parity with that explicit fallback.

- [ ] **Step 6: Commit validation/benchmark code and instructions**

```bash
git add scripts/validate_rknn_parity.py scripts/benchmark_rknn_service.py tests/test_rknn_validation.py docs/rknn-validation.md
git commit -m "test: add RKNN parity and performance gates"
```

## Task 9: 准备 RK3588 宿主机与安全的 ARM64 容器部署

**Files:**
- Create: `requirements-rknn.txt`
- Create: `Dockerfile.rknn`
- Create: `docker-compose.rknn.yml`
- Create: `scripts/check_rknn_device.py`
- Create: `docs/rk3588-deployment.md`
- Create: `tests/test_rknn_device_check.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: the verified RK3588 runtime contract and `.rknn` artifacts/manifest from Tasks 7–8.
- Produces: `python scripts/check_rknn_device.py --model <path>` returning exit code 0 only after Lite2 model load/init/inference succeeds.
- Produces: `docker compose -f docker-compose.rknn.yml up -d` NPU service on port 8000.

- [ ] **Step 1: Write a failing device-check test with a fake `RknnModelSession`**

Create a small unit test which verifies that a failed inference produces nonzero return from `main()` and a successful session prints `runtime=rknn` and the model path. Keep the checker injectable:

```python
def main(argv: list[str] | None = None, session_factory=RknnModelSession) -> int:
    args = build_parser().parse_args(argv)
    try:
        session = session_factory(args.model, load_input_spec(args.manifest), core_mask=args.core_mask)
        session.infer(np.zeros(session.input_spec.shape, dtype=session.input_spec.dtype))
        print(f"runtime=rknn model={args.model}")
        return 0
    except Exception as exc:
        print(f"RKNN device check failed: {exc}", file=sys.stderr)
        return 1
```

- [ ] **Step 2: Run the device-check test to confirm it fails**

Run: `uv run pytest tests/test_rknn_device_check.py -q`

Expected: collection fails because `scripts/check_rknn_device.py` does not exist.

- [ ] **Step 3: Implement host prerequisite checking and Lite2 install instructions**

`check_rknn_device.py` must fail clearly for: missing `/usr/lib/librknnrt.so`, inaccessible `/dev/dri/renderD129`, missing `rknnlite.api`, absent model file, and failed model inference. For the final model test use a zero uint8 array shaped from manifest input metadata, then call the Task 1 session.

Set `requirements-rknn.txt` to a single audited installation source line, with no package resolver ambiguity:

```text
./third_party/rknn/rknn_toolkit_lite2-2.3.2-cp311-cp311-linux_aarch64.whl
```

The deployment document must first run `python3 --version`, require CPython 3.11, then run `python3 -m pip install --no-deps -r requirements-rknn.txt`; if the wheel filename does not match CPython 3.11/aarch64, stop before installation and obtain the official matching 2.3.2 wheel.

- [ ] **Step 4: Create the least-privilege RKNN image and Compose service**

`Dockerfile.rknn` must use `python:3.11-slim`, install only OpenCV/FastAPI runtime system libraries already listed in `Dockerfile`, install `requirements-rknn.txt`, and define a non-root `faceapi` user. Set `LD_LIBRARY_PATH=/usr/lib` but do not copy a host library into the image.

`docker-compose.rknn.yml` must include these concrete constraints:

```yaml
services:
  face-api-rknn:
    build:
      context: .
      dockerfile: Dockerfile.rknn
    devices:
      - /dev/dri/renderD129:/dev/dri/renderD129
    group_add:
      - "${RKNN_RENDER_GID:?set RKNN_RENDER_GID to getent group render | cut -d: -f3}"
    volumes:
      - ./data/face_api:/app/data/face_api
      - ./models/rknn:/app/models/rknn:ro
      - /usr/lib/librknnrt.so:/usr/lib/librknnrt.so:ro
      - ./logs:/app/logs
    environment:
      FACE_API_RUNTIME: rknn
      FACE_API_RKNN_CORE_MASK: auto
    ports:
      - "8000:8000"
```

The command must pass `--runtime rknn`, models under `/app/models/rknn`, the actual RTSP URL and the selected `--rknn-aligner-fallback`. It must not include `--provider cuda`, NVIDIA configuration, `privileged`, or `rknn_server`.

- [ ] **Step 5: Verify on the RK3588 host before and after containerization**

Run on the box host:

```bash
getent group render
python3 scripts/check_rknn_device.py --model models/rknn/yolov12n-face.rknn
python3 -m face_api.main --runtime rknn --host 127.0.0.1 --port 8001 --data-dir /srv/face-api-rknn/data --yolo-face-rknn models/rknn/yolov12n-face.rknn --cvlface-recognition-rknn models/rknn/cvlface_adaface_ir50_webface4m.rknn --cvlface-aligner-rknn models/rknn/cvlface_dfa_mobilenet.rknn
curl --fail http://127.0.0.1:8001/readyz
RKNN_RENDER_GID=$(getent group render | cut -d: -f3) docker compose -f docker-compose.rknn.yml up -d --build
curl --fail http://127.0.0.1:8000/readyz
```

Expected: both ready responses report `runtime: "rknn"`; the container process belongs to the supplied render group and can infer without privileged access.

- [ ] **Step 6: Commit deployment artifacts and documentation**

```bash
git add requirements-rknn.txt Dockerfile.rknn docker-compose.rknn.yml scripts/check_rknn_device.py docs/rk3588-deployment.md README.md tests/test_rknn_device_check.py
git commit -m "docs: add RK3588 RKNN deployment artifacts"
```

## Task 10: 进行发布演练、可观测性确认与回滚验证

**Files:**
- Modify: `docs/rk3588-deployment.md`
- Modify: `docker-compose.rknn.yml`
- Modify: `scripts/benchmark_rknn_service.py`

**Interfaces:**
- Consumes: released `.rknn` artifacts, manifest, rebuilt gallery database and Task 8 benchmark report.
- Produces: a release evidence directory outside Git containing parity report, 10-minute performance report, 24-hour health report and rollback record.

- [ ] **Step 1: Define a failing release-gate check for an incomplete benchmark report**

Add a unit test to `tests/test_rknn_validation.py` that passes a report without `end_to_end_ms.p95` and asserts the release-gate function refuses it:

```python
def test_release_gate_rejects_missing_end_to_end_p95():
    with pytest.raises(SystemExit, match="end_to_end_ms.p95"):
        require_release_evidence({"runtime": "rknn"})
```

- [ ] **Step 2: Run the release-gate test to confirm it fails**

Run: `uv run pytest tests/test_rknn_validation.py::test_release_gate_rejects_missing_end_to_end_p95 -q`

Expected: import succeeds but fails because `require_release_evidence` does not exist.

- [ ] **Step 3: Implement explicit promotion criteria and diagnostics collection**

Add `require_release_evidence()` to `scripts/benchmark_rknn_service.py`. It must require `runtime == "rknn"`, `samples >= 300`, detector/recognition/end-to-end p95 fields, `npu_error_count == 0`, and `end_to_end_p95_improvement >= 0.50`. Emit a failure reason for the first missing or failing field.

Extend the deployment document with exact collection commands:

```bash
mkdir -p /secure/reports/rk3588-$(date +%F)
docker compose -f docker-compose.rknn.yml logs --since 10m face-api-rknn > /secure/reports/rk3588-$(date +%F)/service.log
python scripts/benchmark_rknn_service.py --runtime rknn --video data/videos/input2.mp4 --report /secure/reports/rk3588-$(date +%F)/benchmark-rknn.json
python scripts/benchmark_rknn_service.py --runtime onnx --video data/videos/input2.mp4 --report /secure/reports/rk3588-$(date +%F)/benchmark-onnx-cpu.json
```

Document the 24-hour canary on port `8001`, with a read-only stream/client and `/readyz` polling every minute; preserve logs and reports outside the repository.

- [ ] **Step 4: Verify all automated tests and both runtime modes before traffic cutover**

Run: `uv run pytest -q`

Run on the RK3588 box: `python scripts/validate_rknn_parity.py --images /secure/validation/list.txt --report /secure/reports/parity-report.json`

Expected: tests and parity gate return exit code 0. Capture the service’s `/readyz`, `rknnrt` version, RKNPU driver version and all model SHA-256 values with the release evidence.

- [ ] **Step 5: Execute the reversible traffic switch and prove rollback**

Back up database before re-feature extraction, then start the NPU service under a distinct container name/port. Promote it only after the 24-hour canary passes. The rollback command preserves data and swaps only the service runtime:

```bash
docker compose -f docker-compose.rknn.yml stop face-api-rknn
docker compose -f docker-compose.yml up -d face-api
curl --fail http://127.0.0.1:8000/readyz
```

Record the rollback test time, image tags and `/readyz` response in the release evidence directory. Do not delete `.rknn`, ONNX images, or the database backup during the first complete release cycle.

- [ ] **Step 6: Commit only the release-gate code and documentation change**

```bash
git add scripts/benchmark_rknn_service.py tests/test_rknn_validation.py docs/rk3588-deployment.md docker-compose.rknn.yml
git commit -m "test: add RKNN release readiness gate"
```

## Plan Self-Review

* **Spec coverage:** Tasks 1–4 establish the independent runtime and API behavior; Task 5 prevents feature mixing and reconstructs gallery; Task 6 covers the separate long-video tracking path; Task 7 converts and fingerprints models; Task 8 establishes INT8/parity/performance gates; Task 9 handles host and Docker deployment; Task 10 runs the canary, promotion and rollback. This covers every section of the approved design.
* **Placeholder scan:** The plan has no deferred work markers; every implementation step names exact files, callable interfaces, commands and acceptance conditions.
* **Type consistency:** all new adapter methods use `detect_all`, `detect_largest`, `track_nearest`, `feature_from_detection_with_info`, `RknnModelSession.infer`, and `EmbeddingIdentity` consistently across their producer and consumer tasks.

# RK3588 NPU 人脸服务迁移设计

## 1. 目标与范围

将现有人脸 REST/WebSocket/视频分析服务部署到 RK3588 边缘盒子，并让主要的神经网络推理使用其 NPU，消除当前无 GPU 主机上 ONNX Runtime CPU 推理的主要性能瓶颈。服务对外 HTTP、WebSocket、数据库和人物特征存储接口保持不变。

本次迁移覆盖以下三个推理图：

| 业务环节 | 当前 ONNX 模型 | NPU 目标产物 | 目标执行位置 |
| --- | --- | --- | --- |
| 人脸检测 | `models/onnx/yolov12n-face.onnx` | `models/rknn/yolov12n-face.rknn` | RK3588 NPU |
| 人脸对齐网络 | `models/onnx/cvlface_dfa_mobilenet.onnx` | `models/rknn/cvlface_dfa_mobilenet.rknn` | 优先 RK3588 NPU；转换失败时 CPU 回退 |
| 人脸特征提取 | `models/onnx/cvlface_adaface_ir50_webface4m.onnx` | `models/rknn/cvlface_adaface_ir50_webface4m.rknn` | RK3588 NPU |

不在本次范围内：更换人脸检测/识别模型、重新训练模型、将 OpenCV 几何处理或向量检索迁移到 NPU、改变 API 返回字段、引入分布式任务队列。

## 2. 已验证的目标硬件基线

盒子已经完成真实 NPU 推理验证，以下版本是部署契约的一部分：

| 项目 | 已验证值 |
| --- | --- |
| 设备与系统 | RK3588，Ubuntu 22.04.5 LTS，aarch64 |
| 内核 | `6.1.141` |
| NPU 驱动 | `RKNPU driver v0.9.8` |
| NPU DRM 节点 | `card1` 对应 `/sys/devices/platform/fdab0000.npu`；渲染节点为 `renderD129` |
| 系统软件包 | `rknpu 2.3.0a`（T-Firefly） |
| 本地实际 Runtime | `rknn_api/rknnrt 2.3.2`，构建于 `2025-04-09` |
| rknn_server | `2.3.0`，由 `rknn_server.service` 开机启动 |
| 冒烟测试 | `/usr/share/model/RK3588/mobilenet_v1.rknn` 10 次推理成功，3.23–3.61 ms/次，退出码 0 |

`rknn_server` 是调试/传输服务，不是应用进程调用本地 NPU 的前置条件。生产服务直接经 `librknnrt.so` 和 DRM NPU 节点推理；服务启动时不得重复启动 `rknn_server`。

开发机转换工具和设备 Python 运行时必须选择 RKNN 2.3.2 系列，并在第一次端到端模型验证时再次由模型加载日志确认。包名 `rknpu 2.3.0a` 和 `rknn_server 2.3.0` 不能替代上述 Runtime 实测版本。

## 3. 当前项目与迁移边界

当前业务服务由 `face_api/main.py` 构造 `ApiConfig`，`RecognitionService` 在 `face_api/core/recognition.py` 中创建：

1. `face_demo/yolo_face_onnx.py:YoloFaceOnnxDetector`：BGR 图像字母盒化、BGR→RGB、NCHW、`float32 / 255`，再执行 YOLO 输出解码与 CPU NMS。
2. `face_demo/face_engine/cvlface_onnx.py:CVLFaceOnnxEngine`：人脸裁剪、BGR→RGB、112×112 标准化、DFA 对齐网络、OpenCV 仿射对齐、AdaFace 特征提取及 L2 归一化。
3. `face_api/core/recognition.py`：最大脸选择、特征匹配和 API 业务错误处理。

实时检测和识别分别在不同的工作线程中调用 `RecognitionService`。长视频任务则通过 `_build_video_tracker_factory()` 创建 `YoloFaceTracker`；该实现目前仍调用 Ultralytics/PyTorch 跟踪。因此 NPU 迁移必须同时解决视频分析使用 NPU 检测结果与 CPU 跟踪 ID 的衔接，不能只替换在线 API 的检测器。

当前代码的 `provider` 只有 `auto`、`cuda`、`cpu` 三种 ONNX Runtime 选择；它不能表示 RKNN Runtime。`runtime` 必须成为独立选项，避免把 NPU 误实现成 ONNX provider。

## 4. 目标架构

```text
FastAPI / WebSocket / 视频任务（接口不变）
                 |
          RecognitionService
                 |
     runtime = onnx | rknn（显式选择）
       |                         |
 ONNX 现有适配器           RKNN 适配器（新）
       |                         |
 ONNX Runtime          RKNNLite + librknnrt.so
                                  |
                      /dev/dri/renderD129 → RK3588 NPU

CPU 保留：OpenCV 解码、letterbox、YOLO 解码/NMS、裁剪、仿射变换、
L2 归一化、余弦匹配、ByteTrack 与 API 业务逻辑。
```

新增 RKNN 适配器必须实现与既有调用点相同的能力：

* 检测器：`detect_largest(image)`、可取得全量检测框的内部方法、`describe()`；实时链路行为与 `YoloFaceOnnxDetector` 一致。
* 特征引擎：`feature_from_detection_with_info(frame, detection)`、`feature_from_detection()`、`describe()`；返回现有 `FaceFeature` 数据结构。
* 模型会话：加载、一次推理、关闭和执行元数据报告。单个会话推理必须由互斥锁保护，避免 FastAPI 并发请求共用同一 RKNN context 时发生不确定行为。

`describe()` 统一返回 `runtime: "rknn"`、模型路径、`rknnrt_version`、NPU core mask 与实际模型输入/输出元数据。健康检查的 ready 状态必须在三个必需模型成功加载后才为成功。

## 5. 模型转换与量化设计

### 5.1 通用规则

* 以仓库现有 ONNX 文件作为唯一转换输入；不从 PyTorch 权重重新导出，避免引入另一套算子图。
* 三个模型都构建为目标平台 `rk3588`、固定 `batch=1` 的 `.rknn` 文件。
* 使用 RKNN-Toolkit2 2.3.2，在 x86_64 Linux 开发/CI 环境执行转换；不要在 RK3588 生产盒子上安装完整 Toolkit2。
* 每个模型先执行不量化构建以检查不支持算子，再执行 INT8 PTQ 构建。不能跳过 FP 模型检查直接调参量化。
* 每次产物同时保存转换日志、工具版本、ONNX SHA-256、校准集清单 SHA-256、输入/输出 shape、量化参数以及 `.rknn` SHA-256 到 `models/rknn/manifest.json`。
* 校准样本含有人脸生物信息，不提交到 Git、不复制到 Docker 镜像；存放在受控的内部路径，仅将相对文件名和 SHA-256 写入私有构建记录。

### 5.2 输入语义

| 模型 | 源 ONNX 语义 | RKNN 构建输入语义 | 运行时输入 |
| --- | --- | --- | --- |
| YOLOv12n | RGB，640×640，NCHW，`float32`，值为 `pixel / 255` | `mean=[0,0,0]`，`std=[255,255,255]` | letterbox 后 RGB `uint8`；布局按产物 metadata 提供 |
| DFA | RGB，112×112，NCHW，`float32`，`(pixel - 127.5) / 127.5` | `mean=[127.5,127.5,127.5]`，`std=[127.5,127.5,127.5]` | RGB `uint8`；布局按产物 metadata 提供 |
| AdaFace IR50 | RGB，112×112，NCHW，`float32`，`(pixel - 127.5) / 127.5` | `mean=[127.5,127.5,127.5]`，`std=[127.5,127.5,127.5]` | 对齐后的 RGB `uint8`；布局按产物 metadata 提供 |

转换脚本必须读取生成模型的输入 metadata 决定实际 `NHWC/NCHW`，禁止用 Python 端固定猜测布局。运行输出一律转换为 `float32` 后再复用现有 YOLO 后处理和特征 L2 归一化逻辑。

### 5.3 校准集与验收集

使用 500 张不重复的、已获部署授权的人脸图像建立 INT8 校准集：

* 250 张来自当前摄像头/RTSP 实际画面，覆盖白天、夜间、逆光、侧脸、远近脸、遮挡和多人；
* 150 张来自已注册人员的额外采集图，且不与注册特征构建用图重复；
* 100 张无脸或包含小脸/多人脸的场景；
* 每张图在转换前使用本项目同一套 letterbox、裁剪、对齐预处理生成对应模型输入。

使用独立的 200 张图像作为验收集；校准集和验收集没有重复的原始图片或相邻视频帧。

## 6. 分阶段交付与回退策略

### 阶段 A：YOLOv12n 检测器 NPU 化

先完成 `yolov12n-face.rknn`。保留 `preprocess_image()`、`decode_yolo_output()`、NMS 和最大脸选择，确保框坐标语义不变。若 INT8 验收不通过，先交付该模型的非量化 RKNN 版本用于定位；只有在非量化正确、INT8 错误时才调整校准集或量化配置。

### 阶段 B：AdaFace IR50 特征提取 NPU 化

完成 IR50 的 `.rknn` 并保持当前 OpenCV 几何对齐流程。新注册的人脸与历史 gallery 特征必须使用同一推理 runtime 重新生成；迁移时通过数据库/元数据中的 `embedding_runtime` 和 `embedding_model_sha256` 标记，禁止混合 ONNX 与 INT8-RKNN 特征向量进行阈值判定。

### 阶段 C：DFA 对齐网络 NPU 化或 CPU 回退

尝试转换 DFA。若 Toolkit2 报告不支持算子、模型 build 失败，或其验收结果不合格，则在 manifest 中将 `aligner_runtime` 固定为 `onnx-cpu`，而检测与 IR50 继续使用 NPU。该回退是正式支持的部署模式，不阻断前两阶段交付。

### 阶段 D：视频分析跟踪接入

新增 CPU 跟踪适配器：接收 RKNN 检测器的全量 `FaceDetection` 列表，输入 ByteTrack，返回现有 `YoloTrackResult` 与稳定 `track_id`。不得继续通过 `YoloFaceTracker` 加载 `.pt` 模型，因为这会在 RK3588 上重新引入 PyTorch 检测开销。实时“最大脸”流程不需要 track ID，长视频流程必须要求其存在。

每个阶段独立可部署：`runtime=onnx` 始终保留；`runtime=rknn` 在某个模型加载或自检失败时必须在启动期报出模型名与原始 Runtime 错误并退出，绝不能静默降级到 CPU。

## 7. 配置、部署与依赖

### 7.1 配置接口

新增服务级配置：

* `--runtime {onnx,rknn}` / `FACE_API_RUNTIME`，默认 `onnx`，避免改变已有部署的行为；
* `--yolo-face-rknn` / `FACE_API_YOLO_FACE_RKNN`；
* `--cvlface-recognition-rknn` / `FACE_API_CVLFACE_RECOGNITION_RKNN`；
* `--cvlface-aligner-rknn` / `FACE_API_CVLFACE_ALIGNER_RKNN`；
* `--rknn-core-mask {auto,0,1,2,0_1_2}` / `FACE_API_RKNN_CORE_MASK`，默认 `auto`；
* `--rknn-aligner-fallback {error,onnx-cpu}` / `FACE_API_RKNN_ALIGNER_FALLBACK`，默认 `error`。

`--provider` 继续只服务于 `runtime=onnx`。当 `runtime=rknn` 时传入 `--provider` 不改变 NPU 实现，但在 `describe()` 中仍报告该字段以便诊断。

### 7.2 Python 运行时

目标设备上的 `rknn-toolkit-lite2` AArch64 wheel 必须与 CPython 版本兼容。当前项目要求 Python 3.11–3.12；部署前先在盒子上以目标 Python 安装 2.3.2 Lite2 wheel 并执行模型加载测试。

如果厂商 2.3.2 Lite2 仅提供 Python 3.10 wheel，则最终 RK3588 服务固定到 Python 3.10，并将三个 `pyproject.toml` 的下限统一调整为 `>=3.10`；该变更必须通过全量测试后才能采用。不得通过在 Python 3.11 中强行安装不匹配 wheel 绕过 ABI 检查。

RK3588 的运行时依赖不安装 `onnxruntime-gpu`、CUDA、PyTorch、Ultralytics 或完整 RKNN-Toolkit2。依赖拆分为 `onnx` 与 `rknn` 两套可选组，RKNPU 生产镜像只安装 `rknn` 组以及 FastAPI、OpenCV、NumPy、Pillow、scikit-image 等必要 CPU 依赖。

### 7.3 容器化

先用盒子宿主机 Python 服务完成端到端验收，再制作 ARM64 Docker 镜像。容器必须：

* 使用与 Lite2 wheel 匹配的 Python 基础镜像；
* 显式提供设备 Runtime `librknnrt.so`；
* 只透传 NPU 节点 `/dev/dri/renderD129`，不使用 `privileged: true`；
* 以具备宿主机 `render` 组权限的非 root 用户运行；
* 将 `data/face_api`、`models/rknn` 和日志作为独立挂载卷；
* 不在容器内运行第二个 `rknn_server`。

Docker Compose 的 NPU 服务与现有 GPU Compose 分离，命名为 `docker-compose.rknn.yml`。它不得继承 `docker-compose.gpu.yml` 的 NVIDIA runtime、CUDA 环境变量或 ONNX GPU 依赖。

## 8. 验收与可观测性

使用 `data/videos/input2.mp4` 和第 5.3 节独立验收集，逐阶段执行 ONNX CPU 与 RKNN 对比。固定 `yolo_conf=0.25`、`yolo_iou=0.45`、相同图库和相同视频采样间隔。

| 验收项 | 通过标准 |
| --- | --- |
| NPU 初始化 | 服务启动日志显示三个模型的实际 runtime、RKNPU driver、rknnrt 版本和模型 SHA；模型加载失败则进程非零退出 |
| 检测一致性 | 验收集里 ONNX 置信度 `>=0.30` 的最大脸，RKNN 最大脸与其 IoU `>=0.90` 的比例不少于 98% |
| 特征一致性 | 验收集中 ONNX 与 RKNN 对同一张已对齐脸的余弦相似度 `>=0.98` 的比例不少于 95% |
| 身份结果 | 200 张验收图的 Top-1 `subject_id` 与 ONNX 相同的比例不少于 98%；不通过的样本逐张记录路径、两端相似度和模型 SHA |
| API 回归 | REST、WebSocket、视频分析接口测试全部通过，响应 schema 不变 |
| 性能 | 在盒子上运行 10 分钟 RTSP 或本地循环视频，端到端 p95 帧处理时间相对盒子 ONNX-CPU 基线降低至少 50%，无连续 60 秒 NPU 初始化/推理错误 |
| 稳定性 | 24 小时服务运行后无进程退出、无模型 context 泄漏导致的持续内存增长；重启后可自动重新加载模型 |

每条性能日志包含 detector、aligner、recognition 的推理毫秒数及端到端毫秒数；不得仅用 NPU 单模型耗时代表服务 FPS。实际模型的预处理、CPU 后处理、视频解码和网络推流均计入端到端指标。

## 9. 风险与明确处置

| 风险 | 处置 |
| --- | --- |
| ONNX 存在 Toolkit2 不支持算子 | 先非量化 build；定位算子后只在转换副本做 ONNX 图改写，保留原 ONNX 与变更脚本；DFA 可走正式 CPU 回退 |
| INT8 引发检测或识别漂移 | 先扩大/修正校准样本覆盖范围，再重新量化；不调整线上阈值掩盖模型漂移 |
| RKNN Lite2 没有 Python 3.11 wheel | 将 RK3588 部署运行时固定为 Python 3.10，并在修改项目版本约束后全量测试 |
| 容器无 NPU 访问权限 | 先宿主机验证，再检查 `renderD129` 映射、`render` 组 GID 与 `librknnrt.so` 动态加载；不通过 `privileged` 规避 |
| 视频分析仍使用 PyTorch 检测 | 用 RKNN 检测器输出接 CPU ByteTrack；该适配完成前禁止在 RKNN 配置启用长视频任务 |
| ONNX 与 RKNN 特征混用 | 迁移时重新生成 gallery 特征并存储 runtime/model hash；旧特征作为备份而非同库参与匹配 |

## 10. 发布顺序

1. 在开发机生成并验收模型产物和 manifest。
2. 在 RK3588 宿主机安装匹配的 Lite2 环境，运行单模型和服务集成验收。
3. 备份既有 `data/face_api`，在新特征库上重建 gallery，完成 REST/WebSocket/视频回归。
4. 以 `runtime=rknn` 启动一个只读流量或测试端口实例，观察 24 小时。
5. 切换正式服务；保留原 ONNX 镜像、ONNX 模型和 `runtime=onnx` 启动命令，直到一个完整发布周期内无回归。


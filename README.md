# Face Serve V2：RK3588 混合推理

检测与识别使用 RKNN Lite2，CVLFace 对齐使用 CPU ONNX Runtime。REST、视频分析和 WebSocket 继续共用检测与识别服务；最近目标选择、有界异步识别队列保留。

## 模型与输入契约

| 阶段 | 默认文件 | 执行位置 |
| --- | --- | --- |
| 检测 | models/rknn/retinaface-mobilenet0.25-480x720-int8.rknn | RK3588 NPU core 0 |
| 对齐 | models/onnx/cvlface_dfa_mobilenet.onnx | ONNX Runtime CPU |
| 识别 | models/rknn/recognition-int8.rknn | RK3588 NPU core 1 |

检测输入固定为 NCHW 1×3×480×720：BGR 原始像素，等比例缩放后居中补边。均值 [104,117,123] 已编译进模型，Python 不重复归一化。解码 boxes/scores/landmarks 三个输出，再撤销补边和缩放。

识别输入为 NCHW 1×3×112×112：RGB 原始像素，模型内完成均值/标准差 127.5 的归一化。输出 512 维特征再做 L2 归一化。当前文件内记录的主机输入接口是 float32；模型名称中的 int8 表示量化，不能据此把主机接口强行改成 int8。

对齐网络输入是 112×112；几何变换沿用 ONNX 基线的 160×160 坐标系，最后得到 112×112 识别图像。配置默认 rknn_aligner_fallback=onnx-cpu，在此模式下不加载 aligner-int8.rknn。

models/rknn/manifest.json 包含模型 SHA-256、主机输入输出属性、预处理契约与 ONNX 对齐文件校验。路径相对 manifest 所在目录解析。它由随附 RKNN 文件的嵌入导出元数据生成，target_verified=false 表示尚未完成实机验收。替换模型时应重新生成并在板端验证：

```bash
python scripts/prepare_rknn_manifest.py
```

## RK3588 原生部署

在 Linux aarch64 上使用 Python 3.11（也支持 3.12）。安装 rknn-toolkit-lite2==2.3.2，并由板端镜像提供匹配的 librknnrt.so 2.3.2 与可用 NPU 驱动。参考项目历史日志中的驱动版本为 0.9.8，实际版本须以板端日志为准。官方发布：[RKNN Toolkit2 v2.3.2](https://github.com/airockchip/rknn-toolkit2/tree/v2.3.2)、[Lite2 2.3.2 ARM64 wheel](https://pypi.org/project/rknn-toolkit-lite2/2.3.2/)。

从项目根目录执行：

```bash
uv sync --locked --extra rknn
.venv/bin/python scripts/check_rknn_device.py \
  --model models/rknn/retinaface-mobilenet0.25-480x720-int8.rknn --core-mask 0
.venv/bin/python scripts/check_rknn_device.py \
  --model models/rknn/recognition-int8.rknn --core-mask 1
```

预检默认检查 /usr/lib/librknnrt.so、/dev/dri/renderD129、进程的设备读写权限，并执行一次模型推理。其他设备节点/库路径的板卡应按实际部署位置调整；仅安装 Python wheel 不会安装内核驱动。

启动服务：

```bash
.venv/bin/python -m face_api.main \
  --runtime rknn \
  --retinaface-rknn models/rknn/retinaface-mobilenet0.25-480x720-int8.rknn \
  --cvlface-recognition-rknn models/rknn/recognition-int8.rknn \
  --cvlface-aligner-onnx models/onnx/cvlface_dfa_mobilenet.onnx \
  --rknn-manifest models/rknn/manifest.json \
  --rknn-aligner-fallback onnx-cpu \
  --rknn-detector-core-mask 0 --rknn-recognition-core-mask 1 \
  --data-dir data/face_api_rknn \
  --video-analysis-recognition-interval 0.2 \
  --video-analysis-recognition-queue-size 64 \
  --performance-log-enabled \
  --performance-log-file logs/rknn_performance.jsonl \
  --host 0.0.0.0 --port 8002
```

性能 JSONL 同时写文件和标准输出；用 --no-performance-log-stdout 关闭终端副本。REST/视频/WebSocket 的阶段耗时、端到端耗时和 FPS 沿用现有日志字段。API 文档位于 http://localhost:8002/docs。推理会话串行保护，服务退出时先等待视频队列完成，再释放内部创建的 NPU 会话。

命令行演示也默认使用同一混合推理链，可运行 python run_demo.py --help 查看输入、图库和输出参数。--runtime onnx 切换检测及识别到 ONNX；仅在显式提供 .pth 检测模型时需要 uv sync --extra pytorch。部署默认依赖不加载 torch/CUDA。

## 特征库迁移

特征按 runtime、识别模型 SHA-256 和对齐流程 SHA-256 隔离。旧记录自动增加 legacy 流程标记，不会参与新流程匹配。需用原始登记图像重新生成，不能复用旧 ONNX 或旧几何变换流程的特征。

以下命令会写入新目录，目标数据库已存在时拒绝覆盖。执行前确保源目录是实际已有的人脸库：

```bash
.venv/bin/python scripts/rebuild_gallery_features.py \
  --source-data-dir data/face_api \
  --target-data-dir data/face_api_rknn \
  --runtime rknn --rknn-aligner-fallback onnx-cpu
```

新库生成后，将服务 --data-dir 指向该目录。重建失败时应检查原始图像和模型错误，保留旧库；不要把不完整新库用于验收。SQLite 会对旧库添加缺失的身份列，原记录和原始图像保留。

## Docker

默认 Compose 用于 Windows/WSL 或普通 x86_64 Linux 上的 ONNX CPU 服务，外部端口为 8002：

```bash
docker compose up --build -d
docker compose ps
curl --fail http://127.0.0.1:8002/readyz
docker compose logs -f face-api-onnx
```

`Dockerfile` 使用 `uv.lock` 安装基础运行依赖，不安装 Toolkit2、Lite2、CUDA 或 PyTorch。模型从只读 `models` 卷加载，ONNX 数据与日志分别保存在 `runtime/data-onnx` 和 `runtime/logs`。

RK3588 使用独立的 ARM64 配置；不要在 Windows/WSL 中启动该服务，因为 Docker Desktop 没有板端 `librknnrt.so`、RKNPU 驱动或 `/dev/dri/renderD129`：

```bash
export RKNN_RENDER_GID="$(getent group render | cut -d: -f3)"
export RKNN_RUNTIME_LIBRARY=/usr/lib/librknnrt.so
export RKNN_RENDER_DEVICE=/dev/dri/renderD129
docker compose -f docker-compose.rknn.yml config
docker compose -f docker-compose.rknn.yml up --build -d
curl --fail http://127.0.0.1:8001/readyz
docker compose -f docker-compose.rknn.yml logs -f face-api-rknn
```

`Dockerfile.rknn` 固定构建 `linux/arm64`，从同一 `uv.lock` 安装 Lite2 2.3.2，并以非 root 用户运行。Compose 只映射 NPU render 节点及只读板端运行库，不使用 `privileged`。USB 摄像头需要另外映射实际 `/dev/videoN`；已有原生重建库应放入 `runtime/data-rknn`。Windows 的 ARM64 构建只能验证镜像可构建，不能替代 RK3588 实机推理与性能验收。

## 实机一致性与性能验收

准备 images.txt，每行一个绝对图片路径。用相同原始登记图片分别重建 ONNX 基线库 data/face_api_onnx 和混合库 data/face_api_rknn，不能使用空库替代匹配验收：

```bash
.venv/bin/python scripts/rebuild_gallery_features.py \
  --source-data-dir data/face_api --target-data-dir data/face_api_onnx --runtime onnx
.venv/bin/python scripts/validate_rknn_parity.py \
  --images images.txt --report outputs/rknn_parity.json \
  --onnx-gallery-data-dir data/face_api_onnx \
  --rknn-gallery-data-dir data/face_api_rknn
.venv/bin/python scripts/benchmark_rknn_service.py \
  --images images.txt --report outputs/rknn_benchmark.json \
  --gallery-data-dir data/face_api_rknn --runtime rknn --samples 300
```

报告记录活动模型与哈希。检查框位置、检测数、对齐结果、同人与异人分数和匹配结果，再按相同输入比较 ONNX/RKNN 的 p50/p95。脚本的 recognition_ms 包含检测后的人脸处理与匹配，不代表纯 NPU 算子耗时。还需通过 REST/长视频/WebSocket 实测队列、持续运行、断流重连和内存/NPU 错误日志。

## 本地验证

```bash
uv sync --locked --extra conversion
uv run --no-sync pytest tests/test_retinaface_rknn.py tests/test_rknn_runtime.py tests/test_cvlface_rknn.py -q
```

完整测试包含显式 PyTorch 后端的用例，需额外安装 pytorch 可选依赖。CPU 测试覆盖实际 ONNX 对齐模型、混合链与基线几何一致性、模拟 RKNN 输入输出、检测框/关键点反变换、特征库身份隔离、日志及资源释放；这些测试不替代板端 RKNN 推理与量化精度验证。

# RK3588 RKNN/NPU 部署

本说明适用于已经验收的 RK3588：Ubuntu 22.04.5、aarch64、Linux 6.1.141、RKNPU driver 0.9.8、`librknnrt.so`/`rknn_server` 2.3.x。转换机和盒子运行时必须分离：转换机使用 Toolkit2；盒子和容器仅使用 Lite2 2.3.2 与宿主机 Runtime。

## 1. 在盒子宿主机准备 Lite2

先确认解释器 ABI；本镜像和 wheel 固定为 CPython 3.11/aarch64。不是 3.11 时不要安装该 wheel。

```bash
python3 --version
uname -m
```

预期分别为 `Python 3.11.x` 和 `aarch64`。推荐直接使用锁文件创建环境：

```bash
uv sync --locked --extra rknn
```

没有 uv 时可使用审核过的 requirements；Lite2 文件包含官方 CPython 3.11/aarch64 wheel 的 SHA-256：

```bash
python3 -m pip install --no-cache-dir -r requirements-rknn-runtime.txt
python3 -m pip install --no-deps --require-hashes -r requirements-rknn.txt
```

`requirements-rknn-runtime.txt` 是精确 CPU 运行依赖；它不包含 CUDA、`onnxruntime-gpu`、PyTorch、Ultralytics 或完整 Toolkit2。即使不使用 `onnx-cpu` 对齐回退，也保留 CPU `onnxruntime`，以便配置切换时维持确定的 CPU-only 行为。若 Python 或架构不是 CPython 3.11/aarch64，立刻停止；不要用强制安装绕过 ABI 检查，也不要在盒子安装完整 Toolkit2。

确认驱动、运行库和仅有的 NPU render 节点：

```bash
sudo cat /sys/kernel/debug/rknpu/version
getent group render
ls -l /usr/lib/librknnrt.so /dev/dri/renderD129
systemctl status rknn_server.service --no-pager
```

不要在容器中启动第二个 `rknn_server`；盒子厂家安装的服务保持由宿主机管理。

## 2. 宿主机先验收

将经 Task 7–8 生成、并通过目标机 C metadata probe/manifest finalize 的三个 `.rknn` 文件与 `manifest.json` 放入 `models/rknn/`。路径和 SHA-256 必须与 manifest 相符。

```bash
python3 scripts/check_rknn_device.py \
  --model models/rknn/retinaface-mobilenet0.25-480x720-int8.rknn \
  --manifest models/rknn/manifest.json --core-mask 0

python3 -m face_api.main \
  --runtime rknn --host 127.0.0.1 --port 8001 \
  --data-dir /srv/face-api-rknn/data \
  --retinaface-rknn models/rknn/retinaface-mobilenet0.25-480x720-int8.rknn \
  --cvlface-recognition-rknn models/rknn/recognition-int8.rknn \
  --cvlface-aligner-rknn models/rknn/aligner-int8.rknn \
  --rknn-manifest models/rknn/manifest.json \
  --rknn-aligner-fallback error \
  --live-stream-url 'rtsp://USER:PASSWORD@CAMERA/STREAM'
curl --fail http://127.0.0.1:8001/readyz
```

`check_rknn_device.py` fails before Lite2 initialization when the Runtime library, `renderD129`, Lite2 module, or model is absent; it only succeeds after an actual zero-input inference. The ready response must report `runtime: "rknn"`. Do not proceed to Docker if this host test fails.

For the documented CPU aligner fallback, replace `--rknn-aligner-fallback error` with `onnx-cpu`, supply the matching ONNX aligner path, and use a manifest whose `runtime.aligner_runtime` is `onnx-cpu`; the service rejects any path/SHA mismatch.

## 3. 最小权限容器

Docker Compose v2 with BuildKit is required. `Dockerfile.rknn` 使用 `uv.lock` 中固定版本和哈希的官方 ARM64 Lite2 wheel；完整 Toolkit2 不会进入运行镜像。

```bash
export RKNN_RENDER_GID="$(getent group render | cut -d: -f3)"
export FACE_API_LIVE_STREAM_URL='rtsp://USER:PASSWORD@CAMERA/STREAM'
export FACE_API_RKNN_ALIGNER_FALLBACK=error
export FACE_API_RKNN_HOST_PORT=8001
docker compose -f docker-compose.rknn.yml config
docker compose -f docker-compose.rknn.yml up -d --build
curl --fail http://127.0.0.1:8001/readyz
docker compose -f docker-compose.rknn.yml exec face-api-rknn id
```

`id` 必须显示传入的 `render` GID。该服务只映射 `/dev/dri/renderD129` 和只读 `librknnrt.so`，以非 root `faceapi` 用户运行；不使用 `privileged`、NVIDIA/CUDA 配置或容器内的 `rknn_server`。模型卷只读，数据和日志卷可写。

## 4. 常见故障

| 现象 | 处理 |
| --- | --- |
| `RKNN runtime library is missing` | 先恢复厂商 `/usr/lib/librknnrt.so`，不要把不同版本库复制进镜像。 |
| `renderD129` 不存在或无权限 | 验证主机 NPU/驱动，再核对 Compose 的设备映射和 `RKNN_RENDER_GID`。不要以 `privileged` 绕过。 |
| `rknnlite.api is unavailable` | 重新确认 Python 3.11/aarch64 wheel 标签，使用 `--no-deps` 重新安装官方 Lite2 2.3.2 wheel。 |
| manifest 或模型 SHA 不匹配 | 回到目标机 C probe/finalize 步骤重新生成 manifest；不要手改 digest。 |

正式发布的 10 分钟性能、24 小时 canary、证据保存与回滚步骤见后续发布验收章节；这些硬件命令必须在目标 RK3588 上执行，本文档不把本机单元测试视作设备验收。

## 5. 发布证据、Canary 与回滚（仅在目标 RK3588 执行）

以下步骤是操作清单，不是本仓库已经完成的硬件结果。证据目录必须在仓库外，且应由
受控存储备份。先保留 ONNX 服务于 `8000`，RKNN Compose 默认仅发布到 `8001`：

```bash
export EVIDENCE_DIR="/secure/reports/rk3588-$(date +%F)"
mkdir -p "$EVIDENCE_DIR"
export FACE_API_RKNN_HOST_PORT=8001
docker compose -f docker-compose.rknn.yml up -d --build
curl --fail http://127.0.0.1:8001/readyz | tee "$EVIDENCE_DIR/readyz-initial.json"
```

先用同一 RK3588、同一图像清单和对应 runtime 的已重建 gallery 写入 ONNX-CPU
基线，再进行至少 300 个 warm-up 之后样本的 RKNN 测量。`--onnx-baseline-report`
将已解析的基线报告路径、SHA-256 和 `end_to_end_ms.p95` 写入 RKNN 报告的
`onnx_baseline` 字段，并计算改善比例；
`--require-release-evidence` 要求 RKNN、至少 300 样本、三项 p95、零 NPU 推理错误
以及端到端 p95 改善至少 50%。将服务保持运行至少十分钟，并在该窗口结束后保存日志：

```bash
python scripts/benchmark_rknn_service.py \
  --runtime onnx --images /secure/validation/list.txt \
  --gallery-data-dir /secure/gallery/onnx --provider cpu \
  --report "$EVIDENCE_DIR/benchmark-onnx-cpu.json"

python scripts/benchmark_rknn_service.py \
  --runtime rknn --images /secure/validation/list.txt \
  --gallery-data-dir /secure/gallery/rknn \
  --rknn-manifest /opt/face-serve/models/rknn/manifest.json \
  --yolo-rknn /opt/face-serve/models/rknn/yolo-int8.rknn \
  --recognition-rknn /opt/face-serve/models/rknn/recognition-int8.rknn \
  --aligner-rknn /opt/face-serve/models/rknn/aligner-int8.rknn \
  --onnx-baseline-report "$EVIDENCE_DIR/benchmark-onnx-cpu.json" \
  --require-release-evidence --report "$EVIDENCE_DIR/benchmark-rknn.json"

docker compose -f docker-compose.rknn.yml logs --since 10m face-api-rknn \
  > "$EVIDENCE_DIR/service-10m.log"
python scripts/validate_rknn_parity.py --images /secure/validation/list.txt \
  --onnx-gallery-data-dir /secure/gallery/onnx \
  --rknn-gallery-data-dir /secure/gallery/rknn \
  --report "$EVIDENCE_DIR/parity-report.json"
curl --fail http://127.0.0.1:8001/readyz > "$EVIDENCE_DIR/readyz-after-benchmark.json"
sudo cat /sys/kernel/debug/rknpu/version > "$EVIDENCE_DIR/rknpu-version.txt"
ldconfig -p | grep librknnrt > "$EVIDENCE_DIR/rknnrt-library.txt"
sha256sum /opt/face-serve/models/rknn/*.rknn > "$EVIDENCE_DIR/model-sha256.txt"
```

确认 parity 和 release gate 都返回 0 后，使用只读 RTSP client/stream 对 `8001`
运行 24 小时 canary。每分钟采集 `/readyz`，保留服务日志、健康检查输出以及读取的
镜像标签；未满 24 小时或任何 NPU/服务错误均不得切流：

```bash
for minute in $(seq 1 1440); do
  date -Is >> "$EVIDENCE_DIR/readyz-24h.log"
  curl --fail --silent http://127.0.0.1:8001/readyz >> "$EVIDENCE_DIR/readyz-24h.log" || exit 1
  sleep 60
done
docker compose -f docker-compose.rknn.yml logs face-api-rknn > "$EVIDENCE_DIR/service-24h.log"
docker inspect --format '{{.Config.Image}} {{.Image}}' face-api-rknn > "$EVIDENCE_DIR/rknn-image.txt"
```

在重新提取 gallery 特征之前先备份数据库。只在全部 canary 证据通过后把上游流量
切到 `8001`；第一轮完整发布周期中不要删除 `.rknn`、ONNX 镜像或数据库备份。完成
切流前应演练下面的回滚；它只切换服务运行时，不删除数据。记录执行时间、两个镜像
标签和 `/readyz` 响应到 `$EVIDENCE_DIR/rollback-record.txt`：

```bash
docker compose -f docker-compose.rknn.yml stop face-api-rknn
docker compose -f docker-compose.yml up -d face-api
{
  echo "rollback_at=$(date -Is)"
  echo "rknn_image=$(docker inspect --format '{{.Config.Image}} {{.Image}}' face-api-rknn)"
  echo "onnx_image=$(docker inspect --format '{{.Config.Image}} {{.Image}}' face-api)"
  echo "readyz_response="
  curl --fail --silent http://127.0.0.1:8000/readyz
} | tee "$EVIDENCE_DIR/rollback-record.txt"
```

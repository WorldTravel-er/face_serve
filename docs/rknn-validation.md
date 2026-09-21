# RK3588 RKNN 验收与性能门槛

转换完成后，先把每个 `.rknn` 拷到 RK3588，使用目标机的
`probe_rknn_tensor_metadata` 生成元数据，再回到转换主机执行
`convert_rknn_models.py --finalize --metadata-probe ...`。转换器的实际文件名固定为
`yolo-int8.rknn`、`recognition-int8.rknn`、`aligner-int8.rknn`（或 `-fp`），而不是
模型的 ONNX 文件名。只有这个两阶段流程完成、`manifest.json` 已包含真实目标张量
契约与根级 `runtime.aligner_runtime`，产物才能部署或验收。

例如，全部使用 RKNN aligner 时，对每个已探测的产物执行（替换 probe 文件）：

```bash
python scripts/convert_rknn_models.py --finalize --model yolo --precision int8 \
  --output-dir /secure/build/rknn --metadata-probe /secure/probes/yolo-int8.json \
  --aligner-runtime rknn
python scripts/convert_rknn_models.py --finalize --model recognition --precision int8 \
  --output-dir /secure/build/rknn --metadata-probe /secure/probes/recognition-int8.json \
  --aligner-runtime rknn
python scripts/convert_rknn_models.py --finalize --model aligner --precision int8 \
  --output-dir /secure/build/rknn --metadata-probe /secure/probes/aligner-int8.json \
  --aligner-runtime rknn
```

若明确采用 CPU fallback，最后一次 finalization 使用
`--aligner-runtime onnx-cpu --aligner-onnx-path <已部署同一路径的 ONNX aligner>`。
该路径与 SHA-256 写入 manifest；验证和服务启动都会拒绝与 manifest 不一致的
fallback 选择。

## 一致性门槛

校准集（500 张）只用于 INT8 转换；验证集必须是独立的 200 张图像。为 ONNX
和 RKNN 分别重建 gallery（两者的 embedding identity 不同，不能混用），然后在
RK3588 上运行：

```bash
python scripts/validate_rknn_parity.py \
  --images /secure/validation/list.txt \
  --onnx-gallery-data-dir /secure/gallery/onnx \
  --rknn-gallery-data-dir /secure/gallery/rknn \
  --report /secure/reports/parity-report.json \
  --provider cpu \
  --rknn-manifest /opt/face-serve/models/rknn/manifest.json \
  --yolo-rknn /opt/face-serve/models/rknn/yolo-int8.rknn \
  --recognition-rknn /opt/face-serve/models/rknn/recognition-int8.rknn \
  --aligner-rknn /opt/face-serve/models/rknn/aligner-int8.rknn
```

报告保存每个失败项的图像路径、ONNX/RKNN 检测框、IoU、特征余弦相似度、两个
top-1 subject ID，以及全部 ONNX/RKNN 模型的 SHA-256。三个比率各自有分母：

| 指标 | 合格分子 / 分母 | 门槛 |
| --- | --- | --- |
| `detection_iou_rate` | ONNX score ≥ 0.30 的图中，RKNN/ONNX IoU ≥ 0.90 | 0.98 |
| `feature_cosine_rate` | 两条路径均可抽特征的图中，余弦 ≥ 0.98 | 0.95 |
| `top1_match_rate` | 两条路径均可抽特征的图中，top-1 subject ID 相同且非空 | 0.98 |

没有任何符合某指标资格的图像时，该指标是 `0.0`，不会被当成满分。任何解码、
检测、特征或匹配推理异常都会作为 `inference_failure` 写入报告，并使命令以非零
状态退出；不得从报告中省略不一致项。

若 aligner 的 FP 或 INT8 RKNN 路径构建失败或未通过此门槛，在生产 manifest 明确
记录 `runtime.aligner_runtime: onnx-cpu` 以及 ONNX aligner 路径/SHA（通过上面的
`--finalize` 参数），然后使用同一 ONNX 文件执行：

```bash
python scripts/validate_rknn_parity.py ... \
  --rknn-aligner-fallback onnx-cpu \
  --aligner-onnx /opt/face-serve/models/onnx/cvlface_dfa_mobilenet.onnx \
  --yolo-rknn /opt/face-serve/models/rknn/yolo-int8.rknn \
  --recognition-rknn /opt/face-serve/models/rknn/recognition-int8.rknn
```

此模式不加载、不校验也不报告未使用的 RKNN aligner；禁止静默退回 CPU。

## 性能门槛

先在同一台 RK3588、同一验证图集上记录 ONNX-CPU 基线：

```bash
python scripts/benchmark_rknn_service.py \
  --runtime onnx --images /secure/validation/list.txt \
  --gallery-data-dir /secure/gallery/onnx \
  --report /secure/reports/benchmark-onnx-cpu.json --provider cpu
```

随后以对应 RKNN gallery 测量 RKNN：

```bash
python scripts/benchmark_rknn_service.py \
  --runtime rknn --images /secure/validation/list.txt \
  --gallery-data-dir /secure/gallery/rknn \
  --report /secure/reports/benchmark-rknn.json \
  --rknn-manifest /opt/face-serve/models/rknn/manifest.json \
  --yolo-rknn /opt/face-serve/models/rknn/yolo-int8.rknn \
  --recognition-rknn /opt/face-serve/models/rknn/recognition-int8.rknn \
  --aligner-rknn /opt/face-serve/models/rknn/aligner-int8.rknn
```

每个命令固定丢弃 10 次 warm-up，并要求至少 300 个后续样本。脚本用
`perf_counter_ns()` 分别测量 `detect_frame`、
`extract_feature_from_detection` 与 `recognize_detection`；端到端延迟另行测量
`detect + recognize_detection`，因此不把 recognition 内部的特征提取重复相加。
报告中 p50、p95 和 FPS 都是数值毫秒/帧率。验收要求 RKNN 的端到端 p95 比
ONNX-CPU 基线低至少 50%，10 分钟基准无错误，并完成 24 小时无错误稳定性运行。

这两项命令是部署外部验收门，代码库不会伪造任何 RK3588 硬件结果。

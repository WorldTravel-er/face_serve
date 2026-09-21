**在 RK3588 上生成 probe JSON**

分别 probe 三个模型：

```
./probe_rknn_tensor_metadata models/rknn/yolo-int8.rknn \
  > yolo-int8.probe.json

./probe_rknn_tensor_metadata models/rknn/recognition-int8.rknn \
  > recognition-int8.probe.json

./probe_rknn_tensor_metadata models/rknn/aligner-int8.rknn \
  > aligner-int8.probe.json
```

生成的 JSON 大概长这样：

```
{
  "rknn_path": "models/rknn/yolo-int8.rknn",
  "rknn_sha256": "...",
  "inputs": [
    {
      "name": "...",
      "layout": "nchw",
      "shape": [1, 3, 640, 640],
      "dtype": "uint8"
    }
  ],
  "outputs": [
    {
      "name": "...",
      "layout": "undefined",
      "shape": [1, 5, 8400],
      "dtype": "float32"
    }
  ]
}
```

 **finalize 写入 manifest.json**

在项目根目录执行：

```
python scripts/convert_rknn_models.py \
  --model yolo \
  --precision int8 \
  --output-dir models/rknn \
  --finalize \
  --metadata-probe models/rknn/probes/yolo-int8.probe.json
python scripts/convert_rknn_models.py \
  --model recognition \
  --precision int8 \
  --output-dir models/rknn \
  --finalize \
  --metadata-probe models/rknn/probes/recognition-int8.probe.json
python scripts/convert_rknn_models.py \
  --model aligner \
  --precision int8 \
  --output-dir models/rknn \
  --finalize \
  --metadata-probe models/rknn/probes/aligner-int8.probe.json \
  --aligner-runtime rknn
```

执行完成后会生成或更新：

```
models/rknn/manifest.json
```





**aligner 用 ONNX CPU fallback**

也就是：

```
YOLO:        RKNN
Recognition: RKNN
Aligner:     ONNX Runtime CPU
```

这正好适合 `GridSample` 这种 NPU/runtime 不支持的算子。

接下来你不要再 probe 这个 aligner RKNN，先只 probe 能成功加载的模型：

```
./probe_rknn_tensor_metadata models/rknn/yolo-int8.rknn \
  > yolo-int8.probe.json

./probe_rknn_tensor_metadata models/rknn/recognition-int8.rknn \
  > recognition-int8.probe.json
```

finalize：

```
uv run python scripts/convert_rknn_models.py \
  --model yolo \
  --precision int8 \
  --output-dir models/rknn \
  --finalize \
  --metadata-probe models/rknn/probes/yolo-int8.probe.json \
  --aligner-runtime onnx-cpu \
  --aligner-onnx-path models/onnx/cvlface_dfa_mobilenet.onnx
uv run python scripts/convert_rknn_models.py \
  --model recognition \
  --precision int8 \
  --output-dir models/rknn \
  --finalize \
  --metadata-probe models/rknn/probes/recognition-int8.probe.json \
  --aligner-runtime onnx-cpu \
  --aligner-onnx-path models/onnx/cvlface_dfa_mobilenet.onnx
```

**如果 aligner 是 onnx-cpu fallback，推荐这样启动**

结合你前面 `GridSample` 报错，当前最现实的运行方式是：

```
YOLO 检测：RKNN/NPU
人脸识别：RKNN/NPU
人脸对齐：ONNX CPU
```

启动命令：

```
uv run python face_core/cli.py \
  --runtime rknn \
  --rknn-manifest models/rknn/manifest.json \
  --rknn-aligner-fallback onnx-cpu \
  --yolo-face-rknn models/rknn/yolo-int8.rknn \
  --cvlface-recognition-rknn models/rknn/recognition-int8.rknn \
  --cvlface-aligner-onnx models/onnx/cvlface_dfa_mobilenet.onnx \
  --video data/videos/exo.mp4 \
  --gallery data/gallery \
  --output outputs/rknn_test \
  --sample-fps 2
```

**如果要跑摄像头**

```
uv run python face_core/cli.py \
  --runtime rknn \
  --rknn-manifest models/rknn/manifest.json \
  --rknn-aligner-fallback onnx-cpu \
  --yolo-face-rknn models/rknn/yolo-int8.rknn \
  --cvlface-recognition-rknn models/rknn/recognition-int8.rknn \
  --cvlface-aligner-onnx models/onnx/cvlface_dfa_mobilenet.onnx \
  --camera 0 \
  --camera-fps 8 \
  --max-seconds 30 \
  --gallery data/gallery \
  --output outputs/rknn_camera_test
```

```
uv run python -m face_api.main \
  --runtime rknn \
  --host 0.0.0.0 \
  --port 8000 \
  --data-dir data/face_api \
  --rknn-manifest models/rknnfp/manifest.json \
  --yolo-face-rknn models/rknnfp/yolo-fp.rknn \
  --cvlface-recognition-rknn models/rknnfp/recognition-fp.rknn \
  --rknn-aligner-fallback onnx-cpu \
  --cvlface-aligner-onnx models/onnx/cvlface_dfa_mobilenet.onnx
```

```
uv run python -m face_api.main \
  --runtime rknn \
  --host 127.0.0.1 \
  --port 8000 \
  --data-dir data/face_api \
  --rknn-manifest models/rknnfp/manifest.json \
  --yolo-face-rknn models/rknnfp/yolo-fp.rknn \
  --cvlface-recognition-rknn models/rknnfp/recognition-fp.rknn \
  --rknn-yolo-core-mask 0 \
  --rknn-recognition-core-mask 1 \
  --rknn-aligner-fallback onnx-cpu \
  --cvlface-aligner-onnx models/onnx/cvlface_dfa_mobilenet.onnx \
  --video-analysis-recognition-interval 0.05 \
  --video-analysis-recognition-queue-size 256
```

请将在rk3588上进行rest_api，video_analysis和websocket连接时对人脸检测，人脸对齐和人脸识别的延迟时间做log记录，给出规划
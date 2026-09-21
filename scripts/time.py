#!/usr/bin/env python3
"""
视频人脸检测与识别性能测试脚本
使用YOLOv12n进行人脸检测，使用AdaFace进行人脸识别
支持MP4视频文件分析
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
from collections import deque
import sys
import os


class YOLOv12FaceDetector:
    """YOLOv12人脸检测器 - 专门用于人脸检测模型"""

    def __init__(self, model_path, input_size=640, conf_threshold=0.5, iou_threshold=0.4):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 创建ONNX推理会话
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider', 'CUDAExecutionProvider']
        )

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 获取输入形状
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4:
            self.batch_size = input_shape[0]
            self.input_height = input_shape[2] if input_shape[2] else input_size
            self.input_width = input_shape[3] if input_shape[3] else input_size
        else:
            self.input_height = input_size
            self.input_width = input_size

        # 打印模型信息以便调试
        print(f"YOLO模型输入: {self.input_name}, 形状: {input_shape}, 类型: {self.session.get_inputs()[0].type}")
        print(f"YOLO模型输出: {self.output_name}, 形状: {self.session.get_outputs()[0].shape}")

    def preprocess(self, image):
        """预处理图像"""
        # 调整大小
        resized = cv2.resize(image, (self.input_width, self.input_height))
        # 转换为RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 归一化并转换为NCHW格式，确保是float32类型
        input_data = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        return input_data

    def postprocess(self, outputs, original_shape):
        """后处理，提取人脸边界框"""
        # 处理不同的输出格式
        if isinstance(outputs, list):
            output = outputs[0]
        else:
            output = outputs

        # 检查输出形状
        if len(output.shape) == 3:
            # 形状: [batch, num_boxes, features]
            num_boxes = output.shape[1]
            num_features = output.shape[2]

            boxes = []
            orig_h, orig_w = original_shape[:2]

            for i in range(num_boxes):
                box_data = output[0, i, :]

                # 根据特征数量解析
                if num_features == 5:
                    # 人脸检测模型: [x1, y1, x2, y2, conf]
                    x1, y1, x2, y2, conf = box_data
                elif num_features == 6:
                    # 标准YOLO: [x1, y1, x2, y2, conf, class_id]
                    x1, y1, x2, y2, conf, class_id = box_data
                    # 对于人脸检测，我们接受所有检测（或者可以检查class_id）
                elif num_features == 7:
                    # 某些模型: [x1, y1, x2, y2, conf, class_id, extra]
                    x1, y1, x2, y2, conf, class_id, _ = box_data
                else:
                    # 尝试前5个值
                    try:
                        x1, y1, x2, y2, conf = box_data[:5]
                    except:
                        continue

                if conf < self.conf_threshold:
                    continue

                # 转换坐标到原始图像尺寸
                x1 = int(x1 * orig_w / self.input_width)
                y1 = int(y1 * orig_h / self.input_height)
                x2 = int(x2 * orig_w / self.input_width)
                y2 = int(y2 * orig_h / self.input_height)

                # 确保坐标有效
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))

                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2, conf))

            # 应用NMS
            if boxes:
                boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
                nms_boxes = []
                for box in boxes:
                    if not nms_boxes:
                        nms_boxes.append(box)
                    else:
                        keep = True
                        for kept_box in nms_boxes:
                            if self._compute_iou(box, kept_box) > self.iou_threshold:
                                keep = False
                                break
                        if keep:
                            nms_boxes.append(box)
                return nms_boxes

        return []

    def _compute_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def detect(self, image):
        """执行检测"""
        # 预处理
        input_tensor = self.preprocess(image)

        # 推理
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = time.time() - start_time

        # 后处理
        boxes = self.postprocess(outputs, image.shape)

        return boxes, inference_time


class AdaFaceRecognizer:
    """AdaFace人脸识别器"""

    def __init__(self, model_path, input_size=112):
        self.model_path = model_path
        self.input_size = input_size

        # 创建ONNX推理会话
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider', 'CUDAExecutionProvider']
        )

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 获取输入形状
        input_shape = self.session.get_inputs()[0].shape
        print(f"IR50模型输入: {self.input_name}, 形状: {input_shape}, 类型: {self.session.get_inputs()[0].type}")
        print(f"IR50模型输出: {self.output_name}, 形状: {self.session.get_outputs()[0].shape}")

    def preprocess(self, face_image):
        """预处理人脸图像"""
        # 调整大小为112x112
        resized = cv2.resize(face_image, (self.input_size, self.input_size))
        # 转换为RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 归一化并转换为NCHW格式，确保是float32类型
        input_data = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0

        # 标准化 (使用ImageNet的均值和标准差)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        input_data = (input_data - mean) / std

        # 添加batch维度
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        return input_data

    def extract_feature(self, face_image):
        """提取人脸特征"""
        if face_image.size == 0:
            return None, 0

        # 预处理
        input_tensor = self.preprocess(face_image)

        # 推理
        start_time = time.time()
        features = self.session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = time.time() - start_time

        # 返回特征向量并归一化
        feature = features[0][0]
        feature = feature / np.linalg.norm(feature)

        return feature, inference_time


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.yolo_times = deque(maxlen=window_size)
        self.ir50_times = deque(maxlen=window_size)
        self.frame_times = deque(maxlen=window_size)
        self.frame_count = 0
        self.total_yolo_time = 0
        self.total_ir50_time = 0
        self.yolo_count = 0
        self.ir50_count = 0
        self.total_faces = 0

    def update_yolo(self, inference_time):
        self.yolo_times.append(inference_time * 1000)  # 转换为毫秒
        self.total_yolo_time += inference_time * 1000
        self.yolo_count += 1

    def update_ir50(self, inference_time):
        self.ir50_times.append(inference_time * 1000)
        self.total_ir50_time += inference_time * 1000
        self.ir50_count += 1

    def update_frame(self, frame_time, num_faces=0):
        self.frame_times.append(frame_time * 1000)
        self.frame_count += 1
        self.total_faces += num_faces

    def get_stats(self):
        stats = {}

        if self.yolo_times:
            stats['yolo'] = {
                'avg': np.mean(self.yolo_times),
                'max': np.max(self.yolo_times),
                'min': np.min(self.yolo_times),
                'std': np.std(self.yolo_times),
                'count': self.yolo_count,
                'total': self.total_yolo_time
            }

        if self.ir50_times:
            stats['ir50'] = {
                'avg': np.mean(self.ir50_times),
                'max': np.max(self.ir50_times),
                'min': np.min(self.ir50_times),
                'std': np.std(self.ir50_times),
                'count': self.ir50_count,
                'total': self.total_ir50_time
            }

        if self.frame_times:
            stats['frame'] = {
                'avg': np.mean(self.frame_times),
                'max': np.max(self.frame_times),
                'min': np.min(self.frame_times),
                'std': np.std(self.frame_times),
                'fps': 1000 / np.mean(self.frame_times) if np.mean(self.frame_times) > 0 else 0,
                'count': self.frame_count,
                'total_faces': self.total_faces
            }

        return stats


def get_video_info(cap):
    """获取视频信息"""
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(
            cv2.CAP_PROP_FPS) > 0 else 0
    }
    return info


def main():
    parser = argparse.ArgumentParser(description='视频人脸检测与识别性能测试')
    parser.add_argument('--video-file', type=str, required=True, help='输入视频文件路径 (MP4格式)')
    parser.add_argument('--yolo-model', type=str, default='models/onnx/yolov12n-face.onnx', help='YOLOv12模型路径')
    parser.add_argument('--ir50-model', type=str, default='models/onnx/cvlface_adaface_ir50_webface4m.onnx',
                        help='AdaFace IR50模型路径')
    parser.add_argument('--yolo-input-size', type=int, default=640, help='YOLO输入尺寸')
    parser.add_argument('--ir50-input-size', type=int, default=112, help='IR50输入尺寸')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='YOLO置信度阈值')
    parser.add_argument('--display', action='store_true', help='是否显示视频窗口')
    parser.add_argument('--max-frames', type=int, default=0, help='处理的最大帧数 (0表示全部)')
    parser.add_argument('--output-video', type=str, default='', help='输出视频文件路径 (可选)')
    parser.add_argument('--skip-frames', type=int, default=0, help='跳过的帧数 (用于加速处理)')

    args = parser.parse_args()

    # 检查视频文件是否存在
    if not os.path.exists(args.video_file):
        print(f"错误: 视频文件不存在: {args.video_file}")
        sys.exit(1)

    print("=" * 70)
    print("视频人脸检测与识别性能测试")
    print("=" * 70)
    print(f"视频文件: {args.video_file}")
    print(f"YOLO模型: {args.yolo_model}")
    print(f"IR50模型: {args.ir50_model}")
    print(f"YOLO输入尺寸: {args.yolo_input_size}x{args.yolo_input_size}")
    print(f"IR50输入尺寸: {args.ir50_input_size}x{args.ir50_input_size}")
    print("=" * 70)

    # 初始化视频读取
    cap = cv2.VideoCapture(args.video_file)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {args.video_file}")
        sys.exit(1)

    # 获取视频信息
    video_info = get_video_info(cap)
    print(f"\n视频信息:")
    print(f"  分辨率: {video_info['width']}x{video_info['height']}")
    print(f"  帧率: {video_info['fps']:.2f} FPS")
    print(f"  总帧数: {video_info['total_frames']}")
    print(f"  时长: {video_info['duration']:.2f} 秒")
    print("-" * 70)

    # 初始化模型
    try:
        yolo_detector = YOLOv12FaceDetector(
            args.yolo_model,
            args.yolo_input_size,
            args.conf_threshold
        )
        print("✓ YOLOv12模型加载成功")
    except Exception as e:
        print(f"✗ YOLOv12模型加载失败: {e}")
        cap.release()
        sys.exit(1)

    try:
        ir50_recognizer = AdaFaceRecognizer(args.ir50_model, args.ir50_input_size)
        print("✓ AdaFace IR50模型加载成功")
    except Exception as e:
        print(f"✗ AdaFace IR50模型加载失败: {e}")
        cap.release()
        sys.exit(1)

    # 初始化性能监控
    monitor = PerformanceMonitor(window_size=30)

    # 准备输出视频
    video_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.output_video,
            fourcc,
            video_info['fps'],
            (video_info['width'], video_info['height'])
        )
        print(f"✓ 输出视频: {args.output_video}")

    print("\n开始处理视频...")
    print("-" * 70)

    frame_count = 0
    processed_frames = 0
    skip_counter = 0

    # 进度条
    total_frames = min(video_info['total_frames'], args.max_frames) if args.max_frames > 0 else video_info[
        'total_frames']

    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 跳过帧
        if args.skip_frames > 0:
            skip_counter += 1
            if skip_counter <= args.skip_frames:
                continue
            skip_counter = 0

        # 检查最大帧数
        if args.max_frames > 0 and processed_frames >= args.max_frames:
            break

        frame_start_time = time.time()
        processed_frames += 1

        # YOLO检测
        boxes, yolo_time = yolo_detector.detect(frame)
        monitor.update_yolo(yolo_time)

        # 对每个检测到的人脸进行识别
        ir50_times = []
        for box in boxes:
            x1, y1, x2, y2, conf = box
            # 提取人脸区域
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                try:
                    feature, ir50_time = ir50_recognizer.extract_feature(face_roi)
                    if feature is not None:
                        ir50_times.append(ir50_time)
                        monitor.update_ir50(ir50_time)
                except Exception as e:
                    # 如果某个人脸识别失败，继续处理其他人脸
                    print(f"\n警告: 人脸识别失败: {e}")
                    continue

        # 计算平均IR50时间
        avg_ir50_time = np.mean(ir50_times) if ir50_times else 0

        # 计算帧处理时间
        frame_time = time.time() - frame_start_time
        monitor.update_frame(frame_time, len(boxes))

        # 获取统计信息
        stats = monitor.get_stats()

        # 输出信息
        progress = (processed_frames / total_frames * 100) if total_frames > 0 else 0
        faces_text = f"人脸: {len(boxes)}" if len(boxes) > 0 else "无人脸"

        print(f"\r[{progress:5.1f}%] 帧 {processed_frames:4d}/{total_frames} | "
              f"YOLO: {yolo_time * 1000:6.2f}ms (avg: {stats['yolo']['avg']:6.2f}ms) | "
              f"IR50: {avg_ir50_time * 1000:6.2f}ms (avg: {stats['ir50']['avg']:6.2f}ms) | "
              f"总帧: {frame_time * 1000:6.2f}ms | {faces_text}", end='')
        sys.stdout.flush()

        # 显示视频窗口
        if args.display:
            display_frame = frame.copy()
            # 绘制检测结果
            for box in boxes:
                x1, y1, x2, y2, conf = box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f'{conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 显示性能信息
            info_text = [
                f"帧: {processed_frames}/{total_frames}",
                f"YOLO: {yolo_time * 1000:.1f}ms",
                f"IR50: {avg_ir50_time * 1000:.1f}ms" if avg_ir50_time > 0 else "IR50: -",
                f"Faces: {len(boxes)}"
            ]
            for i, text in enumerate(info_text):
                cv2.putText(display_frame, text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow('Video Analysis', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 写入输出视频
        if video_writer:
            # 在输出视频上绘制结果
            output_frame = frame.copy()
            for box in boxes:
                x1, y1, x2, y2, conf = box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_frame, f'{conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            video_writer.write(output_frame)

    # 释放资源
    cap.release()
    if video_writer:
        video_writer.release()
    if args.display:
        cv2.destroyAllWindows()

    # 输出最终统计信息
    print("\n\n" + "=" * 70)
    print("分析完成 - 统计摘要")
    print("=" * 70)

    stats = monitor.get_stats()

    # 视频信息
    print(f"\n视频信息:")
    print(f"  总帧数: {processed_frames}")
    print(f"  总人脸数: {stats['frame']['total_faces']}")
    print(f"  平均每帧人脸数: {stats['frame']['total_faces'] / processed_frames:.2f}")

    # YOLO检测性能
    if 'yolo' in stats:
        print(f"\nYOLOv12人脸检测:")
        print(f"  平均推理时间: {stats['yolo']['avg']:.2f} ms")
        print(f"  最小推理时间: {stats['yolo']['min']:.2f} ms")
        print(f"  最大推理时间: {stats['yolo']['max']:.2f} ms")
        print(f"  标准差: {stats['yolo']['std']:.2f} ms")
        print(f"  总推理时间: {stats['yolo']['total']:.2f} ms")
        print(f"  处理帧数: {stats['yolo']['count']}")

    # IR50识别性能
    if 'ir50' in stats:
        print(f"\nAdaFace IR50人脸识别:")
        print(f"  平均推理时间: {stats['ir50']['avg']:.2f} ms")
        print(f"  最小推理时间: {stats['ir50']['min']:.2f} ms")
        print(f"  最大推理时间: {stats['ir50']['max']:.2f} ms")
        print(f"  标准差: {stats['ir50']['std']:.2f} ms")
        print(f"  总推理时间: {stats['ir50']['total']:.2f} ms")
        print(f"  处理人脸数: {stats['ir50']['count']}")

    # 整体性能
    if 'frame' in stats:
        print(f"\n整体性能:")
        print(f"  平均帧处理时间: {stats['frame']['avg']:.2f} ms")
        print(f"  最小帧处理时间: {stats['frame']['min']:.2f} ms")
        print(f"  最大帧处理时间: {stats['frame']['max']:.2f} ms")
        print(f"  标准差: {stats['frame']['std']:.2f} ms")
        print(f"  平均FPS: {stats['frame']['fps']:.2f}")

        # 计算处理速度与实时性的关系
        original_fps = video_info['fps']
        if original_fps > 0:
            speed_ratio = stats['frame']['fps'] / original_fps
            if speed_ratio >= 1:
                print(f"  处理速度: {speed_ratio:.2f}x 实时 (快于实时)")
            else:
                print(f"  处理速度: {speed_ratio:.2f}x 实时 (慢于实时)")

    # 导出详细报告
    report_file = f"performance_report_{os.path.splitext(os.path.basename(args.video_file))[0]}.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("视频人脸检测与识别性能报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"视频文件: {args.video_file}\n")
        f.write(f"YOLO模型: {args.yolo_model}\n")
        f.write(f"IR50模型: {args.ir50_model}\n")
        f.write(f"处理帧数: {processed_frames}\n")
        f.write(f"总人脸数: {stats['frame']['total_faces']}\n\n")

        if 'yolo' in stats:
            f.write("YOLOv12检测性能:\n")
            f.write(f"  平均: {stats['yolo']['avg']:.2f}ms\n")
            f.write(f"  最小: {stats['yolo']['min']:.2f}ms\n")
            f.write(f"  最大: {stats['yolo']['max']:.2f}ms\n")
            f.write(f"  标准差: {stats['yolo']['std']:.2f}ms\n\n")

        if 'ir50' in stats:
            f.write("AdaFace IR50识别性能:\n")
            f.write(f"  平均: {stats['ir50']['avg']:.2f}ms\n")
            f.write(f"  最小: {stats['ir50']['min']:.2f}ms\n")
            f.write(f"  最大: {stats['ir50']['max']:.2f}ms\n")
            f.write(f"  标准差: {stats['ir50']['std']:.2f}ms\n\n")

        if 'frame' in stats:
            f.write("整体性能:\n")
            f.write(f"  平均帧处理: {stats['frame']['avg']:.2f}ms\n")
            f.write(f"  平均FPS: {stats['frame']['fps']:.2f}\n")

    print(f"\n详细报告已保存到: {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
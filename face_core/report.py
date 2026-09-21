from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import format_timestamp, write_csv, write_json
from .video_processor import FrameResult, VideoMeta


def build_reports(
    output_dir: Path,
    video_path: Path,
    meta: VideoMeta,
    results: list[FrameResult],
    segments: list[dict],
    annotated_video: str = "",
    feature_meta: dict[str, Any] | None = None,
) -> dict:
    best = max(results, key=lambda r: r.raw_similarity) if results else None
    avg_raw = sum(r.raw_similarity for r in results) / len(results) if results else 0.0
    report = {
        "video": str(video_path),
        "video_meta": {
            "fps": meta.fps,
            "frame_count": meta.frame_count,
            "width": meta.width,
            "height": meta.height,
            "duration": meta.duration,
            "duration_text": format_timestamp(meta.duration),
        },
        "target_selection": "RetinaFace detection + ByteTrack; lock nearest target track_id when available",
        "feature_model": feature_meta or {},
        "recognition": {
            "name": best.best_name if best else "",
            "id_card": best.id_card if best else "",
            "best_timestamp": format_timestamp(best.timestamp) if best else "",
            "best_raw_similarity": best.raw_similarity if best else 0.0,
            # "best_raw_align_score":  None,
            "average_raw_similarity": avg_raw,
            "best_raw_face": best.raw_face_path if best else "",
            # "best_raw_aligned_face": best.raw_aligned_face_path if best else "",
        },
        "segments": [
            {
                "start": s["start"],
                "end": s["end"],
                "start_text": format_timestamp(s["start"]),
                "end_text": format_timestamp(s["end"]),
            }
            for s in segments
        ],
        "annotated_video": annotated_video,
        "frame_result_count": len(results),
    }
    write_json(output_dir / "recognition_result.json", report)
    write_json(output_dir / "target_tracks.json", [_frame_to_dict(r) for r in results])
    _write_similarity_csv(output_dir / "face_similarity.csv", results)
    _write_summary(output_dir / "summary.md", report)
    return report


def _write_similarity_csv(path: Path, results: list[FrameResult]) -> None:
    rows = []
    for r in results:
        rows.append(
            {
                "frame_index": r.frame_index,
                "timestamp": format_timestamp(r.timestamp),
                "raw_similarity": f"{r.raw_similarity:.6f}",
                # "raw_align_score": _fmt_optional(r.raw_align_score),
                # "raw_feature_valid": r.raw_feature_valid,
                "best_match_name": r.best_name,
                "id_card": r.id_card,
                "face_box": r.box,
                "raw_face_path": r.raw_face_path,
                "raw_aligned_face_path": r.raw_aligned_face_path,
                "track_source": r.track_source,
                "track_id": r.track_id,
                "lost_frames": r.lost_frames,
            }
        )
    write_csv(
        path,
        rows,
        [
            "frame_index",
            "timestamp",
            "raw_similarity",
            "raw_align_score",
            "raw_feature_valid",
            "best_match_name",
            "id_card",
            "face_box",
            "raw_face_path",
            "raw_aligned_face_path",
            "track_source",
            "track_id",
            "lost_frames",
        ],
    )


def _write_summary(path: Path, report: dict) -> None:
    rec = report["recognition"]
    feature = report.get("feature_model", {})
    lines = [
        "# 第一阶段人脸识别 Demo 报告",
        "",
        f"- 视频：`{report['video']}`",
        f"- 视频时长：{report['video_meta']['duration_text']}",
        f"- 目标选择策略：{report['target_selection']}",
        f"- 特征引擎：{feature.get('feature_engine', '')}",
        f"- 识别模型：{feature.get('recognition_model_path', feature.get('recognition_model_id', ''))}",
        f"- 对齐模型：{feature.get('aligner_model_path', feature.get('aligner_model_id', ''))}",
        f"- 识别结果：{rec['name']}",
        f"- 身份证：{rec['id_card']}",
        f"- 最佳时间戳：{rec['best_timestamp']}",
        f"- 原图最佳相似度：{rec['best_raw_similarity']:.4f}",
        # f"- 原图最佳对齐分：{_fmt_optional(rec['best_raw_align_score'])}",
        f"- 原图平均相似度：{rec['average_raw_similarity']:.4f}",
        "",
        "## 出现片段",
        "",
    ]
    for idx, segment in enumerate(report["segments"], start=1):
        lines.append(f"{idx}. {segment['start_text']} - {segment['end_text']}")
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `recognition_result.json`：完整识别结果",
            "- `target_tracks.json`：最近目标人逐帧轨迹，包含人脸 track_id、CVLFace 对齐分和特征有效性",
            "- `face_similarity.csv`：逐帧原图相似度与对齐质量",
            "- `annotated_video.mp4`：完整标注视频",
            "- `frames/face_raw`：原始人脸裁剪",
            "- `frames/face_aligned`：CVLface aligner 原图对齐结果",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _frame_to_dict(r: FrameResult) -> dict:
    return {
        "frame_index": r.frame_index,
        "timestamp": r.timestamp,
        "timestamp_text": format_timestamp(r.timestamp),
        "box": r.box,
        "raw_similarity": r.raw_similarity,
        # "raw_align_score": r.raw_align_score,
        # "raw_feature_valid": r.raw_feature_valid,
        "best_name": r.best_name,
        "id_card": r.id_card,
        "raw_face_path": r.raw_face_path,
        # "raw_aligned_face_path": r.raw_aligned_face_path,
        "target_score": r.target_score,
        "track_source": r.track_source,
        "track_id": r.track_id,
        "lost_frames": r.lost_frames,
    }


def _fmt_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"

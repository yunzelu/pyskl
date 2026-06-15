"""
extract_lab_hpe_jsonl.py

Purpose:
    Process the lab-collected RGB dataset and save HPE results as JSONL.

Dataset structure:
    dataset_root/
        1-saad-walk/
            output_video.avi
            frame_labels.csv
        15-han-fall2/
            output_video.avi
            frame_labels.csv
        ...

Each frame_labels.csv:
    Frame,Label
    1394,Walking
    1471,Transition-Stand-to-Sit
    1520,Sit-Stationary
    3046,Transition-Sit-to-Stand
    7311,DELETE
    17639,END

Important assumptions:
    - One and only one real subject in each video.
    - If YOLO returns multiple detections, select the one with highest bbox confidence.
    - Reported video FPS may be wrong, so timestamp uses --assumed-fps, default 30.
    - The script saves per-frame activity label info so later scripts do not need
      to parse the video/CSV again.

Example:
    python extract_lab_hpe_jsonl.py ^
        --root "D:/lu/project/lab_dataset" ^
        --output "D:/lu/project/lab_hpe_jsonl/yolo26x" ^
        --model "yolo26x-pose.pt" ^
        --imgsz 640 ^
        --device 0 ^
        --assumed-fps 30
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_subject_from_folder(folder_name: str) -> str:
    """
    Example:
        1-saad-walk -> saad
        15-han-fall2 -> han
    """
    parts = folder_name.split("-")
    if len(parts) >= 3:
        return parts[1]
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def label_to_group(label: str | None) -> str:
    """
    Coarse grouping for later audit.

    You can modify this mapping later without rerunning HPE,
    because the original label is also saved.
    """
    if label is None:
        return "unlabeled"

    text = label.strip().lower()

    if text in {"delete", "end"}:
        return "ignore"

    if "stationary" in text:
        return "stationary"

    if "transition" in text:
        return "transition"

    if "fall" in text:
        return "transition"

    if "walk" in text:
        return "walking"

    return "other"


def read_frame_labels(csv_path: Path) -> list[dict[str, Any]]:
    """
    Convert frame_labels.csv into segments.

    Each row marks the start frame of a label.
    END marks the end of the previous labeled period.
    DELETE marks a period to ignore.

    Returns:
        [
            {
                "start_frame": int,
                "end_frame": int,
                "label": str,
                "label_group": str,
                "use_for_audit": bool
            },
            ...
        ]
    """
    rows = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = safe_int(row.get("Frame"))
            label = row.get("Label")

            if frame is None or label is None:
                continue

            rows.append(
                {
                    "frame": frame,
                    "label": label.strip(),
                }
            )

    rows = sorted(rows, key=lambda x: x["frame"])

    segments = []

    for i, row in enumerate(rows):
        start = row["frame"]
        label = row["label"]

        if label.upper() == "END":
            continue

        if i + 1 < len(rows):
            end = rows[i + 1]["frame"] - 1
        else:
            # If no END exists, leave open. Later we will clip by video frame count.
            end = None

        group = label_to_group(label)

        segments.append(
            {
                "start_frame": start,
                "end_frame": end,
                "label": label,
                "label_group": group,
                "use_for_audit": group in {"stationary", "transition"},
            }
        )

    return segments


def clip_segments_to_video(
    segments: list[dict[str, Any]],
    frame_count: int,
) -> list[dict[str, Any]]:
    clipped = []
    has_known_frame_count = frame_count > 0

    for seg in segments:
        start = int(seg["start_frame"])
        end = seg["end_frame"]

        if end is None and has_known_frame_count:
            end = frame_count - 1

        if end is not None:
            end = int(end)
            if has_known_frame_count:
                end = min(end, frame_count - 1)

        if start < 0:
            start = 0

        if end is not None and start > end:
            continue

        new_seg = dict(seg)
        new_seg["start_frame"] = start
        new_seg["end_frame"] = end
        new_seg["length_frames"] = None if end is None else end - start + 1
        clipped.append(new_seg)

    return clipped


def build_frame_label_arrays(
    segments: list[dict[str, Any]],
    frame_count: int,
) -> tuple[list[str | None], list[str], list[int | None]]:
    """
    Build fast frame -> label lookup arrays.
    """
    labels: list[str | None] = [None] * frame_count
    groups: list[str] = ["unlabeled"] * frame_count
    segment_ids: list[int | None] = [None] * frame_count

    for seg_idx, seg in enumerate(segments):
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])

        for t in range(start, end + 1):
            labels[t] = seg["label"]
            groups[t] = seg["label_group"]
            segment_ids[t] = seg_idx

    return labels, groups, segment_ids


def find_segment_id_for_frame(
    frame_idx: int,
    segments: list[dict[str, Any]],
    cursor: int,
) -> tuple[int | None, int]:
    """
    Find the annotation segment for monotonically increasing frame indices.
    Used when OpenCV cannot report a valid frame count, so we cannot build
    dense frame-label arrays up front.
    """
    if not segments:
        return None, cursor

    while cursor + 1 < len(segments):
        next_start = int(segments[cursor + 1]["start_frame"])
        if frame_idx < next_start:
            break
        cursor += 1

    seg = segments[cursor]
    start = int(seg["start_frame"])
    end = seg["end_frame"]

    if frame_idx < start:
        return None, cursor

    if end is not None and frame_idx > int(end):
        return None, cursor

    return cursor, cursor


def discover_sessions(root: Path) -> list[dict[str, Any]]:
    """
    Find folders containing both output_video.avi and frame_labels.csv.
    """
    sessions = []

    for folder in [root, *sorted(root.rglob("*"))]:
        if not folder.is_dir():
            continue

        video_path = folder / "output_video.avi"
        label_path = folder / "frame_labels.csv"

        if video_path.exists() and label_path.exists():
            sessions.append(
                {
                    "folder": folder,
                    "session_name": folder.name,
                    "subject": parse_subject_from_folder(folder.name),
                    "video_path": video_path,
                    "label_path": label_path,
                }
            )

    return sessions


def make_output_path(
    output_root: Path,
    dataset_root: Path,
    session_folder: Path,
    model_path: str,
    imgsz: list[int],
) -> Path:
    rel = session_folder.relative_to(dataset_root)
    if str(rel) == ".":
        rel = Path(session_folder.name)

    model_name = Path(model_path).stem

    if len(imgsz) == 1:
        imgsz_name = str(imgsz[0])
    else:
        imgsz_name = "x".join(str(x) for x in imgsz)

    out_dir = output_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir / f"{session_folder.name}__{model_name}__imgsz{imgsz_name}.jsonl"


def to_builtin(obj: Any) -> Any:
    """
    Convert numpy values to JSON-serializable Python values.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    return obj


def select_one_person_result(result: Any, kpt_conf_thr: float) -> dict[str, Any]:
    """
    Select one person from a YOLO pose result.

    Since the video should contain one real subject, we keep only the detection
    with highest bbox confidence, but still save num_detections for diagnostics.
    """
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {
            "detected": False,
            "num_detections": 0,
            "selected_idx": None,
            "bbox_xyxy": None,
            "bbox_conf": None,
            "keypoints_xy": None,
            "keypoints_conf": None,
            "avg_keypoint_conf": None,
            "valid_keypoint_count": 0,
            "total_keypoint_count": 0,
        }

    bbox_xyxy = boxes.xyxy.detach().cpu().numpy()
    bbox_conf = boxes.conf.detach().cpu().numpy()

    selected_idx = int(np.argmax(bbox_conf))

    keypoints_xy = None
    keypoints_conf = None
    avg_keypoint_conf = None
    valid_keypoint_count = 0
    total_keypoint_count = 0

    if result.keypoints is not None:
        kxy = result.keypoints.xy.detach().cpu().numpy()

        if result.keypoints.conf is not None:
            kconf = result.keypoints.conf.detach().cpu().numpy()
        else:
            kconf = None

        if len(kxy) > selected_idx:
            keypoints_xy = kxy[selected_idx].astype(float).tolist()
            total_keypoint_count = len(keypoints_xy)

        if kconf is not None and len(kconf) > selected_idx:
            keypoints_conf = kconf[selected_idx].astype(float).tolist()
            conf_arr = np.asarray(keypoints_conf, dtype=float)

            if conf_arr.size > 0:
                avg_keypoint_conf = float(np.nanmean(conf_arr))
                valid_keypoint_count = int(np.sum(conf_arr >= kpt_conf_thr))

    return {
        "detected": True,
        "num_detections": int(len(boxes)),
        "selected_idx": selected_idx,
        "bbox_xyxy": bbox_xyxy[selected_idx].astype(float).tolist(),
        "bbox_conf": float(bbox_conf[selected_idx]),
        "keypoints_xy": keypoints_xy,
        "keypoints_conf": keypoints_conf,
        "avg_keypoint_conf": avg_keypoint_conf,
        "valid_keypoint_count": valid_keypoint_count,
        "total_keypoint_count": total_keypoint_count,
    }


def process_one_session(
    session: dict[str, Any],
    dataset_root: Path,
    output_root: Path,
    model: YOLO,
    model_path: str,
    imgsz: list[int],
    device: str,
    conf: float,
    iou: float,
    assumed_fps: float,
    kpt_conf_thr: float,
    skip_existing: bool,
) -> None:
    video_path = session["video_path"]
    label_path = session["label_path"]
    session_folder = session["folder"]

    output_path = make_output_path(
        output_root=output_root,
        dataset_root=dataset_root,
        session_folder=session_folder,
        model_path=model_path,
        imgsz=imgsz,
    )

    if skip_existing and output_path.exists():
        print(f"[SKIP] {output_path}")
        return

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_known = frame_count > 0

    if not frame_count_known:
        print(
            f"[WARN] {session['session_name']}: cv2 reported frame_count={frame_count}. "
            "Keeping annotation segments unclipped and assigning labels by frame_idx."
        )

    raw_segments = read_frame_labels(label_path)
    segments = clip_segments_to_video(raw_segments, frame_count)

    if frame_count_known:
        frame_labels, frame_groups, frame_segment_ids = build_frame_label_arrays(
            segments=segments,
            frame_count=frame_count,
        )
    else:
        frame_labels, frame_groups, frame_segment_ids = [], [], []

    label_set = sorted({seg["label"] for seg in segments})
    label_group_set = sorted({seg["label_group"] for seg in segments})

    metadata = {
        "type": "metadata",
        "dataset_info": {
            "dataset_root": str(dataset_root),
            "session_folder": str(session_folder),
            "session_name": session["session_name"],
            "subject": session["subject"],
            "structure_note": "Each session folder contains output_video.avi and frame_labels.csv.",
            "one_subject_assumption": True,
        },
        "video_info": {
            "video_path": str(video_path),
            "video_name": video_path.name,
            "reported_fps_from_cv2": reported_fps,
            "assumed_fps_used_for_timestamp": assumed_fps,
            "width": width,
            "height": height,
            "frame_count_from_cv2": frame_count,
            "frame_count_known_from_cv2": frame_count_known,
        },
        "annotation_info": {
            "annotation_path": str(label_path),
            "format_note": "Each CSV row gives the starting frame of a label. DELETE means skipped period. END marks the end boundary.",
            "label_set": label_set,
            "label_group_set": label_group_set,
            "segments": segments,
        },
        "model_info": {
            "model_path": model_path,
            "imgsz": imgsz,
            "conf": conf,
            "iou": "NA",
            "device": device,
            "kpt_conf_thr": kpt_conf_thr,
        },
        "format_note": {
            "selected_person_rule": "If multiple detections exist, select the person with the highest bbox confidence.",
            "coordinate_system": "Pixel coordinates in original video frame.",
            "timestamp_sec": "frame_idx / assumed_fps, because video metadata FPS may be incorrect.",
            "label_group": "stationary, transition, walking, ignore, unlabeled, or other.",
            "future_pyskl_note": "keypoints_xy and keypoints_conf can later be converted to keypoint [M,T,V,C] and keypoint_score [M,T,V].",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    yolo_imgsz: int | list[int]
    if len(imgsz) == 1:
        yolo_imgsz = imgsz[0]
    else:
        yolo_imgsz = imgsz

    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=False, default=to_builtin) + "\n")

        pbar = tqdm(
            total=frame_count if frame_count_known else None,
            desc=session["session_name"],
            unit="frame",
        )

        frame_idx = 0
        segment_cursor = 0
        total_wall_time_sec = 0.0

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            t0 = time.perf_counter()

            results = model.predict(
                source=frame,
                imgsz=yolo_imgsz,
                conf=conf,
                # iou=iou,
                device=device,
                verbose=False,
            )

            wall_time_ms = (time.perf_counter() - t0) * 1000.0
            total_wall_time_sec += wall_time_ms / 1000.0

            result = results[0]
            selected = select_one_person_result(
                result=result,
                kpt_conf_thr=kpt_conf_thr,
            )

            speed = getattr(result, "speed", {}) or {}
            yolo_speed_ms = {
                "preprocess": safe_float(speed.get("preprocess")),
                "inference": safe_float(speed.get("inference")),
                "postprocess": safe_float(speed.get("postprocess")),
            }

            valid_speed_values = [
                v for v in yolo_speed_ms.values()
                if v is not None
            ]
            yolo_speed_ms["total"] = (
                float(sum(valid_speed_values))
                if valid_speed_values
                else None
            )

            if frame_idx < len(frame_labels):
                label = frame_labels[frame_idx]
                label_group = frame_groups[frame_idx]
                segment_id = frame_segment_ids[frame_idx]
            else:
                segment_id, segment_cursor = find_segment_id_for_frame(
                    frame_idx=frame_idx,
                    segments=segments,
                    cursor=segment_cursor,
                )

                if segment_id is not None:
                    label = segments[segment_id]["label"]
                    label_group = segments[segment_id]["label_group"]
                else:
                    label = None
                    label_group = "unlabeled"

            if segment_id is not None:
                seg = segments[segment_id]
                segment_start = seg["start_frame"]
                segment_end = seg["end_frame"]
                segment_length = seg["length_frames"]
            else:
                segment_start = None
                segment_end = None
                segment_length = None

            frame_record = {
                "type": "frame",
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / assumed_fps,
                "wall_time_ms": wall_time_ms,
                "yolo_speed_ms": yolo_speed_ms,
                "label": label,
                "label_group": label_group,
                "segment_id": segment_id,
                "segment_start_frame": segment_start,
                "segment_end_frame": segment_end,
                "segment_length_frames": segment_length,
            }

            frame_record.update(selected)

            f.write(json.dumps(frame_record, ensure_ascii=False, default=to_builtin) + "\n")

            frame_idx += 1
            pbar.update(1)

        summary_record = {
            "type": "process_summary",
            "frames_processed": frame_idx,
            "total_wall_time_sec": total_wall_time_sec,
            "effective_processing_fps": frame_idx / total_wall_time_sec if total_wall_time_sec > 0 else None,
        }
        f.write(json.dumps(summary_record, ensure_ascii=False, default=to_builtin) + "\n")

        pbar.close()

    cap.release()

    print(f"[DONE] Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
        help="Root folder of the lab-collected dataset.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output root folder for JSONL files.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="YOLO pose model path, e.g., yolo26x-pose.pt or OpenVINO model folder.",
    )

    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[640],
        help="YOLO image size. Use one value, e.g. --imgsz 640, or two values, e.g. --imgsz 352 640.",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="YOLO device. Use 0 for first GPU, cpu for CPU.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO bbox confidence threshold.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="YOLO NMS IoU threshold.",
    )

    parser.add_argument(
        "--assumed-fps",
        type=float,
        default=30.0,
        help="FPS used for timestamp. Use 30 because video metadata FPS may be wrong.",
    )

    parser.add_argument(
        "--kpt-conf-thr",
        type=float,
        default=0.0,
        help="Keypoint confidence threshold for valid_keypoint_count.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sessions whose output JSONL already exists.",
    )

    args = parser.parse_args()

    dataset_root = Path(args.root)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(dataset_root)

    if not sessions:
        raise RuntimeError(
            f"No sessions found under {dataset_root}. "
            "Expected folders with output_video.avi and frame_labels.csv."
        )

    print(f"[INFO] Found {len(sessions)} sessions.")
    print(f"[INFO] Loading model: {args.model}")

    model = YOLO(args.model)

    for session in sessions:
        process_one_session(
            session=session,
            dataset_root=dataset_root,
            output_root=output_root,
            model=model,
            model_path=args.model,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            assumed_fps=args.assumed_fps,
            kpt_conf_thr=args.kpt_conf_thr,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()

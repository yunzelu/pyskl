"""
Create stride-downsampled PYSKL annotation pkls from existing radar_v4 pkls.

The default conversion simulates 30 fps -> 10 fps by creating three phase
variants per source clip:

    phase 1: 1, 4, 7, ...
    phase 2: 2, 5, 8, ...
    phase 3: 3, 6, 9, ...

Phase numbers in the CLI are 1-based to match that notation. Internally, the
arrays are indexed from zero. By default, clips whose length is not divisible
by the stride are shortened to floor(T / stride) frames for every phase, so all
phase variants from the same source clip have the same temporal length.

The split is preserved by source annotation id: every generated phase variant
is assigned to the same train/val/test split as the source clip. This keeps the
existing subject split intact when the input pkl was already split by subject.

Examples:
    python tools/data/radar_v4/downsample_pyskl_pkl.py ^
        data/radar_v4/pyskl/radarv4_yolo26xpose_clip60_val_mia_test_yunze.pkl

    python tools/data/radar_v4/downsample_pyskl_pkl.py ^
        data/radar_v4/pyskl ^
        --glob "radarv4_yolo26xpose_clip60_val_mia_test_*.pkl"
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sanitize_suffix(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Suffix cannot be empty.")
    return value if value.startswith("_") else f"_{value}"


def default_suffix(stride: int, phases: list[int]) -> str:
    phase_text = "".join(str(phase) for phase in phases)
    return f"_ds{stride}p{phase_text}"


def load_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"{path} does not contain a top-level dict.")
    if "annotations" not in data or "split" not in data:
        raise KeyError(f"{path} must contain 'annotations' and 'split'.")
    if not isinstance(data["annotations"], list):
        raise TypeError(f"{path}['annotations'] must be a list.")
    if not isinstance(data["split"], dict):
        raise TypeError(f"{path}['split'] must be a dict.")

    return data


def dump_pkl(data: dict[str, Any], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def annotation_id_key(annotations: list[dict[str, Any]]) -> str:
    if not annotations:
        raise ValueError("Input pkl has no annotations.")

    first = annotations[0]
    if "frame_dir" in first:
        return "frame_dir"
    if "filename" in first:
        return "filename"

    raise KeyError("Annotations must contain either 'frame_dir' or 'filename'.")


def validate_phases(stride: int, phases: list[int]) -> list[int]:
    if stride <= 0:
        raise ValueError("--stride must be positive.")

    if not phases:
        phases = list(range(1, stride + 1))

    seen = set()
    cleaned: list[int] = []
    for phase in phases:
        if phase < 1 or phase > stride:
            raise ValueError(f"Phase {phase} is invalid for stride {stride}.")
        if phase not in seen:
            cleaned.append(phase)
            seen.add(phase)

    return cleaned


def make_indices(
    total_frames: int,
    stride: int,
    phase: int,
    tail_policy: str,
) -> np.ndarray:
    offset = phase - 1

    if tail_policy == "floor":
        count = total_frames // stride
        if count <= 0:
            return np.empty((0,), dtype=np.int64)
        return offset + stride * np.arange(count, dtype=np.int64)

    if tail_policy == "natural":
        return np.arange(offset, total_frames, stride, dtype=np.int64)

    raise ValueError(f"Unknown tail policy: {tail_policy}")


def split_lookup(split: dict[str, list[str]]) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = defaultdict(list)

    for split_name, ids in split.items():
        if not isinstance(ids, list):
            raise TypeError(f"split[{split_name!r}] must be a list.")
        for item_id in ids:
            lookup[str(item_id)].append(split_name)

    return lookup


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def downsample_annotation(
    item: dict[str, Any],
    id_key: str,
    indices: np.ndarray,
    stride: int,
    phase: int,
    tail_policy: str,
    source_fps: float,
    target_fps: float,
) -> dict[str, Any]:
    source_id = str(item[id_key])
    keypoint = item.get("keypoint")
    if not isinstance(keypoint, np.ndarray):
        raise TypeError(f"{source_id}: keypoint must be a numpy array.")
    if keypoint.ndim < 2:
        raise ValueError(f"{source_id}: keypoint shape {keypoint.shape} is invalid.")

    source_total_frames = int(keypoint.shape[1])
    declared_total_frames = int_or_none(item.get("total_frames"))
    if declared_total_frames is not None and declared_total_frames != source_total_frames:
        raise ValueError(
            f"{source_id}: total_frames={declared_total_frames} but "
            f"keypoint.shape[1]={source_total_frames}."
        )

    if len(indices) == 0:
        raise ValueError(f"{source_id}: downsampled clip would have zero frames.")

    new_id = f"{source_id}__ds{stride}p{phase}"
    new_item = dict(item)
    new_item[id_key] = new_id
    new_item["keypoint"] = np.ascontiguousarray(keypoint[:, indices, ...])
    new_item["total_frames"] = int(len(indices))

    keypoint_score = item.get("keypoint_score")
    if keypoint_score is not None:
        if not isinstance(keypoint_score, np.ndarray):
            raise TypeError(f"{source_id}: keypoint_score must be a numpy array.")
        if keypoint_score.ndim < 2 or keypoint_score.shape[1] != source_total_frames:
            raise ValueError(
                f"{source_id}: keypoint_score shape {keypoint_score.shape} "
                f"does not match keypoint frame count {source_total_frames}."
            )
        new_item["keypoint_score"] = np.ascontiguousarray(keypoint_score[:, indices, ...])

    new_item["downsampled_from"] = source_id
    new_item["downsample_stride"] = int(stride)
    new_item["downsample_phase"] = int(phase)
    new_item["downsample_offset"] = int(phase - 1)
    new_item["downsample_tail_policy"] = tail_policy
    new_item["downsample_source_fps"] = float(source_fps)
    new_item["downsample_target_fps"] = float(target_fps)
    new_item["downsample_source_total_frames"] = source_total_frames
    new_item["downsample_relative_start_index"] = int(indices[0])
    new_item["downsample_relative_end_index"] = int(indices[-1])

    source_start_frame = int_or_none(item.get("start_frame"))
    if source_start_frame is not None:
        new_item["downsample_source_start_frame"] = source_start_frame
        new_item["downsample_source_end_frame"] = int_or_none(item.get("end_frame"))
        new_item["start_frame"] = source_start_frame + int(indices[0])
        new_item["end_frame"] = source_start_frame + int(indices[-1])

    return new_item


def label_key(item: dict[str, Any]) -> str:
    for key in ("label_name", "cleaned_label", "original_label", "label"):
        if key in item and item[key] is not None:
            return str(item[key])
    return "unknown"


def count_by_label(
    annotations: list[dict[str, Any]],
    ids: list[str] | None,
    id_key: str,
) -> dict[str, int]:
    allowed = set(ids) if ids is not None else None
    counts: Counter[str] = Counter()
    for item in annotations:
        if allowed is None or str(item[id_key]) in allowed:
            counts[label_key(item)] += 1
    return dict(sorted(counts.items()))


def count_by_subject(
    annotations: list[dict[str, Any]],
    ids: list[str] | None,
    id_key: str,
) -> dict[str, int]:
    allowed = set(ids) if ids is not None else None
    counts: Counter[str] = Counter()
    for item in annotations:
        if allowed is None or str(item[id_key]) in allowed:
            counts[str(item.get("subject", "unknown"))] += 1
    return dict(sorted(counts.items()))


def frame_stats(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(item["total_frames"]) for item in annotations]
    counts = Counter(lengths)
    if not lengths:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "most_common": [],
        }

    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": float(np.mean(lengths)),
        "most_common": [
            {"total_frames": length, "count": count}
            for length, count in counts.most_common(20)
        ],
    }


def subject_split_conflicts(
    annotations: list[dict[str, Any]],
    split: dict[str, list[str]],
    id_key: str,
) -> dict[str, list[str]]:
    id_to_subject = {
        str(item[id_key]): str(item.get("subject", "unknown"))
        for item in annotations
    }
    subject_to_splits: dict[str, set[str]] = defaultdict(set)

    for split_name, ids in split.items():
        for item_id in ids:
            subject = id_to_subject.get(str(item_id))
            if subject is not None:
                subject_to_splits[subject].add(split_name)

    return {
        subject: sorted(split_names)
        for subject, split_names in sorted(subject_to_splits.items())
        if len(split_names) > 1
    }


def make_summary(
    input_pkl: Path,
    output_pkl: Path,
    input_annotations: list[dict[str, Any]],
    output_annotations: list[dict[str, Any]],
    output_split: dict[str, list[str]],
    id_key: str,
    stride: int,
    phases: list[int],
    tail_policy: str,
    source_fps: float,
    target_fps: float,
    skipped_zero_frame: int,
) -> dict[str, Any]:
    return {
        "input_pkl": str(input_pkl),
        "output_pkl": str(output_pkl),
        "id_key": id_key,
        "stride": stride,
        "phases": phases,
        "tail_policy": tail_policy,
        "source_fps": source_fps,
        "target_fps": target_fps,
        "num_input_annotations": len(input_annotations),
        "num_output_annotations": len(output_annotations),
        "skipped_zero_frame_phase_variants": skipped_zero_frame,
        "input_total_frames": frame_stats(input_annotations),
        "output_total_frames": frame_stats(output_annotations),
        "samples_per_label": count_by_label(output_annotations, None, id_key),
        "samples_per_subject": count_by_subject(output_annotations, None, id_key),
        "split_counts": {
            split_name: len(ids)
            for split_name, ids in output_split.items()
        },
        "samples_per_label_per_split": {
            split_name: count_by_label(output_annotations, ids, id_key)
            for split_name, ids in output_split.items()
        },
        "samples_per_subject_per_split": {
            split_name: count_by_subject(output_annotations, ids, id_key)
            for split_name, ids in output_split.items()
        },
        "subject_split_conflicts": subject_split_conflicts(
            output_annotations,
            output_split,
            id_key,
        ),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_summary(summary: dict[str, Any], output_pkl: Path, overwrite: bool) -> Path:
    summary_path = output_pkl.with_name(f"{output_pkl.stem}_summary.json")
    if summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"{summary_path} already exists. Use --overwrite to replace it."
        )

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=json_default)
    return summary_path


def convert_pkl(
    input_pkl: Path,
    output_pkl: Path,
    stride: int,
    phases: list[int],
    tail_policy: str,
    source_fps: float,
    target_fps: float,
    overwrite: bool,
    write_summary_file: bool,
) -> None:
    data = load_pkl(input_pkl)
    input_annotations = data["annotations"]
    input_split = data["split"]
    id_key = annotation_id_key(input_annotations)
    source_to_splits = split_lookup(input_split)

    output_annotations: list[dict[str, Any]] = []
    output_split: dict[str, list[str]] = {
        split_name: []
        for split_name in input_split
    }
    skipped_zero_frame = 0

    for item in input_annotations:
        source_id = str(item[id_key])
        keypoint = item.get("keypoint")
        if not isinstance(keypoint, np.ndarray):
            raise TypeError(f"{source_id}: keypoint must be a numpy array.")
        total_frames = int(keypoint.shape[1])

        for phase in phases:
            indices = make_indices(
                total_frames=total_frames,
                stride=stride,
                phase=phase,
                tail_policy=tail_policy,
            )
            if len(indices) == 0:
                skipped_zero_frame += 1
                continue

            new_item = downsample_annotation(
                item=item,
                id_key=id_key,
                indices=indices,
                stride=stride,
                phase=phase,
                tail_policy=tail_policy,
                source_fps=source_fps,
                target_fps=target_fps,
            )
            output_annotations.append(new_item)

            for split_name in source_to_splits.get(source_id, []):
                output_split[split_name].append(str(new_item[id_key]))

    output_data = {
        "split": output_split,
        "annotations": output_annotations,
    }
    dump_pkl(output_data, output_pkl, overwrite=overwrite)

    print(
        f"[DONE] {output_pkl} "
        f"(annotations={len(output_annotations)}, "
        + ", ".join(f"{name}={len(ids)}" for name, ids in output_split.items())
        + ")"
    )

    if write_summary_file:
        summary = make_summary(
            input_pkl=input_pkl,
            output_pkl=output_pkl,
            input_annotations=input_annotations,
            output_annotations=output_annotations,
            output_split=output_split,
            id_key=id_key,
            stride=stride,
            phases=phases,
            tail_policy=tail_policy,
            source_fps=source_fps,
            target_fps=target_fps,
            skipped_zero_frame=skipped_zero_frame,
        )
        summary_path = write_summary(summary, output_pkl, overwrite=overwrite)
        print(f"[DONE] Summary: {summary_path}")

        conflicts = summary["subject_split_conflicts"]
        if conflicts:
            print(f"[WARN] Subjects appearing in multiple splits: {conflicts}")


def resolve_output_path(
    input_pkl: Path,
    input_root: Path,
    output: Path | None,
    output_dir: Path | None,
    suffix: str,
    input_is_dir: bool,
) -> Path:
    if output is not None:
        if input_is_dir:
            raise ValueError("--output can only be used when the input is one pkl file.")
        return output

    if output_dir is not None:
        rel = input_pkl.relative_to(input_root) if input_is_dir else Path(input_pkl.name)
        return output_dir / rel.with_name(f"{rel.stem}{suffix}{rel.suffix}")

    return input_pkl.with_name(f"{input_pkl.stem}{suffix}{input_pkl.suffix}")


def iter_input_pkls(input_path: Path, glob_pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob(glob_pattern) if path.is_file())
    raise FileNotFoundError(input_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample an existing radar_v4 PYSKL pkl into stride-phase variants."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input pkl file, or a directory containing pkl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output pkl path. Only valid when input is a single pkl file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. For directory input, preserves the relative input "
            "layout under this directory."
        ),
    )
    parser.add_argument(
        "--glob",
        default="*.pkl",
        help="Glob used when input is a directory. Default: *.pkl",
    )
    parser.add_argument(
        "--suffix",
        help="Output filename suffix. Default: _ds{stride}p{phases}, e.g. _ds3p123.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Temporal stride. Use 3 for 30 fps -> 10 fps.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        help="1-based phases to create. Default: all phases for the stride.",
    )
    parser.add_argument(
        "--tail-policy",
        choices=["floor", "natural"],
        default="floor",
        help=(
            "floor uses floor(T / stride) frames for every phase, dropping tails; "
            "natural keeps phase::stride and can produce lengths differing by one."
        ),
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=30.0,
        help="Source fps recorded in metadata fields only. Default: 30.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        help="Target fps recorded in metadata fields only. Default: source_fps / stride.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output pkl and summary JSON.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write the companion *_summary.json file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phases = validate_phases(args.stride, args.phases or [])
    suffix = sanitize_suffix(args.suffix) if args.suffix else default_suffix(args.stride, phases)
    target_fps = args.target_fps if args.target_fps is not None else args.source_fps / args.stride

    input_path = args.input
    input_pkls = iter_input_pkls(input_path, args.glob)
    if not input_pkls:
        raise RuntimeError(f"No pkl files found under {input_path} with glob {args.glob!r}.")

    input_is_dir = input_path.is_dir()
    input_root = input_path if input_is_dir else input_path.parent

    print(
        f"[INFO] Converting {len(input_pkls)} pkl file(s): "
        f"stride={args.stride}, phases={phases}, tail_policy={args.tail_policy}, "
        f"target_fps={target_fps:g}"
    )

    for input_pkl in input_pkls:
        output_pkl = resolve_output_path(
            input_pkl=input_pkl,
            input_root=input_root,
            output=args.output,
            output_dir=args.output_dir,
            suffix=suffix,
            input_is_dir=input_is_dir,
        )
        convert_pkl(
            input_pkl=input_pkl,
            output_pkl=output_pkl,
            stride=args.stride,
            phases=phases,
            tail_policy=args.tail_policy,
            source_fps=args.source_fps,
            target_fps=target_fps,
            overwrite=args.overwrite,
            write_summary_file=not args.no_summary,
        )


if __name__ == "__main__":
    main()

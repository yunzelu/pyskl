"""Build per-fold PYSKL annotations from fused S6 pseudo labels."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.s6.common import DEFAULT_CONTINUOUS_TEACHER_PKL, LABELS, TeacherSpec, selected_specs


DEFAULT_PSEUDO_ROOT = Path("work_dirs/thesis/s6/pseudo_labels")
DEFAULT_OUT_DIR = Path("data/radar_v4/pyskl/s6_pseudo")
MODE_FILES = {
    "hard": "fusion_hard_pseudo_labels.npz",
    "raw_soft": "fusion_raw_soft_probabilities.npz",
    "calibrated_soft": "fusion_calibrated_soft_probabilities.npz",
    "mc_calibrated_soft": "fusion_mc_calibrated_soft_probabilities.npz",
}
MODE_TAGS = {
    "hard": "fusion1to1_hard",
    "raw_soft": "fusion1to1_raw_soft",
    "calibrated_soft": "fusion1to1_calibrated_soft",
    "mc_calibrated_soft": "fusion1to1_mc_calibrated_soft",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one fold-level pseudo-label annotation pkl from S6 fused outputs."
    )
    parser.add_argument("--source-pkl", type=Path, default=DEFAULT_CONTINUOUS_TEACHER_PKL)
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds", nargs="+", default=["a", "b", "c"])
    parser.add_argument(
        "--mode",
        choices=("hard", "raw_soft", "calibrated_soft", "mc_calibrated_soft"),
        default="hard",
    )
    parser.add_argument("--split-template", default="fold_{fold}_pseudo_train")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-incomplete-folds",
        action="store_true",
        help="Skip a fold if any of its teacher fusion files are missing.",
    )
    parser.add_argument(
        "--include-probability-passes",
        action="store_true",
        help="For MC mode, store [K, C] per-pass fused probabilities in each sample.",
    )
    return parser.parse_args()


def read_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or "annotations" not in data or "split" not in data:
        raise ValueError(f"{path} is not a PYSKL annotation dict with annotations and split")
    return data


def write_pickle(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def source_index(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    duplicates = []
    for item in source["annotations"]:
        frame_dir = str(item.get("frame_dir", ""))
        if not frame_dir:
            continue
        if frame_dir in index:
            duplicates.append(frame_dir)
        index[frame_dir] = item
    if duplicates:
        raise ValueError(f"Duplicate frame_dir values in source pkl, first examples: {duplicates[:5]}")
    return index


def label_group(label_name: str) -> str:
    if label_name.startswith("Transition-"):
        return "transition"
    if label_name == "Walking":
        return "walking"
    if label_name == "Falling":
        return "falling"
    return "state"


def scalar(value: np.ndarray | np.generic | Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def frame_key_from_npz(data: dict[str, np.ndarray], index: int) -> str:
    return str(scalar(data["frame_dir"][index]))


def validate_window_match(source_item: dict[str, Any], data: dict[str, np.ndarray], index: int) -> None:
    checks = (
        ("subject_id", str(source_item.get("subject_id") or source_item.get("subject") or ""), str(scalar(data["subject_id"][index]))),
        ("recording_id", str(source_item.get("recording_id") or ""), str(scalar(data["recording_id"][index]))),
        ("start_frame", int(source_item.get("start_frame", -1)), int(scalar(data["window_start_frame"][index]))),
        ("end_frame", int(source_item.get("end_frame", -1)), int(scalar(data["window_end_frame"][index]))),
        ("center_frame", int(source_item.get("center_frame", -1)), int(scalar(data["center_frame"][index]))),
    )
    mismatches = [
        (name, left, right)
        for name, left, right in checks
        if left != right
    ]
    if mismatches:
        raise ValueError(
            f"Pseudo/source metadata mismatch for {frame_key_from_npz(data, index)!r}: {mismatches}"
        )
    source_ts = float(source_item.get("center_timestamp", np.nan))
    pseudo_ts = float(scalar(data["center_timestamp"][index]))
    if not np.isclose(source_ts, pseudo_ts, atol=1e-6, rtol=0):
        raise ValueError(
            f"Pseudo/source timestamp mismatch for {frame_key_from_npz(data, index)!r}: "
            f"{source_ts} vs {pseudo_ts}"
        )


def add_pseudo_fields(
    source_item: dict[str, Any],
    data: dict[str, np.ndarray],
    index: int,
    spec: TeacherSpec,
    mode: str,
    source_path: Path,
    include_probability_passes: bool,
) -> dict[str, Any]:
    item = copy.deepcopy(source_item)
    original_label = int(item.get("label", item.get("hard_label", -1)))
    original_label_name = str(item.get("label_name") or (LABELS[original_label] if 0 <= original_label < len(LABELS) else ""))
    original_group = str(item.get("gt_group") or item.get("label_group") or label_group(original_label_name))

    pseudo_label = int(scalar(data["pseudo_label_id"][index]))
    pseudo_label_name = LABELS[pseudo_label]
    probabilities = np.asarray(data["probabilities"][index], dtype=np.float32)
    confidence = float(np.max(probabilities))

    item["manual_label"] = original_label
    item["manual_label_name"] = original_label_name
    item["manual_gt_group"] = original_group
    item["pseudo_label"] = pseudo_label
    item["pseudo_label_name"] = pseudo_label_name
    item["pseudo_label_group"] = label_group(pseudo_label_name)
    item["pseudo_confidence"] = confidence
    item["pseudo_probs"] = probabilities
    item["pseudo_teacher"] = spec.teacher
    item["pseudo_teacher_split"] = spec.pseudo_split
    item["pseudo_fold"] = spec.fold_dir
    item["pseudo_subjects"] = tuple(spec.pseudo_subjects)
    item["pseudo_source"] = f"s6_{MODE_TAGS[mode]}"
    item["pseudo_source_file"] = str(source_path)
    item["pseudo_stream"] = "joint_limb_fusion_1to1"
    item["pseudo_joint_weight"] = float(np.asarray(data.get("joint_weight", [0.5]))[0])
    item["pseudo_limb_weight"] = float(np.asarray(data.get("limb_weight", [0.5]))[0])

    if "logits" in data:
        item["pseudo_logits"] = np.asarray(data["logits"][index], dtype=np.float32)
    if mode == "mc_calibrated_soft":
        for key in (
            "predictive_entropy",
            "expected_entropy",
            "mutual_information",
            "variation_ratio",
            "mean_probability_variance",
        ):
            if key in data:
                item[f"pseudo_{key}"] = float(scalar(data[key][index]))
        if include_probability_passes and "probability_passes" in data:
            item["pseudo_probability_passes"] = np.asarray(data["probability_passes"][index], dtype=np.float32)

    item["label"] = pseudo_label
    item["label_name"] = pseudo_label_name
    item["hard_label"] = pseudo_label
    item["hard_label_name"] = pseudo_label_name
    item["gt_group"] = item["pseudo_label_group"]
    item["label_source"] = item["pseudo_source"]
    return item


def teacher_fusion_path(pseudo_root: Path, spec: TeacherSpec, mode: str) -> Path:
    return pseudo_root / spec.fold_dir / spec.teacher / "fusion_1to1" / MODE_FILES[mode]


def build_fold(
    source: dict[str, Any],
    source_by_frame_dir: dict[str, dict[str, Any]],
    specs: list[TeacherSpec],
    fold: str,
    args: argparse.Namespace,
) -> tuple[Path | None, dict[str, Any]]:
    fold_dir = f"fold_{fold}"
    split_name = args.split_template.format(fold=fold, fold_dir=fold_dir, mode=args.mode)
    missing_paths = [teacher_fusion_path(args.pseudo_root, spec, args.mode) for spec in specs if not teacher_fusion_path(args.pseudo_root, spec, args.mode).exists()]
    if missing_paths:
        summary = {
            "fold": fold_dir,
            "mode": args.mode,
            "status": "skipped_missing_fusion_files" if args.skip_incomplete_folds else "missing_fusion_files",
            "missing": [str(path) for path in missing_paths],
        }
        if args.skip_incomplete_folds:
            return None, summary
        raise FileNotFoundError(f"Missing fusion files for {fold_dir}: {summary['missing']}")

    annotations: list[dict[str, Any]] = []
    split_ids: list[str] = []
    seen: set[str] = set()
    per_teacher = Counter()
    per_subject = Counter()
    per_class = Counter()
    unmatched = []

    for spec in specs:
        path = teacher_fusion_path(args.pseudo_root, spec, args.mode)
        data = load_npz(path)
        count = len(data["frame_dir"])
        for index in range(count):
            frame_dir = frame_key_from_npz(data, index)
            if frame_dir in seen:
                raise ValueError(f"Duplicate pseudo window in {fold_dir}: {frame_dir}")
            seen.add(frame_dir)
            source_item = source_by_frame_dir.get(frame_dir)
            if source_item is None:
                unmatched.append(frame_dir)
                continue
            validate_window_match(source_item, data, index)
            item = add_pseudo_fields(
                source_item,
                data,
                index,
                spec,
                args.mode,
                path,
                args.include_probability_passes,
            )
            annotations.append(item)
            split_ids.append(str(item["frame_dir"]))
            per_teacher[spec.teacher] += 1
            per_subject[str(item.get("subject_id") or item.get("subject") or "")] += 1
            per_class[str(item["label_name"])] += 1

    if unmatched:
        preview = unmatched[:8]
        raise ValueError(
            f"{fold_dir} has {len(unmatched)} fused windows not found in {args.source_pkl}. "
            f"First examples: {preview}. Use the exact source pkl that generated pseudo inference."
        )

    output = {
        "split": {split_name: split_ids},
        "annotations": annotations,
        "labels": list(source.get("labels", LABELS)),
        "protocol": {
            "study": "s6",
            "source_pkl": str(args.source_pkl),
            "pseudo_root": str(args.pseudo_root),
            "mode": args.mode,
            "fusion": "joint_limb_1to1",
            "split": split_name,
            "fold": fold_dir,
            "teacher_splits": {
                spec.teacher: {
                    "pseudo_split": spec.pseudo_split,
                    "pseudo_subjects": list(spec.pseudo_subjects),
                    "train_subjects": list(spec.train_subjects),
                    "validation_subject": spec.val_subject,
                    "calibration_subject": spec.calibration_subject,
                    "original_test_subject": spec.original_test_subject,
                }
                for spec in specs
            },
        },
    }
    tag = MODE_TAGS[args.mode]
    out_path = args.out_dir / f"radarv4_yolo26xpose_clip60_s6_{tag}_{fold_dir}_pseudo.pkl"
    write_pickle(out_path, output, args.overwrite)
    summary = {
        "fold": fold_dir,
        "mode": args.mode,
        "status": "written",
        "output_pkl": str(out_path),
        "split_name": split_name,
        "num_samples": len(annotations),
        "samples_per_teacher": dict(sorted(per_teacher.items())),
        "samples_per_subject": dict(sorted(per_subject.items())),
        "samples_per_class": {label: int(per_class.get(label, 0)) for label in LABELS},
        "source_pkl": str(args.source_pkl),
        "pseudo_root": str(args.pseudo_root),
    }
    write_json(out_path.with_name(f"{out_path.stem}_summary.json"), summary, args.overwrite)
    return out_path, summary


def specs_by_fold(folds: list[str]) -> dict[str, list[TeacherSpec]]:
    specs = selected_specs(folds, ["t1", "t2", "t3", "t4"])
    grouped: dict[str, list[TeacherSpec]] = defaultdict(list)
    for spec in specs:
        grouped[spec.fold].append(spec)
    return {
        fold: sorted(items, key=lambda item: item.teacher)
        for fold, items in grouped.items()
    }


def main() -> None:
    args = parse_args()
    source = read_pickle(args.source_pkl)
    source_by_frame_dir = source_index(source)
    all_summaries = []
    for fold, specs in sorted(specs_by_fold(args.folds).items()):
        _, summary = build_fold(source, source_by_frame_dir, specs, fold, args)
        all_summaries.append(summary)
        print(f"[INFO] {summary['fold']} {summary['status']}: {summary.get('num_samples', 0)} samples")
    combined = {
        "mode": args.mode,
        "source_pkl": str(args.source_pkl),
        "pseudo_root": str(args.pseudo_root),
        "folds": all_summaries,
    }
    write_json(args.out_dir / f"radarv4_yolo26xpose_clip60_s6_{MODE_TAGS[args.mode]}_pseudo_summary.json", combined, args.overwrite)


if __name__ == "__main__":
    main()

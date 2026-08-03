"""Create S6 teacher-specific continuous split keys from the S2 window pkl."""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .common import (
        DEFAULT_CONTINUOUS_SOURCE_PKL,
        DEFAULT_CONTINUOUS_TEACHER_PKL,
        LABELS,
        TeacherSpec,
        selected_specs,
        write_json,
    )
except ImportError:
    from common import (
        DEFAULT_CONTINUOUS_SOURCE_PKL,
        DEFAULT_CONTINUOUS_TEACHER_PKL,
        LABELS,
        TeacherSpec,
        selected_specs,
        write_json,
    )


def annotation_subject(item: dict[str, Any]) -> str:
    subject = item.get("subject_id", item.get("subject"))
    if subject is None:
        raise KeyError(f"Annotation {item.get('frame_dir', '<unknown>')} has no subject field")
    return str(subject).lower()


def annotation_label(item: dict[str, Any]) -> str:
    label_name = item.get("label_name")
    if isinstance(label_name, str) and label_name in LABELS:
        return label_name
    label_id = item.get("hard_label", item.get("label"))
    return LABELS[int(label_id)]


def make_split_for_spec(
    spec: TeacherSpec,
    annotations: list[dict[str, Any]],
) -> dict[str, list[str]]:
    subjects = {
        "train": set(spec.train_subjects),
        "val": {spec.val_subject},
        "calib": {spec.calibration_subject},
        "pseudo": set(spec.pseudo_subjects),
    }
    seen: dict[str, str] = {}
    for split_name, split_subjects in subjects.items():
        for subject in split_subjects:
            if subject in seen:
                raise ValueError(
                    f"{spec.fold_dir} {spec.teacher}: subject {subject!r} is in both "
                    f"{seen[subject]} and {split_name}"
                )
            seen[subject] = split_name

    split = {"train": [], "val": [], "calib": [], "pseudo": [], "unused": []}
    for item in annotations:
        subject = annotation_subject(item)
        frame_dir = str(item["frame_dir"])
        if subject in subjects["train"]:
            split["train"].append(frame_dir)
        elif subject in subjects["val"]:
            split["val"].append(frame_dir)
        elif subject in subjects["calib"]:
            split["calib"].append(frame_dir)
        elif subject in subjects["pseudo"]:
            split["pseudo"].append(frame_dir)
        else:
            split["unused"].append(frame_dir)
    return split


def count_by_class(
    annotations_by_id: dict[str, dict[str, Any]],
    frame_dirs: list[str],
) -> dict[str, int]:
    counts = Counter(annotation_label(annotations_by_id[frame_dir]) for frame_dir in frame_dirs)
    return {label: int(counts.get(label, 0)) for label in LABELS}


def count_by_subject(
    annotations_by_id: dict[str, dict[str, Any]],
    frame_dirs: list[str],
) -> dict[str, int]:
    counts = Counter(annotation_subject(annotations_by_id[frame_dir]) for frame_dir in frame_dirs)
    return dict(sorted((subject, int(count)) for subject, count in counts.items()))


def build_continuous_teacher_splits(
    source_pkl: Path,
    output_pkl: Path,
    specs: list[TeacherSpec],
    overwrite: bool,
) -> dict[str, Any]:
    if output_pkl.exists() and not overwrite:
        raise FileExistsError(f"{output_pkl} exists; pass --overwrite")
    with source_pkl.open("rb") as handle:
        data = pickle.load(handle)

    annotations = data.get("annotations", [])
    if not annotations:
        raise ValueError(f"{source_pkl} does not contain annotations")
    annotations_by_id = {str(item["frame_dir"]): item for item in annotations}

    split = dict(data.get("split", {}))
    spec_summaries: dict[str, Any] = {}
    for spec in specs:
        spec_split = make_split_for_spec(spec, annotations)
        split[spec.train_split] = spec_split["train"]
        split[spec.val_split] = spec_split["val"]
        split[spec.calib_split] = spec_split["calib"]
        split[spec.pseudo_split] = spec_split["pseudo"]
        split[spec.test_split] = spec_split["pseudo"]

        key = f"{spec.fold_dir}_{spec.teacher}"
        spec_summaries[key] = {
            "fold": spec.fold,
            "teacher": spec.teacher,
            "train_subjects": list(spec.train_subjects),
            "val_subject": spec.val_subject,
            "calibration_subject": spec.calibration_subject,
            "pseudo_subjects": list(spec.pseudo_subjects),
            "original_test_subject": spec.original_test_subject,
            "split_keys": {
                "train": spec.train_split,
                "val": spec.val_split,
                "calib": spec.calib_split,
                "pseudo": spec.pseudo_split,
                "test_alias": spec.test_split,
            },
            "split_counts": {name: len(ids) for name, ids in spec_split.items()},
            "samples_per_class_per_split": {
                name: count_by_class(annotations_by_id, ids)
                for name, ids in spec_split.items()
            },
            "samples_per_subject_per_split": {
                name: count_by_subject(annotations_by_id, ids)
                for name, ids in spec_split.items()
            },
        }

    data["split"] = split
    protocol = data.get("protocol")
    if isinstance(protocol, dict):
        protocol = dict(protocol)
        protocol["s6_teacher4_splits"] = {
            "enabled": True,
            "source": str(source_pkl),
            "policy": "Each teacher trains on 6 outer-train subjects and holds out 2 pseudo-label target subjects.",
        }
        data["protocol"] = protocol

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "stage": "s6_continuous_teacher_splits",
        "source_pkl": str(source_pkl),
        "output_pkl": str(output_pkl),
        "num_annotations": len(annotations),
        "labels": LABELS,
        "teacher_splits": spec_summaries,
        "source_protocol": data.get("protocol", {}),
    }
    summary_path = output_pkl.with_name(f"{output_pkl.stem}_summary.json")
    write_json(summary_path, summary, overwrite=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build S6 teacher split keys on top of the S2 continuous pkl.")
    parser.add_argument("--source-pkl", type=Path, default=DEFAULT_CONTINUOUS_SOURCE_PKL)
    parser.add_argument("--output-pkl", type=Path, default=DEFAULT_CONTINUOUS_TEACHER_PKL)
    parser.add_argument("--folds", nargs="+", help="Fold ids to include. Default: a b c.")
    parser.add_argument("--teachers", nargs="+", help="Teacher ids to include. Default: t1 t2 t3 t4.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_pkl.exists():
        raise FileNotFoundError(args.source_pkl)
    specs = selected_specs(args.folds, args.teachers)
    summary = build_continuous_teacher_splits(
        source_pkl=args.source_pkl,
        output_pkl=args.output_pkl,
        specs=specs,
        overwrite=args.overwrite,
    )
    print(f"[DONE] wrote S6 continuous teacher pkl: {args.output_pkl}")
    print(f"[DONE] teacher split groups: {len(summary['teacher_splits'])}")


if __name__ == "__main__":
    main()

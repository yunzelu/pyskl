"""Concatenate inner-teacher pseudo labels into canonical fold-level OOF files.

This is step 6 of the pseudo-labeling protocol. It combines the four
teacher-level pseudo-target files within one outer fold, validates the
cross-fitting partition, and writes:

- oof_skeleton_pseudo_labels.parquet
- oof_skeleton_pseudo_labels_audit.parquet
- radar_teacher_alignment.parquet

The first and third files are radar-training-safe and do not contain manual
activity labels. The audit file contains manual labels and correctness fields
for diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rerun.pseudo_labeling.run_inner_teacher_oof_pseudo_labeling import (  # noqa: E402
    DATASET_ID,
    FOLDS,
    ID_TO_LABEL,
    LABELS,
    SCHEMA_VERSION,
    WINDOW_SIZE,
    write_csv,
    write_global_metadata_files,
    write_json,
    write_parquet,
    read_parquet,
)


TRAINING_SAFE_FORBIDDEN_FIELDS = {
    "manual_label_at_skeleton_center",
    "manual_label_name_at_skeleton_center",
    "manual_center_raw_label",
    "manual_segment_index",
    "manual_segment_start_frame",
    "manual_segment_end_frame",
    "manual_boundary_source",
    "source_pyskl_frame_dir",
    "frame_dir",
    "pseudo_label_correct",
    "distance_to_manual_boundary_frames",
}

ALIGNMENT_COLUMNS = [
    "schema_version",
    "dataset_id",
    "outer_fold",
    "inner_teacher_id",
    "subject_id",
    "recording_id",
    "skeleton_sample_id",
    "window_candidate_index",
    "accepted_window_index_in_recording",
    "window_start_retained_idx",
    "center_retained_idx",
    "window_end_retained_idx_exclusive",
    "source_frame_start",
    "source_frame_center",
    "source_frame_end",
    "source_timestamp_start_sec",
    "source_timestamp_center_sec",
    "source_timestamp_end_sec",
    "center_timestamp_sec",
    "nominal_camera_time_center_sec",
    "camera_fps_for_nominal_time",
    "center_timestamp_policy",
    "max_adjacent_gap_sec",
    "window_span_sec",
    "hard_pseudo_label_id",
    "hard_pseudo_label_name",
    "mc_predictive_entropy",
    "mc_expected_entropy",
    "mc_mi_raw",
    "temperature",
    "calibrated_argmax_id",
    "calibrated_argmax_name",
    "mi_q95_calibration",
    "reliability_weight",
] + [f"mc_raw_p{index}" for index in range(len(LABELS))] + [
    f"mc_cal_p{index}" for index in range(len(LABELS))
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def teacher_root(output_root: Path, fold: str, teacher: str) -> Path:
    return output_root / f"fold_{fold}" / teacher


def write_rows(path: Path, rows: list[dict[str, Any]], write_csv_copy: bool) -> None:
    write_parquet(path, rows)
    if write_csv_copy:
        write_csv(path.with_suffix(".csv"), rows)


def sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["subject_id"]),
            str(row["recording_id"]),
            int(row["window_candidate_index"]),
            int(row["window_start_retained_idx"]),
            int(row["source_frame_center"]),
            str(row["inner_teacher_id"]),
        ),
    )


def assert_no_manual_fields(rows: list[dict[str, Any]], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} is empty")
    present = TRAINING_SAFE_FORBIDDEN_FIELDS & set().union(*(set(row) for row in rows))
    if present:
        raise RuntimeError(f"{name} contains manual/audit-only fields: {sorted(present)}")


def assert_unique(rows: list[dict[str, Any]], key: str, name: str) -> None:
    values = [str(row[key]) for row in rows]
    duplicates = len(values) - len(set(values))
    if duplicates:
        raise RuntimeError(f"{name} contains {duplicates} duplicate {key} values")


def source_window_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["subject_id"],
        row["recording_id"],
        int(row["window_start_retained_idx"]),
        int(row["source_frame_center"]),
    )


def validate_fold_rows(
    fold: str,
    safe_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    teacher_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fold_spec = FOLDS[fold]
    train_pool = set(fold_spec["train_pool"])
    val_subjects = set(fold_spec["val"])
    calib_subjects = set(fold_spec["calib"])
    test_subjects = set(fold_spec["test"])
    forbidden_subjects = val_subjects | calib_subjects | test_subjects

    assert_no_manual_fields(safe_rows, f"fold {fold} training-safe rows")
    assert_unique(safe_rows, "skeleton_sample_id", f"fold {fold} training-safe rows")
    assert_unique(audit_rows, "skeleton_sample_id", f"fold {fold} audit rows")
    if {row["skeleton_sample_id"] for row in safe_rows} != {row["skeleton_sample_id"] for row in audit_rows}:
        raise RuntimeError(f"fold {fold}: safe and audit sample-id sets differ")

    source_keys = [source_window_key(row) for row in safe_rows]
    source_duplicates = len(source_keys) - len(set(source_keys))
    if source_duplicates:
        raise RuntimeError(f"fold {fold}: duplicate source skeleton windows={source_duplicates}")

    subject_to_teachers: dict[str, set[str]] = defaultdict(set)
    teacher_counts = Counter()
    subject_counts = Counter()
    pseudo_label_counts = Counter()

    for row in safe_rows:
        if str(row["schema_version"]) != SCHEMA_VERSION:
            raise RuntimeError(f"fold {fold}: unexpected schema version {row['schema_version']}")
        if str(row["dataset_id"]) != DATASET_ID:
            raise RuntimeError(f"fold {fold}: unexpected dataset id {row['dataset_id']}")
        if str(row["outer_fold"]).lower() != fold:
            raise RuntimeError(f"fold {fold}: row has outer_fold={row['outer_fold']}")

        subject = str(row["subject_id"]).lower()
        teacher = str(row["inner_teacher_id"]).lower()
        if subject in forbidden_subjects:
            raise RuntimeError(f"fold {fold}: forbidden subject {subject} appears in pseudo labels")
        if subject not in train_pool:
            raise RuntimeError(f"fold {fold}: non-training subject {subject} appears in pseudo labels")
        teacher_spec = FOLDS[fold]["teachers"][teacher]
        if subject not in set(teacher_spec["pseudo_target"]):
            raise RuntimeError(
                f"fold {fold} {teacher}: subject {subject} was not a pseudo-target subject"
            )
        if subject in set(teacher_spec["train"]):
            raise RuntimeError(f"fold {fold} {teacher}: subject {subject} leaked from teacher training")

        subject_to_teachers[subject].add(teacher)
        teacher_counts[teacher] += 1
        subject_counts[subject] += 1
        pseudo_label_counts[int(row["hard_pseudo_label_id"])] += 1

    if set(subject_to_teachers) != train_pool:
        raise RuntimeError(
            f"fold {fold}: pseudo subjects do not equal train pool. "
            f"got={sorted(subject_to_teachers)}, expected={sorted(train_pool)}"
        )
    repeated_subjects = {
        subject: sorted(teachers)
        for subject, teachers in subject_to_teachers.items()
        if len(teachers) != 1
    }
    if repeated_subjects:
        raise RuntimeError(f"fold {fold}: subjects covered by multiple teachers: {repeated_subjects}")

    expected_total = 0
    expected_teacher_counts = {}
    for teacher, teacher_spec in FOLDS[fold]["teachers"].items():
        expected = int(teacher_spec["expected_pseudo_target_windows"])
        expected_total += expected
        expected_teacher_counts[teacher] = expected
        if teacher_counts[teacher] != expected:
            raise RuntimeError(
                f"fold {fold} {teacher}: row count {teacher_counts[teacher]} != expected {expected}"
            )
        metadata = teacher_metadata.get(teacher)
        if metadata is None:
            raise RuntimeError(f"fold {fold}: missing metadata for {teacher}")
        if set(metadata["pseudo_target_subjects"]) != set(teacher_spec["pseudo_target"]):
            raise RuntimeError(f"fold {fold} {teacher}: metadata pseudo-target subjects mismatch")

    if len(safe_rows) != expected_total:
        raise RuntimeError(f"fold {fold}: total rows {len(safe_rows)} != expected {expected_total}")

    return {
        "fold": fold,
        "schema_version": SCHEMA_VERSION,
        "total_rows": len(safe_rows),
        "expected_total_rows": expected_total,
        "subjects": sorted(subject_to_teachers),
        "subject_to_teacher": {subject: sorted(teachers)[0] for subject, teachers in subject_to_teachers.items()},
        "subject_counts": dict(sorted(subject_counts.items())),
        "teacher_counts": dict(sorted(teacher_counts.items())),
        "expected_teacher_counts": expected_teacher_counts,
        "hard_pseudo_label_counts": {
            ID_TO_LABEL[index]: int(pseudo_label_counts.get(index, 0))
            for index in range(len(LABELS))
        },
        "validation_subject_absent": sorted(val_subjects),
        "calibration_subject_absent": sorted(calib_subjects),
        "outer_test_subject_absent": sorted(test_subjects),
    }


def alignment_rows_from_safe_rows(safe_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in safe_rows:
        rows.append({column: row.get(column) for column in ALIGNMENT_COLUMNS})
    return rows


def fold_summary_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for subject, count in summary["subject_counts"].items():
        rows.append(
            {
                "fold": summary["fold"],
                "group": "subject",
                "name": subject,
                "count": count,
            }
        )
    for teacher, count in summary["teacher_counts"].items():
        rows.append(
            {
                "fold": summary["fold"],
                "group": "teacher",
                "name": teacher,
                "count": count,
            }
        )
    for label, count in summary["hard_pseudo_label_counts"].items():
        rows.append(
            {
                "fold": summary["fold"],
                "group": "hard_pseudo_label",
                "name": label,
                "count": count,
            }
        )
    return rows


def aggregate_fold(args: argparse.Namespace, fold: str) -> dict[str, Any]:
    fold = fold.lower()
    fold_dir = args.output_root / f"fold_{fold}"
    safe_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    metadata_by_teacher: dict[str, dict[str, Any]] = {}

    for teacher in sorted(FOLDS[fold]["teachers"]):
        root = teacher_root(args.output_root, fold, teacher)
        safe_path = root / "pseudo_predictions.parquet"
        audit_path = root / "pseudo_predictions_audit.parquet"
        metadata_path = root / "teacher_metadata.json"
        for path in [safe_path, audit_path, metadata_path, root / "mc_fused_samples.npz"]:
            if not path.exists():
                raise FileNotFoundError(path)

        safe_rows.extend(read_parquet(safe_path))
        audit_rows.extend(read_parquet(audit_path))
        metadata_by_teacher[teacher] = read_json(metadata_path)

    safe_rows = sorted_rows(safe_rows)
    audit_rows = sorted_rows(audit_rows)
    summary = validate_fold_rows(fold, safe_rows, audit_rows, metadata_by_teacher)
    alignment_rows = alignment_rows_from_safe_rows(safe_rows)
    assert_no_manual_fields(alignment_rows, f"fold {fold} radar alignment rows")
    assert_unique(alignment_rows, "skeleton_sample_id", f"fold {fold} radar alignment rows")

    write_global_metadata_files(args.output_root)
    write_rows(fold_dir / "oof_skeleton_pseudo_labels.parquet", safe_rows, args.write_csv_copy)
    write_rows(fold_dir / "oof_skeleton_pseudo_labels_audit.parquet", audit_rows, args.write_csv_copy)
    write_rows(fold_dir / "radar_teacher_alignment.parquet", alignment_rows, args.write_csv_copy)
    write_json(
        fold_dir / "fold_metadata.json",
        {
            "protocol": "OOF skeleton pseudo-label fold aggregation",
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "fold": fold.upper(),
            "window_size": WINDOW_SIZE,
            "safe_output": fold_dir / "oof_skeleton_pseudo_labels.parquet",
            "audit_output": fold_dir / "oof_skeleton_pseudo_labels_audit.parquet",
            "alignment_output": fold_dir / "radar_teacher_alignment.parquet",
            "teacher_metadata": metadata_by_teacher,
            "validation_checks": summary,
            "training_safe_files_exclude_manual_labels": True,
        },
    )
    write_csv(args.report_dir / f"fold_{fold}_oof_pseudo_label_counts.csv", fold_summary_csv_rows(summary))
    print(f"[DONE] fold={fold} rows={summary['total_rows']} -> {fold_dir}")
    return summary


def run(args: argparse.Namespace) -> None:
    folds = [fold.lower() for fold in args.folds]
    for fold in folds:
        if fold not in FOLDS:
            raise ValueError(f"Unknown fold {fold!r}; expected one of {sorted(FOLDS)}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    summaries = [aggregate_fold(args, fold) for fold in folds]
    combined_rows = []
    for summary in summaries:
        combined_rows.extend(fold_summary_csv_rows(summary))
    write_csv(args.report_dir / "oof_pseudo_label_counts.csv", combined_rows)
    write_json(
        args.report_dir / "oof_pseudo_label_aggregation_summary.json",
        {
            "protocol": "OOF skeleton pseudo-label aggregation",
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "output_root": args.output_root,
            "folds": folds,
            "summaries": summaries,
            "training_safe_file": "fold_<fold>/oof_skeleton_pseudo_labels.parquet",
            "audit_file": "fold_<fold>/oof_skeleton_pseudo_labels_audit.parquet",
            "radar_alignment_manifest": "fold_<fold>/radar_teacher_alignment.parquet",
        },
    )
    print(f"[DONE] wrote aggregation reports under {args.report_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", default=sorted(FOLDS))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/radar_v4/rerun/yolo26xpose/pseudo_labels_v1"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("rerun/pseudo_labeling/reports/oof_pseudo_labels_v1"),
    )
    parser.add_argument("--write-csv-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

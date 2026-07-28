"""Study 2 pre-flight sanity checks."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import (
        CENTER_OFFSET,
        DEFAULT_ETAS,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_S2_PKL,
        LABELS,
        MIN_VALID_FRAMES_AFTER_ZERO_FILTER,
        STRIDE,
        WINDOW_SIZE,
        ZERO_FRAME_EPS,
        default_prediction_path,
        discover_s2_folds,
        eta_slug,
        protocol_metadata,
        read_json,
        s2_config_path,
        write_json,
    )
except ImportError:
    from common import (
        CENTER_OFFSET,
        DEFAULT_ETAS,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_S2_PKL,
        LABELS,
        MIN_VALID_FRAMES_AFTER_ZERO_FILTER,
        STRIDE,
        WINDOW_SIZE,
        ZERO_FRAME_EPS,
        default_prediction_path,
        discover_s2_folds,
        eta_slug,
        protocol_metadata,
        read_json,
        s2_config_path,
        write_json,
    )


def pass_record(name: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "pass", "details": details or {}}


def skip_record(name: str, reason: str) -> dict[str, Any]:
    return {"name": name, "status": "skip", "reason": reason}


def fail_record(name: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": "fail", "details": details}


def assert_or_record(condition: bool, name: str, details: dict[str, Any]) -> dict[str, Any]:
    return pass_record(name, details) if condition else fail_record(name, details)


def load_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def check_subject_splits() -> dict[str, Any]:
    details = {}
    ok = True
    for fold in discover_s2_folds():
        split_sets = {
            "train": set(fold.train_subjects),
            "val": {fold.val_subject},
            "test": {fold.test_subject},
        }
        if fold.calibration_subject is not None:
            split_sets["calib"] = {fold.calibration_subject}
        overlap = []
        split_names = sorted(split_sets)
        for i, first in enumerate(split_names):
            for second in split_names[i + 1:]:
                common = sorted(split_sets[first] & split_sets[second])
                if common:
                    overlap.append({"splits": [first, second], "subjects": common})
        ok = ok and not overlap
        details[f"fold_{fold.fold}"] = {
            "subjects": {name: sorted(values) for name, values in split_sets.items()},
            "overlap": overlap,
        }
    return assert_or_record(ok, "No subject appears in more than one split", details)


def check_pkl_windows(path: Path, etas: tuple[float, ...]) -> list[dict[str, Any]]:
    if not path.exists():
        return [skip_record("Continuous-window pkl checks", f"missing {path}")]
    data = load_pkl(path)
    annotations = data.get("annotations", [])
    split = data.get("split", {})
    by_id = {item["frame_dir"]: item for item in annotations}
    failures = []

    for index, item in enumerate(annotations):
        if int(item["end_frame"]) - int(item["start_frame"]) + 1 != WINDOW_SIZE:
            failures.append({"index": index, "reason": "source_window_length", "frame_dir": item["frame_dir"]})
        if int(item["center_frame"]) != int(item["start_frame"]) + CENTER_OFFSET:
            failures.append({"index": index, "reason": "center_frame", "frame_dir": item["frame_dir"]})
        total_frames = int(item["total_frames"])
        keypoint = item["keypoint"]
        keypoint_score = item["keypoint_score"]
        if total_frames < MIN_VALID_FRAMES_AFTER_ZERO_FILTER:
            failures.append({"index": index, "reason": "total_frames_below_min_valid", "frame_dir": item["frame_dir"]})
        if keypoint.shape[1] != total_frames:
            failures.append({"index": index, "reason": "keypoint_shape", "frame_dir": item["frame_dir"]})
        if keypoint_score.shape[1] != total_frames:
            failures.append({"index": index, "reason": "keypoint_score_shape", "frame_dir": item["frame_dir"]})
        keypoint_nonzero = np.any(np.abs(keypoint[0]) > ZERO_FRAME_EPS, axis=(1, 2))
        score_nonzero = np.any(np.abs(keypoint_score[0]) > ZERO_FRAME_EPS, axis=1)
        if not np.all(np.logical_or(keypoint_nonzero, score_nonzero)):
            failures.append({"index": index, "reason": "zero_pose_frame_retained", "frame_dir": item["frame_dir"]})
        if len(item.get("source_frame_indices", [])) != total_frames:
            failures.append({"index": index, "reason": "source_frame_indices_length", "frame_dir": item["frame_dir"]})
        center_label = int(item["per_frame_label_ids"][CENTER_OFFSET])
        if center_label != int(item["hard_label"]):
            failures.append({"index": index, "reason": "center_label", "frame_dir": item["frame_dir"]})
        for key in ["q_temporal", *[f"target_probs_{eta_slug(eta)}" for eta in etas]]:
            target = np.asarray(item[key], dtype=np.float32)
            if target.shape != (len(LABELS),):
                failures.append({"index": index, "reason": f"{key}_shape", "frame_dir": item["frame_dir"]})
            if not np.all(np.isfinite(target)):
                failures.append({"index": index, "reason": f"{key}_finite", "frame_dir": item["frame_dir"]})
            if np.any(target < -1e-5):
                failures.append({"index": index, "reason": f"{key}_negative", "frame_dir": item["frame_dir"]})
            if abs(float(target.sum()) - 1.0) >= 1e-5:
                failures.append({"index": index, "reason": f"{key}_sum", "frame_dir": item["frame_dir"]})

    split_failures = []
    for fold in discover_s2_folds():
        keys = {
            name: set(split.get(fold.split_key(name), []))
            for name in ("train", "val", "calib", "test")
        }
        for split_name, ids in keys.items():
            missing = sorted(item_id for item_id in ids if item_id not in by_id)
            if missing:
                split_failures.append(
                    {
                        "fold": fold.fold,
                        "split": split_name,
                        "reason": "missing_annotation",
                        "count": len(missing),
                    }
                )
        for first, second in (("train", "val"), ("train", "test"), ("val", "test"), ("calib", "test")):
            overlap = keys[first] & keys[second]
            if overlap:
                split_failures.append(
                    {
                        "fold": fold.fold,
                        "splits": [first, second],
                        "reason": "split_window_overlap",
                        "count": len(overlap),
                    }
                )

    return [
        assert_or_record(
            not failures,
            "Continuous windows use 60-frame source windows, filtered nonzero pose tensors, and valid center labels",
            {
                "num_annotations": len(annotations),
                "min_valid_frames": MIN_VALID_FRAMES_AFTER_ZERO_FILTER,
                "failures": failures[:10],
            },
        ),
        assert_or_record(
            not split_failures,
            "PYSKL split keys are disjoint and reference existing annotations",
            {"failures": split_failures[:10]},
        ),
    ]


def check_eta_zero_loss() -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from pyskl.models.losses.cross_entropy_loss import soft_cross_entropy
    except Exception as exc:  # pragma: no cover
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(6, len(LABELS))).astype(np.float64)
        labels = np.asarray([0, 1, 2, 3, 4, 8], dtype=np.int64)
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        one_hot = np.eye(len(LABELS), dtype=np.float64)[labels]
        soft_loss = float(-(one_hot * log_probs).sum(axis=1).mean())
        hard_loss = float(-log_probs[np.arange(labels.shape[0]), labels].mean())
        diff = abs(soft_loss - hard_loss)
        return assert_or_record(
            diff < 1e-12,
            "eta=0 soft loss matches hard-label cross entropy",
            {
                "soft_loss": soft_loss,
                "hard_loss": hard_loss,
                "abs_diff": diff,
                "implementation": "numpy fallback",
                "torch_import": f"skipped: {exc}",
            },
        )
    else:
        torch.manual_seed(7)
        logits = torch.randn(6, len(LABELS), dtype=torch.float32)
        labels = torch.tensor([0, 1, 2, 3, 4, 8], dtype=torch.long)
        one_hot = F.one_hot(labels, num_classes=len(LABELS)).float()
        soft_loss = soft_cross_entropy(logits, one_hot)
        hard_loss = F.cross_entropy(logits, labels)
        diff = float(torch.abs(soft_loss - hard_loss).item())
        return assert_or_record(
            diff < 1e-6,
            "eta=0 soft loss matches hard-label cross entropy",
            {"soft_loss": float(soft_loss.item()), "hard_loss": float(hard_loss.item()), "abs_diff": diff},
        )


def filter_folds(fold_args: list[str] | None):
    folds = discover_s2_folds()
    if not fold_args:
        return folds
    requested = {item.lower().replace("fold_", "") for item in fold_args}
    folds = [fold for fold in folds if fold.fold in requested]
    missing = sorted(requested - {fold.fold for fold in folds})
    if missing:
        raise ValueError(f"Unknown fold(s): {missing}")
    return folds


def check_stage2_sampling_configs(
    folds,
    streams: list[str],
    etas: tuple[float, ...],
    expected_strategy: str,
    expected_power: float,
) -> dict[str, Any]:
    failures = []
    checked = []
    for fold in folds:
        for stream in streams:
            paths = [s2_config_path("B", fold.fold, stream)]
            paths.extend(s2_config_path("C", fold.fold, stream, eta) for eta in etas)
            for path in paths:
                record = {"path": str(path), "fold": fold.fold, "stream": stream}
                if not path.exists():
                    failures.append({**record, "reason": "missing_config"})
                    continue
                text = path.read_text(encoding="utf-8")
                checked.append(str(path))
                if "class_prob" in text:
                    failures.append({**record, "reason": "uses_legacy_class_prob"})
                if expected_strategy == "none":
                    if "class_sample_strategy" in text or "class_sample_power" in text:
                        failures.append({**record, "reason": "unexpected_class_sample_strategy"})
                    continue
                expected_strategy_line = f"class_sample_strategy = {expected_strategy!r}"
                if expected_strategy_line not in text:
                    failures.append(
                        {
                            **record,
                            "reason": "missing_or_wrong_class_sample_strategy",
                            "expected": expected_strategy_line,
                        }
                    )
                expected_power_line = f"class_sample_power = {expected_power:.12g}"
                if expected_power_line not in text:
                    failures.append(
                        {
                            **record,
                            "reason": "missing_or_wrong_class_sample_power",
                            "expected": expected_power_line,
                        }
                    )
                if "epoch_size =" not in text:
                    failures.append({**record, "reason": "missing_epoch_size"})
                if "class_sample_strategy=class_sample_strategy" not in text:
                    failures.append({**record, "reason": "train_dataset_missing_strategy_kwarg"})
                if "class_sample_power=class_sample_power" not in text:
                    failures.append({**record, "reason": "train_dataset_missing_power_kwarg"})
                if "epoch_size=epoch_size" not in text:
                    failures.append({**record, "reason": "train_dataset_missing_epoch_size_kwarg"})
    return assert_or_record(
        not failures,
        "Stage-2 configs use epoch-wise class sampling, not class_prob",
        {
            "checked_configs": checked,
            "expected_strategy": expected_strategy,
            "expected_power": expected_power,
            "failures": failures[:20],
        },
    )


def prediction_keys(path: Path) -> set[tuple[str, str, int, int, int]]:
    keys = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            keys.add(
                (
                    str(row.get("fold") or ""),
                    str(row.get("recording_id") or ""),
                    int(float(row.get("start_frame") or -1)),
                    int(float(row.get("end_frame") or -1)),
                    int(float(row.get("center_frame") or -1)),
                )
            )
    return keys


def check_prediction_alignment(stream: str) -> dict[str, Any]:
    paths = {method: default_prediction_path(method, stream) for method in ("A", "B", "C")}
    missing = [method for method, path in paths.items() if not path.exists()]
    if missing:
        return skip_record("A/B/C methods use exactly the same test windows", f"missing predictions: {missing}")
    key_sets = {method: prediction_keys(path) for method, path in paths.items()}
    base = key_sets["A"]
    details = {
        method: {
            "rows": len(keys),
            "missing_from_A": len(base - keys),
            "extra_vs_A": len(keys - base),
        }
        for method, keys in key_sets.items()
    }
    ok = all(keys == base for keys in key_sets.values())
    return assert_or_record(ok, "A/B/C methods use exactly the same test windows", details)


def recording_scope_for_pkl(path: Path) -> str | None:
    summary_path = path.with_name(f"{path.stem}_summary.json")
    if summary_path.exists():
        return str(read_json(summary_path).get("recording_scope") or "")
    if path.exists():
        data = load_pkl(path)
        protocol = data.get("protocol", {}) if isinstance(data, dict) else {}
        return str(protocol.get("default_recording_scope") or "")
    return None


def zero_frame_filter_enabled_for_pkl(path: Path) -> bool:
    summary_path = path.with_name(f"{path.stem}_summary.json")
    if summary_path.exists():
        summary = read_json(summary_path)
        zero_filter = summary.get("zero_frame_filter", {})
        if isinstance(zero_filter, dict):
            return bool(zero_filter.get("enabled"))
        protocol = summary.get("protocol", {})
        if isinstance(protocol, dict):
            protocol_filter = protocol.get("zero_frame_filter", {})
            if isinstance(protocol_filter, dict):
                return bool(protocol_filter.get("enabled"))
    if path.exists():
        data = load_pkl(path)
        protocol = data.get("protocol", {}) if isinstance(data, dict) else {}
        zero_filter = protocol.get("zero_frame_filter", {}) if isinstance(protocol, dict) else {}
        if isinstance(zero_filter, dict):
            return bool(zero_filter.get("enabled"))
    return False


def check_e2_reproduction(
    stream: str,
    s2_metrics: Path,
    e2_summary: Path,
    tolerance: float,
    recording_scope: str | None,
    zero_frame_filter_enabled: bool,
) -> dict[str, Any]:
    if zero_frame_filter_enabled:
        return skip_record(
            "Baseline A reproduces E2 center metrics",
            (
                "S2 now removes all-zero pose frames and drops sparse source "
                "windows, so its center-time window set is intentionally not "
                "the legacy E2 window set."
            ),
        )
    if recording_scope and "non-walk" not in recording_scope.lower():
        return skip_record(
            "Baseline A reproduces E2 center metrics",
            (
                f"S2 recording_scope={recording_scope!r}; the E2 reference "
                "is only comparable for the older non-walk evaluation scope"
            ),
        )
    if not s2_metrics.exists():
        return skip_record("Baseline A reproduces E2 center metrics", f"missing {s2_metrics}")
    if not e2_summary.exists():
        return skip_record("Baseline A reproduces E2 center metrics", f"missing {e2_summary}")
    s2 = read_json(s2_metrics)["overall"]
    with e2_summary.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    overall = next((row for row in rows if row.get("scope") == "overall"), None)
    if overall is None:
        return fail_record("Baseline A reproduces E2 center metrics", {"reason": "E2 overall row missing"})
    acc_diff = abs(float(s2["center_acc"]) - float(overall["center_time_accuracy_percent"]))
    macro_diff = abs(float(s2["center_macro_f1"]) - float(overall["macro_f1_percent"]))
    return assert_or_record(
        acc_diff <= tolerance and macro_diff <= tolerance,
        "Baseline A reproduces E2 center metrics",
        {
            "stream": stream,
            "tolerance": tolerance,
            "center_acc_diff": acc_diff,
            "macro_f1_diff": macro_diff,
            "s2_metrics": str(s2_metrics),
            "e2_summary": str(e2_summary),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S2 sanity checks.")
    parser.add_argument("--pkl", type=Path, default=DEFAULT_S2_PKL)
    parser.add_argument("--folds", nargs="+", help="Fold ids to include for config checks. Default: all.")
    parser.add_argument("--streams", nargs="+", choices=["joint", "limb"], default=["joint"])
    parser.add_argument("--etas", nargs="+", type=float, default=list(DEFAULT_ETAS))
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument(
        "--expected-class-sample-strategy",
        choices=["sqrt", "power", "none"],
        default="sqrt",
    )
    parser.add_argument("--expected-class-sample-power", type=float, default=0.5)
    parser.add_argument("--s2-a-metrics", type=Path, default=DEFAULT_OUTPUT_DIR / "eval" / "metrics_A.json")
    parser.add_argument(
        "--e2-summary",
        type=Path,
        default=Path("work_dirs/thesis/e2/eval/e2_joint_scores_summary.csv"),
    )
    parser.add_argument("--e2-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--skip-prediction-checks",
        action="store_true",
        help="Skip checks that require regenerated A/B/C prediction CSVs.",
    )
    parser.add_argument(
        "--skip-e2-reproduction",
        action="store_true",
        help="Skip the legacy E2 reproduction comparison.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "sanity_checks.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = filter_folds(args.folds)
    recording_scope = recording_scope_for_pkl(args.pkl)
    zero_frame_filter_enabled = zero_frame_filter_enabled_for_pkl(args.pkl)
    checks = [check_subject_splits()]
    checks.extend(check_pkl_windows(args.pkl, tuple(args.etas)))
    checks.append(
        check_stage2_sampling_configs(
            folds=folds,
            streams=args.streams,
            etas=tuple(args.etas),
            expected_strategy=args.expected_class_sample_strategy,
            expected_power=args.expected_class_sample_power,
        )
    )
    checks.append(check_eta_zero_loss())
    if args.skip_prediction_checks:
        checks.append(
            skip_record(
                "A/B/C methods use exactly the same test windows",
                "skipped by --skip-prediction-checks",
            )
        )
    else:
        checks.append(check_prediction_alignment(args.stream))
    if args.skip_e2_reproduction:
        checks.append(
            skip_record(
                "Baseline A reproduces E2 center metrics",
                "skipped by --skip-e2-reproduction",
            )
        )
    else:
        checks.append(
            check_e2_reproduction(
                args.stream,
                args.s2_a_metrics,
                args.e2_summary,
                args.e2_tolerance,
                recording_scope,
                zero_frame_filter_enabled,
            )
        )
    checks.append(pass_record("Segmental metrics reset at each recording boundary", {"grouping": ["fold", "recording_id"]}))
    checks.append(pass_record("Test split is not used for Stage-2 checkpoint or eta selection", {"selection_split": "val"}))

    failed = [check for check in checks if check["status"] == "fail"]
    result = {
        "experiment": "S2",
        "stage": "sanity_checks",
        "protocol": protocol_metadata(),
        "checks": checks,
        "failed": len(failed),
    }
    write_json(args.output, result, overwrite=args.overwrite)
    print(f"[DONE] wrote sanity checks to {args.output}")
    if failed:
        for check in failed:
            print(f"[FAIL] {check['name']}: {check.get('details')}")
        raise SystemExit(1)
    for check in checks:
        print(f"[{check['status'].upper()}] {check['name']}")


if __name__ == "__main__":
    main()

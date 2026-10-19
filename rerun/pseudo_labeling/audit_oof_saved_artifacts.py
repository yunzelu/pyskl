#!/usr/bin/env python3
"""Independently reconstruct saved OOF artifacts; no inference or label changes.

Run from the repository root with Python and NumPy. The only write is the JSON
audit report. Assertions fail before saving a successful report on discrepancy.
"""

import argparse
import csv
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("data/radar_v4/rerun/yolo26xpose/pseudo_labels_v1")
DEFAULT_OUTPUT = Path(
    "rerun/pseudo_labeling/reports/oof_pseudo_labels_v1/"
    "e1b_e3_quality/saved_artifact_audit.json"
)
EPS = 1e-12


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def values(rows, field, dtype=float):
    return np.array([dtype(row[field]) for row in rows])


def probabilities(rows, prefix):
    return np.array([[float(row[f"{prefix}{i}"]) for i in range(9)] for row in rows])


def entropy(prob):
    prob = np.clip(np.asarray(prob, dtype=np.float64), EPS, 1.0)
    return -(prob * np.log(prob)).sum(axis=-1)


def normalized(prob):
    prob = np.clip(np.asarray(prob, dtype=np.float64), EPS, 1.0)
    return prob / prob.sum(axis=-1, keepdims=True)


def calibrate(prob, temperature):
    logits = np.log(normalized(prob)) / temperature
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def nll(prob, labels):
    return float(-np.log(np.clip(prob[np.arange(len(labels)), labels], EPS, 1.0)).mean())


def audit(root):
    maxima, thresholds = {}, {}
    counts = Counter()
    population = Counter()
    subjects, recordings = set(), set()
    teachers, folds = [], []

    def require(condition, message):
        counts["assertions_passed"] += 1
        if not condition:
            raise AssertionError(message)

    def compare(name, actual, expected, tolerance=0.0):
        residual = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        maxima[name] = max(maxima.get(name, 0.0), residual)
        thresholds[name] = tolerance
        require(np.isfinite(residual) and residual <= tolerance,
                f"{name}: residual {residual} exceeds {tolerance}")

    fold_paths = sorted(root.glob("fold_*"))
    require([path.name for path in fold_paths] == ["fold_a", "fold_b", "fold_c"],
            "Expected fold_a, fold_b, fold_c")
    for fold in fold_paths:
        aggregate = read_csv(fold / "oof_skeleton_pseudo_labels_audit.csv")
        aggregate_safe = read_csv(fold / "oof_skeleton_pseudo_labels.csv")
        aggregate_by_id = {row["skeleton_sample_id"]: row for row in aggregate}
        require(len(aggregate_by_id) == len(aggregate), f"Duplicate IDs: {fold}")
        require(len(aggregate_safe) == len(aggregate), f"Safe/audit count: {fold}")
        for row in aggregate_safe:
            audit_row = aggregate_by_id[row["skeleton_sample_id"]]
            require(all(value == audit_row[key] for key, value in row.items()),
                    f"Aggregate safe/audit mismatch: {row['skeleton_sample_id']}")
        teacher_ids = set()
        teacher_paths = sorted(fold.glob("t[1-4]"))
        require(len(teacher_paths) == 4, f"Expected 4 teachers: {fold}")
        for teacher in teacher_paths:
            metadata = read_json(teacher / "teacher_metadata.json")
            roles = {key: set(metadata[key]) for key in (
                "train_subjects", "validation_subject", "calibration_subject",
                "pseudo_target_subjects", "outer_test_subject_excluded")}
            role_sets = list(roles.values())
            intersections = [sorted(a & b) for i, a in enumerate(role_sets)
                             for b in role_sets[i + 1:] if a & b]
            require(not intersections, f"Subject role overlap: {teacher}")
            require(metadata["num_passes"] == 30, f"Teacher pass count: {teacher}")
            temperature = metadata["temperature"]
            require(np.isfinite(temperature) and temperature > 0, f"Temperature: {teacher}")
            teacher_result = {
                "teacher": teacher.relative_to(root).as_posix(),
                "temperature": temperature,
                "subject_roles": {key: sorted(value) for key, value in roles.items()},
                "subject_role_intersections": intersections,
            }
            for split, csv_name, npz_name in (
                ("pseudo_target", "pseudo_predictions.csv", "mc_fused_samples.npz"),
                ("calib", "calibration_predictions.csv", "calibration_mc_fused_samples.npz"),
            ):
                rows = read_csv(teacher / csv_name)
                audit_rows = (read_csv(teacher / "pseudo_predictions_audit.csv")
                              if split == "pseudo_target" else rows)
                ids = [row["skeleton_sample_id"] for row in rows]
                require(len(ids) == len(set(ids)), f"Duplicate teacher IDs: {teacher}/{split}")
                require(len(rows) == len(audit_rows), f"Safe/audit count: {teacher}/{split}")
                labels = values(audit_rows, "manual_label_at_skeleton_center", int)
                expected_subjects = roles["pseudo_target_subjects" if split == "pseudo_target"
                                          else "calibration_subject"]
                require({row["subject_id"] for row in rows} == expected_subjects,
                        f"Unexpected subject population: {teacher}/{split}")
                require(all(int(row["mc_passes"]) == 30 for row in rows), "CSV pass count")
                with np.load(teacher / npz_name, allow_pickle=False) as archive:
                    fused = archive["fused_mc_probabilities"]
                    require(list(archive["sample_ids"]) == ids, "NPZ sample order")
                    require(fused.shape == (30, len(rows), 9), "NPZ shape")
                    compare("source_frame_centers", archive["source_frame_indices"][:, 30],
                            values(rows, "source_frame_center", int))
                    if split == "calib":
                        compare("calibration_archive_labels", archive["manual_labels"], labels)
                require(np.isfinite(fused).all() and fused.min() >= 0 and fused.max() <= 1,
                        "Invalid fused probabilities")
                compare("per_pass_probability_sum_error", fused.sum(axis=-1), 1.0, 1e-5)
                mean = fused.astype(np.float64).mean(axis=0)
                predictive = entropy(mean)
                expected = entropy(fused).mean(axis=0)
                mi = np.maximum(predictive - expected, 0.0).astype(np.float32)
                raw = probabilities(rows, "mc_raw_p")
                calibrated = probabilities(rows, "mc_cal_p")
                compare("raw_mean_from_archived_passes", raw, mean.astype(np.float32))
                compare("predictive_entropy_from_archived_passes", values(rows, "mc_predictive_entropy"),
                        predictive.astype(np.float32))
                compare("expected_entropy_from_archived_passes", values(rows, "mc_expected_entropy"),
                        expected.astype(np.float32))
                compare("mi_from_archived_passes", values(rows, "mc_mi_raw"), mi)
                compare("calibrated_mean_reconstruction", calibrated, calibrate(raw, temperature), 1e-12)
                compare("raw_mean_probability_sum_error", raw.sum(axis=-1), 1.0, 1e-5)
                compare("calibrated_probability_sum_error", calibrated.sum(axis=-1), 1.0, 1e-12)
                pred = values(rows, "hard_pseudo_label_id", int)
                compare("hard_prediction_vs_raw_argmax", pred, raw.argmax(axis=-1))
                compare("hard_prediction_vs_calibrated_argmax", pred, calibrated.argmax(axis=-1))
                compare("saved_calibrated_argmax", pred, values(rows, "calibrated_argmax_id", int))
                compare("row_temperature", values(rows, "temperature"), temperature)
                compare("row_mi_q95", values(rows, "mi_q95_calibration"), metadata["mi_q95_calibration"])
                reliability = 0.1 + 0.9 * (1.0 - np.minimum(
                    mi.astype(np.float64) / max(metadata["mi_q95_calibration"], EPS), 1.0))
                compare("reliability_weight_reconstruction", values(rows, "reliability_weight"), reliability)
                stream_probs = []
                for stream in ("joint", "bone"):
                    stream_root = teacher / "stream_outputs" / split / stream
                    stream_ids = read_json(stream_root / "sample_ids.json")
                    require([row["skeleton_sample_id"] for row in stream_ids] == ids,
                            f"Stream sample order: {stream_root}")
                    compare("stream_labels_vs_audit", np.load(stream_root / "labels.npy"), labels)
                    stream_prob = np.load(stream_root / "mc_prob_passes.npy", allow_pickle=False)
                    require(stream_prob.shape == fused.shape, f"Stream shape: {stream_root}")
                    stream_probs.append(stream_prob)
                    stream_metadata = read_json(stream_root / "metadata.json")
                    require(stream_metadata["num_passes"] == 30, "Stream metadata pass count")
                    sanity = stream_metadata.get("sanity_check")
                    require(sanity and sanity["deterministic_max_abs_diff"] <= sanity["atol"]
                            and sanity["stochastic_mean_abs_diff"] > sanity["atol"],
                            f"Recorded inference sanity check: {stream_root}")
                    counts["stream_artifact_sets_checked"] += 1
                compare("fused_vs_joint_bone_average", fused, 0.5 * (stream_probs[0] + stream_probs[1]))
                counts[f"{split}_rows"] += len(rows)
                counts[f"{split}_per_pass_probability_values"] += fused.size
                teacher_result[f"{split}_rows"] = len(rows)
                count_key = "pseudo_target_windows" if split == "pseudo_target" else "calibration_windows"
                require(len(rows) == metadata["counts"][count_key], "Metadata row count")
                if split == "calib":
                    compare("calibration_mi_q95_reconstruction", float(np.quantile(mi, 0.95)),
                            metadata["mi_q95_calibration"], 1e-8)
                    # Fitting normalizes float32 means in float64 before computing raw NLL.
                    compare("normalized_raw_calibration_nll", nll(normalized(raw), labels),
                            metadata["temperature_fit"]["raw_calibration_nll"], 1e-12)
                    compare("calibrated_calibration_nll", nll(calibrated, labels),
                            metadata["temperature_fit"]["calibrated_calibration_nll"], 1e-12)
                else:
                    require(not teacher_ids.intersection(ids), "Duplicate IDs between teachers")
                    teacher_ids.update(ids)
                    for row, audit_row in zip(rows, audit_rows):
                        require(all(value == audit_row[key] for key, value in row.items()),
                                f"Teacher safe/audit mismatch: {row['skeleton_sample_id']}")
                        aggregate_row = aggregate_by_id[row["skeleton_sample_id"]]
                        require(all(value == aggregate_row[key] for key, value in row.items()
                                    if "timestamp" not in key and "nominal" not in key),
                                f"Teacher/aggregate non-timestamp mismatch: {row['skeleton_sample_id']}")
                        population[(row["recording_id"], row["window_start_retained_idx"],
                                    row["source_frame_center"]) ] += 1
                        subjects.add(row["subject_id"])
                        recordings.add(row["recording_id"])
            teachers.append(teacher_result)
        require(set(aggregate_by_id) == teacher_ids, f"Aggregate/teacher population: {fold}")
        fold_metadata = read_json(fold / "fold_metadata.json")
        timestamp_info = fold_metadata["real_timestamp_lookup"]
        require(timestamp_info["real_timestamp_applied"] and timestamp_info["rows_updated"] == len(aggregate),
                f"Aggregate real timestamp metadata: {fold}")
        folds.append({"fold": fold.name, "rows": len(aggregate),
                      "timestamp_provenance": {key: value for key, value in timestamp_info.items()
                                               if key != "recording_timestamp_files"}})
        print(f"Audited {fold.name}: {len(aggregate):,} target rows", flush=True)
    return {
        "audit_passed": True,
        "input_root": root.as_posix(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "counts": dict(counts),
        "maximum_absolute_residuals": maxima,
        "residual_tolerances": thresholds,
        "unique_population": {
            "key": ["recording_id", "window_start_retained_idx", "source_frame_center"],
            "source_windows": len(population),
            "source_window_occurrences_across_folds": dict(Counter(population.values())),
            "recordings": len(recordings), "subjects": sorted(subjects),
        },
        "folds": folds, "teachers": teachers,
        "interpretation": [
            "Raw means are float64 averages of 30 fused float32 passes, cast back to float32.",
            "Calibrated probabilities apply one teacher temperature to the predictive mean, not to each pass.",
            "MC MI q95 uses NumPy quantile on float32 MI; saved/runtime interpolation precision can differ at about 1e-9. The audit allows 1e-8 and records the residual.",
            "The recorded raw calibration NLL uses float64-renormalized probabilities; direct NLL of unnormalized float32 means can differ at about 1e-9.",
            "Teacher CSV and NPZ timestamps retain nominal source-frame time. Aggregate CSVs record real timestamp lookup; timestamp differences are intentional and excluded from teacher/aggregate identity checks.",
            "Fold rows include repeat source windows. Pooled row metrics are descriptive; they are not an independent sample of that many windows.",
            "Subject separation is verified from saved metadata and row identities, not reconstructed from historical training execution.",
            "Recorded inference sanity checks are inspected; no inference or temperature fitting is rerun.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved successful audit: {args.output}")


if __name__ == "__main__":
    main()

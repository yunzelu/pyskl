"""Audit saved OOF targets using the actual E1-B, E2B and E3 evaluators.

No inference or temperature fitting is performed. E1's train split is the OOF
target population; labels and temporal ordering come from that canonical pkl.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
# E1's summary script imports its sibling evaluator by this module name.
sys.path.insert(0, str(REPO / "rerun/e1"))
import evaluate_continuous_segmental as segmental
import summarize_results as e1
from rerun.e2 import evaluate_e2b_reliability as e2
from rerun.e3 import run_e3_mc_temperature_scaling as e3
from rerun.pseudo_labeling import analyze_oof_pseudo_labels as legacy

LABELS = e1.LABELS
ECE_BINS = 15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty input: {path}")
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def align_to_annotations(rows, annotations):
    """Join by stable identity, rejecting duplicate, missing and extra windows."""
    indexed = {row["source_pyskl_frame_dir"]: row for row in rows}
    require(len(indexed) == len(rows), "Duplicate pseudo source_pyskl_frame_dir")
    require(len({row["skeleton_sample_id"] for row in rows}) == len(rows),
            "Duplicate skeleton_sample_id")
    expected = {item["frame_dir"] for item in annotations}
    require(len(expected) == len(annotations), "Duplicate E1 annotation frame_dir")
    missing, extra = expected - indexed.keys(), indexed.keys() - expected
    require(not missing and not extra,
            f"OOF / E1 train coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    return [indexed[item["frame_dir"]] for item in annotations]


def validate_annotations(rows, annotations, metadata, fold):
    boundaries = Counter()
    timestamps = Counter()
    subjects = set()
    teacher_counts = Counter()
    for row, item in zip(rows, annotations):
        name = item["frame_dir"]
        def check(ok, description):
            require(bool(ok), f"{name}: {description}")
        pairs = {
            "manual_label_at_skeleton_center": "label",
            "source_frame_center": "center_source_frame",
            "window_start_retained_idx": "window_row_start",
            "window_end_retained_idx_exclusive": "window_row_end_exclusive",
            "window_size": "window_size", "stride": "stride",
            "center_offset": "center_row_offset",
        }
        for column, key in pairs.items():
            check(int(row[column]) == int(item[key]), f"{column} differs from E1 {key}")
        check(row["recording_id"] == item["session_name"], "recording differs")
        check(row["subject_id"] == item["subject"], "subject differs")
        check(item["sample_protocol"] == "continuous_window_w60_s12", "wrong protocol")
        check((int(row["window_size"]), int(row["stride"]), int(row["center_offset"]))
              == (60, 12, 30), "wrong window definition")
        frames = np.asarray(item["source_frame_indices"])
        check(len(frames) == 60 and int(frames[30]) == int(row["source_frame_center"]),
              "center is not source_frame_indices[30]")
        check(int(frames[0]) == int(row["source_frame_start"]), "start frame differs")
        check(int(frames[-1]) == int(row["source_frame_end"]), "end frame differs")
        check(int(row["center_retained_idx"]) == int(item["window_row_start"]) + 30,
              "retained center differs")
        check(float(item["max_adjacent_gap_sec"]) <= 0.5 + 1e-8, "gap exceeds E1 limit")
        check(float(item["window_span_sec"]) <= 2.5 + 1e-8, "span exceeds E1 limit")
        check(row["outer_fold"].lower() == fold and row["split_role"] == "pseudo_target",
              "wrong fold/split")
        check(int(row["mc_passes"]) == 30, "not a 30-pass pseudo target")
        teacher = row["inner_teacher_id"].lower()
        info = metadata["teacher_metadata"][teacher]
        subject = row["subject_id"]
        excluded = (info["train_subjects"] + info["validation_subject"]
                    + info["calibration_subject"] + info["outer_test_subject_excluded"])
        check(subject in info["pseudo_target_subjects"] and subject not in excluded,
              "pseudo target subject leaked into teacher train/val/calib/test")
        for column in ("temperature", "mi_q95_calibration"):
            if row.get(column):
                check(np.isclose(float(row[column]), float(info[column]), rtol=0, atol=1e-12),
                      f"{column} differs from teacher metadata")
        if row.get("manual_segment_start_frame") and row.get("manual_segment_end_frame"):
            check(int(row["manual_segment_start_frame"]) <= int(frames[30])
                  <= int(row["manual_segment_end_frame"]), "center outside audited segment")
        boundaries[row["manual_boundary_source"]] += 1
        timestamps[row["source_timestamp_source"]] += 1
        teacher_counts[teacher] += 1
        subjects.add(subject)
    owners = [subject for info in metadata["teacher_metadata"].values()
              for subject in info["pseudo_target_subjects"]]
    require(len(owners) == len(set(owners)) and set(owners) == subjects,
            "Teacher target subjects must partition E1 train subjects")
    return {"num_verified_windows": len(rows), "num_subjects": len(subjects),
            "subjects": sorted(subjects), "teacher_counts": dict(teacher_counts),
            "manual_boundary_sources": dict(boundaries), "timestamp_sources": dict(timestamps),
            "annotation_label_mismatches": 0, "identity_or_source_frame_mismatches": 0,
            "missing_or_extra_windows": 0, "teacher_subject_leakage": 0}


def saved_probabilities(rows, prefix):
    columns = [f"{prefix}_p{i}" for i in range(len(LABELS))]
    if any(not row.get(column) for row in rows for column in columns):
        return None
    values = np.asarray([[float(row[column]) for column in columns] for row in rows])
    e3.validate_probabilities(values, prefix)
    return values


def temporal_metrics(annotations, scores, fold):
    records = []
    for recording, items in segmental.grouped_sequences(annotations, scores).items():
        records.append({"fold": fold.upper(), "recording_id": recording,
                        **segmental.evaluate_recording(items, set(), "sequence_index")})
    result = {"num_recordings": len(records),
              "edit": float(np.mean([row["edit"] for row in records]))}
    for threshold in segmental.THRESHOLDS:
        suffix = f"{int(100 * threshold):02d}"
        counts = [sum(row[f"{kind}_{suffix}"] for row in records) for kind in ("tp", "fp", "fn")]
        result.update({f"{kind}_{suffix}": value for kind, value in zip(("tp", "fp", "fn"), counts)})
        result[f"f1_{suffix}"] = segmental.f1_from_counts(*counts)
    return result, records


def evaluate_fold(root, data_root, fold):
    folder = root / f"fold_{fold}"
    audit_path = folder / "oof_skeleton_pseudo_labels_audit.csv"
    pkl = data_root / f"radarv4_yolo26xpose_continuous_window_w60_s12_fold_{fold}.pkl"
    metadata_path = folder / "fold_metadata.json"
    annotations = segmental.split_annotations(pkl, "train")
    rows = align_to_annotations(read_rows(audit_path), annotations)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validation = validate_annotations(rows, annotations, metadata, fold)
    validation.update({"fold": fold.upper(), "audit_path": str(audit_path),
                       "audit_sha256": sha256(audit_path), "e1_annotation_path": str(pkl),
                       "e1_annotation_sha256": sha256(pkl),
                       "fold_metadata_sha256": sha256(metadata_path)})
    targets = np.asarray([int(item["label"]) for item in annotations])
    hard = np.asarray([int(row["hard_pseudo_label_id"]) for row in rows])
    require(bool(np.all((hard >= 0) & (hard < len(LABELS)))), "Invalid hard pseudo class")
    # One-hot scores here represent hard decisions only, never calibration probabilities.
    decisions = np.eye(len(LABELS))[hard]
    metrics = {"fold": fold.upper(), "num_windows": len(rows), **e1.metric_dict(decisions, targets)}
    temporal, recordings = temporal_metrics(annotations, decisions, fold)
    metrics.update(temporal)
    errors = (hard != targets).astype(np.int64)
    metrics.update(num_errors=int(errors.sum()), error_rate=float(errors.mean()),
                   random_auprc_baseline=float(errors.mean()))
    skipped = []
    probabilities = {}
    for version, prefix in (("raw", "mc_raw"), ("calibrated", "mc_cal")):
        values = saved_probabilities(rows, prefix)
        if values is None:
            skipped.append(f"{version} NLL/Brier/ECE: incomplete saved {prefix}_p0..8")
            continue
        probabilities[version] = values
        require(np.array_equal(values.argmax(axis=1), hard), f"{version} argmax differs from hard label")
        scores = e3.classification_metrics(values, targets, ECE_BINS)
        require(scores["center_accuracy"] == metrics["top1_acc"]
                and scores["center_macro_f1"] == metrics["macro_f1"], "E1/E3 metric mismatch")
        for name in ("nll", "brier", "ece"):
            metrics[f"{version}_{name}"] = scores[name]
        validation[f"{version}_probability_sum_max_abs_error"] = float(np.max(np.abs(values.sum(axis=1) - 1)))
    if len(probabilities) == 2:
        diffs = []
        for teacher in metadata["teacher_metadata"]:
            mask = np.asarray([row["inner_teacher_id"].lower() == teacher for row in rows])
            temperature = float(metadata["teacher_metadata"][teacher]["temperature"])
            reconstructed = e3.apply_pool_temperature(probabilities["raw"][mask], temperature)
            diffs.append(float(np.max(np.abs(reconstructed - probabilities["calibrated"][mask]))))
        validation["calibrated_reconstruction_max_abs_error"] = max(diffs)
        require(max(diffs) < 1e-6, "Saved calibrated probabilities do not match E3 temperature rule")
    rankings = []
    if all(row.get("mc_mi_raw") for row in rows):
        mi = np.asarray([float(row["mc_mi_raw"]) for row in rows])
        require(bool(np.all(np.isfinite(mi)) and np.all(mi >= -1e-8)), "Invalid raw MI")
        scores = {"raw_mi": np.maximum(mi, 0)}
        if all(row.get("mi_q95_calibration") for row in rows):
            quantiles = np.asarray([float(row["mi_q95_calibration"]) for row in rows])
            require(bool(np.all(np.isfinite(quantiles)) and np.all(quantiles >= 0)), "Invalid MI q95")
            scores["u_norm"] = np.minimum(mi / np.maximum(quantiles, 1e-12), 1)
            validation["u_norm_clipped_windows"] = int(np.count_nonzero(scores["u_norm"] == 1))
            if all(row.get("reliability_weight") for row in rows):
                weights = np.asarray([float(row["reliability_weight"]) for row in rows])
                diff = float(np.max(np.abs(weights - (0.1 + 0.9 * (1 - scores["u_norm"])))))
                validation["reliability_weight_max_abs_error"] = diff
                require(diff < 1e-8, "Saved weight differs from normalized MI")
        else:
            skipped.append("Normalized MI ranking: incomplete saved mi_q95_calibration")
        for score_name, uncertainty in scores.items():
            for subset, mask in (("all", np.ones(len(rows), dtype=bool)),
                                 ("state", np.isin(targets, e2.STATE_CLASS_IDS)),
                                 ("transition", np.isin(targets, e2.TRANSITION_CLASS_IDS))):
                rank = {"fold": fold.upper(), "score": score_name, "subset": subset,
                        **e2.ranking_metrics_for_subset(errors, uncertainty, mask, "")}
                rankings.append(rank)
                if subset == "all":
                    metrics[f"{score_name}_error_auroc"] = rank["error_auroc"]
                    metrics[f"{score_name}_error_auprc_ap"] = rank["error_auprc_ap"]
    else:
        skipped.append("Error AUROC/AP: raw MI is not saved for every evaluated window")
    f1 = e3.per_class_f1(hard, targets, len(LABELS))
    classes = [{"fold": fold.upper(), "class_id": i, "class_name": name,
                "support": int(np.count_nonzero(targets == i)),
                "predicted": int(np.count_nonzero(hard == i)), "f1": float(f1[i])}
               for i, name in enumerate(LABELS)]
    # Independently reproduce the previous report to expose definition changes.
    if len(probabilities) == 2 and "u_norm_error_auroc" in metrics:
        old = legacy.analyze_fold(root, fold)
        comparisons = {"top1_acc": "center_top1_accuracy", "macro_f1": "center_macro_f1",
                       "u_norm_error_auroc": "all_error_auroc", "u_norm_error_auprc_ap": "all_error_auprc_ap"}
        validation["legacy_metric_max_abs_difference"] = max(abs(metrics[k] - old[v]) for k, v in comparisons.items())
        require(validation["legacy_metric_max_abs_difference"] < 1e-12, "Legacy reproduction failed")
    return metrics, recordings, rankings, classes, validation, [f"Fold {fold.upper()}: {item}" for item in skipped]


def summarize(rows, exclude=()):
    summary = []
    keys = dict.fromkeys(key for row in rows for key in row)
    for key in keys:
        if key in exclude or key == "fold":
            continue
        numeric = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not numeric:
            continue
        values = [float(value) for value in numeric if np.isfinite(value)]
        summary.append({"metric": key, "mean": float(np.mean(values)) if values else None,
                        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "num_folds": len(values)})
    return summary


def markdown(payload):
    folds = payload["fold_metrics"]
    summary = {row["metric"]: row for row in payload["mean_sd"]}
    lines = ["# OOF pseudo-label quality: E1-B / E3 protocol audit", "",
             "All requested metrics are computable from the saved artifacts in this run."
             if not payload["skipped"] else "Some metrics are skipped; see the availability section.", "",
             "Manual annotations are used only for this evaluation. No inference, retraining,"
             " filtering, resampling, or temperature refitting was performed.", "",
             "## Results", "",
             "Mean and sample SD (ddof=1) are over the three outer-fold OOF target sets. "
             "Edit and segmental F1 are on a 0-100 scale; other metrics use their native scale.", "",
             "| Metric | A | B | C | Mean +/- SD |", "| --- | ---: | ---: | ---: | ---: |"]
    specs = [("top1_acc", "Top-1 accuracy"), ("macro_f1", "Macro F1 (9 classes)"),
             ("edit", "Edit"), ("f1_10", "Segmental F1@10"), ("f1_25", "Segmental F1@25"),
             ("f1_50", "Segmental F1@50"), ("error_rate", "Error rate"),
             ("random_auprc_baseline", "Random AUPRC"), ("raw_mi_error_auroc", "Error AUROC (raw MI)"),
             ("raw_mi_error_auprc_ap", "Error AUPRC/AP (raw MI)"),
             ("raw_nll", "NLL (raw)"), ("calibrated_nll", "NLL (calibrated)"),
             ("raw_brier", "Brier (raw)"), ("calibrated_brier", "Brier (calibrated)"),
             ("raw_ece", "ECE (raw)"), ("calibrated_ece", "ECE (calibrated)"),
             ("u_norm_error_auroc", "Error AUROC (legacy normalized MI)"),
             ("u_norm_error_auprc_ap", "Error AUPRC/AP (legacy normalized MI)")]
    def fmt(value):
        return "NA" if value is None or not np.isfinite(value) else f"{value:.4f}"
    for key, label in specs:
        item = summary.get(key, {})
        lines.append(f"| {label} | " + " | ".join(fmt(row.get(key)) for row in folds)
                     + f" | {fmt(item.get('mean'))} +/- {fmt(item.get('sd'))} |")
    if not payload["skipped"]:
        deltas = ", ".join(f"{name.upper()} {summary[f'calibrated_{name}']['mean'] - summary[f'raw_{name}']['mean']:+.4f}"
                           for name in ("nll", "brier", "ece"))
        lines += ["", f"Calibration changes in fold means (calibrated minus raw): {deltas}. "
                  "Negative values indicate improvement; the table also shows variation by fold. "
                  "Error AP should be interpreted alongside the error-prevalence baseline.", "",
                  "### Per-class F1", "", "| Class | A | B | C | Mean +/- SD |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for class_id, label in enumerate(LABELS):
            values = [row["f1"] for row in payload["class_metrics"] if row["class_id"] == class_id]
            lines.append(f"| {label} | " + " | ".join(fmt(value) for value in values)
                         + f" | {fmt(np.mean(values))} +/- {fmt(np.std(values, ddof=1))} |")
    lines += ["", "## Annotation and artifact checks", "",
              "Every audit row is joined by `source_pyskl_frame_dir` to the E1-B continuous "
              "pkl's outer **train** split. Exact coverage, unique identities, recording, subject, "
              "source start/center/end frames, center label, window offsets and OOF exclusions are checked.", "",
              "| Fold | Windows | Recordings | Subjects | Label mismatches | Missing/extra windows |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row, check in zip(folds, payload["validation"]):
        lines.append(f"| {row['fold']} | {row['num_windows']} | {row['num_recordings']} | "
                     f"{check['num_subjects']} | {check['annotation_label_mismatches']} | "
                     f"{check['missing_or_extra_windows']} |")
    if not payload["skipped"]:
        lines += ["", "Saved raw and calibrated argmax decisions agree with the hard pseudo labels. "
              "E1 and E3 classification implementations give identical results. Calibrated "
              "probabilities are checked against each teacher's saved temperature, and reliability "
              "weights against reconstructed normalized MI. The previous pseudo report's accuracy, "
              "macro F1 and normalized-MI ranking results are reproduced."]
    else:
        lines += ["", "Available probability/uncertainty checks and their residuals are recorded "
                  "in the JSON validation section; incomplete metric inputs are explicitly skipped."]
    lines += ["",
              "## Metric equivalence", "",
              "| Metric | Definition and reused implementation |", "| --- | --- |",
              "| Top-1, macro F1 | `rerun/e1/summarize_results.py::metric_dict`; checked against "
              "`rerun/e3/run_e3_mc_temperature_scaling.py::classification_metrics`. Nine-class "
              "macro F1 includes absent classes as zero, with no class weighting. |",
              "| Edit | E1 `grouped_sequences` and `evaluate_recording`: collapse consecutive "
              "labels, normalize Levenshtein by the larger segment count, then average recording scores. |",
              "| Segmental F1 | E1 `evaluate_recording` / `f1_from_counts`: IoU thresholds "
              "0.10/0.25/0.50; sum TP/FP/FN across recordings before F1. No background excluded. |",
              "| Error rate / random AUPRC | Error is hard argmax != manual center label. "
              "Random-ranking AUPRC baseline equals error prevalence. |",
              "| Error AUROC/AP | E2B `ranking_metrics_for_subset`: tied-score average-rank AUROC; "
              "AP is sum of precision times recall increments at distinct scores, not trapezoidal PR area. |",
              "| NLL | E3: mean -ln(p_true), clipping p_true to [1e-12,1]. |",
              "| Brier | E3: mean over windows of sum over 9 classes of (p-onehot)^2; "
              "no division by class count. |",
              "| ECE | E3: top-label confidence, 15 equal-width bins; [lower,upper), "
              "last bin includes 1. Weighted absolute bin accuracy-confidence difference. |", "",
              "NLL/Brier/ECE use the saved `mc_raw_p0..8` and `mc_cal_p0..8` as stored; "
              "they are not replaced by one-hot labels or refitted probabilities.", "",
              "## Ambiguities resolved and comparison limits", "",
              "1. **Center-time protocol:** E1-B takes retained detection index `start + 30` "
              "in a 60-detection, stride-12 window (gap <=0.5 s, span <=2.5 s). Its original "
              "source-frame annotation supplies the label. It is not a timestamp midpoint or "
              "majority label. Aggregation replaced nominal times with real camera timestamps; "
              "identity/source-frame alignment is used, not equality of those timestamps.",
              "2. **Temporal axis:** E1's segment metrics use ordered window indices within each "
              "recording, sorted by window start then source center frame. They do not use "
              "elapsed-second IoU, interpolate missing windows, or split sequences at rejected "
              "window gaps. Different recordings are never concatenated.",
              "3. **E1 report version:** current E1 uses official MS-TCN best-overlap-then-check-matched "
              "matching, without a fallback if the best reference is already matched. Compare "
              "with `rerun/e1/reports/ms_tcn_official`, not the original root reports using "
              "the earlier best-unmatched rule.",
              "4. **Ranking score:** E1/E3 do not define error AUROC/AP. Their rerun companion "
              "E2B uses raw MI; that score is primary here. The previous pseudo report uses "
              "`min(mc_mi_raw / max(mi_q95_calibration,1e-12),1)`. Teacher-specific normalization "
              "changes pooled ranking and clipping creates ties, so both are reported separately. "
              "Larger score means more likely wrong. MI remains the raw MC uncertainty even "
              "when evaluating the order-preserving calibrated probabilities.",
              "5. **Evaluation population:** these are held-out inner-teacher targets covering "
              "each outer fold's eight training subjects, not E1/E3 outer-test subjects. Subjects "
              "and windows overlap across outer folds. Equal-fold mean +/- SD is descriptive, "
              "not an independent-sample confidence interval or a paired improvement over E1/E3.",
              "6. **Probability variants:** report both raw 30-pass joint/bone MC means and saved "
              "pool-temperature-calibrated probabilities. Classification/segment scores coincide "
              "because their argmax decisions are identical. Calibration temperatures are "
              "teacher-specific and fitted on the fixed calibration subject only.", "",
              "## Availability and reproducibility", ""]
    lines += [f"- {message}" for message in payload["skipped"]] or ["No requested metric was skipped."]
    lines += ["", "Run from the repository root:", "", "```powershell",
              "python rerun/pseudo_labeling/evaluate_oof_pseudo_labels.py", "```", "",
              "The JSON contains input/code SHA-256 hashes and numerical validation residuals. "
              "CSV outputs include fold metrics, fold mean/SD, per-recording segment counts, "
              "per-class F1 and state/transition error ranking with both uncertainty scores.", "",
              "The separate [saved artifact audit](saved_artifact_audit.json) reconstructs "
              "MC fusion, means, uncertainty and calibration from archived passes and verifies "
              "teacher/stream sample ordering and saved subject roles. Regenerate it with "
              "`python rerun/pseudo_labeling/audit_oof_saved_artifacts.py`. Its population counts "
              "distinguish repeated fold rows from unique source windows. Subject separation "
              "is verified from saved metadata, not historical training execution.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-label-root", type=Path,
                        default=Path("data/radar_v4/rerun/yolo26xpose/pseudo_labels_v1"))
    parser.add_argument("--annotation-root", type=Path,
                        default=Path("data/radar_v4/rerun/yolo26xpose/pyskl/continuous_window_w60_s12"))
    parser.add_argument("--report-dir", type=Path,
                        default=Path("rerun/pseudo_labeling/reports/oof_pseudo_labels_v1/e1b_e3_quality"))
    args = parser.parse_args()
    payload: dict[str, Any] = {"fold_metrics": [], "recording_metrics": [], "ranking_metrics": [],
                               "class_metrics": [], "validation": [], "skipped": []}
    for fold in ("a", "b", "c"):
        result = evaluate_fold(args.pseudo_label_root, args.annotation_root, fold)
        for key, value in zip(("fold_metrics", "recording_metrics", "ranking_metrics",
                               "class_metrics", "validation", "skipped"), result):
            payload[key].extend(value if isinstance(value, list) else [value])
        print(f"Verified fold {fold.upper()}: {result[0]['num_windows']} windows", flush=True)
    payload["mean_sd"] = summarize(payload["fold_metrics"])
    payload["protocol"] = {"annotation_split": "train", "center_offset": 30,
                           "axis": "sequence_index", "background_labels": [],
                           "matching_rule": segmental.MATCHING_RULE, "ece_bins": ECE_BINS,
                           "primary_uncertainty_score": "mc_mi_raw", "fold_sd_ddof": 1}
    code_paths = [Path(__file__), Path(segmental.__file__), Path(e1.__file__), Path(e2.__file__), Path(e3.__file__)]
    payload["code_sha256"] = {str(path.relative_to(REPO)): sha256(path) for path in code_paths}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for key in ("fold_metrics", "mean_sd", "recording_metrics", "ranking_metrics", "class_metrics"):
        if payload[key]:
            e3.write_csv(args.report_dir / f"{key}.csv", payload[key])
    e3.write_json(args.report_dir / "summary.json", payload)
    (args.report_dir / "summary.md").write_text(markdown(payload), encoding="utf-8")
    print(f"Wrote {args.report_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()

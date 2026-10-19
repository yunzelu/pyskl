"""Evaluate saved R2/R3 student targets against retained RADAR-center annotations.

Run in pyskl; PointCam inputs are read only. Identity, selection, all saved variants
and saved teacher associations must validate before any report is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from rerun.pseudo_labeling import evaluate_oof_pseudo_labels as reference

require = reference.require
LABELS = reference.LABELS
VARIANTS = ("hard", "raw_mc_soft", "weighted_raw_mc_soft", "weighted_calibrated_mc_soft",
            "weighted_hard", "calibrated_mc_soft")
R2_VARIANTS = VARIANTS[:4]


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def load_pointcam_helpers(project):
    path = project / "src/rerun/r2/pseudo_targets.py"
    spec = importlib.util.spec_from_file_location("pointcam_r2_assignment_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compact_radar_ground_truth(path, fold):
    payload = reference.segmental.load_pickle(path)
    metadata = payload["metadata"]
    require(metadata["label_names"] == LABELS, "Radar class mapping differs")
    require(metadata["fold"] == fold and metadata["dataset_type"] == "continuous_window",
            "Unexpected radar fold/type")
    ids = payload["split"]["train"]
    all_ids = [fid for split in payload["split"].values() for fid in split]
    require(len(all_ids) == len(set(all_ids)), "Duplicate radar IDs across splits")
    annotations = payload["annotations"]
    index = {ann["frame_id"]: ann for ann in annotations}
    require(len(index) == len(annotations) and set(index) == set(all_ids), "Radar coverage differs")
    labels = {}
    for fid in ids:
        ann = index[fid]
        require(int(ann["total_frames"]) == 40 and ann["points"].shape == (40, 150, 4),
                f"Not a 40-frame radar point window: {fid}")
        labels[fid] = int(ann["label"])
    result = (ids, labels, metadata, payload["split"])
    # Release multi-GB point arrays; only identity and manual label enter metrics.
    del payload, annotations, index, ann
    gc.collect()
    return result


def validate_selection(root, fold, ids, labels, targets):
    path = root / "selection" / f"fold_{fold}_training_windows.csv"
    rows = reference.read_rows(path)
    require(len({r["frame_id"] for r in rows}) == len(rows), "Duplicate selection row")
    counts = Counter()
    kept = []
    for row in rows:
        if row["recording_id"] == "38-mia-fall2":
            status = "excluded_recording"
        else:
            gap = abs(float(row["radar_center_timestamp_sec"]) - float(row["teacher_center_timestamp_sec"]))
            require(np.isfinite(gap) and abs(gap - float(row["abs_time_diff_sec"])) <= 1e-9,
                    "Invalid saved selection gap")
            status = "kept" if gap <= 0.2 else "teacher_center_gap_gt_0.2s"
        require(row["status"] == status, "Selection differs from deterministic R3 rule")
        counts[status] += 1
        if status == "kept":
            fid = row["frame_id"]
            kept.append(fid)
            require(int(row["manual_label"]) == labels[fid], "Selection/radar GT differs")
            require(int(row["pseudo_label"]) == targets[fid]["hard_label"], "Selection/pseudo differs")
    require(kept == ids, "Selection does not exactly reproduce saved radar train IDs/order")
    return {"original_radar_train_windows": len(rows), "kept_radar_windows": counts["kept"],
            "excluded_recording_windows": counts["excluded_recording"],
            "excluded_gap_windows": counts["teacher_center_gap_gt_0.2s"],
            "selection_sha256": reference.sha256(path)}


def evaluate_fold(project, skeleton_root, fold, pointcam, stage="r3"):
    require(stage in ("r2", "r3"), "Unknown radar evaluation stage")
    root = project / "data/rerun" / ("r3" if stage == "r3" else "radar_dataset")
    target_root = root / "pseudo_targets" if stage == "r3" else project / "data/rerun/r2_pseudo_targets"
    gt_path = root / "continuous_window" / f"fold_{fold}.pkl"
    ids, labels, metadata, splits = compact_radar_ground_truth(gt_path, fold)
    if stage == "r3":
        protocol = metadata["r3_protocol"]
        require(protocol["filter_split"] == "train" and protocol["max_teacher_center_gap_sec"] == 0.2
                and protocol["excluded_training_recordings"] == ["38-mia-fall2"], "Unexpected R3 filter")
        require(digest(ids) == protocol["ordered_train_ids_sha256"], "Radar ID digest differs")
        variants = VARIANTS
    else:
        original_ids = ids
        ids = [fid for fid in ids if pointcam.parse_continuous_frame_id(fid).recording_id != "38-mia-fall2"]
        splits = {**splits, "train": ids}
        variants = R2_VARIANTS
    base, soft, checks, soft_variants = None, {}, [], {}
    common_fields = ("frame_id", "total_frames", "label", "hard_label", "manual_label",
                     "recording_id", "recording_uid", "subject_id", "window_start_retained_idx",
                     "window_end_retained_idx_inclusive", "center_retained_idx", "radar_center_timestamp_sec",
                     "teacher_center_timestamp_sec", "abs_time_diff_sec", "teacher_row_index",
                     "skeleton_sample_id", "inner_teacher_id", "source_frame_center", "source_timestamp_center_sec",
                     "reliability_weight", "mc_mi_raw", "mc_predictive_entropy", "mc_expected_entropy")
    for variant in variants:
        path = target_root / variant / f"fold_{fold}.pkl"
        payload = reference.segmental.load_pickle(path)
        rows = payload["annotations"]
        require(payload["split"] == splits and [r["frame_id"] for r in rows] == ids,
                f"{variant}: retained radar IDs/order differ")
        require(payload["metadata"]["label_names"] == LABELS, f"{variant}: class mapping differs")
        if stage == "r2":
            stats = payload["metadata"]["stats"]
            require("max_pseudo_time_diff_sec" in stats and stats["max_pseudo_time_diff_sec"] is None,
                    "R2 unexpectedly has a teacher association gap cutoff")
            require(stats["candidate_train_windows"] == len(original_ids)
                    and stats["kept_train_windows"] == len(ids)
                    and stats["dropped_train_windows"] == len(original_ids) - len(ids), "R2 count mismatch")
            require(stats["dropped_by_recording"] == ({"38-mia-fall2": len(original_ids) - len(ids)}
                                                     if len(original_ids) != len(ids) else {}),
                    "Unexpected R2 recording exclusions")
        if base is None:
            base = rows
        is_calibrated = variant in ("calibrated_mc_soft", "weighted_calibrated_mc_soft")
        source = "calibrated_mc" if is_calibrated else "raw_mc"
        weighted = variant.startswith("weighted_")
        for ann, other in zip(rows, base):
            fid = ann["frame_id"]
            require(all(ann[k] == other[k] for k in common_fields), f"{variant}: changed association {fid}")
            require(all(np.array_equal(ann[k], other[k]) for k in ("raw_mc", "calibrated_mc")),
                    f"{variant}: changed assigned probabilities")
            require(ann["manual_label"] == labels[fid], f"{variant}: GT does not match radar pkl")
            expected = pointcam.probability_vector(ann[source])
            require(np.array_equal(ann["soft_label"], expected), f"{variant}: soft target differs from routing")
            require(ann["sample_weight"] == (ann["reliability_weight"] if weighted else 1),
                    f"{variant}: unexpected sample weight")
            require(ann["label"] == ann["hard_label"] == int(np.argmax(ann["raw_mc"]))
                    == int(np.argmax(ann["calibrated_mc"])), "Pseudo argmax disagreement")
        if variant in ("raw_mc_soft", "calibrated_mc_soft", "weighted_calibrated_mc_soft"):
            version = "calibrated" if is_calibrated else "raw"
            soft[version] = np.stack([r["soft_label"] for r in rows]).astype(np.float64)
            soft_variants[version] = variant
        csv_path = path.with_name(f"fold_{fold}_assignments.csv")
        csv_rows = reference.read_rows(csv_path)
        require([r["frame_id"] for r in csv_rows] == ids, "Assignment CSV IDs/order differs")
        for ann, csv_row in zip(rows, csv_rows):
            require(csv_row["skeleton_sample_id"] == ann["skeleton_sample_id"]
                    and int(csv_row["manual_label"]) == ann["manual_label"]
                    and int(csv_row["hard_label"]) == ann["hard_label"], "Assignment CSV label/identity differs")
            for field, prefix in (("raw_mc", "raw_mc_p"), ("calibrated_mc", "calibrated_mc_p"), ("soft_label", "soft_p")):
                require(np.array_equal(np.asarray([float(csv_row[f"{prefix}{i}"]) for i in range(9)], dtype=np.float32),
                                       ann[field]), "Assignment CSV probabilities differ")
        checks.append({"variant": variant, "num_windows": len(rows), "all_ids_order_labels_associations_equal": True,
                       "pkl_sha256": reference.sha256(path), "assignments_sha256": reference.sha256(csv_path)})
    targets = {r["frame_id"]: r for r in base}
    if stage == "r3":
        selection = validate_selection(root, fold, ids, labels, targets)
    else:
        selection = {"original_radar_train_windows": len(original_ids), "kept_radar_windows": len(ids),
                     "excluded_recording_windows": len(original_ids) - len(ids), "excluded_gap_windows": 0,
                     "original_train_ids_sha256": digest(original_ids)}
    alignment_path = skeleton_root / f"fold_{fold.lower()}" / "radar_teacher_alignment.csv"
    alignment = pointcam.TeacherAlignmentIndex.load(alignment_path, LABELS)
    skeleton_audit_path = alignment_path.with_name("oof_skeleton_pseudo_labels_audit.csv")
    skeleton_rows = reference.read_rows(skeleton_audit_path)
    skeleton = {r["skeleton_sample_id"]: r for r in skeleton_rows}
    require(len(skeleton) == len(skeleton_rows), "Duplicate source skeleton ID")
    annotated, joined, unique_teacher = [], [], set()
    for ann in base:
        fid = ann["frame_id"]
        parsed = pointcam.parse_continuous_frame_id(fid)
        require(parsed.center_retained_idx == parsed.window_start_retained_idx + 20
                and parsed.window_end_retained_idx_inclusive == parsed.window_start_retained_idx + 39,
                "Not a radar 40-frame center window")
        require(parsed.recording_id == ann["recording_id"] and parsed.subject_id == ann["subject_id"]
                and parsed.center_retained_idx == ann["center_retained_idx"]
                and parsed.window_start_retained_idx == ann["window_start_retained_idx"], "Radar window metadata differs")
        match = alignment.nearest(ann["subject_id"], ann["recording_id"], ann["radar_center_timestamp_sec"])
        require(match is not None and match.row.skeleton_sample_id == ann["skeleton_sample_id"]
                and match.row.row_index == ann["teacher_row_index"], "Saved teacher is not the original nearest match")
        require(np.isfinite(match.abs_time_diff_sec) and match.abs_time_diff_sec == ann["abs_time_diff_sec"]
                and (stage == "r2" or match.abs_time_diff_sec <= 0.2),
                "Retained association gap differs")
        require(match.row.source_frame_center == ann["source_frame_center"]
                and match.row.center_timestamp_sec == ann["teacher_center_timestamp_sec"],
                "Saved teacher center metadata differs")
        require(np.array_equal(match.row.raw_mc, ann["raw_mc"])
                and np.array_equal(match.row.calibrated_mc, ann["calibrated_mc"])
                and match.row.mc_mi_raw == ann["mc_mi_raw"], "Saved assigned probabilities/MI changed")
        source = skeleton[ann["skeleton_sample_id"]]
        source_label = int(source["manual_label_at_skeleton_center"])
        require(source["recording_id"] == ann["recording_id"] and source["subject_id"] == ann["subject_id"],
                "Cross-recording/subject teacher association")
        unique_teacher.add(ann["skeleton_sample_id"])
        # E1 temporal adapter: ordering metadata is exclusively RADAR metadata.
        annotated.append({"frame_dir": fid, "session_name": ann["recording_id"], "label": labels[fid],
                          "window_row_start": ann["window_start_retained_idx"],
                          "center_source_frame": ann["center_retained_idx"]})
        joined.append({"fold": fold, "radar_frame_id": fid, "recording_id": ann["recording_id"],
                       "radar_manual_label": labels[fid], "hard_pseudo_label": ann["hard_label"],
                       "error": int(labels[fid] != ann["hard_label"]),
                       "source_skeleton_sample_id": ann["skeleton_sample_id"],
                       "source_skeleton_manual_label": source_label,
                       "radar_vs_skeleton_gt_differs": int(source_label != labels[fid]),
                       "radar_center_timestamp_sec": ann["radar_center_timestamp_sec"],
                       "teacher_center_timestamp_sec": ann["teacher_center_timestamp_sec"],
                       "abs_time_diff_sec": ann["abs_time_diff_sec"], "mc_mi_raw": ann["mc_mi_raw"]})
    y = np.asarray([labels[fid] for fid in ids])
    hard = np.asarray([r["hard_label"] for r in base])
    decisions = np.eye(9)[hard]
    temporal, recordings = reference.temporal_metrics(annotated, decisions, fold)
    metrics = {"fold": fold, "num_windows": len(ids), **reference.e1.metric_dict(decisions, y), **temporal}
    errors = (hard != y).astype(np.int64)
    mi = np.asarray([r["mc_mi_raw"] for r in base])
    require(np.isfinite(mi).all() and (mi >= 0).all(), "Invalid saved MI")
    rankings = []
    for subset, mask in (("all", np.ones(len(y), dtype=bool)), ("state", np.isin(y, (0, 1, 2))),
                         ("transition", np.isin(y, (3, 4, 5, 6, 7, 8)))):
        rank = {"fold": fold, "subset": subset, **reference.e2.ranking_metrics_for_subset(errors, mi, mask, "")}
        rankings.append(rank)
        if subset == "all":
            metrics.update({k: rank[k] for k in ("num_errors", "error_rate", "random_auprc_baseline", "error_auroc", "error_auprc_ap")})
    roundoff, assigned_metrics = {}, []
    for name, probabilities in soft.items():
        result = reference.e3.classification_metrics(probabilities, y, 15)
        require(result["center_accuracy"] == metrics["top1_acc"] and result["center_macro_f1"] == metrics["macro_f1"],
                "E1/E3 classification disagreement")
        for key in ("nll", "brier", "ece"):
            metrics[f"{name}_{key}"] = result[key]
        assigned = np.stack([r["raw_mc" if name == "raw" else "calibrated_mc"] for r in base]).astype(np.float64)
        roundoff[name] = float(np.max(np.abs(assigned - probabilities)))
        assigned_metrics.append({"fold": fold, "probabilities": name,
                                 **reference.e3.classification_metrics(assigned, y, 15)})
    class_f1 = reference.e3.per_class_f1(hard, y, 9)
    classes = [{"fold": fold, "class_id": i, "class_name": label,
                "support": int(np.count_nonzero(y == i)), "f1": float(class_f1[i])} for i, label in enumerate(LABELS)]
    validation = {"fold": fold, **selection, "variants": checks, "radar_gt_mismatches": 0,
                  "soft_target_variants": soft_variants,
                  "unique_associated_skeleton_windows": len(unique_teacher),
                  "radar_vs_associated_skeleton_gt_mismatches": sum(r["radar_vs_skeleton_gt_differs"] for r in joined),
                  "gt_label_identity_sha256": digest([[fid, labels[fid]] for fid in ids]),
                  "radar_pkl": str(gt_path), "radar_pkl_size_bytes": gt_path.stat().st_size,
                  "teacher_alignment_sha256": reference.sha256(alignment_path),
                  "skeleton_audit_sha256": reference.sha256(skeleton_audit_path),
                  "soft_target_vs_assigned_probability_max_abs_diff": roundoff,
                  "radar_annotation_metadata": metadata["annotation_alignment"],
                  "max_retained_gap_sec": max(r["abs_time_diff_sec"] for r in base),
                  "windows_with_association_gap_gt_0_2_sec": sum(r["abs_time_diff_sec"] > 0.2 for r in base)}
    return metrics, recordings, rankings, classes, validation, joined, assigned_metrics


def ground_truth_explanation(payload):
    """Distinguish reference-label disagreement from pseudo-prediction error."""
    count = sum(r["radar_vs_associated_skeleton_gt_mismatches"] for r in payload["validation"])
    errors = sum(r["num_errors"] for r in payload["fold_metrics"])
    total = payload["population"]["fold_row_instances"]
    return ["## Radar-center versus skeleton-center ground truth", "",
            "Each radar window is paired with a skeleton window whose center can occur at a different time. "
            "Both reference labels come from human activity annotations, evaluated at their respective centers:", "",
            "- **Radar-center ground truth:** the manual activity label at the radar window's center timestamp. "
            "This is the reference used for every quality metric in this report.",
            "- **Skeleton-center ground truth:** the manual activity label at the associated skeleton window's "
            "center. This is recorded only for the association audit, not substituted for radar ground truth.", "",
            "Joining by exact radar `frame_id` retrieves the manual label of the same radar window receiving "
            "the pseudo target. `skeleton_sample_id` separately identifies the teacher window. All radar joins "
            "succeeded; the label disagreements below are not failed joins.", "",
            "For example (illustrative), an action boundary may lie between the two centers: the radar "
            "center is still in walk while the skeleton center is already in fall. Their manual labels "
            "then differ even though the association selected the correct nearest teacher window.", "",
            f"**{count:,} of {total:,} fold-row instances ({100 * count / total:.2f}%) have different "
            "radar-center and associated skeleton-center ground-truth labels.** This counts "
            "`radar_manual_label != source_skeleton_manual_label`; it does not count pseudo-label mistakes. "
            f"The separate pseudo-label error count is **{errors:,}**, computed as "
            "`hard_pseudo_label != radar_manual_label`. A pseudo label can match either reference or neither.", "",
            "These counts include repeated windows across folds. Evaluating with skeleton-center ground truth "
            "would measure teacher quality at the teacher's time and could overlook errors from transferring "
            "the target to the radar window. The student-target comparison therefore uses radar-center ground truth.", ""]


def render_report(payload):
    stage = payload.get("stage", "r3")
    is_r3 = stage == "r3"
    gt_directory = "r3" if is_r3 else "radar_dataset"
    variants_text = "six" if is_r3 else "four"
    means = {r["metric"]: r for r in payload["mean_sd"]}
    lines = ["# " + ("R3 retained radar-window pseudo-label quality" if is_r3 else
                      "R2 radar-window pseudo-label quality without an association-gap cutoff"), "",
             "The evaluated objects are retained **radar training windows**, not skeleton windows. "
             f"Every ground-truth label is read from PointCam `data/rerun/{gt_directory}/continuous_window/fold_<fold>.pkl` "
             "and joined by exact radar `frame_id`. No teacher inference, reassignment or filtering changes were made.", "",
             "## Retention and identity checks", "",
             f"All {variants_text} saved pseudo-target variants have identical ordered retained radar IDs, manual labels, "
             "hard labels, raw/calibrated probabilities and skeleton associations. Weighted variants change "
             "loss weights, not sample selection. Assignment CSV IDs, labels, associated skeleton IDs "
             "and probability vectors match their target PKLs.", "",
             "| Fold | Original radar train | Recording drops | Gap drops | Retained radar | Distinct associated skeleton | Radar/skeleton GT differences |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in payload["validation"]:
        keys = ("original_radar_train_windows", "excluded_recording_windows", "excluded_gap_windows",
                "kept_radar_windows", "unique_associated_skeleton_windows", "radar_vs_associated_skeleton_gt_mismatches")
        lines.append(f"| {r['fold']} | " + " | ".join(str(r[k]) for k in keys) + " |")
    selection_text = ("The saved selection audit exactly reproduces the retained IDs using "
              "`recording_id != 38-mia-fall2` and `abs(teacher_center_timestamp_sec - radar_center_timestamp_sec) <= 0.2`. "
              if is_r3 else
              "The saved R2 IDs exactly equal the original radar train split minus `38-mia-fall2`. "
              "Saved R2 metadata records this as `missing_pseudo_recording`, with `max_pseudo_time_diff_sec = null`. "
              "No teacher-to-radar center-gap threshold is applied. The original radar-window validity limits "
              "(adjacent radar gaps <=0.5 s; span <=2.5 s) still apply: 'no gap filter' here refers only to "
              "the teacher-to-radar association gap. ")
    lines += ["", selection_text +
              "These are exclusions from radar training candidates, not subtraction from the original skeleton-window count. "
              "There are zero pseudo-target `manual_label` mismatches against the radar ground-truth PKLs.", ""]
    if not is_r3:
        lines += ["| Fold | Association gap >0.2 s (included) | Maximum association gap (s) |",
                  "| --- | ---: | ---: |"]
        for r in payload["validation"]:
            lines.append(f"| {r['fold']} | {r['windows_with_association_gap_gt_0_2_sec']} | {r['max_retained_gap_sec']:.6f} |")
        lines += [""]
    lines += ground_truth_explanation(payload)
    lines += [
              "## Metrics", "", "Equal-fold mean and sample SD (ddof=1). Edit and segmental F1 use a 0-100 scale. "
              "All metrics are unweighted over retained windows, retaining E1/E3 comparability; sample loss weights "
              "and stochastic training sampler repetitions are not evaluation weights.", "",
              "| Metric | A | B | C | Mean +/- SD |", "| --- | ---: | ---: | ---: | ---: |"]
    specs = [("top1_acc", "Top-1 accuracy"), ("macro_f1", "Macro F1 (9 classes)"), ("edit", "Edit"),
             ("f1_10", "Segmental F1@10"), ("f1_25", "Segmental F1@25"), ("f1_50", "Segmental F1@50"),
             ("error_rate", "Error rate"), ("random_auprc_baseline", "Random AUPRC"),
             ("error_auroc", "Error AUROC (raw MI)"), ("error_auprc_ap", "Error AUPRC/AP (raw MI)")]
    specs += [(f"{version}_{metric}", f"{metric.upper() if metric != 'brier' else 'Brier'} ({version} soft target)")
              for metric in ("nll", "brier", "ece") for version in ("raw", "calibrated")]
    for key, name in specs:
        row = means[key]
        lines.append(f"| {name} | " + " | ".join(f"{r[key]:.4f}" for r in payload["fold_metrics"])
                     + f" | {row['mean']:.4f} +/- {row['sd']:.4f} |")
    lines += ["", "All requested metrics are available. NLL/Brier/ECE above use the exact stored "
              "`soft_label` arrays in `raw_mc_soft` and "
              + ("`calibrated_mc_soft`" if is_r3 else "`weighted_calibrated_mc_soft`")
              + ", which are the student soft targets. "
              "The calibrated target's training loss weight is not applied to evaluation metrics. "
              "PointCam normalizes the float32 assigned vectors again when materializing `soft_label`; "
              "small rounding differences are recorded in JSON. Metrics of the assigned `raw_mc` and "
              "`calibrated_mc` vectors are also saved in `assigned_probability_metrics.csv`. "
              "Hard targets are evaluated as class decisions; no one-hot confidence is invented.", "",
              "## Ground truth and temporal protocol", "",
              "- Radar windows contain 40 retained frames, stride 8, center offset 20. The source builder "
              "requires adjacent timestamp gaps <=0.5 s and total window span <=2.5 s.",
              "- Radar ground truth is the manual camera annotation interval active at the radar-center "
              "timestamp: camera annotation boundary frames map through `timestamps.csv` to half-open "
              "time intervals. The saved radar PKL label is authoritative for this evaluation. "
              "The associated skeleton window instead has 60 detections and center offset 30. "
              "The saved `source_frame_center` in a pseudo target belongs to that skeleton window.",
              "- Each saved teacher association is verified against PointCam's original nearest-center "
              "lookup within the same subject and recording, including its original CSV-row tie-break rule. "
              "The existing saved association is used for evaluation.",
              "- E1 top-1/nine-class macro F1 and current official MS-TCN Edit/F1 functions are reused. "
              "Sequences are grouped by radar recording and sorted by numeric radar window start/center. "
              "Segment IoU uses ordinal positions of retained radar windows. Rejected windows are absent; "
              "there is no interpolation, duration weighting, or new split at a removed gap. Thus these "
              "are retained-sequence metrics, not full original recording segmentation metrics.",
              "- Edit is averaged across recordings; F1 pools TP/FP/FN across recordings. No background "
              "class is excluded. Compare the matching convention with E1 `reports/ms_tcn_official`.",
              "- Error ranking reuses E2B raw MI, with pseudo-label error against radar GT as positive. "
              "AP is threshold-grouped average precision, and random AUPRC is error prevalence.",
              "- E3 NLL uses natural log with 1e-12 clipping; Brier sums squared error over nine classes "
              "without dividing by nine; ECE is 15-bin equal-width top-label calibration.", "",
              "## Interpretation limits", "",
              "This measures assigned teacher-target quality on retained radar training windows. "
              "It does not measure a trained radar student's performance. Radar and skeleton centers "
              "can have different manual labels at their respective center times; their observed disagreement counts "
              "are reported above, so skeleton-center GT is not substituted. Multiple radar windows "
              "may share one skeleton target. Folds overlap, so their mean/SD is descriptive, not an "
              "independent-sample confidence interval. Differences from the preceding skeleton report "
              "combine modality/window/GT changes with filtering and cannot isolate the filter's effect.", "",
              f"Across folds: {payload['population']['fold_row_instances']} radar row instances, "
              f"{payload['population']['unique_radar_frame_ids']} unique radar frame IDs.", "",
              "## Reproduction", "", "Run from the pyskl repository:", "", "```powershell",
              f"python rerun/pseudo_labeling/evaluate_{stage}_radar_pseudo_labels.py", "```", "",
              "`joined_radar_audit.csv` explicitly separates radar GT from associated skeleton GT. "
              "Other outputs include fold/mean-SD, per-recording, per-class and state/transition ranking "
              "CSVs. JSON stores input hashes, identity checks and probability-rounding residuals. "
              "PointCam data and source files are read only.", ""]
    return "\n".join(lines)


def main(default_stage="r3"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("r2", "r3"), default=default_stage)
    parser.add_argument("--pointcam-root", type=Path, default=REPO.parent / "PointCam")
    parser.add_argument("--skeleton-root", type=Path, default=REPO / "data/radar_v4/rerun/yolo26xpose/pseudo_labels_v1")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    if args.report_dir is None:
        directory = "r3_retained_radar_quality" if args.stage == "r3" else "r2_no_gap_radar_quality"
        args.report_dir = REPO / "rerun/pseudo_labeling/reports/oof_pseudo_labels_v1" / directory
    helpers = load_pointcam_helpers(args.pointcam_root)
    keys = ("fold_metrics", "recording_metrics", "ranking_metrics", "class_metrics", "validation",
            "joined_radar_audit", "assigned_probability_metrics")
    payload = {key: [] for key in keys}
    for fold in ("A", "B", "C"):
        result = evaluate_fold(args.pointcam_root, args.skeleton_root, fold, helpers, args.stage)
        for key, value in zip(keys, result):
            payload[key].extend(value if isinstance(value, list) else [value])
        print(f"Verified fold {fold}: {result[0]['num_windows']} retained radar windows", flush=True)
    payload["mean_sd"] = reference.summarize(payload["fold_metrics"])
    payload["stage"] = args.stage
    population = Counter(r["radar_frame_id"] for r in payload["joined_radar_audit"])
    payload["population"] = {"fold_row_instances": sum(population.values()), "unique_radar_frame_ids": len(population),
                             "occurrence_counts": dict(Counter(population.values()))}
    code = [Path(__file__), Path(reference.__file__), Path(reference.segmental.__file__), Path(reference.e1.__file__),
            Path(reference.e2.__file__), Path(reference.e3.__file__), Path(helpers.__file__),
            args.pointcam_root / "src/rerun/r3/dataset.py", args.pointcam_root / "src/rerun/dataset/build_radar_dataset.py"]
    if args.stage == "r2":
        code.append(Path(__file__).with_name("evaluate_r2_radar_pseudo_labels.py"))
    payload["code_sha256"] = {str(p): reference.sha256(p) for p in code}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for key in (*keys[:4], "joined_radar_audit", "assigned_probability_metrics", "mean_sd"):
        reference.e3.write_csv(args.report_dir / f"{key}.csv", payload[key])
    del payload["joined_radar_audit"]
    reference.e3.write_json(args.report_dir / "summary.json", payload)
    (args.report_dir / "summary.md").write_text(render_report(payload), encoding="utf-8")
    print(f"Wrote {args.report_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()

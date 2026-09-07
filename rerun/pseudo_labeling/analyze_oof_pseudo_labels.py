from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


LABELS = [
    "lie-stationary",
    "sit-stationary",
    "walk",
    "fall",
    "transition-lie-to-sit",
    "transition-lie-to-stand",
    "transition-sit-to-lie",
    "transition-sit-to-stand",
    "transition-stand-to-sit",
]
STATE_CLASS_IDS = {0, 1, 2}
TRANSITION_CLASS_IDS = {3, 4, 5, 6, 7, 8}
PROBABILITY_COLUMNS = [f"mc_raw_p{i}" for i in range(len(LABELS))]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze out-of-fold skeleton pseudo labels against the audit-only "
            "manual center labels."
        )
    )
    parser.add_argument(
        "--pseudo-label-root",
        default="data/radar_v4/rerun/yolo26xpose/pseudo_labels_v1",
        help="Root containing fold_a/fold_b/fold_c pseudo-label outputs.",
    )
    parser.add_argument(
        "--report-dir",
        default="rerun/pseudo_labeling/reports/oof_pseudo_labels_v1",
        help="Directory for CSV, JSON, and Markdown reports.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["a", "b", "c"],
        help="Fold suffixes to analyze.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing integer field {key}")
    return int(float(value))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing float field {key}")
    return float(value)


def per_class_f1(predictions: list[int], labels: list[int], num_classes: int) -> list[float]:
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, prediction in zip(labels, predictions):
        if not (0 <= label < num_classes):
            raise ValueError(f"Label {label} is outside [0, {num_classes})")
        if not (0 <= prediction < num_classes):
            raise ValueError(f"Prediction {prediction} is outside [0, {num_classes})")
        confusion[label][prediction] += 1

    f1_scores: list[float] = []
    for class_id in range(num_classes):
        tp = confusion[class_id][class_id]
        predicted = sum(confusion[label][class_id] for label in range(num_classes))
        support = sum(confusion[class_id])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        denom = precision + recall
        f1_scores.append((2.0 * precision * recall / denom) if denom else 0.0)
    return f1_scores


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean_value = average(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def binary_roc_auc(error_labels: list[int], scores: list[float]) -> float:
    if len(error_labels) != len(scores):
        raise ValueError("error_labels and scores must have the same length")
    if not error_labels:
        return math.nan

    n_pos = sum(1 for value in error_labels if value == 1)
    n_neg = len(error_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan

    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0 for _ in scores]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        avg_rank = 0.5 * ((start + 1) + end)
        for order_index in range(start, end):
            ranks[order[order_index]] = avg_rank
        start = end

    rank_sum_pos = sum(ranks[index] for index, label in enumerate(error_labels) if label == 1)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(error_labels: list[int], scores: list[float]) -> float:
    if len(error_labels) != len(scores):
        raise ValueError("error_labels and scores must have the same length")
    if not error_labels:
        return math.nan

    n_pos = sum(1 for value in error_labels if value == 1)
    if n_pos == 0:
        return math.nan

    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    sorted_labels = [error_labels[index] for index in order]
    sorted_scores = [scores[index] for index in order]

    tp = 0
    fp = 0
    previous_recall = 0.0
    ap = 0.0
    for index, label in enumerate(sorted_labels):
        if label == 1:
            tp += 1
        else:
            fp += 1

        is_last_for_score = index == len(sorted_labels) - 1 or sorted_scores[index + 1] != sorted_scores[index]
        if not is_last_for_score:
            continue

        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += (recall - previous_recall) * precision
        previous_recall = recall

    return float(ap)


def top2_accuracy(probabilities: list[list[float]], labels: list[int]) -> float:
    correct = 0
    for probability, label in zip(probabilities, labels):
        top2 = sorted(range(len(probability)), key=lambda index: probability[index], reverse=True)[:2]
        correct += int(label in top2)
    return correct / len(labels) if labels else math.nan


def reliability_weight_to_u_norm(weight: float) -> float:
    return 1.0 - ((weight - 0.1) / 0.9)


def subset_ranking_metrics(
    labels: list[int],
    errors: list[int],
    uncertainty: list[float],
    class_ids: set[int] | None,
    prefix: str,
) -> dict[str, Any]:
    if class_ids is None:
        mask = [True for _ in labels]
    else:
        mask = [label in class_ids for label in labels]

    subset_errors = [error for error, keep in zip(errors, mask) if keep]
    subset_scores = [score for score, keep in zip(uncertainty, mask) if keep]
    n = len(subset_errors)
    error_count = sum(subset_errors)
    correct_count = n - error_count
    error_rate = error_count / n if n else math.nan
    return {
        f"{prefix}num_samples": n,
        f"{prefix}num_errors": error_count,
        f"{prefix}num_correct": correct_count,
        f"{prefix}error_rate": error_rate,
        f"{prefix}random_auprc_baseline": error_rate,
        f"{prefix}error_auroc": binary_roc_auc(subset_errors, subset_scores),
        f"{prefix}error_auprc_ap": average_precision(subset_errors, subset_scores),
    }


def analyze_fold(root: Path, fold: str) -> dict[str, Any]:
    audit_path = root / f"fold_{fold}" / "oof_skeleton_pseudo_labels_audit.csv"
    rows = read_rows(audit_path)
    if not rows:
        raise ValueError(f"No rows in {audit_path}")

    header = set(rows[0])
    missing_prob_cols = [column for column in PROBABILITY_COLUMNS if column not in header]
    if missing_prob_cols:
        raise ValueError(f"{audit_path} is missing probability columns: {missing_prob_cols}")

    explicit_u_norm_columns = [
        column
        for column in (
            "u_norm",
            "uinorm",
            "uncertainty_u_norm",
            "mc_mi_u_norm",
            "normalized_uncertainty",
        )
        if column in header
    ]

    labels = [as_int(row, "manual_label_at_skeleton_center") for row in rows]
    predictions = [as_int(row, "hard_pseudo_label_id") for row in rows]
    probabilities = [[as_float(row, column) for column in PROBABILITY_COLUMNS] for row in rows]

    errors = [int(prediction != label) for prediction, label in zip(predictions, labels)]
    u_norm = [
        min(as_float(row, "mc_mi_raw") / max(as_float(row, "mi_q95_calibration"), EPS), 1.0)
        for row in rows
    ]

    weight_diffs: list[float] = []
    if "reliability_weight" in header:
        for row, reconstructed in zip(rows, u_norm):
            inferred = reliability_weight_to_u_norm(as_float(row, "reliability_weight"))
            weight_diffs.append(abs(inferred - reconstructed))

    f1 = per_class_f1(predictions, labels, len(LABELS))
    n = len(rows)
    result: dict[str, Any] = {
        "fold": fold.upper(),
        "audit_path": str(audit_path),
        "num_windows": n,
        "center_top1_accuracy": sum(1 for error in errors if error == 0) / n,
        "center_top2_accuracy": top2_accuracy(probabilities, labels),
        "center_macro_f1": average(f1),
        "state_macro_f1": average([f1[class_id] for class_id in sorted(STATE_CLASS_IDS)]),
        "transition_macro_f1": average([f1[class_id] for class_id in sorted(TRANSITION_CLASS_IDS)]),
        "explicit_normalized_uncertainty_columns": explicit_u_norm_columns,
        "normalized_uncertainty_saved_explicitly": bool(explicit_u_norm_columns),
        "normalized_uncertainty_reconstructed": True,
        "normalized_uncertainty_source_fields": "mc_mi_raw,mi_q95_calibration",
        "u_norm_min": min(u_norm),
        "u_norm_mean": average(u_norm),
        "u_norm_max": max(u_norm),
        "reliability_weight_u_norm_max_abs_diff": max(weight_diffs) if weight_diffs else math.nan,
    }
    result.update(subset_ranking_metrics(labels, errors, u_norm, None, "all_"))
    result.update(subset_ranking_metrics(labels, errors, u_norm, STATE_CLASS_IDS, "state_"))
    result.update(subset_ranking_metrics(labels, errors, u_norm, TRANSITION_CLASS_IDS, "transition_"))
    return result


def summarize_folds(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = [
        "num_windows",
        "center_top1_accuracy",
        "center_top2_accuracy",
        "center_macro_f1",
        "state_macro_f1",
        "transition_macro_f1",
        "u_norm_min",
        "u_norm_mean",
        "u_norm_max",
        "all_num_samples",
        "all_num_errors",
        "all_error_rate",
        "all_random_auprc_baseline",
        "all_error_auroc",
        "all_error_auprc_ap",
        "state_num_samples",
        "state_num_errors",
        "state_error_rate",
        "state_random_auprc_baseline",
        "state_error_auroc",
        "state_error_auprc_ap",
        "transition_num_samples",
        "transition_num_errors",
        "transition_error_rate",
        "transition_random_auprc_baseline",
        "transition_error_auroc",
        "transition_error_auprc_ap",
        "reliability_weight_u_norm_max_abs_diff",
    ]
    summary: list[dict[str, Any]] = []
    for key in metric_keys:
        values = [
            float(row[key])
            for row in fold_rows
            if key in row and isinstance(row[key], (float, int)) and math.isfinite(float(row[key]))
        ]
        summary.append(
            {
                "metric": key,
                "mean": average(values),
                "sd": sample_sd(values),
                "num_folds": len(values),
            }
        )
    return summary


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def format_value(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (float, int)):
        return str(value)
    value = float(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def format_mean_sd(row: dict[str, Any]) -> str:
    mean_value = row["mean"]
    sd_value = row["sd"]
    if not isinstance(mean_value, (float, int)) or not math.isfinite(float(mean_value)):
        return "NA"
    if not isinstance(sd_value, (float, int)) or not math.isfinite(float(sd_value)):
        return f"{float(mean_value):.4f} +- NA"
    return f"{float(mean_value):.4f} +- {float(sd_value):.4f}"


def render_markdown(fold_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> str:
    summary_by_metric = {row["metric"]: row for row in summary_rows}

    lines = [
        "# OOF Skeleton Pseudo-Label Analysis",
        "",
        "Rows are fold-level out-of-fold pseudo-target windows from the audit CSVs.",
        "Manual center labels are used only for this diagnostic report.",
        "",
        "## Metric Definitions",
        "",
        "- Center top1 accuracy: `hard_pseudo_label_id == manual_label_at_skeleton_center`.",
        "- Center top2 accuracy: manual label appears in the top two `mc_raw_p*` probabilities.",
        "- Center macro-F1: unweighted mean of per-class F1 over all nine final classes.",
        "- State macro-F1: unweighted mean over lie-stationary, sit-stationary, and walk.",
        "- Transition macro-F1: unweighted mean over fall and the five transition classes.",
        "- Error AUROC/AUPRC: binary error detection with pseudo-label error as the positive class and reconstructed `u_norm` as the score.",
        "- Error AUPRC uses the average-precision definition; its random baseline is the error rate.",
        "",
        "## Normalized Uncertainty",
        "",
    ]

    any_explicit = any(row["normalized_uncertainty_saved_explicitly"] for row in fold_rows)
    if any_explicit:
        columns = sorted(
            {
                column
                for row in fold_rows
                for column in row["explicit_normalized_uncertainty_columns"]
            }
        )
        lines.append(f"An explicit normalized uncertainty column is present: `{', '.join(columns)}`.")
    else:
        lines.append(
            "No explicit normalized uncertainty column is present in the fold audit CSVs. "
            "The report reconstructs `u_norm = min(mc_mi_raw / max(mi_q95_calibration, 1e-12), 1)`."
        )
    lines.extend(
        [
            "",
            "## Center Classification",
            "",
            "| Fold | Windows | Top1 Acc | Top2 Acc | Macro-F1 | State Macro-F1 | Transition Macro-F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fold_rows:
        lines.append(
            "| {fold} | {n} | {top1} | {top2} | {macro} | {state} | {transition} |".format(
                fold=row["fold"],
                n=row["num_windows"],
                top1=format_value(row["center_top1_accuracy"]),
                top2=format_value(row["center_top2_accuracy"]),
                macro=format_value(row["center_macro_f1"]),
                state=format_value(row["state_macro_f1"]),
                transition=format_value(row["transition_macro_f1"]),
            )
        )
    lines.append(
        "| Mean +- SD | {n} | {top1} | {top2} | {macro} | {state} | {transition} |".format(
            n=format_mean_sd(summary_by_metric["num_windows"]),
            top1=format_mean_sd(summary_by_metric["center_top1_accuracy"]),
            top2=format_mean_sd(summary_by_metric["center_top2_accuracy"]),
            macro=format_mean_sd(summary_by_metric["center_macro_f1"]),
            state=format_mean_sd(summary_by_metric["state_macro_f1"]),
            transition=format_mean_sd(summary_by_metric["transition_macro_f1"]),
        )
    )

    ranking_specs = [
        ("all", "All Windows"),
        ("state", "State Windows"),
        ("transition", "Transition/Action Windows"),
    ]
    for prefix, title in ranking_specs:
        lines.extend(
            [
                "",
                f"## {title} Error Ranking",
                "",
                "| Fold | Windows | Errors | Error Rate | Random AUPRC | Error AUROC | Error AUPRC/AP |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in fold_rows:
            lines.append(
                "| {fold} | {n} | {errors} | {rate} | {baseline} | {auroc} | {auprc} |".format(
                    fold=row["fold"],
                    n=row[f"{prefix}_num_samples"],
                    errors=row[f"{prefix}_num_errors"],
                    rate=format_value(row[f"{prefix}_error_rate"]),
                    baseline=format_value(row[f"{prefix}_random_auprc_baseline"]),
                    auroc=format_value(row[f"{prefix}_error_auroc"]),
                    auprc=format_value(row[f"{prefix}_error_auprc_ap"]),
                )
            )
        lines.append(
            "| Mean +- SD | {n} | - | {rate} | {baseline} | {auroc} | {auprc} |".format(
                n=format_mean_sd(summary_by_metric[f"{prefix}_num_samples"]),
                rate=format_mean_sd(summary_by_metric[f"{prefix}_error_rate"]),
                baseline=format_mean_sd(summary_by_metric[f"{prefix}_random_auprc_baseline"]),
                auroc=format_mean_sd(summary_by_metric[f"{prefix}_error_auroc"]),
                auprc=format_mean_sd(summary_by_metric[f"{prefix}_error_auprc_ap"]),
            )
        )

    lines.extend(
        [
            "",
            "## Reproducibility Notes",
            "",
            "- `u_norm` is reconstructed from `mc_mi_raw` and `mi_q95_calibration` when no explicit normalized column is saved.",
            "- `reliability_weight` is checked against `0.1 + 0.9 * (1 - u_norm)`.",
            "- State/transition subsets are selected by the manual center label in the audit table.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.pseudo_label_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = [analyze_fold(root, fold.lower()) for fold in args.folds]
    summary_rows = summarize_folds(fold_rows)

    fold_csv = report_dir / "oof_pseudo_label_analysis_fold_metrics.csv"
    summary_csv = report_dir / "oof_pseudo_label_analysis_mean_sd.csv"
    summary_json = report_dir / "oof_pseudo_label_analysis_summary.json"
    summary_md = report_dir / "oof_pseudo_label_analysis_summary.md"

    write_csv(fold_csv, fold_rows)
    write_csv(summary_csv, summary_rows, fieldnames=["metric", "mean", "sd", "num_folds"])

    payload = {
        "pseudo_label_root": str(root),
        "folds": fold_rows,
        "mean_sd": summary_rows,
        "metadata": {
            "labels": LABELS,
            "state_class_ids": sorted(STATE_CLASS_IDS),
            "transition_action_class_ids": sorted(TRANSITION_CLASS_IDS),
            "top2_probability_columns": PROBABILITY_COLUMNS,
            "uncertainty_direction": "larger u_norm means more likely to be wrong",
            "normalized_uncertainty_formula": "min(mc_mi_raw / max(mi_q95_calibration, 1e-12), 1)",
            "error_positive_class": "hard_pseudo_label_id != manual_label_at_skeleton_center",
            "auroc_definition": "rank-based binary AUROC with average ranks for ties",
            "auprc_definition": "average precision, matching sklearn.metrics.average_precision_score convention",
        },
    }
    summary_json.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    summary_md.write_text(render_markdown(fold_rows, summary_rows), encoding="utf-8")

    print(f"Wrote {fold_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    for row in fold_rows:
        print(
            "{fold}: n={n} top1={top1:.4f} top2={top2:.4f} macro_f1={macro:.4f} "
            "all_error_auroc={auroc:.4f} all_error_auprc={auprc:.4f}".format(
                fold=row["fold"],
                n=row["num_windows"],
                top1=row["center_top1_accuracy"],
                top2=row["center_top2_accuracy"],
                macro=row["center_macro_f1"],
                auroc=row["all_error_auroc"],
                auprc=row["all_error_auprc_ap"],
            )
        )


if __name__ == "__main__":
    main()

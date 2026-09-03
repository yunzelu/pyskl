"""E2B self-consistent error-ranking evaluation.

E2B evaluates whether a branch's own uncertainty score identifies that same
branch's activity errors. Positive class is prediction error:

    e_i = 1[argmax_c p_i,c != y_i]

and the ranking score is that branch's mutual information. Larger uncertainty
must mean more likely to be wrong.

No scikit-learn dependency is required. AUROC and average precision are
implemented directly with the same AP definition used by
``sklearn.metrics.average_precision_score``:

    AP = sum_n (R_n - R_{n-1}) P_n

where thresholds are evaluated at distinct uncertainty-score values.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np


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
STATE_CLASS_IDS = [0, 1, 2]
TRANSITION_CLASS_IDS = [3, 4, 5, 6, 7, 8]
FOLDS = ["a", "b", "c"]
BRANCHES = ["mc_dropout", "laplace"]
CONDITION_DIR = "b_continuous_window"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def branch_paths(args: argparse.Namespace, branch: str, fold: str) -> dict[str, Path]:
    if branch == "mc_dropout":
        root = args.mc_root
        quantity_name = "mc_quantities.npz"
        mean_name = "mc_mean_probabilities.npy"
    elif branch == "laplace":
        root = args.laplace_root
        quantity_name = "laplace_quantities.npz"
        mean_name = "laplace_mean_probabilities.npy"
    else:
        raise ValueError(f"Unsupported branch {branch!r}")

    base = root / f"fold_{fold}" / "fusion" / CONDITION_DIR / "validation"
    return {
        "base": base,
        "quantities": base / quantity_name,
        "mean_probabilities": base / mean_name,
        "labels": base / "labels.npy",
        "sample_ids": base / "sample_ids.json",
        "metrics": base / "metrics.json",
    }


def validate_probabilities(values: np.ndarray, name: str, atol: float = 1e-5) -> None:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS):
        raise ValueError(f"{name} expected [N, {len(LABELS)}], got {probabilities.shape}")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(probabilities < -atol) or np.any(probabilities > 1.0 + atol):
        raise ValueError(f"{name} contains values outside [0, 1]")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=atol, rtol=0.0):
        max_error = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(f"{name} rows do not sum to 1; max error={max_error}")


def load_branch_artifact(
    args: argparse.Namespace,
    branch: str,
    fold: str,
) -> dict[str, Any]:
    paths = branch_paths(args, branch, fold)
    for key in ["quantities", "labels", "sample_ids"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing {branch} fold {fold} artifact: {paths[key]}")

    quantities = np.load(paths["quantities"])
    required = ["mean_probabilities", "mutual_information", "label"]
    for key in required:
        if key not in quantities.files:
            raise KeyError(f"{paths['quantities']} is missing key {key!r}")

    probabilities = np.asarray(quantities["mean_probabilities"], dtype=np.float64)
    labels = np.asarray(quantities["label"], dtype=np.int64).reshape(-1)
    labels_from_file = np.load(paths["labels"]).astype(np.int64, copy=False).reshape(-1)
    uncertainty = np.asarray(quantities["mutual_information"], dtype=np.float64).reshape(-1)
    sample_ids = json.loads(paths["sample_ids"].read_text(encoding="utf-8"))

    validate_probabilities(probabilities, f"{branch} fold {fold} predictive mean")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError(f"{branch} fold {fold}: probability/label length mismatch")
    if not np.array_equal(labels, labels_from_file):
        raise ValueError(f"{branch} fold {fold}: labels.npy does not match quantities label")
    if len(sample_ids) != labels.shape[0]:
        raise ValueError(f"{branch} fold {fold}: sample_ids length mismatch")
    if not np.all(np.isfinite(uncertainty)):
        raise ValueError(f"{branch} fold {fold}: uncertainty contains NaN or Inf")
    if np.any(uncertainty < -args.mi_negative_atol):
        raise ValueError(f"{branch} fold {fold}: MI contains values below zero")
    uncertainty = np.maximum(uncertainty, 0.0)

    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    errors = (predictions != labels).astype(np.int64)
    if "prediction" in quantities.files:
        saved_predictions = np.asarray(quantities["prediction"], dtype=np.int64).reshape(-1)
        if not np.array_equal(predictions, saved_predictions):
            raise ValueError(f"{branch} fold {fold}: saved predictions do not match argmax")
    if "error" in quantities.files:
        saved_errors = np.asarray(quantities["error"]).astype(np.int64).reshape(-1)
        if not np.array_equal(errors, saved_errors):
            raise ValueError(f"{branch} fold {fold}: saved errors do not match branch argmax errors")

    metrics = load_json(paths["metrics"]) if paths["metrics"].exists() else {}
    return {
        "branch": branch,
        "fold": fold,
        "paths": paths,
        "probabilities": probabilities,
        "labels": labels,
        "predictions": predictions,
        "errors": errors,
        "uncertainty": uncertainty,
        "sample_ids": sample_ids,
        "source_metrics": metrics,
    }


def binary_roc_auc(error_labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(error_labels, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape[0] != values.shape[0]:
        raise ValueError("error_labels and scores must have the same length")
    if labels.shape[0] == 0:
        return float("nan")
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(np.count_nonzero(positives))
    n_neg = int(np.count_nonzero(negatives))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(values, kind="mergesort")
    sorted_scores = values[order]
    ranks = np.empty(labels.shape[0], dtype=np.float64)
    start = 0
    while start < labels.shape[0]:
        end = start + 1
        while end < labels.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_pos = float(np.sum(ranks[positives]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(error_labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(error_labels, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape[0] != values.shape[0]:
        raise ValueError("error_labels and scores must have the same length")
    if labels.shape[0] == 0:
        return float("nan")
    n_pos = int(np.count_nonzero(labels == 1))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-values, kind="mergesort")
    y_sorted = labels[order]
    scores_sorted = values[order]
    distinct = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct, labels.shape[0] - 1]
    tps = np.cumsum(y_sorted)[threshold_idxs].astype(np.float64)
    fps = (1 + threshold_idxs - tps).astype(np.float64)
    precision = tps / (tps + fps)
    recall = tps / float(n_pos)
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def ranking_metrics_for_subset(
    errors: np.ndarray,
    uncertainty: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    subset_errors = errors[mask].astype(np.int64)
    subset_uncertainty = uncertainty[mask].astype(np.float64)
    n = int(subset_errors.shape[0])
    error_count = int(np.count_nonzero(subset_errors))
    correct_count = int(n - error_count)
    error_rate = float(error_count / n) if n else float("nan")
    return {
        f"{prefix}num_samples": n,
        f"{prefix}num_errors": error_count,
        f"{prefix}num_correct": correct_count,
        f"{prefix}error_rate": error_rate,
        f"{prefix}random_auprc_baseline": error_rate,
        f"{prefix}error_auroc": binary_roc_auc(subset_errors, subset_uncertainty),
        f"{prefix}error_auprc_ap": average_precision(subset_errors, subset_uncertainty),
    }


def fold_branch_metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    labels = artifact["labels"]
    errors = artifact["errors"]
    uncertainty = artifact["uncertainty"]
    state_mask = np.isin(labels, STATE_CLASS_IDS)
    transition_mask = np.isin(labels, TRANSITION_CLASS_IDS)

    row = {
        "branch": artifact["branch"],
        "fold": artifact["fold"],
        "split": "val",
        "condition": "b",
        "uncertainty_score": "mutual_information",
        "positive_class": "branch_specific_activity_error",
        "larger_score_means": "more_likely_wrong",
        "source_path": str(artifact["paths"]["quantities"]),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "median_uncertainty": float(np.median(uncertainty)),
        "max_uncertainty": float(np.max(uncertainty)),
    }
    row.update(ranking_metrics_for_subset(errors, uncertainty, np.ones_like(errors, dtype=bool), ""))
    row.update(ranking_metrics_for_subset(errors, uncertainty, state_mask, "state_"))
    row.update(ranking_metrics_for_subset(errors, uncertainty, transition_mask, "transition_"))

    source_metrics = artifact["source_metrics"]
    if source_metrics:
        row["center_accuracy"] = float(source_metrics.get("center_accuracy", 1.0 - row["error_rate"]))
        row["center_macro_f1"] = float(source_metrics.get("center_macro_f1", np.nan))
    else:
        row["center_accuracy"] = float(1.0 - row["error_rate"])
        row["center_macro_f1"] = float("nan")
    return row


def mean_sd(values: list[float]) -> tuple[float, float, int]:
    finite = np.array([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    sd = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return float(np.mean(finite)), sd, int(finite.size)


def aggregate_mean_sd(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = [
        "error_rate",
        "random_auprc_baseline",
        "error_auroc",
        "error_auprc_ap",
        "state_error_rate",
        "state_error_auroc",
        "transition_error_rate",
        "transition_error_auroc",
        "mean_uncertainty",
        "median_uncertainty",
        "max_uncertainty",
        "center_accuracy",
        "center_macro_f1",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["branch"]), []).append(row)

    summary_rows = []
    for branch in sorted(grouped):
        branch_rows = sorted(grouped[branch], key=lambda row: str(row["fold"]))
        summary: dict[str, Any] = {
            "branch": branch,
            "split": "val",
            "condition": "b",
            "folds": len(branch_rows),
        }
        for key in metric_keys:
            values = [float(row.get(key, np.nan)) for row in branch_rows]
            mean, sd, valid = mean_sd(values)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_sd"] = sd
            summary[f"{key}_valid_folds"] = valid
        summary_rows.append(summary)
    return summary_rows


def format_value(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def format_mean_sd(mean: float, sd: float) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(sd):
        return f"{mean:.4f} +- NA"
    return f"{mean:.4f} +- {sd:.4f}"


def branch_display(branch: str) -> str:
    return {
        "mc_dropout": "MC dropout",
        "laplace": "Laplace",
    }.get(branch, branch)


def subject_from_recording(recording_id: str) -> str:
    parts = str(recording_id).split("-")
    return parts[1].lower() if len(parts) >= 2 else ""


def normalized_sample_key(item: dict[str, Any]) -> tuple[str, str, int, int]:
    recording_id = str(
        item.get(
            "recording_id",
            item.get("session_name", item.get("session", "")),
        )
    )
    subject_id = str(item.get("subject_id", item.get("subject", ""))).lower()
    if not subject_id:
        subject_id = subject_from_recording(recording_id)
    return (
        subject_id,
        recording_id,
        int(item["window_row_start"]),
        int(item["center_source_frame"]),
    )


def markdown_report(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# E2B Self-Consistent Error Ranking",
        "",
        "Split: validation. Protocol: E1-B continuous windows.",
        "",
        "Positive class is branch-specific activity error:",
        "`argmax(p_branch) != manual_center_label`.",
        "Ranking score is the same branch's mutual information; larger MI means more likely wrong.",
        "Error AUPRC is average precision, not trapezoidal PR area.",
        "State/transition AUROCs use subsets defined by the manual center label.",
        "",
        "## Mean +- SD Across Folds",
        "",
        "| Branch | Error rate | Random AUPRC baseline | Error AUROC | Error AUPRC/AP | State Error AUROC | Transition Error AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {branch} | {err} | {base} | {auc} | {ap} | {state_auc} | {trans_auc} |".format(
                branch=branch_display(str(row["branch"])),
                err=format_mean_sd(row["error_rate_mean"], row["error_rate_sd"]),
                base=format_mean_sd(row["random_auprc_baseline_mean"], row["random_auprc_baseline_sd"]),
                auc=format_mean_sd(row["error_auroc_mean"], row["error_auroc_sd"]),
                ap=format_mean_sd(row["error_auprc_ap_mean"], row["error_auprc_ap_sd"]),
                state_auc=format_mean_sd(row["state_error_auroc_mean"], row["state_error_auroc_sd"]),
                trans_auc=format_mean_sd(row["transition_error_auroc_mean"], row["transition_error_auroc_sd"]),
            )
        )

    lines.extend(
        [
            "",
            "## Fold Metrics",
            "",
            "| Branch | Fold | N | Errors | Error rate | Random AUPRC baseline | Error AUROC | Error AUPRC/AP | State Error AUROC | Transition Error AUROC |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["branch"]), str(item["fold"]))):
        lines.append(
            "| {branch} | {fold} | {n} | {err_n} | {err} | {base} | {auc} | {ap} | {state_auc} | {trans_auc} |".format(
                branch=branch_display(str(row["branch"])),
                fold=str(row["fold"]).upper(),
                n=int(row["num_samples"]),
                err_n=int(row["num_errors"]),
                err=format_value(float(row["error_rate"])),
                base=format_value(float(row["random_auprc_baseline"])),
                auc=format_value(float(row["error_auroc"])),
                ap=format_value(float(row["error_auprc_ap"])),
                state_auc=format_value(float(row["state_error_auroc"])),
                trans_auc=format_value(float(row["transition_error_auroc"])),
            )
        )
    return "\n".join(lines) + "\n"


def panel_svg(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    metric: str,
    title: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> list[str]:
    by_branch = {row["branch"]: row for row in summary_rows}
    fold_values: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        fold_values.setdefault(str(row["branch"]), []).append((str(row["fold"]).upper(), float(row[metric])))

    top = y0 + 30
    bottom = y0 + height - 40
    left = x0 + 50
    right = x0 + width - 18
    plot_h = bottom - top

    def sx(branch_index: int) -> float:
        centers = [left + (right - left) * 0.33, left + (right - left) * 0.67]
        return centers[branch_index]

    def sy(value: float) -> float:
        value = min(max(value, 0.0), 1.0)
        return bottom - value * plot_h

    colors = {
        "mc_dropout": "#2f6fed",
        "laplace": "#c75a2a",
    }
    lines = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + 18}" text-anchor="middle" font-size="14" font-weight="600">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" stroke-width="1" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222" stroke-width="1" />',
    ]
    for tick in [0.0, 0.5, 1.0]:
        y = sy(tick)
        lines.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#222" stroke-width="1" />')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10">{tick:.1f}</text>')
        if tick == 0.5:
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#888" stroke-width="1" stroke-dasharray="4 4" />')

    for branch_index, branch in enumerate(BRANCHES):
        if branch not in by_branch:
            continue
        summary = by_branch[branch]
        mean = float(summary[f"{metric}_mean"])
        sd = float(summary[f"{metric}_sd"])
        if not np.isfinite(mean):
            continue
        x = sx(branch_index)
        bar_w = 42
        y = sy(mean)
        color = colors.get(branch, "#666")
        lines.append(
            f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w}" height="{bottom - y:.1f}" fill="{color}" opacity="0.78" />'
        )
        if np.isfinite(sd):
            y_low = sy(mean - sd)
            y_high = sy(mean + sd)
            lines.extend(
                [
                    f'<line x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" stroke="#111" stroke-width="1.4" />',
                    f'<line x1="{x - 9:.1f}" y1="{y_high:.1f}" x2="{x + 9:.1f}" y2="{y_high:.1f}" stroke="#111" stroke-width="1.4" />',
                    f'<line x1="{x - 9:.1f}" y1="{y_low:.1f}" x2="{x + 9:.1f}" y2="{y_low:.1f}" stroke="#111" stroke-width="1.4" />',
                ]
            )
        for fold_index, (fold, value) in enumerate(sorted(fold_values.get(branch, []))):
            if not np.isfinite(value):
                continue
            jitter = [-13, 0, 13][fold_index % 3]
            lines.append(
                f'<circle cx="{x + jitter:.1f}" cy="{sy(value):.1f}" r="4" fill="#fff" stroke="{color}" stroke-width="2">'
                f'<title>{html.escape(branch_display(branch))} fold {fold}: {value:.4f}</title></circle>'
            )
        lines.append(
            f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" font-size="11">{html.escape(branch_display(branch))}</text>'
        )
    return lines


def write_svg_chart(path: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 980
    height = 620
    panels = [
        ("error_auroc", "Error AUROC", 20, 45, 460, 245),
        ("error_auprc_ap", "Error AUPRC/AP", 500, 45, 460, 245),
        ("state_error_auroc", "State Error AUROC", 20, 330, 460, 245),
        ("transition_error_auroc", "Transition Error AUROC", 500, 330, 460, 245),
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="490" y="24" text-anchor="middle" font-size="18" font-weight="700">E2B Error-Ranking Reliability, Mean +- SD Across Folds</text>',
        '<text x="490" y="604" text-anchor="middle" font-size="11" fill="#555">Bars show fold mean; whiskers show SD; points show individual folds. Dashed line marks AUROC chance level.</text>',
    ]
    for metric, title, x0, y0, panel_w, panel_h in panels:
        lines.extend(panel_svg(rows, summary_rows, metric, title, x0, y0, panel_w, panel_h))
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def validate_cross_branch_alignment(artifacts: list[dict[str, Any]]) -> None:
    by_fold: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        by_fold.setdefault(str(artifact["fold"]), []).append(artifact)

    for fold, fold_artifacts in by_fold.items():
        if len(fold_artifacts) < 2:
            continue
        reference = fold_artifacts[0]
        reference_keys = [normalized_sample_key(item) for item in reference["sample_ids"]]
        for artifact in fold_artifacts[1:]:
            current_keys = [normalized_sample_key(item) for item in artifact["sample_ids"]]
            if current_keys != reference_keys:
                raise ValueError(f"Sample IDs differ across E2B branches for fold {fold}")
            if not np.array_equal(artifact["labels"], reference["labels"]):
                raise ValueError(f"Labels differ across E2B branches for fold {fold}")


def run(args: argparse.Namespace) -> None:
    folds = [item.lower() for item in args.folds]
    branches = [item.lower() for item in args.branches]
    for fold in folds:
        if fold not in FOLDS:
            raise ValueError(f"Unknown fold {fold!r}; expected one of {FOLDS}")
    for branch in branches:
        if branch not in BRANCHES:
            raise ValueError(f"Unknown branch {branch!r}; expected one of {BRANCHES}")

    artifacts = [
        load_branch_artifact(args, branch, fold)
        for fold in folds
        for branch in branches
    ]
    validate_cross_branch_alignment(artifacts)
    rows = [fold_branch_metrics(artifact) for artifact in artifacts]
    summary_rows = aggregate_mean_sd(rows)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "e2b_reliability_fold_metrics.csv", rows)
    write_csv(args.report_dir / "e2b_reliability_mean_sd.csv", summary_rows)
    write_json(
        args.report_dir / "e2b_reliability_summary.json",
        {
            "experiment": "E2B self-consistent error ranking",
            "split": "val",
            "condition": "b",
            "branches": branches,
            "folds": folds,
            "positive_class": "branch-specific activity prediction error",
            "uncertainty_score": "branch-specific mutual_information",
            "direction": "larger uncertainty means more likely to be wrong",
            "error_auprc_definition": "average precision: sum_n (R_n - R_{n-1}) P_n",
            "auroc_definition": "rank-based binary AUROC with average ranks for ties",
            "state_subset_definition": "manual center label in lie-stationary, sit-stationary, walk",
            "transition_subset_definition": "manual center label in fall or transition-* classes",
            "state_class_ids": STATE_CLASS_IDS,
            "transition_class_ids": TRANSITION_CLASS_IDS,
            "labels": LABELS,
            "fold_metrics": rows,
            "mean_sd": summary_rows,
        },
    )
    (args.report_dir / "e2b_reliability_summary.md").write_text(
        markdown_report(rows, summary_rows),
        encoding="utf-8",
        newline="\n",
    )
    write_svg_chart(args.report_dir / "e2b_reliability_spread.svg", rows, summary_rows)
    print(f"[DONE] wrote E2B reliability reports under {args.report_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", default=FOLDS)
    parser.add_argument("--branches", nargs="+", default=BRANCHES)
    parser.add_argument(
        "--mc-root",
        type=Path,
        default=Path("work_dirs/rerun/e2/e2a_mc_dropout"),
        help="Root containing fused MC-dropout validation artifacts.",
    )
    parser.add_argument(
        "--laplace-root",
        type=Path,
        default=Path("work_dirs/rerun/e2/e2a_laplace"),
        help="Root containing fused Laplace validation artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("rerun/e2/reports"),
        help="Directory where E2B CSV/JSON/Markdown/SVG reports are written.",
    )
    parser.add_argument(
        "--mi-negative-atol",
        type=float,
        default=1e-8,
        help="Tolerance for small negative MI values before clipping to zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

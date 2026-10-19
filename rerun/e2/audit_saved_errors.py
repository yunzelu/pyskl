"""Audit saved E2 errors before fold averaging, without running inference.

Run: python rerun/e2/audit_saved_errors.py
Requires NumPy only. Reads trusted local E1 prediction pickles and E2 arrays.
Validation comparisons use canonical sample IDs; E1 row order follows the
original PoseDataset order (its prediction pickles do not contain sample IDs).
Test results are explicitly report-only when the raw arrays are unavailable.
"""

import argparse
import csv
import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np


METHODS = ("deterministic", "laplace", "mc_dropout")
STOCHASTIC = {"laplace": "laplace_prob_samples.npy", "mc_dropout": "mc_prob_passes.npy"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_ids(path):
    return [
        (x.get("recording_id", x.get("session_name")),
         int(x["window_row_start"]), int(x["center_source_frame"]))
        for x in read_json(path)
    ]


def validate_probabilities(p):
    require(p.shape[-1] == 9, "Expected nine classes")
    require(np.isfinite(p).all(), "Nonfinite probabilities")
    require(p.min() >= -1e-6 and p.max() <= 1 + 1e-6, "Invalid probabilities")
    require(np.allclose(p.sum(axis=-1), 1, atol=1e-5, rtol=0), "Rows do not sum to one")


def metrics(p, y):
    pred = p.argmax(axis=-1)
    error = pred != y
    confusion = np.bincount(y * 9 + pred, minlength=81).reshape(9, 9)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * confusion.diagonal(), denominator,
                   out=np.zeros(9), where=denominator != 0)
    return {"num_samples": len(y), "num_errors": int(error.sum()),
            "error_rate": float(error.mean()), "accuracy": float(1 - error.mean()),
            "macro_f1": float(f1.mean())}


def check_metrics(actual, saved):
    require(actual["num_samples"] == saved["num_samples"], "Sample count mismatch")
    require(actual["num_errors"] == saved["error_count"], "Error count mismatch")
    for a, b in [("accuracy", "center_accuracy"), ("macro_f1", "center_macro_f1")]:
        require(np.isclose(actual[a], saved[b], atol=1e-12, rtol=0), f"Mismatch: {b}")


def error_overlap(predictions, labels):
    """Partition erroneous windows; shared errors may predict different classes."""
    d, m, l = (predictions[k] != labels for k in ("deterministic", "mc_dropout", "laplace"))
    regions = {
        "all_three_wrong": d & m & l,
        "deterministic_mc_only": d & m & ~l,
        "deterministic_laplace_only": d & ~m & l,
        "mc_laplace_only": ~d & m & l,
        "deterministic_only": d & ~m & ~l,
        "mc_only": ~d & m & ~l,
        "laplace_only": ~d & ~m & l,
    }
    union = d | m | l
    require(np.array_equal(np.stack(list(regions.values())).sum(0), union.astype(int)),
            "Error regions are not a disjoint partition")
    counts = {name: int(mask.sum()) for name, mask in regions.items()}
    require(counts["all_three_wrong"] + counts["deterministic_mc_only"] +
            counts["deterministic_laplace_only"] + counts["deterministic_only"] == int(d.sum()),
            "Deterministic regions do not reconstruct total errors")
    require(counts["all_three_wrong"] + counts["deterministic_mc_only"] +
            counts["mc_laplace_only"] + counts["mc_only"] == int(m.sum()),
            "MC regions do not reconstruct total errors")
    require(counts["all_three_wrong"] + counts["deterministic_laplace_only"] +
            counts["mc_laplace_only"] + counts["laplace_only"] == int(l.sum()),
            "Laplace regions do not reconstruct total errors")
    pair_union = int((m | l).sum())
    return {
        "num_samples": len(labels), "deterministic_errors": int(d.sum()),
        "mc_errors": int(m.sum()), "laplace_errors": int(l.sum()), **counts,
        "any_method_wrong": int(union.sum()), "all_three_correct": int((~union).sum()),
        "mc_laplace_shared_including_deterministic": int((m & l).sum()),
        "mc_wrong_laplace_correct": int((m & ~l).sum()),
        "laplace_wrong_mc_correct": int((~m & l).sum()),
        "mc_laplace_either_wrong": pair_union,
        "mc_laplace_error_jaccard": float((m & l).sum() / pair_union) if pair_union else None,
    }, regions


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("rerun/e2/reports/error_audit"))
    parser.add_argument("--verify-dataset-order", action="store_true",
                        help="Load each original dataset pickle to verify baseline sample ordering.")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = (root / args.output_dir).resolve()
    fold_rows, pair_rows, pass_rows, pass_summaries, disagreements, checks = [], [], [], [], [], []
    overlap_rows, error_membership_rows = [], []
    validation_report = read_json(root / "rerun/e2/reports/e2a_raw_predictive_summary.json")
    report_lookup = {(x["fold"], x["branch"]): x for x in validation_report["fold_metrics"]}

    dataset_checks = []
    for fold in "abc":
        original_ids, original_labels = None, None
        if args.verify_dataset_order:
            dataset_path = root / ("data/radar_v4/rerun/yolo26xpose/pyskl/continuous_window_w60_s12/"
                                   f"radarv4_yolo26xpose_continuous_window_w60_s12_fold_{fold}.pkl")
            with dataset_path.open("rb") as handle:
                dataset = pickle.load(handle)
            identifier = "filename" if "filename" in dataset["annotations"][0] else "frame_dir"
            selected = set(dataset["split"]["val"])
            annotations = [x for x in dataset["annotations"] if x[identifier] in selected]
            original_ids = [(x.get("recording_id", x.get("session_name")), int(x["window_row_start"]),
                             int(x["center_source_frame"])) for x in annotations]
            original_labels = np.array([int(x["label"]) for x in annotations])
            del annotations, dataset
        for stream in ("joint", "bone", "fusion"):
            probabilities = {}
            ids, labels = None, None
            for method, filename in STOCHASTIC.items():
                folder = root / f"work_dirs/rerun/e2/e2a_{method}/fold_{fold}/{stream}/b_continuous_window/validation"
                prefix = "mc" if method == "mc_dropout" else "laplace"
                y = np.load(folder / "labels.npy", allow_pickle=False)
                current_ids = canonical_ids(folder / "sample_ids.json")
                require(len(set(current_ids)) == len(current_ids) == len(y), "Duplicate/missing sample IDs")
                if ids is None:
                    ids, labels = current_ids, y
                require(ids == current_ids and np.array_equal(labels, y), "Sample alignment mismatch")
                if args.verify_dataset_order:
                    require(original_ids == current_ids and np.array_equal(original_labels, y),
                            "Original dataset sample order mismatch")
                p = np.load(folder / f"{prefix}_mean_probabilities.npy", allow_pickle=False)
                draws = np.load(folder / filename, allow_pickle=False)
                validate_probabilities(p)
                validate_probabilities(draws)
                require(draws.ndim == 3 and draws.shape[1:] == p.shape == (len(y), 9), "Shape mismatch")
                mean_difference = float(np.max(np.abs(draws.mean(0) - p)))
                require(mean_difference <= 1e-6, "Saved mean does not match draw mean")
                actual = metrics(p, y)
                saved = read_json(folder / "metrics.json")
                check_metrics(actual, saved)
                if stream == "fusion":
                    check_metrics(actual, report_lookup[(fold, method)])
                    with np.load(folder / f"{prefix}_quantities.npz", allow_pickle=False) as z:
                        for name, expected in [("label", y), ("prediction", p.argmax(1)),
                                               ("error", p.argmax(1) != y), ("mean_probabilities", p)]:
                            require(np.array_equal(z[name], expected), f"Saved NPZ {name} mismatch")
                probabilities[method] = p
                fold_rows.append(dict(split="validation", fold=fold, stream=stream, method=method,
                                      evidence="raw_predictions", **actual))
                errors = draws.argmax(2) != y
                counts = errors.sum(1)
                for i, count in enumerate(counts):
                    pass_rows.append(dict(split="validation", fold=fold, stream=stream, method=method,
                                          inference_draw_index=i + 1, num_samples=len(y), num_errors=int(count),
                                          error_rate=float(count / len(y))))
                pass_summaries.append(dict(
                    split="validation", fold=fold, stream=stream, method=method, num_draws=len(draws),
                    minimum_errors=int(counts.min()), maximum_errors=int(counts.max()),
                    mean_draw_errors=float(counts.mean()), errors_from_mean_probabilities=actual["num_errors"],
                    distinct_error_counts=len(np.unique(counts)),
                    distinct_error_vectors=len(np.unique(np.packbits(errors, axis=1), axis=0)),
                    distinct_probability_arrays=len({hashlib.sha256(d.tobytes()).hexdigest() for d in draws}),
                    samples_with_varying_correctness=int((errors != errors[0]).any(0).sum()),
                    samples_with_varying_prediction=int((draws.argmax(2) != draws[0].argmax(1)).any(0).sum())))
                checks.append(dict(fold=fold, stream=stream, method=method,
                                   sample_alignment_and_saved_metrics="passed", max_mean_difference=mean_difference))

            det_path = root / f"work_dirs/rerun/e1/fold_{fold}/{stream}/b_continuous_window/validation/best_pred.pkl"
            with det_path.open("rb") as handle:
                det = np.asarray(pickle.load(handle), dtype=np.float64)
            if not (det.min() >= -1e-6 and det.max() <= 1 + 1e-6 and
                    np.allclose(det.sum(1), 1, atol=1e-4, rtol=0)):
                det = np.exp(det - det.max(1, keepdims=True))
                det /= det.sum(1, keepdims=True)
            validate_probabilities(det)
            require(det.shape == (len(labels), 9), "Deterministic shape mismatch")
            probabilities["deterministic"] = det
            actual = metrics(det, labels)
            if stream == "fusion":
                check_metrics(actual, report_lookup[(fold, "deterministic")])
                saved_e1 = read_json(det_path.with_name("best_eval.json"))
                require(actual["accuracy"] == saved_e1["top1_acc"] and actual["macro_f1"] == saved_e1["macro_f1"],
                        "E1 fusion metrics mismatch")
            fold_rows.append(dict(split="validation", fold=fold, stream=stream, method="deterministic",
                                  evidence="raw_predictions", **actual))
            for left, right in itertools.combinations(METHODS, 2):
                a, b = probabilities[left], probabilities[right]
                pa, pb = a.argmax(1), b.argmax(1)
                ea, eb = pa != labels, pb != labels
                pair_rows.append(dict(
                    split="validation", fold=fold, stream=stream, left_method=left, right_method=right,
                    num_samples=len(labels), left_errors=int(ea.sum()), right_errors=int(eb.sum()),
                    prediction_disagreements=int((pa != pb).sum()), error_vector_disagreements=int((ea != eb).sum()),
                    both_wrong=int((ea & eb).sum()), left_wrong_right_correct=int((ea & ~eb).sum()),
                    left_correct_right_wrong=int((~ea & eb).sum()),
                    both_wrong_different_class=int((ea & eb & (pa != pb)).sum()),
                    max_abs_probability_difference=float(np.abs(a - b).max()),
                    mean_abs_probability_difference=float(np.abs(a - b).mean())))
            preds = {k: p.argmax(1) for k, p in probabilities.items()}
            overlap, regions = error_overlap(preds, labels)
            overlap_rows.append(dict(split="validation", fold=fold, stream=stream, **overlap))
            if stream == "fusion":
                for region, mask in regions.items():
                    for i in np.flatnonzero(mask):
                        error_membership_rows.append(dict(
                            split="validation", fold=fold, row_index=int(i), region=region,
                            recording_id=ids[i][0], window_row_start=ids[i][1],
                            center_source_frame=ids[i][2], label=int(labels[i]),
                            **{k + "_prediction": int(preds[k][i]) for k in METHODS},
                            **{k + "_error": int(preds[k][i] != labels[i]) for k in METHODS}))
                changed = (preds["deterministic"] != preds["laplace"]) | (preds["deterministic"] != preds["mc_dropout"])
                for i in np.flatnonzero(changed):
                    disagreements.append(dict(
                        fold=fold, row_index=int(i), recording_id=ids[i][0], window_row_start=ids[i][1],
                        center_source_frame=ids[i][2], label=int(labels[i]),
                        **{k + "_prediction": int(preds[k][i]) for k in METHODS},
                        **{k + "_error": int(preds[k][i] != labels[i]) for k in METHODS}))
        if args.verify_dataset_order:
            dataset_checks.append(dict(fold=fold, num_samples=len(original_labels),
                                       ordered_sample_ids_and_labels="passed"))

    test_path = root / "rerun/e2/reports/test/e2a_raw_predictive_fold_metrics.csv"
    missing_test_sources = []
    if test_path.exists():
        with test_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                n, count = int(row["num_samples"]), int(row["error_count"])
                require(np.isclose(1 - count / n, float(row["center_accuracy"]), atol=1e-12, rtol=0), "Test report inconsistency")
                fold_rows.append(dict(split="test", fold=row["fold"], stream="fusion", method=row["branch"],
                                      evidence="saved_report_only", num_samples=n, num_errors=count, error_rate=count / n,
                                      accuracy=float(row["center_accuracy"]), macro_f1=float(row["center_macro_f1"])))
                if not (root / row["source_path"]).exists():
                    missing_test_sources.append(row["source_path"])

    mean_rows = []
    for split in ("validation", "test"):
        for method in METHODS:
            rows = [r for r in fold_rows if r["split"] == split and r["stream"] == "fusion" and r["method"] == method]
            if not rows:
                continue
            rates = np.array([r["error_rate"] for r in rows])
            mean_rows.append(dict(split=split, method=method, folds=len(rows), total_errors=sum(r["num_errors"] for r in rows),
                                  total_samples=sum(r["num_samples"] for r in rows), mean_fold_error_rate=float(rates.mean()),
                                  sd_fold_error_rate=float(rates.std(ddof=1)), displayed_error_rate=f"{rates.mean():.4f}",
                                  mean_fold_accuracy=float(1 - rates.mean()), displayed_accuracy=f"{1 - rates.mean():.4f}"))
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("fold_errors", fold_rows), ("pairwise_errors", pair_rows), ("inference_draw_errors", pass_rows),
                       ("inference_draw_summary", pass_summaries), ("fusion_disagreements", disagreements),
                       ("mean_fold_errors", mean_rows), ("error_overlap", overlap_rows),
                       ("fusion_error_membership", error_membership_rows)]:
        write_csv(out / f"{name}.csv", rows)
    summary = {
        "validation_raw_checks": checks,
        "original_dataset_order_checks": dataset_checks,
        "missing_test_report_sources": missing_test_sources,
        "overlap_definitions": {
            "shared_error": "The same window is misclassified by the specified methods; wrong predicted classes may differ.",
            "seven_regions": "The seven *_only/all_three_wrong regions are mutually exclusive and sum to any_method_wrong.",
            "mc_only": "MC is wrong; deterministic and Laplace are both correct.",
            "mc_wrong_laplace_correct": "MC is wrong and Laplace is correct, irrespective of deterministic correctness.",
            "mc_laplace_shared_including_deterministic": "MC and Laplace are both wrong, whether deterministic is correct or wrong.",
            "mc_laplace_error_jaccard": "Number wrong for both MC and Laplace divided by number wrong for either.",
        },
        "interpretation": [
            "The three folds are evaluated separately; 30 inference draws are not 30 independent training runs.",
            "Predictive probabilities are averaged within a method before argmax; fold metrics are averaged afterward.",
            "Errors of mean probabilities need not equal the mean number of errors across draws.",
            "Equal error counts do not imply identical predicted classes or error sample identities.",
            "Laplace draw indices are batchwise samples, not necessarily one fixed posterior weight draw across the dataset.",
            "Test rows use reports only; absent test predictions prevent checking exact test error overlap or individual draws.",
            "Deterministic sample order relies on E1 PoseDataset ordering; its prediction pickles do not store sample IDs.",
        ],
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "validated_stochastic_artifacts": len(checks),
                      "inference_draw_rows": len(pass_rows), "missing_test_sources": len(missing_test_sources),
                      "fusion_means": mean_rows}, indent=2))


if __name__ == "__main__":
    main()

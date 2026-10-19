# OOF pseudo-label quality: E1-B / E3 protocol audit

All requested metrics are computable from the saved artifacts in this run.

Manual annotations are used only for this evaluation. No inference, retraining, filtering, resampling, or temperature refitting was performed.

## Results

Mean and sample SD (ddof=1) are over the three outer-fold OOF target sets. Edit and segmental F1 are on a 0-100 scale; other metrics use their native scale.

| Metric | A | B | C | Mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| Top-1 accuracy | 0.9423 | 0.9435 | 0.9452 | 0.9437 +/- 0.0015 |
| Macro F1 (9 classes) | 0.9027 | 0.9042 | 0.9079 | 0.9049 +/- 0.0027 |
| Edit | 97.1748 | 97.7236 | 96.8106 | 97.2363 +/- 0.4596 |
| Segmental F1@10 | 98.3169 | 99.0539 | 98.3239 | 98.5649 +/- 0.4235 |
| Segmental F1@25 | 98.1680 | 98.9593 | 98.2613 | 98.4629 +/- 0.4325 |
| Segmental F1@50 | 96.4701 | 96.6414 | 96.4442 | 96.5186 +/- 0.1072 |
| Error rate | 0.0577 | 0.0565 | 0.0548 | 0.0563 +/- 0.0015 |
| Random AUPRC | 0.0577 | 0.0565 | 0.0548 | 0.0563 +/- 0.0015 |
| Error AUROC (raw MI) | 0.9023 | 0.8964 | 0.9025 | 0.9004 +/- 0.0035 |
| Error AUPRC/AP (raw MI) | 0.2684 | 0.2675 | 0.2713 | 0.2691 +/- 0.0020 |
| NLL (raw) | 0.1505 | 0.1507 | 0.1440 | 0.1484 +/- 0.0038 |
| NLL (calibrated) | 0.1508 | 0.1498 | 0.1427 | 0.1478 +/- 0.0044 |
| Brier (raw) | 0.0850 | 0.0834 | 0.0806 | 0.0830 +/- 0.0022 |
| Brier (calibrated) | 0.0850 | 0.0832 | 0.0799 | 0.0827 +/- 0.0026 |
| ECE (raw) | 0.0069 | 0.0126 | 0.0106 | 0.0100 +/- 0.0029 |
| ECE (calibrated) | 0.0064 | 0.0117 | 0.0044 | 0.0075 +/- 0.0038 |
| Error AUROC (legacy normalized MI) | 0.9024 | 0.8973 | 0.9034 | 0.9011 +/- 0.0033 |
| Error AUPRC/AP (legacy normalized MI) | 0.2635 | 0.2616 | 0.2721 | 0.2657 +/- 0.0056 |

Calibration changes in fold means (calibrated minus raw): NLL -0.0006, BRIER -0.0003, ECE -0.0025. Negative values indicate improvement; the table also shows variation by fold. Error AP should be interpreted alongside the error-prevalence baseline.

### Per-class F1

| Class | A | B | C | Mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| lie-stationary | 0.9313 | 0.9246 | 0.9296 | 0.9285 +/- 0.0035 |
| sit-stationary | 0.9610 | 0.9579 | 0.9622 | 0.9604 +/- 0.0022 |
| walk | 0.9795 | 0.9817 | 0.9807 | 0.9806 +/- 0.0011 |
| fall | 0.8827 | 0.8634 | 0.8796 | 0.8753 +/- 0.0104 |
| transition-lie-to-sit | 0.8976 | 0.9084 | 0.8871 | 0.8977 +/- 0.0106 |
| transition-lie-to-stand | 0.9528 | 0.9513 | 0.9540 | 0.9527 +/- 0.0014 |
| transition-sit-to-lie | 0.7944 | 0.8028 | 0.8544 | 0.8172 +/- 0.0325 |
| transition-sit-to-stand | 0.8612 | 0.8733 | 0.8488 | 0.8611 +/- 0.0122 |
| transition-stand-to-sit | 0.8637 | 0.8742 | 0.8749 | 0.8709 +/- 0.0063 |

## Annotation and artifact checks

Every audit row is joined by `source_pyskl_frame_dir` to the E1-B continuous pkl's outer **train** split. Exact coverage, unique identities, recording, subject, source start/center/end frames, center label, window offsets and OOF exclusions are checked.

| Fold | Windows | Recordings | Subjects | Label mismatches | Missing/extra windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 43293 | 29 | 8 | 0 | 0 |
| B | 43742 | 33 | 8 | 0 | 0 |
| C | 43335 | 34 | 8 | 0 | 0 |

Saved raw and calibrated argmax decisions agree with the hard pseudo labels. E1 and E3 classification implementations give identical results. Calibrated probabilities are checked against each teacher's saved temperature, and reliability weights against reconstructed normalized MI. The previous pseudo report's accuracy, macro F1 and normalized-MI ranking results are reproduced.

## Metric equivalence

| Metric | Definition and reused implementation |
| --- | --- |
| Top-1, macro F1 | `rerun/e1/summarize_results.py::metric_dict`; checked against `rerun/e3/run_e3_mc_temperature_scaling.py::classification_metrics`. Nine-class macro F1 includes absent classes as zero, with no class weighting. |
| Edit | E1 `grouped_sequences` and `evaluate_recording`: collapse consecutive labels, normalize Levenshtein by the larger segment count, then average recording scores. |
| Segmental F1 | E1 `evaluate_recording` / `f1_from_counts`: IoU thresholds 0.10/0.25/0.50; sum TP/FP/FN across recordings before F1. No background excluded. |
| Error rate / random AUPRC | Error is hard argmax != manual center label. Random-ranking AUPRC baseline equals error prevalence. |
| Error AUROC/AP | E2B `ranking_metrics_for_subset`: tied-score average-rank AUROC; AP is sum of precision times recall increments at distinct scores, not trapezoidal PR area. |
| NLL | E3: mean -ln(p_true), clipping p_true to [1e-12,1]. |
| Brier | E3: mean over windows of sum over 9 classes of (p-onehot)^2; no division by class count. |
| ECE | E3: top-label confidence, 15 equal-width bins; [lower,upper), last bin includes 1. Weighted absolute bin accuracy-confidence difference. |

NLL/Brier/ECE use the saved `mc_raw_p0..8` and `mc_cal_p0..8` as stored; they are not replaced by one-hot labels or refitted probabilities.

## Ambiguities resolved and comparison limits

1. **Center-time protocol:** E1-B takes retained detection index `start + 30` in a 60-detection, stride-12 window (gap <=0.5 s, span <=2.5 s). Its original source-frame annotation supplies the label. It is not a timestamp midpoint or majority label. Aggregation replaced nominal times with real camera timestamps; identity/source-frame alignment is used, not equality of those timestamps.
2. **Temporal axis:** E1's segment metrics use ordered window indices within each recording, sorted by window start then source center frame. They do not use elapsed-second IoU, interpolate missing windows, or split sequences at rejected window gaps. Different recordings are never concatenated.
3. **E1 report version:** current E1 uses official MS-TCN best-overlap-then-check-matched matching, without a fallback if the best reference is already matched. Compare with `rerun/e1/reports/ms_tcn_official`, not the original root reports using the earlier best-unmatched rule.
4. **Ranking score:** E1/E3 do not define error AUROC/AP. Their rerun companion E2B uses raw MI; that score is primary here. The previous pseudo report uses `min(mc_mi_raw / max(mi_q95_calibration,1e-12),1)`. Teacher-specific normalization changes pooled ranking and clipping creates ties, so both are reported separately. Larger score means more likely wrong. MI remains the raw MC uncertainty even when evaluating the order-preserving calibrated probabilities.
5. **Evaluation population:** these are held-out inner-teacher targets covering each outer fold's eight training subjects, not E1/E3 outer-test subjects. Subjects and windows overlap across outer folds. Equal-fold mean +/- SD is descriptive, not an independent-sample confidence interval or a paired improvement over E1/E3.
6. **Probability variants:** report both raw 30-pass joint/bone MC means and saved pool-temperature-calibrated probabilities. Classification/segment scores coincide because their argmax decisions are identical. Calibration temperatures are teacher-specific and fitted on the fixed calibration subject only.

## Availability and reproducibility

No requested metric was skipped.

Run from the repository root:

```powershell
python rerun/pseudo_labeling/evaluate_oof_pseudo_labels.py
```

The JSON contains input/code SHA-256 hashes and numerical validation residuals. CSV outputs include fold metrics, fold mean/SD, per-recording segment counts, per-class F1 and state/transition error ranking with both uncertainty scores.

The separate [saved artifact audit](saved_artifact_audit.json) reconstructs MC fusion, means, uncertainty and calibration from archived passes and verifies teacher/stream sample ordering and saved subject roles. Regenerate it with `python rerun/pseudo_labeling/audit_oof_saved_artifacts.py`. Its population counts distinguish repeated fold rows from unique source windows. Subject separation is verified from saved metadata, not historical training execution.

# OOF Skeleton Pseudo-Label Analysis

Rows are fold-level out-of-fold pseudo-target windows from the audit CSVs.
Manual center labels are used only for this diagnostic report.

## Metric Definitions

- Center top1 accuracy: `hard_pseudo_label_id == manual_label_at_skeleton_center`.
- Center top2 accuracy: manual label appears in the top two `mc_raw_p*` probabilities.
- Center macro-F1: unweighted mean of per-class F1 over all nine final classes.
- State macro-F1: unweighted mean over lie-stationary, sit-stationary, and walk.
- Transition macro-F1: unweighted mean over fall and the five transition classes.
- Error AUROC/AUPRC: binary error detection with pseudo-label error as the positive class and reconstructed `u_norm` as the score.
- Error AUPRC uses the average-precision definition; its random baseline is the error rate.

## Normalized Uncertainty

No explicit normalized uncertainty column is present in the fold audit CSVs. The report reconstructs `u_norm = min(mc_mi_raw / max(mi_q95_calibration, 1e-12), 1)`.

## Center Classification

| Fold | Windows | Top1 Acc | Top2 Acc | Macro-F1 | State Macro-F1 | Transition Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 43293 | 0.9423 | 0.9967 | 0.9027 | 0.9573 | 0.8754 |
| B | 43742 | 0.9435 | 0.9971 | 0.9042 | 0.9547 | 0.8789 |
| C | 43335 | 0.9452 | 0.9973 | 0.9079 | 0.9575 | 0.8831 |
| Mean +- SD | 43456.6667 +- 247.9966 | 0.9437 +- 0.0015 | 0.9970 +- 0.0003 | 0.9049 +- 0.0027 | 0.9565 +- 0.0016 | 0.8792 +- 0.0039 |

## All Windows Error Ranking

| Fold | Windows | Errors | Error Rate | Random AUPRC | Error AUROC | Error AUPRC/AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 43293 | 2496 | 0.0577 | 0.0577 | 0.9024 | 0.2635 |
| B | 43742 | 2473 | 0.0565 | 0.0565 | 0.8973 | 0.2616 |
| C | 43335 | 2373 | 0.0548 | 0.0548 | 0.9034 | 0.2721 |
| Mean +- SD | 43456.6667 +- 247.9966 | - | 0.0563 +- 0.0015 | 0.0563 +- 0.0015 | 0.9011 +- 0.0033 | 0.2657 +- 0.0056 |

## State Windows Error Ranking

| Fold | Windows | Errors | Error Rate | Random AUPRC | Error AUROC | Error AUPRC/AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 31236 | 1599 | 0.0512 | 0.0512 | 0.9253 | 0.2890 |
| B | 31868 | 1419 | 0.0445 | 0.0445 | 0.9323 | 0.2900 |
| C | 31174 | 1363 | 0.0437 | 0.0437 | 0.9254 | 0.2738 |
| Mean +- SD | 31426.0000 +- 384.0365 | - | 0.0465 +- 0.0041 | 0.0465 +- 0.0041 | 0.9277 +- 0.0040 | 0.2843 +- 0.0091 |

## Transition/Action Windows Error Ranking

| Fold | Windows | Errors | Error Rate | Random AUPRC | Error AUROC | Error AUPRC/AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 12057 | 897 | 0.0744 | 0.0744 | 0.8472 | 0.2383 |
| B | 11874 | 1054 | 0.0888 | 0.0888 | 0.8030 | 0.2312 |
| C | 12161 | 1010 | 0.0831 | 0.0831 | 0.8496 | 0.2715 |
| Mean +- SD | 12030.6667 +- 145.3008 | - | 0.0821 +- 0.0072 | 0.0821 +- 0.0072 | 0.8333 +- 0.0262 | 0.2470 +- 0.0215 |

## Reproducibility Notes

- `u_norm` is reconstructed from `mc_mi_raw` and `mi_q95_calibration` when no explicit normalized column is saved.
- `reliability_weight` is checked against `0.1 + 0.9 * (1 - u_norm)`.
- State/transition subsets are selected by the manual center label in the audit table.

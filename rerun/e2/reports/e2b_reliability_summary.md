# E2B Self-Consistent Error Ranking

Split: validation. Protocol: E1-B continuous windows.

Positive class is branch-specific activity error:
`argmax(p_branch) != manual_center_label`.
Ranking score is the same branch's mutual information; larger MI means more likely wrong.
Error AUPRC is average precision, not trapezoidal PR area.
State/transition AUROCs use subsets defined by the manual center label.

## Mean +- SD Across Folds

| Branch | Error rate | Random AUPRC baseline | Error AUROC | Error AUPRC/AP | State Error AUROC | Transition Error AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laplace | 0.0532 +- 0.0102 | 0.0532 +- 0.0102 | 0.9189 +- 0.0179 | 0.3179 +- 0.0115 | 0.9451 +- 0.0150 | 0.8702 +- 0.1088 |
| MC dropout | 0.0533 +- 0.0106 | 0.0533 +- 0.0106 | 0.9217 +- 0.0202 | 0.3134 +- 0.0412 | 0.9428 +- 0.0035 | 0.8778 +- 0.0799 |

## Fold Metrics

| Branch | Fold | N | Errors | Error rate | Random AUPRC baseline | Error AUROC | Error AUPRC/AP | State Error AUROC | Transition Error AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Laplace | A | 5766 | 357 | 0.0619 | 0.0619 | 0.8983 | 0.3121 | 0.9624 | 0.7449 |
| Laplace | B | 3870 | 216 | 0.0558 | 0.0558 | 0.9304 | 0.3311 | 0.9366 | 0.9401 |
| Laplace | C | 4505 | 189 | 0.0420 | 0.0420 | 0.9280 | 0.3105 | 0.9362 | 0.9256 |
| MC dropout | A | 5766 | 357 | 0.0619 | 0.0619 | 0.8985 | 0.3072 | 0.9467 | 0.7855 |
| MC dropout | B | 3870 | 219 | 0.0566 | 0.0566 | 0.9352 | 0.3573 | 0.9421 | 0.9270 |
| MC dropout | C | 4505 | 187 | 0.0415 | 0.0415 | 0.9315 | 0.2757 | 0.9397 | 0.9208 |

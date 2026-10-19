# E2B Self-Consistent Error Ranking

Split: test. Protocol: E1-B continuous windows.

Positive class is branch-specific activity error:
`argmax(p_branch) != manual_center_label`.
Ranking score is the same branch's mutual information; larger MI means more likely wrong.
Error AP is non-interpolated average precision: `AP = sum_n (R_n - R_{n-1}) P_n`.
State/transition AUROCs use subsets defined by the manual center label.

## Mean +- SD Across Folds

| Branch | Error rate | Random AP baseline | Error AUROC | Error AP | State Error AUROC | Transition Error AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laplace | 0.0452 +- 0.0071 | 0.0452 +- 0.0071 | 0.9249 +- 0.0052 | 0.2890 +- 0.0529 | 0.9499 +- 0.0028 | 0.8734 +- 0.0376 |
| MC dropout | 0.0452 +- 0.0074 | 0.0452 +- 0.0074 | 0.9139 +- 0.0026 | 0.2552 +- 0.0521 | 0.9360 +- 0.0143 | 0.8696 +- 0.0290 |

## Fold Metrics

| Branch | Fold | N | Errors | Error rate | Random AP baseline | Error AUROC | Error AP | State Error AUROC | Transition Error AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Laplace | A | 4334 | 228 | 0.0526 | 0.0526 | 0.9192 | 0.3498 | 0.9494 | 0.8396 |
| Laplace | B | 4934 | 220 | 0.0446 | 0.0446 | 0.9261 | 0.2636 | 0.9530 | 0.8667 |
| Laplace | C | 4964 | 191 | 0.0385 | 0.0385 | 0.9294 | 0.2537 | 0.9474 | 0.9138 |
| MC dropout | A | 4334 | 230 | 0.0531 | 0.0531 | 0.9109 | 0.3131 | 0.9221 | 0.8848 |
| MC dropout | B | 4934 | 218 | 0.0442 | 0.0442 | 0.9151 | 0.2403 | 0.9507 | 0.8362 |
| MC dropout | C | 4964 | 190 | 0.0383 | 0.0383 | 0.9156 | 0.2121 | 0.9351 | 0.8880 |

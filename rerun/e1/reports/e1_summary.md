# E1 Result Summary

Fusion uses `0.5 * (joint_probability + bone_probability)`. The saved
`best_pred.pkl` files already contain softmax probabilities from PYSKL, so
fusion is applied directly to those probabilities.

## Mean +- SD Across Folds

| Condition | Stream | Top-1 Acc | Macro F1 |
| --- | --- | ---: | ---: |
| A1 | joint | 0.9924 +- 0.0050 | 0.9918 +- 0.0063 |
| A1 | bone | 0.9973 +- 0.0035 | 0.9972 +- 0.0036 |
| A1 | fusion | 0.9991 +- 0.0016 | 0.9989 +- 0.0018 |
| A2 | joint | 0.7951 +- 0.0367 | 0.7088 +- 0.0454 |
| A2 | bone | 0.7832 +- 0.0455 | 0.6693 +- 0.0780 |
| A2 | fusion | 0.8056 +- 0.0340 | 0.7037 +- 0.0678 |
| B | joint | 0.9506 +- 0.0095 | 0.9201 +- 0.0167 |
| B | bone | 0.9524 +- 0.0069 | 0.9207 +- 0.0145 |
| B | fusion | 0.9548 +- 0.0074 | 0.9262 +- 0.0147 |

## Fold Metrics

| Condition | Fold | Stream | N | Top-1 Acc | Macro F1 |
| --- | --- | --- | ---: | ---: | ---: |
| A1 | A | joint | 709 | 0.9873 | 0.9849 |
| A1 | A | bone | 709 | 0.9986 | 0.9984 |
| A1 | A | fusion | 709 | 0.9972 | 0.9968 |
| A1 | B | joint | 744 | 0.9973 | 0.9972 |
| A1 | B | bone | 744 | 0.9933 | 0.9931 |
| A1 | B | fusion | 744 | 1.0000 | 1.0000 |
| A1 | C | joint | 808 | 0.9926 | 0.9934 |
| A1 | C | bone | 808 | 1.0000 | 1.0000 |
| A1 | C | fusion | 808 | 1.0000 | 1.0000 |
| A2 | A | joint | 4334 | 0.8366 | 0.7367 |
| A2 | A | bone | 4334 | 0.7545 | 0.6283 |
| A2 | A | fusion | 4334 | 0.8039 | 0.6836 |
| A2 | B | joint | 4934 | 0.7669 | 0.6564 |
| A2 | B | bone | 4934 | 0.7594 | 0.6202 |
| A2 | B | fusion | 4934 | 0.7726 | 0.6482 |
| A2 | C | joint | 4964 | 0.7818 | 0.7333 |
| A2 | C | bone | 4964 | 0.8356 | 0.7592 |
| A2 | C | fusion | 4964 | 0.8405 | 0.7793 |
| B | A | joint | 4334 | 0.9400 | 0.9030 |
| B | A | bone | 4334 | 0.9467 | 0.9106 |
| B | A | fusion | 4334 | 0.9469 | 0.9124 |
| B | B | joint | 4934 | 0.9532 | 0.9207 |
| B | B | bone | 4934 | 0.9503 | 0.9141 |
| B | B | fusion | 4934 | 0.9558 | 0.9245 |
| B | C | joint | 4964 | 0.9585 | 0.9365 |
| B | C | bone | 4964 | 0.9601 | 0.9372 |
| B | C | fusion | 4964 | 0.9617 | 0.9416 |

## Continuous Segmental Metrics

Segmental metrics are computed only for continuous-window evaluation
conditions A2 and B. Each recording is evaluated independently;
segmental F1 pools TP, FP, and FN counts across recordings within
a fold, while Edit is normalized per recording and then averaged.

| Condition | Stream | Edit | F1@10 | F1@25 | F1@50 |
| --- | --- | ---: | ---: | ---: | ---: |
| A2 | joint | 78.32 +- 6.38 | 85.99 +- 3.59 | 84.06 +- 4.75 | 67.87 +- 6.34 |
| A2 | bone | 77.40 +- 8.75 | 82.10 +- 7.71 | 80.49 +- 8.98 | 66.23 +- 9.85 |
| A2 | fusion | 79.24 +- 4.42 | 85.08 +- 5.05 | 84.06 +- 5.74 | 69.64 +- 8.67 |
| B | joint | 98.70 +- 0.86 | 99.29 +- 0.47 | 99.21 +- 0.52 | 97.75 +- 0.25 |
| B | bone | 98.27 +- 1.46 | 98.94 +- 1.01 | 98.90 +- 1.08 | 97.39 +- 1.61 |
| B | fusion | 99.08 +- 0.88 | 99.52 +- 0.39 | 99.48 +- 0.46 | 98.28 +- 0.63 |

| Condition | Fold | Stream | Recordings | Windows | Edit | F1@10 | F1@25 | F1@50 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 | A | joint | 3 | 4334 | 84.07 | 90.01 | 89.51 | 74.66 |
| A2 | A | bone | 3 | 4334 | 70.06 | 76.92 | 75.40 | 60.72 |
| A2 | A | fusion | 3 | 4334 | 76.89 | 82.54 | 81.81 | 67.92 |
| A2 | B | joint | 3 | 4934 | 79.43 | 84.83 | 81.78 | 62.10 |
| A2 | B | bone | 3 | 4934 | 75.07 | 78.42 | 75.21 | 60.36 |
| A2 | B | fusion | 3 | 4934 | 76.50 | 81.80 | 79.79 | 61.96 |
| A2 | C | joint | 3 | 4964 | 71.45 | 83.11 | 80.88 | 66.86 |
| A2 | C | bone | 3 | 4964 | 87.08 | 90.97 | 90.86 | 77.61 |
| A2 | C | fusion | 3 | 4964 | 84.35 | 90.90 | 90.58 | 79.05 |
| B | A | joint | 3 | 4334 | 98.86 | 99.39 | 99.25 | 98.02 |
| B | A | bone | 3 | 4334 | 99.63 | 99.79 | 99.79 | 99.25 |
| B | A | fusion | 3 | 4334 | 99.42 | 99.66 | 99.66 | 98.97 |
| B | B | joint | 3 | 4934 | 97.76 | 98.79 | 98.67 | 97.69 |
| B | B | bone | 3 | 4934 | 96.72 | 97.82 | 97.70 | 96.37 |
| B | B | fusion | 3 | 4934 | 98.07 | 99.09 | 98.96 | 98.11 |
| B | C | joint | 3 | 4964 | 99.46 | 99.71 | 99.71 | 97.52 |
| B | C | bone | 3 | 4964 | 98.45 | 99.20 | 99.20 | 96.56 |
| B | C | fusion | 3 | 4964 | 99.74 | 99.83 | 99.83 | 97.75 |

## Continuous Segmental Metric Readiness

Continuous-window pkls support per-recording temporal evaluation.
Every test sample has `session_name`, `window_row_start`,
`center_source_frame`, `center_timestamp_sec`, `label`, and
`frame_dir`. For segmental Edit/F1@k, group by `session_name` and
sort each group by `window_row_start` or `center_source_frame`.
Do not concatenate different `session_name` groups.

| Fold | Test Sequences | Test Windows | Missing Required Fields | Ordering Issues |
| --- | ---: | ---: | ---: | ---: |
| A | 3 | 4334 | 0 | 0 |
| B | 3 | 4934 | 0 | 0 |
| C | 3 | 4964 | 0 | 0 |

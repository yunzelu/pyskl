# R2 radar-window pseudo-label quality without an association-gap cutoff

The evaluated objects are retained **radar training windows**, not skeleton windows. Every ground-truth label is read from PointCam `data/rerun/radar_dataset/continuous_window/fold_<fold>.pkl` and joined by exact radar `frame_id`. No teacher inference, reassignment or filtering changes were made.

## Retention and identity checks

All four saved pseudo-target variants have identical ordered retained radar IDs, manual labels, hard labels, raw/calibrated probabilities and skeleton associations. Weighted variants change loss weights, not sample selection. Assignment CSV IDs, labels, associated skeleton IDs and probability vectors match their target PKLs.

| Fold | Original radar train | Recording drops | Gap drops | Retained radar | Distinct associated skeleton | Radar/skeleton GT differences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 26609 | 86 | 0 | 26523 | 23140 | 1097 |
| B | 27190 | 86 | 0 | 27104 | 24078 | 1049 |
| C | 27696 | 0 | 0 | 27696 | 24397 | 1012 |

The saved R2 IDs exactly equal the original radar train split minus `38-mia-fall2`. Saved R2 metadata records this as `missing_pseudo_recording`, with `max_pseudo_time_diff_sec = null`. No teacher-to-radar center-gap threshold is applied. The original radar-window validity limits (adjacent radar gaps <=0.5 s; span <=2.5 s) still apply: 'no gap filter' here refers only to the teacher-to-radar association gap. These are exclusions from radar training candidates, not subtraction from the original skeleton-window count. There are zero pseudo-target `manual_label` mismatches against the radar ground-truth PKLs.

| Fold | Association gap >0.2 s (included) | Maximum association gap (s) |
| --- | ---: | ---: |
| A | 3604 | 6.599564 |
| B | 3187 | 6.211313 |
| C | 3481 | 6.599564 |

## Radar-center versus skeleton-center ground truth

Each radar window is paired with a skeleton window whose center can occur at a different time. Both reference labels come from human activity annotations, evaluated at their respective centers:

- **Radar-center ground truth:** the manual activity label at the radar window's center timestamp. This is the reference used for every quality metric in this report.
- **Skeleton-center ground truth:** the manual activity label at the associated skeleton window's center. This is recorded only for the association audit, not substituted for radar ground truth.

Joining by exact radar `frame_id` retrieves the manual label of the same radar window receiving the pseudo target. `skeleton_sample_id` separately identifies the teacher window. All radar joins succeeded; the label disagreements below are not failed joins.

For example (illustrative), an action boundary may lie between the two centers: the radar center is still in walk while the skeleton center is already in fall. Their manual labels then differ even though the association selected the correct nearest teacher window.

**3,158 of 81,323 fold-row instances (3.88%) have different radar-center and associated skeleton-center ground-truth labels.** This counts `radar_manual_label != source_skeleton_manual_label`; it does not count pseudo-label mistakes. The separate pseudo-label error count is **5,364**, computed as `hard_pseudo_label != radar_manual_label`. A pseudo label can match either reference or neither.

These counts include repeated windows across folds. Evaluating with skeleton-center ground truth would measure teacher quality at the teacher's time and could overlook errors from transferring the target to the radar window. The student-target comparison therefore uses radar-center ground truth.

## Metrics

Equal-fold mean and sample SD (ddof=1). Edit and segmental F1 use a 0-100 scale. All metrics are unweighted over retained windows, retaining E1/E3 comparability; sample loss weights and stochastic training sampler repetitions are not evaluation weights.

| Metric | A | B | C | Mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| Top-1 accuracy | 0.9282 | 0.9358 | 0.9379 | 0.9340 +/- 0.0051 |
| Macro F1 (9 classes) | 0.8051 | 0.8372 | 0.8403 | 0.8275 +/- 0.0195 |
| Edit | 88.0467 | 89.8635 | 89.2990 | 89.0697 +/- 0.9298 |
| Segmental F1@10 | 93.4814 | 94.9486 | 94.4143 | 94.2815 +/- 0.7426 |
| Segmental F1@25 | 92.9673 | 94.6762 | 94.1851 | 93.9428 +/- 0.8798 |
| Segmental F1@50 | 90.2529 | 91.7208 | 91.4131 | 91.1289 +/- 0.7741 |
| Error rate | 0.0718 | 0.0642 | 0.0621 | 0.0660 +/- 0.0051 |
| Random AUPRC | 0.0718 | 0.0642 | 0.0621 | 0.0660 +/- 0.0051 |
| Error AUROC (raw MI) | 0.9043 | 0.8971 | 0.8986 | 0.9000 +/- 0.0038 |
| Error AUPRC/AP (raw MI) | 0.3233 | 0.2946 | 0.2962 | 0.3047 +/- 0.0161 |
| NLL (raw soft target) | 0.1963 | 0.1896 | 0.1829 | 0.1896 +/- 0.0067 |
| NLL (calibrated soft target) | 0.1960 | 0.1884 | 0.1757 | 0.1867 +/- 0.0103 |
| Brier (raw soft target) | 0.1069 | 0.0988 | 0.0955 | 0.1004 +/- 0.0059 |
| Brier (calibrated soft target) | 0.1069 | 0.0985 | 0.0941 | 0.0998 +/- 0.0065 |
| ECE (raw soft target) | 0.0273 | 0.0277 | 0.0247 | 0.0265 +/- 0.0016 |
| ECE (calibrated soft target) | 0.0266 | 0.0270 | 0.0190 | 0.0242 +/- 0.0045 |

All requested metrics are available. NLL/Brier/ECE above use the exact stored `soft_label` arrays in `raw_mc_soft` and `weighted_calibrated_mc_soft`, which are the student soft targets. The calibrated target's training loss weight is not applied to evaluation metrics. PointCam normalizes the float32 assigned vectors again when materializing `soft_label`; small rounding differences are recorded in JSON. Metrics of the assigned `raw_mc` and `calibrated_mc` vectors are also saved in `assigned_probability_metrics.csv`. Hard targets are evaluated as class decisions; no one-hot confidence is invented.

## Ground truth and temporal protocol

- Radar windows contain 40 retained frames, stride 8, center offset 20. The source builder requires adjacent timestamp gaps <=0.5 s and total window span <=2.5 s.
- Radar ground truth is the manual camera annotation interval active at the radar-center timestamp: camera annotation boundary frames map through `timestamps.csv` to half-open time intervals. The saved radar PKL label is authoritative for this evaluation. The associated skeleton window instead has 60 detections and center offset 30. The saved `source_frame_center` in a pseudo target belongs to that skeleton window.
- Each saved teacher association is verified against PointCam's original nearest-center lookup within the same subject and recording, including its original CSV-row tie-break rule. The existing saved association is used for evaluation.
- E1 top-1/nine-class macro F1 and current official MS-TCN Edit/F1 functions are reused. Sequences are grouped by radar recording and sorted by numeric radar window start/center. Segment IoU uses ordinal positions of retained radar windows. Rejected windows are absent; there is no interpolation, duration weighting, or new split at a removed gap. Thus these are retained-sequence metrics, not full original recording segmentation metrics.
- Edit is averaged across recordings; F1 pools TP/FP/FN across recordings. No background class is excluded. Compare the matching convention with E1 `reports/ms_tcn_official`.
- Error ranking reuses E2B raw MI, with pseudo-label error against radar GT as positive. AP is threshold-grouped average precision, and random AUPRC is error prevalence.
- E3 NLL uses natural log with 1e-12 clipping; Brier sums squared error over nine classes without dividing by nine; ECE is 15-bin equal-width top-label calibration.

## Interpretation limits

This measures assigned teacher-target quality on retained radar training windows. It does not measure a trained radar student's performance. Radar and skeleton centers can have different manual labels at their respective center times; their observed disagreement counts are reported above, so skeleton-center GT is not substituted. Multiple radar windows may share one skeleton target. Folds overlap, so their mean/SD is descriptive, not an independent-sample confidence interval. Differences from the preceding skeleton report combine modality/window/GT changes with filtering and cannot isolate the filter's effect.

Across folds: 81323 radar row instances, 36644 unique radar frame IDs.

## Reproduction

Run from the pyskl repository:

```powershell
python rerun/pseudo_labeling/evaluate_r2_radar_pseudo_labels.py
```

`joined_radar_audit.csv` explicitly separates radar GT from associated skeleton GT. Other outputs include fold/mean-SD, per-recording, per-class and state/transition ranking CSVs. JSON stores input hashes, identity checks and probability-rounding residuals. PointCam data and source files are read only.

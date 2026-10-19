# R3 retained radar-window pseudo-label quality

The evaluated objects are retained **radar training windows**, not skeleton windows. Every ground-truth label is read from PointCam `data/rerun/r3/continuous_window/fold_<fold>.pkl` and joined by exact radar `frame_id`. No teacher inference, reassignment or filtering changes were made.

## Retention and identity checks

All six saved pseudo-target variants have identical ordered retained radar IDs, manual labels, hard labels, raw/calibrated probabilities and skeleton associations. Weighted variants change loss weights, not sample selection. Assignment CSV IDs, labels, associated skeleton IDs and probability vectors match their target PKLs.

| Fold | Original radar train | Recording drops | Gap drops | Retained radar | Distinct associated skeleton | Radar/skeleton GT differences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 26609 | 86 | 3604 | 22919 | 22869 | 894 |
| B | 27190 | 86 | 3187 | 23917 | 23863 | 869 |
| C | 27696 | 0 | 3481 | 24215 | 24160 | 832 |

The saved selection audit exactly reproduces the retained IDs using `recording_id != 38-mia-fall2` and `abs(teacher_center_timestamp_sec - radar_center_timestamp_sec) <= 0.2`. These are exclusions from radar training candidates, not subtraction from the original skeleton-window count. There are zero pseudo-target `manual_label` mismatches against the radar ground-truth PKLs.

## Radar-center versus skeleton-center ground truth

Each radar window is paired with a skeleton window whose center can occur at a different time. Both reference labels come from human activity annotations, evaluated at their respective centers:

- **Radar-center ground truth:** the manual activity label at the radar window's center timestamp. This is the reference used for every quality metric in this report.
- **Skeleton-center ground truth:** the manual activity label at the associated skeleton window's center. This is recorded only for the association audit, not substituted for radar ground truth.

Joining by exact radar `frame_id` retrieves the manual label of the same radar window receiving the pseudo target. `skeleton_sample_id` separately identifies the teacher window. All radar joins succeeded; the label disagreements below are not failed joins.

For example (illustrative), an action boundary may lie between the two centers: the radar center is still in walk while the skeleton center is already in fall. Their manual labels then differ even though the association selected the correct nearest teacher window.

**2,595 of 71,051 fold-row instances (3.65%) have different radar-center and associated skeleton-center ground-truth labels.** This counts `radar_manual_label != source_skeleton_manual_label`; it does not count pseudo-label mistakes. The separate pseudo-label error count is **4,642**, computed as `hard_pseudo_label != radar_manual_label`. A pseudo label can match either reference or neither.

These counts include repeated windows across folds. Evaluating with skeleton-center ground truth would measure teacher quality at the teacher's time and could overlook errors from transferring the target to the radar window. The student-target comparison therefore uses radar-center ground truth.

## Metrics

Equal-fold mean and sample SD (ddof=1). Edit and segmental F1 use a 0-100 scale. All metrics are unweighted over retained windows, retaining E1/E3 comparability; sample loss weights and stochastic training sampler repetitions are not evaluation weights.

| Metric | A | B | C | Mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| Top-1 accuracy | 0.9296 | 0.9365 | 0.9377 | 0.9346 +/- 0.0044 |
| Macro F1 (9 classes) | 0.8158 | 0.8457 | 0.8476 | 0.8364 +/- 0.0178 |
| Edit | 88.1019 | 89.9075 | 89.3334 | 89.1143 +/- 0.9225 |
| Segmental F1@10 | 93.5204 | 94.9879 | 94.3672 | 94.2919 +/- 0.7366 |
| Segmental F1@25 | 93.3966 | 94.9669 | 94.3045 | 94.2227 +/- 0.7883 |
| Segmental F1@50 | 91.2299 | 92.3190 | 91.9845 | 91.8445 +/- 0.5579 |
| Error rate | 0.0704 | 0.0635 | 0.0623 | 0.0654 +/- 0.0044 |
| Random AUPRC | 0.0704 | 0.0635 | 0.0623 | 0.0654 +/- 0.0044 |
| Error AUROC (raw MI) | 0.9061 | 0.9001 | 0.8965 | 0.9009 +/- 0.0048 |
| Error AUPRC/AP (raw MI) | 0.3198 | 0.2956 | 0.2894 | 0.3016 +/- 0.0160 |
| NLL (raw soft target) | 0.1845 | 0.1801 | 0.1731 | 0.1792 +/- 0.0058 |
| NLL (calibrated soft target) | 0.1846 | 0.1790 | 0.1678 | 0.1771 +/- 0.0085 |
| Brier (raw soft target) | 0.1034 | 0.0970 | 0.0940 | 0.0981 +/- 0.0048 |
| Brier (calibrated soft target) | 0.1034 | 0.0968 | 0.0926 | 0.0976 +/- 0.0054 |
| ECE (raw soft target) | 0.0239 | 0.0256 | 0.0225 | 0.0240 +/- 0.0016 |
| ECE (calibrated soft target) | 0.0232 | 0.0249 | 0.0165 | 0.0215 +/- 0.0044 |

All requested metrics are available. NLL/Brier/ECE above use the exact stored `soft_label` arrays in `raw_mc_soft` and `calibrated_mc_soft`, which are the student soft targets. The calibrated target's training loss weight is not applied to evaluation metrics. PointCam normalizes the float32 assigned vectors again when materializing `soft_label`; small rounding differences are recorded in JSON. Metrics of the assigned `raw_mc` and `calibrated_mc` vectors are also saved in `assigned_probability_metrics.csv`. Hard targets are evaluated as class decisions; no one-hot confidence is invented.

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

Across folds: 71051 radar row instances, 32276 unique radar frame IDs.

## Reproduction

Run from the pyskl repository:

```powershell
python rerun/pseudo_labeling/evaluate_r3_radar_pseudo_labels.py
```

`joined_radar_audit.csv` explicitly separates radar GT from associated skeleton GT. Other outputs include fold/mean-SD, per-recording, per-class and state/transition ranking CSVs. JSON stores input hashes, identity checks and probability-rounding residuals. PointCam data and source files are read only.

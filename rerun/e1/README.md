# E1 Skeleton Training-Inference Alignment

This rerun experiment compares activity-aligned skeleton training with direct
continuous-window skeleton training under matched continuous-window inference.

Implemented conditions:

- A1: train on activity-aligned samples, validate on activity-aligned samples,
  evaluate on activity-aligned test samples.
- A2: use the A1 activity-aligned checkpoint and evaluate on continuous-window
  test samples using the B continuous-window config.
- B: train on continuous-window samples, validate on continuous-window samples,
  evaluate on continuous-window test samples.
- C: train on continuous-window samples with triangular temporal-composition
  soft targets, validate and evaluate on continuous-window samples with hard
  center labels.

A2 does not have a separate config file. Its test dataset, preprocessing, and
metrics are the same as B; the only difference is the checkpoint being loaded.

## Generate Configs

```powershell
python rerun/e1/generate_stgcnpp_configs.py --overwrite
```

Generated config root:

```text
configs/stgcn++/stgcn++_radarv4/rerun/e1
```

Generated jobs:

```text
rerun/e1/slurm/run_a1_a2_fold_<fold>_<stream>.sh
rerun/e1/slurm/run_b_fold_<fold>_<stream>.sh
rerun/e1/slurm/run_c_fold_<fold>_<stream>.sh
```

There are 18 generated jobs total: A1+A2, B, and C for each fold
(`a`, `b`, `c`) and stream (`joint`, `bone`). A1+A2 jobs request 4 hours.
B and C jobs request 6 hours.

## Training Protocol

Each A1 config uses:

- ST-GCN++ with `num_person=1`
- `GCNHead` with `dropout=0.5`
- `num_classes=9`
- `CrossEntropyLoss` without class weights
- `videos_per_gpu=16`
- `workers_per_gpu=2`
- `lr=0.05`
- `GPUS=4` in the generated job scripts
- `total_epochs=20`
- validation checkpoint selection by `macro_f1`
- metrics `macro_f1` and top-1 accuracy only

A1, B, and C use the same model/runtime settings. A1 uses the
activity-aligned pkl. B uses
`data/radar_v4/rerun/yolo26xpose/pyskl/continuous_window_w60_s12`. C uses
`data/radar_v4/rerun/yolo26xpose/pyskl/continuous_window_w60_s12_triangular`.

The training pipeline is:

```python
Flip -> PreNormalize2D(mode='auto') -> GenSkeFeat -> MonotonicUniformResample(60)
-> PoseDecode -> FormatGCNInput(num_person=1)
```

Validation and test omit `Flip` and use the same deterministic monotonic
resampling.

For C training only, the pipeline inserts:

```python
dict(type='UseSoftLabel', source_key='label_soft_triangular', num_classes=9)
```

This replaces the training batch `label` with the triangular soft target before
the loss. The underlying dataset item still keeps the hard center `label`, so
the square-root sampler is unchanged. C validation and test do not use this
transform and therefore evaluate against hard center labels, like B.

## Square-Root Sampler

Training uses `class_sample_strategy='sqrt'` with `epoch_size=N_train`.
The sampler assigns each sample weight `1 / sqrt(n_yi)` and samples with
replacement. Validation and test datasets do not use the sampler.

For reproducibility, run the jobs with the same seed for joint and bone streams.
The generated jobs default to `SEED=42`. A1+A2 jobs train A1 and then evaluate
the selected A1 checkpoint on continuous windows; B jobs train directly on
continuous windows. Per-epoch sampled index sequences are written under each
training work directory:

```text
work_dirs/rerun/e1/fold_<fold>/<stream>/a1_activity_aligned/sampler_indices
work_dirs/rerun/e1/fold_<fold>/<stream>/b_continuous_window/sampler_indices
work_dirs/rerun/e1/fold_<fold>/<stream>/c_triangular_temporal_composition/sampler_indices
```

With 4-GPU distributed training, PYSKL pads the rank slices when `N_train` is not
divisible by 4. The saved `sampled_indices` field is the requested natural
`N_train` sequence; `ddp_padded_indices` records the padded sequence actually
split across ranks.

## Result Summary

After the three folds finish for both streams, generate the joint/bone/fusion
test-subject summary with:

```powershell
python rerun/e1/summarize_results.py
```

The script reads available `best_pred.pkl` and `best_eval.json` files from
`work_dirs/rerun/e1`, verifies the saved single-stream metrics, fuses joint and
bone as `0.5 * (joint_probability + bone_probability)`, and writes:

```text
rerun/e1/reports/e1_fold_metrics.csv
rerun/e1/reports/e1_mean_sd.csv
rerun/e1/reports/e1_summary.json
rerun/e1/reports/e1_summary.md
rerun/e1/reports/e1_continuous_segmental_fold_metrics.csv
rerun/e1/reports/e1_continuous_segmental_recording_metrics.csv
rerun/e1/reports/e1_continuous_segmental_mean_sd.csv
rerun/e1/reports/e1_continuous_segmental_summary.json
rerun/e1/reports/e1_continuous_segmental_summary.md
```

Fusion predictions and metrics are written under
`work_dirs/rerun/e1/fold_<fold>/fusion/<condition>/`.

## Validation-Subject Reporting

For method-selection reporting, evaluate each selected checkpoint on the
corresponding fold validation subject, then calculate joint, bone, and fusion
metrics.

Generate validation inference configs and jobs with:

```powershell
python rerun/e1/generate_validation_eval_scripts.py --overwrite
```

Generated validation configs:

```text
configs/stgcn++/stgcn++_radarv4/rerun/e1/fold_<fold>/<stream>/validation/<condition>_validation.py
```

Generated validation jobs:

```text
rerun/e1/slurm/run_validation_fold_<fold>_<stream>.sh
```

There are 6 validation jobs total, one per fold and stream. Each job evaluates
A1, A2, B, and C for that fold/stream. The checkpoint rule is:

- A1: use the stream checkpoint saved as `best_macro_f1_epoch_*.pth` under
  `a1_activity_aligned`.
- A2: use the same A1 checkpoint, but evaluate on the continuous-window
  validation split.
- B: use the stream checkpoint saved as `best_macro_f1_epoch_*.pth` under
  `b_continuous_window`.
- C: use the stream checkpoint saved as `best_macro_f1_epoch_*.pth` under
  `c_triangular_temporal_composition`.

Submit all validation inference jobs from the repository root with:

```bash
bash rerun/e1/slurm/submit_validation_jobs.sh
```

After all validation jobs finish, summarize validation metrics with:

```powershell
python rerun/e1/summarize_validation_results.py
```

The validation summarizer writes:

```text
rerun/e1/reports/e1_validation_fold_metrics.csv
rerun/e1/reports/e1_validation_mean_sd.csv
rerun/e1/reports/e1_validation_summary.json
rerun/e1/reports/e1_validation_summary.md
```

Validation fusion predictions and metrics are written under:

```text
work_dirs/rerun/e1/fold_<fold>/fusion/<condition>/validation/
```

## Continuous Segmental Metrics

The main summary script also evaluates available continuous-window conditions
A2, B, and C with MS-TCN-style segmental metrics:

- normalized Edit score
- segmental F1@10
- segmental F1@25
- segmental F1@50

This does not require rerunning inference. The saved `best_pred.pkl` files
provide the per-window class probabilities, and the continuous-window pkl
provides the matching test order, `session_name`, `window_row_start`, and
`center_source_frame` metadata.

Protocol details:

- Treat every `session_name` as an independent temporal sequence.
- Never concatenate windows from different recordings.
- Sort windows within a recording by `window_row_start`, then
  `center_source_frame`.
- Collapse consecutive identical ground-truth labels into ground-truth
  segments.
- Collapse consecutive identical predicted labels into predicted segments.
- Ignore background labels if explicitly provided; the current 9-class E1 label
  set has no background class by default.
- Compute segment IoU using ordered window-center sequence indices by default,
  matching the official sequence-label evaluation convention.
- Match predicted and ground-truth segments only within the same recording and
  only when the class labels match.
- For each threshold, sum TP, FP, and FN counts across recordings in the fold,
  then compute one fold-level F1 score.
- Compute normalized Levenshtein Edit per recording after collapsing labels,
  then average recording Edit scores within the fold.

Standalone segmental evaluation can be run with:

```powershell
python rerun/e1/evaluate_continuous_segmental.py
```

Optional flags:

```powershell
python rerun/e1/evaluate_continuous_segmental.py --background-labels <label-or-id>
python rerun/e1/evaluate_continuous_segmental.py --overlap-axis center_source_frame
```

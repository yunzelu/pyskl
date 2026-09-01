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
```

There are 12 generated jobs total: A1+A2 and B for each fold
(`a`, `b`, `c`) and stream (`joint`, `bone`). A1+A2 jobs request 4 hours.
B jobs request 6 hours.

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

A1 and B use the same model/runtime settings. A1 uses the activity-aligned pkl;
B uses `data/radar_v4/rerun/yolo26xpose/pyskl/continuous_window_w60_s12`.

The training pipeline is:

```python
Flip -> PreNormalize2D(mode='auto') -> GenSkeFeat -> MonotonicUniformResample(60)
-> PoseDecode -> FormatGCNInput(num_person=1)
```

Validation and test omit `Flip` and use the same deterministic monotonic
resampling.

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
```

With 4-GPU distributed training, PYSKL pads the rank slices when `N_train` is not
divisible by 4. The saved `sampled_indices` field is the requested natural
`N_train` sequence; `ddp_padded_indices` records the padded sequence actually
split across ranks.

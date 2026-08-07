# S6: Out-of-Fold Skeleton Teachers for Pseudo Labeling

S6 rebuilds the old `8111teacher4` teacher setup with the current zero-frame
filtering policy.

The split mirrors E5:

- Each outer 8/1/1/1 fold has four teachers.
- Each teacher trains on six of the outer-fold training subjects.
- The remaining two outer-fold training subjects are stored as the pkl `test`
  split for later pseudo-label inference.
- The outer validation and calibration subjects stay unchanged.
- The outer test subject is unused by these teacher-training jobs.

The trimmed pkl builder now removes all-zero pose frames from each sample and
drops samples with fewer than `30` retained frames for `clip_len=60`.

Prepare trimmed teacher pkls:

```bash
bash thesis/s6/build_trimmed_pkls.sh
python thesis/s6/generate_trimmed_configs.py --overwrite
```

Train trimmed teachers. These three jobs can be submitted in parallel:

```bash
sbatch thesis/s6/train_trimmed_fold_a.sh
sbatch thesis/s6/train_trimmed_fold_b.sh
sbatch thesis/s6/train_trimmed_fold_c.sh
```

Prepare the continuous teacher split pkl from the already-built S2 continuous
windows. Run this once, not inside parallel fold jobs:

```bash
bash thesis/s6/build_continuous_teacher_splits.sh
```

Fine-tune the continuous teachers after trimmed checkpoints exist. These jobs
can also be submitted in parallel:

```bash
sbatch thesis/s6/train_continuous_fold_a.sh
sbatch thesis/s6/train_continuous_fold_b.sh
sbatch thesis/s6/train_continuous_fold_c.sh
```

Useful restricted runs:

```bash
RUN_TEACHERS="t1" RUN_STREAMS="joint limb" sbatch thesis/s6/train_trimmed_fold_a.sh
RUN_TEACHERS="t3 t4" RUN_STREAMS="joint" sbatch thesis/s6/train_continuous_fold_b.sh
```

Outputs:

- Trimmed pkls: `data/radar_v4/pyskl/8111teacher4_s6/`
- Trimmed configs: `thesis/s6/configs/trimmed/`
- Trimmed checkpoints: `work_dirs/thesis/s6/trimmed/`
- Continuous teacher pkl: `data/radar_v4/pyskl/s2/radarv4_yolo26xpose_clip60_s2_teacher4_s6_continuous.pkl`
- Continuous configs: `thesis/s6/configs/continuous/`
- Continuous checkpoints: `work_dirs/thesis/s6/continuous_hard/`

## Pseudo Label Export

After continuous fine-tuning, each teacher labels only its own pseudo split. For
example, `fold_a/t1` labels `fold_a_t1_pseudo`. Temperature is fitted from the
same teacher's calibration split, for example `fold_a_t1_calib`, and is reused
for deterministic calibrated and MC calibrated outputs.

Recommended order:

```bash
sbatch thesis/s6/pseudo_fit_temperature.sh
```

After the temperature job has finished, export the pseudo-label variants:

```bash
sbatch thesis/s6/pseudo_hard_labels.sh
sbatch thesis/s6/pseudo_raw_soft.sh
sbatch thesis/s6/pseudo_calibrated_soft.sh
sbatch thesis/s6/pseudo_mc_calibrated_soft.sh
```

Useful restricted run:

```bash
RUN_FOLDS="a" RUN_TEACHERS="t1" RUN_STREAMS="joint limb" sbatch thesis/s6/pseudo_mc_calibrated_soft.sh
```

Pseudo-label outputs are written under:

```text
work_dirs/thesis/s6/pseudo_labels/fold_<fold>/<teacher>/<stream>/
```

Main files per teacher/stream:

- `temperature.json`
- `deterministic_hard_pseudo_labels.csv`
- `raw_soft_probabilities.csv`
- `calibrated_soft_probabilities.csv`
- `mc_calibrated_soft_probabilities.csv`
- matching `.npz` files with arrays and metadata

The MC file stores calibrated per-pass probabilities with shape `[N, K, C]`,
the calibrated predictive mean, hard pseudo labels from the predictive mean,
and uncertainty quantities. Set `SAVE_MC_LOGITS=1` for the MC job if raw MC
logits also need to be stored.

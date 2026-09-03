# Pseudo-Labeling Inner Teachers

This folder owns the inner cross-fitting stage used to generate out-of-fold
pseudo-labels for the outer-training subjects. It is separate from E2
uncertainty evaluation.

## Protocol

The inner teachers reuse the frozen E1-B continuous-window training protocol:

- ST-GCN++
- joint and bone streams
- equal probability fusion after stream inference
- direct continuous-window training
- hard center-label target
- 60 retained skeleton detections
- stride 12
- center offset 30
- maximum adjacent gap 0.5 seconds
- maximum window span 2.5 seconds
- `GCNHead(dropout=0.5)`
- online square-root sampling on training only
- hard-label `CrossEntropyLoss`
- checkpoint selection by validation center macro-F1

For each outer fold, the eight outer-training subjects are split into four
disjoint pseudo-target pairs. Each inner teacher trains on the remaining six
outer-training subjects and predicts the held-out pair. The fixed outer
validation subject is used for checkpoint selection, the fixed outer
calibration subject is preserved as `calib` for later temperature fitting, and
the outer test subject is excluded entirely.

Each teacher system has one joint checkpoint and one bone checkpoint:

```text
3 outer folds x 4 inner teachers x 2 streams = 24 trained checkpoints
```

## Generate Artifacts

Generate split pkls, configs, jobs, and count reports:

```bash
python rerun/pseudo_labeling/generate_inner_teacher_training_artifacts.py --overwrite
```

The local workspace may not have enough free disk for the 24 full pkls. To
generate only configs, jobs, and count reports locally:

```bash
python rerun/pseudo_labeling/generate_inner_teacher_training_artifacts.py --overwrite --skip-pkl-write
```

On the cluster, use the preparation job to write the large pkls:

```bash
sbatch rerun/pseudo_labeling/slurm/prepare_inner_teacher_artifacts.sh
```

## Generated Paths

Inner pkl directory:

```text
data/radar_v4/rerun/yolo26xpose/pyskl/inner_teachers_continuous_window_w60_s12/
```

Config directory:

```text
configs/stgcn++/stgcn++_radarv4/rerun/pseudo_labeling/inner_teachers/fold_<fold>/t<id>/<stream>.py
```

SLURM jobs:

```text
rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_<fold>_t<id>_<stream>.sh
rerun/pseudo_labeling/slurm/inner_teachers/submit_inner_teacher_jobs.sh
```

Reports:

```text
rerun/pseudo_labeling/reports/inner_teacher_split_counts.csv
rerun/pseudo_labeling/generated_inner_teacher_artifacts.json
```

## Train Inner Teachers

After the pkl preparation job finishes, submit all 24 stream jobs:

```bash
bash rerun/pseudo_labeling/slurm/inner_teachers/submit_inner_teacher_jobs.sh
```

Each config sets:

- `epoch_size = N_inner_train`
- `data.val` to the fixed outer validation subject
- `data.test` to split `pseudo_target` for `--test-best`
- `work_dir = work_dirs/rerun/pseudo_labeling/inner_teachers/fold_<fold>/t<id>/<stream>`

## Verified Counts

The generator validates these counts against the current continuous-window pkl:

```text
A T1 train 32482 pseudo 10811
A T2 train 32105 pseudo 11188
A T3 train 33190 pseudo 10103
A T4 train 32102 pseudo 11191

B T1 train 32725 pseudo 11017
B T2 train 33010 pseudo 10732
B T3 train 32684 pseudo 11058
B T4 train 32807 pseudo 10935

C T1 train 32524 pseudo 10811
C T2 train 32774 pseudo 10561
C T3 train 32307 pseudo 11028
C T4 train 32400 pseudo 10935
```

## Checkpoint Ties

No custom pseudo-labeling tie-breaker is added. The configs preserve E1-B
`save_best='macro_f1'` and `rule='greater'`. Standard MMCV best-checkpoint
logic treats a strictly greater score as a new best, so an exact tie does not
intentionally replace the current best. If multiple best checkpoint files
remain, PYSKL's post-training `--test-best` fallback selects the best file with
the largest epoch id.

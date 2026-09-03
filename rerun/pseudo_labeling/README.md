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

## Generate OOF Pseudo Labels

After the 24 inner-teacher stream checkpoints exist under:

```text
work_dirs/rerun/pseudo_labeling/inner_teachers/fold_<fold>/t<id>/<stream>/
```

generate one Joint+Bone teacher system at a time:

```bash
python rerun/pseudo_labeling/run_inner_teacher_oof_pseudo_labeling.py \
  --fold a \
  --teacher t1 \
  --device cuda \
  --num-passes 30
```

Each teacher job performs:

- 30-pass MC dropout on the fixed outer calibration subject
- equal probability fusion inside each pass
- post-fusion pool-temperature fitting on the calibration subject
- calibration-set raw MC MI 95th percentile estimation
- 30-pass MC dropout on the teacher's two pseudo-target subjects
- training-safe pseudo table, audit table, and compressed MC archives

Default output root:

```text
data/radar_v4/rerun/yolo26xpose/pseudo_labels_v1/
```

Per-teacher output:

```text
fold_<fold>/t<id>/
├── teacher_metadata.json
├── calibration_predictions.csv
├── calibration_mc_fused_samples.npz
├── pseudo_predictions.csv
├── pseudo_predictions_audit.csv
└── mc_fused_samples.npz
```

`pseudo_predictions.csv` is radar-training-safe and does not contain manual
activity labels. `pseudo_predictions_audit.csv` contains the same rows plus
manual center label, pseudo-label correctness, and distance to the nearest
manual annotation boundary in original RGB frame units.

The full MC archive stores:

```text
sample_ids
fused_mc_probabilities    # [30, N, 9]
source_frame_indices      # [N, 60]
timestamps_sec            # [N, 60]
```

The main table stores only compact start/center/end frame fields; the NPZ keeps
the complete 60-frame source mapping for gap and alignment audits.

Index convention:

- `window_start_retained_idx`, `center_retained_idx`, and
  `window_end_retained_idx_exclusive` are indices in the filtered retained
  skeleton sequence.
- `window_candidate_index = window_start_retained_idx / 12`, the original
  sliding-window candidate index within that retained sequence.
- `accepted_window_index_in_recording` is parsed from the generated
  `win000000` sample token and counts only accepted windows.
- source time uses the original RGB frame/timestamp fields, never
  `center_retained_idx / 30`.

### H100 Jobs

Generate H100 SLURM scripts:

```bash
python rerun/pseudo_labeling/generate_oof_pseudo_label_slurm.py --overwrite
```

This writes:

```text
rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_<fold>_t<id>_h100.sh
rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_aggregate_fold_<fold>_h100.sh
rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/submit_oof_teacher_jobs_h100.sh
rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/submit_oof_aggregation_jobs_h100.sh
rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/submit_oof_all_with_dependencies_h100.sh
```

Submit the full stage with fold-level aggregation dependencies:

```bash
bash rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/submit_oof_all_with_dependencies_h100.sh
```

The teacher jobs use the H100 environment:

```bash
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
module purge
module load StdEnv/2023
module load python/3.10
module load opencv/4.8.1
source "/project/def-mbolic/yunzelu/pyskl/.venv/bin/activate"
cd "/project/def-mbolic/yunzelu/pyskl"
```

Aggregation is CPU-only and should run after the four teacher jobs for the
corresponding fold.

### Step 6 Aggregation

Fold aggregation is:

```bash
python rerun/pseudo_labeling/aggregate_oof_pseudo_labels.py --folds a b c
```

It writes:

```text
fold_<fold>/oof_skeleton_pseudo_labels.csv
fold_<fold>/oof_skeleton_pseudo_labels_audit.csv
fold_<fold>/radar_teacher_alignment.csv
fold_<fold>/fold_metadata.json
```

Validation checks:

- every outer-training subject appears exactly once in the fold
- every skeleton sample ID appears exactly once
- every row was predicted by a teacher that excluded that subject
- validation, calibration, and outer test subjects are absent
- the radar-training-safe files contain no manual-label fields

The row-level tables are CSV files because the H100 cluster environment does
not provide a Parquet backend. Existing stream MC arrays are reused when
present, so rerunning a failed teacher job can continue to CSV/NPZ table
writing without recomputing completed stream outputs.

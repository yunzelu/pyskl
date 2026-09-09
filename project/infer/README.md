# Project inference from pose CSVs

Use `infer_csv.py` directly on pose CSVs and their multiperson masks. A
separate PKL is unnecessary: the reader keeps a rolling window and passes
up to 10,000 valid windows at a time to the four models. This avoids storing
overlapping poses repeatedly and preserves continuity across chunks.

## Selected inputs

For `auto_labeling_pipeline/outputs/0903`, use:

- `pose_2026-06-23_ds_cf.csv`
- `pose_2026-07-20_ds_cf.csv`

Both have approximately 10 fps timing. April 18 has approximately 30 fps
timing and is excluded from the PowerShell launcher. Here `_ds` means duplicate
suppression, not downsampling. The generic CLI processes all selected rows
without changing their sampling rate; pass only the intended 10 fps sources.

## Local commands

From the repository root, the supplied PowerShell launcher selects June and
July explicitly:

```powershell
# Audit both entire input files, without loading models.
./project/infer/run_0903_10fps.ps1 -ValidateOnly

# Run complete inference on both files.
./project/infer/run_0903_10fps.ps1

# A small run in a separate output directory for checking the setup.
./project/infer/run_0903_10fps.ps1 -MaxWindows 12 -ChunkSize 5 -BatchSize 4 `
  -OutputDir data/project/inference/smoke_0903_10fps
```

The launcher defaults to the local `har` Python, CPU, and two Torch threads.
It temporarily adds `.cache/project_infer_mmcv1` to `PYTHONPATH` if that
isolated compatibility cache exists, then restores the previous environment.
The cache contains MMCV 1.7.2 and its missing helper dependencies; the installed
`har` packages were not changed. Override `-Python`, `-Device`, `-BatchSize`,
`-ChunkSize`, `-NumThreads`, `-InputRoot`, or `-OutputDir` as needed. Use
`-Overwrite` to replace matching outputs from an earlier run.

In a compatible PYSKL training environment, the Python CLI can run directly:

```bash
python project/infer/infer_csv.py /path/to/outputs/0903 \
  --exclude '*2026-04-18*' \
  --device cuda:0 --batch-size 128 --chunk-size 10000 \
  --output-dir data/project/inference/0903_10fps
```

It also accepts one or more individual CSV paths. `--validate-only` needs
only Python and NumPy. Model inference requires the repository's compatible
PyTorch/MMCV 1.x runtime; MMCV 2.x does not provide the required APIs.

`--max-windows N` stops after N valid windows per CSV. Its summaries are
explicitly marked `limited` / `scan_complete=false`; they do not certify
unread portions of the file. Use a separate output directory for such runs.

## H100 SLURM job

Submit from the repository root on the cluster:

```bash
sbatch project/slurm/run_inference_0903_10fps_h100.sh
```

The job requests `--gpus=h100_1g.10gb:1`, 8 CPUs, 32 GB RAM, and
`--time=00:30:00`. It processes **every `*_cf.csv` in `data/project/csv/`**
with its paired `*_mm.json`, and writes results under
`work_dirs/project/inference/0903_10fps/`. There is no filename/date exclusion
in this job; place the intended 10 fps inputs in that directory.

All four streams run on `cuda:0` in one Python process, using the existing
H100 modules (`StdEnv/2023`, Python 3.10, OpenCV 4.8.1) and repository `.venv`.
The default remote root is `/project/def-mbolic/yunzelu/pyskl`, overridable
with `PROJECT_REPO_ROOT`. `BATCH_SIZE`, `CHUNK_SIZE`, and `NUM_THREADS` default
to 128, 10,000, and 4. Extra script arguments are passed to the inference CLI:

```bash
sbatch project/slurm/run_inference_0903_10fps_h100.sh --overwrite
```

## CSV and mask contract

- Each CSV needs `Timestamp`, `ID`, and `KP0_X`, `KP0_Y`, `KP0_C` through
  `KP16_X`, `KP16_Y`, `KP16_C`. Coordinates are COCO-17 pixel coordinates;
  confidence values must be finite and between 0 and 1. Bounding boxes are
  not used by the classifier.
- The companion for `prefix_cf.csv` is `prefix_mm.json` in the same directory.
  Its format is a list of `start_unix_time` / `end_unix_time` intervals.
  Both endpoints are included; overlapping intervals are merged.
- Timestamp comparisons use the exact decimal Unix values, without rounding
  or tolerance. Numerically equal values such as `1`, `1.0`, and `1.00`
  belong to the same frame. CSV timestamps must be sorted.
- All rows within masked intervals are skipped. Every remaining timestamp
  must contain exactly one detection, even if duplicate rows share the same
  tracking ID. An unmasked duplicate raises an error with its timestamp and
  CSV row number. The code never chooses a person by confidence or ID.
- Tracking IDs do not split sequences. All retained detections form one
  continuous sequence with one person slot. Output `ID` is always 0; the
  original center detection's ID is preserved as `source_id` for tracing.

## Windows and fusion

The confirmed project settings match training:

- 20 retained frames per window, stride 4, center offset 10 (the 11th frame).
- Maximum adjacent timestamp gap <= 0.5 seconds.
- First-to-last timestamp span <= 2.5 seconds.
- Drop incomplete tails and invalid windows. No interpolation, zero filling,
  extra downsampling, or manual center-label filtering is applied.

The window grid starts at retained index 0 for each CSV and continues across
chunks and tracking-ID changes. Masked rows are removed before windowing;
remaining frames on either side may share a window only if both timing
checks pass. Source timestamps are preserved across those gaps. A chunk
boundary does not reset the stride or lose overlapping windows.

The four streams use their saved `test_pipeline` and the unique
`best_macro_f1*.pth` checkpoint under
`work_dirs/project/stgcnpp/fps10_phase0/<stream>/`. The model adapter prefers
each saved training config, with a fallback to
`project/configs/stgcnpp/fps10_phase0/`. Model/config mismatches or ambiguous
checkpoints fail explicitly. Current selected checkpoints are:

| Stream | Feature | Best checkpoint epoch |
| --- | --- | --- |
| Joint | `j` | 2 |
| Bone | `b` | 20 |
| Joint motion | `jm` | 2 |
| Bone motion | `bm` | 9 |

Each stream runs in evaluation mode and returns all nine probabilities.
Fusion is exactly:

```text
p_fused = (2 * p_j + 2 * p_b + p_jm + p_bm) / 6
label = argmax(p_fused)
```

The recognizer applies softmax once. The fusion step averages probabilities;
it does not apply another softmax, temperature scaling, or voting.
Normalization uses the saved `PreNormalize2D(mode='auto')` transform.
The 1280x720 image metadata matches the input pipeline's documented coordinate
system; automatic normalization uses each window's valid keypoint bounds.

`--chunk-size` (default 10,000) limits the number of valid windows held for one
inference chunk. `--batch-size` (default 128) independently controls each model
minibatch. Four models are loaded once and reused for all chunks/files.

## Outputs

Each source produces:

- `*_cf__center_predictions.csv`: one row per valid window center, with
  original center timestamp, original CSV row, original source ID, retained
  indices, source frame indices, fused label/confidence, and all nine fused
  probabilities plus the nine probabilities from each stream.
- `*_cf__inference_summary.json`: row/mask/window counts, class counts,
  exact checkpoint/config paths, fusion rule, and runtime settings.

Validation-only mode writes `*_cf__validation_summary.json`. A corresponding
`run_inference_summary.json` or `run_validation_summary.json` summarizes the
selected input files. The standard local output is
`data/project/inference/0903_10fps/`.

`source_frame_*` counts unique timestamp groups in the input CSV from zero.
`*_retained_idx` counts rows after masking from zero. `source_csv_row_*` uses
the original physical CSV line number, including the header as line 1.
Timestamp strings retain the source CSV precision.

Predictions cover window centers only. Frames skipped by the mask, invalid
windows, and noncenter frames receive no inferred label. Output CSVs are
written to a partial file and finalized only after that input finishes
successfully. Existing finalized outputs require `--overwrite` to replace.

## Checks

```powershell
python -m unittest project.infer.test_streaming project.infer.test_infer_csv
```

Tests cover masks, strict timestamp grouping, ID changes, exact timing
thresholds, center selection, chunk continuity, probability fusion, and output
provenance. The four real checkpoints also passed a CPU inference smoke check
on actual June 23 windows.

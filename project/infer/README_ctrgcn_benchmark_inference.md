# CTR-GCN inference on the June 23 and July 20 benchmarks

This job runs the four provided CTR-GCN streams on the two previously
preprocessed benchmark recordings and writes a separate set of prediction CSVs.
It does not modify or replace the earlier 10 fps inference scripts or outputs.

## Install

Transfer `ctrgcn_benchmark_inference_bundle.zip` to the PYSKL repository root on
the cluster. From that root, inspect and extract it:

```bash
cd /project/def-mbolic/yunzelu/pyskl
unzip -l ctrgcn_benchmark_inference_bundle.zip
unzip -n ctrgcn_benchmark_inference_bundle.zip
```

The `-n` option preserves existing files. If any listed destination already
exists, check that it contains the intended version before submitting. The
bundle adds these files:

- `project/infer/infer_ctrgcn_benchmark.py`
- `project/infer/models_ctrgcn_10fps.py`
- `project/infer/compare_prediction_centers.py`
- `project/infer/ctrgcn_benchmark_local.sha256`
- `project/slurm/run_inference_ctrgcn_benchmark_h100.sh`
- `README_ctrgcn_benchmark_inference.md`

The job uses the existing repository code, `project/infer/streaming.py`, and
the existing cluster `.venv`.

## Inputs and models

The SLURM job explicitly expects both files directly in
`data/project/csv/benchmark/`:

- `pose_2026-06-23_ds_cf.csv`
- `pose_2026-07-20_ds_cf.csv`

Each requires a paired multiperson mask in the same directory. For a CSV ending
`_ds_cf.csv`, either `_ds_mm.json` or `_ds_cf_mm.json` is accepted when exactly
one is present. If both exist, the job stops so that you can select the intended
mask using `--mask-suffix legacy` or `--mask-suffix confidence`. A filename alone
does not prove which preprocessing stage produced a mask. Use the same masks
as the earlier run for a meaningful comparison.

The four configurations are read from
`configs/ctrgcn/ctrgcn_pyskl_radarv4/ds3p123/{j,b,jm,bm}.py`. Checkpoints are
`work_dirs/ctrgcn/radar_v4_yolo26xpose_clip60_ds3p123/{j,b,jm,bm}/epoch_20.pth`.
The script verifies the model type, stream, class count, expected test pipeline,
and checkpoint presence. The actual cluster artifacts have not been loaded by
these local tests.

## Submit

From the PYSKL repository root, start a GPU smoke test of 16 windows per input:

```bash
sbatch project/slurm/run_inference_ctrgcn_benchmark_h100.sh \
  --max-windows 16 \
  --output-dir work_dirs/project/inference/ctrgcn_benchmark_10fps_smoke
```

The smoke run is intentionally incomplete. Inspect its SLURM log under
`project/slurm/` and confirm four checkpoint loads and output rows for both
files. Then run the full job:

```bash
sbatch project/slurm/run_inference_ctrgcn_benchmark_h100.sh
```

Full results go to `work_dirs/project/inference/ctrgcn_benchmark_10fps/`.
Output collisions are refused by default. Use a different `--output-dir` for
a separate attempt, or `--overwrite` only when replacement is intended. Run
`sbatch` from the repository root because SLURM's log path is relative to the
submission directory.

The job requests one H100 10 GB slice, 32 GB RAM, and two hours. Batch size 32
and chunk size 2,000 are unbenchmarked starting points. If needed, for example:

```bash
sbatch --export=ALL,BATCH_SIZE=16 project/slurm/run_inference_ctrgcn_benchmark_h100.sh
```

If both mask names exist for either recording, pass the same intended suffix
for both inputs, for example:

```bash
sbatch project/slurm/run_inference_ctrgcn_benchmark_h100.sh --mask-suffix legacy
```

For a case requiring different suffix choices, run the Python program separately
for the two input CSVs with different `--output-dir` values.

## Compatibility and comparison

The reader uses the earlier 10 fps inference rules: 20 frames per window, stride
4, center offset 10, maximum adjacent gap 0.5 seconds, and maximum span 2.5
seconds. It does not downsample the input. For a 20-frame input, the CTR-GCN
test sampler selects all 20 frames in order. The saved test preprocessing is
used for each stream.

The CTR-GCN training data uses a different class-index order. Each model's
probabilities are reordered into the prior inference label order before
fusion and export. The native labels for transitions involving lying are more
specific: the source specifies bed-to-sit, floor-to-stand, and sit-to-bed, while
the exported labels use the broader prior lie-to-sit, lie-to-stand, and
sit-to-lie names. This should be considered when interpreting results.

Fusion is `(2*j + 2*b + jm + bm) / 6`. Prediction CSVs retain the usual metadata,
class names and probabilities, and all four stream probability columns. Do not
apply the old postprocessing fusion script's 1:1:2:2 default if comparing these
2:2:1:1 results to the original inference CSVs. If you apply Viterbi or reliability
filtering, apply the same settings to both models.

The input and mask SHA-256 hashes are recorded in the new inference summaries.
The old summaries do not record hashes. Before submitting, you can check the
cluster inputs against the local 0903 copies used for validation:

```bash
cd /project/def-mbolic/yunzelu/pyskl/data/project/csv/benchmark
sha256sum -c /project/def-mbolic/yunzelu/pyskl/project/infer/ctrgcn_benchmark_local.sha256
cd /project/def-mbolic/yunzelu/pyskl
```

The manifest names the local `_ds_mm.json` masks. If the cluster masks use a
different name, compare their individual hashes to the manifest manually. These
checks establish whether the cluster files match the local validation copies,
not whether the old cluster input bytes were identical to those local copies.

The included comparison utility checks
every center timestamp and shared window/source metadata field, for example:

```bash
python project/infer/compare_prediction_centers.py \
  work_dirs/project/inference/0903_10fps/pose_2026-06-23_ds_cf__center_predictions.csv \
  work_dirs/project/inference/ctrgcn_benchmark_10fps/pose_2026-06-23_ds_cf__center_predictions.csv
```

Run it for both dates. Matching centers and metadata do not by themselves prove
that all input poses or masks are identical; compare file hashes when the earlier
input files are available. The two trained model sets also used different
training splits, so this comparison concerns the trained systems and does not
isolate architecture alone.

## Local validation

The new reader was validated on local copies of both benchmark inputs and masks.
It found 66,043 valid windows for June 23 and 50,637 for July 20, with all
recorded scan counts matching the earlier inference summaries. Reconstructed
window centers and key window metadata also matched the earlier prediction rows
for both dates. Synthetic tests checked the class permutation, fusion, output
schema, and mask handling. No real CTR-GCN model or GPU job has been run by
this preparation.

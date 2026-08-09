# S2: Continuous-Window Adaptation of the PoseC3D Skeleton Teacher

Study 2 keeps the E2 trimmed-only teacher unchanged, then adds continuous-window
fine-tuning on the same subject-wise folds.

Default protocol:

- FPS: `30`
- Window: `60` frames
- Stride: `10` frames
- Center frame: `start + 30`
- No tail windows and no temporal zero padding
- All-zero pose frames are removed from the saved model tensor after a candidate
  source window is accepted on the original timeline. The original
  `start_frame`, `end_frame`, `center_frame`, timestamps, and 60-frame
  per-frame label timeline are preserved, and retained source frames are stored
  in `source_frame_indices`.
- Default retained-pose threshold: at least `30` nonzero pose frames from the
  60-frame source window (`--min-valid-ratio 0.5`)
- Default recording scope: non-walk JSONL recordings under
  `data/radar_v4/raw_jsonl/yolo26xpose`; walk-only sessions ending in
  `-walk` are excluded from both training and evaluation
- Segmental metrics are computed on the center-time grid: one step every 10
  video frames, about `0.333` seconds

Prepare S2 artifacts:

```bash
python thesis/s2/stage1_report.py --overwrite
python thesis/s2/build_continuous_windows.py --min-valid-ratio 0.5 --overwrite
python thesis/s2/generate_configs.py --overwrite
python thesis/s2/class_sampling_report.py --overwrite
python thesis/s2/sanity_check.py --skip-prediction-checks --overwrite
```

The generated configs default to `videos_per_gpu=8`, `workers_per_gpu=1`,
`pin_memory=False`, and `find_unused_parameters=False`. The continuous-window
pickle is large, so these defaults are intentionally lower than the E2
trimmed-clip batch settings to avoid CPU RAM exhaustion when DDP ranks and
dataloader workers replicate dataset state.

Stage-2 configs default to `--class-sample-strategy sqrt` and
`--class-sample-power 0.5`. Training first pre-grids every valid stride-10
window, then each epoch draws `epoch_size` windows with replacement. The class
draw rule is `P(c) = sqrt(n_c) / sum_j sqrt(n_j)`, where `n_c` is the fold's
training-window count for class `c`; after drawing a class, one pre-gridded
window from that class is sampled uniformly. This reduces redundant centers from
long state intervals while still oversampling transition classes relative to the
natural sliding-window distribution.

Compute-saving local Method A reproduction from existing E2 scores:

```bash
python thesis/s2/materialize_A_from_e2.py --overwrite
python thesis/s2/evaluate_predictions.py --method A --overwrite
```

Full deterministic Method A inference, if raw logits are needed:

```bash
python thesis/s2/infer.py --method A --stream joint --split test --overwrite
python thesis/s2/evaluate_predictions.py --method A --stream joint --overwrite
```

Train Stage 2 on one fold and the joint stream:

```bash
bash tools/dist_train.sh thesis/s2/configs/fold_a/joint/posec3d_continuous_hard_B.py 4 --validate --seed 42 --deterministic
python thesis/s2/select_stage2_checkpoint.py --method B --stream joint --folds a --overwrite
python thesis/s2/infer.py --method B --stream joint --split test --folds a --overwrite
python thesis/s2/evaluate_predictions.py --method B --stream joint --overwrite

bash tools/dist_train.sh thesis/s2/configs/fold_a/joint/posec3d_continuous_soft_C_eta050.py 4 --validate --seed 42 --deterministic
python thesis/s2/select_stage2_checkpoint.py --method C --stream joint --folds a --etas 0.50 --overwrite
python thesis/s2/infer.py --method C --stream joint --split test --folds a --eta selected --overwrite
python thesis/s2/evaluate_predictions.py --method C --stream joint --overwrite
```

After A, B, and C are evaluated:

```bash
python thesis/s2/summarize.py --stream joint --overwrite
```

After the selected method is also trained for the limb stream:

```bash
python thesis/s2/fuse.py --method B --overwrite
python thesis/s2/evaluate_predictions.py --method B --stream fusion --overwrite
```

Run `eta=0.25` and `eta=0.75` only after the `eta=0.50` soft-target run is
worth expanding, then re-run `select_stage2_checkpoint.py --method C` with all
three eta values.

Pure temporal soft-target Method C uses `eta=1.0`, because:

```text
q_final = (1 - eta) * one_hot(center_label) + eta * q_temporal
```

Run the full three-fold joint/limb `eta=1.0` job, then evaluate fusion and
write summary tables:

```bash
sbatch thesis/s2/run_c_eta100_full.sh
```

This refreshes Method C outputs and summaries under `work_dirs/thesis/s2`.
The summary tables compare the new Method C eta100 result against the existing
Method A and B metrics.

If a Slurm job is still OOM-killed, reduce loader fan-out before increasing the
experiment scope:

```bash
S2_VIDEOS_PER_GPU=4 S2_WORKERS_PER_GPU=0 sbatch thesis/s2/run.sh
```

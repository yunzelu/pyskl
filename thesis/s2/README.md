# S2: Continuous-Window Adaptation of the PoseC3D Skeleton Teacher

Study 2 keeps the E2 trimmed-only teacher unchanged, then adds continuous-window
fine-tuning on the same subject-wise folds.

Default protocol:

- FPS: `30`
- Window: `60` frames
- Stride: `10` frames
- Center frame: `start + 30`
- No tail windows and no temporal zero padding
- Default recording scope: non-walk sessions, matching `thesis/e2`
- Segmental metrics are computed on the center-time grid: one step every 10
  video frames, about `0.333` seconds

Prepare S2 artifacts:

```bash
python thesis/s2/stage1_report.py --overwrite
python thesis/s2/build_continuous_windows.py --overwrite
python thesis/s2/generate_configs.py --overwrite
python thesis/s2/sanity_check.py --overwrite
```

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

# E2: Untrimmed Skeleton Sliding-Window Evaluation

E2 is split into reusable stages so later experiments can insert temperature
calibration, temporal decoding, or other score transforms between inference and
evaluation.

Run from the repo root:

```bash
python thesis/e2/infer_scores.py --stream joint --overwrite
python thesis/e2/infer_scores.py --stream limb --overwrite
python thesis/e2/fuse_scores.py --overwrite

python thesis/e2/evaluate_scores.py --scores work_dirs/thesis/e2/scores/e2_joint_logits.csv --overwrite
python thesis/e2/evaluate_scores.py --scores work_dirs/thesis/e2/scores/e2_limb_logits.csv --overwrite
python thesis/e2/evaluate_scores.py --scores work_dirs/thesis/e2/scores/e2_fusion_joint_limb_logits.csv --overwrite
```

Default protocol:

- FPS: `30`
- Window: `60` frames
- Stride: `10` frames
- Center frame: `start + 30`
- No tail window
- Test subjects are parsed from `work_dirs/posec3d/8111/*_test_<subject>`.
- Only non-walk JSONL folders for each fold's test subject are evaluated.
- Checkpoint auto-selection uses `best_macro_f1_epoch_*.pth`.
- Ground-truth `null`, `DELETE`, `END`, and kneeling labels are excluded.
- Class names and class order follow `tools/data/radar_v4/build_pyskl_pkl.py`.

Score files are the boundary between components:

- `infer_scores.py` creates `work_dirs/thesis/e2/scores/e2_<stream>_logits.csv` by default.
- `fuse_scores.py` reads joint/limb score files and writes a fused score file.
- `evaluate_scores.py` can evaluate any score file with the same schema.

Fusion defaults to joint:limb `1:1`. To change the ratio later:

```bash
python thesis/e2/fuse_scores.py --joint-weight 2 --limb-weight 1 --output work_dirs/thesis/e2/scores/e2_fusion_j2_l1_logits.csv --overwrite
```

To export probabilities instead of logits:

```bash
python thesis/e2/infer_scores.py --stream joint --score-output prob --overwrite
python thesis/e2/infer_scores.py --stream limb --score-output prob --overwrite
python thesis/e2/fuse_scores.py --joint-scores work_dirs/thesis/e2/scores/e2_joint_scores.csv --limb-scores work_dirs/thesis/e2/scores/e2_limb_scores.csv --output work_dirs/thesis/e2/scores/e2_fusion_joint_limb_scores.csv --overwrite
```

The evaluator uses argmax over the score columns, so it can evaluate either
probability or logit score files. Fusion requires both inputs to use the same
`score_type`.

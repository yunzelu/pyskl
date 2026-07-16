# E4: HMM/Viterbi Temporal Refinement

E4 uses the same 8/1/1/1 PoseC3D checkpoints as E2/E3. It consumes E3 temperature-scaled probabilities, tunes a Viterbi smoothing strength on the validation subject, then applies the selected setting to the test subject.

Prerequisites:

```bash
python thesis/e3/infer_logits.py --stream joint --split test --overwrite
python thesis/e3/infer_logits.py --stream joint --split calib --overwrite
python thesis/e3/fit_temperature.py --stream joint --overwrite
```

Run E4:

```bash
sbatch thesis/e4/run.sh
```

Useful restricted/manual commands:

```bash
python thesis/e4/infer_val_logits.py --stream joint --overwrite
python thesis/e4/calibrate_scores.py --stream joint --split val --overwrite
python thesis/e4/calibrate_scores.py --stream joint --split test --overwrite
python thesis/e4/tune_viterbi.py --stream joint --overwrite
python thesis/e4/apply_viterbi.py --stream joint --overwrite
python thesis/e4/evaluate.py --stream joint --overwrite
```

Outputs:

- validation logits: `work_dirs/thesis/e4/logits/`
- calibrated probabilities: `work_dirs/thesis/e4/scores/`
- tuning grid and manual transition matrix: `work_dirs/thesis/e4/tuning/`
- final comparison report: `work_dirs/thesis/e4/eval/`

Manual transition matrix rule:

- `Lying-Stationary`, `Sit-Stationary`, and `Walking` are the state vertices.
- Since the 9-class label set has no explicit standing class, `Walking` is used as the standing/walking vertex.
- State rows give equal valid weight to self and outgoing transition edges.
- Transition and `Falling` rows prefer the end state over staying in the transition, approximately `0.6 / 0.4`.
- Topologically connected but wrong-direction moves receive a small soft-impossible probability.
- Unconnected opposite-side moves receive a smaller hard-impossible probability.

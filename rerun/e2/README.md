# E2 Raw Uncertainty Estimation and Reliability Ranking

E2 evaluates uncertainty branches on the validation subjects only. The current
implementation covers the E2A MC-dropout branch for the E1-B continuous-window
protocol and includes the existing deterministic E1-B validation fusion as the
raw deterministic baseline.

## E2A MC Dropout

The MC-dropout runner uses the E1-B checkpoints selected by validation
center macro-F1:

```text
work_dirs/rerun/e1/fold_<fold>/<stream>/b_continuous_window/best_macro_f1_epoch_*.pth
```

For each fold and stream, it verifies that the loaded ST-GCN++ model has:

- `GCNHead`
- `dropout_ratio=0.5`
- `cls_head.dropout` as `torch.nn.Dropout(p=0.5)`
- `fc_cls.out_features == 9`

During stochastic inference, the complete model is put in evaluation mode and
only `cls_head.dropout` is switched to training mode. BatchNorm and all other
modules remain in evaluation mode.

Run all three folds with:

```bash
python rerun/e2/run_e2a_mc_dropout.py
```

Defaults:

- validation split only
- E1-B continuous-window pkl
- 10 MC passes
- seed 42, set once before the full run
- no temperature scaling
- top-label ECE with 15 bins

Useful overrides:

```bash
python rerun/e2/run_e2a_mc_dropout.py --device cuda:0 --batch-size 128
python rerun/e2/run_e2a_mc_dropout.py --device cpu --batch-size 64 --num-workers 0
python rerun/e2/run_e2a_mc_dropout.py --folds a --overwrite
```

Local CPU wrapper:

```bash
bash rerun/e2/local/run_e2a_mc_dropout_cpu.sh
```

SLURM wrapper:

```bash
sbatch rerun/e2/slurm/run_e2a_mc_dropout.sh
```

## E2A Laplace Wrapper Check

Before fitting last-layer Laplace, verify that the raw-logit wrapper reproduces
the stored PYSKL validation probabilities for the selected E1-B checkpoints:

```bash
sbatch rerun/e2/slurm/run_e2a_laplace_wrapper_check_h100.sh
```

This check runs over 3 folds x 2 streams. It does not import or require
`laplace-torch`; it only verifies the integration point needed by the later
Laplace fitting script:

- checkpoint source:
  `work_dirs/rerun/e1/fold_<fold>/<stream>/b_continuous_window/best_macro_f1_epoch_*.pth`
- validation config:
  `configs/stgcn++/stgcn++_radarv4/rerun/e1/fold_<fold>/<stream>/validation/b_validation.py`
- final probabilistic layer for Laplace: `cls_head.fc_cls`
- expected final layer shape: `Linear(in_features=256, out_features=9)`
- dropout disabled during the check through `wrapper.eval()`
- probability comparison:
  `softmax(wrapper(keypoint))` versus the matching stream's validation
  `best_pred.pkl`

Outputs are written under:

```text
rerun/e2/reports/laplace_wrapper_check/
```

For the H100 cluster, the wrapper uses:

```text
#SBATCH --gpus-per-node=h100:4
module purge
module load StdEnv/2023
module load python/3.10
module load opencv/4.8.1
source "/project/def-mbolic/yunzelu/pyskl/.venv/bin/activate"
cd "/project/def-mbolic/yunzelu/pyskl"
```

## E2A Last-Layer Laplace

After the wrapper check passes, fit and evaluate the raw Laplace branch on the
H100 cluster:

```bash
sbatch rerun/e2/slurm/run_e2a_laplace_h100.sh
```

The runner uses the frozen E2A Laplace protocol:

- base checkpoint: selected E1-B ST-GCN++ checkpoint
- fit split: the fold's E1-B outer training subjects
- fitting examples: every unique valid training window once
- fitting preprocessing: deterministic validation-style Joint or Bone pipeline
- disabled during fitting: random augmentation, square-root sampling, dropout
- last-layer module: `cls_head.fc_cls`
- approximation: last-layer Laplace with `hessian_structure="kron"`
- backend: `CurvlinopsGGN`
- likelihood temperature: `1.0`
- prior precision: one scalar per fold and stream, optimized by marginal
  likelihood
- posterior samples: `30`
- prediction method: `pred_type="nn"`
- fusion: equal probability fusion per posterior sample

The H100 script fits the six stream approximations first:

```text
Fold A Joint, Fold A Bone
Fold B Joint, Fold B Bone
Fold C Joint, Fold C Bone
```

Then it runs a fusion/report pass. Joint posterior samples use seed `42001`,
Bone posterior samples use seed `42002`.

Laplace outputs:

```text
work_dirs/rerun/e2/e2a_laplace/fold_<fold>/<stream>/b_continuous_window/validation/
work_dirs/rerun/e2/e2a_laplace/fold_<fold>/fusion/b_continuous_window/validation/
```

Key fused files:

```text
laplace_prob_samples.npy
laplace_mean_probabilities.npy
laplace_mean_pred.pkl
laplace_quantities.npz
metrics.json
sample_ids.json
```

The combined E2A reports are rewritten to include deterministic, MC-dropout
when available, and Laplace:

```text
rerun/e2/reports/e2a_raw_predictive_fold_metrics.csv
rerun/e2/reports/e2a_raw_predictive_mean_sd.csv
rerun/e2/reports/e2a_raw_predictive_summary.json
rerun/e2/reports/e2a_raw_predictive_summary.md
```

## Outputs

Per-stream MC arrays:

```text
work_dirs/rerun/e2/e2a_mc_dropout/fold_<fold>/<stream>/b_continuous_window/validation/
```

Fused MC arrays and uncertainty quantities:

```text
work_dirs/rerun/e2/e2a_mc_dropout/fold_<fold>/fusion/b_continuous_window/validation/
```

Key fused files:

```text
mc_prob_passes.npy
mc_mean_probabilities.npy
mc_mean_pred.pkl
mc_quantities.npz
metrics.json
sample_ids.json
```

Reports:

```text
rerun/e2/reports/e2a_raw_predictive_fold_metrics.csv
rerun/e2/reports/e2a_raw_predictive_mean_sd.csv
rerun/e2/reports/e2a_mc_dropout_summary.json
rerun/e2/reports/e2a_mc_dropout_summary.md
```

The report metrics are:

- center accuracy: argmax equals the center-frame hard label; this is aligned
  with E1 `top1_acc`
- center macro-F1: unweighted mean of per-class F1 over all nine final classes;
  this is aligned with E1 `macro_f1`
- state macro-F1 over `lie-stationary`, `sit-stationary`, and `walk`
- transition macro-F1 over `fall`, `transition-lie-to-sit`,
  `transition-lie-to-stand`, `transition-sit-to-lie`,
  `transition-sit-to-stand`, and `transition-stand-to-sit`
- raw NLL
- raw Brier score
- raw ECE

The deterministic branch performs a runtime alignment check against the E1-B
validation fusion metrics. For each fold, E2 `center_accuracy` must match E1
`top1_acc`, and E2 `center_macro_f1` must match E1 `macro_f1`. This check uses
`work_dirs/rerun/e1/fold_<fold>/fusion/b_continuous_window/validation/best_eval.json`
when present, otherwise `rerun/e1/reports/e1_validation_fold_metrics.csv`.

The fused MC branch implements:

```text
p_MC^(k) = 0.5 * (p_joint^(k) + p_bone^(k))
pbar_MC = mean_k p_MC^(k)
```

The saved `mc_quantities.npz` also contains predictive entropy, expected
entropy, mutual information, predictions, labels, and branch-specific errors
for the later E2B reliability-ranking analysis.

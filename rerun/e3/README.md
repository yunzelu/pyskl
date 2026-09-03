# E3 MC Pool-Then-Calibrate Temperature Scaling

E3 uses the selected E2 branch: MC dropout. It compares the raw MC predictive
mean against the temperature-calibrated MC predictive mean.

## Data Flow

For each fold, the script generates MC dropout predictions for the calibration
and test splits using the selected E1-B Joint and Bone checkpoints.

Per MC pass:

```text
p_J^(k) = softmax(z_J^(k))
p_B^(k) = softmax(z_B^(k))
p_F^(k) = 0.5 * (p_J^(k) + p_B^(k))
```

Then:

```text
pbar_MC = mean_k p_F^(k)
```

Temperature scaling is applied after pooling:

```text
s = log(max(pbar_MC, 1e-12))
q(T) = softmax(s / T)
```

The temperature is fitted on the calibration subject only and then applied to
the outer test subject.

## Metric Conventions

E3 keeps the E2 conventions:

- NLL: `mean_over_samples(-log(p_true))` using natural logarithms
- Brier: `mean_over_samples(sum_over_classes((p-onehot)^2))`
- `divide_by_num_classes`: `false`
- ECE: top-label ECE with 15 fixed equal-width bins
- ECE bin edges: `[lower, upper)` except the final bin `[lower, upper]`

The ECE bin convention is inherited from `rerun/e2/run_e2a_mc_dropout.py`.

## Run

H100:

```bash
sbatch rerun/e3/slurm/run_e3_mc_temperature_scaling_h100.sh
```

Direct Python:

```bash
python rerun/e3/run_e3_mc_temperature_scaling.py
```

The default temperature optimizer is PyTorch LBFGS over `log(T)` with:

```text
max_iter = 100
lr = 0.1
```

No extra library is needed beyond the existing PYSKL runtime. The optional
`--temperature-optimizer numpy_golden` mode exists only for dependency-light
local report checks after MC artifacts have already been generated.

## Outputs

MC artifacts:

```text
work_dirs/rerun/e3/mc_temperature_scaling/fold_<fold>/<stream>/b_continuous_window/<split>/
work_dirs/rerun/e3/mc_temperature_scaling/fold_<fold>/fusion/b_continuous_window/<split>/
```

Fold temperatures:

```text
work_dirs/rerun/e3/mc_temperature_scaling/fold_<fold>/temperature_scaling/temperature.json
```

Reports:

```text
rerun/e3/reports/e3_calibration_fit_sanity.csv
rerun/e3/reports/e3_test_fold_metrics.csv
rerun/e3/reports/e3_test_mean_sd.csv
rerun/e3/reports/e3_mc_temperature_scaling_summary.json
rerun/e3/reports/e3_mc_temperature_scaling_summary.md
rerun/e3/reports/e3_test_calibration_deltas.svg
```

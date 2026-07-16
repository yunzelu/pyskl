# E3: Temperature Calibration and Uncertainty Analysis

E3 uses the same trained PoseC3D checkpoints as E2. Checkpoints are selected
from `work_dirs/posec3d/8111` by reading `best_macro_f1_epoch_N.pth` and then
loading the matching `epoch_N.pth`.

Run from the repo root:

```bash
python thesis/e3/infer_logits.py --stream joint --split both --overwrite
python thesis/e3/fit_temperature.py --stream joint --overwrite
python thesis/e3/evaluate_uncertainty.py --stream joint --overwrite
```

For limb:

```bash
python thesis/e3/infer_logits.py --stream limb --split both --overwrite
python thesis/e3/fit_temperature.py --stream limb --overwrite
python thesis/e3/evaluate_uncertainty.py --stream limb --overwrite
```

The fold name provides the calibration and test subjects:

- `fold_a`: calib `dengdeng`, test `chenzhe`
- `fold_b`: calib `li`, test `jiadi`
- `fold_c`: calib `xilai`, test `saad`

Outputs:

- logits: `work_dirs/thesis/e3/logits/`
- temperatures: `work_dirs/thesis/e3/temperatures/`
- uncertainty reports: `work_dirs/thesis/e3/analysis/`

Reports include:

- ECE before and after temperature scaling
- confidence-accuracy curve
- coverage-vs-accuracy curve
- per-window confidence, entropy, top1/top2 margin, correctness, nearest boundary distance, and state/transition grouping


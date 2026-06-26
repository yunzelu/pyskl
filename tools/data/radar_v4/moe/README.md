# Radar v4 MoE Gate

This folder trains and applies a small mixture-of-experts gate over four
single-stream CTR-GCN prediction timelines.

The gate input is the concatenation of stream probability vectors:

```text
x_t = [p_t^j, p_t^jm, p_t^b, p_t^bm] in R^(4 * num_classes)
```

The gate architecture is:

```text
Linear(36, 16) -> ReLU -> Dropout(0.2) -> Linear(16, 4)
```

At inference time:

```text
alpha_t = softmax(gate(x_t))
p_t = sum_s alpha_t,s * p_t^s
```

## Manifest

Both scripts use a CSV manifest with one row per session:

```csv
session,origin_session,j,jm,b,bm
35-mia-sit,data/radar_v4/origin/35-mia-sit,path/to/j.csv,path/to/jm.csv,path/to/b.csv,path/to/bm.csv
```

For `apply_gate_csv.py`, `origin_session` may be empty because labels are not
needed.

## Train

```powershell
python tools/data/radar_v4/moe/train_gate.py `
  --manifest tmp/moe_valid_manifest.csv `
  --output work_dirs/radar_v4_moe/gate.pt `
  --metrics-json work_dirs/radar_v4_moe/gate_metrics.json `
  --overwrite
```

Use the validation-subject stream CSVs in the training manifest. The trainer
uses frames with valid ground truth, selected skeleton input, and a prediction
from every stream.

## Apply

```powershell
python tools/data/radar_v4/moe/apply_gate_csv.py `
  --manifest tmp/moe_test_manifest.csv `
  --gate work_dirs/radar_v4_moe/gate.pt `
  --output-dir work_dirs/radar_v4_moe/predictions `
  --include-gate-columns `
  --overwrite
```

By default, frames with no selected skeleton input are written as `NoDetection`
or `NoPrediction`, matching the fixed late-fusion behavior.

#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=e3_mc_temp_val
#SBATCH --output=rerun/e3/slurm/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.10

PROJECT_ROOT="${PROJECT_ROOT:-/project/def-mbolic/yunzelu/pyskl}"
source "${PROJECT_ROOT}/.venv/bin/activate"
cd "${PROJECT_ROOT}"

# CPU evaluation only: reuse the existing 30-pass E2 validation predictions
# and full-precision E3 temperatures fitted on each calibration subject.
# No inference, temperature optimization, or GPU is needed.
export MC_OUTPUT_ROOT="${MC_OUTPUT_ROOT:-work_dirs/rerun/e2/e2a_mc_dropout}"
export E3_OUTPUT_ROOT="${E3_OUTPUT_ROOT:-work_dirs/rerun/e3/mc_temperature_scaling}"
export REPORT_DIR="${REPORT_DIR:-rerun/e3/reports/validation}"

# E3's CLI only accepts calib/test and its temperature mode always fits T.
# Reuse its metric and report functions here without changing Python sources.
python - <<'PY'
import os
from pathlib import Path

import numpy as np

from rerun.dataset.build_radar_v4_yolo26xpose_datasets import FOLDS as SUBJECT_SPLITS
from rerun.e3 import run_e3_mc_temperature_scaling as e3

mc_root = Path(os.environ["MC_OUTPUT_ROOT"])
e3_root = Path(os.environ["E3_OUTPUT_ROOT"])
report_dir = Path(os.environ["REPORT_DIR"])
rows = []
sources = []

# Check every fold's inputs before writing any results. Never fall back to
# fitting T or to the four-decimal temperatures printed in the old report.
for fold in e3.FOLDS:
    source = e3.fusion_output_dir(mc_root, fold, "validation")
    temperature_path = e3.temperature_output_dir(e3_root, fold) / "temperature.json"
    for path in [temperature_path, source / "mc_mean_probabilities.npy",
                 source / "labels.npy", source / "sample_ids.json", source / "metrics.json"]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved input: {path}")

for fold in e3.FOLDS:
    source = e3.fusion_output_dir(mc_root, fold, "validation")
    temperature_path = e3.temperature_output_dir(e3_root, fold) / "temperature.json"
    saved = e3.load_json(temperature_path)
    metadata = e3.load_json(source / "metrics.json")
    if saved["fold"] != fold or saved["calibration_subject_split"] != "calib":
        raise ValueError(f"Wrong fold or temperature fit split: {temperature_path}")
    if (metadata["fold"] != fold or metadata["split"] != "val"
            or metadata["branch"] != "mc_dropout" or metadata["num_passes"] != 30):
        raise ValueError(f"Expected fold {fold} raw 30-pass MC validation predictions: {source}")
    temperature = float(saved["temperature"])
    eps = float(saved["eps"])
    ece_bins = int(saved["ece_bins"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Invalid saved temperature: {temperature_path}")

    raw = np.load(source / "mc_mean_probabilities.npy").astype(np.float64)
    labels = np.load(source / "labels.npy").astype(np.int64)
    # E2 records the subject in session_name (<index>-<subject>-<session>).
    # Convert its IDs to E3's subject_id/recording_id schema.
    sample_ids = [
        {"subject_id": item["session_name"].split("-", 2)[1],
         "recording_id": item["session_name"],
         "window_row_start": item["window_row_start"],
         "center_source_frame": item["center_source_frame"]}
        for item in e3.load_json(source / "sample_ids.json")
    ]
    if raw.shape != (labels.size, len(e3.LABELS)) or len(sample_ids) != labels.size:
        raise ValueError(f"Probability/label/sample ID dimensions differ: {source}")
    if {item["subject_id"] for item in sample_ids} != set(SUBJECT_SPLITS[f"fold_{fold}"]["val"]):
        raise ValueError(f"Sample IDs do not belong to the validation subject: {source}")
    e3.assert_unique_sample_ids(sample_ids, str(source))
    calibrated = e3.apply_pool_temperature(raw, temperature, eps=eps)
    metrics = e3.compare_raw_calibrated_metrics(raw, calibrated, labels, ece_bins)
    row = {"fold": fold, "split": "val", "temperature": temperature, **metrics}
    rows.append(row)
    sources.append({"fold": fold, "raw_predictions": str(source),
                    "temperature_file": str(temperature_path), "temperature": temperature,
                    "eps": eps, "ece_bins": ece_bins, "num_passes": 30})

    out_dir = e3.fusion_output_dir(e3_root, fold, "validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "raw_mc_mean_probabilities.npy", raw.astype(np.float32))
    np.save(out_dir / "temperature_calibrated_probabilities.npy", calibrated.astype(np.float32))
    np.save(out_dir / "labels.npy", labels)
    e3.write_json(out_dir / "sample_ids.json", sample_ids)
    e3.write_json(out_dir / "temperature_calibrated_metrics.json", {
        "fold": fold, "split": "val", "temperature": temperature, "metrics": metrics,
        "sample_ids_head": sample_ids[:5], "sample_ids_tail": sample_ids[-5:],
        "temperature_refitted": False, "source": sources[-1],
    })
    print(f"[DONE] fold={fold} frozen T={temperature:.9g} "
          f"val_nll={metrics['raw_nll']:.4f}->{metrics['calibrated_nll']:.4f}")

summary = e3.aggregate_rows(rows, "val")
report_dir.mkdir(parents=True, exist_ok=True)
e3.write_csv(report_dir / "e3_validation_fold_metrics.csv", rows)
e3.write_csv(report_dir / "e3_validation_mean_sd.csv", [summary])
e3.write_json(report_dir / "e3_mc_temperature_scaling_summary.json", {
    "experiment": "E3 MC pool-then-calibrate temperature scaling",
    "selected_branch": "mc_dropout", "main_split": "val",
    "temperature_fit_split": "calib", "temperature_refitted": False,
    "num_passes": 30, "sources": sources,
    "fold_validation_metrics": rows, "validation_mean_sd": summary,
})
markdown = e3.markdown_table(rows, summary).replace(
    "Main result split: outer test subject.", "Main result split: validation subject."
)
(report_dir / "e3_mc_temperature_scaling_summary.md").write_text(
    markdown, encoding="utf-8", newline="\n"
)
svg_path = report_dir / "e3_validation_calibration_deltas.svg"
e3.write_delta_svg(svg_path, rows, summary)
svg_path.write_text(svg_path.read_text(encoding="utf-8").replace(
    "E3 Test Calibration Deltas", "E3 Validation Calibration Deltas"
), encoding="utf-8", newline="\n")
print(f"[DONE] wrote E3 validation reports under {report_dir}")
PY

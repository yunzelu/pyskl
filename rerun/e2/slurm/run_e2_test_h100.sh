#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --job-name=e2_test
#SBATCH --output=rerun/e2/slurm/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Submit from the repository root:
#   sbatch rerun/e2/slurm/run_e2_test_h100.sh
# Produces both original E2 report formats under rerun/e2/reports/test/.
# Uses the existing selected checkpoints, E1 test predictions and E2 validation
# metadata. MC uses the original run's pass count (30 in the saved results).
# Laplace rebuilds training curvature with the saved prior precision; it does
# not reselect the prior or train the network. No temperature scaling is used.
# All six streams run sequentially on the one allocated GPU.

module purge
module load StdEnv/2023
module load python/3.10
module load opencv/4.8.1

PROJECT_ROOT="${PROJECT_ROOT:-/project/def-mbolic/yunzelu/pyskl}"
source "${PROJECT_ROOT}/.venv/bin/activate"
cd "${PROJECT_ROOT}"

export BATCH_SIZE="${BATCH_SIZE:-128}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export REPORT_DIR="${REPORT_DIR:-rerun/e2/reports/test}"

# Existing E2 entry points hardcode validation. These process-local adapters
# reuse their inference, fusion and report functions without editing .py files.
python -u - <<'PY'
import os
import time
from pathlib import Path

import numpy as np

from rerun.e2 import run_e2a_mc_dropout as mc
from rerun.e2 import run_e2a_laplace as la
from rerun.e2 import evaluate_e2b_reliability as reliability


def test_metadata(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "split" and item == "val":
                item = "test"
            if key == "num_validation_samples":
                key = "num_test_samples"
            result[key] = test_metadata(item)
        return result
    if isinstance(value, (list, tuple)):
        return [test_metadata(item) for item in value]
    return value


def adapt_report_writers(module):
    write_json = module.write_json
    write_csv = module.write_csv
    markdown_report = module.markdown_report
    module.write_json = lambda path, data: write_json(path, test_metadata(data))
    module.write_csv = lambda path, rows: write_csv(path, test_metadata(rows))
    module.markdown_report = lambda rows, summary: markdown_report(rows, summary).replace(
        "Split: validation.", "Split: test."
    )


def stream_output_dir(root, fold, stream):
    return root / f"fold_{fold}" / stream / mc.CONDITION_DIR / "test"


def fusion_output_dir(root, fold):
    return stream_output_dir(root, fold, "fusion")


def base_config_path(root, fold, stream):
    return root / f"fold_{fold}" / stream / "b_continuous_window.py"


def test_labels_and_ids(module, root, fold):
    annotations = module.split_annotations(module.continuous_pkl_path(root, fold), "test")
    return (
        np.array([int(item["label"]) for item in annotations], dtype=np.int64),
        [module.sample_id_from_annotation(item) for item in annotations],
    )


for module in (mc, la):
    module.stream_output_dir = stream_output_dir
    module.fusion_output_dir = fusion_output_dir
    module.validation_config_path = base_config_path
    module.validation_labels_and_ids = (
        lambda root, fold, module=module: test_labels_and_ids(module, root, fold)
    )

for module in (mc, la, reliability):
    adapt_report_writers(module)

# MC reads data.test from the base E1 config. Laplace retains data.val's
# deterministic preprocessing and changes only the requested evaluation split.
original_dataset_cfg = la.deterministic_dataset_cfg
la.deterministic_dataset_cfg = lambda cfg, split: original_dataset_cfg(
    cfg, "test" if split == "val" else split
)

# Original E1 test artifacts are directly under b_continuous_window/.
la.e1_stream_prediction_path = lambda root, fold, stream: (
    root / f"fold_{fold}" / stream / mc.CONDITION_DIR / "best_pred.pkl"
)
la.e1_fusion_prediction_path = lambda root, fold: la.e1_stream_prediction_path(root, fold, "fusion")
original_deterministic_fusion = la.deterministic_fusion_probabilities


def deterministic_test_fusion(root, fold):
    probabilities, source = original_deterministic_fusion(root, fold)
    source["source"] = source["source"].replace("validation", "test")
    return probabilities, source


mc.deterministic_fusion_probabilities = deterministic_test_fusion
la.deterministic_fusion_probabilities = deterministic_test_fusion


def test_metric_reference(args, fold):
    path = args.e1_work_root / f"fold_{fold}" / "fusion" / mc.CONDITION_DIR / "best_eval.json"
    metrics = mc.load_json(path)
    return {"source": str(path), "top1_acc": metrics["top1_acc"], "macro_f1": metrics["macro_f1"]}


mc.e1_metric_reference = test_metric_reference
la.mc_dropout_metrics_for_fold = lambda args, fold: mc.load_json(
    fusion_output_dir(args.mc_output_root, fold) / "metrics.json"
)

original_branch_paths = reliability.branch_paths


def test_branch_paths(args, branch, fold):
    paths = original_branch_paths(args, branch, fold)
    base = paths["base"].with_name("test")
    return {key: base if key == "base" else base / path.name for key, path in paths.items()}


reliability.branch_paths = test_branch_paths

mc_args = mc.parse_args()
la_args = la.parse_args()
reliability_args = reliability.parse_args()
report_dir = Path(os.environ["REPORT_DIR"])
if report_dir.resolve() == Path("rerun/e2/reports").resolve():
    raise ValueError("REPORT_DIR must be separate from the original E2 validation reports")
for args in (mc_args, la_args, reliability_args):
    args.report_dir = report_dir
mc_args.device = la_args.device = "cuda:0"
mc_args.batch_size = int(os.environ["BATCH_SIZE"])
mc_args.num_workers = la_args.num_workers = int(os.environ["NUM_WORKERS"])

# Preflight all dependencies before starting inference. Take stochastic counts,
# seeds and Laplace batch sizes from the actual saved run, not Slurm defaults.
mc_metadata = []
source_fits = {}
source_metrics = {}
for fold in mc_args.folds:
    labels, _ = mc.validation_labels_and_ids(mc_args.data_root, fold)
    probabilities, _ = deterministic_test_fusion(mc_args.e1_work_root, fold)
    if len(probabilities) != len(labels):
        raise ValueError(f"Fold {fold}: E1 test probabilities do not match test labels")
    mc.assert_e1_metric_alignment(mc_args, fold, mc.predictive_metrics(probabilities, labels, mc_args.ece_bins))
    for stream in mc.STREAMS:
        checkpoint = mc.find_selected_checkpoint(mc_args.e1_work_root, fold, stream)
        config = base_config_path(mc_args.config_root, fold, stream)
        if not config.is_file():
            raise FileNotFoundError(config)
        mc_source = mc_args.output_root / f"fold_{fold}" / stream / mc.CONDITION_DIR / "validation"
        metadata = mc.load_json(mc_source / "metadata.json")
        mc_metrics = mc.load_json(mc_source / "metrics.json")
        if Path(mc_metrics["checkpoint"]).name != checkpoint.name:
            raise ValueError(f"MC checkpoint differs from the original E2 run: {fold}/{stream}")
        mc_metadata.append(metadata)
        la_source = la_args.output_root / f"fold_{fold}" / stream / la.CONDITION_DIR / "validation"
        fit_path = la_source / "fit_metadata.json"
        fit = la.load_json(fit_path)
        if Path(fit["checkpoint"]).name != checkpoint.name:
            raise ValueError(f"Laplace checkpoint differs from the original E2 run: {fold}/{stream}")
        expected = {"backend": "CurvlinopsGGN", "hessian_structure": "kron", "subset_of_weights": "last_layer",
                    "last_layer_name": la.LAST_LAYER_NAME, "temperature": 1.0}
        if any(fit[key] != value for key, value in expected.items()):
            raise ValueError(f"Unexpected original Laplace protocol: {fit_path}")
        prior = float(fit["selected_prior_precision"])
        if not np.isfinite(prior) or prior <= 0:
            raise ValueError(f"Invalid saved prior precision: {fit_path}")
        source_fits[fold, stream] = (fit_path, fit)
        source_metrics[fold, stream] = la.load_json(la_source / "metrics.json")


def shared_value(rows, key):
    values = {row[key] for row in rows}
    if len(values) != 1:
        raise ValueError(f"Original E2 runs disagree on {key}: {values}")
    return values.pop()


mc_args.num_passes = int(shared_value(mc_metadata, "num_passes"))
mc_args.seed = la_args.seed = int(shared_value(mc_metadata, "seed_set_once_by_script"))
la_args.num_posterior_samples = int(shared_value(source_metrics.values(), "num_posterior_samples"))
la_args.fit_batch_size = int(shared_value(source_metrics.values(), "fit_batch_size"))
la_args.eval_batch_size = int(shared_value(source_metrics.values(), "eval_batch_size"))
for stream in la.STREAMS:
    setattr(la_args, f"posterior_seed_{stream}", int(shared_value(
        [source_metrics[fold, stream] for fold in la_args.folds], "posterior_seed"
    )))

# Curvature was not saved by the original default job. Reconstruct it from
# outer TRAIN windows with the original prior fixed; never optimize on test.
original_run_stream = la.run_stream


def run_laplace_test_stream(args, fold, stream, device):
    args.source_fit_path, args.source_fit = source_fits[fold, stream]
    args.source_num_fit_samples = int(source_metrics[fold, stream]["num_fit_samples"])
    return original_run_stream(args, fold, stream, device)


def rebuild_laplace_with_fixed_prior(wrapper, train_loader, args):
    import torch
    from laplace import Laplace
    from laplace.curvature import CurvlinopsGGN

    if len(train_loader.dataset) != args.source_num_fit_samples:
        raise ValueError("Training window count differs from the original Laplace fit")
    wrapper.eval()
    la.verify_laplace_ready(wrapper, expected_in_features=args.expected_in_features)
    prior = float(args.source_fit["selected_prior_precision"])
    model = Laplace(
        model=wrapper, likelihood="classification", subset_of_weights="last_layer",
        hessian_structure="kron", backend=CurvlinopsGGN,
        last_layer_name=la.LAST_LAYER_NAME, prior_precision=prior, temperature=1.0,
    )
    started = time.time()
    kwargs = {"progress_bar": args.progress_bar} if la.accepts_keyword(model.fit, "progress_bar") else {}
    model.fit(train_loader, **kwargs)
    wrapper.eval()
    model.model.eval()
    metadata = dict(args.source_fit)
    metadata.update({
        "selected_prior_precision": float(model.prior_precision.detach().cpu().reshape(-1)[0].item()),
        "prior_precision_tensor": model.prior_precision.detach().cpu().numpy(),
        "prior_source": str(args.source_fit_path),
        "prior_optimization_method": "reused_original_E2_marglik_prior",
        "prior_reoptimized": False, "prior_steps": 0,
        "curvature_rebuilt_on": "train", "fit_seconds": time.time() - started,
        "optimize_prior_seconds": 0.0, "total_fit_seconds": time.time() - started,
        "torch_version": torch.__version__,
    })
    return model, metadata


la.run_stream = run_laplace_test_stream
la.fit_laplace_model = rebuild_laplace_with_fixed_prior
print(f"[INFO] Test evaluation: MC passes={mc_args.num_passes}, Laplace samples={la_args.num_posterior_samples}")
mc.run(mc_args)
la.run(la_args)
reliability.run(reliability_args)
print(f"[DONE] {report_dir / 'e2a_raw_predictive_summary.md'}")
print(f"[DONE] {report_dir / 'e2b_reliability_summary.md'}")
PY

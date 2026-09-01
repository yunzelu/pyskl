"""Generate E1 ST-GCN++ configs and fold job scripts.

E1 studies skeleton training-inference alignment:

- A1: train activity-aligned, evaluate activity-aligned.
- A2: use the same activity-aligned checkpoint, evaluate continuous windows.
- B: train continuous windows directly, evaluate continuous windows.
- C: train continuous windows directly with triangular temporal-composition
  soft targets, evaluate continuous windows with hard center labels.

A2 does not need a separate config file because it shares the same continuous
test dataset and evaluation protocol as B.
"""

from __future__ import annotations

import argparse
import json
import pickle
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = [
    "lie-stationary",
    "sit-stationary",
    "walk",
    "fall",
    "transition-lie-to-sit",
    "transition-lie-to-stand",
    "transition-sit-to-lie",
    "transition-sit-to-stand",
    "transition-stand-to-sit",
]

FOLDS = {
    "a": {
        "config_key": "fold_a",
        "train": ["chenzhe", "dengdeng", "hui", "jiadi", "mia", "rose", "xilai", "yunze"],
        "val": ["han"],
        "calib": ["saad"],
        "test": ["li"],
    },
    "b": {
        "config_key": "fold_b",
        "train": ["han", "jiadi", "li", "mia", "rose", "saad", "xilai", "yunze"],
        "val": ["hui"],
        "calib": ["chenzhe"],
        "test": ["dengdeng"],
    },
    "c": {
        "config_key": "fold_c",
        "train": ["chenzhe", "dengdeng", "han", "hui", "jiadi", "li", "saad", "xilai"],
        "val": ["rose"],
        "calib": ["mia"],
        "test": ["yunze"],
    },
}

STREAMS = {
    "joint": "j",
    "bone": "b",
}

DEFAULT_ACTIVITY_PKL_DIR = Path(
    "data/radar_v4/rerun/yolo26xpose/pyskl/activity_aligned"
)
DEFAULT_CONTINUOUS_PKL_DIR = Path(
    "data/radar_v4/rerun/yolo26xpose/pyskl/continuous_window_w60_s12"
)
DEFAULT_TRIANGULAR_PKL_DIR = Path(
    "data/radar_v4/rerun/yolo26xpose/pyskl/continuous_window_w60_s12_triangular"
)
DEFAULT_CONFIG_ROOT = Path("configs/stgcn++/stgcn++_radarv4/rerun/e1")
DEFAULT_JOB_ROOT = Path("rerun/e1/slurm")
DEFAULT_REPORT_PATH = Path("rerun/e1/generated_configs.json")


def load_pkl(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        return pickle.load(f)


def activity_pkl_path(activity_pkl_dir: Path, fold: str) -> Path:
    return activity_pkl_dir / f"radarv4_yolo26xpose_activity_aligned_fold_{fold}.pkl"


def continuous_pkl_path(continuous_pkl_dir: Path, fold: str) -> Path:
    return (
        continuous_pkl_dir
        / f"radarv4_yolo26xpose_continuous_window_w60_s12_fold_{fold}.pkl"
    )


def triangular_pkl_path(triangular_pkl_dir: Path, fold: str) -> Path:
    return (
        triangular_pkl_dir
        / f"radarv4_yolo26xpose_continuous_window_w60_s12_triangular_fold_{fold}.pkl"
    )


def annotations_by_frame_dir(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["frame_dir"]: item for item in data["annotations"]}


def split_count_by_label(pkl_path: Path, split_name: str) -> list[int]:
    data = load_pkl(pkl_path)
    lookup = annotations_by_frame_dir(data)
    counts = Counter(int(lookup[frame_dir]["label"]) for frame_dir in data["split"][split_name])
    return [counts.get(label_id, 0) for label_id in range(len(LABELS))]


def sqrt_expected_draws(counts: list[int], epoch_size: int) -> list[float]:
    weights = [count ** 0.5 if count > 0 else 0.0 for count in counts]
    total = sum(weights)
    if total <= 0:
        raise ValueError("Cannot compute square-root probabilities for empty counts.")
    return [weight / total * epoch_size for weight in weights]


def fold_comment(fold: str) -> str:
    spec = FOLDS[fold]
    return "\n".join(
        [
            "# Subject-wise outer split:",
            f"# Train subjects: {', '.join(spec['train'])}",
            f"# Validation subject: {', '.join(spec['val'])}",
            f"# Calibration subject: {', '.join(spec['calib'])}",
            f"# Test subject: {', '.join(spec['test'])}",
        ]
    )


def class_sampling_comment(counts: list[int], epoch_size: int) -> str:
    expected = sqrt_expected_draws(counts, epoch_size)
    lines = [
        "# Square-root training sampler:",
        "# Each sample i has weight a_i = 1 / sqrt(n_yi).",
        "# Sampling is with replacement; one epoch requests N_train samples.",
        "# DDP may add <= world_size - 1 padded samples internally for equal rank lengths.",
        f"# N_train = {epoch_size}",
        "# Natural train counts and expected requested draws per epoch:",
    ]
    for label_id, (label, count, draw_count) in enumerate(zip(LABELS, counts, expected)):
        lines.append(
            f"# {label_id}: {label}: n={count}, expected={draw_count:.1f}"
        )
    return "\n".join(lines)


def model_block() -> str:
    return """model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        in_channels=3,
        num_person=1,
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=9,
        in_channels=256,
        dropout=0.5,
        loss_cls=dict(type='CrossEntropyLoss')
    )
)
"""


def pipeline_block(stream_var: str = "stream", train_soft_label: bool = False) -> str:
    soft_label_step = ""
    train_tensor_keys = "['keypoint']"
    if train_soft_label:
        soft_label_step = (
            "    dict(type='UseSoftLabel', source_key='label_soft_triangular', "
            "num_classes=9),\n"
        )
        train_tensor_keys = "['keypoint', 'label']"

    return f"""# COCO-17 left/right keypoint ids
coco_left = [1, 3, 5, 7, 9, 11, 13, 15]
coco_right = [2, 4, 6, 8, 10, 12, 14, 16]

train_pipeline = [
    dict(
        type='Flip',
        flip_ratio=0.5,
        direction='horizontal',
        left_kp=coco_left,
        right_kp=coco_right
    ),
    dict(type='PreNormalize2D', mode='auto'),
    dict(type='GenSkeFeat', dataset='coco', feats=[{stream_var}]),
    dict(type='MonotonicUniformResample', clip_len=60),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
{soft_label_step}    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys={train_tensor_keys})
]

val_pipeline = [
    dict(type='PreNormalize2D', mode='auto'),
    dict(type='GenSkeFeat', dataset='coco', feats=[{stream_var}]),
    dict(type='MonotonicUniformResample', clip_len=60),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize2D', mode='auto'),
    dict(type='GenSkeFeat', dataset='coco', feats=[{stream_var}]),
    dict(type='MonotonicUniformResample', clip_len=60),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
"""


def runtime_block(work_dir: str) -> str:
    return f"""optimizer = dict(
    type='SGD',
    lr=0.05,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 20
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1,
    metrics=['top_k_accuracy', 'macro_f1'],
    metric_options=dict(top_k_accuracy=dict(topk=(1,))),
    save_best='macro_f1',
    rule='greater'
)
test_evaluation = dict(
    metrics=['top_k_accuracy', 'macro_f1'],
    metric_options=dict(top_k_accuracy=dict(topk=(1,)))
)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
load_from = None
resume_from = None
auto_resume = False
find_unused_parameters = False
work_dir = {work_dir!r}
"""


def a1_config_text(
    fold: str,
    stream_name: str,
    activity_pkl: Path,
    train_counts: list[int],
) -> str:
    stream = STREAMS[stream_name]
    work_dir = f"./work_dirs/rerun/e1/fold_{fold}/{stream_name}/a1_activity_aligned"
    epoch_size = sum(train_counts)
    return (
        "# ============================================================\n"
        "# Rerun E1 A1: activity-aligned training and activity-aligned evaluation\n"
        f"# Fold {fold.upper()}, stream {stream_name}\n"
        "# Generated by rerun/e1/generate_stgcnpp_configs.py\n"
        "# ============================================================\n\n"
        f"stream = {stream!r}\n"
        "dataset_type = 'PoseDataset'\n"
        f"ann_file = {activity_pkl.as_posix()!r}\n\n"
        + model_block()
        + "\n"
        + fold_comment(fold)
        + "\n"
        + class_sampling_comment(train_counts, epoch_size)
        + "\n"
        + f"class_sample_strategy = 'sqrt'\nclass_sample_power = 0.5\nepoch_size = {epoch_size}\n\n"
        + pipeline_block()
        + "\n"
        + runtime_block(work_dir)
        + "\n"
        + "sampler_indices_output_dir = f'{work_dir}/sampler_indices'\n\n"
        + """data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    persistent_workers=False,
    train_dataloader=dict(
        sampler_indices_output_dir=sampler_indices_output_dir,
        sampler_indices_output_prefix='sqrt_sampler'
    ),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline,
        split='train',
        class_sample_strategy=class_sample_strategy,
        class_sample_power=class_sample_power,
        epoch_size=epoch_size
    ),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test')
)
"""
    )


def b_config_text(
    fold: str,
    stream_name: str,
    continuous_pkl: Path,
    train_counts: list[int],
) -> str:
    stream = STREAMS[stream_name]
    work_dir = f"./work_dirs/rerun/e1/fold_{fold}/{stream_name}/b_continuous_window"
    epoch_size = sum(train_counts)
    return (
        "# ============================================================\n"
        "# Rerun E1 B: continuous-window training and continuous-window evaluation\n"
        f"# Fold {fold.upper()}, stream {stream_name}\n"
        "# A2 uses this config for continuous-window evaluation of the A1 checkpoint.\n"
        "# Generated by rerun/e1/generate_stgcnpp_configs.py\n"
        "# ============================================================\n\n"
        f"stream = {stream!r}\n"
        "dataset_type = 'PoseDataset'\n"
        f"ann_file = {continuous_pkl.as_posix()!r}\n\n"
        + model_block()
        + "\n"
        + fold_comment(fold)
        + "\n"
        + class_sampling_comment(train_counts, epoch_size)
        + "\n"
        + f"class_sample_strategy = 'sqrt'\nclass_sample_power = 0.5\nepoch_size = {epoch_size}\n\n"
        + pipeline_block()
        + "\n"
        + runtime_block(work_dir)
        + "\n"
        + "sampler_indices_output_dir = f'{work_dir}/sampler_indices'\n\n"
        + """data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    persistent_workers=False,
    train_dataloader=dict(
        sampler_indices_output_dir=sampler_indices_output_dir,
        sampler_indices_output_prefix='sqrt_sampler'
    ),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline,
        split='train',
        class_sample_strategy=class_sample_strategy,
        class_sample_power=class_sample_power,
        epoch_size=epoch_size
    ),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test')
)
"""
    )


def c_config_text(
    fold: str,
    stream_name: str,
    triangular_pkl: Path,
    train_counts: list[int],
) -> str:
    stream = STREAMS[stream_name]
    work_dir = (
        f"./work_dirs/rerun/e1/fold_{fold}/{stream_name}/"
        "c_triangular_temporal_composition"
    )
    epoch_size = sum(train_counts)
    return (
        "# ============================================================\n"
        "# Rerun E1 C: continuous-window training with triangular temporal-composition targets\n"
        "# Evaluation keeps the hard center-label continuous-window protocol.\n"
        f"# Fold {fold.upper()}, stream {stream_name}\n"
        "# Generated by rerun/e1/generate_stgcnpp_configs.py\n"
        "# ============================================================\n\n"
        f"stream = {stream!r}\n"
        "dataset_type = 'PoseDataset'\n"
        f"ann_file = {triangular_pkl.as_posix()!r}\n\n"
        + model_block()
        + "\n"
        + fold_comment(fold)
        + "\n"
        + class_sampling_comment(train_counts, epoch_size)
        + "\n"
        + "# The sampler uses hard center labels from `label`; only the training\n"
        + "# pipeline replaces `label` with `label_soft_triangular` for the loss.\n"
        + f"class_sample_strategy = 'sqrt'\nclass_sample_power = 0.5\nepoch_size = {epoch_size}\n\n"
        + pipeline_block(train_soft_label=True)
        + "\n"
        + runtime_block(work_dir)
        + "\n"
        + "sampler_indices_output_dir = f'{work_dir}/sampler_indices'\n\n"
        + """data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    persistent_workers=False,
    train_dataloader=dict(
        sampler_indices_output_dir=sampler_indices_output_dir,
        sampler_indices_output_prefix='sqrt_sampler'
    ),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline,
        split='train',
        class_sample_strategy=class_sample_strategy,
        class_sample_power=class_sample_power,
        epoch_size=epoch_size
    ),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test')
)
"""
    )


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def slurm_header(condition: str, fold: str, stream: str, time_limit: str) -> str:
    return f"""#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time={time_limit}
#SBATCH --job-name=e1_{condition}_fold_{fold}_{stream}
#SBATCH --output=rerun/e1/slurm/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

GPUS="${{GPUS:-4}}"
SEED="${{SEED:-42}}"
FOLD="{fold}"
STREAM="{stream}"
CONFIG_ROOT="configs/stgcn++/stgcn++_radarv4/rerun/e1/fold_${{FOLD}}"
WORK_ROOT="work_dirs/rerun/e1/fold_${{FOLD}}"
"""


def a1_a2_job_text(fold: str, stream: str) -> str:
    return slurm_header("a1_a2", fold, stream, "04:00:00") + f"""
a1_config="${{CONFIG_ROOT}}/${{STREAM}}/a1_activity_aligned.py"
b_config="${{CONFIG_ROOT}}/${{STREAM}}/b_continuous_window.py"
a1_work_dir="${{WORK_ROOT}}/${{STREAM}}/a1_activity_aligned"
a2_eval_dir="${{WORK_ROOT}}/${{STREAM}}/a2_activity_checkpoint_on_continuous"

bash tools/dist_train.sh "${{a1_config}}" "${{GPUS}}" --validate --test-best --seed "${{SEED}}" --deterministic

mapfile -t best_ckpts < <(find "${{a1_work_dir}}" -maxdepth 1 -name 'best_macro_f1_epoch_*.pth' | sort)
if [[ "${{#best_ckpts[@]}}" -ne 1 ]]; then
  echo "[ERROR] Expected one best_macro_f1 checkpoint in ${{a1_work_dir}}, found ${{#best_ckpts[@]}}" >&2
  printf '%s\\n' "${{best_ckpts[@]}}" >&2
  exit 1
fi
best_ckpt="${{best_ckpts[0]}}"

mkdir -p "${{a2_eval_dir}}"
bash tools/dist_test.sh "${{b_config}}" "${{best_ckpt}}" "${{GPUS}}" \\
  --out "${{a2_eval_dir}}/best_pred.pkl" \\
  --eval-out "${{a2_eval_dir}}/best_eval.json"
"""


def b_job_text(fold: str, stream: str) -> str:
    return slurm_header("b", fold, stream, "06:00:00") + f"""
b_config="${{CONFIG_ROOT}}/${{STREAM}}/b_continuous_window.py"

bash tools/dist_train.sh "${{b_config}}" "${{GPUS}}" --validate --test-best --seed "${{SEED}}" --deterministic
"""


def c_job_text(fold: str, stream: str) -> str:
    return slurm_header("c", fold, stream, "06:00:00") + f"""
c_config="${{CONFIG_ROOT}}/${{STREAM}}/c_triangular_temporal_composition.py"

bash tools/dist_train.sh "${{c_config}}" "${{GPUS}}" --validate --test-best --seed "${{SEED}}" --deterministic
"""


def generate(
    folds: list[str],
    streams: list[str],
    activity_pkl_dir: Path,
    continuous_pkl_dir: Path,
    triangular_pkl_dir: Path,
    config_root: Path,
    job_root: Path,
    report_path: Path,
    overwrite: bool,
) -> None:
    outputs: dict[str, Any] = {
        "configs": [],
        "jobs": [],
        "a1_a2_jobs": [],
        "b_jobs": [],
        "c_jobs": [],
        "deleted_legacy_configs": [],
        "deleted_legacy_jobs": [],
        "folds": {},
    }
    for legacy_job_path in job_root.glob("run_fold_*.sh"):
        legacy_job_path.unlink()
        outputs["deleted_legacy_jobs"].append(str(legacy_job_path))

    for fold in folds:
        if fold not in FOLDS:
            raise ValueError(f"Unknown fold {fold!r}; expected one of {sorted(FOLDS)}")
        activity_pkl = activity_pkl_path(activity_pkl_dir, fold)
        continuous_pkl = continuous_pkl_path(continuous_pkl_dir, fold)
        triangular_pkl = triangular_pkl_path(triangular_pkl_dir, fold)
        activity_train_counts = split_count_by_label(activity_pkl, "train")
        continuous_train_counts = split_count_by_label(continuous_pkl, "train")
        triangular_train_counts = split_count_by_label(triangular_pkl, "train")
        if triangular_train_counts != continuous_train_counts:
            raise RuntimeError(
                f"Fold {fold}: C triangular hard-label train counts do not "
                "match B continuous train counts."
            )

        outputs["folds"][fold] = {
            "activity_pkl": str(activity_pkl),
            "continuous_pkl": str(continuous_pkl),
            "triangular_pkl": str(triangular_pkl),
            "activity_train_counts": dict(zip(LABELS, activity_train_counts)),
            "activity_epoch_size": sum(activity_train_counts),
            "continuous_train_counts": dict(zip(LABELS, continuous_train_counts)),
            "continuous_epoch_size": sum(continuous_train_counts),
            "triangular_train_counts": dict(zip(LABELS, triangular_train_counts)),
            "triangular_epoch_size": sum(triangular_train_counts),
        }

        for stream in streams:
            if stream not in STREAMS:
                raise ValueError(
                    f"Unknown stream {stream!r}; expected one of {sorted(STREAMS)}"
                )
            stream_dir = config_root / f"fold_{fold}" / stream
            a1_path = stream_dir / "a1_activity_aligned.py"
            b_path = stream_dir / "b_continuous_window.py"
            c_path = stream_dir / "c_triangular_temporal_composition.py"
            legacy_a2_path = stream_dir / "a2_continuous_window.py"
            if legacy_a2_path.exists():
                legacy_a2_path.unlink()
                outputs["deleted_legacy_configs"].append(str(legacy_a2_path))
            write_text(
                a1_path,
                a1_config_text(fold, stream, activity_pkl, activity_train_counts),
                overwrite=overwrite,
            )
            write_text(
                b_path,
                b_config_text(fold, stream, continuous_pkl, continuous_train_counts),
                overwrite=overwrite,
            )
            write_text(
                c_path,
                c_config_text(fold, stream, triangular_pkl, triangular_train_counts),
                overwrite=overwrite,
            )
            outputs["configs"].extend([str(a1_path), str(b_path), str(c_path)])

        for stream in streams:
            a_job_path = job_root / f"run_a1_a2_fold_{fold}_{stream}.sh"
            b_job_path = job_root / f"run_b_fold_{fold}_{stream}.sh"
            c_job_path = job_root / f"run_c_fold_{fold}_{stream}.sh"
            write_text(a_job_path, a1_a2_job_text(fold, stream), overwrite=overwrite)
            write_text(b_job_path, b_job_text(fold, stream), overwrite=overwrite)
            write_text(c_job_path, c_job_text(fold, stream), overwrite=overwrite)
            outputs["jobs"].extend([str(a_job_path), str(b_job_path), str(c_job_path)])
            outputs["a1_a2_jobs"].append(str(a_job_path))
            outputs["b_jobs"].append(str(b_job_path))
            outputs["c_jobs"].append(str(c_job_path))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", default=["a", "b", "c"])
    parser.add_argument("--streams", nargs="+", default=["joint", "bone"])
    parser.add_argument("--activity-pkl-dir", type=Path, default=DEFAULT_ACTIVITY_PKL_DIR)
    parser.add_argument("--continuous-pkl-dir", type=Path, default=DEFAULT_CONTINUOUS_PKL_DIR)
    parser.add_argument("--triangular-pkl-dir", type=Path, default=DEFAULT_TRIANGULAR_PKL_DIR)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(
        folds=[fold.lower() for fold in args.folds],
        streams=[stream.lower() for stream in args.streams],
        activity_pkl_dir=args.activity_pkl_dir,
        continuous_pkl_dir=args.continuous_pkl_dir,
        triangular_pkl_dir=args.triangular_pkl_dir,
        config_root=args.config_root,
        job_root=args.job_root,
        report_path=args.report_path,
        overwrite=args.overwrite,
    )
    print(f"[DONE] wrote E1 configs under {args.config_root}")
    print(f"[DONE] wrote E1 SLURM jobs under {args.job_root}")
    print(f"[DONE] wrote generation report to {args.report_path}")


if __name__ == "__main__":
    main()

"""Generate pseudo-labeling inner-teacher split pkls, configs, and SLURM jobs.

The inner teachers reuse the E1-B continuous-window protocol:

- ST-GCN++
- joint and bone streams
- hard center-label continuous-window training
- GCNHead(dropout=0.5)
- online square-root training sampler
- validation checkpoint selection by center macro-F1

For each outer fold, four inner teachers train on six outer-training subjects
and predict the remaining two outer-training subjects as ``pseudo_target``.
The outer validation subject remains the validation split for checkpoint
selection, the outer calibration subject is preserved as ``calib`` for later
temperature fitting, and the outer test subject is excluded entirely.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
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
        "train_pool": ["chenzhe", "dengdeng", "hui", "jiadi", "mia", "rose", "xilai", "yunze"],
        "val": ["han"],
        "calib": ["saad"],
        "test": ["li"],
        "teachers": {
            "t1": {
                "train": ["dengdeng", "jiadi", "mia", "rose", "xilai", "yunze"],
                "pseudo_target": ["hui", "chenzhe"],
                "expected_train_windows": 32482,
                "expected_pseudo_target_windows": 10811,
            },
            "t2": {
                "train": ["chenzhe", "dengdeng", "hui", "jiadi", "xilai", "yunze"],
                "pseudo_target": ["rose", "mia"],
                "expected_train_windows": 32105,
                "expected_pseudo_target_windows": 11188,
            },
            "t3": {
                "train": ["chenzhe", "hui", "mia", "rose", "xilai", "yunze"],
                "pseudo_target": ["dengdeng", "jiadi"],
                "expected_train_windows": 33190,
                "expected_pseudo_target_windows": 10103,
            },
            "t4": {
                "train": ["chenzhe", "dengdeng", "hui", "jiadi", "mia", "rose"],
                "pseudo_target": ["yunze", "xilai"],
                "expected_train_windows": 32102,
                "expected_pseudo_target_windows": 11191,
            },
        },
    },
    "b": {
        "train_pool": ["han", "jiadi", "li", "mia", "rose", "saad", "xilai", "yunze"],
        "val": ["hui"],
        "calib": ["chenzhe"],
        "test": ["dengdeng"],
        "teachers": {
            "t1": {
                "train": ["han", "jiadi", "rose", "saad", "xilai", "yunze"],
                "pseudo_target": ["li", "mia"],
                "expected_train_windows": 32725,
                "expected_pseudo_target_windows": 11017,
            },
            "t2": {
                "train": ["han", "jiadi", "li", "mia", "saad", "yunze"],
                "pseudo_target": ["rose", "xilai"],
                "expected_train_windows": 33010,
                "expected_pseudo_target_windows": 10732,
            },
            "t3": {
                "train": ["han", "jiadi", "li", "mia", "rose", "xilai"],
                "pseudo_target": ["yunze", "saad"],
                "expected_train_windows": 32684,
                "expected_pseudo_target_windows": 11058,
            },
            "t4": {
                "train": ["li", "mia", "rose", "saad", "xilai", "yunze"],
                "pseudo_target": ["jiadi", "han"],
                "expected_train_windows": 32807,
                "expected_pseudo_target_windows": 10935,
            },
        },
    },
    "c": {
        "train_pool": ["chenzhe", "dengdeng", "han", "hui", "jiadi", "li", "saad", "xilai"],
        "val": ["rose"],
        "calib": ["mia"],
        "test": ["yunze"],
        "teachers": {
            "t1": {
                "train": ["dengdeng", "han", "jiadi", "li", "saad", "xilai"],
                "pseudo_target": ["hui", "chenzhe"],
                "expected_train_windows": 32524,
                "expected_pseudo_target_windows": 10811,
            },
            "t2": {
                "train": ["chenzhe", "dengdeng", "han", "hui", "jiadi", "saad"],
                "pseudo_target": ["li", "xilai"],
                "expected_train_windows": 32774,
                "expected_pseudo_target_windows": 10561,
            },
            "t3": {
                "train": ["chenzhe", "han", "hui", "jiadi", "li", "xilai"],
                "pseudo_target": ["dengdeng", "saad"],
                "expected_train_windows": 32307,
                "expected_pseudo_target_windows": 11028,
            },
            "t4": {
                "train": ["chenzhe", "dengdeng", "hui", "li", "saad", "xilai"],
                "pseudo_target": ["jiadi", "han"],
                "expected_train_windows": 32400,
                "expected_pseudo_target_windows": 10935,
            },
        },
    },
}

STREAMS = {
    "joint": "j",
    "bone": "b",
}

DEFAULT_SOURCE_PKL_DIR = Path(
    "data/radar_v4/rerun/yolo26xpose/pyskl/continuous_window_w60_s12"
)
DEFAULT_OUTPUT_PKL_DIR = Path(
    "data/radar_v4/rerun/yolo26xpose/pyskl/inner_teachers_continuous_window_w60_s12"
)
DEFAULT_CONFIG_ROOT = Path(
    "configs/stgcn++/stgcn++_radarv4/rerun/pseudo_labeling/inner_teachers"
)
DEFAULT_JOB_ROOT = Path("rerun/pseudo_labeling/slurm/inner_teachers")
DEFAULT_REPORT_DIR = Path("rerun/pseudo_labeling/reports")
DEFAULT_MANIFEST = Path("rerun/pseudo_labeling/generated_inner_teacher_artifacts.json")


@dataclass(frozen=True)
class InnerArtifact:
    fold: str
    teacher: str
    stream: str
    pkl_path: Path
    config_path: Path
    job_path: Path
    work_dir: str
    train_subjects: list[str]
    pseudo_target_subjects: list[str]
    val_subjects: list[str]
    calib_subjects: list[str]
    outer_test_subjects: list[str]
    train_windows: int
    pseudo_target_windows: int
    val_windows: int
    calib_windows: int
    train_counts: list[int]
    pseudo_target_counts: list[int]


def install_numpy_pickle_compat_aliases() -> None:
    try:
        importlib.import_module("numpy._core.numeric")
        return
    except ModuleNotFoundError:
        pass

    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }
    for alias, target in aliases.items():
        try:
            sys.modules.setdefault(alias, importlib.import_module(target))
        except ModuleNotFoundError:
            continue


def load_pkl(path: Path) -> dict[str, Any]:
    install_numpy_pickle_compat_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def source_pkl_path(source_pkl_dir: Path, fold: str) -> Path:
    return (
        source_pkl_dir
        / f"radarv4_yolo26xpose_continuous_window_w60_s12_fold_{fold}.pkl"
    )


def inner_pkl_path(output_pkl_dir: Path, fold: str, teacher: str, stream: str) -> Path:
    return (
        output_pkl_dir
        / f"radarv4_yolo26xpose_continuous_window_w60_s12_inner_fold_{fold}_{teacher}_{stream}.pkl"
    )


def config_path(config_root: Path, fold: str, teacher: str, stream: str) -> Path:
    return config_root / f"fold_{fold}" / teacher / f"{stream}.py"


def job_path(job_root: Path, fold: str, teacher: str, stream: str) -> Path:
    return job_root / f"run_inner_fold_{fold}_{teacher}_{stream}.sh"


def work_dir_for(fold: str, teacher: str, stream: str) -> str:
    return f"./work_dirs/rerun/pseudo_labeling/inner_teachers/fold_{fold}/{teacher}/{stream}"


def annotations_by_subject(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for item in data["annotations"]:
        by_subject.setdefault(str(item["subject"]).lower(), []).append(item)
    return by_subject


def frame_dirs_for_subjects(
    by_subject: dict[str, list[dict[str, Any]]],
    subjects: list[str],
) -> list[str]:
    subject_set = {subject.lower() for subject in subjects}
    return [
        str(item["frame_dir"])
        for subject in subjects
        for item in by_subject.get(subject.lower(), [])
        if str(item["subject"]).lower() in subject_set
    ]


def label_counts_for_subjects(
    by_subject: dict[str, list[dict[str, Any]]],
    subjects: list[str],
) -> list[int]:
    counts = Counter()
    for subject in subjects:
        for item in by_subject.get(subject.lower(), []):
            counts[int(item["label"])] += 1
    return [counts.get(label_id, 0) for label_id in range(len(LABELS))]


def sqrt_expected_draws(counts: list[int], epoch_size: int) -> list[float]:
    weights = [count ** 0.5 if count > 0 else 0.0 for count in counts]
    total = sum(weights)
    if total <= 0:
        raise ValueError("Cannot compute square-root probabilities for empty counts.")
    return [weight / total * epoch_size for weight in weights]


def validate_inner_spec(fold: str) -> None:
    fold_spec = FOLDS[fold]
    train_pool = set(fold_spec["train_pool"])
    covered: list[str] = []
    for teacher, teacher_spec in fold_spec["teachers"].items():
        train_subjects = set(teacher_spec["train"])
        target_subjects = set(teacher_spec["pseudo_target"])
        if len(train_subjects) != 6:
            raise ValueError(f"Fold {fold} {teacher}: expected six train subjects")
        if len(target_subjects) != 2:
            raise ValueError(f"Fold {fold} {teacher}: expected two pseudo-target subjects")
        if train_subjects | target_subjects != train_pool:
            raise ValueError(f"Fold {fold} {teacher}: train + pseudo target does not equal U_f")
        if train_subjects & target_subjects:
            raise ValueError(f"Fold {fold} {teacher}: train/target subjects overlap")
        covered.extend(teacher_spec["pseudo_target"])
    if set(covered) != train_pool or len(covered) != len(set(covered)):
        raise ValueError(f"Fold {fold}: pseudo-target pairs do not partition U_f")


def artifact_for(
    output_pkl_dir: Path,
    config_root: Path,
    job_root: Path,
    data: dict[str, Any],
    fold: str,
    teacher: str,
    stream: str,
) -> InnerArtifact:
    fold_spec = FOLDS[fold]
    teacher_spec = fold_spec["teachers"][teacher]
    by_subject = annotations_by_subject(data)

    train_subjects = list(teacher_spec["train"])
    pseudo_subjects = list(teacher_spec["pseudo_target"])
    val_subjects = list(fold_spec["val"])
    calib_subjects = list(fold_spec["calib"])
    outer_test_subjects = list(fold_spec["test"])

    train_count = sum(len(by_subject[subject]) for subject in train_subjects)
    pseudo_count = sum(len(by_subject[subject]) for subject in pseudo_subjects)
    if train_count != int(teacher_spec["expected_train_windows"]):
        raise RuntimeError(
            f"Fold {fold} {teacher}: train count {train_count} does not match "
            f"expected {teacher_spec['expected_train_windows']}"
        )
    if pseudo_count != int(teacher_spec["expected_pseudo_target_windows"]):
        raise RuntimeError(
            f"Fold {fold} {teacher}: pseudo-target count {pseudo_count} does not "
            f"match expected {teacher_spec['expected_pseudo_target_windows']}"
        )

    return InnerArtifact(
        fold=fold,
        teacher=teacher,
        stream=stream,
        pkl_path=inner_pkl_path(output_pkl_dir, fold, teacher, stream),
        config_path=config_path(config_root, fold, teacher, stream),
        job_path=job_path(job_root, fold, teacher, stream),
        work_dir=work_dir_for(fold, teacher, stream),
        train_subjects=train_subjects,
        pseudo_target_subjects=pseudo_subjects,
        val_subjects=val_subjects,
        calib_subjects=calib_subjects,
        outer_test_subjects=outer_test_subjects,
        train_windows=train_count,
        pseudo_target_windows=pseudo_count,
        val_windows=sum(len(by_subject[subject]) for subject in val_subjects),
        calib_windows=sum(len(by_subject[subject]) for subject in calib_subjects),
        train_counts=label_counts_for_subjects(by_subject, train_subjects),
        pseudo_target_counts=label_counts_for_subjects(by_subject, pseudo_subjects),
    )


def inner_role(subject: str, artifact: InnerArtifact) -> str:
    value = subject.lower()
    if value in set(artifact.train_subjects):
        return "train"
    if value in set(artifact.pseudo_target_subjects):
        return "pseudo_target"
    if value in set(artifact.val_subjects):
        return "val"
    if value in set(artifact.calib_subjects):
        return "calib"
    raise ValueError(f"Subject {subject!r} is not included in {artifact.fold}/{artifact.teacher}")


def build_inner_pkl(data: dict[str, Any], artifact: InnerArtifact) -> dict[str, Any]:
    included_subjects = (
        set(artifact.train_subjects)
        | set(artifact.pseudo_target_subjects)
        | set(artifact.val_subjects)
        | set(artifact.calib_subjects)
    )
    forbidden_subjects = set(artifact.outer_test_subjects)
    annotations = []
    split = {
        "train": [],
        "val": [],
        "calib": [],
        "pseudo_target": [],
    }

    for item in data["annotations"]:
        subject = str(item["subject"]).lower()
        if subject in forbidden_subjects:
            continue
        if subject not in included_subjects:
            continue
        role = inner_role(subject, artifact)
        frame_dir = str(item["frame_dir"])
        split[role].append(frame_dir)
        new_item = dict(item)
        new_item["outer_fold"] = artifact.fold
        new_item["inner_teacher"] = artifact.teacher
        new_item["inner_stream"] = artifact.stream
        new_item["inner_split_role"] = role
        new_item["outer_test_excluded"] = True
        annotations.append(new_item)

    actual = {
        "train": len(split["train"]),
        "val": len(split["val"]),
        "calib": len(split["calib"]),
        "pseudo_target": len(split["pseudo_target"]),
    }
    expected = {
        "train": artifact.train_windows,
        "val": artifact.val_windows,
        "calib": artifact.calib_windows,
        "pseudo_target": artifact.pseudo_target_windows,
    }
    if actual != expected:
        raise RuntimeError(
            f"Split counts for {artifact.fold}/{artifact.teacher}/{artifact.stream} "
            f"do not match expected values: actual={actual}, expected={expected}"
        )

    return {
        "split": split,
        "annotations": annotations,
        "metadata": {
            "protocol": "pseudo_labeling_inner_teacher_continuous_window_w60_s12",
            "source_protocol": "e1_b_continuous_window_w60_s12",
            "outer_fold": artifact.fold,
            "inner_teacher": artifact.teacher,
            "stream": artifact.stream,
            "train_subjects": artifact.train_subjects,
            "pseudo_target_subjects": artifact.pseudo_target_subjects,
            "val_subjects": artifact.val_subjects,
            "calib_subjects": artifact.calib_subjects,
            "outer_test_subjects_excluded": artifact.outer_test_subjects,
            "train_windows": artifact.train_windows,
            "pseudo_target_windows": artifact.pseudo_target_windows,
            "val_windows": artifact.val_windows,
            "calib_windows": artifact.calib_windows,
            "label_map": dict(enumerate(LABELS)),
        },
    }


def write_inner_pkl(
    data: dict[str, Any],
    artifact: InnerArtifact,
    overwrite: bool,
) -> int:
    if artifact.pkl_path.exists() and not overwrite:
        raise FileExistsError(f"{artifact.pkl_path} exists; pass --overwrite to replace it.")
    payload = build_inner_pkl(data, artifact)
    artifact.pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact.pkl_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return artifact.pkl_path.stat().st_size


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
        lines.append(f"# {label_id}: {label}: n={count}, expected={draw_count:.1f}")
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


def pipeline_block() -> str:
    return """# COCO-17 left/right keypoint ids
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
    dict(type='GenSkeFeat', dataset='coco', feats=[stream]),
    dict(type='MonotonicUniformResample', clip_len=60),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize2D', mode='auto'),
    dict(type='GenSkeFeat', dataset='coco', feats=[stream]),
    dict(type='MonotonicUniformResample', clip_len=60),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize2D', mode='auto'),
    dict(type='GenSkeFeat', dataset='coco', feats=[stream]),
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


def config_text(artifact: InnerArtifact) -> str:
    stream_code = STREAMS[artifact.stream]
    return (
        "# ============================================================\n"
        "# Rerun pseudo-labeling inner teacher: direct continuous-window ST-GCN++\n"
        f"# Outer fold {artifact.fold.upper()}, teacher {artifact.teacher.upper()}, stream {artifact.stream}\n"
        "# Generated by rerun/pseudo_labeling/generate_inner_teacher_training_artifacts.py\n"
        "# ============================================================\n\n"
        f"stream = {stream_code!r}\n"
        "dataset_type = 'PoseDataset'\n"
        f"ann_file = {artifact.pkl_path.as_posix()!r}\n\n"
        + model_block()
        + "\n"
        + "# Fixed outer roles:\n"
        + f"# Outer validation subject: {', '.join(artifact.val_subjects)}\n"
        + f"# Outer calibration subject: {', '.join(artifact.calib_subjects)}\n"
        + f"# Outer test subject excluded: {', '.join(artifact.outer_test_subjects)}\n"
        + "# Inner cross-fitting roles:\n"
        + f"# Train subjects: {', '.join(artifact.train_subjects)}\n"
        + f"# Pseudo-target subjects: {', '.join(artifact.pseudo_target_subjects)}\n"
        + f"# Pseudo-target windows: {artifact.pseudo_target_windows}\n"
        + "# `data.test` intentionally uses split='pseudo_target' for --test-best.\n"
        + "# The outer test subject is not present in this pkl.\n"
        + class_sampling_comment(artifact.train_counts, artifact.train_windows)
        + "\n"
        + "class_sample_strategy = 'sqrt'\n"
        + "class_sample_power = 0.5\n"
        + f"epoch_size = {artifact.train_windows}\n\n"
        + pipeline_block()
        + "\n"
        + runtime_block(artifact.work_dir)
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
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='pseudo_target')
)
"""
    )


def slurm_job_text(artifact: InnerArtifact) -> str:
    return f"""#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=02:00:00
#SBATCH --job-name=pl_inner_{artifact.fold}_{artifact.teacher}_{artifact.stream}
#SBATCH --output=rerun/pseudo_labeling/slurm/inner_teachers/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

GPUS="${{GPUS:-4}}"
SEED="${{SEED:-42}}"
CONFIG="{artifact.config_path.as_posix()}"

bash tools/dist_train.sh "${{CONFIG}}" "${{GPUS}}" --validate --test-best --seed "${{SEED}}" --deterministic
"""


def submit_script_text(artifacts: list[InnerArtifact]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'cd "$(dirname "$0")/../../../.."',
        "",
        "# Submit one job per inner-teacher stream.",
    ]
    for artifact in artifacts:
        lines.append(f"sbatch {artifact.job_path.as_posix()}")
    lines.append("")
    return "\n".join(lines)


def artifact_row(artifact: InnerArtifact, pkl_size_bytes: int | None) -> dict[str, Any]:
    row = {
        "fold": artifact.fold,
        "teacher": artifact.teacher,
        "stream": artifact.stream,
        "pkl_path": str(artifact.pkl_path),
        "config_path": str(artifact.config_path),
        "job_path": str(artifact.job_path),
        "work_dir": artifact.work_dir,
        "train_subjects": ",".join(artifact.train_subjects),
        "pseudo_target_subjects": ",".join(artifact.pseudo_target_subjects),
        "val_subjects": ",".join(artifact.val_subjects),
        "calib_subjects": ",".join(artifact.calib_subjects),
        "outer_test_subjects_excluded": ",".join(artifact.outer_test_subjects),
        "train_windows": artifact.train_windows,
        "pseudo_target_windows": artifact.pseudo_target_windows,
        "val_windows": artifact.val_windows,
        "calib_windows": artifact.calib_windows,
        "epoch_size": artifact.train_windows,
        "pkl_size_bytes": "" if pkl_size_bytes is None else pkl_size_bytes,
    }
    for label_id, label in enumerate(LABELS):
        row[f"train_count_{label_id}_{label}"] = artifact.train_counts[label_id]
        row[f"pseudo_target_count_{label_id}_{label}"] = artifact.pseudo_target_counts[label_id]
    return row


def free_space_bytes(path: Path) -> int | None:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    if not probe.exists():
        return None
    return shutil.disk_usage(probe).free


def generate(args: argparse.Namespace) -> None:
    folds = [fold.lower() for fold in args.folds]
    streams = [stream.lower() for stream in args.streams]
    for fold in folds:
        if fold not in FOLDS:
            raise ValueError(f"Unknown fold {fold!r}; expected one of {sorted(FOLDS)}")
        validate_inner_spec(fold)
    for stream in streams:
        if stream not in STREAMS:
            raise ValueError(f"Unknown stream {stream!r}; expected one of {sorted(STREAMS)}")

    free_bytes = free_space_bytes(args.output_pkl_dir)
    if not args.skip_pkl_write and free_bytes is not None and free_bytes < args.min_free_bytes:
        raise RuntimeError(
            f"Refusing to write full inner pkls with only {free_bytes / 1024**3:.2f} GiB free "
            f"under {args.output_pkl_dir}. Re-run on storage with enough space, or pass "
            "--skip-pkl-write to generate configs/jobs/count reports only."
        )

    artifacts: list[InnerArtifact] = []
    rows: list[dict[str, Any]] = []
    pkl_status: dict[str, Any] = {}

    for fold in folds:
        source_path = source_pkl_path(args.source_pkl_dir, fold)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        data = load_pkl(source_path)
        for teacher in sorted(FOLDS[fold]["teachers"]):
            for stream in streams:
                artifact = artifact_for(
                    output_pkl_dir=args.output_pkl_dir,
                    config_root=args.config_root,
                    job_root=args.job_root,
                    data=data,
                    fold=fold,
                    teacher=teacher,
                    stream=stream,
                )
                artifacts.append(artifact)

                pkl_size: int | None = None
                if args.skip_pkl_write:
                    pkl_status[str(artifact.pkl_path)] = "skipped"
                else:
                    pkl_size = write_inner_pkl(data, artifact, overwrite=args.overwrite)
                    pkl_status[str(artifact.pkl_path)] = "written"

                write_text(artifact.config_path, config_text(artifact), overwrite=args.overwrite)
                write_text(artifact.job_path, slurm_job_text(artifact), overwrite=args.overwrite)
                rows.append(artifact_row(artifact, pkl_size))

    submit_path = args.job_root / "submit_inner_teacher_jobs.sh"
    write_text(submit_path, submit_script_text(artifacts), overwrite=args.overwrite)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "inner_teacher_split_counts.csv", rows)
    write_json(
        args.manifest,
        {
            "protocol": "pseudo_labeling_inner_teacher_continuous_window_w60_s12",
            "source_pkl_dir": str(args.source_pkl_dir),
            "output_pkl_dir": str(args.output_pkl_dir),
            "config_root": str(args.config_root),
            "job_root": str(args.job_root),
            "report_dir": str(args.report_dir),
            "skip_pkl_write": bool(args.skip_pkl_write),
            "artifact_count": len(artifacts),
            "pkl_status": pkl_status,
            "submit_script": str(submit_path),
            "rows": rows,
            "checkpoint_tie_behavior": (
                "No custom pseudo-labeling tie-breaker is added. The config keeps E1-B's "
                "MMCV/PYSKL save_best='macro_f1', rule='greater' behavior. "
                "Standard MMCV EvalHook treats a strictly greater score as a "
                "new best; exact ties do not intentionally replace the existing "
                "best. If multiple best checkpoint files remain, PYSKL's "
                "--test-best fallback selects the one with the largest epoch id."
            ),
        },
    )

    print(f"[DONE] artifacts described: {len(artifacts)}")
    print(f"[DONE] wrote configs under {args.config_root}")
    print(f"[DONE] wrote jobs under {args.job_root}")
    print(f"[DONE] wrote counts to {args.report_dir / 'inner_teacher_split_counts.csv'}")
    print(f"[DONE] wrote manifest to {args.manifest}")
    if args.skip_pkl_write:
        print("[INFO] skipped pkl writing by request")
    else:
        print(f"[DONE] wrote pkls under {args.output_pkl_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", default=["a", "b", "c"])
    parser.add_argument("--streams", nargs="+", default=["joint", "bone"])
    parser.add_argument("--source-pkl-dir", type=Path, default=DEFAULT_SOURCE_PKL_DIR)
    parser.add_argument("--output-pkl-dir", type=Path, default=DEFAULT_OUTPUT_PKL_DIR)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-pkl-write",
        action="store_true",
        help="Generate configs/jobs/count reports without writing large pkl files.",
    )
    parser.add_argument(
        "--min-free-bytes",
        type=int,
        default=20 * 1024**3,
        help="Safety threshold before writing the full pkl set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        generate(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

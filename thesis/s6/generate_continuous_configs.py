"""Generate S6 continuous hard-adaptation configs for teacher-4 PoseC3D."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from .common import (
        DEFAULT_CONTINUOUS_CONFIG_DIR,
        DEFAULT_CONTINUOUS_TEACHER_PKL,
        DEFAULT_CONTINUOUS_WORK_ROOT,
        DEFAULT_TRIMMED_WORK_ROOT,
        LABELS,
        TeacherSpec,
        continuous_config_path,
        continuous_work_dir,
        read_json,
        selected_specs,
        trimmed_work_dir,
        write_json,
    )
except ImportError:
    from common import (
        DEFAULT_CONTINUOUS_CONFIG_DIR,
        DEFAULT_CONTINUOUS_TEACHER_PKL,
        DEFAULT_CONTINUOUS_WORK_ROOT,
        DEFAULT_TRIMMED_WORK_ROOT,
        LABELS,
        TeacherSpec,
        continuous_config_path,
        continuous_work_dir,
        read_json,
        selected_specs,
        trimmed_work_dir,
        write_json,
    )


CLASS_SAMPLE_STRATEGIES = ("sqrt", "power", "none")


SKELETON_BLOCK = """# COCO-17 left/right keypoint ids
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
skeletons = [[0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
             [11, 13], [13, 15], [6, 12], [12, 14], [14, 16], [0, 1], [0, 2],
             [1, 3], [2, 4], [11, 12]]
left_limb = [0, 2, 3, 6, 7, 8, 12, 14]
right_limb = [1, 4, 5, 9, 10, 11, 13, 15]
"""


MODEL_BLOCK = """model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)
    ),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=9,
        dropout=0.5,
        loss_cls=dict(type='CrossEntropyLoss')
    ),
    test_cfg=dict(average_clips='prob')
)
"""


PIPELINE_BLOCK = """generate_pose_target = dict(
    joint=dict(with_kp=True, with_limb=False),
    limb=dict(with_kp=False, with_limb=True, skeletons=skeletons)
)[stream]

generate_pose_target_test = dict(
    joint=dict(
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=left_kp,
        right_kp=right_kp
    ),
    limb=dict(
        with_kp=False,
        with_limb=True,
        skeletons=skeletons,
        double=True,
        left_kp=left_kp,
        right_kp=right_kp,
        left_limb=left_limb,
        right_limb=right_limb
    )
)[stream]

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=60),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', **generate_pose_target),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=60, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', **generate_pose_target),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=60, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', **generate_pose_target_test),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
"""


def epoch_from_name(path: Path) -> int:
    match = re.search(r"epoch_(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else -1


def resolve_trimmed_checkpoint(
    spec: TeacherSpec,
    stream: str,
    work_root: Path,
    allow_missing: bool,
) -> Path:
    work_dir = trimmed_work_dir(spec, stream, work_root)
    best = sorted(
        work_dir.glob("best_macro_f1*.pth"),
        key=lambda path: (epoch_from_name(path), path.stat().st_mtime),
    )
    if best:
        return best[-1]

    latest = work_dir / "latest.pth"
    if latest.exists():
        return latest

    epochs = sorted(
        work_dir.glob("epoch_*.pth"),
        key=lambda path: (epoch_from_name(path), path.stat().st_mtime),
    )
    if epochs:
        return epochs[-1]

    if allow_missing:
        return latest
    raise FileNotFoundError(
        f"No trimmed checkpoint found for {spec.fold_dir} {spec.teacher} {stream} under {work_dir}. "
        "Train the trimmed teacher first, or pass --allow-missing-trimmed-checkpoints."
    )


def summary_path_for_pkl(path: Path) -> Path:
    return path.with_name(f"{path.stem}_summary.json")


def train_counts_from_summary(summary_path: Path, spec: TeacherSpec) -> list[int]:
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing continuous teacher summary: {summary_path}. "
            "Run thesis/s6/build_continuous_teacher_splits.py first."
        )
    summary = read_json(summary_path)
    key = f"{spec.fold_dir}_{spec.teacher}"
    teacher = summary.get("teacher_splits", {}).get(key)
    if not isinstance(teacher, dict):
        raise KeyError(f"{summary_path} does not contain teacher split {key!r}")
    counts = teacher.get("samples_per_class_per_split", {}).get("train")
    if not isinstance(counts, dict):
        raise KeyError(f"{summary_path} does not contain train class counts for {key!r}")
    return [int(counts.get(label, 0)) for label in LABELS]


def class_sampling_probs(counts: list[int], power: float) -> list[float]:
    weights = [float(count) ** power if count > 0 else 0.0 for count in counts]
    total = sum(weights)
    if total <= 0:
        raise ValueError("Cannot compute class sampling probabilities without positive class counts")
    return [weight / total for weight in weights]


def class_sampling_comment(
    counts: list[int],
    strategy: str,
    power: float,
    epoch_size: int,
) -> str:
    lines = [
        "# Stage-2 teacher fine-tuning samples are drawn each epoch from pre-gridded train windows.",
        f"# class_sample_strategy: {strategy}; class_sample_power: {power:g}; epoch_size: {epoch_size}",
    ]
    if strategy == "none":
        lines.append("# Training uses the natural materialized-window distribution.")
        lines.append("# Train window counts:")
        lines.extend(f"# {label}: n={int(count)}" for label, count in zip(LABELS, counts))
    else:
        probs = class_sampling_probs(counts, power)
        lines.append("# Class draw rule: P(c) = n_c ** class_sample_power / sum_j n_j ** class_sample_power.")
        lines.append("# After drawing a class, one pre-gridded window from that class is sampled uniformly with replacement.")
        lines.append("# Train window counts and expected draws per epoch:")
        for label, count, prob in zip(LABELS, counts, probs):
            lines.append(f"# {label}: n={int(count)}, p={prob:.6f}, expected_epoch_samples={prob * epoch_size:.1f}")
    return "\n".join(lines) + "\n"


def class_sampling_config(strategy: str, power: float, epoch_size: int) -> str:
    if strategy == "none":
        return ""
    return (
        f"class_sample_strategy = {strategy!r}\n"
        f"class_sample_power = {power:.12g}\n"
        f"epoch_size = {int(epoch_size)}\n\n"
    )


def class_sampling_train_kwargs(strategy: str) -> str:
    if strategy == "none":
        return ""
    return (
        "        class_sample_strategy=class_sample_strategy,\n"
        "        class_sample_power=class_sample_power,\n"
        "        epoch_size=epoch_size\n"
    )


def config_text(
    spec: TeacherSpec,
    stream: str,
    ann_file: Path,
    trimmed_checkpoint: Path,
    class_counts: list[int],
    class_sample_strategy: str,
    class_sample_power: float,
    epoch_size: int,
    total_epochs: int,
    lr: float,
    videos_per_gpu: int,
    workers_per_gpu: int,
    test_videos_per_gpu: int,
    continuous_work_root: Path,
) -> str:
    work_dir = continuous_work_dir(spec, stream, continuous_work_root).as_posix()
    return (
        "# ============================================================\n"
        "# Study 6 continuous hard center-time adaptation of skeleton teacher\n"
        f"# Fold {spec.fold}, teacher {spec.teacher}, stream {stream}\n"
        "# Generated by thesis/s6/generate_continuous_configs.py\n"
        "# ============================================================\n\n"
        + f"stream = {stream!r}\n"
        + f"ann_file = {ann_file.as_posix()!r}\n"
        + "dataset_type = 'PoseDataset'\n\n"
        + MODEL_BLOCK
        + "\n"
        + SKELETON_BLOCK
        + "\n"
        + "# S6 teacher-4 continuous split:\n"
        + f"# Source fold: {spec.fold_dir}\n"
        + f"# Teacher: {spec.teacher}\n"
        + f"# Train subjects: {', '.join(spec.train_subjects)}\n"
        + f"# Validation subject: {spec.val_subject}\n"
        + f"# Calibration subject: {spec.calibration_subject}\n"
        + f"# Pseudo-label target/test split: {', '.join(spec.pseudo_subjects)}\n"
        + f"# Original fold test subject unused here: {spec.original_test_subject}\n"
        + "# Continuous pkl is derived from the zero-frame-filtered S2 windows.\n"
        + f"train_split = {spec.train_split!r}\n"
        + f"val_split = {spec.val_split!r}\n"
        + f"test_split = {spec.pseudo_split!r}\n"
        + class_sampling_comment(class_counts, class_sample_strategy, class_sample_power, epoch_size)
        + class_sampling_config(class_sample_strategy, class_sample_power, epoch_size)
        + PIPELINE_BLOCK
        + "\n"
        + "find_unused_parameters = False\n"
        + "data = dict(\n"
        + f"    videos_per_gpu={videos_per_gpu},\n"
        + f"    workers_per_gpu={workers_per_gpu},\n"
        + "    persistent_workers=False,\n"
        + "    train_dataloader=dict(pin_memory=False),\n"
        + "    val_dataloader=dict(pin_memory=False),\n"
        + f"    test_dataloader=dict(videos_per_gpu={test_videos_per_gpu}, pin_memory=False),\n"
        + "    train=dict(\n"
        + "        type=dataset_type,\n"
        + "        ann_file=ann_file,\n"
        + "        pipeline=train_pipeline,\n"
        + "        split=train_split,\n"
        + class_sampling_train_kwargs(class_sample_strategy)
        + "    ),\n"
        + "    val=dict(type=dataset_type, ann_file=ann_file, split=val_split, pipeline=val_pipeline),\n"
        + "    test=dict(type=dataset_type, ann_file=ann_file, split=test_split, pipeline=test_pipeline)\n"
        + ")\n\n"
        + f"optimizer = dict(type='SGD', lr={lr:g}, momentum=0.9, weight_decay=0.0003)\n"
        + "optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))\n"
        + "lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)\n"
        + f"total_epochs = {total_epochs}\n"
        + "checkpoint_config = dict(interval=1)\n"
        + "evaluation = dict(\n"
        + "    interval=1,\n"
        + "    metrics=['top_k_accuracy', 'mean_class_accuracy', 'macro_f1', 'state_macro_f1', 'transition_macro_f1'],\n"
        + "    topk=(1, 5),\n"
        + "    save_best='macro_f1',\n"
        + "    rule='greater'\n"
        + ")\n"
        + "test_evaluation = dict(\n"
        + "    metrics=['top_k_accuracy', 'mean_class_accuracy', 'macro_f1', 'state_macro_f1',\n"
        + "             'transition_macro_f1', 'per_class_f1', 'confusion_matrix'],\n"
        + "    topk=(1, 5)\n"
        + ")\n"
        + "log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])\n"
        + "log_level = 'INFO'\n"
        + f"load_from = {trimmed_checkpoint.as_posix()!r}\n"
        + "resume_from = None\n"
        + "auto_resume = False\n"
        + f"work_dir = './{work_dir}'\n"
    )


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate S6 continuous teacher fine-tuning configs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONTINUOUS_CONFIG_DIR)
    parser.add_argument("--ann-file", type=Path, default=DEFAULT_CONTINUOUS_TEACHER_PKL)
    parser.add_argument("--trimmed-work-root", type=Path, default=DEFAULT_TRIMMED_WORK_ROOT)
    parser.add_argument("--continuous-work-root", type=Path, default=DEFAULT_CONTINUOUS_WORK_ROOT)
    parser.add_argument("--folds", nargs="+", help="Fold ids to include. Default: a b c.")
    parser.add_argument("--teachers", nargs="+", help="Teacher ids to include. Default: t1 t2 t3 t4.")
    parser.add_argument("--streams", nargs="+", choices=["joint", "limb"], default=["joint", "limb"])
    parser.add_argument("--total-epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--videos-per-gpu", type=int, default=8)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--test-videos-per-gpu", type=int, default=1)
    parser.add_argument("--class-sample-strategy", choices=CLASS_SAMPLE_STRATEGIES, default="sqrt")
    parser.add_argument("--class-sample-power", type=float, default=0.5)
    parser.add_argument("--epoch-size", type=int)
    parser.add_argument("--allow-missing-trimmed-checkpoints", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_epochs <= 0:
        raise ValueError("--total-epochs must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.videos_per_gpu <= 0:
        raise ValueError("--videos-per-gpu must be positive")
    if args.workers_per_gpu < 0:
        raise ValueError("--workers-per-gpu must be non-negative")
    if args.test_videos_per_gpu <= 0:
        raise ValueError("--test-videos-per-gpu must be positive")
    if args.class_sample_power < 0:
        raise ValueError("--class-sample-power must be non-negative")
    if args.class_sample_strategy == "sqrt" and abs(args.class_sample_power - 0.5) > 1e-12:
        raise ValueError("--class-sample-strategy sqrt requires --class-sample-power 0.5")
    if args.epoch_size is not None and args.epoch_size <= 0:
        raise ValueError("--epoch-size must be positive")

    specs = selected_specs(args.folds, args.teachers)
    summary_path = summary_path_for_pkl(args.ann_file)
    outputs: list[str] = []
    manifest_records: list[dict[str, object]] = []
    for spec in specs:
        class_counts = train_counts_from_summary(summary_path, spec)
        epoch_size = int(args.epoch_size) if args.epoch_size is not None else int(sum(class_counts))
        for stream in args.streams:
            checkpoint = resolve_trimmed_checkpoint(
                spec=spec,
                stream=stream,
                work_root=args.trimmed_work_root,
                allow_missing=args.allow_missing_trimmed_checkpoints,
            )
            path = continuous_config_path(spec, stream, args.output_dir)
            write_text(
                path,
                config_text(
                    spec=spec,
                    stream=stream,
                    ann_file=args.ann_file,
                    trimmed_checkpoint=checkpoint,
                    class_counts=class_counts,
                    class_sample_strategy=args.class_sample_strategy,
                    class_sample_power=args.class_sample_power,
                    epoch_size=epoch_size,
                    total_epochs=args.total_epochs,
                    lr=args.lr,
                    videos_per_gpu=args.videos_per_gpu,
                    workers_per_gpu=args.workers_per_gpu,
                    test_videos_per_gpu=args.test_videos_per_gpu,
                    continuous_work_root=args.continuous_work_root,
                ),
                overwrite=args.overwrite,
            )
            outputs.append(str(path))
            manifest_records.append(
                {
                    "fold": spec.fold,
                    "teacher": spec.teacher,
                    "stream": stream,
                    "config": str(path),
                    "load_from": str(checkpoint),
                    "train_split": spec.train_split,
                    "val_split": spec.val_split,
                    "pseudo_split": spec.pseudo_split,
                    "epoch_size": epoch_size,
                    "class_counts": class_counts,
                }
            )

    manifest = args.output_dir / "config_manifest.json"
    write_json(
        manifest,
        {
            "stage": "s6_continuous_teacher_configs",
            "ann_file": str(args.ann_file),
            "trimmed_work_root": str(args.trimmed_work_root),
            "continuous_work_root": str(args.continuous_work_root),
            "folds": sorted({spec.fold for spec in specs}),
            "teachers": sorted({spec.teacher for spec in specs}),
            "streams": args.streams,
            "total_epochs": args.total_epochs,
            "lr": args.lr,
            "videos_per_gpu": args.videos_per_gpu,
            "workers_per_gpu": args.workers_per_gpu,
            "test_videos_per_gpu": args.test_videos_per_gpu,
            "class_sample_strategy": args.class_sample_strategy,
            "class_sample_power": args.class_sample_power,
            "labels": LABELS,
            "configs": outputs,
            "records": manifest_records,
        },
        overwrite=args.overwrite,
    )
    print(f"[DONE] wrote {len(outputs)} S6 continuous configs under {args.output_dir}")
    print(f"[DONE] config manifest: {manifest}")


if __name__ == "__main__":
    main()

"""Generate S6 trimmed teacher-4 PoseC3D configs."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import (
        DEFAULT_TRIMMED_CONFIG_DIR,
        DEFAULT_TRIMMED_PKL_DIR,
        LABELS,
        TeacherSpec,
        count_summary_labels,
        selected_specs,
        trimmed_config_path,
        trimmed_pkl_path,
        trimmed_summary_path,
        trimmed_work_dir,
        write_json,
    )
except ImportError:
    from common import (
        DEFAULT_TRIMMED_CONFIG_DIR,
        DEFAULT_TRIMMED_PKL_DIR,
        LABELS,
        TeacherSpec,
        count_summary_labels,
        selected_specs,
        trimmed_config_path,
        trimmed_pkl_path,
        trimmed_summary_path,
        trimmed_work_dir,
        write_json,
    )


DEFAULT_CLASS_PROB = [2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]


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


def split_comment(spec: TeacherSpec, counts: list[int] | None) -> str:
    lines = [
        "# S6 teacher-4 out-of-fold trimmed split:",
        f"# Source fold: {spec.fold_dir}",
        f"# Teacher: {spec.teacher}",
        f"# Train subjects: {', '.join(spec.train_subjects)}",
        f"# Validation subject: {spec.val_subject}",
        f"# Calibration subject: {spec.calibration_subject}",
        f"# Pseudo-label target/test split: {', '.join(spec.pseudo_subjects)}",
        f"# Original fold test subject unused here: {spec.original_test_subject}",
        "# Zero-frame policy: build_pyskl_pkl removes all-zero pose frames and drops samples with <30 retained frames.",
    ]
    if counts is not None:
        lines.append("# Train counts after zero-frame filtering:")
        lines.extend(f"# {label}: {count}" for label, count in zip(LABELS, counts))
    return "\n".join(lines) + "\n"


def config_text(
    spec: TeacherSpec,
    stream: str,
    pkl_dir: Path,
    class_prob: list[float],
    total_epochs: int,
    lr: float,
    videos_per_gpu: int,
    workers_per_gpu: int,
    test_videos_per_gpu: int,
) -> str:
    ann_file = trimmed_pkl_path(spec, pkl_dir).as_posix()
    work_dir = trimmed_work_dir(spec, stream).as_posix()
    counts = count_summary_labels(trimmed_summary_path(spec, pkl_dir), split="train")
    return (
        "# ============================================================\n"
        "# Study 6 trimmed PoseC3D out-of-fold skeleton teacher\n"
        f"# Fold {spec.fold}, teacher {spec.teacher}, stream {stream}\n"
        "# Generated by thesis/s6/generate_trimmed_configs.py\n"
        "# ============================================================\n\n"
        + f"stream = {stream!r}\n"
        + f"pkl = {spec.pkl_stem!r}\n"
        + f"ann_file = {ann_file!r}\n"
        + "dataset_type = 'PoseDataset'\n\n"
        + MODEL_BLOCK
        + "\n"
        + SKELETON_BLOCK
        + "\n"
        + split_comment(spec, counts)
        + f"class_prob = {class_prob!r}\n\n"
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
        + "        split='train',\n"
        + "        class_prob=class_prob\n"
        + "    ),\n"
        + "    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline),\n"
        + "    test=dict(type=dataset_type, ann_file=ann_file, split='test', pipeline=test_pipeline)\n"
        + ")\n\n"
        + f"optimizer = dict(type='SGD', lr={lr:g}, momentum=0.9, weight_decay=0.0003)\n"
        + "optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))\n"
        + "lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)\n"
        + f"total_epochs = {total_epochs}\n"
        + "checkpoint_config = dict(interval=1)\n"
        + "evaluation = dict(\n"
        + "    interval=1,\n"
        + "    metrics=['top_k_accuracy', 'mean_class_accuracy', 'macro_f1'],\n"
        + "    topk=(1, 5),\n"
        + "    save_best='macro_f1',\n"
        + "    rule='greater'\n"
        + ")\n"
        + "test_evaluation = dict(\n"
        + "    metrics=['top_k_accuracy', 'mean_class_accuracy', 'macro_f1', 'per_class_f1', 'confusion_matrix'],\n"
        + "    topk=(1, 5)\n"
        + ")\n"
        + "log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])\n"
        + "log_level = 'INFO'\n"
        + "load_from = None\n"
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
    parser = argparse.ArgumentParser(description="Generate S6 trimmed teacher configs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRIMMED_CONFIG_DIR)
    parser.add_argument("--pkl-dir", type=Path, default=DEFAULT_TRIMMED_PKL_DIR)
    parser.add_argument("--folds", nargs="+", help="Fold ids to include. Default: a b c.")
    parser.add_argument("--teachers", nargs="+", help="Teacher ids to include. Default: t1 t2 t3 t4.")
    parser.add_argument("--streams", nargs="+", choices=["joint", "limb"], default=["joint", "limb"])
    parser.add_argument("--total-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--videos-per-gpu", type=int, default=32)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--test-videos-per-gpu", type=int, default=1)
    parser.add_argument("--class-prob", nargs=9, type=float, default=DEFAULT_CLASS_PROB)
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
    if any(value < 1.0 for value in args.class_prob):
        raise ValueError("--class-prob values must be >= 1.0 for the legacy sampler")

    specs = selected_specs(args.folds, args.teachers)
    outputs: list[str] = []
    for spec in specs:
        for stream in args.streams:
            path = trimmed_config_path(spec, stream, args.output_dir)
            write_text(
                path,
                config_text(
                    spec=spec,
                    stream=stream,
                    pkl_dir=args.pkl_dir,
                    class_prob=list(args.class_prob),
                    total_epochs=args.total_epochs,
                    lr=args.lr,
                    videos_per_gpu=args.videos_per_gpu,
                    workers_per_gpu=args.workers_per_gpu,
                    test_videos_per_gpu=args.test_videos_per_gpu,
                ),
                overwrite=args.overwrite,
            )
            outputs.append(str(path))

    manifest = args.output_dir / "config_manifest.json"
    write_json(
        manifest,
        {
            "stage": "s6_trimmed_teacher_configs",
            "pkl_dir": str(args.pkl_dir),
            "output_dir": str(args.output_dir),
            "folds": sorted({spec.fold for spec in specs}),
            "teachers": sorted({spec.teacher for spec in specs}),
            "streams": args.streams,
            "total_epochs": args.total_epochs,
            "lr": args.lr,
            "videos_per_gpu": args.videos_per_gpu,
            "workers_per_gpu": args.workers_per_gpu,
            "test_videos_per_gpu": args.test_videos_per_gpu,
            "class_prob": list(args.class_prob),
            "labels": LABELS,
            "configs": outputs,
        },
        overwrite=args.overwrite,
    )
    print(f"[DONE] wrote {len(outputs)} S6 trimmed configs under {args.output_dir}")
    print(f"[DONE] config manifest: {manifest}")


if __name__ == "__main__":
    main()

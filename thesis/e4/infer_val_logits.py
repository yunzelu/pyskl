"""Infer E4 validation-subject continuous-window logits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .common import (
        DEFAULT_CONFIG_ROOT,
        DEFAULT_JSONL_ROOT,
        DEFAULT_WORK_ROOT,
        LABELS,
        default_val_logits_path,
        protocol_metadata,
        write_json,
        write_score_csv,
    )
except ImportError:
    from common import (
        DEFAULT_CONFIG_ROOT,
        DEFAULT_JSONL_ROOT,
        DEFAULT_WORK_ROOT,
        LABELS,
        default_val_logits_path,
        protocol_metadata,
        write_json,
        write_score_csv,
    )

from thesis.e2.common import discover_subject_sessions  # noqa: E402
from thesis.e3.common import discover_e3_folds  # noqa: E402
from thesis.e3.infer_logits import infer_session_logits  # noqa: E402
from infer_processed_pose_csv import resolve_device  # noqa: E402


def infer_val_logits(
    stream: str,
    jsonl_root: Path,
    config_root: Path,
    work_root: Path,
    output: Path,
    batch_size: int,
    device: str,
    overwrite: bool,
    quiet: bool,
) -> None:
    folds = discover_e3_folds(config_root, work_root, stream)
    all_rows = []
    fold_infos: list[dict[str, Any]] = []
    session_infos: list[dict[str, Any]] = []

    for fold in folds:
        subject = fold.val_subject
        sessions = discover_subject_sessions(jsonl_root, subject)
        fold_infos.append(
            {
                "fold": fold.fold,
                "val_subject": fold.val_subject,
                "calib_subject": fold.calib_subject,
                "test_subject": fold.test_subject,
                "active_subject": subject,
                "config": str(fold.config_path),
                "checkpoint": str(fold.checkpoint_path),
                "sessions": [path.parent.name for path in sessions],
            }
        )

        if not quiet:
            print(
                f"[INFO] fold {fold.fold} val: subject={subject}, "
                f"sessions={len(sessions)}, checkpoint={fold.checkpoint_path.name}"
            )

        for jsonl_path in sessions:
            if not quiet:
                print(f"[INFO] inferring {stream} val: {jsonl_path.parent.name}")
            rows, info = infer_session_logits(
                fold=fold,
                split="val",
                subject=subject,
                jsonl_path=jsonl_path,
                stream=stream,
                batch_size=batch_size,
                device=device,
                quiet=quiet,
            )
            all_rows.extend(rows)
            session_infos.append({"fold": fold.fold, "subject": subject, **info})

    if not all_rows:
        raise ValueError("No E4 validation logit rows were produced")

    write_score_csv(output, all_rows, overwrite=overwrite)
    write_json(
        output.with_name(f"{output.stem}_metadata.json"),
        {
            "experiment": "E4",
            "stage": "validation_logit_inference",
            "stream": stream,
            "score_output": "logit",
            "protocol": protocol_metadata(),
            "labels": LABELS,
            "folds": fold_infos,
            "sessions": session_infos,
            "score_csv": str(output),
        },
        overwrite=overwrite,
    )
    print(f"[DONE] wrote {len(all_rows)} {stream} val logit rows to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E4 validation-subject continuous-window logit inference.")
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument("--jsonl-root", type=Path, default=DEFAULT_JSONL_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    infer_val_logits(
        stream=args.stream,
        jsonl_root=args.jsonl_root,
        config_root=args.config_root,
        work_root=args.work_root,
        output=args.output or default_val_logits_path(args.stream),
        batch_size=args.batch_size,
        device=resolve_device(args.device),
        overwrite=args.overwrite,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()

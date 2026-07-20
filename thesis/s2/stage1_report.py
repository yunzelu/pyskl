"""Record existing trimmed PoseC3D checkpoints used to initialize Study 2."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import (
        DEFAULT_STAGE1_REPORT,
        LABELS,
        discover_s2_folds,
        protocol_metadata,
        stage1_checkpoint_metadata,
        write_json,
    )
except ImportError:
    from common import (
        DEFAULT_STAGE1_REPORT,
        LABELS,
        discover_s2_folds,
        protocol_metadata,
        stage1_checkpoint_metadata,
        write_json,
    )


def build_stage1_report(streams: list[str]) -> dict:
    folds = discover_s2_folds()
    records = []
    for fold in folds:
        for stream in streams:
            records.append(stage1_checkpoint_metadata(fold, stream))
    return {
        "experiment": "S2",
        "stage": "stage1_trimmed_checkpoint_report",
        "protocol": protocol_metadata(),
        "labels": LABELS,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the S2 Stage-1 checkpoint report.")
    parser.add_argument("--streams", nargs="+", choices=["joint", "limb"], default=["joint", "limb"])
    parser.add_argument("--output", type=Path, default=DEFAULT_STAGE1_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_stage1_report(args.streams)
    write_json(args.output, report, overwrite=args.overwrite)
    print(f"[DONE] wrote Stage-1 checkpoint report to {args.output}")
    for record in report["records"]:
        print(
            f"fold={record['fold']} stream={record['stream']} "
            f"epoch={record['epoch']} val_acc={record['validation_accuracy']} "
            f"val_macro_f1={record['validation_macro_f1']} "
            f"checkpoint={record['checkpoint']}"
        )


if __name__ == "__main__":
    main()

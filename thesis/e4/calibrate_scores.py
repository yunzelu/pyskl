"""Materialize E4 calibrated probability CSVs from E3/E4 logits."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import (
        LABELS,
        calibrate_logit_rows,
        default_calibrated_path,
        default_e3_logits_path,
        default_e3_temperature_path,
        default_val_logits_path,
        load_temperatures,
        protocol_metadata,
        read_score_csv,
        write_json,
        write_score_csv,
    )
except ImportError:
    from common import (
        LABELS,
        calibrate_logit_rows,
        default_calibrated_path,
        default_e3_logits_path,
        default_e3_temperature_path,
        default_val_logits_path,
        load_temperatures,
        protocol_metadata,
        read_score_csv,
        write_json,
        write_score_csv,
    )


def default_logits_for_split(stream: str, split: str) -> Path:
    if split == "val":
        return default_val_logits_path(stream)
    if split == "test":
        return default_e3_logits_path(stream, "test")
    raise ValueError(f"Unsupported split: {split}")


def calibrate_file(
    logits_path: Path,
    temperatures_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    rows = read_score_csv(logits_path)
    temperatures = load_temperatures(temperatures_path)
    calibrated = calibrate_logit_rows(rows, temperatures)
    write_score_csv(output_path, calibrated, overwrite=overwrite)
    write_json(
        output_path.with_name(f"{output_path.stem}_metadata.json"),
        {
            "experiment": "E4",
            "stage": "temperature_calibration",
            "logits": str(logits_path),
            "temperatures": str(temperatures_path),
            "output_scores": str(output_path),
            "score_output": "prob",
            "protocol": protocol_metadata(),
            "labels": LABELS,
            "num_rows": len(calibrated),
            "fold_temperatures": temperatures,
        },
        overwrite=overwrite,
    )
    print(f"[DONE] wrote {len(calibrated)} calibrated rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply E3 temperature scaling to E4 logits.")
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--logits", type=Path)
    parser.add_argument("--temperatures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logits_path = args.logits or default_logits_for_split(args.stream, args.split)
    temperatures_path = args.temperatures or default_e3_temperature_path(args.stream)
    output_path = args.output or default_calibrated_path(args.stream, args.split)
    calibrate_file(
        logits_path=logits_path,
        temperatures_path=temperatures_path,
        output_path=output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

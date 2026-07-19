"""Materialize E4 raw or calibrated probability CSVs from E3/E4 logits."""

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
        default_raw_path,
        default_val_logits_path,
        load_temperatures,
        protocol_metadata,
        raw_probability_rows,
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
        default_raw_path,
        default_val_logits_path,
        load_temperatures,
        protocol_metadata,
        raw_probability_rows,
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
    temperatures_path: Path | None,
    output_path: Path,
    score_kind: str,
    overwrite: bool,
) -> None:
    rows = read_score_csv(logits_path)
    if score_kind == "raw":
        probabilities = raw_probability_rows(rows)
        temperatures = None
    elif score_kind == "calibrated":
        if temperatures_path is None:
            raise ValueError("calibrated mode requires --temperatures")
        temperatures = load_temperatures(temperatures_path)
        probabilities = calibrate_logit_rows(rows, temperatures)
    else:
        raise ValueError(f"Unsupported score kind: {score_kind}")

    write_score_csv(output_path, probabilities, overwrite=overwrite)
    write_json(
        output_path.with_name(f"{output_path.stem}_metadata.json"),
        {
            "experiment": "E4",
            "stage": "probability_materialization",
            "score_kind": score_kind,
            "logits": str(logits_path),
            "temperatures": "" if temperatures_path is None else str(temperatures_path),
            "output_scores": str(output_path),
            "score_output": "prob",
            "protocol": protocol_metadata(),
            "labels": LABELS,
            "num_rows": len(probabilities),
            "fold_temperatures": {} if temperatures is None else temperatures,
        },
        overwrite=overwrite,
    )
    print(f"[DONE] wrote {len(probabilities)} {score_kind} probability rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize E4 raw/calibrated probabilities from logits.")
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--score-kind", choices=["raw", "calibrated"], default="calibrated")
    parser.add_argument("--logits", type=Path)
    parser.add_argument("--temperatures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logits_path = args.logits or default_logits_for_split(args.stream, args.split)
    temperatures_path = args.temperatures or (
        default_e3_temperature_path(args.stream) if args.score_kind == "calibrated" else None
    )
    if args.output:
        output_path = args.output
    elif args.score_kind == "raw":
        output_path = default_raw_path(args.stream, args.split)
    else:
        output_path = default_calibrated_path(args.stream, args.split)
    calibrate_file(
        logits_path=logits_path,
        temperatures_path=temperatures_path,
        output_path=output_path,
        score_kind=args.score_kind,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

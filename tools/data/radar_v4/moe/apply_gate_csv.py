"""Apply a trained radar v4 MoE gate to aligned stream prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_STREAMS,
    aligned_stream_rows,
    load_gate_checkpoint,
    read_manifest,
    row_has_model_prediction,
    safe_int,
    split_values,
    stream_scores,
    truthy,
)


BASE_COLUMNS = [
    "frame_index",
    "timestamp",
    "timestamp_unix",
    "prediction",
    "prediction_id",
    "confidence",
    "contributing_windows",
    "covering_windows",
    "detection_count",
    "selected_detection",
    "assigned_center_frame",
    "assigned_window_start",
    "assigned_window_end",
    "center_distance",
    "is_prediction_center",
]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def output_path_for_session(output_dir: Path, session: str, suffix: str) -> Path:
    return output_dir / f"{session}{suffix}"


def build_features_for_rows(
    rows_by_stream: dict[str, list[dict[str, str]]],
    labels: list[str],
    streams: list[str],
    keep_empty_frames: bool,
    keep_no_prediction_frames: bool,
) -> tuple[list[list[float]], list[int], list[bool]]:
    features: list[list[float]] = []
    valid_indices: list[int] = []
    valid_mask = [False] * len(rows_by_stream[streams[0]])

    for frame_index in range(len(valid_mask)):
        reference_row = rows_by_stream[streams[0]][frame_index]
        if not keep_empty_frames and not truthy(reference_row.get("selected_detection")):
            continue

        vector: list[float] = []
        valid = True
        for stream in streams:
            row = rows_by_stream[stream][frame_index]
            if not keep_no_prediction_frames and not row_has_model_prediction(row):
                valid = False
                break

            scores = stream_scores(row, labels)
            if scores is None or any(not math.isfinite(value) for value in scores):
                valid = False
                break
            vector.extend(scores)

        if valid:
            valid_indices.append(frame_index)
            valid_mask[frame_index] = True
            features.append(vector)

    return features, valid_indices, valid_mask


def run_gate(
    gate: Any,
    features: list[list[float]],
    batch_size: int,
    device: str,
    num_streams: int,
    num_classes: int,
) -> tuple[list[list[float]], list[list[float]]]:
    import torch

    from common import fuse_expert_probs

    if not features:
        return [], []

    x = torch.tensor(features, dtype=torch.float32, device=device)
    fused_batches = []
    alpha_batches = []
    gate.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch_x = x[start : start + batch_size]
            gate_logits = gate(batch_x)
            fused, alpha = fuse_expert_probs(
                features=batch_x,
                gate_logits=gate_logits,
                num_streams=num_streams,
                num_classes=num_classes,
            )
            fused_batches.append(fused.detach().cpu())
            alpha_batches.append(alpha.detach().cpu())

    fused_all = torch.cat(fused_batches, dim=0).tolist()
    alpha_all = torch.cat(alpha_batches, dim=0).tolist()
    return fused_all, alpha_all


def output_columns(labels: list[str], streams: list[str], include_gate_columns: bool) -> list[str]:
    columns = [*BASE_COLUMNS, *[f"score_{label}" for label in labels]]
    if include_gate_columns:
        columns.extend(f"gate_alpha_{stream}" for stream in streams)
    return columns


def write_fused_csv(
    output_path: Path,
    template_rows: list[dict[str, str]],
    labels: list[str],
    streams: list[str],
    valid_indices: list[int],
    fused_probs: list[list[float]],
    alphas: list[list[float]],
    include_gate_columns: bool,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = output_columns(labels, streams, include_gate_columns)
    score_columns = [f"score_{label}" for label in labels]
    valid_lookup = {
        frame_index: (fused_probs[index], alphas[index])
        for index, frame_index in enumerate(valid_indices)
    }

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for row_index, template in enumerate(template_rows):
            output_row = {column: template.get(column, "") for column in columns}

            if row_index in valid_lookup:
                scores, alpha_values = valid_lookup[row_index]
                prediction_id = max(range(len(scores)), key=lambda index: scores[index])
                output_row["prediction"] = labels[prediction_id]
                output_row["prediction_id"] = str(prediction_id)
                output_row["confidence"] = f"{float(scores[prediction_id]):.8f}"
                output_row["contributing_windows"] = template.get("contributing_windows", "1")
                for column, value in zip(score_columns, scores):
                    output_row[column] = f"{float(value):.8f}"
                if include_gate_columns:
                    for stream, value in zip(streams, alpha_values):
                        output_row[f"gate_alpha_{stream}"] = f"{float(value):.8f}"
            else:
                output_row["prediction"] = (
                    "NoPrediction"
                    if safe_int(template.get("detection_count"), default=0) > 0
                    else "NoDetection"
                )
                output_row["prediction_id"] = "-1"
                output_row["confidence"] = "0.00000000"
                output_row["contributing_windows"] = "0"
                output_row["assigned_center_frame"] = "-1"
                output_row["assigned_window_start"] = "-1"
                output_row["assigned_window_end"] = "-1"
                output_row["center_distance"] = "-1"
                output_row["is_prediction_center"] = "0"
                for column in score_columns:
                    output_row[column] = "0.00000000"
                if include_gate_columns:
                    for stream in streams:
                        output_row[f"gate_alpha_{stream}"] = "0.00000000"

            writer.writerow(output_row)


def apply_manifest(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    device = resolve_device(args.device)
    gate, checkpoint = load_gate_checkpoint(args.gate, device=device)
    labels = list(checkpoint["labels"])
    streams = split_values(args.streams) if args.streams else list(checkpoint["streams"])
    if streams != list(checkpoint["streams"]):
        raise ValueError(
            f"Stream order {streams} does not match checkpoint streams {checkpoint['streams']}"
        )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    specs = read_manifest(args.manifest, streams=streams, require_origin=False)
    outputs = []
    for spec in specs:
        rows_by_stream = aligned_stream_rows(spec, labels=labels, streams=streams)
        features, valid_indices, _valid_mask = build_features_for_rows(
            rows_by_stream=rows_by_stream,
            labels=labels,
            streams=streams,
            keep_empty_frames=args.keep_empty_frames,
            keep_no_prediction_frames=args.keep_no_prediction_frames,
        )
        fused_probs, alphas = run_gate(
            gate=gate,
            features=features,
            batch_size=args.batch_size,
            device=device,
            num_streams=len(streams),
            num_classes=len(labels),
        )
        output_path = output_path_for_session(
            output_dir=args.output_dir,
            session=spec.session,
            suffix=args.suffix,
        )
        write_fused_csv(
            output_path=output_path,
            template_rows=rows_by_stream[streams[0]],
            labels=labels,
            streams=streams,
            valid_indices=valid_indices,
            fused_probs=fused_probs,
            alphas=alphas,
            include_gate_columns=args.include_gate_columns,
            overwrite=args.overwrite,
        )
        outputs.append(
            {
                "session": spec.session,
                "output": output_path,
                "frames": len(rows_by_stream[streams[0]]),
                "gated_frames": len(valid_indices),
            }
        )
        if not args.quiet:
            print(
                f"[DONE] {spec.session}: wrote {output_path} "
                f"({len(valid_indices)}/{len(rows_by_stream[streams[0]])} gated frames)"
            )

    # Keep torch imported so old checkpoint pickle modules resolve before function exit.
    _ = torch
    return {"outputs": outputs, "labels": labels, "streams": streams}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a trained radar v4 MoE gate to stream prediction CSVs."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True, help="Gate checkpoint from train_gate.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--suffix",
        default="_moe_predictions.csv",
        help="Output filename suffix appended to manifest session. Default: _moe_predictions.csv.",
    )
    parser.add_argument(
        "--streams",
        help="Override stream order. Must match the checkpoint. Default: checkpoint streams.",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--keep-empty-frames",
        action="store_true",
        help="Gate frames where selected_detection=0. Default masks them to NoDetection/NoPrediction.",
    )
    parser.add_argument(
        "--keep-no-prediction-frames",
        action="store_true",
        help="Gate frames where one or more streams have no model prediction.",
    )
    parser.add_argument(
        "--include-gate-columns",
        action="store_true",
        help="Append gate_alpha_<stream> columns for debugging.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_manifest(args)


if __name__ == "__main__":
    main()

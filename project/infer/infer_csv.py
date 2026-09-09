"""Infer four-stream center labels directly from *_cf.csv and paired *_mm.json."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.infer.streaming import iter_chunks, iter_frames, iter_windows, new_stats, read_mask
from rerun.dataset.build_radar_v4_yolo26xpose_datasets import FINAL_LABELS


WEIGHTS = {"j": 2.0, "b": 2.0, "jm": 1.0, "bm": 1.0}
DEFAULT_MODEL_ROOT = Path("work_dirs/project/stgcnpp/fps10_phase0")
DEFAULT_CONFIG_ROOT = Path("project/configs/stgcnpp/fps10_phase0")


def fuse_probabilities(probabilities: dict[str, np.ndarray]) -> np.ndarray:
    if set(probabilities) != set(WEIGHTS):
        raise ValueError("Fusion requires exactly j, b, jm, and bm probabilities")
    result = None
    for stream, weight in WEIGHTS.items():
        values = np.asarray(probabilities[stream], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(FINAL_LABELS):
            raise ValueError(f"Invalid probability shape for {stream}: {values.shape}")
        if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
                or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5)):
            raise ValueError(f"Invalid probabilities for {stream}; expected normalized class probabilities")
        if result is None:
            result = np.zeros_like(values)
        if result.shape != values.shape:
            raise ValueError("All streams must predict the same windows in the same order")
        result += weight * values
    return result / sum(WEIGHTS.values())


def find_inputs(inputs: list[Path], excludes: list[str]) -> list[tuple[Path, Path]]:
    csv_paths = set()
    for path in inputs:
        if path.is_dir():
            csv_paths.update(path.glob("*_cf.csv"))
        elif path.is_file() and path.name.endswith("_cf.csv"):
            csv_paths.add(path)
        else:
            raise ValueError(f"Expected an existing *_cf.csv or a directory: {path}")
    selected = sorted({path.resolve() for path in csv_paths
                       if not any(path.match(pattern) for pattern in excludes)})
    if not selected:
        raise ValueError("No input *_cf.csv files selected")
    stems = set()
    pairs = []
    for path in selected:
        if path.stem in stems:
            raise ValueError(f"Input filenames must be distinct for separate outputs: {path.name}")
        stems.add(path.stem)
        mask = path.with_name(path.name[:-len("_cf.csv")] + "_mm.json")
        if not mask.is_file():
            raise FileNotFoundError(f"Required paired multiperson mask not found: {mask}")
        pairs.append((path, mask))
    return pairs


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def prediction_fields() -> list[str]:
    return [
        "recording_id", "window_candidate_index", "window_start_retained_idx", "center_retained_idx",
        "window_end_retained_idx_exclusive", "source_frame_start", "source_frame_center", "source_frame_end",
        "source_csv_row_start", "source_csv_row_center", "source_csv_row_end", "ID", "source_id",
        "start_timestamp_sec", "center_timestamp_sec", "end_timestamp_sec",
        "window_size", "stride", "max_adjacent_gap_sec", "window_span_sec", "label_id", "label_name", "confidence",
        *[f"fused_p{index}" for index in range(len(FINAL_LABELS))],
        *[f"{stream}_p{index}" for stream in WEIGHTS for index in range(len(FINAL_LABELS))],
    ]


def prediction_row(recording: str, window, fused: np.ndarray, probabilities: dict[str, np.ndarray]) -> dict:
    start, center, end = window.frames[0], window.center, window.frames[-1]
    label_id = int(np.argmax(fused))
    return {
        "recording_id": recording,
        "window_candidate_index": window.candidate_index,
        "window_start_retained_idx": window.row_start,
        "center_retained_idx": center.retained_index,
        "window_end_retained_idx_exclusive": window.row_start + len(window.frames),
        "source_frame_start": start.frame_index,
        "source_frame_center": center.frame_index,
        "source_frame_end": end.frame_index,
        "source_csv_row_start": start.csv_row_number,
        "source_csv_row_center": center.csv_row_number,
        "source_csv_row_end": end.csv_row_number,
        "ID": 0,
        "source_id": center.original_id,
        "start_timestamp_sec": start.timestamp_text,
        "center_timestamp_sec": center.timestamp_text,
        "end_timestamp_sec": end.timestamp_text,
        "window_size": len(window.frames),
        "stride": 4,
        "max_adjacent_gap_sec": window.max_gap_sec,
        "window_span_sec": window.span_sec,
        "label_id": label_id,
        "label_name": FINAL_LABELS[label_id],
        "confidence": float(fused[label_id]),
        **{f"fused_p{index}": float(value) for index, value in enumerate(fused)},
        **{f"{stream}_p{index}": float(value)
           for stream, values in probabilities.items() for index, value in enumerate(values)},
    }


def process_one(csv_path: Path, mask_path: Path, args: argparse.Namespace, predictor=None) -> dict:
    started = time.monotonic()
    stats = new_stats()
    mask = read_mask(mask_path)
    frames = iter_frames(csv_path, mask, stats)
    windows = iter_windows(frames, stats)
    if args.max_windows is not None:
        windows = islice(windows, args.max_windows)
    suffix = "validation" if args.validate_only else "inference"
    summary_path = args.output_dir / f"{csv_path.stem}__{suffix}_summary.json"
    prediction_path = args.output_dir / f"{csv_path.stem}__center_predictions.csv"
    temporary = prediction_path.with_suffix(".csv.partial")
    label_counts = Counter()
    chunks = 0
    output_rows = 0
    handle = None
    try:
        writer = None
        if not args.validate_only:
            if predictor is None:
                raise ValueError("A predictor is required outside validation-only mode")
            handle = temporary.open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(handle, fieldnames=prediction_fields())
            writer.writeheader()
        for chunk in iter_chunks(windows, args.chunk_size):
            chunks += 1
            if writer is not None:
                probabilities = predictor.predict([window.to_annotation() for window in chunk])
                fused = fuse_probabilities(probabilities)
                if fused.shape[0] != len(chunk):
                    raise ValueError("Prediction count does not match the input window chunk")
                for index, window in enumerate(chunk):
                    row = prediction_row(csv_path.stem, window, fused[index],
                                         {stream: values[index] for stream, values in probabilities.items()})
                    writer.writerow(row)
                    label_counts[row["label_name"]] += 1
                output_rows += len(chunk)
                handle.flush()
            print(f"[{csv_path.name}] chunk {chunks}: {len(chunk)} windows; "
                  f"{stats['valid_windows']} valid so far", flush=True)
        if handle is not None:
            handle.close()
            handle = None
            temporary.replace(prediction_path)
    except BaseException:
        if handle is not None:
            handle.close()
        if not args.validate_only:
            temporary.unlink(missing_ok=True)
        raise
    result = {
        "status": "limited" if args.max_windows is not None else "complete",
        "mode": suffix,
        "input_csv": str(csv_path),
        "multiperson_mask": str(mask_path),
        "input_csv_bytes": csv_path.stat().st_size,
        "mask_intervals_after_union": len(mask.intervals),
        "mask_endpoints": "inclusive",
        "timestamp_comparison": "exact decimal Unix seconds; no tolerance or rounding",
        "identity_policy": "one constant person ID 0; input tracking IDs ignored for window construction",
        "sampling": "every unique unmasked timestamp; no downsampling",
        "window_size": 20,
        "stride": 4,
        "center_offset": 10,
        "max_adjacent_gap_sec": 0.5,
        "max_window_span_sec": 2.5,
        "chunk_size": args.chunk_size,
        "chunks_processed": chunks,
        "max_windows": args.max_windows,
        "scan_complete": args.max_windows is None,
        "counts": stats,
        "output_rows": output_rows,
        "prediction_csv": str(prediction_path) if not args.validate_only else None,
        "fusion_weights": WEIGHTS,
        "fusion_rule": "(2 * p_j + 2 * p_b + p_jm + p_bm) / 6",
        "label_to_id": {label: index for index, label in enumerate(FINAL_LABELS)},
        "predicted_class_counts": {label: label_counts[label] for label in FINAL_LABELS},
        "model": predictor.describe() if predictor is not None else None,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(summary_path, result)
    print(f"[DONE] {csv_path.name}: {stats['valid_windows']} valid windows; {summary_path}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="One or more *_cf.csv files or directories")
    parser.add_argument("--exclude", action="append", default=[], help="Filename glob to exclude; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("data/project/inference/fps10_phase0"))
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0, etc.; default cpu")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Maximum valid windows held per inference chunk")
    parser.add_argument("--batch-size", type=int, default=128, help="Model minibatch size within each chunk")
    parser.add_argument("--num-threads", type=int, default=0, help="Torch CPU threads; 0 keeps its default")
    parser.add_argument("--validate-only", action="store_true", help="Audit every CSV/mask/window without loading models")
    parser.add_argument("--max-windows", type=int, help="Stop after this many valid windows per CSV (partial smoke run)")
    parser.add_argument("--overwrite", action="store_true", help="Replace matching generated output files")
    args = parser.parse_args()
    if args.chunk_size < 1 or args.batch_size < 1 or args.num_threads < 0:
        parser.error("chunk-size and batch-size must be positive; num-threads must be nonnegative")
    if args.max_windows is not None and args.max_windows < 1:
        parser.error("max-windows must be positive")
    return args


def main() -> None:
    args = parse_args()
    pairs = find_inputs(args.inputs, args.exclude)
    suffix = "validation" if args.validate_only else "inference"
    # Check all masks/output collisions before model allocation or writing files.
    for csv_path, mask_path in pairs:
        read_mask(mask_path)
        expected_outputs = [args.output_dir / f"{csv_path.stem}__{suffix}_summary.json"]
        if not args.validate_only:
            expected_outputs.append(args.output_dir / f"{csv_path.stem}__center_predictions.csv")
        for output in expected_outputs:
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists; use --overwrite or another --output-dir: {output}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Selected inputs: " + ", ".join(csv_path.name for csv_path, _ in pairs), flush=True)
    predictor = None
    if not args.validate_only:
        from project.infer.models import FourStreamPredictor
        if args.num_threads:
            import torch
            torch.set_num_threads(args.num_threads)
        predictor = FourStreamPredictor(args.model_root, args.config_root, args.device, args.batch_size)
    results = [process_one(csv_path, mask_path, args, predictor) for csv_path, mask_path in pairs]
    write_json(args.output_dir / f"run_{suffix}_summary.json", {
        "status": "limited" if args.max_windows is not None else "complete",
        "mode": suffix,
        "files": [{"input_csv": item["input_csv"], "valid_windows": item["counts"]["valid_windows"],
                   "output_rows": item["output_rows"], "prediction_csv": item["prediction_csv"]} for item in results],
        "excluded_patterns": args.exclude,
        "total_valid_windows": sum(item["counts"]["valid_windows"] for item in results),
    })


if __name__ == "__main__":
    main()

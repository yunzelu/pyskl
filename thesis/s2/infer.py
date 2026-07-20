"""Run deterministic S2 center-window inference for Methods A, B, and C."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import (
        CENTER_OFFSET,
        DEFAULT_CONFIG_DIR,
        DEFAULT_JSONL_ROOT,
        DEFAULT_PREDICTION_DIR,
        FPS,
        LABELS,
        METHOD_A,
        METHOD_B,
        METHOD_C,
        METHODS,
        STRIDE,
        WINDOW_SIZE,
        S2FoldSpec,
        clean_group,
        clean_label,
        default_prediction_path,
        discover_s2_folds,
        eta_slug,
        logit_column_name,
        prob_column_name,
        protocol_metadata,
        read_json,
        selection_path,
        softmax,
        stage1_checkpoint_path,
        stage1_config_path,
        stage2_work_dir,
        write_json,
    )
except ImportError:
    from common import (
        CENTER_OFFSET,
        DEFAULT_CONFIG_DIR,
        DEFAULT_JSONL_ROOT,
        DEFAULT_PREDICTION_DIR,
        FPS,
        LABELS,
        METHOD_A,
        METHOD_B,
        METHOD_C,
        METHODS,
        STRIDE,
        WINDOW_SIZE,
        S2FoldSpec,
        clean_group,
        clean_label,
        default_prediction_path,
        discover_s2_folds,
        eta_slug,
        logit_column_name,
        prob_column_name,
        protocol_metadata,
        read_json,
        selection_path,
        softmax,
        stage1_checkpoint_path,
        stage1_config_path,
        stage2_work_dir,
        write_json,
    )

from infer_hpe_jsonl_timeline import load_jsonl_records, read_jsonl_frame_grid  # noqa: E402
from infer_late_fusion_timeline import StreamSpec, infer_fused_window_predictions  # noqa: E402
from infer_processed_pose_csv import resolve_device  # noqa: E402
from thesis.e2.common import discover_subject_sessions, find_checkpoint, score_column_name  # noqa: E402


def generated_config_path(
    config_dir: Path,
    method: str,
    fold: str,
    stream: str,
    eta: float | None,
) -> Path:
    method = method.upper()
    if method == METHOD_A:
        filename = "posec3d_trimmed_baseline_A.py"
    elif method == METHOD_B:
        filename = "posec3d_continuous_hard_B.py"
    elif method == METHOD_C:
        if eta is None:
            raise ValueError("Method C requires eta")
        filename = f"posec3d_continuous_soft_C_{eta_slug(eta)}.py"
    else:
        raise ValueError(method)
    return config_dir / f"fold_{fold}" / stream / filename


def parse_eta_arg(value: str | None) -> float | str | None:
    if value is None:
        return None
    text = value.strip().lower()
    if text == "selected":
        return "selected"
    if text.startswith("eta"):
        return int(text[3:]) / 100.0
    return float(text)


def selection_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = read_json(path)
    records = data.get("records", [])
    if isinstance(records, list):
        return [item for item in records if isinstance(item, dict)]
    selected = data.get("selected")
    if isinstance(selected, list):
        return [item for item in selected if isinstance(item, dict)]
    return []


def selected_record(
    selection_file: Path,
    method: str,
    fold: str,
    stream: str,
) -> dict[str, Any] | None:
    for record in selection_records(selection_file):
        if str(record.get("method", "")).upper() != method.upper():
            continue
        if str(record.get("fold", "")).lower() != fold.lower():
            continue
        if str(record.get("stream", "")) != stream:
            continue
        return record
    return None


def resolve_runtime_paths(
    fold: S2FoldSpec,
    method: str,
    stream: str,
    eta: float | str | None,
    config_dir: Path,
    selection_file: Path | None,
) -> tuple[Path, Path, float | None, dict[str, Any]]:
    method = method.upper()
    if method == METHOD_A:
        return (
            stage1_config_path(fold, stream),
            stage1_checkpoint_path(fold, stream),
            None,
            {"selection_source": "stage1_trimmed_checkpoint"},
        )

    record = selected_record(selection_file or selection_path(method, stream), method, fold.fold, stream)
    resolved_eta: float | None = None
    metadata: dict[str, Any] = {}
    if method == METHOD_C:
        if eta == "selected":
            if record is None:
                raise FileNotFoundError(
                    f"No selected Method C eta/checkpoint for fold {fold.fold} stream {stream}; "
                    f"run thesis/s2/select_stage2_checkpoint.py first or pass --eta."
                )
            resolved_eta = float(record["eta"])
            metadata["selection_source"] = str(selection_file or selection_path(method, stream))
        else:
            if eta is None:
                raise ValueError("Method C requires --eta or --eta selected")
            resolved_eta = float(eta)
            if record is not None and float(record.get("eta", -1.0)) != resolved_eta:
                record = None
            metadata["selection_source"] = "explicit_eta"
        config_path = generated_config_path(config_dir, method, fold.fold, stream, resolved_eta)
        work_dir = stage2_work_dir(method, fold.fold, stream, resolved_eta)
    elif method == METHOD_B:
        config_path = generated_config_path(config_dir, method, fold.fold, stream, None)
        work_dir = stage2_work_dir(method, fold.fold, stream)
        metadata["selection_source"] = "selected_checkpoint" if record else "best_macro_f1_marker"
    else:
        raise ValueError(method)

    if record is not None and record.get("checkpoint"):
        checkpoint_path = Path(str(record["checkpoint"]))
    else:
        checkpoint_path = find_checkpoint(work_dir)

    return config_path, checkpoint_path, resolved_eta, metadata


def prediction_fieldnames() -> list[str]:
    return [
        "model_variant",
        "method",
        "stream",
        "eta",
        "fold",
        "subject_id",
        "validation_subject",
        "test_subject",
        "recording_id",
        "jsonl_path",
        "start_frame",
        "end_frame",
        "center_frame",
        "center_timestamp",
        "raw_ground_truth_center_label",
        "ground_truth_center_label",
        "ground_truth_group",
        "predicted_label",
        "predicted_id",
        "confidence",
        "correct",
        "valid_detection_frames",
        "selected_detection_center",
        "checkpoint",
        "config",
        *[logit_column_name(label) for label in LABELS],
        *[prob_column_name(label) for label in LABELS],
        *[score_column_name(label) for label in LABELS],
    ]


def write_prediction_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_session(
    fold: S2FoldSpec,
    method: str,
    stream: str,
    eta: float | None,
    subject: str,
    jsonl_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    device: str,
    quiet: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata, frame_records = load_jsonl_records(jsonl_path)
    grid, img_shape = read_jsonl_frame_grid(
        jsonl_path=jsonl_path,
        kp_threshold=0.0,
        max_frames=None,
        trust_metadata_count=False,
    )
    if grid.total_frames < WINDOW_SIZE:
        return [], {
            "recording_id": jsonl_path.parent.name,
            "jsonl_path": str(jsonl_path),
            "total_frames": grid.total_frames,
            "total_windows": 0,
            "scored_windows": 0,
            "note": "skipped: shorter than 60 frames",
        }

    window_predictions, _covering_windows, total_windows = infer_fused_window_predictions(
        grid=grid,
        specs=[
            StreamSpec(
                name=stream,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                weight=1.0,
            )
        ],
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        num_classes=len(LABELS),
        batch_size=batch_size,
        device=device,
        img_shape=img_shape,
        min_valid_ratio=0.0,
        min_valid_frames=None,
        include_tail=False,
        normalize_weights=True,
        score_output="logit",
        quiet=quiet,
    )

    rows: list[dict[str, Any]] = []
    model_variant = method

    for prediction in sorted(window_predictions, key=lambda item: item.start):
        center = prediction.start + CENTER_OFFSET
        record = frame_records.get(center, {})
        raw_label_value = record.get("label") if isinstance(record, dict) else None
        raw_label = "" if raw_label_value is None else str(raw_label_value).strip()
        gt_label = clean_label(raw_label_value)
        gt_group = clean_group(record.get("label_group") if isinstance(record, dict) else None, gt_label)
        logits = np.asarray(prediction.scores, dtype=np.float32)
        probabilities = softmax(logits)
        pred_id = int(np.argmax(logits))
        confidence = float(probabilities[pred_id])
        row = {
            "model_variant": model_variant,
            "method": method,
            "stream": stream,
            "eta": "" if eta is None else f"{eta:.2f}",
            "fold": fold.fold,
            "subject_id": subject,
            "validation_subject": fold.val_subject,
            "test_subject": fold.test_subject,
            "recording_id": jsonl_path.parent.name,
            "jsonl_path": str(jsonl_path),
            "start_frame": prediction.start,
            "end_frame": prediction.start + WINDOW_SIZE - 1,
            "center_frame": center,
            "center_timestamp": f"{center / FPS:.6f}",
            "raw_ground_truth_center_label": raw_label,
            "ground_truth_center_label": gt_label or "",
            "ground_truth_group": gt_group,
            "predicted_label": LABELS[pred_id],
            "predicted_id": pred_id,
            "confidence": f"{confidence:.8f}",
            "correct": int(bool(gt_label) and gt_label == LABELS[pred_id]),
            "valid_detection_frames": prediction.valid_frames,
            "selected_detection_center": int(bool(grid.selected_detection[center])),
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
        }
        for label, value in zip(LABELS, logits):
            row[logit_column_name(label)] = f"{float(value):.8f}"
            row[score_column_name(label)] = f"{float(value):.8f}"
        for label, value in zip(LABELS, probabilities):
            row[prob_column_name(label)] = f"{float(value):.8f}"
        rows.append(row)

    video_info = metadata.get("video_info", {}) if isinstance(metadata, dict) else {}
    return rows, {
        "recording_id": jsonl_path.parent.name,
        "jsonl_path": str(jsonl_path),
        "total_frames": grid.total_frames,
        "img_shape": list(img_shape),
        "metadata_assumed_fps": video_info.get("assumed_fps_used_for_timestamp"),
        "s2_fps": FPS,
        "total_windows": total_windows,
        "scored_windows": len(window_predictions),
    }


def infer_method(
    method: str,
    stream: str,
    split: str,
    jsonl_root: Path,
    config_dir: Path,
    output: Path,
    eta: float | str | None,
    selection_file: Path | None,
    batch_size: int,
    device: str,
    folds: list[S2FoldSpec],
    overwrite: bool,
    quiet: bool,
) -> dict[str, Any]:
    method = method.upper()
    all_rows: list[dict[str, Any]] = []
    fold_infos: list[dict[str, Any]] = []
    session_infos: list[dict[str, Any]] = []

    for fold in folds:
        subject = fold.subject_for_split(split)
        sessions = discover_subject_sessions(jsonl_root, subject)
        config_path, checkpoint_path, resolved_eta, selection_meta = resolve_runtime_paths(
            fold=fold,
            method=method,
            stream=stream,
            eta=eta,
            config_dir=config_dir,
            selection_file=selection_file,
        )
        if not config_path.exists():
            raise FileNotFoundError(f"Missing S2 config: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        fold_infos.append(
            {
                "fold": fold.fold,
                "split": split,
                "active_subject": subject,
                "validation_subject": fold.val_subject,
                "test_subject": fold.test_subject,
                "method": method,
                "stream": stream,
                "eta": resolved_eta,
                "config": str(config_path),
                "checkpoint": str(checkpoint_path),
                **selection_meta,
                "sessions": [path.parent.name for path in sessions],
            }
        )
        if not quiet:
            print(
                f"[INFO] method={method} fold={fold.fold} split={split} "
                f"subject={subject} stream={stream} checkpoint={checkpoint_path.name}"
            )

        for jsonl_path in sessions:
            if not quiet:
                print(f"[INFO] inferring {jsonl_path.parent.name}")
            rows, info = infer_session(
                fold=fold,
                method=method,
                stream=stream,
                eta=resolved_eta,
                subject=subject,
                jsonl_path=jsonl_path,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                batch_size=batch_size,
                device=device,
                quiet=quiet,
            )
            all_rows.extend(rows)
            session_infos.append({"fold": fold.fold, "split": split, "subject": subject, **info})

    if not all_rows:
        raise ValueError("No prediction rows were produced")

    write_prediction_csv(output, all_rows, overwrite=overwrite)
    metadata = {
        "experiment": "S2",
        "stage": "inference",
        "method": method,
        "stream": stream,
        "split": split,
        "score_output": "logit_and_probability",
        "protocol": protocol_metadata(),
        "labels": LABELS,
        "folds": fold_infos,
        "sessions": session_infos,
        "prediction_csv": str(output),
        "rows": len(all_rows),
    }
    write_json(output.with_name(f"{output.stem}_metadata.json"), metadata, overwrite=overwrite)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S2 deterministic continuous-window inference.")
    parser.add_argument("--method", choices=list(METHODS), required=True)
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--eta", default="selected", help="Method C eta: selected, 0.25, 0.50, 0.75, or eta050.")
    parser.add_argument("--jsonl-root", type=Path, default=DEFAULT_JSONL_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--folds", nargs="+", help="Fold ids to include. Default: all.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    folds = discover_s2_folds()
    if args.folds:
        requested = {item.lower().replace("fold_", "") for item in args.folds}
        folds = [fold for fold in folds if fold.fold in requested]
        missing = sorted(requested - {fold.fold for fold in folds})
        if missing:
            raise ValueError(f"Unknown fold(s): {missing}")

    method = args.method.upper()
    eta = parse_eta_arg(args.eta) if method == METHOD_C else None
    output = args.output or default_prediction_path(method, args.stream)
    metadata = infer_method(
        method=method,
        stream=args.stream,
        split=args.split,
        jsonl_root=args.jsonl_root,
        config_dir=args.config_dir,
        output=output,
        eta=eta,
        selection_file=args.selection_file,
        batch_size=args.batch_size,
        device=resolve_device(args.device),
        folds=folds,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )
    print(f"[DONE] wrote {metadata['rows']} S2 prediction rows to {metadata['prediction_csv']}")


if __name__ == "__main__":
    main()

"""Select S2 Stage-2 checkpoints from continuous-validation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .common import (
        DEFAULT_ETAS,
        DEFAULT_SELECTION_DIR,
        LABELS,
        METHOD_B,
        METHOD_C,
        discover_s2_folds,
        eta_slug,
        read_log_json_records,
        safe_float,
        safe_int,
        selection_path,
        stage2_work_dir,
        write_json,
        write_rows_csv,
    )
except ImportError:
    from common import (
        DEFAULT_ETAS,
        DEFAULT_SELECTION_DIR,
        LABELS,
        METHOD_B,
        METHOD_C,
        discover_s2_folds,
        eta_slug,
        read_log_json_records,
        safe_float,
        safe_int,
        selection_path,
        stage2_work_dir,
        write_json,
        write_rows_csv,
    )


def log_json_paths(work_dir: Path) -> list[Path]:
    paths = sorted(work_dir.glob("*.log.json"))
    if not paths:
        raise FileNotFoundError(f"No .log.json found under {work_dir}")
    return paths


def latest_log_json(work_dir: Path) -> Path:
    return log_json_paths(work_dir)[-1]


def checkpoint_for_epoch(work_dir: Path, epoch: int) -> Path:
    checkpoint = work_dir / f"epoch_{epoch}.pth"
    if checkpoint.exists():
        return checkpoint
    best = sorted(work_dir.glob(f"*epoch_{epoch}.pth"))
    if best:
        return best[-1]
    raise FileNotFoundError(f"No checkpoint for epoch {epoch} under {work_dir}")


def val_records(work_dir: Path) -> list[dict[str, Any]]:
    records = []
    for log_path in log_json_paths(work_dir):
        for record in read_log_json_records(log_path):
            if record.get("mode") != "val":
                continue
            annotated = dict(record)
            annotated["_log_json"] = str(log_path)
            records.append(annotated)
    if not records:
        raise ValueError(f"No validation records found under {work_dir}")
    return records


def record_key(record: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        safe_float(record.get("macro_f1"), -1.0),
        safe_float(record.get("transition_macro_f1"), -1.0),
        safe_float(record.get("top1_acc"), -1.0),
        safe_int(record.get("epoch"), -1),
    )


def select_from_work_dir(
    method: str,
    fold: str,
    stream: str,
    work_dir: Path,
    eta: float | None,
) -> dict[str, Any]:
    log_paths = log_json_paths(work_dir)
    records = val_records(work_dir)
    selected = max(records, key=record_key)
    epoch = safe_int(selected.get("epoch"), -1)
    checkpoint = checkpoint_for_epoch(work_dir, epoch)
    return {
        "method": method,
        "fold": fold,
        "stream": stream,
        "eta": eta,
        "work_dir": str(work_dir),
        "log_json": str(selected.get("_log_json") or log_paths[-1]),
        "log_jsons": [str(path) for path in log_paths],
        "checkpoint": str(checkpoint),
        "selected_epoch": epoch,
        "validation_accuracy": safe_float(selected.get("top1_acc")),
        "validation_macro_f1": safe_float(selected.get("macro_f1")),
        "validation_transition_macro_f1": safe_float(selected.get("transition_macro_f1")),
        "selection_key": list(record_key(selected)),
    }


def training_log_rows(
    method: str,
    stream: str,
    selected_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_run = {
        (
            str(record["fold"]),
            "" if record.get("eta") is None else eta_slug(float(record["eta"])),
        ): record
        for record in selected_records
    }
    rows = []
    for key, selected in sorted(selected_by_run.items()):
        fold, eta_key = key
        rows_by_key: dict[tuple[str, int, int | str], dict[str, Any]] = {}
        log_paths = selected.get("log_jsons") or [selected["log_json"]]
        for log_path_value in log_paths:
            log_path = Path(str(log_path_value))
            for record in read_log_json_records(log_path):
                mode = str(record.get("mode", ""))
                if mode not in {"train", "val"}:
                    continue
                epoch = safe_int(record.get("epoch"), -1)
                iteration = safe_int(record.get("iter"), "")
                rows_by_key[(mode, epoch, iteration)] = {
                    "method": method,
                    "stream": stream,
                    "fold": fold,
                    "eta": eta_key,
                    "mode": mode,
                    "epoch": epoch,
                    "iter": iteration,
                    "training_loss": "" if mode != "train" else safe_float(record.get("loss")),
                    "training_loss_cls": "" if mode != "train" else safe_float(record.get("loss_cls")),
                    "validation_center_accuracy": "" if mode != "val" else safe_float(record.get("top1_acc")),
                    "validation_macro_f1": "" if mode != "val" else safe_float(record.get("macro_f1")),
                    "validation_transition_macro_f1": "" if mode != "val" else safe_float(record.get("transition_macro_f1")),
                    "learning_rate": safe_float(record.get("lr")),
                    "selected_epoch": int(epoch == int(selected["selected_epoch"])),
                    "selected_eta": int(bool(selected.get("selected_eta", True))),
                    "checkpoint": selected["checkpoint"] if epoch == int(selected["selected_epoch"]) else "",
                }
        rows.extend(
            rows_by_key[key]
            for key in sorted(
                rows_by_key,
                key=lambda item: (
                    item[1],
                    0 if item[0] == "train" else 1,
                    item[2] if isinstance(item[2], int) else -1,
                ),
            )
        )
    return rows


def select_method(
    method: str,
    stream: str,
    folds: list[str] | None,
    etas: tuple[float, ...],
) -> dict[str, Any]:
    method = method.upper()
    fold_specs = discover_s2_folds()
    if folds:
        requested = {fold.lower().replace("fold_", "") for fold in folds}
        fold_specs = [fold for fold in fold_specs if fold.fold in requested]
        missing = sorted(requested - {fold.fold for fold in fold_specs})
        if missing:
            raise ValueError(f"Unknown fold(s): {missing}")

    records: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    for fold in fold_specs:
        if method == METHOD_B:
            selected = select_from_work_dir(
                method=method,
                fold=fold.fold,
                stream=stream,
                work_dir=stage2_work_dir(method, fold.fold, stream),
                eta=None,
            )
            selected["selected_eta"] = True
            records.append(selected)
            all_candidates.append(selected)
            continue

        if method == METHOD_C:
            candidates = [
                select_from_work_dir(
                    method=method,
                    fold=fold.fold,
                    stream=stream,
                    work_dir=stage2_work_dir(method, fold.fold, stream, eta),
                    eta=eta,
                )
                for eta in etas
            ]
            selected = max(
                candidates,
                key=lambda item: (
                    safe_float(item.get("validation_macro_f1"), -1.0),
                    safe_float(item.get("validation_transition_macro_f1"), -1.0),
                    -abs(float(item["eta"]) - 0.50),
                    safe_int(item.get("selected_epoch"), -1),
                ),
            )
            for candidate in candidates:
                candidate["selected_eta"] = candidate is selected
            records.append(selected)
            all_candidates.extend(candidates)
            continue

        raise ValueError(f"Unsupported method: {method}")

    return {
        "experiment": "S2",
        "stage": "stage2_checkpoint_selection",
        "method": method,
        "stream": stream,
        "selection_rule": "maximize continuous validation macro_f1; tie-break by transition_macro_f1",
        "labels": LABELS,
        "records": records,
        "candidates": all_candidates,
    }


def write_training_log_csv(path: Path, method: str, stream: str, records: list[dict[str, Any]], overwrite: bool) -> None:
    fieldnames = [
        "method",
        "stream",
        "fold",
        "eta",
        "mode",
        "epoch",
        "iter",
        "training_loss",
        "training_loss_cls",
        "validation_center_accuracy",
        "validation_macro_f1",
        "validation_transition_macro_f1",
        "learning_rate",
        "selected_epoch",
        "selected_eta",
        "checkpoint",
    ]
    write_rows_csv(path, fieldnames, training_log_rows(method, stream, records), overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select S2 Stage-2 checkpoints.")
    parser.add_argument("--method", choices=[METHOD_B, METHOD_C], required=True)
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument("--folds", nargs="+", help="Fold ids to include. Default: all.")
    parser.add_argument("--etas", nargs="+", type=float, default=list(DEFAULT_ETAS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = select_method(
        method=args.method,
        stream=args.stream,
        folds=args.folds,
        etas=tuple(args.etas),
    )
    output = args.output or selection_path(args.method, args.stream)
    training_log = args.training_log or DEFAULT_SELECTION_DIR / f"training_log_{args.method}_{args.stream}.csv"
    write_json(output, result, overwrite=args.overwrite)
    write_training_log_csv(training_log, args.method, args.stream, result["candidates"], args.overwrite)
    print(f"[DONE] wrote selection to {output}")
    print(f"[DONE] wrote training log table to {training_log}")
    for record in result["records"]:
        eta = "" if record.get("eta") is None else f" eta={record['eta']:.2f}"
        print(
            f"fold={record['fold']} stream={record['stream']}{eta} "
            f"epoch={record['selected_epoch']} "
            f"macro_f1={record['validation_macro_f1']} "
            f"transition_macro_f1={record['validation_transition_macro_f1']}"
        )


if __name__ == "__main__":
    main()

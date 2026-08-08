"""Shared helpers for S6 skeleton-teacher pseudo-label export."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyskl.utils.temperature_scaling import fit_temperature, softmax_np
from pyskl.utils.uncertainty_metrics import predictive_quantities, validate_probabilities
from thesis.s6.common import (
    DEFAULT_CONTINUOUS_CONFIG_DIR,
    DEFAULT_CONTINUOUS_TEACHER_PKL,
    DEFAULT_CONTINUOUS_WORK_ROOT,
    LABELS,
    TeacherSpec,
    continuous_config_path,
    continuous_work_dir,
    selected_specs,
)
from tools.uncertainty.run_posec3d_mc_dropout import (
    assert_monotonic_by_sequence,
    assert_valid_labels,
    build_dataset_and_loader,
    deterministic_logits,
    load_config,
    mc_logits,
    sample_metadata,
)


DEFAULT_PSEUDO_OUT_DIR = Path("work_dirs/thesis/s6/pseudo_labels")
STREAMS = ("joint", "limb")


def add_common_args(parser: argparse.ArgumentParser, include_temperature: bool = False) -> None:
    parser.add_argument("--folds", nargs="+", default=["a", "b", "c"])
    parser.add_argument("--teachers", nargs="+", default=["t1", "t2", "t3", "t4"])
    parser.add_argument("--streams", nargs="+", default=["joint"])
    parser.add_argument("--ann-file", type=Path, default=DEFAULT_CONTINUOUS_TEACHER_PKL)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_CONTINUOUS_WORK_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONTINUOUS_CONFIG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PSEUDO_OUT_DIR)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    if include_temperature:
        parser.add_argument(
            "--temperature-mode",
            choices=("deterministic", "mc"),
            default="mc",
            help="How to fit T when temperature.json is missing or --refit-temperature is used.",
        )
        parser.add_argument("--refit-temperature", action="store_true")
        parser.add_argument("--num-passes", type=int, default=10)


def set_global_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selected_streams(values: list[str]) -> list[str]:
    streams = []
    for value in values:
        stream = value.lower()
        if stream not in STREAMS:
            raise ValueError(f"Unsupported stream {value!r}; expected one of {STREAMS}")
        streams.append(stream)
    return streams


def iter_requested_specs(args: argparse.Namespace) -> list[TeacherSpec]:
    return selected_specs(args.folds, args.teachers)


def output_dir_for(args: argparse.Namespace, spec: TeacherSpec, stream: str) -> Path:
    return args.out_dir / spec.fold_dir / spec.teacher / stream


def resolve_config(args: argparse.Namespace, spec: TeacherSpec, stream: str) -> Path:
    work_config = continuous_work_dir(spec, stream, args.work_root) / f"{stream}.py"
    generated_config = continuous_config_path(spec, stream, args.config_root)
    for path in (work_config, generated_config):
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No continuous config found for {spec.fold_dir}/{spec.teacher}/{stream}. "
        f"Checked {work_config} and {generated_config}."
    )


def _checkpoint_epoch(path: Path) -> int:
    name = path.stem
    if name.startswith("epoch_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return -1
    return -1


def resolve_checkpoint(args: argparse.Namespace, spec: TeacherSpec, stream: str) -> Path:
    work_dir = continuous_work_dir(spec, stream, args.work_root)
    if not work_dir.exists():
        raise FileNotFoundError(f"Missing continuous work dir: {work_dir}")

    for pattern in ("best_macro_f1*.pth", "best_*.pth"):
        candidates = sorted(work_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]

    latest = work_dir / "latest.pth"
    if latest.exists():
        return latest

    epochs = sorted(work_dir.glob("epoch_*.pth"), key=_checkpoint_epoch, reverse=True)
    if epochs:
        return epochs[0]
    raise FileNotFoundError(f"No checkpoint found in {work_dir}")


def run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def environment_summary() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "packages": {
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "mmcv": package_version("mmcv"),
            "pyskl": "local",
        },
    }


def write_json(path: Path, data: Any, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool, fieldnames: list[str] | None = None) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, overwrite: bool, **arrays: Any) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def label_column(label: str, prefix: str = "prob") -> str:
    slug = label.lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    return f"{prefix}_{slug}"


def sample_arrays(samples: list[Any]) -> dict[str, np.ndarray]:
    return {
        "subject_id": np.asarray([item.subject_id for item in samples]),
        "recording_id": np.asarray([item.recording_id for item in samples]),
        "sequence_id": np.asarray([item.sequence_id for item in samples]),
        "frame_dir": np.asarray([item.frame_dir for item in samples]),
        "window_start_frame": np.asarray([item.window_start_frame for item in samples], dtype=np.int64),
        "window_end_frame": np.asarray([item.window_end_frame for item in samples], dtype=np.int64),
        "center_frame": np.asarray([item.center_frame for item in samples], dtype=np.int64),
        "center_timestamp": np.asarray([item.center_timestamp for item in samples], dtype=np.float64),
        "manual_label_id": np.asarray([item.label_id for item in samples], dtype=np.int64),
        "manual_label_name": np.asarray([item.label_name for item in samples]),
        "label_group": np.asarray([item.label_group for item in samples]),
    }


def sample_row(spec: TeacherSpec, stream: str, item: Any) -> dict[str, Any]:
    return {
        "fold": spec.fold_dir,
        "teacher": spec.teacher,
        "stream": stream,
        "pseudo_split": spec.pseudo_split,
        "calibration_split": spec.calib_split,
        "sample_index": item.index,
        "subject_id": item.subject_id,
        "recording_id": item.recording_id,
        "sequence_id": item.sequence_id,
        "frame_dir": item.frame_dir,
        "window_start_frame": item.window_start_frame,
        "window_end_frame": item.window_end_frame,
        "center_frame": item.center_frame,
        "center_timestamp": f"{item.center_timestamp:.8f}",
        "manual_label_id": item.label_id,
        "manual_label_name": item.label_name,
        "label_group": item.label_group,
    }


def probability_rows(
    spec: TeacherSpec,
    stream: str,
    samples: list[Any],
    probabilities: np.ndarray,
    temperature: float | None = None,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    validate_probabilities(probabilities)
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    rows = []
    for index, (item, pred_id, confidence, probs) in enumerate(
        zip(samples, predictions, confidences, probabilities)
    ):
        row = sample_row(spec, stream, item)
        row.update(
            {
                "pseudo_label_id": int(pred_id),
                "pseudo_label_name": LABELS[int(pred_id)],
                "confidence": f"{float(confidence):.8f}",
            }
        )
        if temperature is not None:
            row["temperature"] = f"{float(temperature):.8f}"
        if extra_columns:
            for name, values in extra_columns.items():
                row[name] = f"{float(values[index]):.8f}"
        for class_id, label in enumerate(LABELS):
            row[label_column(label)] = f"{float(probs[class_id]):.8f}"
        rows.append(row)
    return rows


def hard_rows(
    spec: TeacherSpec,
    stream: str,
    samples: list[Any],
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    rows = probability_rows(spec, stream, samples, probabilities)
    keep = {
        "fold",
        "teacher",
        "stream",
        "pseudo_split",
        "calibration_split",
        "sample_index",
        "subject_id",
        "recording_id",
        "sequence_id",
        "frame_dir",
        "window_start_frame",
        "window_end_frame",
        "center_frame",
        "center_timestamp",
        "manual_label_id",
        "manual_label_name",
        "label_group",
        "pseudo_label_id",
        "pseudo_label_name",
        "confidence",
    }
    return [{key: value for key, value in row.items() if key in keep} for row in rows]


def build_loader_and_samples(
    config: Any,
    ann_file: Path,
    split: str,
    batch_size: int | None,
    num_workers: int | None,
) -> tuple[Any, list[Any], np.ndarray]:
    dataset, loader = build_dataset_and_loader(config, ann_file, split, batch_size, num_workers)
    samples = sample_metadata(dataset)
    labels = assert_valid_labels(samples, split)
    assert_monotonic_by_sequence(samples, split)
    return loader, samples, labels


def build_model(config: Any, checkpoint: Path, device: str) -> Any:
    from pyskl.apis import init_recognizer

    return init_recognizer(config, str(checkpoint), device=device)


def write_run_manifest(
    path: Path,
    args: argparse.Namespace,
    spec: TeacherSpec,
    stream: str,
    config_path: Path,
    checkpoint_path: Path,
    mode: str,
    sample_count: int,
    overwrite: bool,
    extra: dict[str, Any] | None = None,
) -> None:
    manifest = {
        "mode": mode,
        "fold": spec.fold_dir,
        "teacher": spec.teacher,
        "stream": stream,
        "pseudo_split": spec.pseudo_split,
        "pseudo_subjects": list(spec.pseudo_subjects),
        "calibration_split": spec.calib_split,
        "calibration_subject": spec.calibration_subject,
        "validation_split_not_loaded": spec.val_split,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "ann_file": str(args.ann_file),
        "out_dir": str(path.parent),
        "sample_count": int(sample_count),
        "command": " ".join(sys.argv),
        "environment": environment_summary(),
    }
    if extra:
        manifest.update(extra)
    write_json(path, manifest, overwrite=overwrite)


def deterministic_pseudo_logits(
    args: argparse.Namespace,
    spec: TeacherSpec,
    stream: str,
    split: str,
) -> tuple[np.ndarray, list[Any], np.ndarray, Path, Path]:
    config_path = resolve_config(args, spec, stream)
    checkpoint_path = resolve_checkpoint(args, spec, stream)
    config = load_config(config_path)
    loader, samples, labels = build_loader_and_samples(
        config,
        args.ann_file,
        split,
        args.batch_size,
        args.num_workers,
    )
    model = build_model(config, checkpoint_path, args.device)
    logits = deterministic_logits(model, loader, len(samples))
    return logits, samples, labels, config_path, checkpoint_path


def temperature_path(args: argparse.Namespace, spec: TeacherSpec, stream: str) -> Path:
    return output_dir_for(args, spec, stream) / "temperature.json"


def load_temperature(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["temperature"])


def fit_or_load_temperature(
    args: argparse.Namespace,
    spec: TeacherSpec,
    stream: str,
) -> tuple[float, dict[str, Any], Path, Path]:
    path = temperature_path(args, spec, stream)
    if path.exists() and not getattr(args, "refit_temperature", False):
        data = json.loads(path.read_text(encoding="utf-8"))
        config_value = data.get("config")
        checkpoint_value = data.get("checkpoint")
        config_path = Path(config_value) if config_value else resolve_config(args, spec, stream)
        checkpoint_path = Path(checkpoint_value) if checkpoint_value else resolve_checkpoint(args, spec, stream)
        return float(data["temperature"]), data, config_path, checkpoint_path

    temperature_mode = getattr(args, "temperature_mode", "mc")
    config_path = resolve_config(args, spec, stream)
    checkpoint_path = resolve_checkpoint(args, spec, stream)
    config = load_config(config_path)
    loader, samples, labels = build_loader_and_samples(
        config,
        args.ann_file,
        spec.calib_split,
        args.batch_size,
        args.num_workers,
    )
    model = build_model(config, checkpoint_path, args.device)
    if temperature_mode == "deterministic":
        logits = deterministic_logits(model, loader, len(samples))[:, None, :]
        dropout_info: list[dict[str, Any]] = []
        first_batch_diff = None
    else:
        logits, dropout_info, first_batch_diff = mc_logits(
            model,
            loader,
            len(samples),
            int(getattr(args, "num_passes", 10)),
        )

    fit = fit_temperature(logits, labels, num_bins=args.ece_bins)
    record = {
        **fit,
        "temperature_mode": temperature_mode,
        "num_passes": int(getattr(args, "num_passes", 1)) if temperature_mode == "mc" else 1,
        "fold": spec.fold_dir,
        "teacher": spec.teacher,
        "stream": stream,
        "calibration_split": spec.calib_split,
        "calibration_subject": spec.calibration_subject,
        "calibration_sample_count": len(samples),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "ann_file": str(args.ann_file),
        "dropout_modules": dropout_info,
        "mc_first_batch_logit_mean_abs_diff": first_batch_diff,
        "command": " ".join(sys.argv),
        "environment": environment_summary(),
    }
    write_json(path, record, overwrite=True)
    save_npz(
        output_dir_for(args, spec, stream) / "calibration_temperature_logits.npz",
        overwrite=True,
        logits=logits.astype(np.float32, copy=False),
        labels=labels,
        **sample_arrays(samples),
    )
    write_csv(
        output_dir_for(args, spec, stream) / "calibration_samples.csv",
        [sample_row(spec, stream, item) for item in samples],
        overwrite=True,
    )
    return float(record["temperature"]), record, config_path, checkpoint_path


def export_hard(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    for spec in iter_requested_specs(args):
        for stream in selected_streams(args.streams):
            out_dir = output_dir_for(args, spec, stream)
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Hard pseudo labels: {spec.fold_dir} {spec.teacher} {stream}")
            logits, samples, labels, config_path, checkpoint_path = deterministic_pseudo_logits(
                args,
                spec,
                stream,
                spec.pseudo_split,
            )
            probs = softmax_np(logits)
            rows = hard_rows(spec, stream, samples, probs)
            write_csv(out_dir / "deterministic_hard_pseudo_labels.csv", rows, overwrite=args.overwrite)
            save_npz(
                out_dir / "deterministic_hard_pseudo_labels.npz",
                overwrite=args.overwrite,
                logits=logits.astype(np.float32, copy=False),
                probabilities=probs.astype(np.float32, copy=False),
                pseudo_label_id=np.argmax(probs, axis=1).astype(np.int64),
                **sample_arrays(samples),
            )
            write_run_manifest(
                out_dir / "deterministic_hard_manifest.json",
                args,
                spec,
                stream,
                config_path,
                checkpoint_path,
                "deterministic_hard",
                len(samples),
                overwrite=args.overwrite,
            )


def export_raw_soft(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    for spec in iter_requested_specs(args):
        for stream in selected_streams(args.streams):
            out_dir = output_dir_for(args, spec, stream)
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Raw soft probabilities: {spec.fold_dir} {spec.teacher} {stream}")
            logits, samples, labels, config_path, checkpoint_path = deterministic_pseudo_logits(
                args,
                spec,
                stream,
                spec.pseudo_split,
            )
            probs = softmax_np(logits)
            rows = probability_rows(spec, stream, samples, probs)
            write_csv(out_dir / "raw_soft_probabilities.csv", rows, overwrite=args.overwrite)
            save_npz(
                out_dir / "raw_soft_probabilities.npz",
                overwrite=args.overwrite,
                logits=logits.astype(np.float32, copy=False),
                probabilities=probs.astype(np.float32, copy=False),
                pseudo_label_id=np.argmax(probs, axis=1).astype(np.int64),
                **sample_arrays(samples),
            )
            write_run_manifest(
                out_dir / "raw_soft_manifest.json",
                args,
                spec,
                stream,
                config_path,
                checkpoint_path,
                "raw_soft",
                len(samples),
                overwrite=args.overwrite,
            )


def export_calibrated_soft(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    for spec in iter_requested_specs(args):
        for stream in selected_streams(args.streams):
            out_dir = output_dir_for(args, spec, stream)
            out_dir.mkdir(parents=True, exist_ok=True)
            temperature, temperature_record, _, _ = fit_or_load_temperature(args, spec, stream)
            print(
                f"[INFO] Calibrated soft probabilities: {spec.fold_dir} {spec.teacher} "
                f"{stream}, T={temperature:.6f}"
            )
            logits, samples, labels, config_path, checkpoint_path = deterministic_pseudo_logits(
                args,
                spec,
                stream,
                spec.pseudo_split,
            )
            probs = softmax_np(logits, temperature=temperature)
            rows = probability_rows(spec, stream, samples, probs, temperature=temperature)
            write_csv(out_dir / "calibrated_soft_probabilities.csv", rows, overwrite=args.overwrite)
            save_npz(
                out_dir / "calibrated_soft_probabilities.npz",
                overwrite=args.overwrite,
                logits=logits.astype(np.float32, copy=False),
                probabilities=probs.astype(np.float32, copy=False),
                temperature=np.asarray([temperature], dtype=np.float32),
                pseudo_label_id=np.argmax(probs, axis=1).astype(np.int64),
                **sample_arrays(samples),
            )
            write_run_manifest(
                out_dir / "calibrated_soft_manifest.json",
                args,
                spec,
                stream,
                config_path,
                checkpoint_path,
                "calibrated_soft",
                len(samples),
                overwrite=args.overwrite,
                extra={"temperature": temperature, "temperature_record": temperature_record},
            )


def export_mc_calibrated_soft(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    for spec in iter_requested_specs(args):
        for stream in selected_streams(args.streams):
            out_dir = output_dir_for(args, spec, stream)
            out_dir.mkdir(parents=True, exist_ok=True)
            temperature, temperature_record, _, _ = fit_or_load_temperature(args, spec, stream)
            print(
                f"[INFO] MC calibrated probabilities: {spec.fold_dir} {spec.teacher} "
                f"{stream}, K={args.num_passes}, T={temperature:.6f}"
            )
            config_path = resolve_config(args, spec, stream)
            checkpoint_path = resolve_checkpoint(args, spec, stream)
            config = load_config(config_path)
            loader, samples, labels = build_loader_and_samples(
                config,
                args.ann_file,
                spec.pseudo_split,
                args.batch_size,
                args.num_workers,
            )
            model = build_model(config, checkpoint_path, args.device)
            logits, dropout_info, first_batch_diff = mc_logits(model, loader, len(samples), args.num_passes)
            prob_passes = softmax_np(logits, temperature=temperature)
            quantities = predictive_quantities(prob_passes)
            rows = probability_rows(
                spec,
                stream,
                samples,
                quantities["probabilities"],
                temperature=temperature,
                extra_columns={
                    "predictive_entropy": quantities["predictive_entropy"],
                    "expected_entropy": quantities["expected_entropy"],
                    "mutual_information": quantities["mutual_information"],
                    "variation_ratio": quantities["variation_ratio"],
                    "mean_probability_variance": quantities["mean_probability_variance"],
                },
            )
            write_csv(out_dir / "mc_calibrated_soft_probabilities.csv", rows, overwrite=args.overwrite)
            arrays = {
                "probabilities": quantities["probabilities"].astype(np.float32, copy=False),
                "probability_passes": prob_passes.astype(np.float32, copy=False),
                "temperature": np.asarray([temperature], dtype=np.float32),
                "pseudo_label_id": quantities["prediction"].astype(np.int64, copy=False),
                "predictive_entropy": quantities["predictive_entropy"].astype(np.float32, copy=False),
                "expected_entropy": quantities["expected_entropy"].astype(np.float32, copy=False),
                "mutual_information": quantities["mutual_information"].astype(np.float32, copy=False),
                "variation_ratio": quantities["variation_ratio"].astype(np.float32, copy=False),
                "mean_probability_variance": quantities["mean_probability_variance"].astype(np.float32, copy=False),
                **sample_arrays(samples),
            }
            if args.save_mc_logits:
                arrays["logits"] = logits.astype(np.float32, copy=False)
            save_npz(out_dir / "mc_calibrated_soft_probabilities.npz", overwrite=args.overwrite, **arrays)
            write_run_manifest(
                out_dir / "mc_calibrated_soft_manifest.json",
                args,
                spec,
                stream,
                config_path,
                checkpoint_path,
                "mc_calibrated_soft",
                len(samples),
                overwrite=args.overwrite,
                extra={
                    "temperature": temperature,
                    "temperature_record": temperature_record,
                    "num_passes": args.num_passes,
                    "dropout_modules": dropout_info,
                    "mc_first_batch_logit_mean_abs_diff": first_batch_diff,
                },
            )


def fit_temperatures(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    for spec in iter_requested_specs(args):
        for stream in selected_streams(args.streams):
            temperature, _, _, _ = fit_or_load_temperature(args, spec, stream)
            print(f"[INFO] Temperature: {spec.fold_dir} {spec.teacher} {stream} T={temperature:.6f}")

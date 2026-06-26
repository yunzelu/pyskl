"""Train a small mixture-of-experts gate from four stream prediction CSVs.

Manifest format:

    session,origin_session,j,jm,b,bm
    35-mia-sit,data/radar_v4/origin/35-mia-sit,path/to/j.csv,path/to/jm.csv,path/to/b.csv,path/to/bm.csv

Each stream CSV must be in the frame-level prediction format written by the
radar v4 inference scripts and must contain score_<label> columns.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_LABEL_MAP,
    DEFAULT_STREAMS,
    fuse_expert_probs,
    load_label_map,
    load_session_examples,
    make_gate_model,
    metrics_from_predictions,
    read_manifest,
    split_values,
    write_json,
)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def collect_examples(
    manifest_path: Path,
    labels: list[str],
    streams: list[str],
    require_detection: bool,
    require_prediction: bool,
    frame_stride: int,
) -> tuple[list[list[float]], list[int], dict[str, Any]]:
    specs = read_manifest(manifest_path, streams=streams, require_origin=True)
    features: list[list[float]] = []
    targets: list[int] = []
    summary: dict[str, Any] = {
        "manifest": manifest_path,
        "sessions": [],
        "total_examples": 0,
        "skipped": {
            "no_gt": 0,
            "no_detection": 0,
            "no_prediction": 0,
            "bad_scores": 0,
        },
    }

    for spec in specs:
        examples = load_session_examples(
            spec=spec,
            labels=labels,
            streams=streams,
            require_detection=require_detection,
            require_prediction=require_prediction,
        )
        session_features = examples.features[::frame_stride]
        session_targets = examples.labels[::frame_stride]
        features.extend(session_features)
        targets.extend(session_targets)

        for key, value in examples.skipped.items():
            summary["skipped"][key] += value
        summary["sessions"].append(
            {
                "session": spec.session,
                "origin_session": spec.origin_session,
                "examples": len(session_features),
                "raw_examples": len(examples.features),
                "skipped": examples.skipped,
            }
        )

    summary["total_examples"] = len(features)
    return features, targets, summary


def evaluate_gate(
    gate: Any,
    features: Any,
    targets: Any,
    batch_size: int,
    num_streams: int,
    num_classes: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    gate.eval()
    losses: list[float] = []
    predictions: list[int] = []
    target_values: list[int] = []
    alpha_sum = torch.zeros(num_streams, dtype=torch.float64, device=features.device)
    alpha_count = 0

    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch_x = features[start : start + batch_size]
            batch_y = targets[start : start + batch_size]
            gate_logits = gate(batch_x)
            fused, alpha = fuse_expert_probs(
                features=batch_x,
                gate_logits=gate_logits,
                num_streams=num_streams,
                num_classes=num_classes,
            )
            loss = F.nll_loss(torch.log(fused.clamp_min(1e-8)), batch_y)
            losses.append(float(loss.item()) * int(batch_x.shape[0]))
            predictions.extend(torch.argmax(fused, dim=1).cpu().tolist())
            target_values.extend(batch_y.cpu().tolist())
            alpha_sum += alpha.double().sum(dim=0)
            alpha_count += int(alpha.shape[0])

    metrics = metrics_from_predictions(
        predictions=predictions,
        targets=target_values,
        num_classes=num_classes,
    )
    metrics["nll"] = sum(losses) / max(1, features.shape[0])
    metrics["mean_alpha"] = (
        (alpha_sum / max(1, alpha_count)).detach().cpu().tolist()
        if alpha_count
        else [0.0] * num_streams
    )
    return metrics


def train_gate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    labels = load_label_map(args.label_map)
    streams = split_values(args.streams)
    if len(streams) < 2:
        raise ValueError("MoE training needs at least two streams")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    num_classes = len(labels)
    input_dim = len(streams) * num_classes

    train_features, train_targets, train_summary = collect_examples(
        manifest_path=args.manifest,
        labels=labels,
        streams=streams,
        require_detection=not args.keep_empty_frames,
        require_prediction=not args.keep_no_prediction_frames,
        frame_stride=args.frame_stride,
    )
    if not train_features:
        raise ValueError("No training examples were loaded")

    eval_features = None
    eval_targets = None
    eval_summary = None
    if args.eval_manifest:
        eval_features, eval_targets, eval_summary = collect_examples(
            manifest_path=args.eval_manifest,
            labels=labels,
            streams=streams,
            require_detection=not args.keep_empty_frames,
            require_prediction=not args.keep_no_prediction_frames,
            frame_stride=args.frame_stride,
        )
        if not eval_features:
            raise ValueError("No evaluation examples were loaded")

    if args.max_examples and args.max_examples > 0 and len(train_features) > args.max_examples:
        indices = list(range(len(train_features)))
        random.shuffle(indices)
        indices = sorted(indices[: args.max_examples])
        train_features = [train_features[index] for index in indices]
        train_targets = [train_targets[index] for index in indices]

    x_train = torch.tensor(train_features, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_targets, dtype=torch.long, device=device)
    x_eval = (
        torch.tensor(eval_features, dtype=torch.float32, device=device)
        if eval_features is not None
        else None
    )
    y_eval = (
        torch.tensor(eval_targets, dtype=torch.long, device=device)
        if eval_targets is not None
        else None
    )

    gate = make_gate_model(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_streams=len(streams),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        gate.train()
        order = torch.randperm(x_train.shape[0], device=device)
        total_loss = 0.0

        for start in range(0, x_train.shape[0], args.batch_size):
            batch_indices = order[start : start + args.batch_size]
            batch_x = x_train[batch_indices]
            batch_y = y_train[batch_indices]

            optimizer.zero_grad(set_to_none=True)
            gate_logits = gate(batch_x)
            fused, _alpha = fuse_expert_probs(
                features=batch_x,
                gate_logits=gate_logits,
                num_streams=len(streams),
                num_classes=num_classes,
            )
            loss = F.nll_loss(torch.log(fused.clamp_min(1e-8)), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(batch_x.shape[0])

        train_metrics = evaluate_gate(
            gate=gate,
            features=x_train,
            targets=y_train,
            batch_size=args.batch_size,
            num_streams=len(streams),
            num_classes=num_classes,
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, x_train.shape[0]),
            "train": train_metrics,
        }
        monitor = train_metrics["nll"]

        if x_eval is not None and y_eval is not None:
            eval_metrics = evaluate_gate(
                gate=gate,
                features=x_eval,
                targets=y_eval,
                batch_size=args.batch_size,
                num_streams=len(streams),
                num_classes=num_classes,
            )
            record["eval"] = eval_metrics
            monitor = eval_metrics["nll"]

        history.append(record)
        if not args.quiet and (epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs):
            message = (
                f"[INFO] epoch={epoch} train_nll={train_metrics['nll']:.6f} "
                f"train_acc={train_metrics['accuracy']:.4f}"
            )
            if "eval" in record:
                message += (
                    f" eval_nll={record['eval']['nll']:.6f} "
                    f"eval_acc={record['eval']['accuracy']:.4f}"
                )
            print(message)

        if monitor < best_metric - args.min_delta:
            best_metric = monitor
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in gate.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if args.patience > 0 and stale_epochs >= args.patience:
            if not args.quiet:
                print(f"[INFO] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        gate.load_state_dict(best_state)

    final_train = evaluate_gate(
        gate=gate,
        features=x_train,
        targets=y_train,
        batch_size=args.batch_size,
        num_streams=len(streams),
        num_classes=num_classes,
    )
    final_eval = None
    if x_eval is not None and y_eval is not None:
        final_eval = evaluate_gate(
            gate=gate,
            features=x_eval,
            targets=y_eval,
            batch_size=args.batch_size,
            num_streams=len(streams),
            num_classes=num_classes,
        )

    checkpoint = {
        "state_dict": gate.state_dict(),
        "labels": labels,
        "streams": streams,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_classes": num_classes,
        "num_streams": len(streams),
        "best_epoch": best_epoch,
        "best_monitor_nll": best_metric,
        "train_metrics": final_train,
        "eval_metrics": final_eval,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    result = {
        "checkpoint": args.output,
        "labels": labels,
        "streams": streams,
        "best_epoch": best_epoch,
        "train_summary": train_summary,
        "eval_summary": eval_summary,
        "train_metrics": final_train,
        "eval_metrics": final_eval,
        "history": history,
        "args": vars(args),
    }
    if args.metrics_json:
        write_json(args.metrics_json, result, overwrite=args.overwrite)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a radar v4 MoE gate from aligned stream prediction CSVs."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output .pt gate checkpoint.")
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument(
        "--streams",
        default=",".join(DEFAULT_STREAMS),
        help="Comma- or colon-separated stream columns. Default: j,jm,b,bm.",
    )
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--keep-empty-frames",
        action="store_true",
        help="Use frames where the template stream has selected_detection=0.",
    )
    parser.add_argument(
        "--keep-no-prediction-frames",
        action="store_true",
        help="Use frames where one or more streams have no model prediction.",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    result = train_gate(args)
    if not args.quiet:
        print(f"[DONE] Wrote gate checkpoint to {args.output}")
        print(
            f"[DONE] train_acc={result['train_metrics']['accuracy']:.4f} "
            f"train_mca={result['train_metrics']['mean_class_accuracy']:.4f}"
        )
        if result["eval_metrics"]:
            print(
                f"[DONE] eval_acc={result['eval_metrics']['accuracy']:.4f} "
                f"eval_mca={result['eval_metrics']['mean_class_accuracy']:.4f}"
            )


if __name__ == "__main__":
    main()

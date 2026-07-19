"""Shared helpers for E4 HMM/Viterbi temporal refinement."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.e2.common import (  # noqa: E402
    DEFAULT_JSONL_ROOT,
    DEFAULT_CONFIG_ROOT,
    DEFAULT_WORK_ROOT,
    LABELS,
    ScoreRow,
    protocol_metadata as e2_protocol_metadata,
    read_score_csv,
    write_json,
    write_score_csv,
)
from thesis.e3.common import (  # noqa: E402
    DEFAULT_LOGIT_DIR as DEFAULT_E3_LOGIT_DIR,
    DEFAULT_TEMPERATURE_DIR as DEFAULT_E3_TEMPERATURE_DIR,
    softmax,
)

DEFAULT_OUTPUT_DIR = Path("work_dirs/thesis/e4")
DEFAULT_LOGIT_DIR = DEFAULT_OUTPUT_DIR / "logits"
DEFAULT_SCORE_DIR = DEFAULT_OUTPUT_DIR / "scores"
DEFAULT_TUNE_DIR = DEFAULT_OUTPUT_DIR / "tuning"
DEFAULT_EVAL_DIR = DEFAULT_OUTPUT_DIR / "eval"

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

LYING = "Lying-Stationary"
SITTING = "Sit-Stationary"
STANDING_WALKING = "Walking"

VERTICES = [LYING, SITTING, STANDING_WALKING]

# With no separate standing class in the 9-class skeleton label set, Walking is
# used as the standing/walking vertex for stand-related transitions.
EDGE_ENDPOINTS = {
    "Falling": (STANDING_WALKING, LYING),
    "Transition-LayBed-to-Sit": (LYING, SITTING),
    "Transition-LayFloor-to-Stand": (LYING, STANDING_WALKING),
    "Transition-Sit-to-LayBed": (SITTING, LYING),
    "Transition-Sit-to-Stand": (SITTING, STANDING_WALKING),
    "Transition-Stand-to-Sit": (STANDING_WALKING, SITTING),
}


def protocol_metadata() -> dict[str, Any]:
    data = e2_protocol_metadata()
    data.update(
        {
            "experiment": "E4",
            "temporal_model": "first-order HMM with Viterbi decoding",
            "objective": "sum_t log p_T(y_t | x_t) + lambda sum_t log A[y_{t-1}, y_t]",
            "transition_matrix": "manual RADAR v4 9-class topology",
        }
    )
    return data


def outgoing_edges(vertex: str) -> list[str]:
    return [edge for edge, (start, _end) in EDGE_ENDPOINTS.items() if start == vertex]


def incoming_edges(vertex: str) -> list[str]:
    return [edge for edge, (_start, end) in EDGE_ENDPOINTS.items() if end == vertex]


def connected_edges(label: str) -> set[str]:
    if label in VERTICES:
        return set(outgoing_edges(label) + incoming_edges(label))

    start, end = EDGE_ENDPOINTS[label]
    return {
        edge
        for edge, (edge_start, edge_end) in EDGE_ENDPOINTS.items()
        if edge_start in {start, end} or edge_end in {start, end}
    }


def transition_level(from_label: str, to_label: str) -> str:
    if from_label in VERTICES:
        valid = {from_label, *outgoing_edges(from_label)}
        if to_label in valid:
            return "valid"
        if to_label in incoming_edges(from_label):
            return "soft_impossible"
        return "hard_impossible"

    start, end = EDGE_ENDPOINTS[from_label]
    if to_label == from_label or to_label == end:
        return "valid"
    if to_label == start or to_label in connected_edges(from_label):
        return "soft_impossible"
    return "hard_impossible"


def manual_transition_matrix(
    vertex_valid_weight: float = 1.0,
    transition_self_weight: float = 0.4,
    transition_end_weight: float = 0.6,
    soft_impossible_weight: float = 1e-3,
    hard_impossible_weight: float = 1e-6,
) -> np.ndarray:
    if min(
        vertex_valid_weight,
        transition_self_weight,
        transition_end_weight,
        soft_impossible_weight,
        hard_impossible_weight,
    ) <= 0:
        raise ValueError("All transition weights must be positive.")

    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.float64)

    for from_label in LABELS:
        from_id = LABEL_TO_ID[from_label]
        weights = np.full(len(LABELS), hard_impossible_weight, dtype=np.float64)

        if from_label in VERTICES:
            for edge in incoming_edges(from_label):
                weights[LABEL_TO_ID[edge]] = soft_impossible_weight
            for to_label in [from_label, *outgoing_edges(from_label)]:
                weights[LABEL_TO_ID[to_label]] = vertex_valid_weight
        else:
            _start, end = EDGE_ENDPOINTS[from_label]
            for edge in connected_edges(from_label):
                weights[LABEL_TO_ID[edge]] = soft_impossible_weight
            weights[LABEL_TO_ID[EDGE_ENDPOINTS[from_label][0]]] = soft_impossible_weight
            weights[LABEL_TO_ID[from_label]] = transition_self_weight
            weights[LABEL_TO_ID[end]] = transition_end_weight

        matrix[from_id] = weights / np.sum(weights)

    return matrix


def transition_matrix_rows(matrix: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for from_id, from_label in enumerate(LABELS):
        for to_id, to_label in enumerate(LABELS):
            rows.append(
                {
                    "from_id": from_id,
                    "from_label": from_label,
                    "to_id": to_id,
                    "to_label": to_label,
                    "level": transition_level(from_label, to_label),
                    "probability": float(matrix[from_id, to_id]),
                }
            )
    return rows


def write_transition_matrix(output_dir: Path, overwrite: bool) -> dict[str, str]:
    matrix = manual_transition_matrix()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "e4_manual_transition_matrix.csv"
    json_path = output_dir / "e4_manual_transition_matrix.json"

    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"{csv_path} exists; pass --overwrite to replace it")
    if json_path.exists() and not overwrite:
        raise FileExistsError(f"{json_path} exists; pass --overwrite to replace it")

    rows = transition_matrix_rows(matrix)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "probability": f"{row['probability']:.12f}",
                }
            )

    write_json(
        json_path,
        {
            "experiment": "E4",
            "stage": "transition_matrix",
            "labels": LABELS,
            "vertices": VERTICES,
            "edge_endpoints": EDGE_ENDPOINTS,
            "rows_sum_to_one": [float(value) for value in matrix.sum(axis=1)],
            "matrix": matrix.tolist(),
            "levels": {
                "valid": "self/outgoing edge from state, or transition self/end state",
                "soft_impossible": "topologically connected but wrong direction or too abrupt",
                "hard_impossible": "unconnected opposite-side transition",
            },
        },
        overwrite=overwrite,
    )
    return {"csv": str(csv_path), "json": str(json_path)}


def load_temperatures(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    temperatures = data.get("temperatures", {})
    if not isinstance(temperatures, dict) or not temperatures:
        raise ValueError(f"{path} has no temperatures")

    output: dict[str, float] = {}
    for fold, item in temperatures.items():
        value = item.get("temperature") if isinstance(item, dict) else None
        if value is None:
            raise ValueError(f"{path} has no temperature for fold {fold}")
        temperature = float(value)
        if temperature <= 0:
            raise ValueError(f"Temperature for fold {fold} must be positive")
        output[str(fold)] = temperature
    return output


def probability_rows_from_logits(
    rows: list[ScoreRow],
    temperatures: dict[str, float] | None,
    source_suffix: str,
) -> list[ScoreRow]:
    probability_rows: list[ScoreRow] = []
    for row in rows:
        if row.score_type != "logit":
            raise ValueError(f"Probability materialization requires logits, got {row.score_type!r}")
        if temperatures is not None and row.fold not in temperatures:
            raise ValueError(f"No temperature found for fold {row.fold}")

        temperature = 1.0 if temperatures is None else temperatures[row.fold]
        probabilities = softmax(row.scores.astype(np.float64) / temperature)
        probability_rows.append(
            ScoreRow(
                source=f"{row.source}_{source_suffix}",
                score_type="prob",
                fold=row.fold,
                test_subject=row.test_subject,
                session=row.session,
                jsonl_path=row.jsonl_path,
                window_start=row.window_start,
                window_end=row.window_end,
                center_frame=row.center_frame,
                center_time_sec=row.center_time_sec,
                raw_gt_label=row.raw_gt_label,
                gt_label=row.gt_label,
                gt_group=row.gt_group,
                valid_detection_frames=row.valid_detection_frames,
                selected_detection_center=row.selected_detection_center,
                scores=np.asarray(probabilities, dtype=np.float32),
            )
        )
    return probability_rows


def raw_probability_rows(rows: list[ScoreRow]) -> list[ScoreRow]:
    return probability_rows_from_logits(rows, temperatures=None, source_suffix="raw_softmax")


def calibrate_logit_rows(rows: list[ScoreRow], temperatures: dict[str, float]) -> list[ScoreRow]:
    return probability_rows_from_logits(rows, temperatures=temperatures, source_suffix="calibrated")


def grouped_sequences(rows: list[ScoreRow]) -> list[list[ScoreRow]]:
    grouped: dict[tuple[str, str], list[ScoreRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.fold, row.session)].append(row)
    return [
        sorted(grouped[key], key=lambda item: item.window_start)
        for key in sorted(grouped)
    ]


def viterbi_decode(probabilities: np.ndarray, transition_matrix: np.ndarray, lam: float) -> np.ndarray:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS):
        raise ValueError(f"Expected probabilities with shape (T, {len(LABELS)})")
    if probabilities.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if lam < 0:
        raise ValueError("lambda must be non-negative")

    eps = 1e-12
    emissions = np.log(np.clip(probabilities, eps, 1.0))
    transitions = np.log(np.clip(transition_matrix, eps, 1.0))

    num_steps, num_classes = emissions.shape
    scores = np.zeros((num_steps, num_classes), dtype=np.float64)
    backpointers = np.zeros((num_steps, num_classes), dtype=np.int64)
    scores[0] = emissions[0]

    for step in range(1, num_steps):
        candidates = scores[step - 1, :, None] + lam * transitions
        backpointers[step] = np.argmax(candidates, axis=0)
        scores[step] = candidates[backpointers[step], np.arange(num_classes)] + emissions[step]

    path = np.zeros(num_steps, dtype=np.int64)
    path[-1] = int(np.argmax(scores[-1]))
    for step in range(num_steps - 2, -1, -1):
        path[step] = backpointers[step + 1, path[step + 1]]
    return path


def decoded_rows_from_path(rows: list[ScoreRow], path: np.ndarray, source: str) -> list[ScoreRow]:
    decoded_rows: list[ScoreRow] = []
    for row, class_id in zip(rows, path):
        scores = np.zeros(len(LABELS), dtype=np.float32)
        scores[int(class_id)] = 1.0
        decoded_rows.append(
            ScoreRow(
                source=source,
                score_type="decoded",
                fold=row.fold,
                test_subject=row.test_subject,
                session=row.session,
                jsonl_path=row.jsonl_path,
                window_start=row.window_start,
                window_end=row.window_end,
                center_frame=row.center_frame,
                center_time_sec=row.center_time_sec,
                raw_gt_label=row.raw_gt_label,
                gt_label=row.gt_label,
                gt_group=row.gt_group,
                valid_detection_frames=row.valid_detection_frames,
                selected_detection_center=row.selected_detection_center,
                scores=scores,
            )
        )
    return decoded_rows


def apply_viterbi(rows: list[ScoreRow], transition_matrix: np.ndarray, lam: float) -> list[ScoreRow]:
    decoded: list[ScoreRow] = []
    source = f"{rows[0].source}_viterbi_lam_{lam:g}" if rows else f"viterbi_lam_{lam:g}"
    for sequence_rows in grouped_sequences(rows):
        probabilities = np.stack([row.scores for row in sequence_rows]).astype(np.float64, copy=False)
        path = viterbi_decode(probabilities, transition_matrix, lam)
        decoded.extend(decoded_rows_from_path(sequence_rows, path, source=source))
    return decoded


def parse_lambda_grid(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise ValueError("lambda grid values must be non-negative")
        values.append(value)

    if not values:
        raise ValueError("lambda grid is empty")
    return sorted(set(values))


def default_val_logits_path(stream: str) -> Path:
    return DEFAULT_LOGIT_DIR / f"e4_{stream}_val_logits.csv"


def default_calibrated_path(stream: str, split: str) -> Path:
    return DEFAULT_SCORE_DIR / f"e4_{stream}_{split}_calibrated_probs.csv"


def default_raw_path(stream: str, split: str) -> Path:
    return DEFAULT_SCORE_DIR / f"e4_{stream}_{split}_raw_probs.csv"


def default_scores_path(stream: str, split: str, score_kind: str) -> Path:
    if score_kind == "raw":
        return default_raw_path(stream, split)
    if score_kind == "calibrated":
        return default_calibrated_path(stream, split)
    raise ValueError(f"Unsupported score kind: {score_kind}")


def default_viterbi_path(stream: str, score_kind: str = "calibrated") -> Path:
    return DEFAULT_SCORE_DIR / f"e4_{stream}_test_{score_kind}_viterbi.csv"


def default_tuning_path(stream: str, score_kind: str = "calibrated") -> Path:
    return DEFAULT_TUNE_DIR / f"e4_{stream}_{score_kind}_viterbi_tuning.json"


def default_e3_logits_path(stream: str, split: str) -> Path:
    return DEFAULT_E3_LOGIT_DIR / f"e3_{stream}_{split}_logits.csv"


def default_e3_temperature_path(stream: str) -> Path:
    return DEFAULT_E3_TEMPERATURE_DIR / f"e3_{stream}_temperatures.json"


__all__ = [
    "DEFAULT_CONFIG_ROOT",
    "DEFAULT_EVAL_DIR",
    "DEFAULT_JSONL_ROOT",
    "DEFAULT_LOGIT_DIR",
    "DEFAULT_SCORE_DIR",
    "DEFAULT_TUNE_DIR",
    "DEFAULT_WORK_ROOT",
    "LABELS",
    "ScoreRow",
    "apply_viterbi",
    "calibrate_logit_rows",
    "default_calibrated_path",
    "default_e3_logits_path",
    "default_e3_temperature_path",
    "default_raw_path",
    "default_scores_path",
    "default_tuning_path",
    "default_val_logits_path",
    "default_viterbi_path",
    "load_temperatures",
    "manual_transition_matrix",
    "parse_lambda_grid",
    "probability_rows_from_logits",
    "protocol_metadata",
    "read_score_csv",
    "raw_probability_rows",
    "write_json",
    "write_score_csv",
    "write_transition_matrix",
]

"""Apply tuned E4 Viterbi decoding to calibrated test probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import (
        LABELS,
        apply_viterbi,
        default_scores_path,
        default_tuning_path,
        default_viterbi_path,
        manual_transition_matrix,
        protocol_metadata,
        read_score_csv,
        write_json,
        write_score_csv,
        write_transition_matrix,
    )
except ImportError:
    from common import (
        LABELS,
        apply_viterbi,
        default_scores_path,
        default_tuning_path,
        default_viterbi_path,
        manual_transition_matrix,
        protocol_metadata,
        read_score_csv,
        write_json,
        write_score_csv,
        write_transition_matrix,
    )


def selected_lambda_from_tuning(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "selected_lambda" not in data:
        raise ValueError(f"{path} has no selected_lambda")
    value = float(data["selected_lambda"])
    if value < 0:
        raise ValueError("selected_lambda must be non-negative")
    return value


def apply_to_file(
    test_scores: Path,
    tuning: Path,
    output: Path,
    score_kind: str,
    lambda_value: float | None,
    overwrite: bool,
) -> None:
    rows = read_score_csv(test_scores)
    if any(row.score_type != "prob" for row in rows):
        raise ValueError("Viterbi decoding expects probability rows")

    lam = selected_lambda_from_tuning(tuning) if lambda_value is None else float(lambda_value)
    if lam < 0:
        raise ValueError("--lambda-value must be non-negative")

    transition_matrix = manual_transition_matrix()
    decoded = apply_viterbi(rows, transition_matrix, lam)
    write_score_csv(output, decoded, overwrite=overwrite)
    matrix_outputs = write_transition_matrix(output.parent, overwrite=overwrite)
    write_json(
        output.with_name(f"{output.stem}_metadata.json"),
        {
            "experiment": "E4",
            "stage": "viterbi_test_decode",
            "score_kind": score_kind,
            "test_scores": str(test_scores),
            "tuning": str(tuning),
            "output_scores": str(output),
            "lambda": lam,
            "protocol": protocol_metadata(),
            "labels": LABELS,
            "transition_matrix": matrix_outputs,
            "score_encoding": (
                "one-hot decoded labels; use for classification/sequence metrics, "
                "not calibrated confidence"
            ),
            "num_rows": len(decoded),
        },
        overwrite=overwrite,
    )
    print(f"[DONE] wrote {len(decoded)} Viterbi-decoded rows to {output} with lambda={lam:g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply E4 Viterbi decoding to test scores.")
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument("--score-kind", choices=["raw", "calibrated"], default="calibrated")
    parser.add_argument("--test-scores", type=Path)
    parser.add_argument("--tuning", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lambda-value", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_to_file(
        test_scores=args.test_scores or default_scores_path(args.stream, "test", args.score_kind),
        tuning=args.tuning or default_tuning_path(args.stream, args.score_kind),
        output=args.output or default_viterbi_path(args.stream, args.score_kind),
        score_kind=args.score_kind,
        lambda_value=args.lambda_value,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

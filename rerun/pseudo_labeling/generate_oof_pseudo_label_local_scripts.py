"""Generate local scripts for OOF pseudo-label generation and aggregation.

These scripts mirror the H100 OOF scripts but run directly on the current
machine without SLURM. They are intended for WSL/Linux-style local execution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rerun.pseudo_labeling.generate_inner_teacher_training_artifacts import FOLDS  # noqa: E402


DEFAULT_LOCAL_ROOT = Path("rerun/pseudo_labeling/local/oof_pseudo_labels")


def repo_root_block() -> str:
    return """SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."
"""


def teacher_script_text(fold: str, teacher: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

{repo_root_block()}

PYTHON="${{PYTHON:-python}}"
NUM_PASSES="${{NUM_PASSES:-30}}"
SEED="${{SEED:-42}}"
BATCH_SIZE="${{BATCH_SIZE:-64}}"
NUM_WORKERS="${{NUM_WORKERS:-0}}"
NUM_THREADS="${{NUM_THREADS:-16}}"
DEVICE="${{DEVICE:-cpu}}"
EXTRA_ARGS=()

if [[ "${{OVERWRITE:-0}}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

if [[ "${{SKIP_SANITY_CHECK:-0}}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-sanity-check)
fi

if [[ "${{SKIP_CHECKPOINT_HASH:-0}}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-checkpoint-hash)
fi

if [[ "${{SAVE_STREAM_PASS_PROBABILITIES:-0}}" == "1" ]]; then
  EXTRA_ARGS+=(--save-stream-pass-probabilities)
fi

"${{PYTHON}}" rerun/pseudo_labeling/run_inner_teacher_oof_pseudo_labeling.py \\
  --fold {fold} \\
  --teacher {teacher} \\
  --device "${{DEVICE}}" \\
  --num-passes "${{NUM_PASSES}}" \\
  --seed "${{SEED}}" \\
  --batch-size "${{BATCH_SIZE}}" \\
  --num-workers "${{NUM_WORKERS}}" \\
  --num-threads "${{NUM_THREADS}}" \\
  "${{EXTRA_ARGS[@]}}"
"""


def aggregate_script_text(fold: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

{repo_root_block()}

PYTHON="${{PYTHON:-python}}"

"${{PYTHON}}" rerun/pseudo_labeling/aggregate_oof_pseudo_labels.py \\
  --folds {fold}
"""


def submit_teacher_scripts_text(folds: list[str]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        repo_root_block().rstrip(),
        "",
        "# Runs teacher systems sequentially on the local machine.",
    ]
    for fold in folds:
        for teacher in sorted(FOLDS[fold]["teachers"]):
            lines.append(
                f"bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_{fold}_{teacher}_local.sh"
            )
    lines.append("")
    return "\n".join(lines)


def submit_aggregation_scripts_text(folds: list[str]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        repo_root_block().rstrip(),
        "",
    ]
    for fold in folds:
        lines.append(
            f"bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_aggregate_fold_{fold}_local.sh"
        )
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def generate(args: argparse.Namespace) -> None:
    folds = [fold.lower() for fold in args.folds]
    for fold in folds:
        if fold not in FOLDS:
            raise ValueError(f"Unknown fold {fold!r}; expected one of {sorted(FOLDS)}")

    for fold in folds:
        for teacher in sorted(FOLDS[fold]["teachers"]):
            write_text(
                args.local_root / f"run_oof_fold_{fold}_{teacher}_local.sh",
                teacher_script_text(fold, teacher),
                args.overwrite,
            )
        write_text(
            args.local_root / f"run_aggregate_fold_{fold}_local.sh",
            aggregate_script_text(fold),
            args.overwrite,
        )
    write_text(
        args.local_root / "run_all_oof_teachers_local.sh",
        submit_teacher_scripts_text(folds),
        args.overwrite,
    )
    write_text(
        args.local_root / "run_all_aggregations_local.sh",
        submit_aggregation_scripts_text(folds),
        args.overwrite,
    )
    print(f"[DONE] wrote local OOF pseudo-label scripts under {args.local_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", default=sorted(FOLDS))
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        generate(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

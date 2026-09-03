"""Generate H100 SLURM jobs for OOF pseudo-label generation.

The generated jobs are separate from the inner-teacher training jobs:

- 12 teacher-system jobs run steps 1-5, one per outer-fold/inner-teacher pair.
- 3 aggregation jobs run step 6, one per outer fold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rerun.pseudo_labeling.generate_inner_teacher_training_artifacts import FOLDS  # noqa: E402


DEFAULT_JOB_ROOT = Path("rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100")


def h100_runtime_block() -> str:
    return """module purge
module load StdEnv/2023
module load python/3.10
module load opencv/4.8.1

source "/project/def-mbolic/yunzelu/pyskl/.venv/bin/activate"
cd "/project/def-mbolic/yunzelu/pyskl"
"""


def teacher_job_text(fold: str, teacher: str) -> str:
    return f"""#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=00:20:00
#SBATCH --job-name=oof_pl_{fold}_{teacher}
#SBATCH --output=rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

{h100_runtime_block()}

NUM_PASSES="${{NUM_PASSES:-30}}"
SEED="${{SEED:-42}}"
BATCH_SIZE="${{BATCH_SIZE:-128}}"
NUM_WORKERS="${{NUM_WORKERS:-2}}"
DEVICE="${{DEVICE:-cuda}}"
EXTRA_ARGS=()

if [[ "${{OVERWRITE:-0}}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

if [[ "${{SAVE_STREAM_PASS_PROBABILITIES:-0}}" == "1" ]]; then
  EXTRA_ARGS+=(--save-stream-pass-probabilities)
fi

python rerun/pseudo_labeling/run_inner_teacher_oof_pseudo_labeling.py \\
  --fold {fold} \\
  --teacher {teacher} \\
  --device "${{DEVICE}}" \\
  --num-passes "${{NUM_PASSES}}" \\
  --seed "${{SEED}}" \\
  --batch-size "${{BATCH_SIZE}}" \\
  --num-workers "${{NUM_WORKERS}}" \\
  "${{EXTRA_ARGS[@]}}"
"""


def aggregate_job_text(fold: str) -> str:
    return f"""#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=oof_aggr_{fold}
#SBATCH --output=rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

{h100_runtime_block()}

python rerun/pseudo_labeling/aggregate_oof_pseudo_labels.py \\
  --folds {fold}
"""


def submit_teacher_jobs_text(folds: list[str]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "${SCRIPT_DIR}/../../../.."',
        "",
    ]
    for fold in folds:
        for teacher in sorted(FOLDS[fold]["teachers"]):
            lines.append(
                f"sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_{fold}_{teacher}_h100.sh"
            )
    lines.append("")
    return "\n".join(lines)


def submit_aggregate_jobs_text(folds: list[str]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "${SCRIPT_DIR}/../../../.."',
        "",
        "# Submit after the corresponding teacher-system jobs have completed.",
    ]
    for fold in folds:
        lines.append(
            f"sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_aggregate_fold_{fold}_h100.sh"
        )
    lines.append("")
    return "\n".join(lines)


def submit_all_with_dependencies_text(folds: list[str]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "${SCRIPT_DIR}/../../../.."',
        "",
        "declare -A FOLD_DEPENDENCIES",
        "",
    ]
    for fold in folds:
        lines.append(f"FOLD_DEPENDENCIES[{fold}]=\"\"")
        for teacher in sorted(FOLDS[fold]["teachers"]):
            job = f"rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_{fold}_{teacher}_h100.sh"
            lines.extend(
                [
                    f'job_id="$(sbatch --parsable {job})"',
                    f'echo "submitted fold {fold} {teacher}: ${{job_id}}"',
                    f'if [[ -z "${{FOLD_DEPENDENCIES[{fold}]}}" ]]; then',
                    f'  FOLD_DEPENDENCIES[{fold}]="${{job_id}}"',
                    "else",
                    f'  FOLD_DEPENDENCIES[{fold}]="${{FOLD_DEPENDENCIES[{fold}]}}:${{job_id}}"',
                    "fi",
                ]
            )
        agg = f"rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_aggregate_fold_{fold}_h100.sh"
        lines.extend(
            [
                f'sbatch --dependency=afterok:${{FOLD_DEPENDENCIES[{fold}]}} {agg}',
                f'echo "submitted fold {fold} aggregation after ${{FOLD_DEPENDENCIES[{fold}]}}"',
                "",
            ]
        )
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
                args.job_root / f"run_oof_fold_{fold}_{teacher}_h100.sh",
                teacher_job_text(fold, teacher),
                args.overwrite,
            )
        write_text(
            args.job_root / f"run_aggregate_fold_{fold}_h100.sh",
            aggregate_job_text(fold),
            args.overwrite,
        )

    write_text(
        args.job_root / "submit_oof_teacher_jobs_h100.sh",
        submit_teacher_jobs_text(folds),
        args.overwrite,
    )
    write_text(
        args.job_root / "submit_oof_aggregation_jobs_h100.sh",
        submit_aggregate_jobs_text(folds),
        args.overwrite,
    )
    write_text(
        args.job_root / "submit_oof_all_with_dependencies_h100.sh",
        submit_all_with_dependencies_text(folds),
        args.overwrite,
    )
    print(f"[DONE] wrote OOF pseudo-labeling H100 jobs under {args.job_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", default=sorted(FOLDS))
    parser.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
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

"""Export deterministic hard pseudo-labels from S6 teachers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.s6.pseudo_common import add_common_args, export_hard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic hard S6 pseudo-labels.")
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    export_hard(parse_args())


if __name__ == "__main__":
    main()

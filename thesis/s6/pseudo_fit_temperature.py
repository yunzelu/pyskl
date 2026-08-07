"""Fit one scalar temperature per S6 teacher/stream."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.s6.pseudo_common import add_common_args, fit_temperatures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit S6 teacher temperatures on calibration splits.")
    add_common_args(parser, include_temperature=True)
    return parser.parse_args()


def main() -> None:
    fit_temperatures(parse_args())


if __name__ == "__main__":
    main()

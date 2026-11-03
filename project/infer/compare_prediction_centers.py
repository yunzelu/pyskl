"""Compare prediction window metadata from two inference CSVs.

Matching metadata supports a like-for-like window comparison. It cannot prove
that the underlying pose coordinates, confidence values, or masks are equal.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from decimal import Decimal, InvalidOperation
from itertools import zip_longest
from pathlib import Path

FIELDS = (
    "window_candidate_index", "window_start_retained_idx", "center_retained_idx",
    "window_end_retained_idx_exclusive", "source_frame_start", "source_frame_center",
    "source_frame_end", "source_csv_row_start", "source_csv_row_center",
    "source_csv_row_end", "ID", "source_id", "start_timestamp_sec",
    "center_timestamp_sec", "end_timestamp_sec", "window_size", "stride",
    "max_adjacent_gap_sec", "window_span_sec",
)
DECIMAL_FIELDS = {"start_timestamp_sec", "center_timestamp_sec", "end_timestamp_sec",
                  "max_adjacent_gap_sec", "window_span_sec"}


def canonical(value: str, field: str) -> str | Decimal:
    if field not in DECIMAL_FIELDS:
        return value
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"Nonfinite {field}: {value!r}")
    return result


def compare(old_csv: Path, new_csv: Path) -> dict:
    with old_csv.open(newline="", encoding="utf-8-sig") as before, \
            new_csv.open(newline="", encoding="utf-8-sig") as after:
        old = csv.DictReader(before)
        new = csv.DictReader(after)
        for reader, path in ((old, old_csv), (new, new_csv)):
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"Missing or duplicate header in {path}")
            if "center_timestamp_sec" not in reader.fieldnames:
                raise ValueError(f"Missing center_timestamp_sec in {path}")
        checked = [f for f in FIELDS if f in old.fieldnames and f in new.fieldnames]
        not_checked = [f for f in FIELDS if f not in checked]
        old_count = new_count = differences = 0
        examples = []
        old_previous = new_previous = None
        for record, (old_row, new_row) in enumerate(zip_longest(old, new), 2):
            old_count += old_row is not None
            new_count += new_row is not None
            for row, side in ((old_row, "old"), (new_row, "new")):
                if row is None:
                    continue
                center = canonical(row["center_timestamp_sec"], "center_timestamp_sec")
                previous = old_previous if side == "old" else new_previous
                if previous is not None and center <= previous:
                    raise ValueError(f"{side} timestamps not strictly increasing at CSV record {record}")
                if side == "old":
                    old_previous = center
                else:
                    new_previous = center
            if old_row is None or new_row is None:
                differences += 1
                if len(examples) < 10:
                    examples.append({"csv_record": record, "reason": "row present in only one file"})
                continue
            fields = [field for field in checked if canonical(old_row[field], field) != canonical(new_row[field], field)]
            if fields:
                differences += 1
                if len(examples) < 10:
                    examples.append({"csv_record": record, "mismatched_fields": fields,
                                     "old_center": old_row["center_timestamp_sec"],
                                     "new_center": new_row["center_timestamp_sec"]})
        return {
            "old_csv": str(old_csv), "new_csv": str(new_csv),
            "old_rows": old_count, "new_rows": new_count,
            "compared_fields": checked, "unavailable_fields": not_checked,
            "different_rows": differences,
            "match": old_count == new_count and differences == 0,
            "examples": examples,
            "limitation": "Matching centers and metadata do not prove raw inputs, pose values, or multiperson masks are identical.",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_csv", type=Path)
    parser.add_argument("new_csv", type=Path)
    parser.add_argument("--json-output", type=Path, help="Write report to a new JSON file")
    args = parser.parse_args()
    result = compare(args.old_csv, args.new_csv)
    text = json.dumps(result, indent=2) + "\n"
    if args.json_output:
        with args.json_output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    sys.stdout.write(text)
    return 0 if result["match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Summarize LOSO stream metrics from a pyskl training log."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CONFIG_RE = re.compile(
    r"\+\s*CONFIG=.*?ctrgcn_pyskl_radarv4_loso_2d[\\/]"
    r"(?P<subject>[^\\/]+)[\\/](?P<stream>[^\\/]+)\.py"
)
STREAM_RE = re.compile(r"Config:\s*stream\s*=\s*['\"](?P<stream>[^'\"]+)['\"]")
PKL_RE = re.compile(r"\bpkl\s*=\s*['\"][^'\"]*?_test_(?P<subject>[^'\"]+)['\"]")
ANN_FILE_RE = re.compile(
    r"\bann_file\s*=\s*['\"][^'\"]*?_test_(?P<subject>[^'\".\\/]+)\.pkl['\"]"
)
TOP1_RE = re.compile(r"\btop1_acc:\s*(?P<value>[0-9]*\.?[0-9]+)")
MEAN_CLASS_RE = re.compile(
    r"\bmean_class_accuracy:\s*(?P<value>[0-9]*\.?[0-9]+)"
)


@dataclass
class RunContext:
    config_subject: str | None = None
    config_stream: str | None = None
    pkl_subject: str | None = None
    ann_subject: str | None = None
    stream: str | None = None
    start_line: int | None = None


@dataclass
class Result:
    stream: str
    subject: str
    top1_acc: float
    mean_class_accuracy: float
    line: int
    config_subject: str | None = None
    config_stream: str | None = None
    pkl_subject: str | None = None
    ann_subject: str | None = None


@dataclass
class PendingBestTest:
    context: RunContext
    line: int
    top1_acc: float | None = None
    mean_class_accuracy: float | None = None


@dataclass
class ParseOutput:
    results: list[Result] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a pyskl training log and summarize best-checkpoint top1_acc "
            "and mean_class_accuracy by stream and LOSO test subject."
        )
    )
    parser.add_argument("log", type=Path, help="Path to the .out training log.")
    parser.add_argument(
        "--subject-source",
        choices=("auto", "dataset", "config"),
        default="auto",
        help=(
            "Where to read the test subject from. 'dataset' uses pkl/ann_file; "
            "'config' uses the CONFIG path; 'auto' prefers dataset and falls "
            "back to config. Default: auto."
        ),
    )
    parser.add_argument(
        "--std-ddof",
        type=int,
        default=0,
        choices=(0, 1),
        help="Use 0 for population std or 1 for sample std. Default: 0.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write the per-subject best rows as CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Optional path to write stream-level summary rows as CSV.",
    )
    parser.add_argument(
        "--quiet-warnings",
        action="store_true",
        help="Do not print parser warnings to stderr.",
    )
    return parser.parse_args()


def choose_subject(context: RunContext, source: str) -> str | None:
    dataset_subject = context.pkl_subject or context.ann_subject
    if source == "dataset":
        return dataset_subject
    if source == "config":
        return context.config_subject
    return dataset_subject or context.config_subject


def choose_stream(context: RunContext) -> str | None:
    return context.stream or context.config_stream


def parse_log(log_path: Path, subject_source: str) -> ParseOutput:
    output = ParseOutput()
    context = RunContext()
    pending: PendingBestTest | None = None

    # pyskl progress bars use carriage returns heavily. Splitting only on LF keeps
    # reported source lines aligned with tools such as rg and most editors.
    with log_path.open("r", encoding="utf-8", errors="replace", newline="\n") as handle:
        for line_no, line in enumerate(handle, start=1):
            config_match = CONFIG_RE.search(line)
            if config_match:
                context = RunContext(
                    config_subject=config_match.group("subject"),
                    config_stream=config_match.group("stream"),
                    start_line=line_no,
                )
                pending = None
                continue

            stream_match = STREAM_RE.search(line)
            if stream_match:
                context.stream = stream_match.group("stream")

            pkl_match = PKL_RE.search(line)
            if pkl_match:
                context.pkl_subject = pkl_match.group("subject")

            ann_match = ANN_FILE_RE.search(line)
            if ann_match:
                context.ann_subject = ann_match.group("subject")

            if "Testing results of the best checkpoint" in line:
                pending = PendingBestTest(
                    context=RunContext(
                        config_subject=context.config_subject,
                        config_stream=context.config_stream,
                        pkl_subject=context.pkl_subject,
                        ann_subject=context.ann_subject,
                        stream=context.stream,
                        start_line=context.start_line,
                    ),
                    line=line_no,
                )
                continue

            if pending is None:
                continue

            top1_match = TOP1_RE.search(line)
            if top1_match:
                pending.top1_acc = float(top1_match.group("value"))

            mean_class_match = MEAN_CLASS_RE.search(line)
            if mean_class_match:
                pending.mean_class_accuracy = float(mean_class_match.group("value"))

            if (
                pending.top1_acc is not None
                and pending.mean_class_accuracy is not None
            ):
                run_context = pending.context
                subject = choose_subject(run_context, subject_source)
                stream = choose_stream(run_context)
                if subject is None or stream is None:
                    output.warnings.append(
                        f"line {pending.line}: skipped result because stream or "
                        f"subject could not be identified"
                    )
                else:
                    add_context_warnings(output.warnings, run_context, pending.line)
                    output.results.append(
                        Result(
                            stream=stream,
                            subject=subject,
                            top1_acc=pending.top1_acc,
                            mean_class_accuracy=pending.mean_class_accuracy,
                            line=pending.line,
                            config_subject=run_context.config_subject,
                            config_stream=run_context.config_stream,
                            pkl_subject=run_context.pkl_subject,
                            ann_subject=run_context.ann_subject,
                        )
                    )
                pending = None

    if pending is not None:
        output.warnings.append(
            f"line {pending.line}: found a best-checkpoint section without both "
            "top1_acc and mean_class_accuracy"
        )

    return output


def add_context_warnings(
    warnings: list[str], context: RunContext, result_line: int
) -> None:
    dataset_subjects = {
        subject for subject in (context.pkl_subject, context.ann_subject) if subject
    }
    if len(dataset_subjects) > 1:
        warnings.append(
            f"line {result_line}: pkl subject {context.pkl_subject!r} and "
            f"ann_file subject {context.ann_subject!r} disagree"
        )

    dataset_subject = context.pkl_subject or context.ann_subject
    if (
        dataset_subject
        and context.config_subject
        and dataset_subject != context.config_subject
    ):
        warnings.append(
            f"line {result_line}: CONFIG subject {context.config_subject!r} "
            f"differs from dataset subject {dataset_subject!r}"
        )

    if context.stream and context.config_stream and context.stream != context.config_stream:
        warnings.append(
            f"line {result_line}: CONFIG stream {context.config_stream!r} "
            f"differs from config-body stream {context.stream!r}"
        )


def best_by_stream_subject(results: Iterable[Result]) -> list[Result]:
    best: dict[tuple[str, str], Result] = {}
    for result in results:
        key = (result.stream, result.subject)
        old = best.get(key)
        if old is None or result.top1_acc > old.top1_acc:
            best[key] = result
    return sorted(best.values(), key=lambda row: (row.stream, row.subject))


def duplicate_counts(results: Iterable[Result]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for result in results:
        key = (result.stream, result.subject)
        counts[key] = counts.get(key, 0) + 1
    return counts


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float], ddof: int) -> float:
    if len(values) - ddof <= 0:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - ddof)
    return variance**0.5


def summarize_by_stream(rows: Iterable[Result], ddof: int) -> list[dict[str, float | int | str]]:
    groups: dict[str, list[Result]] = {}
    for row in rows:
        groups.setdefault(row.stream, []).append(row)

    summary: list[dict[str, float | int | str]] = []
    for stream in sorted(groups):
        stream_rows = groups[stream]
        top1 = [row.top1_acc for row in stream_rows]
        mean_class = [row.mean_class_accuracy for row in stream_rows]
        summary.append(
            {
                "stream": stream,
                "n_subjects": len(stream_rows),
                "mean_top1_acc": mean(top1),
                "std_top1_acc": std(top1, ddof),
                "mean_class_accuracy_mean": mean(mean_class),
                "mean_class_accuracy_std": std(mean_class, ddof),
            }
        )
    return summary


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        if rows
        else len(headers[index])
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def write_per_subject_csv(path: Path, rows: list[Result], counts: dict[tuple[str, str], int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stream",
                "subject",
                "top1_acc",
                "mean_class_accuracy",
                "source_line",
                "num_runs_for_stream_subject",
                "config_subject",
                "config_stream",
                "pkl_subject",
                "ann_subject",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "stream": row.stream,
                    "subject": row.subject,
                    "top1_acc": f"{row.top1_acc:.6f}",
                    "mean_class_accuracy": f"{row.mean_class_accuracy:.6f}",
                    "source_line": row.line,
                    "num_runs_for_stream_subject": counts[(row.stream, row.subject)],
                    "config_subject": row.config_subject or "",
                    "config_stream": row.config_stream or "",
                    "pkl_subject": row.pkl_subject or "",
                    "ann_subject": row.ann_subject or "",
                }
            )


def write_summary_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stream",
                "n_subjects",
                "mean_top1_acc",
                "std_top1_acc",
                "mean_class_accuracy_mean",
                "mean_class_accuracy_std",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def main() -> int:
    args = parse_args()
    if not args.log.is_file():
        print(f"error: log file not found: {args.log}", file=sys.stderr)
        return 2

    parsed = parse_log(args.log, args.subject_source)
    if not parsed.results:
        print("No best-checkpoint test results found.", file=sys.stderr)
        return 1

    counts = duplicate_counts(parsed.results)
    best_rows = best_by_stream_subject(parsed.results)
    summary_rows = summarize_by_stream(best_rows, args.std_ddof)

    print(f"Parsed {len(parsed.results)} best-checkpoint test result(s).")
    print(f"Keeping {len(best_rows)} best row(s) by (stream, subject), ranked by top1_acc.")
    print()

    per_subject_rows = [
        [
            row.stream,
            row.subject,
            f"{row.top1_acc:.4f}",
            f"{row.mean_class_accuracy:.4f}",
            str(counts[(row.stream, row.subject)]),
            str(row.line),
        ]
        for row in best_rows
    ]
    print("Best per stream/subject")
    print_table(
        ["stream", "subject", "top1_acc", "mean_class_accuracy", "runs", "line"],
        per_subject_rows,
    )
    print()

    summary_table_rows = [
        [
            str(row["stream"]),
            str(row["n_subjects"]),
            f"{row['mean_top1_acc']:.4f}",
            f"{row['std_top1_acc']:.4f}",
            f"{row['mean_class_accuracy_mean']:.4f}",
            f"{row['mean_class_accuracy_std']:.4f}",
        ]
        for row in summary_rows
    ]
    std_name = "population" if args.std_ddof == 0 else "sample"
    print(f"Stream summary ({std_name} std, ddof={args.std_ddof})")
    print_table(
        [
            "stream",
            "n",
            "mean_top1",
            "std_top1",
            "mean_mca",
            "std_mca",
        ],
        summary_table_rows,
    )

    duplicate_messages = [
        f"{stream}/{subject}: {count} runs"
        for (stream, subject), count in sorted(counts.items())
        if count > 1
    ]
    if duplicate_messages:
        print("\nDuplicate stream/subject groups were reduced to the best top1_acc:", file=sys.stderr)
        for message in duplicate_messages:
            print(f"  {message}", file=sys.stderr)

    if parsed.warnings and not args.quiet_warnings:
        print("\nWarnings:", file=sys.stderr)
        for warning in parsed.warnings:
            print(f"  {warning}", file=sys.stderr)

    if args.csv:
        write_per_subject_csv(args.csv, best_rows, counts)
        print(f"\nWrote per-subject CSV: {args.csv}")

    if args.summary_csv:
        write_summary_csv(args.summary_csv, summary_rows)
        print(f"Wrote summary CSV: {args.summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

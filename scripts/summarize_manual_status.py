#!/usr/bin/env python3
"""
Summarize manual validation status across Gemini report files.
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path


MANUAL_STATUS_FIELD = "manual_status"


def collect_report_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    def add_path(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or resolved.name.startswith("_"):
            return
        seen.add(resolved)
        paths.append(resolved)

    for raw in inputs:
        expanded = glob(raw, recursive=True) or [raw]
        for entry in expanded:
            candidate = Path(entry)
            if candidate.is_dir():
                for child in sorted(candidate.glob("*.json")):
                    add_path(child)
            elif candidate.suffix.lower() == ".json" and candidate.exists():
                add_path(candidate)
    return paths


def load_status(path: Path) -> str | None:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    status = data.get(MANUAL_STATUS_FIELD)
    if status in ("valid", "invalid"):
        return status
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count manual valid/invalid statuses across reports.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Report JSON path(s), directories, or globs to include.",
    )
    parser.add_argument(
        "--list-unset",
        action="store_true",
        help="Print report paths missing manual status.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_paths = collect_report_paths(args.paths)
    if not report_paths:
        raise SystemExit("No report files found for the given inputs.")

    total = len(report_paths)
    valid = 0
    invalid = 0
    unset_paths: list[Path] = []

    for path in report_paths:
        try:
            status = load_status(path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to read {path}: {exc}")
            unset_paths.append(path)
            continue
        if status == "valid":
            valid += 1
        elif status == "invalid":
            invalid += 1
        else:
            unset_paths.append(path)

    unset = len(unset_paths)
    print(f"Total reports scanned: {total}")
    print(f"Manual valid:   {valid}")
    print(f"Manual invalid: {invalid}")
    print(f"Not set:        {unset}")

    if total > 0:
        pct = lambda count: (count / total) * 100
        print(f"Valid%: {pct(valid):.1f} | Invalid%: {pct(invalid):.1f} | Marked%: {pct(valid + invalid):.1f}")

    if args.list_unset and unset_paths:
        print("\nReports without manual status:")
        for path in unset_paths:
            print(f"- {path}")


if __name__ == "__main__":
    main()

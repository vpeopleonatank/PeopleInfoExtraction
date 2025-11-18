#!/usr/bin/env python3
"""
Utility to inspect Gemini validation reports with passage text on the left
and detected issues on the right.
"""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from glob import glob
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
MANUAL_STATUS_FIELD = "manual_status"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def resolve_source_path(source: str, report_path: Path) -> Path | None:
    if not source:
        return None
    candidate = Path(source)
    if candidate.exists():
        return candidate
    repo_candidate = ROOT / source
    if repo_candidate.exists():
        return repo_candidate
    report_candidate = report_path.parent / source
    if report_candidate.exists():
        return report_candidate
    return candidate if candidate.exists() else None


def wrap_lines(text: str, width: int) -> list[str]:
    if width <= 0:
        return text.splitlines() or [""]
    wrapper = textwrap.TextWrapper(
        width=width,
        replace_whitespace=False,
        drop_whitespace=False,
    )
    lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(wrapper.wrap(paragraph))
    return lines or [""]


def build_issue_text(report: dict) -> str:
    issues = report.get("issues") or []
    missing_people = report.get("missing_people") or []
    lines: list[str] = [
        f"Verdict: {report.get('verdict', 'unknown')}",
        f"Issues: {len(issues)} | Missing people: {len(missing_people)}",
    ]
    if issues:
        lines.append("")
        lines.append("Issues")
        for idx, issue in enumerate(issues, start=1):
            header = f"[{idx}] {issue.get('type', 'unknown')}"
            person = issue.get("person_name")
            field = issue.get("field")
            if person:
                header += f" | Person: {person}"
            if field:
                header += f" | Field: {field}"
            lines.append(header)
            description = issue.get("description")
            if description:
                lines.append(f"  {description}")
            evidence = issue.get("evidence")
            if evidence:
                lines.append(f"  Evidence: {evidence}")
            lines.append("")
    if missing_people:
        lines.append("Missing People")
        for mp in missing_people:
            snippet = f" — {mp.get('snippet')}" if mp.get("snippet") else ""
            lines.append(f"- {mp.get('name', 'unknown')}{snippet}")
    return "\n".join(lines).rstrip()


def chunk_columns(left_lines: Iterable[str], right_lines: Iterable[str], left_width: int) -> list[str]:
    output: list[str] = []
    separator = " | "
    left_list = list(left_lines)
    right_list = list(right_lines)
    total_rows = max(len(left_list), len(right_list))
    for idx in range(total_rows):
        left = left_list[idx] if idx < len(left_list) else ""
        right = right_list[idx] if idx < len(right_list) else ""
        output.append(f"{left.ljust(left_width)}{separator}{right}")
    return output


def collect_report_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    def add_path(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        if resolved.name.startswith("_"):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render passage text beside Gemini validation issues.",
    )
    parser.add_argument(
        "reports",
        nargs="+",
        help="Report JSON paths, directories, or glob patterns (e.g., data/.../*.json).",
    )
    parser.add_argument(
        "--left-ratio",
        type=float,
        default=0.55,
        help="Portion of terminal width reserved for the passage column (0-1).",
    )
    parser.add_argument(
        "--mark",
        choices=["valid", "invalid", "unset"],
        help="Set manual status for all provided reports without prompting.",
    )
    parser.add_argument(
        "--prompt-mark",
        action="store_true",
        help="Prompt after each report to mark it as valid/invalid/unset.",
    )
    return parser.parse_args()


def apply_manual_status(report: dict, report_path: Path, status: Optional[str]) -> None:
    if status:
        report[MANUAL_STATUS_FIELD] = status
    else:
        report.pop(MANUAL_STATUS_FIELD, None)
    write_json(report_path, report)
    print(f"[Saved {MANUAL_STATUS_FIELD} = {status or 'unset'}]")


def prompt_manual_status(current: Optional[str]) -> Optional[str]:
    prompt = (
        f"Mark report as [v]alid, [i]nvalid, [u]nset, [s]kip "
        f"(current: {current or 'unset'}): "
    )
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return None
        if not answer or answer in {"s", "skip"}:
            return None
        if answer in {"v", "valid"}:
            return "valid"
        if answer in {"i", "invalid"}:
            return "invalid"
        if answer in {"u", "unset", "clear"}:
            return "unset"
        print("Please enter v/i/u/s.")


def render_report(
    report_path: Path,
    left_ratio: float,
    is_first: bool,
    args: argparse.Namespace,
) -> None:
    report = load_json(report_path)
    source_file = resolve_source_path(report.get("source_file", ""), report_path)
    passage_text = ""
    if source_file and source_file.exists():
        try:
            original = load_json(source_file)
            passage_text = original.get("passage", {}).get("text") or ""
        except Exception as exc:  # pylint: disable=broad-except
            passage_text = f"[Error loading passage from {source_file}: {exc}]"
    else:
        passage_text = "[Original passage not found]"

    terminal_width = shutil.get_terminal_size((120, 20)).columns
    left_width = max(10, int(terminal_width * left_ratio) - 1)
    right_width = max(10, terminal_width - left_width - 3)

    issue_text = build_issue_text(report)
    left_lines = wrap_lines(passage_text, left_width)
    right_lines = wrap_lines(issue_text, right_width)

    manual_status = report.get(MANUAL_STATUS_FIELD)
    header = (
        f"Doc: {report.get('doc_id')} | Passage: {report.get('passage_id')} | "
        f"Article: {report.get('article_title') or 'N/A'} | "
        f"Manual: {manual_status or 'unset'}"
    )
    prefix = "" if is_first else "\n"
    print(f"{prefix}=== {report_path} ===")
    print(header)
    print(f"Source: {source_file or report.get('source_file', 'unknown')}")
    print("-" * terminal_width)
    for line in chunk_columns(left_lines, right_lines, left_width):
        print(line)

    manual_action: Optional[str] = None
    force_unset = False
    if args.mark:
        if args.mark == "unset":
            force_unset = True
        else:
            manual_action = args.mark
    elif args.prompt_mark:
        chosen = prompt_manual_status(manual_status)
        if chosen == "unset":
            force_unset = True
        else:
            manual_action = chosen

    if manual_action is not None or force_unset:
        apply_manual_status(report, report_path, manual_action if not force_unset else None)


def main() -> None:
    args = parse_args()
    report_paths = collect_report_paths(args.reports)
    if not report_paths:
        raise SystemExit("No report files found. Provide JSON paths or directories.")
    for idx, report_path in enumerate(report_paths):
        render_report(report_path, args.left_ratio, is_first=idx == 0, args=args)


if __name__ == "__main__":
    main()

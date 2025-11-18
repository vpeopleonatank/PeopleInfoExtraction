#!/usr/bin/env python3
"""
Batch validator that sends workflow_b outputs to Gemini 2.5 Flash.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.gemini_validator import GeminiFlashValidator
from src.validation.models import GeminiBatchSummary, GeminiValidationReport


def load_extraction_file(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def process_file(
    file_path: Path,
    validator: GeminiFlashValidator,
    output_dir: Path,
) -> GeminiValidationReport:
    data = load_extraction_file(file_path)
    passage = data.get("passage", {})
    response = data.get("response", {})
    parsed_people = response.get("parsed", {}).get("people", []) or []

    # Fallback: try to parse raw JSON if parsed.people missing
    if not parsed_people and isinstance(response.get("raw"), str):
        try:
            raw_obj = json.loads(response["raw"])
            parsed_people = raw_obj.get("people", []) or []
        except json.JSONDecodeError:
            parsed_people = []

    report = validator.validate_extraction(
        doc_id=data.get("doc_id", "unknown"),
        passage_id=data.get("passage_id", 0),
        passage_text=passage.get("text", ""),
        extracted_people=parsed_people,
        raw_response_text=response.get("raw"),
        original_model=data.get("model"),
        article_title=data.get("article_title"),
        published_at=data.get("published_at"),
        source_file=str(file_path),
        metadata={
            "status": data.get("status"),
            "error": data.get("error"),
            "model": data.get("model"),
        },
    )

    output_path = output_dir / file_path.name
    save_json(report.model_dump(mode="json"), output_path)
    return report


def build_batch_summary(output_dir: Path, validator_model: str) -> GeminiBatchSummary:
    summary = GeminiBatchSummary(validator_model=validator_model)
    if not output_dir.exists():
        return summary
    for file_path in sorted(output_dir.glob("*.json")):
        if file_path.name.startswith("_"):
            continue
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        try:
            report = GeminiValidationReport.model_validate(data)
        except Exception:
            continue
        summary.register_report(report)
    save_json(summary.model_dump(mode="json"), output_dir / "_batch_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate workflow_b outputs using Gemini 2.5 Flash.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/cache/workflow_b_fast/pass1"),
        help="Directory containing extraction JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cache/workflow_b_fast/gemini_reports"),
        help="Directory to store Gemini validation reports.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to process.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-run validation even when report exists.")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model to use.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="Maximum output tokens.")
    parser.add_argument("--max-passage-chars", type=int, default=12000, help="Max passage characters sent to Gemini.")
    parser.add_argument("--api-key", type=str, default=None, help="Google API key (fallback to GOOGLE_API_KEY env).")
    parser.add_argument("--rate-limit-ms", type=int, default=0, help="Sleep between requests (milliseconds).")
    parser.add_argument("--summary-only", action="store_true", help="Only recompute batch summary from existing reports.")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls and emit mocked Gemini responses.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    skip_existing = not args.no_skip_existing

    if args.summary_only:
        summary = build_batch_summary(output_dir, validator_model=args.model)
        print(f"Summary generated for {summary.total_files_processed} reports in {output_dir}")
        return

    if not input_dir.exists():
        raise SystemExit(f"Input directory {input_dir} does not exist.")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {input_dir}")
    if args.limit:
        json_files = json_files[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)

    validator = GeminiFlashValidator(
        model_name=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        max_passage_chars=args.max_passage_chars,
        api_key=args.api_key,
        dry_run=args.dry_run,
    )

    summary = GeminiBatchSummary(validator_model=args.model)
    processed = 0
    skipped = 0
    failed = 0

    for file_path in tqdm(json_files, desc="Gemini validating"):
        output_path = output_dir / file_path.name
        if skip_existing and output_path.exists():
            skipped += 1
            continue
        try:
            report = process_file(file_path, validator, output_dir)
            summary.register_report(report)
            processed += 1
            if args.dry_run and validator.last_prompt:
                print("\n=== Gemini Dry-Run Prompt ===")
                print(validator.last_prompt)
                print("=== End Prompt ===\n")
        except Exception as exc:  # pylint: disable=broad-except
            failed += 1
            print(f"\nError processing {file_path.name}: {exc}")
        if args.rate_limit_ms > 0:
            time.sleep(args.rate_limit_ms / 1000)

    save_json(summary.model_dump(mode="json"), output_dir / "_batch_summary.json")

    print(
        f"Processed: {processed}, Skipped: {skipped}, Failed: {failed}, "
        f"Issues logged: {summary.total_issues}, Missing people: {summary.total_missing_people}"
    )


if __name__ == "__main__":
    main()

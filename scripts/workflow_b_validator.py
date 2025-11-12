#!/usr/bin/env python3
"""
Validate workflow_b_fast extraction outputs using a larger local LLM.

This script:
1. Loads JSON files from workflow_b_fast/pass1/
2. Uses a larger Ollama model to validate the extraction
3. Checks for hallucinations, wrong positions, missing entities, and schema violations
4. Outputs validation reports to workflow_b_fast/validation_reports/
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.llm_validator import PeopleValidator
from src.validation.models import BatchValidationSummary


def load_extraction_file(file_path: Path) -> dict:
    """Load a workflow_b extraction JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_validation_report(report_data: dict, output_path: Path) -> None:
    """Save validation report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)


def process_file(
    file_path: Path,
    validator: PeopleValidator,
    output_dir: Path,
    skip_existing: bool = True,
) -> tuple[bool, dict | None]:
    """
    Process a single extraction file and generate validation report.

    Returns:
        (success, report_dict) tuple
    """
    # Determine output path
    output_path = output_dir / file_path.name

    # Skip if already validated
    if skip_existing and output_path.exists():
        return True, None

    try:
        # Load extraction data
        data = load_extraction_file(file_path)

        # Extract required fields
        doc_id = data.get("doc_id")
        passage_id = data.get("passage_id", 0)
        article_title = data.get("article_title")
        original_model = data.get("model", "unknown")
        passage_text = data.get("passage", {}).get("text", "")

        # Get extracted people
        response_data = data.get("response", {})
        parsed_data = response_data.get("parsed", {})
        extracted_people = parsed_data.get("people", [])

        # Handle case where response might be unparsed
        if not extracted_people and response_data.get("raw"):
            try:
                raw_json = json.loads(response_data["raw"])
                extracted_people = raw_json.get("people", [])
            except json.JSONDecodeError:
                pass

        # Validate
        report = validator.validate_extraction(
            doc_id=doc_id,
            passage_id=passage_id,
            passage_text=passage_text,
            extracted_people=extracted_people,
            original_model=original_model,
            article_title=article_title,
            source_file=str(file_path),
        )

        # Convert to dict for saving
        report_dict = report.model_dump(mode="json")

        # Save report
        save_validation_report(report_dict, output_path)

        return True, report_dict

    except Exception as e:
        print(f"\nError processing {file_path.name}: {e}")
        return False, None


def main():
    parser = argparse.ArgumentParser(
        description="Validate workflow_b_fast extraction outputs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/cache/workflow_b_fast/pass1"),
        help="Directory containing extraction JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cache/workflow_b_fast/validation_reports"),
        help="Directory to save validation reports",
    )
    parser.add_argument(
        "--validator-model",
        type=str,
        default="qwen2.5:14b",
        help="Ollama model to use for validation (e.g., qwen2.5:14b, llama3.1:70b)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-validate files that already have reports",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only generate batch summary from existing reports",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    skip_existing = not args.no_skip_existing

    # Ensure directories exist
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        sys.exit(1)

    if args.limit:
        json_files = json_files[: args.limit]

    print(f"Found {len(json_files)} files to process")
    print(f"Validator model: {args.validator_model}")
    print(f"Output directory: {output_dir}")
    print()

    # Initialize validator
    if not args.summary_only:
        validator = PeopleValidator(
            validator_model=args.validator_model,
            temperature=0.0,
            ollama_base_url=args.ollama_url,
        )

        # Process files
        success_count = 0
        error_count = 0
        skipped_count = 0

        for file_path in tqdm(json_files, desc="Validating extractions"):
            if skip_existing:
                output_path = output_dir / file_path.name
                if output_path.exists():
                    skipped_count += 1
                    continue

            success, _ = process_file(
                file_path,
                validator,
                output_dir,
                skip_existing=skip_existing,
            )

            if success:
                success_count += 1
            else:
                error_count += 1

        print(f"\n✓ Successfully validated: {success_count}")
        if skipped_count > 0:
            print(f"⊘ Skipped (already validated): {skipped_count}")
        if error_count > 0:
            print(f"✗ Errors: {error_count}")

    # Generate batch summary
    print("\nGenerating batch summary...")

    report_files = sorted(output_dir.glob("*.json"))
    if not report_files:
        print("No validation reports found")
        sys.exit(0)

    batch_summary = BatchValidationSummary(
        total_files_processed=len(report_files),
        total_files_with_issues=0,
        total_issues=0,
        total_people_extracted=0,
        validator_model=args.validator_model,
        original_model="unknown",
    )

    for report_file in report_files:
        with open(report_file, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        summary = report_data.get("summary", {})

        # Update batch summary
        batch_summary.total_people_extracted += summary.get("total_people_extracted", 0)
        batch_summary.total_issues += summary.get("total_issues_found", 0)

        if summary.get("total_issues_found", 0) > 0:
            batch_summary.total_files_with_issues += 1

        # Aggregate severity counts
        batch_summary.critical_issues += summary.get("critical_issues", 0)
        batch_summary.errors += summary.get("errors", 0)
        batch_summary.warnings += summary.get("warnings", 0)
        batch_summary.info_issues += summary.get("info_issues", 0)

        # Aggregate type counts
        batch_summary.hallucinations += summary.get("hallucinations", 0)
        batch_summary.wrong_positions += summary.get("wrong_positions", 0)
        batch_summary.missing_entities += summary.get("missing_entities", 0)
        batch_summary.schema_violations += summary.get("schema_violations", 0)
        batch_summary.wrong_law_articles += summary.get("wrong_law_articles", 0)

        # Update original model (take from first report)
        if batch_summary.original_model == "unknown":
            batch_summary.original_model = report_data.get("original_model", "unknown")

    # Save batch summary
    batch_summary_path = output_dir / "_batch_summary.json"
    with open(batch_summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("BATCH VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Files processed: {batch_summary.total_files_processed}")
    print(f"Files with issues: {batch_summary.total_files_with_issues} ({batch_summary.total_files_with_issues / batch_summary.total_files_processed * 100:.1f}%)")
    print(f"Total people extracted: {batch_summary.total_people_extracted}")
    print(f"Total issues found: {batch_summary.total_issues}")
    print()
    print("Issues by severity:")
    print(f"  Critical: {batch_summary.critical_issues}")
    print(f"  Errors:   {batch_summary.errors}")
    print(f"  Warnings: {batch_summary.warnings}")
    print(f"  Info:     {batch_summary.info_issues}")
    print()
    print("Issues by type:")
    print(f"  Hallucinations:       {batch_summary.hallucinations}")
    print(f"  Wrong positions:      {batch_summary.wrong_positions}")
    print(f"  Missing entities:     {batch_summary.missing_entities}")
    print(f"  Wrong law articles:   {batch_summary.wrong_law_articles}")
    print(f"  Schema violations:    {batch_summary.schema_violations}")
    print()
    print(f"Summary saved to: {batch_summary_path}")


if __name__ == "__main__":
    main()

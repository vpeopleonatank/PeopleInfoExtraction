#!/usr/bin/env python3
"""
Demo script to test validation logic without requiring Ollama.

This shows how the validator detects issues in the sample file.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.models import (
    IssueType,
    IssueSeverity,
    ValidationIssue,
    ValidationReport,
    ValidationSummary,
)


def analyze_sample_file():
    """Analyze the sample file and demonstrate validation issues."""

    sample_file = Path("data/cache/workflow_b_fast/pass1/019a2f6a-4fb7-7b90-840c-f5c3d20f5093_p0.json")

    if not sample_file.exists():
        print(f"Sample file not found: {sample_file}")
        return

    # Load the file
    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    passage_text = data["passage"]["text"]
    extracted_people = data["response"]["parsed"]["people"]

    print("=" * 80)
    print("VALIDATION DEMO - Manual Analysis")
    print("=" * 80)
    print(f"\nFile: {sample_file.name}")
    print(f"Passage length: {len(passage_text)} characters")
    print(f"People extracted: {len(extracted_people)}")
    print("\n" + "=" * 80)

    # Manually check for issues we can detect
    issues_found = []

    for idx, person in enumerate(extracted_people):
        person_name = person.get("name", {}).get("text", "Unknown")
        print(f"\n[Person {idx + 1}] {person_name}")
        print("-" * 40)

        # Check name position
        name_info = person.get("name", {})
        expected_name = name_info.get("text")
        start = name_info.get("start", -1)
        end = name_info.get("end", -1)

        if start != -1 and end != -1:
            actual_text = passage_text[start:end] if start < len(passage_text) and end <= len(passage_text) and start < end else "[OUT OF BOUNDS]"

            if actual_text != expected_name:
                correct_pos = passage_text.find(expected_name)
                print(f"  ❌ NAME POSITION ERROR")
                print(f"     Expected: '{expected_name}' at position {start}:{end}")
                print(f"     Actual: '{actual_text}'")
                if correct_pos != -1:
                    print(f"     Correct position: {correct_pos}:{correct_pos + len(expected_name)}")
                    issues_found.append(f"{person_name}: Wrong name position (claimed {start}:{end}, actually at {correct_pos})")
                else:
                    print(f"     ⚠️  Name not found in text - HALLUCINATION!")
                    issues_found.append(f"{person_name}: Name not found in passage (HALLUCINATION)")
            else:
                print(f"  ✓ Name position correct: {start}:{end}")

        # Check age position
        age_info = person.get("age", {})
        if age_info.get("value") is not None:
            age_start = age_info.get("start", -1)
            age_end = age_info.get("end", -1)
            print(f"  Age: {age_info['value']} (position: {age_start}:{age_end})")

            if age_start != -1 and age_end != -1 and age_start == age_end:
                print(f"    ⚠️  Age position has same start/end - likely incorrect")
                issues_found.append(f"{person_name}: Age position invalid (start == end)")

        # Check law articles
        actions = person.get("actions", [])
        for action_idx, action in enumerate(actions):
            law_article = action.get("law_article", "")
            if law_article:
                # Normalize and check
                normalized_law = " ".join(law_article.split())
                normalized_passage = " ".join(passage_text.split())

                if normalized_law not in normalized_passage:
                    print(f"  ❌ LAW ARTICLE ERROR in action {action_idx}")
                    print(f"     Claimed: '{law_article}'")
                    print(f"     Not found in passage!")

                    # Try to find what law article is actually mentioned
                    import re
                    law_pattern = r"(điểm\s+\w+\s+)?khoản\s+\d+\s+điều\s+\d+"
                    actual_laws = re.findall(law_pattern, passage_text, re.IGNORECASE)
                    if actual_laws:
                        print(f"     Found in text: {actual_laws}")

                    issues_found.append(f"{person_name}: Wrong law article (claimed '{law_article}')")
                else:
                    print(f"  ✓ Law article correct: '{law_article}'")

        # Check birth_place
        birth_place = person.get("birth_place", {})
        if birth_place.get("text"):
            bp_text = birth_place.get("text")
            bp_start = birth_place.get("start", -1)
            bp_end = birth_place.get("end", -1)

            if bp_start != -1 and bp_end != -1 and bp_start == bp_end:
                print(f"  ⚠️  Birth place '{bp_text}' has invalid position: {bp_start}:{bp_end}")
                issues_found.append(f"{person_name}: Birth place position invalid")

    # Check for missing people
    print("\n" + "=" * 80)
    print("CHECKING FOR MISSING PEOPLE")
    print("=" * 80)

    # Look for patterns like "ông/bà [Name]"
    import re
    name_pattern = r"(?:ông|bà|anh|chị)\s+([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ][a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+(?:\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ][a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+)+)"

    found_names = re.findall(name_pattern, passage_text)
    extracted_names_set = {p.get("name", {}).get("text", "").lower() for p in extracted_people}

    missing = []
    for name in found_names:
        if name.lower() not in extracted_names_set and len(name.split()) >= 2:
            # Check if this is actually in the extracted list (case-insensitive)
            if not any(name.lower() in en for en in extracted_names_set):
                missing.append(name)

    if missing:
        print(f"\n⚠️  Potentially missing people: {set(missing)}")
        for name in set(missing):
            issues_found.append(f"Missing person: '{name}' appears in text but not extracted")
    else:
        print("✓ No obviously missing people detected")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total issues found: {len(issues_found)}")

    if issues_found:
        print("\nIssues:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
    else:
        print("✓ No issues detected")

    print("\n" + "=" * 80)
    print("This demonstrates what the LLM-based validator will check automatically.")
    print("Run with Ollama server active to get full LLM-powered validation.")
    print("=" * 80)


if __name__ == "__main__":
    analyze_sample_file()

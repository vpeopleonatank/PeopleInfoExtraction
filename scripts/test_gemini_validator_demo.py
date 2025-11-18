#!/usr/bin/env python3
"""
Demo script that exercises the Gemini validator with a stubbed Gemini client.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# Allow importing src modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.gemini_validator import GeminiFlashValidator


class StubGeminiClient:
    """Simple stub that mimics python-genai's client.models interface."""

    class _Models:
        def __init__(self, parent: "StubGeminiClient") -> None:
            self.parent = parent

        def generate_content(self, *, model: str, contents: str, config=None):
            self.parent.messages.append(contents)
            return SimpleNamespace(text=json.dumps(self.parent.response, ensure_ascii=False))

    def __init__(self, response: dict) -> None:
        self.response = response
        self.messages: list[str] = []
        self.models = self._Models(self)


def load_file(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini validator demo using stubbed responses.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/cache/workflow_b_fast/pass1/019a2f6a-4fb7-7b90-840c-f5c3d20f5093_p0.json"),
        help="Extraction JSON to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_file(args.file)

    stub_response = {
        "verdict": "unsupported",
        "issues": [
            {
                "type": "hallucination",
                "person_name": "Ông A",
                "field": "actions[0].law_article",
                "description": "Law article 999 is not present in the passage.",
                "evidence": None,
            }
        ],
        "missing_people": [
            {"name": "Nguyễn Văn B", "snippet": "Nhắc tới trong đoạn thứ hai nhưng không có trong extraction."}
        ],
    }

    stub_client = StubGeminiClient(stub_response)
    validator = GeminiFlashValidator(
        client=stub_client,
        model_name="gemini-2.5-flash-stub",
        api_key="stub",
    )

    passage = data.get("passage", {})
    response = data.get("response", {})
    parsed_people = response.get("parsed", {}).get("people", []) or []

    report = validator.validate_extraction(
        doc_id=data.get("doc_id", "unknown"),
        passage_id=data.get("passage_id", 0),
        passage_text=passage.get("text", ""),
        extracted_people=parsed_people,
        raw_response_text=response.get("raw"),
        original_model=data.get("model"),
        article_title=data.get("article_title"),
        published_at=data.get("published_at"),
        source_file=str(args.file),
    )

    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

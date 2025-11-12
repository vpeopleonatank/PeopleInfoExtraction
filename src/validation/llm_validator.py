"""LLM-based validator for people extraction outputs."""

import json
import re
from datetime import datetime
from typing import Any, Optional

from ..llm.ollama import OllamaClient, OllamaError
from .models import (
    IssueType,
    IssueSeverity,
    ValidationIssue,
    ValidationReport,
    ValidationSummary,
)


class PeopleValidator:
    """Validates LLM extraction outputs using a larger local model."""

    # Allowed values for schema validation
    ALLOWED_ROLES = {"suspect", "victim", "official", "witness", "lawyer", "judge", "other"}
    ALLOWED_PREDICATES = {"arrested", "charged", "sentenced", "confessed", "searched", "seized", "other"}

    def __init__(
        self,
        validator_model: str = "qwen2.5:14b",
        temperature: float = 0.0,
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the validator.

        Args:
            validator_model: Ollama model to use for validation (should be larger/better than original)
            temperature: Temperature for LLM generation (0.0 for deterministic)
            ollama_base_url: Base URL for Ollama API
        """
        self.validator_model = validator_model
        self.client = OllamaClient(
            model=validator_model,
            temperature=temperature,
            base_url=ollama_base_url,
        )

    def validate_extraction(
        self,
        doc_id: str,
        passage_id: int,
        passage_text: str,
        extracted_people: list[dict[str, Any]],
        original_model: str,
        article_title: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> ValidationReport:
        """
        Validate a complete extraction output.

        Args:
            doc_id: Document ID
            passage_id: Passage ID
            passage_text: Original passage text
            extracted_people: List of people extracted by the original model
            original_model: Name of the model that did the extraction
            article_title: Optional article title
            source_file: Optional source file path

        Returns:
            ValidationReport with all issues found
        """
        report = ValidationReport(
            doc_id=doc_id,
            passage_id=passage_id,
            article_title=article_title,
            validator_model=self.validator_model,
            original_model=original_model,
            source_file=source_file,
            summary=ValidationSummary(
                total_people_extracted=len(extracted_people),
                total_issues_found=0,
            ),
            issues=[],
        )

        # Run all validation checks
        for person in extracted_people:
            person_name = person.get("name", {}).get("text", "Unknown")

            # 1. Character position validation
            self._validate_character_positions(report, person, passage_text, person_name)

            # 2. Schema compliance validation
            self._validate_schema_compliance(report, person, person_name)

            # 3. Factual accuracy validation (check for hallucinations)
            self._validate_factual_accuracy(report, person, passage_text, person_name)

        # 4. Completeness check (find missing entities) using LLM
        self._check_completeness(report, passage_text, extracted_people)

        return report

    def _validate_character_positions(
        self,
        report: ValidationReport,
        person: dict[str, Any],
        passage_text: str,
        person_name: str,
    ) -> None:
        """Validate that start/end character positions match actual text."""
        # Check name position
        name_info = person.get("name", {})
        if name_info.get("text"):
            self._check_span_position(
                report,
                person_name,
                "name",
                name_info.get("text"),
                name_info.get("start", -1),
                name_info.get("end", -1),
                passage_text,
            )

        # Check age position
        age_info = person.get("age", {})
        if age_info.get("value") is not None:
            # Age value should be numeric, not directly in text
            # But start/end should point to where it was mentioned
            start = age_info.get("start", -1)
            end = age_info.get("end", -1)
            if start != -1 and end != -1 and start != end:
                # Check if the span at least looks reasonable
                if start < 0 or end > len(passage_text) or start >= end:
                    report.add_issue(
                        ValidationIssue(
                            type=IssueType.WRONG_POSITION,
                            severity=IssueSeverity.ERROR,
                            person_name=person_name,
                            field="age.start/end",
                            expected=f"valid position in range [0, {len(passage_text)}]",
                            actual=f"start={start}, end={end}",
                            description="Age position indices are invalid",
                        )
                    )

        # Check birth_place position
        birth_place_info = person.get("birth_place", {})
        if birth_place_info.get("text"):
            self._check_span_position(
                report,
                person_name,
                "birth_place",
                birth_place_info.get("text"),
                birth_place_info.get("start", -1),
                birth_place_info.get("end", -1),
                passage_text,
            )

        # Check phone numbers
        phones = person.get("phones", [])
        for idx, phone_info in enumerate(phones):
            if phone_info.get("text"):
                self._check_span_position(
                    report,
                    person_name,
                    f"phones[{idx}]",
                    phone_info.get("text"),
                    phone_info.get("start", -1),
                    phone_info.get("end", -1),
                    passage_text,
                )

        # Check national_id
        national_id_info = person.get("national_id", {})
        if national_id_info.get("text"):
            self._check_span_position(
                report,
                person_name,
                "national_id",
                national_id_info.get("text"),
                national_id_info.get("start", -1),
                national_id_info.get("end", -1),
                passage_text,
            )

        # Check aliases
        aliases = person.get("aliases", [])
        for idx, alias_info in enumerate(aliases):
            if alias_info.get("text"):
                self._check_span_position(
                    report,
                    person_name,
                    f"aliases[{idx}]",
                    alias_info.get("text"),
                    alias_info.get("start", -1),
                    alias_info.get("end", -1),
                    passage_text,
                )

    def _check_span_position(
        self,
        report: ValidationReport,
        person_name: str,
        field: str,
        expected_text: str,
        start: int,
        end: int,
        passage_text: str,
    ) -> None:
        """Check a single span's position against the passage text."""
        if start == -1 or end == -1:
            return  # No position provided, skip

        # Check bounds
        if start < 0 or end > len(passage_text) or start >= end:
            report.add_issue(
                ValidationIssue(
                    type=IssueType.WRONG_POSITION,
                    severity=IssueSeverity.ERROR,
                    person_name=person_name,
                    field=f"{field}.start/end",
                    expected=f"valid position in range [0, {len(passage_text)}]",
                    actual=f"start={start}, end={end}",
                    description=f"Position indices are out of bounds or invalid",
                )
            )
            return

        # Extract actual text at position
        actual_text = passage_text[start:end]

        # Check if texts match
        if actual_text != expected_text:
            # Try to find where the text actually appears
            actual_position = passage_text.find(expected_text)
            if actual_position != -1:
                report.add_issue(
                    ValidationIssue(
                        type=IssueType.WRONG_POSITION,
                        severity=IssueSeverity.ERROR,
                        person_name=person_name,
                        field=field,
                        expected=f"'{expected_text}' at position {actual_position}",
                        actual=f"'{actual_text}' at position {start}",
                        description=f"Text at specified position doesn't match. Found at position {actual_position} instead.",
                        text_snippet=passage_text[max(0, actual_position - 20):actual_position + len(expected_text) + 20],
                    )
                )
            else:
                # Text not found anywhere in passage - possible hallucination
                report.add_issue(
                    ValidationIssue(
                        type=IssueType.HALLUCINATION,
                        severity=IssueSeverity.CRITICAL,
                        person_name=person_name,
                        field=field,
                        expected="text present in passage",
                        actual=f"'{expected_text}' not found",
                        description=f"Text '{expected_text}' does not appear in the passage at all",
                    )
                )

    def _validate_schema_compliance(
        self,
        report: ValidationReport,
        person: dict[str, Any],
        person_name: str,
    ) -> None:
        """Validate that role labels and action predicates match allowed values."""
        # Check roles
        roles = person.get("roles", [])
        for idx, role in enumerate(roles):
            label = role.get("label")
            if label and label not in self.ALLOWED_ROLES:
                report.add_issue(
                    ValidationIssue(
                        type=IssueType.SCHEMA_VIOLATION,
                        severity=IssueSeverity.WARNING,
                        person_name=person_name,
                        field=f"roles[{idx}].label",
                        expected=f"one of {self.ALLOWED_ROLES}",
                        actual=label,
                        description=f"Role label '{label}' is not in allowed set",
                    )
                )

        # Check actions
        actions = person.get("actions", [])
        for idx, action in enumerate(actions):
            predicate = action.get("predicate")
            if predicate and predicate not in self.ALLOWED_PREDICATES:
                report.add_issue(
                    ValidationIssue(
                        type=IssueType.SCHEMA_VIOLATION,
                        severity=IssueSeverity.WARNING,
                        person_name=person_name,
                        field=f"actions[{idx}].predicate",
                        expected=f"one of {self.ALLOWED_PREDICATES}",
                        actual=predicate,
                        description=f"Action predicate '{predicate}' is not in allowed set",
                    )
                )

    def _validate_factual_accuracy(
        self,
        report: ValidationReport,
        person: dict[str, Any],
        passage_text: str,
        person_name: str,
    ) -> None:
        """Validate that extracted facts (law articles, amounts, etc.) are accurate."""
        actions = person.get("actions", [])
        for idx, action in enumerate(actions):
            law_article = action.get("law_article", "")
            if law_article:
                # Check if law article appears in text
                # Normalize for comparison (remove extra spaces)
                normalized_law = " ".join(law_article.split())
                normalized_passage = " ".join(passage_text.split())

                if normalized_law not in normalized_passage:
                    # Try to find similar law articles in text
                    law_pattern = r"(điều|khoản|điểm)\s+\w+.*?(bộ luật|luật)\s+\w+"
                    found_laws = re.findall(law_pattern, passage_text, re.IGNORECASE)

                    report.add_issue(
                        ValidationIssue(
                            type=IssueType.WRONG_LAW_ARTICLE,
                            severity=IssueSeverity.CRITICAL,
                            person_name=person_name,
                            field=f"actions[{idx}].law_article",
                            expected=f"law article present in text (found: {found_laws[:3]})" if found_laws else "law article present in text",
                            actual=law_article,
                            description=f"Law article '{law_article}' not found in passage text",
                        )
                    )

            # Check amount_vnd if present
            amount_vnd = action.get("amount_vnd")
            if amount_vnd is not None:
                # Convert to string and check if it appears in some form
                amount_str = str(amount_vnd)
                # This is a soft check - amounts can be written in many ways
                # Just flag if suspiciously large or seems fabricated
                if amount_vnd > 1_000_000_000_000:  # 1 trillion VND
                    report.add_issue(
                        ValidationIssue(
                            type=IssueType.HALLUCINATION,
                            severity=IssueSeverity.WARNING,
                            person_name=person_name,
                            field=f"actions[{idx}].amount_vnd",
                            expected="reasonable amount",
                            actual=str(amount_vnd),
                            description=f"Amount {amount_vnd} VND seems unusually large",
                        )
                    )

    def _check_completeness(
        self,
        report: ValidationReport,
        passage_text: str,
        extracted_people: list[dict[str, Any]],
    ) -> None:
        """Use LLM to check if any people/entities were missed."""
        # Create a prompt asking the validator to find any people not in the extraction
        extracted_names = [p.get("name", {}).get("text", "") for p in extracted_people]
        extracted_names_str = ", ".join([f'"{n}"' for n in extracted_names if n])

        system_prompt = """Bạn là hệ thống kiểm tra chất lượng trích xuất thông tin.
Nhiệm vụ: Tìm tất cả TÊN NGƯỜI được đề cập trong đoạn văn mà CHƯA có trong danh sách đã trích xuất.

Quy tắc:
- Chỉ tìm tên người (không phải tổ chức, địa danh)
- Chỉ liệt kê tên người xuất hiện NGUYÊN VĂN trong văn bản
- Bỏ qua các danh xưng (ông, bà, anh, chị)
- Trả về JSON: {"missing_people": ["Tên 1", "Tên 2", ...]}
- Nếu không thiếu ai, trả về: {"missing_people": []}"""

        user_prompt = f"""Đoạn văn:
{passage_text}

Danh sách đã trích xuất:
{extracted_names_str if extracted_names_str else "(rỗng)"}

Tìm tên người còn thiếu:"""

        try:
            result = self.client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                format="json",
            )

            # Parse response
            try:
                response_data = json.loads(result.content)
                missing_people = response_data.get("missing_people", [])

                for missing_name in missing_people:
                    if missing_name and missing_name.strip():
                        # Verify this name actually appears in text
                        if missing_name in passage_text:
                            report.add_issue(
                                ValidationIssue(
                                    type=IssueType.MISSING_ENTITY,
                                    severity=IssueSeverity.WARNING,
                                    person_name=None,
                                    field="people",
                                    expected=f"extraction should include '{missing_name}'",
                                    actual="not extracted",
                                    description=f"Person '{missing_name}' appears in text but was not extracted",
                                    text_snippet=self._get_context_snippet(passage_text, missing_name),
                                )
                            )
            except json.JSONDecodeError:
                # LLM didn't return valid JSON, skip completeness check
                pass

        except OllamaError as e:
            # If LLM call fails, add an info issue but don't fail validation
            report.add_issue(
                ValidationIssue(
                    type=IssueType.INCOMPLETE_EXTRACTION,
                    severity=IssueSeverity.INFO,
                    person_name=None,
                    field="completeness_check",
                    expected="LLM completeness check",
                    actual="check failed",
                    description=f"Could not perform completeness check: {str(e)}",
                )
            )

    def _get_context_snippet(self, text: str, target: str, context_chars: int = 50) -> str:
        """Get a snippet of text around the target string."""
        pos = text.find(target)
        if pos == -1:
            return ""

        start = max(0, pos - context_chars)
        end = min(len(text), pos + len(target) + context_chars)

        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

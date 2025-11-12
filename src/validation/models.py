"""Pydantic models for validation reports."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IssueType(str, Enum):
    """Types of validation issues."""

    HALLUCINATION = "hallucination"
    WRONG_POSITION = "wrong_position"
    MISSING_ENTITY = "missing_entity"
    SCHEMA_VIOLATION = "schema_violation"
    WRONG_LAW_ARTICLE = "wrong_law_article"
    INCOMPLETE_EXTRACTION = "incomplete_extraction"


class IssueSeverity(str, Enum):
    """Severity levels for validation issues."""

    CRITICAL = "critical"  # Hallucinations, fabricated data
    ERROR = "error"  # Wrong positions, incorrect data
    WARNING = "warning"  # Potential issues, edge cases
    INFO = "info"  # Suggestions, completeness notes


class ValidationIssue(BaseModel):
    """A single validation issue found in the extraction."""

    type: IssueType
    severity: IssueSeverity
    person_name: Optional[str] = None
    field: Optional[str] = None  # e.g., "name.start", "actions[0].law_article"
    expected: Optional[str] = None
    actual: Optional[str] = None
    description: str
    text_snippet: Optional[str] = None  # Relevant text from source

    class Config:
        use_enum_values = True


class ValidationSummary(BaseModel):
    """Summary statistics for a validation report."""

    total_people_extracted: int
    total_issues_found: int
    critical_issues: int = 0
    errors: int = 0
    warnings: int = 0
    info_issues: int = 0

    hallucinations: int = 0
    wrong_positions: int = 0
    missing_entities: int = 0
    schema_violations: int = 0
    wrong_law_articles: int = 0


class ValidationReport(BaseModel):
    """Complete validation report for a single passage extraction."""

    doc_id: str
    passage_id: int
    article_title: Optional[str] = None

    validator_model: str
    original_model: str
    validation_timestamp: datetime = Field(default_factory=datetime.now)

    summary: ValidationSummary
    issues: list[ValidationIssue] = Field(default_factory=list)

    # Original file path for reference
    source_file: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue and update summary counts."""
        self.issues.append(issue)
        self.summary.total_issues_found += 1

        # Update severity counts
        if issue.severity == IssueSeverity.CRITICAL:
            self.summary.critical_issues += 1
        elif issue.severity == IssueSeverity.ERROR:
            self.summary.errors += 1
        elif issue.severity == IssueSeverity.WARNING:
            self.summary.warnings += 1
        elif issue.severity == IssueSeverity.INFO:
            self.summary.info_issues += 1

        # Update type counts
        if issue.type == IssueType.HALLUCINATION:
            self.summary.hallucinations += 1
        elif issue.type == IssueType.WRONG_POSITION:
            self.summary.wrong_positions += 1
        elif issue.type == IssueType.MISSING_ENTITY:
            self.summary.missing_entities += 1
        elif issue.type == IssueType.SCHEMA_VIOLATION:
            self.summary.schema_violations += 1
        elif issue.type == IssueType.WRONG_LAW_ARTICLE:
            self.summary.wrong_law_articles += 1


class BatchValidationSummary(BaseModel):
    """Summary across multiple validation reports."""

    total_files_processed: int
    total_files_with_issues: int
    total_issues: int

    total_people_extracted: int

    critical_issues: int = 0
    errors: int = 0
    warnings: int = 0
    info_issues: int = 0

    hallucinations: int = 0
    wrong_positions: int = 0
    missing_entities: int = 0
    schema_violations: int = 0
    wrong_law_articles: int = 0

    validation_timestamp: datetime = Field(default_factory=datetime.now)
    validator_model: str
    original_model: str

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

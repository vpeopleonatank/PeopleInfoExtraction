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


class GeminiIssueType(str, Enum):
    """Issue categories returned by the Gemini validator."""

    HALLUCINATION = "hallucination"
    MISSING_INFO = "missing_info"
    CONFLICT = "conflict"
    PARSING_ERROR = "parsing_error"
    OTHER = "other"


class GeminiIssueSeverity(str, Enum):
    """Severity levels for Gemini issues."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class GeminiVerdict(str, Enum):
    """High-level verdict returned by the Gemini validator."""

    SUPPORTED = "supported"
    UNSURE = "unsure"
    UNSUPPORTED = "unsupported"


class ManualValidationStatus(str, Enum):
    """Manual override for report validity."""

    VALID = "valid"
    INVALID = "invalid"


class GeminiValidationIssue(BaseModel):
    """Structured issue entry for Gemini validation."""

    type: GeminiIssueType
    description: str
    severity: Optional[GeminiIssueSeverity] = None
    person_name: Optional[str] = None
    field: Optional[str] = None
    evidence: Optional[str] = None

    class Config:
        use_enum_values = True


class GeminiMissingPerson(BaseModel):
    """Represents a potentially missed entity detected by Gemini."""

    name: str
    snippet: Optional[str] = None


class GeminiReportSummary(BaseModel):
    """Summary statistics for a Gemini validation report."""

    verdict: GeminiVerdict
    issue_count: int = 0
    missing_people_count: int = 0
    usable: Optional[bool] = None
    confidence: Optional[float] = None


class GeminiValidationReport(BaseModel):
    """Complete report generated by the Gemini validator."""

    doc_id: str
    passage_id: int
    article_title: Optional[str] = None
    published_at: Optional[str] = None

    validator_model: str
    original_model: Optional[str] = None

    verdict: GeminiVerdict = GeminiVerdict.UNSURE
    manual_status: Optional[ManualValidationStatus] = None
    usable: Optional[bool] = None
    confidence: Optional[float] = None
    issues: list[GeminiValidationIssue] = Field(default_factory=list)
    missing_people: list[GeminiMissingPerson] = Field(default_factory=list)
    summary: GeminiReportSummary

    validation_timestamp: datetime = Field(default_factory=datetime.now)
    source_file: Optional[str] = None

    prompt_truncated: bool = False
    prompt_char_count: int = 0
    raw_response_text: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GeminiBatchSummary(BaseModel):
    """Aggregate statistics across Gemini validation reports."""

    total_files_processed: int = 0
    total_supported: int = 0
    total_unsure: int = 0
    total_unsupported: int = 0
    total_issues: int = 0
    total_missing_people: int = 0
    total_usable: int = 0

    usable_percentage: float = 0.0
    supported_percentage: float = 0.0
    unsure_percentage: float = 0.0
    unsupported_percentage: float = 0.0
    validator_model: str
    generated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def register_report(self, report: GeminiValidationReport) -> None:
        """Update the summary counters based on a single report."""
        self.total_files_processed += 1
        self.total_issues += len(report.issues)
        self.total_missing_people += len(report.missing_people)

        if self._effective_usable(report):
            self.total_usable += 1

        if report.verdict == GeminiVerdict.SUPPORTED:
            self.total_supported += 1
        elif report.verdict == GeminiVerdict.UNSURE:
            self.total_unsure += 1
        else:
            self.total_unsupported += 1

        self._update_percentages()

    def _effective_usable(self, report: GeminiValidationReport) -> bool:
        """
        Respect the model's explicit usable flag when provided, but enforce the rule-based
        guardrails (no critical issues, no missing people, verdict must be supported).
        """
        rule_usable = self._is_report_usable(report)
        if report.usable is None:
            return rule_usable
        return bool(report.usable) and rule_usable

    def _is_report_usable(self, report: GeminiValidationReport) -> bool:
        """
        Define "usable" as supported, with no missing people, and no critical issues.
        Issues missing a severity are treated as critical for safety.
        """
        return (
            report.verdict == GeminiVerdict.SUPPORTED
            and not report.missing_people
            and not self._has_critical_issue(report.issues)
        )

    def _has_critical_issue(self, issues: list[GeminiValidationIssue]) -> bool:
        """Detect if any issue is critical (or missing severity, which is treated as critical)."""
        for issue in issues:
            severity = issue.severity or GeminiIssueSeverity.CRITICAL
            if severity == GeminiIssueSeverity.CRITICAL:
                return True
        return False

    def _update_percentages(self) -> None:
        """Recompute percentage fields after counters are updated."""
        total = self.total_files_processed
        if total <= 0:
            self.usable_percentage = 0.0
            self.supported_percentage = 0.0
            self.unsure_percentage = 0.0
            self.unsupported_percentage = 0.0
            return

        self.usable_percentage = round((self.total_usable / total) * 100, 2)
        self.supported_percentage = round((self.total_supported / total) * 100, 2)
        self.unsure_percentage = round((self.total_unsure / total) * 100, 2)
        self.unsupported_percentage = round((self.total_unsupported / total) * 100, 2)

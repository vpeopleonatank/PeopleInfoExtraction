"""Shared span schema + helpers for deterministic + learned extractors."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, ValidationError


class SpanSource(str, Enum):
    """Origin of a span."""

    REGEX = "regex"
    NER = "ner"
    LLM = "llm"
    HEURISTIC = "heuristic"


class SpanType(str, Enum):
    """Canonical span types we emit across the pipeline."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    PHONE = "phone_number"
    NATIONAL_ID = "national_id"
    DATE_OF_BIRTH = "date_of_birth"
    CURRENCY = "currency_amount"
    LAW = "law_reference"
    OTHER = "other"


@dataclass(slots=True)
class Span:
    """Lightweight container for matched spans."""

    doc_id: str
    span_type: SpanType
    text: str
    start: int
    end: int
    sentence_id: str
    source: SpanSource
    passage_id: Optional[int] = None
    confidence: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "span_type": self.span_type.value,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "sentence_id": self.sentence_id,
            "source": self.source.value,
            "passage_id": self.passage_id,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }


class SpanModel(BaseModel):
    """Pydantic representation mirroring the runtime dataclass."""

    doc_id: str
    span_type: SpanType
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    sentence_id: str
    source: SpanSource
    passage_id: Optional[int] = Field(default=None, ge=0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    attributes: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @property
    def length(self) -> int:
        return self.end - self.start


def validate_span(span: Span | SpanModel | Dict[str, Any]) -> SpanModel:
    """Validate a span dataclass/dict and return a `SpanModel`."""

    if isinstance(span, SpanModel):
        return span
    if isinstance(span, Span):
        payload = span.to_dict()
    else:
        payload = span
    return SpanModel.model_validate(payload)


def validate_spans(spans: Iterable[Span | SpanModel | Dict[str, Any]]) -> List[SpanModel]:
    """Validate a collection of span-like objects."""

    validated: List[SpanModel] = []
    for span in spans:
        try:
            validated.append(validate_span(span))
        except ValidationError as exc:
            raise ValueError(f"Invalid span payload: {span}") from exc
    return validated

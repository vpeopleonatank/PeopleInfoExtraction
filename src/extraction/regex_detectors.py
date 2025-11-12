"""Regex-based detectors for structured person attributes."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

from src.extraction.spans import Span, SpanSource, SpanType

NormalizerResult = Optional[Tuple[str, Dict[str, Any]]]
Normalizer = Callable[[re.Match[str]], NormalizerResult]


@dataclass(slots=True)
class RegexDetector:
    """Wrapper around a compiled regex + normalization logic."""

    name: str
    span_type: SpanType
    pattern: Pattern[str]
    normalizer: Normalizer
    confidence: float = 0.9

    def detect(
        self,
        text: str,
        *,
        doc_id: str,
        sentence_id: str,
        passage_id: Optional[int],
        base_offset: int,
    ) -> List[Span]:
        matches: List[Span] = []
        for match in self.pattern.finditer(text):
            normalized = self.normalizer(match)
            if not normalized:
                continue
            normalized_value, extra_attrs = normalized
            start = base_offset + match.start()
            end = base_offset + match.end()
            attributes = {"detector": self.name, "normalized": normalized_value, "raw": match.group(0)}
            attributes.update(extra_attrs)
            matches.append(
                Span(
                    doc_id=doc_id,
                    span_type=self.span_type,
                    text=match.group(0).strip(),
                    start=start,
                    end=end,
                    sentence_id=sentence_id,
                    source=SpanSource.REGEX,
                    passage_id=passage_id,
                    confidence=self.confidence,
                    attributes=attributes,
                )
            )
        return matches


def _normalize_phone(match: re.Match[str]) -> NormalizerResult:
    raw_digits = re.sub(r"\D", "", match.group(0))
    if len(raw_digits) < 9:
        return None
    if raw_digits.startswith("84"):
        national = raw_digits[2:]
        if not national.startswith("0"):
            national = "0" + national
    else:
        national = raw_digits
    if len(national) not in (10, 11):
        return None
    e164 = "+84" + national.lstrip("0")
    return national, {"e164": e164}


def _normalize_digits(match: re.Match[str]) -> NormalizerResult:
    digits = re.sub(r"\D", "", match.group("id") if "id" in match.groupdict() else match.group(0))
    if len(digits) not in (9, 12):
        return None
    return digits, {"length": len(digits)}


def _normalize_full_date(match: re.Match[str]) -> NormalizerResult:
    day = int(match.group("day"))
    month = int(match.group("month"))
    year = int(match.group("year"))
    if year < 1900:
        year += 1900 if year >= 30 else 2000
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    normalized = f"{year:04d}-{month:02d}-{day:02d}"
    return normalized, {"precision": "date", "year": year, "month": month, "day": day}


def _normalize_birth_year(match: re.Match[str]) -> NormalizerResult:
    year = int(match.group("year"))
    if year < 1900 or year > 2100:
        return None
    return str(year), {"precision": "year", "year": year}


def _parse_vietnamese_number(text: str) -> float:
    sanitized = text.replace(".", "").replace(" ", "")
    sanitized = sanitized.replace(",", ".")
    try:
        return float(sanitized)
    except ValueError:
        return float("nan")


CURRENCY_MULTIPLIERS: Dict[str, int] = {
    "tỷ": 1_000_000_000,
    "ty": 1_000_000_000,
    "triệu": 1_000_000,
    "trieu": 1_000_000,
    "nghìn": 1_000,
    "ngàn": 1_000,
    "ngan": 1_000,
    "đồng": 1,
    "dong": 1,
    "đ": 1,
    "d": 1,
    "vnd": 1,
}


def _normalize_currency(match: re.Match[str]) -> NormalizerResult:
    amount_text = match.group("amount")
    unit = (match.group("unit") or "").lower()
    number = _parse_vietnamese_number(amount_text)
    if number != number:  # NaN check
        return None
    multiplier = CURRENCY_MULTIPLIERS.get(unit, 1)
    normalized_vnd = int(round(number * multiplier))
    normalized = f"{normalized_vnd}"
    return normalized, {"amount_vnd": normalized_vnd, "unit": unit or "đồng"}


def _normalize_law(match: re.Match[str]) -> NormalizerResult:
    article = match.group("article")
    if not article:
        return None
    article_num = re.sub(r"[^\dA-Za-z]", "", article)
    code = match.group("code")
    year = match.group("year")
    attrs: Dict[str, Any] = {}
    if code:
        attrs["code"] = code.strip()
    if year:
        attrs["year"] = int(year)
    return article_num, attrs


PHONE_PATTERN = re.compile(
    r"(?:(?:\+?84)|0)(?:[\s\-.]?\d){8,10}",
    flags=re.UNICODE,
)
NATIONAL_ID_PATTERN = re.compile(
    r"(?:(?:CMND|CCCD)(?:\s+s(?:ố|o))?|s(?:ố|o)\s+(?:CMND|CCCD))[:\s-]*(?P<id>\d[\d\s\.]{8,})",
    flags=re.IGNORECASE,
)
DOB_DATE_PATTERN = re.compile(
    r"(?:(?:sinh\s+(?:ngày|ngày)\s*)|(?:ngày\s+sinh\s*)|)(?P<day>\d{1,2})[./-](?P<month>\d{1,2})[./-](?P<year>\d{2,4})",
    flags=re.IGNORECASE,
)
DOB_YEAR_PATTERN = re.compile(
    r"(?:sinh\s+năm|sinh|sn)\s*(?P<year>19\d{2}|20\d{2})",
    flags=re.IGNORECASE,
)
CURRENCY_PATTERN = re.compile(
    r"(?P<amount>\d{1,3}(?:[\.\s]\d{3})*(?:,\d+)?|\d+(?:,\d+)?)\s*(?P<unit>tỷ|ty|triệu|trieu|nghìn|ngàn|ngan|đồng|dong|đ|d|vnd)",
    flags=re.IGNORECASE,
)
LAW_PATTERN = re.compile(
    r"Điều\s+(?P<article>\d{1,3}[A-Za-z]?)"
    r"(?:\s+(?P<code>(?:Bộ\s+luật\s+Hình\s+sự)|BLHS)(?:\s+(?P<year>\d{4}))?)?",
    flags=re.IGNORECASE,
)


DETECTORS: Sequence[RegexDetector] = (
    RegexDetector(
        name="phone",
        span_type=SpanType.PHONE,
        pattern=PHONE_PATTERN,
        normalizer=_normalize_phone,
        confidence=0.95,
    ),
    RegexDetector(
        name="national_id",
        span_type=SpanType.NATIONAL_ID,
        pattern=NATIONAL_ID_PATTERN,
        normalizer=_normalize_digits,
        confidence=0.9,
    ),
    RegexDetector(
        name="dob_full",
        span_type=SpanType.DATE_OF_BIRTH,
        pattern=DOB_DATE_PATTERN,
        normalizer=_normalize_full_date,
        confidence=0.85,
    ),
    RegexDetector(
        name="dob_year",
        span_type=SpanType.DATE_OF_BIRTH,
        pattern=DOB_YEAR_PATTERN,
        normalizer=_normalize_birth_year,
        confidence=0.8,
    ),
    RegexDetector(
        name="currency",
        span_type=SpanType.CURRENCY,
        pattern=CURRENCY_PATTERN,
        normalizer=_normalize_currency,
        confidence=0.8,
    ),
    RegexDetector(
        name="law_article",
        span_type=SpanType.LAW,
        pattern=LAW_PATTERN,
        normalizer=_normalize_law,
        confidence=0.9,
    ),
)


def run_regex_detectors(
    text: str,
    *,
    doc_id: str,
    sentence_id: str,
    base_offset: int,
    passage_id: Optional[int] = None,
    detectors: Iterable[RegexDetector] = DETECTORS,
) -> List[Span]:
    """Run all registered detectors over *text*."""

    spans: List[Span] = []
    for detector in detectors:
        spans.extend(
            detector.detect(
                text,
                doc_id=doc_id,
                sentence_id=sentence_id,
                passage_id=passage_id,
                base_offset=base_offset,
            )
        )
    return spans


__all__ = [
    "RegexDetector",
    "DETECTORS",
    "run_regex_detectors",
]

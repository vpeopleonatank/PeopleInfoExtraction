"""Verifier + normalizer for span outputs."""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.extraction.spans import SpanModel, SpanType, validate_spans

LOGGER = logging.getLogger(__name__)
WHITESPACE_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"(19|20)\d{2}")


@dataclass(slots=True)
class VerificationResult:
    span: SpanModel
    verified: bool
    reason: Optional[str] = None


def _clean_text(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value or "").strip()


def build_full_text(article: dict) -> str:
    paragraphs = article.get("paragraphs") or []
    return "\n\n".join(paragraphs)


def _derive_birth_year(attributes: Dict[str, object], fallback_text: str) -> Optional[int]:
    candidate = attributes.get("normalized") or fallback_text
    if not isinstance(candidate, str):
        return None
    match = YEAR_RE.search(candidate)
    if match:
        return int(match.group(0))
    return None


def _apply_derivations(span: SpanModel) -> Dict[str, object]:
    attributes: Dict[str, object] = dict(span.attributes)
    derivations: List[Dict[str, object]] = list(attributes.get("derivations", []))  # type: ignore[var-annotated]
    if span.span_type == SpanType.DATE_OF_BIRTH:
        birth_year = _derive_birth_year(attributes, span.text)
        if birth_year:
            attributes["birth_year"] = birth_year
            derivations.append(
                {"field": "birth_year", "derivation": "dob->birth_year", "source_span": [span.start, span.end]}
            )
    if span.span_type == SpanType.CURRENCY and "amount_vnd" in attributes:
        derivations.append(
            {"field": "amount_vnd", "derivation": "currency_normalizer", "source_span": [span.start, span.end]}
        )
    if derivations:
        attributes["derivations"] = derivations
    return attributes


def verify_span(span: SpanModel, *, full_text: str) -> VerificationResult:
    if span.end > len(full_text):
        return VerificationResult(span=span, verified=False, reason="out_of_bounds")
    doc_slice = full_text[span.start : span.end]
    if not doc_slice:
        return VerificationResult(span=span, verified=False, reason="empty_slice")
    if _clean_text(doc_slice) != _clean_text(span.text):
        return VerificationResult(span=span, verified=False, reason="text_mismatch")
    attributes = _apply_derivations(span)
    attributes["verifier_pass"] = True
    attributes["verifier_source"] = "verbatim"
    base_conf = span.confidence if span.confidence is not None else 0.5
    new_span = span.model_copy(update={"confidence": round(base_conf, 4), "attributes": attributes})
    return VerificationResult(span=new_span, verified=True)


def verify_document(article: dict, spans: Sequence[SpanModel]) -> Tuple[List[SpanModel], List[VerificationResult], dict]:
    """Verify all spans for an article."""

    full_text = build_full_text(article)
    verified_spans: List[SpanModel] = []
    failures: List[VerificationResult] = []
    for span in spans:
        result = verify_span(span, full_text=full_text)
        if result.verified:
            verified_spans.append(result.span)
        else:
            failures.append(result)
    stats = _aggregate_stats(spans, verified_spans, failures)
    return verified_spans, failures, stats


def _aggregate_stats(
    original_spans: Sequence[SpanModel],
    verified_spans: Sequence[SpanModel],
    failures: Sequence[VerificationResult],
) -> dict:
    by_type: Dict[str, Dict[str, object]] = defaultdict(lambda: {"total": 0, "verified": 0, "avg_confidence": 0.0})
    for span in original_spans:
        entry = by_type[span.span_type.value]
        entry["total"] = entry.get("total", 0) + 1
    for span in verified_spans:
        entry = by_type[span.span_type.value]
        entry["verified"] = entry.get("verified", 0) + 1
        avg = entry.get("avg_confidence", 0.0)
        count = entry.get("verified", 1)
        entry["avg_confidence"] = round(((avg * (count - 1)) + (span.confidence or 0.0)) / count, 4)
    stats = {
        "input_spans": len(original_spans),
        "verified_spans": len(verified_spans),
        "dropped_spans": len(failures),
        "by_type": dict(by_type),
        "failure_reasons": _summarize_failures(failures),
    }
    return stats


def _summarize_failures(failures: Sequence[VerificationResult]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for result in failures:
        counts[result.reason or "unknown"] += 1
    return dict(counts)


def load_spans_from_path(path: Path) -> Tuple[str, List[SpanModel], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_spans = payload.get("spans", [])
    return payload.get("doc_id") or path.stem, validate_spans(raw_spans), payload.get("stats", {})


def load_article(doc_id: str, *, articles_dir: Path) -> dict:
    article_path = articles_dir / f"{doc_id}.json"
    if not article_path.exists():
        raise FileNotFoundError(f"Article {doc_id} not found under {articles_dir}")
    return json.loads(article_path.read_text(encoding="utf-8"))


__all__ = [
    "VerificationResult",
    "build_full_text",
    "verify_span",
    "verify_document",
    "load_spans_from_path",
    "load_article",
]

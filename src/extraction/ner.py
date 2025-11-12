"""Pluggable Vietnamese NER helpers with a rule-based fallback."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.extraction.spans import Span, SpanSource, SpanType

LOGGER = logging.getLogger(__name__)


class NERBackend(str, Enum):
    AUTO = "auto"
    VNCORENLP = "vncorenlp"
    PHOBERT = "phobert"
    SIMPLE = "simple"
    NONE = "none"


SPAN_LABEL_MAP: Dict[str, SpanType] = {
    "PER": SpanType.PERSON,
    "ORG": SpanType.ORGANIZATION,
    "LOC": SpanType.LOCATION,
}


@dataclass(slots=True)
class NerEntity:
    label: SpanType
    text: str
    start: int
    end: int
    score: float


class BaseNERBackend:
    name: str = "base"

    def entities(self, sentence: str) -> List[NerEntity]:
        raise NotImplementedError


def _align_offsets(sentence: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """Align whitespace-tokenized spans back to character offsets."""

    offsets: List[Tuple[int, int]] = []
    cursor = 0
    lowered = sentence.casefold()
    for token in tokens:
        normalized = token.replace("_", " ")
        normalized = re.sub(r"\s+", " ", normalized)
        needle = normalized.casefold().strip()
        if not needle:
            offsets.append((-1, -1))
            continue
        idx = lowered.find(needle, cursor)
        if idx == -1:
            idx = lowered.find(needle)
        if idx == -1:
            offsets.append((-1, -1))
            continue
        start = idx
        end = idx + len(needle)
        cursor = end
        offsets.append((start, end))
    return offsets


def _collect_bio_entities(
    sentence: str, tagged_tokens: Sequence[Tuple[str, str]], score: float = 0.8
) -> List[NerEntity]:
    offsets = _align_offsets(sentence, [tok for tok, _ in tagged_tokens])
    entities: List[NerEntity] = []
    current_label: Optional[str] = None
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    for (token, tag), (start, end) in zip(tagged_tokens, offsets):
        if start == -1 or end == -1:
            continue
        tag = tag or "O"
        upper_tag = tag.upper()
        if upper_tag.startswith("B-"):
            if current_label and current_start is not None:
                entities.append(
                    NerEntity(
                        label=SPAN_LABEL_MAP.get(current_label, SpanType.OTHER),
                        text=sentence[current_start:current_end],
                        start=current_start,
                        end=current_end,
                        score=score,
                    )
                )
            current_label = upper_tag[2:]
            current_start = start
            current_end = end
        elif upper_tag.startswith("I-") and current_label == upper_tag[2:]:
            current_end = end
        else:
            if current_label and current_start is not None:
                entities.append(
                    NerEntity(
                        label=SPAN_LABEL_MAP.get(current_label, SpanType.OTHER),
                        text=sentence[current_start:current_end],
                        start=current_start,
                        end=current_end,
                        score=score,
                    )
                )
            current_label = None
            current_start = None
            current_end = None
    if current_label and current_start is not None:
        entities.append(
            NerEntity(
                label=SPAN_LABEL_MAP.get(current_label, SpanType.OTHER),
                text=sentence[current_start:current_end],
                start=current_start,
                end=current_end,
                score=score,
            )
        )
    return [entity for entity in entities if entity.label != SpanType.OTHER]


class VnCoreNLPBackend(BaseNERBackend):
    name = "vncorenlp"

    def __init__(self, save_dir: Optional[str] = None) -> None:
        from vncorenlp import VnCoreNLP  # type: ignore

        model_dir = save_dir or os.environ.get("VNCORENLP_HOME")
        if not model_dir:
            raise RuntimeError("VNCORENLP_HOME is required to use the VnCoreNLP backend.")
        self._client = VnCoreNLP(
            save_dir=model_dir,
            annotators="wseg,pos,ner",
            max_heap_size="-Xmx2g",
        )

    def entities(self, sentence: str) -> List[NerEntity]:
        annotated = self._client.ner(sentence)
        if not annotated:
            return []
        tokens: List[Tuple[str, str]] = []
        for token_info in annotated[0]:
            token_text = token_info[0]
            ner_tag = token_info[-1]
            tokens.append((token_text, ner_tag))
        return _collect_bio_entities(sentence, tokens, score=0.85)


class TransformersNERBackend(BaseNERBackend):
    name = "phobert"

    def __init__(self, model_name: Optional[str] = None) -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline  # type: ignore

        self.model_name = model_name or "vinai/phobert-base"
        self._pipeline = pipeline(
            "token-classification",
            model=AutoModelForTokenClassification.from_pretrained(self.model_name),
            tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            aggregation_strategy="simple",
        )

    def entities(self, sentence: str) -> List[NerEntity]:
        results = self._pipeline(sentence)
        entities: List[NerEntity] = []
        for item in results:
            label = item.get("entity_group")
            if not label:
                continue
            span_type = SPAN_LABEL_MAP.get(label.replace("B-", "").replace("I-", ""), SpanType.OTHER)
            if span_type == SpanType.OTHER:
                continue
            score = float(item.get("score", 0.8))
            entities.append(
                NerEntity(
                    label=span_type,
                    text=item["word"],
                    start=int(item["start"]),
                    end=int(item["end"]),
                    score=score,
                )
            )
        return entities


HONORIFICS = (
    "ông",
    "bà",
    "anh",
    "chị",
    "cô",
    "chú",
    "thiếu tá",
    "trung tá",
    "thượng tá",
    "đại tá",
    "thiếu tướng",
    "thượng tướng",
    "trung tướng",
    "luật sư",
    "thẩm phán",
)


@dataclass(slots=True)
class Token:
    text: str
    start: int
    end: int

    @property
    def cleaned(self) -> str:
        return self.text.strip(".,:;!?()[]{}\"'“”‘’")


def _is_name_token(token: Token) -> bool:
    cleaned = token.cleaned
    if not cleaned or any(char.isdigit() for char in cleaned):
        return False
    if cleaned.isupper():
        return False
    return cleaned[0].isupper()


class SimpleRegexBackend(BaseNERBackend):
    """Deterministic fallback approximating PERSON mentions."""

    name = "simple"

    def entities(self, sentence: str) -> List[NerEntity]:
        tokens = [
            Token(text=match.group(0), start=match.start(), end=match.end())
            for match in re.finditer(r"\S+", sentence)
        ]
        spans: List[NerEntity] = []
        current: List[int] = []
        for idx, token in enumerate(tokens):
            if _is_name_token(token):
                current.append(idx)
            else:
                spans.extend(self._flush_span(sentence, tokens, current))
                current = []
        spans.extend(self._flush_span(sentence, tokens, current))
        return spans

    def _flush_span(self, sentence: str, tokens: List[Token], indices: List[int]) -> List[NerEntity]:
        if not indices:
            return []
        if len(indices) == 1:
            token = tokens[indices[0]]
            if len(token.cleaned) <= 2:
                return []
        start_idx = indices[0]
        honorific_bonus = 0.0
        honorific_idx = start_idx - 1
        if honorific_idx >= 0:
            honorific_token = tokens[honorific_idx]
            if honorific_token.cleaned.casefold() in HONORIFICS:
                indices.insert(0, honorific_idx)
                honorific_bonus = 0.1
        start = tokens[indices[0]].start
        end = tokens[indices[-1]].end
        text = sentence[start:end]
        confidence = min(0.55 + honorific_bonus + (0.02 * (len(indices) - 2)), 0.85)
        return [
            NerEntity(
                label=SpanType.PERSON,
                text=text,
                start=start,
                end=end,
                score=round(confidence, 3),
            )
        ]


def create_backend(backend: NERBackend = NERBackend.AUTO) -> Optional[BaseNERBackend]:
    """Instantiate the requested backend, falling back to the rule-based version."""

    if backend == NERBackend.NONE:
        return None

    if backend in {NERBackend.AUTO, NERBackend.VNCORENLP}:
        try:
            return VnCoreNLPBackend()
        except Exception as exc:  # pragma: no cover - depends on external jar
            LOGGER.warning("VnCoreNLP backend unavailable: %s", exc)
            if backend == NERBackend.VNCORENLP:
                raise

    if backend in {NERBackend.AUTO, NERBackend.PHOBERT}:
        try:
            return TransformersNERBackend()
        except Exception as exc:  # pragma: no cover - heavy dependency
            LOGGER.warning("PhoBERT backend unavailable: %s", exc)
            if backend == NERBackend.PHOBERT:
                raise

    return SimpleRegexBackend()


class NerPipeline:
    """Small utility to convert backend outputs into `Span` records."""

    def __init__(self, backend: NERBackend = NERBackend.AUTO) -> None:
        self.backend_name = backend
        self._backend = create_backend(backend)

    def extract(
        self,
        sentence: str,
        *,
        doc_id: str,
        sentence_id: str,
        base_offset: int,
        passage_id: Optional[int] = None,
    ) -> List[Span]:
        if not sentence or self._backend is None:
            return []
        spans: List[Span] = []
        for entity in self._backend.entities(sentence):
            spans.append(
                Span(
                    doc_id=doc_id,
                    span_type=entity.label,
                    text=entity.text,
                    start=base_offset + entity.start,
                    end=base_offset + entity.end,
                    sentence_id=sentence_id,
                    source=SpanSource.NER,
                    passage_id=passage_id,
                    confidence=min(max(entity.score, 0.0), 1.0),
                    attributes={"backend": getattr(self._backend, "name", "unknown")},
                )
            )
        return spans


__all__ = ["NERBackend", "NerPipeline", "create_backend"]

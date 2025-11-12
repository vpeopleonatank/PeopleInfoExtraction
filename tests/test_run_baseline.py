from typing import List

from src.extraction.run_baseline import dedupe_spans, process_article, segment_sentences
from src.extraction.spans import Span, SpanSource, SpanType


def test_segment_sentences_returns_offsets() -> None:
    sentences = segment_sentences("Ông A bị bắt. Ông A khai báo.")
    assert len(sentences) == 2
    assert sentences[0][0].startswith("Ông A")
    assert sentences[0][1] == 0
    assert sentences[1][1] > sentences[0][1]


def test_segment_sentences_preserves_offset_after_stripping() -> None:
    text = "  \t\n  Ông A bị bắt."
    sentences = segment_sentences(text)
    assert sentences == [("Ông A bị bắt.", text.index("Ông"))]


def test_dedupe_spans_prefers_higher_confidence() -> None:
    spans = [
        Span(
            doc_id="doc",
            span_type=SpanType.PERSON,
            text="Ông A",
            start=0,
            end=4,
            sentence_id="0.0",
            source=SpanSource.NER,
            confidence=0.6,
        ),
        Span(
            doc_id="doc",
            span_type=SpanType.PERSON,
            text="Ông A",
            start=0,
            end=4,
            sentence_id="0.1",
            source=SpanSource.NER,
            confidence=0.8,
        ),
    ]
    deduped = dedupe_spans(spans)
    assert len(deduped) == 1
    assert deduped[0].confidence == 0.8


class _FakeNer:
    def extract(
        self,
        sentence: str,
        *,
        doc_id: str,
        sentence_id: str,
        base_offset: int,
        passage_id: int | None = None,
    ) -> List[Span]:
        if "Ông" not in sentence:
            return []
        return [
            Span(
                doc_id=doc_id,
                span_type=SpanType.PERSON,
                text="Ông Nguyễn Văn A",
                start=base_offset,
                end=base_offset + len("Ông Nguyễn Văn A"),
                sentence_id=sentence_id,
                source=SpanSource.NER,
                passage_id=passage_id,
                confidence=0.75,
            )
        ]


def test_process_article_combines_regex_and_ner_spans() -> None:
    article = {
        "doc_id": "doc123",
        "title": "Example",
        "passages": [
            {
                "passage_id": 0,
                "text": "Ông Nguyễn Văn A sinh năm 1984. Điện thoại 0912 345 678.",
                "start": 0,
            }
        ],
    }
    spans, stats = process_article(article, ner_pipeline=_FakeNer())
    assert stats["total_spans"] >= 2
    types = {span["span_type"] for span in spans}
    assert "person" in types
    assert "phone_number" in types

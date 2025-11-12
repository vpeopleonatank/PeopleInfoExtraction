from src.extraction.ner import NERBackend, NerPipeline
from src.extraction.spans import SpanType


def test_simple_backend_merges_honorific() -> None:
    pipeline = NerPipeline(NERBackend.SIMPLE)
    spans = pipeline.extract(
        "Ông Nguyễn Văn A bị bắt.",
        doc_id="doc1",
        sentence_id="0",
        base_offset=0,
    )
    assert spans
    assert spans[0].span_type == SpanType.PERSON
    assert spans[0].text.startswith("Ông")


def test_none_backend_disables_extraction() -> None:
    pipeline = NerPipeline(NERBackend.NONE)
    spans = pipeline.extract(
        "Bà Trần Thị B là giám đốc.",
        doc_id="doc1",
        sentence_id="0",
        base_offset=0,
    )
    assert spans == []

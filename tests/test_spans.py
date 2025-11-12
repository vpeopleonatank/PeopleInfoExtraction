from src.extraction.spans import Span, SpanSource, SpanType, validate_span, validate_spans


def test_validate_span_round_trip() -> None:
    span = Span(
        doc_id="123",
        span_type=SpanType.PERSON,
        text="Ông Nguyễn Văn A",
        start=10,
        end=30,
        sentence_id="0.0",
        source=SpanSource.NER,
        passage_id=0,
        confidence=0.9,
        attributes={"backend": "simple"},
    )
    model = validate_span(span)
    assert model.doc_id == "123"
    assert model.start == 10
    assert model.attributes["backend"] == "simple"


def test_validate_spans_invalid() -> None:
    payload = {
        "doc_id": "1",
        "span_type": "person",
        "text": "Test",
        "start": -1,
        "end": 5,
        "sentence_id": "0.0",
        "source": "ner",
    }
    try:
        validate_spans([payload])
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid span start")

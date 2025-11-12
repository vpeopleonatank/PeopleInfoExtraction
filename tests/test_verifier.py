from src.extraction.spans import SpanModel, SpanSource, SpanType
from src.extraction.verifier import build_full_text, verify_document, verify_span


def _article():
    return {
        "doc_id": "doc",
        "paragraphs": ["Ông Nguyễn Văn A sinh ngày 05/11/1984.", "Ông A bị truy tố theo Điều 174."],
    }


def _span(**kwargs):
    data = {
        "doc_id": "doc",
        "span_type": SpanType.DATE_OF_BIRTH,
        "text": "05/11/1984",
        "start": 27,
        "end": 37,
        "sentence_id": "0.0",
        "source": SpanSource.REGEX,
        "confidence": 0.9,
        "attributes": {"normalized": "1984-11-05"},
    }
    data.update(kwargs)
    return SpanModel.model_validate(data)


def test_build_full_text() -> None:
    article = _article()
    text = build_full_text(article)
    assert "Ông Nguyễn Văn A" in text
    assert "\n\n" in text


def test_verify_span_success_adds_birth_year() -> None:
    article = _article()
    span = _span()
    result = verify_span(span, full_text=build_full_text(article))
    assert result.verified
    assert result.span.attributes["birth_year"] == 1984


def test_verify_document_drops_mismatch() -> None:
    article = _article()
    good = _span()
    bad = _span(text="Sai lệch", start=0, end=3)
    verified, failures, stats = verify_document(article, [good, bad])
    assert len(verified) == 1
    assert len(failures) == 1
    assert stats["verified_spans"] == 1
    assert stats["dropped_spans"] == 1

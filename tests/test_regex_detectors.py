from src.extraction.regex_detectors import run_regex_detectors
from src.extraction.spans import SpanType


def collect(text: str):
    return run_regex_detectors(text, doc_id="doc", sentence_id="0", base_offset=0)


def test_detect_phone_number() -> None:
    spans = collect("Liên hệ qua số +84 912-345-678 để xác minh.")
    phones = [span for span in spans if span.span_type == SpanType.PHONE]
    assert phones
    assert phones[0].attributes["normalized"] == "0912345678"
    assert phones[0].attributes["e164"] == "+84912345678"


def test_detect_national_id() -> None:
    spans = collect("CCCD số 079 123 456 789 cấp tại Hà Nội.")
    ids = [span for span in spans if span.span_type == SpanType.NATIONAL_ID]
    assert ids and ids[0].attributes["normalized"] == "079123456789"


def test_detect_dob_formats() -> None:
    spans = collect("Bị can sinh ngày 05/11/1984, sinh năm 1984.")
    dob_spans = [span for span in spans if span.span_type == SpanType.DATE_OF_BIRTH]
    normalized = {span.attributes["normalized"] for span in dob_spans}
    assert "1984-11-05" in normalized
    assert "1984" in normalized


def test_detect_currency_amount() -> None:
    spans = collect("Số tiền chiếm đoạt khoảng 2,5 tỷ đồng.")
    currency = [span for span in spans if span.span_type == SpanType.CURRENCY]
    assert currency
    assert currency[0].attributes["amount_vnd"] == 2500000000


def test_detect_law_article() -> None:
    spans = collect("Bị cáo bị truy tố theo Điều 174 Bộ luật Hình sự 2015.")
    law = [span for span in spans if span.span_type == SpanType.LAW]
    assert law
    assert law[0].attributes["code"].lower().startswith("bộ luật")
    assert law[0].attributes["year"] == 2015

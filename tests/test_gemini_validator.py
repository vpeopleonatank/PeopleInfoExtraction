import json
from types import SimpleNamespace

from src.validation.gemini_validator import GeminiFlashValidator
from src.validation.models import GeminiIssueType, GeminiVerdict


class StubGeminiClient:
    class _Models:
        def __init__(self, parent: "StubGeminiClient") -> None:
            self.parent = parent

        def generate_content(self, *, model: str, contents: str, config=None):
            self.parent.messages.append(contents)
            return SimpleNamespace(text=json.dumps(self.parent.payload, ensure_ascii=False))

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.messages: list[str] = []
        self.models = self._Models(self)


class BrokenGeminiClient:
    class _Models:
        def generate_content(self, *, model: str, contents: str, config=None):
            return SimpleNamespace(text="not json")

    def __init__(self) -> None:
        self.models = self._Models()


def _base_kwargs() -> dict:
    return {
        "doc_id": "doc",
        "passage_id": 0,
        "passage_text": "Ông A bị bắt ngày 1/1.",
        "extracted_people": [{"name": {"text": "Ông A"}}],
        "raw_response_text": "{}",
        "original_model": "qwen3:4b",
        "article_title": "Title",
        "published_at": "2024-01-01T00:00:00",
    }


def test_validate_extraction_with_stub_response() -> None:
    payload = {
        "verdict": "supported",
        "issues": [
            {
                "type": "missing_info",
                "person_name": "Ông A",
                "field": "actions[0].law_article",
                "description": "Law article missing.",
                "evidence": "Không thấy trong đoạn.",
            }
        ],
        "missing_people": [],
    }
    client = StubGeminiClient(payload)
    validator = GeminiFlashValidator(client=client, api_key="stub", model_name="gemini-test")

    report = validator.validate_extraction(**_base_kwargs())

    assert report.verdict == GeminiVerdict.SUPPORTED
    assert report.summary.issue_count == 1
    assert client.messages is not None
    assert '"passage_text"' in client.messages[0]


def test_invalid_json_response_creates_parsing_issue() -> None:
    validator = GeminiFlashValidator(client=BrokenGeminiClient(), api_key="stub", model_name="gemini-test")
    report = validator.validate_extraction(**_base_kwargs())

    assert len(report.issues) == 1
    assert report.issues[0].type == GeminiIssueType.PARSING_ERROR
    assert report.verdict == GeminiVerdict.UNSURE


def test_passage_truncation_flag() -> None:
    payload = {"verdict": "unsure", "issues": [], "missing_people": []}
    client = StubGeminiClient(payload)
    validator = GeminiFlashValidator(
        client=client,
        api_key="stub",
        model_name="gemini-test",
        max_passage_chars=10,
    )
    kwargs = _base_kwargs()
    kwargs["passage_text"] = "A" * 50
    report = validator.validate_extraction(**kwargs)

    assert report.prompt_truncated is True
    assert report.prompt_char_count == 10

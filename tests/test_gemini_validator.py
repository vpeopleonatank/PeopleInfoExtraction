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


class StubOpenRouterClient:
    class _Completions:
        def __init__(self, parent: "StubOpenRouterClient") -> None:
            self.parent = parent

        def create(self, *, model: str, messages: list[dict], **_kwargs):
            self.parent.calls.append({"model": model, "messages": messages})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.parent.payload))])

    class _Chat:
        def __init__(self, parent: "StubOpenRouterClient") -> None:
            self.completions = StubOpenRouterClient._Completions(parent)

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict] = []
        self.chat = self._Chat(self)


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


def test_openrouter_transport_uses_chat_api() -> None:
    payload = json.dumps({"verdict": "unsupported", "issues": [], "missing_people": []}, ensure_ascii=False)
    client = StubOpenRouterClient(payload)
    validator = GeminiFlashValidator(
        client=client,
        model_name="google/gemini-2.5-flash",
        transport="openrouter",
        openrouter_api_key="stub",
    )

    report = validator.validate_extraction(**_base_kwargs())

    assert report.verdict == GeminiVerdict.UNSUPPORTED
    assert client.calls, "Expected OpenRouter stub to capture at least one call."
    assert client.calls[0]["messages"][0]["role"] == "system"

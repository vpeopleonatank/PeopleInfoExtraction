"""Gemini 2.5 Flash validator for workflow_b outputs."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - dependency may be missing in limited envs
    genai = None
    genai_types = None

try:
    from openai import OpenAI as OpenAIClient
except ImportError:  # pragma: no cover - optional dependency
    OpenAIClient = None

from .models import (
    GeminiIssueType,
    GeminiMissingPerson,
    GeminiReportSummary,
    GeminiValidationIssue,
    GeminiValidationReport,
    GeminiVerdict,
)


class GeminiValidatorError(RuntimeError):
    """Raised when something goes wrong while talking to Gemini."""


class GeminiFlashValidator:
    """High-level helper that calls Gemini 2.5 Flash to cross-check extractions."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert fact-checker. Cross-check the provided PEOPLE extraction against the passage.\n"
        "- Only rely on the passage text; flag anything fabricated or unsupported.\n"
        "- Identify missing people mentioned in the passage but absent from the extraction.\n"
        "- Highlight conflicts (different ages, roles, law articles, etc.).\n"
        "- Respond with STRICT JSON using this schema:\n"
        "{\n"
        '  "verdict": "supported|unsure|unsupported",\n'
        '  "issues": [\n'
        "    {\"type\": \"hallucination|missing_info|conflict|other\", \"person_name\": str|null,\n"
        '     "field": str|null, "description": str, "evidence": str|null}\n'
        "  ],\n"
        '  "missing_people": [{"name": str, "snippet": str|null}]\n'
        "}\n"
        "- If unsure, choose verdict \"unsure\" and explain why."
    )

    def __init__(
        self,
        *,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 32468,
        max_passage_chars: int = 12000,
        api_key: Optional[str] = None,
        transport: str = "google",
        openrouter_api_key: Optional[str] = None,
        openrouter_site_url: Optional[str] = None,
        openrouter_site_name: Optional[str] = None,
        openrouter_base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        client: Optional[Any] = None,
        dry_run: bool = False,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_passage_chars = max_passage_chars
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.dry_run = dry_run

        allowed_transports = {"google", "openrouter"}
        if transport not in allowed_transports:
            raise ValueError(f"Unsupported transport '{transport}'. Expected one of {sorted(allowed_transports)}.")
        self.transport = transport
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.openrouter_site_url = openrouter_site_url or os.getenv("OPENROUTER_SITE_URL")
        self.openrouter_site_name = openrouter_site_name or os.getenv("OPENROUTER_SITE_NAME")
        self.openrouter_base_url = (
            openrouter_base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        )

        self._client = client or self._build_client()
        self._last_prompt: Optional[str] = None

    @property
    def last_prompt(self) -> Optional[str]:
        """Expose the most recent prompt, useful for debugging/dry-run output."""
        return self._last_prompt

    def _build_client(self) -> Any:
        if self.dry_run:
            return None
        if self.transport == "openrouter":
            return self._build_openrouter_client()
        return self._build_google_client()

    def _build_google_client(self) -> Any:
        if genai is None:
            raise GeminiValidatorError(
                "python-genai dependency is missing. Install dependencies or run in dry-run mode."
            )
        client_kwargs: dict[str, Any] = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        try:
            return genai.Client(**client_kwargs)
        except Exception as exc:  # pragma: no cover - network/environment errors
            raise GeminiValidatorError(f"Failed to initialize Gemini client: {exc}") from exc

    def _build_openrouter_client(self) -> Any:
        if OpenAIClient is None:  # pragma: no cover - optional dependency
            raise GeminiValidatorError("openai package is missing. Install openai>=1.0.0 to use OpenRouter transport.")
        api_key = self.openrouter_api_key
        if not api_key:
            raise GeminiValidatorError("OpenRouter API key is required. Pass via argument or OPENROUTER_API_KEY env var.")
        try:
            return OpenAIClient(base_url=self.openrouter_base_url, api_key=api_key)
        except Exception as exc:  # pragma: no cover - network/environment errors
            raise GeminiValidatorError(f"Failed to initialize OpenRouter client: {exc}") from exc

    def _openrouter_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.openrouter_site_url:
            headers["HTTP-Referer"] = self.openrouter_site_url
        if self.openrouter_site_name:
            headers["X-Title"] = self.openrouter_site_name
        return headers

    def _truncate_passage(self, text: str) -> tuple[str, bool]:
        if not text:
            return "", False
        if len(text) <= self.max_passage_chars:
            return text, False
        return text[: self.max_passage_chars], True

    def _build_user_prompt(self, payload: dict[str, Any], truncated: bool) -> str:
        header = (
            "Review the extraction below. Determine whether each person/action is supported solely by the passage text.\n"
        )
        if truncated:
            header += "NOTE: Passage text was truncated because it exceeded the maximum length.\n"
        prompt = f"{header}\nPAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        self._last_prompt = prompt
        return prompt

    def _strip_code_fences(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return stripped

    def _extract_json_block(self, text: str) -> Optional[str]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return match.group(0)
        return None

    def _parse_model_response(self, response_text: str) -> dict[str, Any]:
        attempt_order = [
            response_text,
            self._strip_code_fences(response_text),
            self._extract_json_block(response_text) or "",
        ]
        for candidate in attempt_order:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise GeminiValidatorError("Gemini response was not valid JSON.")

    def _call_model(self, user_prompt: str) -> str:
        if self.dry_run or self._client is None:
            mock = {
                "verdict": "unsure",
                "issues": [
                    {
                        "type": "other",
                        "description": "Dry-run mode is enabled; no live Gemini call was made.",
                    }
                ],
                "missing_people": [],
            }
            return json.dumps(mock, ensure_ascii=False)

        if self.transport == "openrouter":
            return self._call_openrouter(user_prompt)
        return self._call_google(user_prompt)

    def _call_google(self, user_prompt: str) -> str:
        config = None
        if genai_types is not None:
            config = genai_types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
                system_instruction=self.system_prompt,
            )
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            )
        except Exception as exc:  # pragma: no cover - propagate API issues
            raise GeminiValidatorError(f"Gemini request failed: {exc}") from exc
        text = getattr(response, "text", None)
        if text:
            return text
        # Fallback when SDK returns candidates structure
        candidates = getattr(response, "candidates", None)
        if candidates:
            contents = candidates[0].get("content") if isinstance(candidates[0], dict) else getattr(candidates[0], "content", None)
            if contents:
                parts = getattr(contents, "parts", None)
                if parts:
                    return "".join(
                        part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "") for part in parts
                    )
        raise GeminiValidatorError("Gemini response did not contain text.")

    def _call_openrouter(self, user_prompt: str) -> str:
        chat = getattr(self._client, "chat", None)
        if chat is None:
            raise GeminiValidatorError("OpenRouter client is missing chat attribute.")
        completions = getattr(chat, "completions", None)
        if completions is None:
            raise GeminiValidatorError("OpenRouter client is missing chat.completions.")
        headers = self._openrouter_headers()
        kwargs: dict[str, Any] = {}
        if headers:
            kwargs["extra_headers"] = headers
        try:
            response = completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - propagate API issues
            raise GeminiValidatorError(f"OpenRouter request failed: {exc}") from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise GeminiValidatorError("OpenRouter response did not include choices.")
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None and isinstance(first_choice, dict):
            message = first_choice.get("message")
        if message is None:
            raise GeminiValidatorError("OpenRouter response missing message content.")

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            ).strip()
        elif isinstance(content, str):
            text = content
        else:
            text = str(content or "")
        if not text:
            raise GeminiValidatorError("OpenRouter response content was empty.")
        return text

    def _coerce_issue_type(self, value: Optional[str]) -> GeminiIssueType:
        if not value:
            return GeminiIssueType.OTHER
        normalized = value.lower()
        for option in GeminiIssueType:
            if normalized == option.value:
                return option
        if normalized == "missing_info":
            return GeminiIssueType.MISSING_INFO
        if normalized == "hallucination":
            return GeminiIssueType.HALLUCINATION
        if normalized == "conflict":
            return GeminiIssueType.CONFLICT
        return GeminiIssueType.OTHER

    def _coerce_verdict(self, value: Optional[str]) -> GeminiVerdict:
        if not value:
            return GeminiVerdict.UNSURE
        normalized = value.lower()
        for verdict in GeminiVerdict:
            if normalized == verdict.value:
                return verdict
        return GeminiVerdict.UNSURE

    def _build_report_from_payload(
        self,
        *,
        payload: dict[str, Any],
        doc_id: str,
        passage_id: int,
        article_title: Optional[str],
        published_at: Optional[str],
        original_model: Optional[str],
        source_file: Optional[str],
        prompt_truncated: bool,
        prompt_char_count: int,
        raw_response_text: str,
    ) -> GeminiValidationReport:
        verdict = self._coerce_verdict(payload.get("verdict"))

        issues_data = payload.get("issues") or []
        issues: list[GeminiValidationIssue] = []
        if isinstance(issues_data, list):
            for issue in issues_data:
                if not isinstance(issue, dict):
                    continue
                issues.append(
                    GeminiValidationIssue(
                        type=self._coerce_issue_type(issue.get("type")),
                        person_name=issue.get("person_name"),
                        field=issue.get("field"),
                        description=issue.get("description") or "",
                        evidence=issue.get("evidence"),
                    )
                )

        missing_people_data = payload.get("missing_people") or []
        missing_people: list[GeminiMissingPerson] = []
        if isinstance(missing_people_data, list):
            for person in missing_people_data:
                if not isinstance(person, dict):
                    continue
                name = person.get("name")
                if not name:
                    continue
                missing_people.append(
                    GeminiMissingPerson(
                        name=name,
                        snippet=person.get("snippet"),
                    )
                )

        summary = GeminiReportSummary(
            verdict=verdict,
            issue_count=len(issues),
            missing_people_count=len(missing_people),
        )

        return GeminiValidationReport(
            doc_id=doc_id,
            passage_id=passage_id,
            article_title=article_title,
            published_at=published_at,
            validator_model=self.model_name,
            original_model=original_model,
            verdict=verdict,
            issues=issues,
            missing_people=missing_people,
            summary=summary,
            source_file=source_file,
            prompt_truncated=prompt_truncated,
            prompt_char_count=prompt_char_count,
            raw_response_text=raw_response_text,
        )

    def validate_extraction(
        self,
        *,
        doc_id: str,
        passage_id: int,
        passage_text: str,
        extracted_people: list[dict[str, Any]],
        raw_response_text: Optional[str],
        original_model: Optional[str],
        article_title: Optional[str],
        published_at: Optional[str],
        source_file: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GeminiValidationReport:
        """
        Validate a single extraction result by calling Gemini (or returning a mock in dry-run).
        """
        passage_snippet, truncated = self._truncate_passage(passage_text or "")
        payload = {
            "doc_id": doc_id,
            "passage_id": passage_id,
            "article_title": article_title,
            "published_at": published_at,
            "passage_text": passage_snippet,
            "extracted_people": extracted_people,
            "raw_response": raw_response_text,
            "metadata": metadata or {},
        }
        user_prompt = self._build_user_prompt(payload, truncated)

        try:
            model_response = self._call_model(user_prompt)
            parsed = self._parse_model_response(model_response)
            return self._build_report_from_payload(
                payload=parsed,
                doc_id=doc_id,
                passage_id=passage_id,
                article_title=article_title,
                published_at=published_at,
                original_model=original_model,
                source_file=source_file,
                prompt_truncated=truncated,
                prompt_char_count=len(passage_snippet),
                raw_response_text=model_response,
            )
        except GeminiValidatorError as exc:
            issue = GeminiValidationIssue(
                type=GeminiIssueType.PARSING_ERROR,
                description=str(exc),
            )
            summary = GeminiReportSummary(verdict=GeminiVerdict.UNSURE, issue_count=1, missing_people_count=0)
            return GeminiValidationReport(
                doc_id=doc_id,
                passage_id=passage_id,
                article_title=article_title,
                published_at=published_at,
                validator_model=self.model_name,
                original_model=original_model,
                verdict=GeminiVerdict.UNSURE,
                issues=[issue],
                missing_people=[],
                summary=summary,
                source_file=source_file,
                prompt_truncated=truncated,
                prompt_char_count=len(passage_snippet),
                raw_response_text=None,
            )

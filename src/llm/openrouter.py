"""Thin client for interacting with OpenRouter's OpenAI-compatible API."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI, OpenAIError

from .types import LLMGeneration


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, payload: Optional[dict] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


class OpenRouterClient:
    """Client for the OpenRouter /chat/completions endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        timeout: int = 120,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter requires an API key.")
        self.model = model
        self.temperature = temperature
        self.extra_headers = extra_headers or {}
        self.extra_body = extra_body or {}
        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout)

    def _create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        format: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ):
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if format:
            if format == "json":
                payload["response_format"] = {"type": "json_object"}
            else:
                payload["response_format"] = {"type": format}
        if extra_options:
            payload.update(extra_options)
        try:
            return self.client.chat.completions.create(
                **payload,
                extra_headers=self.extra_headers or None,
                extra_body=self.extra_body or None,
            )
        except OpenAIError as exc:  # pragma: no cover - passthrough for unexpected runtime failures
            status_code = getattr(exc, "status_code", None)
            error_payload: dict = {}
            try:
                if getattr(exc, "response", None) is not None and exc.response.text:
                    error_payload = json.loads(exc.response.text)
            except Exception:
                error_payload = {"text": getattr(exc, "response", None) and exc.response.text}
            raise OpenRouterError(str(exc), status_code=status_code, payload=error_payload) from exc
        except Exception as exc:  # pragma: no cover - defensive catch for unexpected parse errors
            # openai client can raise JSONDecodeError when upstream responds with non-JSON
            response_text = getattr(exc, "response", None) and getattr(exc.response, "text", None)
            payload = {"text": response_text or str(exc)}
            message = str(exc)
            if response_text:
                message = f"{message} | response: {response_text[:200]}"
            raise OpenRouterError(message, payload=payload) from exc

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        format: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> LLMGeneration:
        completion = self._create_chat_completion(messages, format=format, extra_options=extra_options)
        choices = completion.choices or []
        if not choices:
            raise OpenRouterError("OpenRouter response did not contain any choices")
        message = choices[0].message or {}
        usage = getattr(completion, "usage", None) or {}
        return LLMGeneration(
            content=getattr(message, "content", "") or "",
            model=getattr(completion, "model", "") or self.model,
            created_at=getattr(completion, "created", None),
            total_duration=None,
            prompt_eval_count=getattr(usage, "prompt_tokens", None),
            eval_count=getattr(usage, "completion_tokens", None),
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        format: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> LLMGeneration:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(messages, format=format, extra_options=extra_options)

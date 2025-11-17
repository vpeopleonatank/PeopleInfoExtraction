"""Thin client for interacting with a vLLM OpenAI-compatible server."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from .types import LLMGeneration


class VLLMError(RuntimeError):
    """Raised when the vLLM server returns an error."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, payload: Optional[dict] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


class VLLMClient:
    """Client for vLLM's OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.0,
        timeout: int = 120,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.api_key = api_key

    def _post(self, path: str, payload: Dict[str, Any]) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if response.status_code != 200:
            error_payload: dict
            try:
                error_payload = response.json()
            except json.JSONDecodeError:
                error_payload = {"text": response.text}
            raise VLLMError(
                f"vLLM request failed ({response.status_code})",
                status_code=response.status_code,
                payload=error_payload,
            )
        try:
            return response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - passthrough for unexpected server replies
            raise VLLMError("Invalid JSON returned by vLLM") from exc

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        format: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> LLMGeneration:
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
        data = self._post("/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise VLLMError("vLLM response did not contain any choices", payload=data)
        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        return LLMGeneration(
            content=message.get("content", ""),
            model=data.get("model", self.model),
            created_at=data.get("created"),
            total_duration=None,
            prompt_eval_count=usage.get("prompt_tokens"),
            eval_count=usage.get("completion_tokens"),
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

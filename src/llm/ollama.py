"""Thin client for interacting with a local Ollama server."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class OllamaError(RuntimeError):
    """Raised when the Ollama server returns an error."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, payload: Optional[dict] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


@dataclass(slots=True)
class OllamaGeneration:
    """Container for a non-streamed Ollama response."""

    content: str
    model: str
    created_at: Optional[str]
    total_duration: Optional[int]
    prompt_eval_count: Optional[int]
    eval_count: Optional[int]


class OllamaClient:
    """Convenience wrapper around the Ollama HTTP API."""

    def __init__(
        self,
        *,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> dict:
        url = f"{self.base_url}{path}"
        response = requests.post(url, json=payload, timeout=self.timeout)
        if response.status_code != 200:
            raise OllamaError(
                f"Ollama request failed ({response.status_code})",
                status_code=response.status_code,
                payload={"text": response.text},
            )
        try:
            return response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - passthrough for unexpected server replies
            raise OllamaError("Invalid JSON returned by Ollama") from exc

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        format: Optional[str] = None,
        keep_alive: Optional[int] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> OllamaGeneration:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if format:
            payload["format"] = format
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if extra_options:
            payload["options"].update(extra_options)
        data = self._post("/api/chat", payload)
        message = data.get("message") or {}
        return OllamaGeneration(
            content=message.get("content", ""),
            model=data.get("model", self.model),
            created_at=data.get("created_at"),
            total_duration=data.get("total_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        format: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> OllamaGeneration:
        """Helper to run a single system+user chat turn."""

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.chat(messages, format=format, extra_options=extra_options)

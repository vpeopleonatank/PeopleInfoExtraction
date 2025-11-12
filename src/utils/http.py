"""HTTP helper utilities with simple retry/backoff handling."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional

import requests

from src import config

LOGGER = logging.getLogger(__name__)


@dataclass
class HttpFetcher:
    """Simple HTTP client with retry + default headers."""

    max_retries: int = config.MAX_RETRIES
    backoff_seconds: float = config.RETRY_BACKOFF
    timeout_seconds: int = config.REQUEST_TIMEOUT
    default_headers: MutableMapping[str, str] = field(
        default_factory=lambda: dict(config.DEFAULT_HEADERS)
    )

    def _merge_headers(self, headers: Optional[Mapping[str, str]]) -> MutableMapping[str, str]:
        merged: MutableMapping[str, str] = dict(self.default_headers)
        if headers:
            merged.update(headers)
        return merged

    def get(self, url: str, *, headers: Optional[Mapping[str, str]] = None, **kwargs: Any) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                LOGGER.debug("Fetching %s (attempt %s/%s)", url, attempt, self.max_retries)
                response = requests.get(
                    url,
                    headers=self._merge_headers(headers),
                    timeout=self.timeout_seconds,
                    **kwargs,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:  # pragma: no cover - network errors hard to deterministically test
                last_error = exc
                LOGGER.warning("Request to %s failed: %s", url, exc)
                if attempt == self.max_retries:
                    break
                time.sleep(self.backoff_seconds * attempt)
        assert last_error is not None
        raise last_error


FETCHER = HttpFetcher()


def fetch_text(url: str, *, headers: Optional[Mapping[str, str]] = None, **kwargs: Any) -> str:
    """Return response text for *url* using the shared fetcher."""

    return FETCHER.get(url, headers=headers, **kwargs).text

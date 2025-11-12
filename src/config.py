"""Project-wide configuration and constants."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"
CLEAN_TEXT_DIR = DATA_DIR / "clean_text"
CACHE_DIR = DATA_DIR / "cache"

TIMELINE_BASE = "https://thanhnien.vn/timelinelist/1855/{page}.htm"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


@dataclass(slots=True, frozen=True)
class TimelineConfig:
    """Parameters used for paginating the Pháp Luật timeline."""

    max_pages: int = 5
    per_page_expected: int = 20


@dataclass(slots=True, frozen=True)
class ExtractionConfig:
    """Parameters controlling passage segmentation and extraction."""

    min_passage_tokens: int = 512
    max_passage_tokens: int = 1000


def ensure_data_dirs() -> None:
    """Ensure local data directories exist."""

    for path in (DATA_DIR, RAW_HTML_DIR, CLEAN_TEXT_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)

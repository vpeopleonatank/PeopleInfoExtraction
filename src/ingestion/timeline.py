"""Scraper for Thanh Niên Pháp Luật timeline pages."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src import config
from src.utils.http import fetch_text

LOGGER = logging.getLogger(__name__)
THANH_NIEN_BASE = "https://thanhnien.vn"


@dataclass(slots=True)
class TimelineArticle:
    """Represents one article entry from the timeline."""

    title: str
    url: str
    canonical_url: str
    category_name: str
    category_url: str
    image_url: Optional[str]
    data_id: Optional[str]
    page: int
    position: int

    def to_dict(self) -> dict:
        return asdict(self)

    def is_category_match(self, keyword: Optional[str]) -> bool:
        """Return True if *keyword* is None or matches category slug/text."""

        if keyword is None:
            return True
        key = keyword.lower()
        return key in self.category_url.lower() or key in self.category_name.lower()


class TimelineScraper:
    """Fetches and parses Thanh Niên timeline pages."""

    def __init__(self, base_template: str = config.TIMELINE_BASE, *, category_keyword: Optional[str] = "phap-luat") -> None:
        self.base_template = base_template
        self.category_keyword = category_keyword

    def build_page_url(self, page: int) -> str:
        return self.base_template.format(page=page)

    def fetch_page(self, page: int) -> List[TimelineArticle]:
        html = fetch_text(self.build_page_url(page))
        return list(self.parse_page(html, page=page))

    def parse_page(self, html: str, *, page: int) -> Iterable[TimelineArticle]:
        soup = BeautifulSoup(html, "lxml")
        for position, item in enumerate(soup.select("div.box-category-item"), start=1):
            title_link = item.select_one("a.box-category-link-title")
            if not title_link:
                continue
            rel_url = title_link.get("href")
            if not rel_url:
                continue
            absolute_url = urljoin(THANH_NIEN_BASE, rel_url)
            canonical_rel = title_link.get("data-io-canonical-url") or rel_url
            canonical_url = urljoin(THANH_NIEN_BASE, canonical_rel)

            category_link = item.select_one("a.box-category-category")
            if not category_link:
                category_name = ""
                category_href = ""
            else:
                category_name = category_link.get_text(strip=True)
                category_href = category_link.get("href", "")

            image_tag = item.select_one("a.box-category-link-with-avatar img")
            image_url = image_tag.get("src") if image_tag else None

            article = TimelineArticle(
                title=title_link.get("title") or title_link.get_text(strip=True),
                url=absolute_url,
                canonical_url=canonical_url,
                category_name=category_name,
                category_url=urljoin(THANH_NIEN_BASE, category_href) if category_href else "",
                image_url=image_url,
                data_id=item.get("data-id"),
                page=page,
                position=position,
            )
            if article.is_category_match(self.category_keyword):
                yield article

    def collect(self, *, limit: int = 20, max_pages: int = 10) -> List[TimelineArticle]:
        results: List[TimelineArticle] = []
        seen: set[str] = set()
        for page in range(1, max_pages + 1):
            LOGGER.info("Fetching timeline page %s", page)
            for article in self.fetch_page(page):
                key = article.canonical_url or article.url
                if key in seen:
                    continue
                seen.add(key)
                results.append(article)
                if len(results) >= limit:
                    return results
        return results


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Thanh Niên Pháp Luật article links")
    parser.add_argument("--limit", type=int, default=20, help="Number of articles to collect")
    parser.add_argument("--max-pages", type=int, default=5, help="Maximum number of timeline pages to crawl")
    parser.add_argument("--category", type=str, default="phap-luat", help="Keyword to match in category URL/name")
    parser.add_argument("--output", type=str, default=str(config.CACHE_DIR / "phap_luat_articles.json"), help="Optional JSON output path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.ensure_data_dirs()
    scraper = TimelineScraper(category_keyword=args.category)
    articles = scraper.collect(limit=args.limit, max_pages=args.max_pages)
    LOGGER.info("Collected %s articles (requested %s)", len(articles), args.limit)

    data = [article.to_dict() for article in articles]
    output_path = (
        config.BASE_DIR / args.output if not args.output.startswith("/") else Path(args.output)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2 if args.pretty else None)
    LOGGER.info("Wrote %s articles to %s", len(data), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

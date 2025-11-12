"""Utilities for downloading and parsing Thanh Niên articles."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Iterable, List, Optional

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from src import config
from src.utils.http import fetch_text

LOGGER = logging.getLogger(__name__)
DOC_ID_RE = re.compile(r"-(?P<doc_id>\d+)\.htm")
WHITESPACE_RE = re.compile(r"\s+")


def extract_doc_id(url: str) -> str:
    """Return numeric doc_id inferred from an article URL."""

    match = DOC_ID_RE.search(url)
    if match:
        return match.group("doc_id")
    # Fallback to slug-ish format
    stripped = url.rstrip("/")
    return stripped.split("/")[-1].replace(".htm", "")


def clean_text(value: str) -> str:
    """Normalize whitespace and HTML entities."""

    text = unescape(value or "")
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def count_tokens(value: str) -> int:
    return len([tok for tok in value.split(" ") if tok])


@dataclass(slots=True)
class Passage:
    doc_id: str
    passage_id: int
    text: str
    start: int
    end: int
    word_count: int
    paragraph_indices: List[int]

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "passage_id": self.passage_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "word_count": self.word_count,
            "paragraph_indices": self.paragraph_indices,
        }


@dataclass(slots=True)
class ArticleDocument:
    doc_id: str
    url: str
    canonical_url: str
    title: str
    summary: Optional[str]
    author: Optional[str]
    category: Optional[str]
    published_at: Optional[datetime]
    paragraphs: List[str]
    passages: List[Passage] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "summary": self.summary,
            "author": self.author,
            "category": self.category,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "paragraphs": self.paragraphs,
            "passages": [p.to_dict() for p in self.passages],
        }

    def full_text(self) -> str:
        return "\n\n".join(self.paragraphs)

    def build_passages(
        self,
        *,
        min_tokens: int = config.ExtractionConfig().min_passage_tokens,
        max_tokens: int = config.ExtractionConfig().max_passage_tokens,
    ) -> List[Passage]:
        if not self.paragraphs:
            self.passages = []
            return self.passages

        paragraph_tokens = [count_tokens(p) for p in self.paragraphs]
        paragraph_spans: List[tuple[int, int]] = []
        cursor = 0
        for idx, paragraph in enumerate(self.paragraphs):
            if idx > 0:
                cursor += 2  # account for double newline between paragraphs
            start = cursor
            cursor += len(paragraph)
            paragraph_spans.append((start, cursor))

        segments: List[tuple[int, int]] = []
        start_idx = 0
        current_tokens = 0
        for idx, tokens in enumerate(paragraph_tokens):
            if idx > start_idx and current_tokens + tokens > max_tokens:
                segments.append((start_idx, idx - 1))
                start_idx = idx
                current_tokens = 0
            current_tokens += tokens
        if start_idx < len(self.paragraphs):
            segments.append((start_idx, len(self.paragraphs) - 1))

        def segment_tokens(segment: tuple[int, int]) -> int:
            s, e = segment
            return sum(paragraph_tokens[s : e + 1])

        merged: List[list[int]] = []
        for seg in segments:
            if merged:
                prev_start, prev_end = merged[-1]
                if segment_tokens((prev_start, prev_end)) < min_tokens:
                    merged[-1][1] = seg[1]
                    continue
            merged.append([seg[0], seg[1]])

        if len(merged) >= 2:
            last_start, last_end = merged[-1]
            if segment_tokens((last_start, last_end)) < min_tokens:
                merged[-2][1] = last_end
                merged.pop()

        passages: List[Passage] = []
        for passage_id, (seg_start, seg_end) in enumerate(merged):
            text = "\n\n".join(self.paragraphs[seg_start : seg_end + 1])
            start_char = paragraph_spans[seg_start][0]
            end_char = paragraph_spans[seg_end][1]
            token_count = segment_tokens((seg_start, seg_end))
            passages.append(
                Passage(
                    doc_id=self.doc_id,
                    passage_id=passage_id,
                    text=text,
                    start=start_char,
                    end=end_char,
                    word_count=token_count,
                    paragraph_indices=list(range(seg_start, seg_end + 1)),
                )
            )

        self.passages = passages
        return self.passages


def parse_article_html(html: str, url: str, *, doc_id: Optional[str] = None) -> ArticleDocument:
    soup = BeautifulSoup(html, "lxml")
    title_node = soup.select_one("h1.detail-title [data-role='title']")
    sapo_node = soup.select_one(".detail-sapo")
    body_node = soup.select_one(".detail-content[data-role='content']")
    canonical_link = soup.find("link", rel="canonical")

    if not title_node or not body_node:
        raise ValueError("Unable to locate title or body in article HTML")

    meta_author = soup.find("meta", attrs={"name": "article:author"})
    meta_section = soup.find("meta", attrs={"property": "article:section"})
    meta_publish = soup.find("meta", attrs={"property": "article:published_time"})

    paragraphs = list(iter_body_paragraphs(body_node))
    published_at = None
    if meta_publish and meta_publish.get("content"):
        try:
            published_at = datetime.fromisoformat(meta_publish["content"].replace("Z", "+00:00"))
        except ValueError:
            LOGGER.warning("Unable to parse publish time: %s", meta_publish["content"])

    article = ArticleDocument(
        doc_id=doc_id or extract_doc_id(url),
        url=url,
        canonical_url=canonical_link["href"] if canonical_link and canonical_link.get("href") else url,
        title=clean_text(title_node.get_text(" ", strip=True)),
        summary=clean_text(sapo_node.get_text(" ", strip=True)) if sapo_node else None,
        author=clean_text(meta_author["content"]) if meta_author and meta_author.get("content") else None,
        category=clean_text(meta_section["content"]) if meta_section and meta_section.get("content") else None,
        published_at=published_at,
        paragraphs=paragraphs,
    )
    return article


def iter_body_paragraphs(body_node: Tag) -> Iterable[str]:
    for child in body_node.children:
        if isinstance(child, NavigableString):
            continue
        if child.name in {"script", "style"}:
            continue
        text: Optional[str] = None
        if child.name == "p":
            text = child.get_text(" ", strip=True)
        elif child.name in {"ul", "ol"}:
            items = [li.get_text(" ", strip=True) for li in child.find_all("li", recursive=False)]
            for item in items:
                cleaned = clean_text(item)
                if cleaned:
                    yield cleaned
            continue
        elif child.name == "figure":
            caption = child.find("figcaption")
            if caption:
                text = caption.get_text(" ", strip=True)
                if text:
                    text = f"[Ảnh] {text}"
        elif child.name in {"h2", "h3", "h4", "blockquote"}:
            text = child.get_text(" ", strip=True)
        else:
            # fallback for generic containers
            text = child.get_text(" ", strip=True)

        cleaned = clean_text(text or "")
        if cleaned:
            yield cleaned


@dataclass(slots=True)
class ArticleFetcher:
    raw_dir: Path = config.RAW_HTML_DIR

    def fetch(self, url: str, *, doc_id: Optional[str] = None, save_html: bool = True) -> ArticleDocument:
        html = fetch_text(url)
        doc = parse_article_html(html, url, doc_id=doc_id)
        if save_html:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            (self.raw_dir / f"{doc.doc_id}.html").write_text(html, encoding="utf-8")
        return doc


def write_article_json(article: ArticleDocument, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(article.to_dict(), fh, ensure_ascii=False, indent=2)

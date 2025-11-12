"""Shared helpers for iterating article passages."""
from __future__ import annotations

from typing import List, Tuple


def passage_records(article: dict) -> List[dict]:
    """Return passage dictionaries for an article, falling back to paragraphs."""

    passages = article.get("passages") or []
    if passages:
        return passages
    paragraphs = article.get("paragraphs") or []
    cursor = 0
    fallback: List[dict] = []
    for idx, paragraph in enumerate(paragraphs):
        paragraph = (paragraph or "").strip()
        if not paragraph:
            continue
        start = cursor
        cursor += len(paragraph) + 2  # assume blank line between paragraphs
        fallback.append(
            {
                "passage_id": idx,
                "text": paragraph,
                "start": start,
                "end": start + len(paragraph),
                "word_count": len(paragraph.split()),
                "paragraph_indices": [idx],
            }
        )
    return fallback


__all__: Tuple[str, ...] = ("passage_records",)

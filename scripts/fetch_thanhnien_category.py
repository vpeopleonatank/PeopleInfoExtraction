#!/usr/bin/env python3
"""
Fetch Thanhnien articles for a specific ingest category and dump their content.

Example:
  python scripts/fetch_thanhnien_articles.py --limit 5 --category phap-luat \\
      --db-url postgresql://crawl_user:crawl_password@localhost:5433/crawl_db
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import create_engine, func, or_, select
from sqlalchemy.orm import Session

# Ensure the repository root is importable when executing from the scripts/ directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import Article  # noqa: E402


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query the articles table for Thanhnien posts in a specific ingest "
            "category and emit their content as JSON."
        )
    )
    parser.add_argument(
        "--db-url",
        help=(
            "SQLAlchemy database URL. Defaults to DATABASE_URL or "
            "CRAWLER_DATABASE_URL from the environment."
        ),
    )
    parser.add_argument(
        "--category",
        default="phap-luat",
        help="Case-insensitive ingest/category slug to filter by. Defaults to 'phap-luat'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of articles to return. Defaults to 10.",
    )
    parser.add_argument(
        "--site-slug",
        default="thanhnien",
        help="Site slug to filter on. Defaults to 'thanhnien'.",
    )
    parser.add_argument(
        "--order",
        choices=("newest", "oldest"),
        default="newest",
        help="Sort by publish_date (falling back to created_at). Defaults to newest first.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional output path. When omitted, JSON is written to stdout. "
            "If provided, directories are created as needed."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "json"),
        default="jsonl",
        help="Output format. 'jsonl' writes one JSON object per line. Defaults to jsonl.",
    )
    return parser.parse_args(argv)


def resolve_db_url(cli_url: str | None) -> str:
    for candidate in (cli_url, os.getenv("DATABASE_URL"), os.getenv("CRAWLER_DATABASE_URL")):
        if candidate:
            return candidate
    raise SystemExit(
        "Database URL not provided. Supply --db-url or configure "
        "DATABASE_URL/CRAWLER_DATABASE_URL in the environment."
    )


def normalise_category(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    return normalized or None


def article_to_dict(article: Article) -> dict[str, Any]:
    def isoformat(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    return {
        "id": str(article.id),
        "site_slug": article.site_slug,
        "title": article.title,
        "url": article.url,
        "category_id": article.category_id,
        "category_name": article.category_name,
        "ingest_category_slug": article.ingest_category_slug,
        "publish_date": isoformat(article.publish_date),
        "created_at": isoformat(article.created_at),
        "updated_at": isoformat(article.updated_at),
        "tags": article.tags.split(",") if article.tags else [],
        "content": article.content,
        "description": article.description,
    }


def fetch_articles(session: Session, args: argparse.Namespace) -> list[Article]:
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")

    category = normalise_category(args.category)
    query = select(Article).where(Article.site_slug == args.site_slug)

    if category:
        query = query.where(
            or_(
                func.lower(Article.ingest_category_slug) == category,
                func.lower(Article.category_id) == category,
                func.lower(Article.category_name) == category,
            )
        )

    if args.order == "newest":
        query = query.order_by(
            Article.publish_date.desc().nullslast(),
            Article.created_at.desc().nullslast(),
        )
    else:
        query = query.order_by(
            Article.publish_date.asc().nullsfirst(),
            Article.created_at.asc().nullsfirst(),
        )

    if args.limit:
        query = query.limit(args.limit)

    return list(session.scalars(query))


def write_output(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    output_data: str
    if args.format == "json":
        output_data = json.dumps(records, ensure_ascii=False, indent=2)
    else:
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        output_data = "\n".join(lines)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_data, encoding="utf-8")
    else:
        sys.stdout.write(output_data + ("\n" if not output_data.endswith("\n") else ""))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    db_url = resolve_db_url(args.db_url)

    engine = create_engine(db_url)
    with Session(engine) as session:
        articles = fetch_articles(session, args)

    records = [article_to_dict(article) for article in articles]
    write_output(records, args)

    guidance = (
        f"Fetched {len(records)} article(s) for site '{args.site_slug}' "
        f"in category '{args.category}'."
    )
    print(guidance, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

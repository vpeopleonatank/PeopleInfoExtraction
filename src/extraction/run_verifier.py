"""CLI to verify span outputs against article text."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

from src import config
from src.extraction.verifier import load_article, load_spans_from_path, verify_document

LOGGER = logging.getLogger("verifier")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify cached spans and emit normalized outputs.")
    parser.add_argument(
        "--spans-dir",
        type=Path,
        default=config.CACHE_DIR / "baseline_spans",
        help="Directory containing span JSON files (output from run_baseline).",
    )
    parser.add_argument(
        "--articles-dir",
        type=Path,
        default=config.CLEAN_TEXT_DIR,
        help="Directory containing cleaned article JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.CACHE_DIR / "verified_spans",
        help="Directory to write verified span JSON files.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of documents to process.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print output JSON.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if not args.spans_dir.exists():
        LOGGER.error("Spans directory %s does not exist", args.spans_dir)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    total_verified = 0
    total_dropped = 0
    for span_path in sorted(args.spans_dir.glob("*.json")):
        if args.limit is not None and processed >= args.limit:
            break
        doc_id, spans, baseline_stats = load_spans_from_path(span_path)
        article = load_article(doc_id, articles_dir=args.articles_dir)
        verified, failures, stats = verify_document(article, spans)
        total_verified += stats["verified_spans"]
        total_dropped += stats["dropped_spans"]
        payload = {
            "doc_id": doc_id,
            "article_title": article.get("title"),
            "baseline_stats": baseline_stats,
            "verifier_stats": stats,
            "spans": [span.model_dump(mode="json") for span in verified],
            "dropped": [
                {
                    "span_type": result.span.span_type.value,
                    "start": result.span.start,
                    "end": result.span.end,
                    "reason": result.reason,
                    "text": result.span.text,
                }
                for result in failures
            ],
        }
        output_path = args.output_dir / f"{doc_id}.json"
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2 if args.pretty else None)
        LOGGER.info(
            "Doc %s verified=%s dropped=%s (input spans=%s)",
            doc_id,
            stats["verified_spans"],
            stats["dropped_spans"],
            stats["input_spans"],
        )
        processed += 1

    LOGGER.info("Verification complete for %s files (verified=%s dropped=%s)", processed, total_verified, total_dropped)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

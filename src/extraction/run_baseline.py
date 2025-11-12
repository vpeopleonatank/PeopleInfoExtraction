"""CLI to run regex + NER baselines over cleaned article passages."""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from src import config
from src.extraction import NERBackend, NerPipeline, run_regex_detectors
from src.extraction.passages import passage_records
from src.extraction.regex_detectors import DETECTORS, RegexDetector
from src.extraction.spans import Span, SpanModel, SpanSource, SpanType, validate_spans

LOGGER = logging.getLogger("baseline")
SENTENCE_PATTERN = re.compile(r"[^.!?\n]+[.!?…]*", flags=re.UNICODE)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic regex + NER spans over cleaned articles.")
    parser.add_argument(
        "--input",
        type=Path,
        default=config.CLEAN_TEXT_DIR,
        help="Directory containing cleaned article JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.CACHE_DIR / "baseline_spans",
        help="Directory to write span JSON outputs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of documents to process.")
    parser.add_argument(
        "--doc-ids",
        nargs="+",
        help="Optional specific doc_ids to process (must correspond to filenames in --input).",
    )
    parser.add_argument(
        "--ner-backend",
        choices=[backend.value for backend in NERBackend],
        default=NERBackend.SIMPLE.value,
        help="NER backend to use (default: simple deterministic fallback).",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=config.CACHE_DIR / "baseline_spans.log",
        help="File to write execution logs.",
    )
    return parser.parse_args(argv)


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)


def iter_article_paths(input_dir: Path, doc_ids: Optional[Sequence[str]] = None) -> Iterator[Path]:
    if doc_ids:
        for doc_id in doc_ids:
            candidate = input_dir / f"{doc_id}.json"
            if candidate.exists():
                yield candidate
            else:
                LOGGER.warning("Doc %s not found under %s", doc_id, input_dir)
    else:
        yield from sorted(input_dir.glob("*.json"))


def segment_sentences(text: str) -> List[Tuple[str, int]]:
    """Return (sentence, offset) pairs where offsets always align to the original text."""

    sentences: List[Tuple[str, int]] = []
    for match in SENTENCE_PATTERN.finditer(text):
        raw = match.group(0)
        sentence = raw.strip()
        if not sentence:
            continue
        leading_ws = len(raw) - len(raw.lstrip())
        sentences.append((sentence, match.start() + leading_ws))
    if not sentences:
        stripped = text.strip()
        if stripped:
            leading_ws = len(text) - len(text.lstrip())
            sentences.append((stripped, leading_ws))
    return sentences


def dedupe_spans(spans: Iterable[Span]) -> List[Span]:
    """Deduplicate spans by (type, start, end, source, text)."""

    ordered: "OrderedDict[tuple, Span]" = OrderedDict()
    for span in spans:
        key = (span.span_type, span.start, span.end, span.source, span.text)
        existing = ordered.get(key)
        if existing is None or (span.confidence or 0.0) > (existing.confidence or 0.0):
            ordered[key] = span
    return sorted(ordered.values(), key=lambda s: (s.start, s.end, s.span_type.value))


def serialize_spans(spans: Sequence[SpanModel]) -> List[dict]:
    serialized: List[dict] = []
    for model in spans:
        payload = model.model_dump()
        payload["span_type"] = model.span_type.value
        payload["source"] = model.source.value
        serialized.append(payload)
    return serialized


def process_article(
    article: dict,
    *,
    detectors: Iterable[RegexDetector] = DETECTORS,
    ner_pipeline: Optional[NerPipeline] = None,
) -> Tuple[List[dict], dict]:
    doc_id = article.get("doc_id") or "unknown"
    spans: List[Span] = []
    regex_count = 0
    ner_count = 0
    sentence_total = 0
    passages_processed = 0
    for passage in passage_records(article):
        text = passage.get("text") or ""
        if not text.strip():
            continue
        passages_processed += 1
        passage_id = passage.get("passage_id", passages_processed - 1)
        passage_start = passage.get("start", 0)
        for sentence_idx, (sentence, rel_offset) in enumerate(segment_sentences(text)):
            sentence_id = f"{passage_id}.{sentence_idx}"
            sentence_total += 1
            base_offset = passage_start + rel_offset
            regex_spans = run_regex_detectors(
                sentence,
                doc_id=doc_id,
                sentence_id=sentence_id,
                base_offset=base_offset,
                passage_id=passage_id,
                detectors=detectors,
            )
            spans.extend(regex_spans)
            regex_count += len(regex_spans)
            ner_results: List[Span] = []
            if ner_pipeline is not None:
                ner_results = ner_pipeline.extract(
                    sentence,
                    doc_id=doc_id,
                    sentence_id=sentence_id,
                    base_offset=base_offset,
                    passage_id=passage_id,
                )
            spans.extend(ner_results)
            ner_count += len(ner_results)

    deduped = dedupe_spans(spans)
    validated = validate_spans(deduped)
    serialized = serialize_spans(validated)
    detector_counts = Counter(
        payload["attributes"].get("detector")
        for payload in serialized
        if payload.get("source") == SpanSource.REGEX.value and payload.get("attributes")
    )
    stats = {
        "regex_spans": regex_count,
        "ner_spans": ner_count,
        "total_spans": len(serialized),
        "sentences": sentence_total,
        "passages": passages_processed,
        "detector_hits": {k: v for k, v in detector_counts.items() if k},
    }
    return serialized, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file)
    if not args.input.exists():
        LOGGER.error("Input directory %s not found", args.input)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline: Optional[NerPipeline]
    try:
        pipeline = NerPipeline(NERBackend(args.ner_backend))
    except Exception as exc:  # pragma: no cover - backend initialization errors
        LOGGER.error("Failed to initialize NER backend %s: %s", args.ner_backend, exc)
        return 2

    total_docs = 0
    total_spans = 0
    for doc_path in iter_article_paths(args.input, args.doc_ids):
        if args.limit is not None and total_docs >= args.limit:
            break
        with doc_path.open("r", encoding="utf-8") as fh:
            article = json.load(fh)
        spans, stats = process_article(article, ner_pipeline=pipeline)
        if not spans:
            LOGGER.info("Doc %s produced no spans.", article.get("doc_id"))
        else:
            LOGGER.info(
                "Doc %s spans=%s regex=%s ner=%s sentences=%s",
                article.get("doc_id"),
                stats["total_spans"],
                stats["regex_spans"],
                stats["ner_spans"],
                stats["sentences"],
            )
        output_path = args.output_dir / f"{article.get('doc_id', doc_path.stem)}.json"
        payload = {
            "doc_id": article.get("doc_id"),
            "title": article.get("title"),
            "source_path": str(doc_path),
            "stats": stats,
            "spans": spans,
        }
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2 if args.pretty else None)
        total_docs += 1
        total_spans += stats["total_spans"]

    LOGGER.info("Completed baseline extraction for %s docs (total spans: %s)", total_docs, total_spans)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

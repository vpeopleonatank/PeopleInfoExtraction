"""Fast Workflow B pass #1 runner using a local LLM backend (Ollama or vLLM)."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

from src import config
from src.extraction.passages import passage_records
from src.llm import OllamaClient, OllamaError, VLLMClient, VLLMError

LOGGER = logging.getLogger("workflow_b_fast_pass1")

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là hệ thống trích xuất thông tin pháp lý bằng tiếng Việt.\n"
    "- Chỉ sử dụng đúng đoạn văn bản đã cho; không suy đoán.\n"
    "- Luôn trả về JSON hợp lệ đúng theo lược đồ.\n"
    "- Nếu thông tin không xuất hiện nguyên văn, đặt null hoặc mảng rỗng.\n"
    "- Với mỗi trường không-null, cung cấp nguyên văn trích dẫn và chỉ số ký tự (start/end) trong passage.\n"
    "- Nếu không thể tuân thủ, trả về {\"error\": \"no_answer\"}."
)

SCHEMA_JSON = json.dumps(
    {
        "people": [
            {
                "name": {"text": "", "start": -1, "end": -1},
                "aliases": [{"text": "", "start": -1, "end": -1}],
                "age": {"value": None, "start": -1, "end": -1},
                "birth_date": {"value": None, "start": -1, "end": -1},
                "address": { "text": "", "start": -1, "end": -1 },
                "occupation": { "text": "", "start": -1, "end": -1 },
                "phones": [{"text": "", "start": -1, "end": -1}],
                "organizations": [{ "text": "", "start": -1, "end": -1 }],
                "national_id": {"text": "", "start": -1, "end": -1},
                "roles": [{"label": "suspect|victim|official|witness|lawyer|judge|other", "sentence_id": 0}],
                "actions": [
                    {
                        "predicate": "arrested|charged|sentenced|confessed|searched|seized|other",
                        "object_description": {
                            "text": "",
                            "start": -1,
                            "end": -1
                        },
                        "items_seized": [{"text": "", "start": -1, "end": -1}],
                        "location": {
                            "text": "",
                            "start": -1,
                            "end": -1
                        },
                        "time": {
                            "text": "",
                            "start": -1,
                            "end": -1
                        },
                        "amount": None,
                        "law_article": "",
                        "sentence_years": None,
                        "sentence_id": 0,
                    }
                ],
                "provenance": {"doc_id": "", "passage_id": 0},
                "confidence": 0.0,
            }
        ]
    },
    ensure_ascii=False,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Workflow B Pass #1 using a local LLM model.")
    parser.add_argument(
        "--backend",
        choices=("ollama", "vllm"),
        default="ollama",
        help="LLM backend to use (default: ollama).",
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
        default=config.CACHE_DIR / "workflow_b_fast" / "pass1",
        help="Directory to write extractor outputs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of articles to process.")
    parser.add_argument("--doc-ids", nargs="+", help="Specific doc_ids to process (default: all).")
    parser.add_argument("--model", default="qwen3:4b", help="Model name to use for the selected backend (default: qwen3:4b).")
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL for the selected backend (default: Ollama http://localhost:11435, vLLM http://localhost:8000/v1).",
    )
    parser.add_argument("--api-key", default=None, help="API key for OpenAI-compatible endpoints (vLLM).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (default: 0).")
    parser.add_argument("--timeout", type=int, default=1200, help="Request timeout in seconds (default: 120).")
    parser.add_argument(
        "--max-passage-chars",
        type=int,
        default=4000,
        help="Maximum number of characters per passage to send to the LLM (default: 4000).",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip passages that already have an output file.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument("--system-prompt-path", type=Path, help="Optional path to override the default system prompt.")
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        help="Optional JSONL file containing raw articles (fields: id, title, content, publish_date, ...). "
        "If set, overrides --articles-dir.",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")


def iter_article_paths(articles_dir: Path, doc_ids: Optional[Sequence[str]] = None) -> Iterator[Path]:
    if doc_ids:
        for doc_id in doc_ids:
            path = articles_dir / f"{doc_id}.json"
            if path.exists():
                yield path
            else:
                LOGGER.warning("Article %s not found under %s", doc_id, articles_dir)
    else:
        yield from sorted(articles_dir.glob("*.json"))


def iter_articles_from_jsonl(path: Path, doc_ids: Optional[Sequence[str]] = None) -> Iterator[dict]:
    filter_ids = set(doc_ids) if doc_ids else None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            article_id = record.get("id") or record.get("doc_id")
            if filter_ids and article_id not in filter_ids:
                continue
            yield jsonl_record_to_article(record)


def jsonl_record_to_article(record: Dict[str, object]) -> dict:
    doc_id = record.get("id") or record.get("doc_id")
    title = record.get("title")
    published_at = record.get("publish_date") or record.get("published_at")
    content = (record.get("content") or "") if isinstance(record.get("content"), str) else ""
    text = content.strip()
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    passage = {
        "passage_id": 0,
        "text": text,
        "start": 0,
        "end": len(text),
        "word_count": len(text.split()),
        "paragraph_indices": list(range(len(paragraphs))) if paragraphs else [0],
    }
    return {
        "doc_id": doc_id,
        "title": title,
        "published_at": published_at,
        "passages": [passage],
        "source": "jsonl",
    }


def load_system_prompt(path: Optional[Path]) -> str:
    if path is None:
        return DEFAULT_SYSTEM_PROMPT
    return path.read_text(encoding="utf-8").strip()


def build_user_prompt(article: dict, passage: dict, text: str) -> str:
    meta_lines = [
        f"doc_id: {article.get('doc_id')}",
        f"title: {article.get('title')}",
        f"published_at: {article.get('published_at')}",
        f"passage_id: {passage.get('passage_id')}",
        f"word_count: {passage.get('word_count') or len(text.split())}",
    ]
    metadata = "\n".join(meta_lines)
    return (
        "Trích xuất tất cả thông tin về con người trong đoạn báo tiếng Việt dưới đây.\n"
        "- Chỉ liệt kê cá nhân được nêu bằng tên riêng/biệt danh (danh từ riêng viết hoa); mỗi người chỉ một entry. Gộp biệt danh vào aliases.\n"
        "- Bỏ qua người chỉ xuất hiện dưới dạng danh từ chung hoặc đại từ không đi kèm tên riêng (ví dụ: \"người đàn ông\", \"tài xế\", \"anh ta\", \"họ\", \"nạn nhân\"); không suy luận tên từ chức danh/quan hệ.\n"
        "- Mọi text/start/end phải trích nguyên văn từ passage; nếu thiếu trường thì để null hoặc mảng rỗng và start/end = -1.\n"
        "- Quan hệ cho graph:\n"
        "  - Tổ chức đi cùng tên người → điền vào organizations với span nguyên văn.\n"
        "  - Hành động chỉ tạo khi câu chứa tên người đó; ghi predicate và spans cho object/location/time/items/law_article; dùng sentence_id của câu chứa tên.\n"
        "- Không tạo giá trị hay quan hệ nếu không có cụm từ rõ ràng trong passage; không đoán thêm.\n"
        "Metadata:\n"
        f"{metadata}\n\n"
        "Đoạn văn:\n<<<\n"
        f"{text}\n"
        ">>>\n\n"
        "Bám sát lược đồ JSON sau và chỉ trả về JSON hợp lệ, không thêm giải thích:\n"
        f"{SCHEMA_JSON}"
    )


def strip_code_fences(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
        return parts[-1].strip()
    return stripped


def try_parse_json(content: str) -> Tuple[Optional[dict], Optional[str]]:
    candidate = strip_code_fences(content)
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = candidate[start : end + 1]
            try:
                return json.loads(snippet), None
            except json.JSONDecodeError as exc:
                return None, str(exc)
        return None, "JSON decode error"


def write_output(path: Path, payload: dict, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    config.ensure_data_dirs()
    if args.jsonl_path and not args.jsonl_path.exists():
        LOGGER.error("JSONL path %s does not exist", args.jsonl_path)
        return 1
    system_prompt = load_system_prompt(args.system_prompt_path)
    default_base_urls = {
        "ollama": "http://localhost:11435",
        "vllm": "http://localhost:8000/v1",
    }
    base_url = args.base_url or default_base_urls[args.backend]

    if args.backend == "ollama":
        client = OllamaClient(
            model=args.model,
            base_url=base_url,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        error_cls = OllamaError
    else:
        client = VLLMClient(
            model=args.model,
            base_url=base_url,
            temperature=args.temperature,
            timeout=args.timeout,
            api_key=args.api_key,
        )
        error_cls = VLLMError

    processed_docs = 0
    success_passages = 0
    error_passages = 0
    run_start = time.perf_counter()

    def article_iter() -> Iterator[tuple[str, dict]]:
        if args.jsonl_path:
            for article in iter_articles_from_jsonl(args.jsonl_path, args.doc_ids):
                doc_id = article.get("doc_id") or "unknown"
                yield doc_id, article
        else:
            for doc_path in iter_article_paths(args.articles_dir, args.doc_ids):
                article = json.loads(doc_path.read_text(encoding="utf-8"))
                doc_id = article.get("doc_id") or doc_path.stem
                yield doc_id, article

    for doc_id, article in article_iter():
        if args.limit is not None and processed_docs >= args.limit:
            break
        doc_start = time.perf_counter()
        doc_attempts = 0
        doc_success = 0
        doc_errors = 0
        for passage in passage_records(article):
            passage_text = (passage.get("text") or "").strip()
            if not passage_text:
                continue
            passage_id = passage.get("passage_id", 0)
            output_path = args.output_dir / f"{doc_id}_p{passage_id}.json"
            if args.skip_existing and output_path.exists():
                LOGGER.info("Skipping %s passage %s (exists)", doc_id, passage_id)
                continue
            truncated_text = passage_text
            was_truncated = False
            if args.max_passage_chars and len(truncated_text) > args.max_passage_chars:
                truncated_text = truncated_text[: args.max_passage_chars]
                was_truncated = True
            user_prompt = build_user_prompt(article, passage, truncated_text)
            start_time = time.perf_counter()
            doc_attempts += 1
            try:
                generation = client.generate(system_prompt=system_prompt, user_prompt=user_prompt, format=None)
            except error_cls as exc:
                error_passages += 1
                doc_errors += 1
                write_output(
                    output_path,
                    {
                        "doc_id": doc_id,
                        "passage_id": passage_id,
                        "status": "request_error",
                        "error": str(exc),
                        "article_title": article.get("title"),
                        "passage": {
                            "text": truncated_text,
                            "original_length": len(passage_text),
                            "truncated": was_truncated,
                            "start": passage.get("start"),
                            "end": passage.get("end"),
                        },
                        "prompts": {"system": system_prompt, "user": user_prompt},
                        "model": args.model,
                    },
                )
                LOGGER.error("LLM call failed for %s passage %s: %s", doc_id, passage_id, exc)
                continue

            elapsed = time.perf_counter() - start_time
            parsed, parse_error = try_parse_json(generation.content)
            status = "ok" if parsed and parse_error is None else "parse_error"
            if status != "ok":
                error_passages += 1
                doc_errors += 1
                LOGGER.warning("JSON parse issue for %s passage %s: %s", doc_id, passage_id, parse_error)
            else:
                success_passages += 1
                doc_success += 1

            payload = {
                "doc_id": doc_id,
                "article_title": article.get("title"),
                "published_at": article.get("published_at"),
                "passage_id": passage_id,
                "status": status,
                "error": parse_error,
                "model": generation.model,
                "prompts": {"system": system_prompt, "user": user_prompt},
                "passage": {
                    "text": truncated_text,
                    "original_length": len(passage_text),
                    "truncated": was_truncated,
                    "start": passage.get("start"),
                    "end": passage.get("end"),
                    "paragraph_indices": passage.get("paragraph_indices"),
                },
                "response": {
                    "raw": generation.content,
                    "parsed": parsed,
                },
                "metrics": {
                    "elapsed_seconds": round(elapsed, 3),
                    "total_duration_ns": generation.total_duration,
                    "prompt_eval_count": generation.prompt_eval_count,
                    "eval_count": generation.eval_count,
                },
            }
            write_output(output_path, payload)
        processed_docs += 1
        if doc_attempts:
            LOGGER.info(
                "Doc %s processed in %.2fs (passages=%s, success=%s, issues=%s)",
                doc_id,
                time.perf_counter() - doc_start,
                doc_attempts,
                doc_success,
                doc_errors,
            )

    LOGGER.info(
        "Workflow B fast pass1 finished in %.2fs (articles=%s, passages=%s, success=%s, issues=%s)",
        time.perf_counter() - run_start,
        processed_docs,
        success_passages + error_passages,
        success_passages,
        error_passages,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

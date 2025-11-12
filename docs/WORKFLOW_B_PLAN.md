# Workflow B Implementation Plan (LLM Extraction → LLM Validation)

## 1. Current State Snapshot
- **Ingestion:** `src/ingestion/timeline.py`, `src/ingestion/article_loader.py`, and `src/ingestion/articles.py` already fetch Thanh Niên Pháp Luật URLs, clean HTML, and emit `data/clean_text/{doc_id}.json` with paragraph + passage offsets.
- **Deterministic/NER baseline:** `src/extraction/run_baseline.py` stitches regex detectors (`regex_detectors.py`) with pluggable NER (`ner.py`) to output spans into `data/cache/baseline_spans/`.
- **Verifier:** `src/extraction/verifier.py` + `run_verifier.py` ensure spans match verbatim text and annotate derivations. Downstream components (linker/upsert) are not yet implemented in this repo.
- **Gap vs Workflow B:** there is no LLM-driven extractor or validator. Existing CLIs assume regex/NER-first pipelines.

## 2. Goal
Run the **LLM-first dual-pass workflow (Workflow B)** so that:
1. Passages from `data/clean_text` go directly to an LLM that emits full structured person profiles (schema per `docs/SYSTEM_DESIGN.md`).
2. A second LLM/classifier validates each extracted field against the passage, downgrading or dropping unverifiable claims.
3. Outputs are persisted in cache directories and fed into the existing verifier/linker work later if needed.

## 3. Architecture Overview
```
article JSON → passage iterator
      │
      ├──► Pass #1 LLM extractor (strict JSON schema, temperature 0)
      │         │
      │         └─ writes raw schema outputs + token usage to cache
      │
      └──► Pass #2 validator (LLM or lightweight classifier) consumes
                passage text + Pass #1 JSON and emits per-field verdicts
                │
                └─ verified person profiles + span/evidence metadata
```

## 4. Work Plan

### Phase 0 – Prerequisites
1. **LLM provider selection**: confirm available endpoint (OpenAI, Azure, local) and authentication strategy; capture in `.env` + `src/config.py`.
2. **Schema contract**: promote the JSON schema from `docs/SYSTEM_DESIGN.md` into a Pydantic model (e.g., `src/extraction/schemas.py`) for serialization + validation.
3. **Data audit**: run `python -m src.ingestion.article_loader --limit N --skip-existing` to ensure fresh passages exist; sample to confirm token ranges within rate limits.

### Phase 1 – Extraction Runner (Pass #1)
1. **Module**: add `src/extraction/llm_extractor.py` with:
   - Prompt builder (system + user) enforcing the schema.
   - Client wrapper that handles retries, temperature=0, response schema validation, and cost logging.
2. **CLI**: implement `python -m src.extraction.run_llm_pass1`:
   - Loads article JSON (reuse `_passage_records` logic from `run_baseline.py` or factor it into a shared helper).
   - Applies passage gating (min tokens, zero-shot filter) and optional deterministic hints (e.g., candidate names from regex/NER if available) as prompt context.
   - Persists outputs under `data/cache/llm_pass1/{doc_id}_{passage_id}.json` with metadata (latency, token usage, prompt hash).
3. **Observability**: log per-passage status (success/error/no_answer) and emit aggregate metrics at the end of the CLI run.

### Phase 2 – Intermediate Normalization
1. Normalize Pass #1 JSON into internal models:
   - Map extracted people → `SpanModel`-compatible structures to reuse verifier + linking later.
   - Store verbatim evidence spans when provided; if LLM returns offsets, rebase to article-level char positions.
2. Cache a combined article-level artifact (`data/cache/llm_pass1_articles/{doc_id}.json`) summarizing per-person attributes for downstream use.

### Phase 3 – Validator Runner (Pass #2)
1. **Prompt**: create a validator template that takes `{passage_text, extracted_json}` and asks the model to label each field as `supported`, `contradicted`, or `not_found`, returning corrections + evidence spans.
2. **Runner** `python -m src.extraction.run_llm_validator`:
   - Iterates over Pass #1 outputs, injects the original passage text, and invokes the validator model.
   - Applies deterministic string matching to confirm evidence offsets; mismatches marked as dropped.
   - Emits `data/cache/llm_pass2/{doc_id}_{passage_id}.json` with per-field verdicts, confidence, and hallucination notes.
3. **Fallback**: if the validator errors or is over budget, fall back to the deterministic verifier for high-signal spans (phones, IDs).

### Phase 4 – Aggregation & Persistence
1. Combine Pass #2 verdicts into person-level profiles with `status` (`verified`, `dropped`, `needs_review`).
2. Store consolidated artifacts under `data/cache/workflow_b/{doc_id}.json`:
   - Original passage metadata.
   - Extracted people + attributes + evidence offsets.
   - Validation verdicts + reasons.
   - Cost/latency metrics for both passes.
3. Provide a thin adapter so later stages (linker/upsert) can consume workflow B outputs just like `verified_spans`.

### Phase 5 – Testing & QA
1. Unit-test prompt builders and schema validators (mock LLM responses).
2. Add integration tests with canned passages + fake LLM clients to ensure both CLIs produce deterministic outputs.
3. Record manual evaluation over ≥5 articles: measure validator drop/keep rates, hallucination rate, and compare to regex baseline.

### Phase 6 – Operationalization
1. **Config toggles**: add CLI flag (or config entry) to choose Workflow A vs B so orchestration scripts can run B-first by default.
2. **Automation**: document `make workflow_b` (or similar) that chains:
   ```
   python -m src.extraction.run_llm_pass1 \
   && python -m src.extraction.run_llm_validator \
   && python scripts/summarize_workflow_b.py
   ```
3. **Monitoring**: capture daily dashboards (token cost, schema error rate, validator drop reasons) and alert on regressions.

## 5. Risks & Mitigations
- **Token cost blow-up**: enforce passage-length guardrails and reuse cached outputs when rerunning validator-only experiments.
- **Schema drift**: keep Pydantic validation strict; fail fast when LLM returns malformed JSON, and retry with a repair prompt.
- **Offset alignment**: add regression tests comparing validator evidence spans to `article["paragraphs"]` slices; reuse `verify_span` helpers.
- **Dependency management**: extend `pyproject.toml` with the selected LLM SDK (e.g., `openai`, `anthropic`) and gate optional installs via extras.

## 6. Open Questions / Approvals Needed
1. Which LLM endpoint + model tier will be available in the deployment environment (impacts prompt size + batching strategy)?
2. Do we want deterministic hints (regex/NER) piped into Pass #1 prompts, or should workflow B stay purely LLM-based initially?
3. What confidence thresholds trigger human review vs automatic acceptance in Pass #2 outputs?

## 7. Fast Testing Harness for Workflow B (Vietnamese Corpus)

### 7.1 Goals
- Push **N Vietnamese articles** through Workflow B as quickly as possible.
- Persist structured outputs so a **stronger reviewer LLM** can analyze hallucinations, evidence coverage, and crime/role accuracy.
- Produce artifacts that feed directly into a **People Profile lookup** (input a name → return roles/actions/evidence).

### 7.2 Script Plan
1. **`python -m scripts.workflow_b_fast_pass1`** *(implemented; targets local Ollama `qwen3:4b`)*
   - Inputs: `--articles-dir data/clean_text`, `--limit N`, optional `--doc-ids`, `--model`, `--base-url`.
   - Uses the Vietnamese system prompt + schema guardrails to query the local Ollama server and capture raw JSON + metadata for each passage (≤4,000 chars by default).
   - Writes `data/cache/workflow_b_fast/pass1/{doc_id}_p{passage_id}.json` with passage text, prompts, raw response, parsed JSON, and latency/token counters emitted by Ollama.
   - Supports quick experiments via `--jsonl-path data/thanhnien_phapluat_sample.jsonl` (each line = raw article with `content` field), so you can test without running the ingestion/cleaning step.
   - Example: `python -m scripts.workflow_b_fast_pass1 --jsonl-path data/thanhnien_phapluat_sample.jsonl --limit 3 --model qwen3:4b`
2. **`python -m scripts.workflow_b_fast_validator`**
   - Consumes Pass #1 files, reuses the validator prompt, and stores verdicts under `data/cache/workflow_b_fast/pass2/{doc_id}_{passage_id}.json`.
   - Flags hallucinations/unsupported fields via `status in {"supported","hallucinated","needs_review"}` and keeps evidence spans.
3. **`python -m scripts.workflow_b_fast_review`**
   - Batches validated outputs (grouped per article) and sends them to a **strong reviewer LLM** (e.g., GPT-4.1 or Claude Opus) with both passage text snippets and structured JSON.
   - Reviewer returns a summary: hallucination count, accuracy score, missing fields, and crime/action highlights.
   - Results saved in `data/cache/workflow_b_fast/reviewer/{doc_id}.json`.
4. **`python -m scripts.workflow_b_profile_index`**
   - Aggregates validated people across all articles into `data/cache/workflow_b_fast/people_index.json`.
   - Structure: `{ "person_key": { "names": [...], "articles": [...], "actions": [{"crime": "...", "evidence": "..."}] } }`.
   - Provides CLI flag `--query "Nguyễn Văn A"` to print all associated roles/actions for manual inspection.

### 7.3 Data & Logging Conventions
- **Directory layout**
  ```
  data/cache/workflow_b_fast/
    ├── pass1/           # raw extractor outputs
    ├── pass2/           # validator verdicts
    ├── reviewer/        # strong-LLM review notes
    └── people_index.json
  ```
- Each JSON blob stores: `doc_id`, `passage_id`, `passage_text`, `llm_model`, `prompt_version`, `token_usage`, `timings`, `people`.
- Logs capture per-article latency + total cost to make budget projections before scaling beyond N.

### 7.4 Suggested System Prompts (Vietnamese-friendly)
1. **Extractor / Pass #1**
   ```
   Bạn là hệ thống trích xuất thông tin pháp lý bằng tiếng Việt. 
   Chỉ dùng đúng văn bản đã cho; không suy đoán. 
   Xuất JSON hợp lệ theo lược đồ cung cấp (people[].name, aliases, roles, actions, chứng cứ với start/end). 
   Nếu không chắc chắn hoặc thông tin không xuất hiện nguyên văn, đặt giá trị null hoặc mảng rỗng. 
   Bắt buộc trả về cả trích dẫn gốc và chỉ số ký tự cho mỗi trường không-null.
   ```
2. **Validator / Pass #2**
   ```
   Bạn là trình kiểm chứng. 
   Đầu vào gồm: đoạn báo + JSON trích xuất. 
   Với từng trường, trả về trạng thái {supported, contradicted, not_found}. 
   Chỉ đánh dấu supported nếu bạn tìm được chuỗi tương ứng trong văn bản (ghi rõ substring + vị trí). 
   Nếu thông tin sai hoặc không có, đặt trạng thái phù hợp và giải thích ngắn gọn bằng tiếng Việt.
   ```
3. **Reviewer (Strong LLM)**
   ```
   Bạn là kiểm toán viên chất lượng cao. 
   Nhận: (a) đoạn báo gốc tiếng Việt, (b) JSON đã được validator duyệt. 
   Nhiệm vụ: 
     - Tính điểm chính xác 0–1 dựa trên số trường supported / tổng trường. 
     - Liệt kê mọi dấu hiệu phóng đại hoặc thông tin thiếu chứng cứ. 
     - Tóm tắt hồ sơ từng người: tên, vai trò (ví dụ: nghi phạm, bị hại), hành động (họ làm gì, tội danh), và câu chứng cứ trích dẫn.
   Luôn viết tiếng Việt, giữ câu ngắn, ưu tiên dữ liệu kiểm chứng được.
   ```

### 7.5 Review & Reporting Flow
1. Run `workflow_b_fast_pass1` + `workflow_b_fast_validator` on the target N articles.
2. Trigger the reviewer script to obtain hallucination/accuracy audits per doc.
3. Generate a markdown or CSV summary (`scripts/workflow_b_fast_report.py`) with:
   - `doc_id`, total people, validator drop rate, reviewer accuracy score, token cost.
   - Top crimes/actions extracted, keyed by person.
4. Use `people_index.json` to answer ad-hoc profile questions (e.g., `python -m scripts.workflow_b_profile_index --query "Trần Văn B"`).

### 7.6 Integration with People Profile System
- The fast-testing artifacts double as seed data for a future **People Profile API**:
  - Index by normalized person name (case- and diacritic-insensitive).
  - Store per-article references (`doc_id`, `sentence_id`, evidence span) so UI can show “Nguyễn Văn A bị bắt vì …” with direct quotes.
  - Capture validator + reviewer verdicts to grade reliability before exposing facts to end users.
- Once satisfied with accuracy, plug `people_index.json` into downstream search or Neo4j prototypes to answer “người này đã làm gì?” queries.

# People Info Extraction PoC Plan (v0)

## Objective
Demonstrate end-to-end extraction of Person entities (with evidence spans, roles, and key attributes) from Vietnamese news articles into Neo4j, with ≤5% hallucinated fields, ≥80% precision on manually verified entities, and a dashboard surfacing provenance for at least 20 articles.

## Scope & Assumptions
- Input: cleaned HTML/text corpus from Thanh Niên "Pháp Luật" articles (timeline endpoint below) or existing crawler dumps.
- Models: existing regex library, Vietnamese NER (PhoBERT/VnCoreNLP), and one LLM endpoint with JSON schema support.
- Infra: local Neo4j instance, Python orchestration, no external services beyond what’s documented in `docs/SYSTEM_DESIGN.md`.
- Out of scope: real-time ingestion, production-hardening, policy/legal review.

## Workstreams & Tasks

### 1. Corpus Ingestion & Segmentation
- Define article metadata contract (doc_id, URL, publish date, section).
- Implement scraper/loader that pages through `https://thanhnien.vn/timelinelist/1855/{page}.htm` (replace `{page}` with 1..N) to collect ≥20 latest "Pháp Luật" article URLs.
- Fetch each article, clean markup, sentence-split, and persist passages (512–1,000 tokens) with offsets.
- Instrument BM25/embedding retriever for passage selection ahead of LLM calls.

### 2. Deterministic & NER Baseline Extraction
- Implement regex detectors for phones, Vietnamese national IDs, DOB, currency, and law articles (normalize formats + spans).
- Integrate lightweight Vietnamese NER to tag PERSON/ORG/LOC per sentence; expose spans + confidence.
- Emit unified span objects (`{type, text, start, end, sentence_id, source}`) to feed later stages.

### 3. LLM Extraction Orchestration
- Package JSON schema + strict prompt described in the system design; enforce temperature 0 and tool/JSON mode.
- Runner: iterate over passages, feed doc metadata + retrieved spans, capture raw JSON, and log latency/cost.
- Implement retries and schema validation; drop/flag passages returning `{"error":"no_answer"}`.

### 4. Verifier, Normalizer & Confidence Aggregation
- Implement verbatim span checker (string slice comparison) and null-out any mismatches.
- Normalize derived fields (e.g., age→birth_year) with provenance metadata (`derivation`, `source_span`).
- Compute field-level confidence (detector_conf * verifier_pass) and aggregate to entity-level scores.

### 5. Intra- & Cross-Doc Entity Linking
- Intra-doc: cluster mentions using name normalization, pronoun heuristics, and NER tags; assign cluster IDs.
- Cross-doc: build blocking keys (normalized name, hometown, birth_year) and implement similarity scorer with thresholds (auto-merge ≥0.9, review 0.6–0.9).
- Emit canonical Person records with alias lists, identifiers, confidence, and provenance bundle.

### 6. Graph Upsert Pipeline
- Define deterministic UIDs and Cypher MERGE templates (Person, Article, CaseEvent) with provenance edges.
- Implement write path that batches upserts per article, preserves doc_id/sentence_id/start/end on relationships.
- Add basic constraints/indexes (Person.uid, Article.doc_id, CaseEvent.eid) and migration scripts.

### 7. QA Harness & Review Dashboard
- Create evaluation set (10 hand-annotated articles) to score precision/recall per field.
- Build lightweight dashboard (Streamlit or notebook) summarizing Persons, roles, evidence quotes, and provenance links.
- Log metrics: extraction latency, LLM cost, verifier drop rate, linker match distribution.

## Timeline & Milestones (1-week PoC sprint)
- Day 0–1: Workstreams 1–2 complete; ingestion + deterministic spans validated.
- Day 2–3: LLM extractor + verifier running on sample corpus; <5% schema errors.
- Day 4: Entity linking and Neo4j upsert produce first 10 Persons with provenance.
- Day 5: QA harness + dashboard demo; precision/recall report + risk log delivered.

## Exit Criteria & Review Checklist
- 20 articles processed end-to-end; ≥80% precision on verified entities, zero unverifiable fields written.
- Neo4j contains Persons with roles/events and evidence spans accessible through dashboard.
- Known issues/backlog documented (e.g., low-confidence matches, missing fields).

## Risks & Mitigations
- **LLM cost/latency:** throttle via passage gating & batch scheduling; cache responses.
- **Schema drift:** enforce Pydantic schema + JSON schema validation before storage.
- **Linking false merges:** keep conservative thresholds, emit review queue.
- **Data quality variance:** maintain retriever quality metrics and fallback to deterministic-only output when LLM fails.

## Immediate Next Steps
1. Audit verifier drop reasons (`data/cache/verified_spans/*.json`) to tighten SIMPLE heuristic/regex behavior until `<10%` mismatches per doc:
   - Run the verifier in explain mode (`python3 -m src.extraction.run_verifier --explain-drops data/cache/verified_spans`) and bucket drop reasons (accent drift, boundary mismatch, null text).
   - Convert the counts into a short log (top offenders, doc_ids, before/after %) and feed concrete fixes (accent folding, punctuation nudges, fallback spans) into `regex_detectors` + `ner.py`.
   - Regenerate `data/cache/verified_spans` and ensure the new heuristics pass the existing verifier unit tests before locking the metric.
   - ✅ Snapshot (2025-11-10) captured in `@docs/VERIFIER_DROP_AUDIT.md`; root cause isolated to `segment_sentences` offset drift.
   - ✅ Offsets patched + regression test added (2025-11-11); drop rate now 0% across the seed corpus per the same audit log.
2. Define explicit confidence contracts for linking + Neo4j using `verified_spans` as the single source of truth:
   - Publish a table mapping `field_confidence` → action (`>=0.8` auto-link, `0.6–0.79` review queue, `<0.6` drop) plus entity-level rollups; store alongside the schema in `docs/SYSTEM_DESIGN.md`.
   - Update the linker to read `verified_spans` payloads (detector + verifier provenance) and emit `linking_decision` + `justification` blobs for Neo4j upserts.
   - Add contract tests that simulate low/high confidence spans to guarantee the Neo4j writer enforces the thresholds.
3. Grow the 10-article QA set ahead of the mid-sprint review:
   - Select 10 representative Pháp Luật articles (mix of crime/court topics), run the full pipeline, and capture manual annotations in `data/qa/hand_labels/{doc_id}.json`.
   - Use the verified outputs as seed labels but require second-pass human confirmation on roles, dates, and identifiers; log disagreements + adjudication notes.
   - Produce a QA tracker (spreadsheet or markdown table) summarizing coverage, precision/recall per field, and unresolved issues to drive the review agenda.

## Status Update (2025-11-10)
- Timeline crawler (`src/ingestion/timeline.py`) is live; `python3 -m src.ingestion.timeline --limit 20 --max-pages 5 --pretty` writes `data/cache/phap_luat_articles.json` with ≥20 unique Pháp Luật links.
- Article loader (`src/ingestion/article_loader.py`) fetches HTML, strips boilerplate, splits into passages, and stores JSON artifacts under `data/clean_text/{doc_id}.json` plus raw HTML in `data/raw_html/`.
- Config module + HTTP helper centralize headers/retry/backoff; data directories (`data/raw_html`, `data/clean_text`, `data/cache`) are auto-created via `config.ensure_data_dirs()`.
- Deterministic span stack landed: shared schema (`src/extraction/spans.py`), regex detectors, pluggable NER, and `python3 -m src.extraction.run_baseline` emitting `data/cache/baseline_spans/{doc_id}.json` (5 docs → 155 spans with detector metrics).
- Verifier/normalizer (`src/extraction/verifier.py` + `python3 -m src.extraction.run_verifier`) validates offsets, annotates derivations, and writes `data/cache/verified_spans/{doc_id}.json` (118 spans verified, 37 dropped for text mismatches).

## Execution Slice – Deterministic & NER (Target: 2025-11-12)

### Goal
- Produce a validated span inventory (regex + baseline Vietnamese NER) so passages from `article_loader` enter the verifier without manual edits.

### Deliverables
- `src/extraction/spans.py`: shared `Span`/`SpanSource` dataclasses + JSON/Pydantic schema helpers.
- `src/extraction/regex_detectors.py`: locale-aware detectors for phone, national ID/CCCD, DOB, currency, and Penal Code article spans (normalized value + verbatim offsets).
- `src/extraction/ner.py`: pluggable wrapper (PhoBERT/VnCoreNLP/simple fallback) emitting PERSON/ORG/LOC spans with honorific merging + confidence.
- CLI `python3 -m src.extraction.run_baseline --input data/clean_text --limit 10 --pretty` that writes `data/cache/baseline_spans/{doc_id}.json` with span + metric summaries.
- Unit tests covering detector normalization edge cases and `Span` schema serialization.

### Task Breakdown
1. Define `Span` schema (type/text/start/end/sentence_id/source/confidence/attributes) plus validation helpers and serialization tests.
2. Implement regex detectors with locale-aware patterns (dashed/spaced phone formats, CCCD prefixes) and a registry for composability.
3. Integrate PhoBERT/VnCoreNLP-ready NER hooks with a default lightweight fallback; merge honorifics (Ông/Bà/etc.) into mentions.
4. Build `run_baseline` CLI: iterate cleaned passages, call detectors + NER per sentence, dedupe overlaps, emit JSON + summary metrics (spans/doc, detector hit ratios).
5. Reuse the Pydantic schema for validation to mirror the upcoming LLM contract; surface schema errors in the CLI output.

### Acceptance Criteria
- 10 sample articles yield ≥90% detector precision on manual spot-checks; misses log to `baseline_spans.log`.
- Span JSON passes schema validation and flows into the verifier pipeline without additional transforms.

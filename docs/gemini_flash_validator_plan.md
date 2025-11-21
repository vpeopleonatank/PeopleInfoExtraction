## Gemini 2.5 Flash Validation Plan

### Goals
- Cross-check `passage.text`, `response.raw`, and `response.parsed.people` in workflow_b outputs for hallucinations or omissions using Google Gemini 2.5 Flash.
- Produce structured reports that highlight unsupported claims, missing people, or inconsistent facts without disrupting current Ollama-based validation.

### Scope & Constraints
1. **Inputs**: Files under `data/cache/workflow_b_fast/pass1/*.json` containing the passage plus either `response.parsed.people` (preferred) or `response.raw`.
2. **Outputs**: Per-file Gemini reports (JSON) saved to `data/cache/workflow_b_fast/gemini_reports/` and an optional batch summary.
3. **Environment**: No outbound network allowed in repo, so implementation must be ready for execution elsewhere; provide mock/dry-run support.

### Architecture Overview
1. **Dependency**: Add the official `python-genai` client as an optional dependency in `pyproject.toml`.
2. **Validator Class**: Implement `src/validation/gemini_validator.py` exposing `GeminiFlashValidator`.
   - Initializes Gemini client with `model="gemini-2.5-flash"` (CLI override allowed).
   - Accepts API key via environment (`GOOGLE_API_KEY`) or CLI flag.
   - Handles prompt construction, response parsing, retries, and error normalization.
3. **Prompt Strategy**:
   - System instructions emphasize factual checking only within provided passage.
   - User payload includes metadata (doc_id, title, published_at, passage_id), the passage text, and serialized extracted people (or raw response fallback).
   - Request output JSON schema:
     ```json
     {
       "verdict": "supported|unsure|unsupported",
       "issues": [
         {
           "type": "hallucination|missing_info|conflict",
           "person_name": "…",
           "field": "…",
           "description": "…",
           "evidence": "quoted span or null"
         }
       ],
       "missing_people": [
         {
           "name": "…",
           "snippet": "…"
         }
       ]
     }
     ```
   - Enforce JSON by wrapping prompt with “respond ONLY with JSON”; add minimal temperature (0 or 0.2) for determinism.
4. **Batch CLI**:
   - New script `scripts/workflow_b_gemini_validator.py` mirroring the Ollama batch tool.
   - CLI options: `--input-dir`, `--output-dir`, `--limit`, `--no-skip-existing`, `--api-key`, `--model`, `--rate-limit-ms`, `--summary-only`, `--dry-run`.
   - Dry-run mode prints the prompt + mocked response for auditing.
5. **Report Schema**:
   - Reuse `ValidationReport` scaffolding where possible or define new Pydantic models (e.g., `GeminiValidationReport`, `GeminiIssue`).
   - Include metadata: doc_id, passage_id, validator_model, original_model, timestamps, source file path, verdict summary counts per issue type.

### Error Handling & Resilience
- Guard against long passages: truncate or split while clearly noting trimming in prompt.
- JSON parsing: attempt `json.loads`, fallback to `json.loads` after stripping surrounding text, final fallback marks response as parsing error.
- Rate limiting: optional sleep between calls; support `--rate-limit-ms`.
- Logging: capture raw Gemini response in report for debugging (optional flag).

### Verification Plan
1. **Unit-style tests**: Add tests under `tests/validation/test_gemini_validator.py` using mocked Gemini client to ensure prompt formation and response parsing.
2. **Dry-run/demo script**: Provide `scripts/test_gemini_validator_demo.py` that loads the sample JSON and uses a stubbed response to demonstrate behavior without API calls.
3. **Documentation**: Update `README_VALIDATION.md` and/or create a short guide detailing setup steps, command examples, and interpretation of Gemini reports.

### Implementation Order
1. Update dependencies & config.
2. Implement validator class + models.
3. Implement batch CLI + demo script.
4. Add tests and documentation updates.
5. Manual dry-run with sample file (mocked) to validate end-to-end flow.

### Usability Metric (LLM-Graded) Strategy
- **Definition**: A result is "usable" when Gemini returns `verdict=supported` and flags no critical issues and no missing people. Non-critical issues (warning/info) are allowed but must be listed.
- **Single-pass rubric prompt**: Extend the Gemini JSON schema to include `usable` and `confidence` with rules baked into the prompt: usable=true only if verdict is supported AND there are no critical issues AND `missing_people` is empty; if unsure, set usable=false. Make the model label issue severity (critical|warning|info) so the rule can be enforced.
- **Prompt stub**:
  ```json
  {
    "verdict": "supported|unsure|unsupported",
    "issues": [
      {
        "type": "hallucination|missing_info|conflict|other",
        "severity": "critical|warning|info",
        "person_name": null,
        "field": null,
        "description": "",
        "evidence": null
      }
    ],
    "missing_people": [...],
    "usable": true,
    "confidence": 0.0,
    "notes": "brief rationale"
  }
  ```
  Instructions: "Usable only if supported AND missing_people is empty AND there are no critical issues (warnings/info allowed but must be listed). If unsure, mark usable=false."
- **Batch aggregation**: In the batch summary, count `usable=true` reports and compute `usable_percentage = usable/total * 100`. This is purely model-judged, no manual review loop.
- **Calibration loop**: Before locking the rule, run Gemini on 3–5 representative files asking it to propose a rubric and expected usable rate. Compare against a small manual spot-check (≈10 files). If the rubric is too strict/lenient, tune the "usable" rule in the prompt (e.g., allow ≤1 warning/info issue but still reject any critical issues).
- **Determinism**: Use temperature 0, request JSON via `response_mime_type=application/json`, and instruct the model to default to `usable=false` when the passage is truncated or it is uncertain.

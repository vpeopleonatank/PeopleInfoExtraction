# LLM Extraction Validation System

A comprehensive validation system for checking Vietnamese people information extraction outputs from Qwen3-4b using a larger, more capable local LLM.

## Quick Start

```bash
# 1. Make sure Ollama is running
ollama serve

# 2. Pull validator model
ollama pull qwen2.5:14b

# 3. Run validation
python scripts/workflow_b_validator.py

# 4. Check results
cat data/cache/workflow_b_fast/validation_reports/_batch_summary.json
```

## What Was Built

### 1. Validation Models (`src/validation/models.py`)

Pydantic models for structured validation reports:

- **`ValidationIssue`**: Individual issues with type, severity, and details
- **`ValidationReport`**: Complete report per file with summary statistics
- **`BatchValidationSummary`**: Aggregate statistics across all files
- **Issue Types**: hallucination, wrong_position, missing_entity, wrong_law_article, schema_violation
- **Severities**: critical, error, warning, info

### 2. Validator Class (`src/validation/llm_validator.py`)

`PeopleValidator` class with four validation methods:

1. **`validate_character_positions()`**: Checks start/end indices match actual text
   - Validates name, age, birth_place, phones, national_id, aliases positions
   - Detects out-of-bounds indices
   - Finds mismatched text spans

2. **`validate_factual_accuracy()`**: Detects hallucinations
   - Verifies law articles appear in source text
   - Checks monetary amounts are reasonable
   - Flags fabricated information

3. **`validate_schema_compliance()`**: Ensures data follows schema
   - Validates role labels (suspect, victim, official, witness, lawyer, judge, other)
   - Validates action predicates (arrested, charged, sentenced, etc.)

4. **`check_completeness()`**: Finds missing entities using LLM
   - Uses validator LLM to identify people not extracted
   - Provides context snippets for missing entities

### 3. Batch Processing Script (`scripts/workflow_b_validator.py`)

Full-featured CLI for batch validation:

- Process directories of extraction files
- Progress tracking with tqdm
- Skip existing reports (configurable)
- Generate per-file validation reports
- Create aggregate batch summary
- Comprehensive error handling

### 4. Demo Script (`scripts/test_validator_demo.py`)

Rule-based demo that works without Ollama:

- Demonstrates validation logic on sample file
- Shows what issues would be detected
- Useful for testing and development

### 5. Documentation (`docs/validation_guide.md`)

Complete guide covering:

- Installation and setup
- Usage examples
- Output format specifications
- Issue types and severities
- Integration with extraction workflow
- Troubleshooting guide

## Example: Issues Found in Sample File

Running the demo on `019a2f6a-4fb7-7b90-840c-f5c3d20f5093_p0.json`:

```
Total issues found: 11

Issues:
  1. Phạm Văn Sử: Wrong name position (claimed 10:16, actually at 16)
  2. Lê Văn Sử: Wrong name position (claimed 100:15, actually at 364)
  3. Trần Trung Cang: Wrong name position (claimed 200:25, actually at 1621)
  4. Trần Trung Cang: Age position invalid (start == end)
  5. Trần Trung Cang: Birth place position invalid
  6. Đinh Công Lý: Wrong name position (claimed 300:35, actually at 2064)
  7. Đinh Công Lý: Age position invalid (start == end)
  8. Tô Hiền Đệ: Wrong name position (claimed 350:40, actually at 2088)
  9. Tô Hiền Đệ: Age position invalid (start == end)
  10. Nguyễn Chí Linh: Wrong name position (claimed 400:45, actually at 2112)
  11. Nguyễn Chí Linh: Age position invalid (start == end)
```

All 6 extracted people had position errors! This shows the validator successfully detects issues.

## Validation Report Example

```json
{
  "doc_id": "019a2f6a-4fb7-7b90-840c-f5c3d20f5093",
  "passage_id": 0,
  "summary": {
    "total_people_extracted": 6,
    "total_issues_found": 11,
    "errors": 6,
    "warnings": 5
  },
  "issues": [
    {
      "type": "wrong_position",
      "severity": "error",
      "person_name": "Phạm Văn Sử",
      "field": "name",
      "expected": "'Phạm Văn Sử' at position 16",
      "actual": "', ông ' at position 10",
      "description": "Text at specified position doesn't match. Found at position 16 instead."
    }
  ]
}
```

## Architecture

```
Extraction Files                 Validator                Output
─────────────────               ──────────               ──────────

workflow_b_fast/pass1/          PeopleValidator          validation_reports/
├─ file1.json ─────────────┐                        ┌──> file1.json
├─ file2.json ─────────────┼──> validate_extraction()─┼──> file2.json
├─ file3.json ─────────────┘    - check positions    └──> file3.json
└─ ...                          - check facts             ...
                                - find missing         _batch_summary.json
                                - validate schema
                                    │
                                    └──> Uses Ollama
                                         (qwen2.5:14b)
```

## Files Created

```
src/validation/
├── __init__.py                 # Module init
├── models.py                   # Pydantic validation models
└── llm_validator.py            # PeopleValidator class

scripts/
├── workflow_b_validator.py     # Main batch validation script
└── test_validator_demo.py      # Demo without Ollama

docs/
└── validation_guide.md         # Complete usage guide
```

## Usage

**Validate all files:**

```bash
python scripts/workflow_b_validator.py
```

**Test on 5 files:**

```bash
python scripts/workflow_b_validator.py --limit 5
```

**Run demo without Ollama:**

```bash
python scripts/test_validator_demo.py
```

**View summary:**

```bash
python scripts/workflow_b_validator.py --summary-only
```

**Optional: Cross-check with Gemini 2.5 Flash**

```bash
export GOOGLE_API_KEY="your-key"
python scripts/workflow_b_gemini_validator.py --limit 5
```

**Review Gemini report with passage + issues side-by-side**

```bash
# single report
python scripts/view_gemini_report.py data/cache/workflow_b_fast/gemini_reports/<file>.json

# or inspect an entire directory
python scripts/view_gemini_report.py data/cache/workflow_b_fast/gemini_reports/*.json

# prompt to mark manual status while reviewing
python scripts/view_gemini_report.py data/cache/workflow_b_fast/gemini_reports/*.json --prompt-mark

# quick bulk mark (sets field without prompting)
python scripts/view_gemini_report.py some_report.json --mark invalid
```

Use `--left-ratio` (e.g., `0.6`) if you want to allocate more terminal width to the passage column. The viewer automatically skips `_batch_summary.json`.

To see how many reports have been reviewed manually, run:

```bash
python scripts/summarize_manual_status.py data/cache/workflow_b_fast/gemini_reports
```

Pass `--list-unset` to print the files you still need to mark.

Gemini reports will be written to `data/cache/workflow_b_fast/gemini_reports/`. Use `--dry-run` to inspect prompts without making network calls or run `scripts/test_gemini_validator_demo.py` for an end-to-end stubbed example.

## Next Steps

1. **Start Ollama**: `ollama serve`
2. **Pull validator model**: `ollama pull qwen2.5:14b`
3. **Run validation**: `python scripts/workflow_b_validator.py --limit 5` (test on 5 files first)
4. **Review results**: Check `data/cache/workflow_b_fast/validation_reports/`
5. **Run full batch**: `python scripts/workflow_b_validator.py` (process all files)
6. **Analyze issues**: Use batch summary to identify improvement areas

## Benefits

- **Quality assurance**: Automatically detect hallucinations and errors
- **Position validation**: Ensure character indices are correct
- **Completeness checking**: Find missed entities
- **Schema enforcement**: Validate data follows expected format
- **Batch reporting**: Aggregate statistics across all extractions
- **Actionable insights**: Detailed issue reports for debugging

For detailed documentation, see: `docs/validation_guide.md`

# Validation System Guide

This guide explains how to use the validation system to check LLM extraction outputs for hallucinations, errors, and missing information.

## Overview

The validation system uses a larger, more capable local LLM (via Ollama) to validate the outputs from the faster extraction model (Qwen3-4b). It performs four types of checks:

1. **Character Position Validation**: Verifies start/end indices match actual text positions
2. **Factual Accuracy Check**: Detects hallucinations and fabricated information
3. **Completeness Check**: Finds people/entities that were missed in extraction
4. **Schema Compliance**: Validates role labels, predicates, and data formats

## Prerequisites

### 1. Install Ollama

```bash
# Install Ollama (if not already installed)
curl https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

### 2. Pull a Validator Model

The validator should use a larger/better model than the original extraction model. Recommended options:

```bash
# Option 1: Qwen 2.5 14B (recommended - good balance of quality and speed)
ollama pull qwen2.5:14b

# Option 2: Qwen 2.5 32B (higher quality, slower)
ollama pull qwen2.5:32b

# Option 3: Llama 3.1 70B (highest quality, requires more resources)
ollama pull llama3.1:70b
```

## Usage

### Basic Usage

Validate all extraction files in the default directory:

```bash
python scripts/workflow_b_validator.py
```

This will:
- Process all JSON files in `data/cache/workflow_b_fast/pass1/`
- Use `qwen2.5:14b` as the validator model
- Output validation reports to `data/cache/workflow_b_fast/validation_reports/`
- Skip files that have already been validated
- Generate a batch summary

### Command-Line Options

```bash
python scripts/workflow_b_validator.py \
  --input-dir data/cache/workflow_b_fast/pass1 \
  --output-dir data/cache/workflow_b_fast/validation_reports \
  --validator-model qwen2.5:14b \
  --ollama-url http://localhost:11434 \
  --limit 10 \
  --no-skip-existing
```

**Options:**

- `--input-dir PATH`: Directory containing extraction JSON files (default: `data/cache/workflow_b_fast/pass1`)
- `--output-dir PATH`: Directory to save validation reports (default: `data/cache/workflow_b_fast/validation_reports`)
- `--validator-model MODEL`: Ollama model to use (default: `qwen2.5:14b`)
- `--ollama-url URL`: Ollama API base URL (default: `http://localhost:11434`)
- `--limit N`: Limit number of files to process (useful for testing)
- `--no-skip-existing`: Re-validate files that already have reports
- `--summary-only`: Only generate batch summary from existing reports

### Examples

**Validate a small batch for testing:**

```bash
python scripts/workflow_b_validator.py --limit 5
```

**Use a different validator model:**

```bash
python scripts/workflow_b_validator.py --validator-model llama3.1:70b
```

**Re-validate all files:**

```bash
python scripts/workflow_b_validator.py --no-skip-existing
```

**Generate summary only (without re-validation):**

```bash
python scripts/workflow_b_validator.py --summary-only
```

## Output Format

### Validation Report (per file)

Each validation report is saved as a JSON file with the following structure:

```json
{
  "doc_id": "019a2f6a-4fb7-7b90-840c-f5c3d20f5093",
  "passage_id": 0,
  "article_title": "Cà Mau sắp xét xử lưu động 2 vụ án IUU...",
  "validator_model": "qwen2.5:14b",
  "original_model": "qwen3:4b",
  "validation_timestamp": "2025-11-12T10:30:00",
  "summary": {
    "total_people_extracted": 6,
    "total_issues_found": 11,
    "critical_issues": 0,
    "errors": 6,
    "warnings": 5,
    "info_issues": 0,
    "hallucinations": 0,
    "wrong_positions": 6,
    "missing_entities": 0,
    "schema_violations": 0,
    "wrong_law_articles": 0
  },
  "issues": [
    {
      "type": "wrong_position",
      "severity": "error",
      "person_name": "Phạm Văn Sử",
      "field": "name",
      "expected": "'Phạm Văn Sử' at position 16",
      "actual": "', ông ' at position 10",
      "description": "Text at specified position doesn't match. Found at position 16 instead.",
      "text_snippet": "Ngày 29.10, ông Phạm Văn Sử, Phó giám đốc"
    }
  ],
  "source_file": "/path/to/original/file.json"
}
```

### Batch Summary

The batch summary (`_batch_summary.json`) aggregates statistics across all validated files:

```json
{
  "total_files_processed": 100,
  "total_files_with_issues": 85,
  "total_issues": 342,
  "total_people_extracted": 456,
  "critical_issues": 12,
  "errors": 234,
  "warnings": 89,
  "info_issues": 7,
  "hallucinations": 12,
  "wrong_positions": 198,
  "missing_entities": 23,
  "schema_violations": 15,
  "wrong_law_articles": 94,
  "validation_timestamp": "2025-11-12T12:00:00",
  "validator_model": "qwen2.5:14b",
  "original_model": "qwen3:4b"
}
```

## Issue Types and Severities

### Issue Types

1. **`hallucination`**: Extracted information not present in source text (most serious)
2. **`wrong_position`**: Character indices don't match actual text positions
3. **`missing_entity`**: Person/entity appears in text but wasn't extracted
4. **`wrong_law_article`**: Law article citation doesn't match source text
5. **`schema_violation`**: Invalid role labels or action predicates
6. **`incomplete_extraction`**: Other completeness issues

### Severity Levels

1. **`critical`**: Hallucinations, fabricated data (requires immediate attention)
2. **`error`**: Wrong positions, incorrect data (should be fixed)
3. **`warning`**: Potential issues, edge cases (review recommended)
4. **`info`**: Suggestions, minor notes (informational only)

## Testing Without Ollama

If Ollama is not available, you can run a demo validator that shows what issues would be detected:

```bash
python scripts/test_validator_demo.py
```

This performs rule-based validation without using an LLM and is useful for:
- Understanding what the validator checks
- Quick testing without LLM overhead
- Development and debugging

## Gemini 2.5 Flash Validator

In addition to the local Ollama validator, you can send extractions to Google Gemini 2.5 Flash for an external fact-check.

### Prerequisites

1. Install project dependencies (the `python-genai` client is already in `pyproject.toml`):
   ```bash
   pip install -e .
   ```
2. Obtain a Google API key and set it in the environment:
   ```bash
   export GOOGLE_API_KEY="your-key"
   ```

### Running the Gemini Batch Validator

```bash
python scripts/workflow_b_gemini_validator.py \
  --input-dir data/cache/workflow_b_fast/pass1 \
  --output-dir data/cache/workflow_b_fast/gemini_reports \
  --limit 5
```

- Reports are saved as JSON files alongside `_batch_summary.json` in the specified output directory.
- Use `--dry-run` to inspect prompts and receive mocked responses without calling the API.
- Use `--rate-limit-ms` to throttle requests if hitting quota limits.
- Generate summaries from existing reports with `--summary-only`.

### Demo Without Network Access

When network access is unavailable, run the stubbed demo:

```bash
python scripts/test_gemini_validator_demo.py
```

This loads a sample file, feeds it to a fake Gemini client, and prints the resulting report to illustrate the workflow.

## Integration with Workflow

### Recommended Workflow

1. **Extract data** using the fast model:
   ```bash
   python scripts/workflow_b_fast_pass1.py
   ```

2. **Validate outputs**:
   ```bash
   python scripts/workflow_b_validator.py
   ```

3. **Review the batch summary** to understand overall quality:
   ```bash
   cat data/cache/workflow_b_fast/validation_reports/_batch_summary.json
   ```

4. **Fix critical issues** in files with high-severity problems

5. **Optionally re-run extraction** with better prompts or different models based on validation findings

### Filtering High-Priority Issues

To find files with critical issues:

```bash
# Find reports with hallucinations
grep -l '"hallucinations": [1-9]' data/cache/workflow_b_fast/validation_reports/*.json

# Find reports with critical severity issues
grep -l '"critical_issues": [1-9]' data/cache/workflow_b_fast/validation_reports/*.json
```

## Performance Considerations

- **Validation speed**: ~2-5 seconds per file with `qwen2.5:14b` on GPU
- **Resource usage**: Larger models (32B, 70B) require more VRAM
- **Batch processing**: Use `--limit` for initial testing, then run full batch overnight
- **Caching**: Use default skip-existing behavior to avoid re-validation

## Troubleshooting

### Ollama Connection Error

```
Error: HTTPConnectionPool(host='localhost', port=11434): Connection refused
```

**Solution**: Make sure Ollama is running:
```bash
ollama serve
```

### Model Not Found

```
Error: model 'qwen2.5:14b' not found
```

**Solution**: Pull the model first:
```bash
ollama pull qwen2.5:14b
```

### Out of Memory

**Solution**: Use a smaller model or reduce batch size:
```bash
# Use smaller model
python scripts/workflow_b_validator.py --validator-model qwen2.5:7b

# Process in smaller batches
python scripts/workflow_b_validator.py --limit 10
```

## Sample Output

When running validation, you'll see progress and a summary:

```
Found 100 files to process
Validator model: qwen2.5:14b
Output directory: data/cache/workflow_b_fast/validation_reports

Validating extractions: 100%|████████████| 100/100 [08:23<00:00,  5.03s/it]

✓ Successfully validated: 100

Generating batch summary...

================================================================
BATCH VALIDATION SUMMARY
================================================================
Files processed: 100
Files with issues: 85 (85.0%)
Total people extracted: 456
Total issues found: 342

Issues by severity:
  Critical: 12
  Errors:   234
  Warnings: 89
  Info:     7

Issues by type:
  Hallucinations:       12
  Wrong positions:      198
  Missing entities:     23
  Wrong law articles:   94
  Schema violations:    15

Summary saved to: data/cache/workflow_b_fast/validation_reports/_batch_summary.json
```

## Next Steps

After validation, you can:

1. **Improve extraction prompts** based on common issues
2. **Adjust model parameters** (temperature, sampling) to reduce hallucinations
3. **Use validation reports** to fine-tune the extraction model
4. **Filter out low-quality extractions** based on issue counts
5. **Create a second pass** that fixes validated issues automatically

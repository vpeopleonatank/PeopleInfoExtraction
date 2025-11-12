# Verifier Drop Audit — 2025-11-10

## Snapshot
- Source data: `data/cache/verified_spans/*.json` (5 docs processed on 2025-11-10).
- Comparison baseline: `data/cache/baseline_spans/*.json` + article text under `data/clean_text`.
- Total baseline spans (persons only in this slice): 118 verified / 37 dropped ⇒ 23.9% drop rate (target <10%).

| doc_id | verified | dropped | drop_rate |
|---|---:|---:|---:|
| 18525110618474009 | 30 | 12 | 28.6% |
| 185251107113106479 | 14 | 4 | 22.2% |
| 18525110812244065 | 17 | 6 | 26.1% |
| 185251109133337638 | 24 | 5 | 17.2% |
| 185251109161718931 | 33 | 10 | 23.3% |

## Failure buckets
- `text_mismatch` (37/37 drops): every failure came from SIMPLE NER spans whose offsets are shifted one character to the left. Slicing the article text at `[start:end]` produces strings like `' An'` vs target `'Anh'`, i.e., a leading space plus a missing trailing character.
- Root cause: `segment_sentences` (`src/extraction/run_baseline.py:54-71`) trims each regex-matched sentence with `.strip()` but does **not** adjust the reported `rel_offset`. When offsets are added to `passage_start`, every span coming from the stripped sentence inherits a `start` that is `len(leading whitespace)` characters too small. The verifier compares untrimmed slices, so the shifted windows fail the verbatim check.
- Regex spans are unaffected because their matcher consumes the same trimmed sentence string; only SIMPLE NER (and any backend using `segment_sentences`) is impacted.

## Remediation plan
1. **Fix offset math in `segment_sentences`:** preserve the raw substring or track the number of leading characters removed. For example, capture `raw = match.group(0)` and `leading_ws = len(raw) - len(raw.lstrip())`, then emit `(raw.strip(), match.start() + leading_ws)`. This keeps the clean sentence text without losing alignment.
2. **Add regression tests:** create a unit test that feeds a sentence with leading whitespace/newline into `segment_sentences`, runs the SIMPLE backend, and asserts that the resulting span indexes slice the original passage text verbatim. This guards all future backends that rely on the helper.
3. **Re-run `python -m src.extraction.run_baseline … && python -m src.extraction.run_verifier …` after the fix:** expect `leading_space_offset` drops to 0 and per-doc drop rates to fall below 10% (regex-only spans already pass). Capture the “before vs. after” metrics in this log once the rerun completes.

Optional hygiene:
- Expose a `--explain-drops` flag in `run_verifier` that prints the offending slice vs. expected text to speed up future audits.
- During NER span creation, defensively trim whitespace when `sentence[offset_start]` is whitespace and update the offsets before emitting the span. This is redundant once `segment_sentences` is fixed but provides an extra safety net for upstream sentence segmentation changes.

## Post-fix verification — 2025-11-11
- Applied the offset fix in `segment_sentences` (`src/extraction/run_baseline.py`) and added `test_segment_sentences_preserves_offset_after_stripping` to lock in the behavior.
- Replayed the deterministic pipeline:
  - `python -m src.extraction.run_baseline --limit 5 --pretty`
  - `python -m src.extraction.run_verifier --limit 5 --pretty`
- Result: 155/155 spans verified, 0 drops (0%) while detector counts and confidence remain unchanged.

| doc_id | verified | dropped | drop_rate |
|---|---:|---:|---:|
| 18525110618474009 | 42 | 0 | 0.0% |
| 185251107113106479 | 18 | 0 | 0.0% |
| 18525110812244065 | 23 | 0 | 0.0% |
| 185251109133337638 | 29 | 0 | 0.0% |
| 185251109161718931 | 43 | 0 | 0.0% |

Next audit step is to rerun once additional docs/NER backends land, but for now the `<10%` mismatch goal is met with margin.

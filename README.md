# Command to extract with vllm
```
python -m scripts.workflow_b_fast_pass1 --backend vllm --model Qwen/Qwen3-8B-FP8 --base-url http://localhost:8000/v1 --jsonl-path data/thanhnien_phapluat_1000.jsonl --limit 1000 --output-dir data/cache/workflow_b_fast/qwen3_8B_FP8
```

# Command to extract with gemini flash 2.5
```
python -m scripts.workflow_b_fast_pass1 --jsonl-path data/thanhnien_phapluat_1000.jsonl --output-dir data/cache/workflow_b_fast/pass1/gemini_flash_2.5 --limit 1000 --backend openrouter --model google/gemini-2.5-flash --api-key ...
```
# Command to validate extracted results with gemini flash 2.5

```
python scripts/workflow_b_gemini_validator.py --input-dir data/cache/workflow_b_fast/pass1/gemini_flash_2.5 --output-dir data/cache/workflow_b_fast/gemini_flash_2.5_gemini_reports --no-skip-existing --max-output-tokens 12000 --transport openrouter --model google/gemini-2.5-flash --openrouter-api-key ...
```

"""Common LLM dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(slots=True)
class LLMGeneration:
    """Container for a non-streamed LLM response."""

    content: str
    model: str
    created_at: Optional[Union[str, int]]
    total_duration: Optional[int]
    prompt_eval_count: Optional[int]
    eval_count: Optional[int]

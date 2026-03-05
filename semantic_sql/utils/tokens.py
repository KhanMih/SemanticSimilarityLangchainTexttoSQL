"""Token counting utilities for prompt budget management."""

from __future__ import annotations

import tiktoken

_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.encoding_for_model("gpt-4o-mini")
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def estimate_example_tokens(question: str, sql: str, explanation: str = "") -> int:
    """Estimate token cost of injecting one few-shot example into the prompt."""
    block = f"Question: {question}\nSQL: {sql}"
    if explanation:
        block += f"\nExplanation: {explanation}"
    return count_tokens(block) + 10  # overhead for formatting/delimiters

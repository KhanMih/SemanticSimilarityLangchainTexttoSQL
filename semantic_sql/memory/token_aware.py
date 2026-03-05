"""Token-aware example selection.

Dynamically adjusts k so that injected few-shot examples + schema context
never blow the context-window budget.
"""

from __future__ import annotations

from semantic_sql.config import settings
from semantic_sql.memory.example_store import ExampleStore
from semantic_sql.models.schemas import TableSchema, VettedExample
from semantic_sql.utils.tokens import count_tokens, estimate_example_tokens


def _schema_tokens(schemas: list[TableSchema]) -> int:
    return sum(count_tokens(s.ddl) for s in schemas)


def select_examples_within_budget(
    store: ExampleStore,
    question: str,
    schema_context: list[TableSchema],
    *,
    max_tokens: int | None = None,
    base_prompt_tokens: int = 500,
) -> list[VettedExample]:
    """Select as many few-shot examples as fit within the token budget.

    Priority order:
      1. Schema context (non-negotiable — the agent needs DDL)
      2. Few-shot examples (highest similarity first, added greedily)
    """
    budget = max_tokens or settings.max_prompt_tokens
    used = base_prompt_tokens + _schema_tokens(schema_context) + count_tokens(question)
    remaining = budget - used

    if remaining <= 0:
        return []

    candidates = store.select_examples(question, k=settings.few_shot_k * 2)

    selected: list[VettedExample] = []
    for ex in candidates:
        cost = estimate_example_tokens(ex.question, ex.sql_query, ex.explanation)
        if cost > remaining:
            break
        selected.append(ex)
        remaining -= cost
        if len(selected) >= settings.few_shot_k:
            break

    return selected

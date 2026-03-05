"""Assembles the 3-layer prompt from Behavioral, Knowledge, and Executive sections."""

from __future__ import annotations

from semantic_sql.agent.prompts.templates import (
    BEHAVIORAL_EXAMPLE_BLOCK,
    BEHAVIORAL_SECTION,
    EXECUTIVE_SECTION,
    KNOWLEDGE_SECTION,
    KNOWLEDGE_TABLE_BLOCK,
    SYSTEM_PROMPT,
)
from semantic_sql.models.schemas import PromptPayload, TableSchema, VettedExample
from semantic_sql.utils.tokens import count_tokens


def _render_behavioral(examples: list[VettedExample]) -> str:
    if not examples:
        return ""
    blocks = [
        BEHAVIORAL_EXAMPLE_BLOCK.format(question=ex.question, sql_query=ex.sql_query)
        for ex in examples
    ]
    return BEHAVIORAL_SECTION.format(examples="\n".join(blocks))


def _render_knowledge(schemas: list[TableSchema]) -> str:
    if not schemas:
        return ""
    blocks = [
        KNOWLEDGE_TABLE_BLOCK.format(
            table_name=s.table_name,
            row_count=s.row_count or "?",
            ddl=s.ddl,
        )
        for s in schemas
    ]
    return KNOWLEDGE_SECTION.format(schemas="\n".join(blocks))


def _render_executive(question: str) -> str:
    return EXECUTIVE_SECTION.format(question=question)


def build_prompt(payload: PromptPayload) -> tuple[str, str]:
    """Build (system_message, user_message) for the LLM.

    Returns a tuple so callers can pass them as role=system / role=user.
    """
    system = SYSTEM_PROMPT

    user_parts: list[str] = []
    behavioral = _render_behavioral(payload.behavioral_examples)
    if behavioral:
        user_parts.append(behavioral)

    knowledge = _render_knowledge(payload.schema_context)
    if knowledge:
        user_parts.append(knowledge)

    user_parts.append(_render_executive(payload.question))
    user_message = "\n".join(user_parts)

    payload.token_budget_used = count_tokens(system) + count_tokens(user_message)
    return system, user_message

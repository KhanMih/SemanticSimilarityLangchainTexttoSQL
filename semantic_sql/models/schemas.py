"""Pydantic models shared across the project."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class FeedbackStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class TableSchema(BaseModel):
    """Represents one table's DDL + metadata (Knowledge Layer)."""

    table_name: str
    ddl: str
    column_descriptions: dict[str, str] = Field(default_factory=dict)
    sample_rows: list[dict] = Field(default_factory=list)
    row_count: int | None = None


class VettedExample(BaseModel):
    """A human-approved (question, SQL) pair (Behavioral Layer)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    sql_query: str
    explanation: str = ""
    tables_used: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    langfuse_trace_id: str | None = None


class QueryResult(BaseModel):
    """Full result of a single agent invocation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    generated_sql: str = ""
    sql_valid: bool = False
    sql_error: str | None = None
    result_data: list[dict] = Field(default_factory=list)
    llm_answer: str = ""
    few_shot_examples_used: list[VettedExample] = Field(default_factory=list)
    tables_used: list[str] = Field(default_factory=list)
    langfuse_trace_id: str | None = None
    feedback_status: FeedbackStatus = FeedbackStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PromptPayload(BaseModel):
    """Assembled prompt content for the 3-layer architecture."""

    behavioral_examples: list[VettedExample] = Field(default_factory=list)
    schema_context: list[TableSchema] = Field(default_factory=list)
    question: str = ""
    token_budget_used: int = 0

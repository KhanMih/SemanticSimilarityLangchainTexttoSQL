"""Query Pipeline — end-to-end flow for answering a natural-language question.

Orchestrates:
  1. TextToSQLAgent.ask()    — 3-layer prompt → SQL → execute → answer
  2. LangfuseTracer          — logs the full interaction as a Langfuse trace
  3. Returns QueryResult     — ready for display and later annotation
"""

from __future__ import annotations

import logging
from typing import Any

from semantic_sql.agent.sql_agent import TextToSQLAgent
from semantic_sql.config import settings
from semantic_sql.feedback.langfuse_client import LangfuseTracer
from semantic_sql.models.schemas import QueryResult

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Full query lifecycle: question → SQL → data → answer → trace."""

    def __init__(
        self,
        *,
        agent: TextToSQLAgent | None = None,
        tracer: LangfuseTracer | None = None,
        enable_tracing: bool = True,
    ):
        self.agent = agent or TextToSQLAgent()
        self.tracer = tracer

        has_real_key = (
            settings.langfuse_secret_key
            and not settings.langfuse_secret_key.startswith("sk-lf-...")
            and len(settings.langfuse_secret_key) > 15
        )
        self.enable_tracing = enable_tracing and has_real_key

        if self.enable_tracing and self.tracer is None:
            try:
                self.tracer = LangfuseTracer()
            except Exception:
                logger.warning("Langfuse tracing disabled — could not initialise client")
                self.enable_tracing = False

    def run(
        self,
        question: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> QueryResult:
        """Execute the full query pipeline."""
        trace = None
        if self.enable_tracing and self.tracer:
            trace = self.tracer.create_trace(
                name="text-to-sql",
                user_id=user_id,
                session_id=session_id,
                tags=tags or ["semantic-sql"],
                metadata={"question": question},
            )

        result = self.agent.ask(question)

        if trace is not None:
            result.langfuse_trace_id = trace.id
            self._log_trace(trace, result)
            if self.tracer:
                self.tracer.flush()

        return result

    def _log_trace(self, trace: Any, result: QueryResult) -> None:
        """Attach generation details to the Langfuse trace."""
        trace.update(
            input=result.question,
            output=result.llm_answer,
            metadata={
                "question": result.question,
                "generated_sql": result.generated_sql,
                "sql_valid": result.sql_valid,
                "sql_error": result.sql_error,
                "tables_used": result.tables_used,
                "few_shot_count": len(result.few_shot_examples_used),
                "row_count": len(result.result_data),
            },
        )

        trace.generation(
            name="sql-generation",
            input=result.question,
            output=result.generated_sql,
            metadata={
                "few_shot_examples": [
                    {"question": ex.question, "sql": ex.sql_query}
                    for ex in result.few_shot_examples_used
                ],
            },
        )

"""The Text-to-SQL agent — orchestrates the 3-layer architecture.

Flow:
  1. Retrieve few-shot examples   (Behavioral Layer)
  2. Retrieve schema context      (Knowledge Layer)
  3. Build prompt & call LLM      (Executive Layer)
  4. Validate & execute SQL
  5. Generate natural-language answer
"""

from __future__ import annotations

import logging
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from semantic_sql.agent.prompt_builder import build_prompt
from semantic_sql.config import settings
from semantic_sql.db.executor import SQLExecutor
from semantic_sql.db.schema_inspector import SchemaInspector
from semantic_sql.memory.example_store import ExampleStore
from semantic_sql.memory.token_aware import select_examples_within_budget
from semantic_sql.models.schemas import PromptPayload, QueryResult

logger = logging.getLogger(__name__)

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL)


def _clean_sql(raw: str) -> str:
    """Strip markdown fences or stray text the LLM might wrap around the SQL."""
    match = _SQL_FENCE_RE.search(raw)
    if match:
        return match.group(1).strip()
    return raw.strip()


ANSWER_SYSTEM = (
    "You are a helpful data analyst. Given the user's question and the SQL "
    "query results, provide a clear, concise natural-language answer. "
    "Refer to specific numbers and values from the data."
)


class TextToSQLAgent:
    """End-to-end Text-to-SQL agent with dynamic few-shot injection."""

    def __init__(
        self,
        *,
        example_store: ExampleStore | None = None,
        schema_inspector: SchemaInspector | None = None,
        sql_executor: SQLExecutor | None = None,
        llm: ChatGoogleGenerativeAI | None = None,
    ):
        self.example_store = example_store or ExampleStore()
        self.schema_inspector = schema_inspector or SchemaInspector()
        self.sql_executor = sql_executor or SQLExecutor()
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.default_llm_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )

    def ask(self, question: str) -> QueryResult:
        result = QueryResult(question=question)

        # ── Layer 2: Knowledge (schema) ──────────────────
        schemas = self.schema_inspector.get_all_schemas()
        result.tables_used = [s.table_name for s in schemas]

        # ── Layer 1: Behavioral (few-shot) ───────────────
        examples = select_examples_within_budget(
            self.example_store, question, schemas
        )
        result.few_shot_examples_used = examples

        # ── Layer 3: Executive (generate SQL) ────────────
        payload = PromptPayload(
            behavioral_examples=examples,
            schema_context=schemas,
            question=question,
        )
        system_msg, user_msg = build_prompt(payload)

        logger.info(
            "Prompt assembled: %d tokens, %d examples, %d tables",
            payload.token_budget_used,
            len(examples),
            len(schemas),
        )

        raw_sql = self.llm.invoke(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        ).content
        result.generated_sql = _clean_sql(raw_sql)

        # ── Validate & Execute ───────────────────────────
        exec_result = self.sql_executor.execute(result.generated_sql)
        result.sql_valid = exec_result.success
        result.sql_error = exec_result.error
        result.result_data = exec_result.rows[:50]  # cap for display

        # ── Natural-language answer ──────────────────────
        if exec_result.success and exec_result.rows:
            result.llm_answer = self._generate_answer(question, result.generated_sql, exec_result.rows)
        elif exec_result.success:
            result.llm_answer = "The query executed successfully but returned no rows."
        else:
            result.llm_answer = f"SQL error: {exec_result.error}"

        return result

    def _generate_answer(
        self, question: str, sql: str, rows: list[dict]
    ) -> str:
        preview = str(rows[:20])
        user_msg = (
            f"Question: {question}\n\n"
            f"SQL executed:\n{sql}\n\n"
            f"Results ({len(rows)} rows, showing first 20):\n{preview}"
        )
        resp = self.llm.invoke(
            [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
        )
        return resp.content

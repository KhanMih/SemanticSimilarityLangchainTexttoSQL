"""Annotation handler — processes Langfuse feedback into behavioral memory.

Implements the full pipeline:
  Langfuse positive score → SQL validation → deduplication → vector store injection
"""

from __future__ import annotations

import logging
from typing import Any

from semantic_sql.feedback.langfuse_client import FeedbackPoller
from semantic_sql.feedback.validation import SQLValidator
from semantic_sql.memory.deduplication import Deduplicator
from semantic_sql.memory.example_store import ExampleStore
from semantic_sql.models.schemas import FeedbackStatus, VettedExample

logger = logging.getLogger(__name__)


class AnnotationHandler:
    """Processes human feedback from Langfuse into the behavioral memory store."""

    def __init__(
        self,
        *,
        store: ExampleStore | None = None,
        poller: FeedbackPoller | None = None,
        validator: SQLValidator | None = None,
        deduplicator: Deduplicator | None = None,
    ):
        self.store = store or ExampleStore()
        self.poller = poller or FeedbackPoller()
        self.validator = validator or SQLValidator()
        self.deduplicator = deduplicator or Deduplicator(self.store)

    def process_feedback(self, traces: list[dict[str, Any]] | None = None) -> dict[str, int]:
        """Process positive traces from Langfuse and inject approved ones into memory.

        Returns a summary dict: {approved, rejected_validation, rejected_duplicate, skipped}.
        """
        if traces is None:
            traces = self.poller.fetch_positive_traces()

        stats = {"approved": 0, "rejected_validation": 0, "rejected_duplicate": 0, "skipped": 0}

        for trace in traces:
            question = trace.get("question", "")
            sql = trace.get("generated_sql", "")
            trace_id = trace.get("trace_id", "")

            if not question or not sql:
                stats["skipped"] += 1
                logger.warning("Skipping trace %s: missing question or SQL", trace_id)
                continue

            # Gate 1: Automated SQL validation
            validation = self.validator.validate(sql)
            if not validation.is_valid:
                stats["rejected_validation"] += 1
                logger.info(
                    "Rejected trace %s (validation): %s", trace_id, validation.reason
                )
                continue

            # Gate 2: Semantic de-duplication
            example = VettedExample(
                question=question,
                sql_query=sql,
                explanation=trace.get("score_comment", ""),
                langfuse_trace_id=trace_id,
            )

            was_added = self.deduplicator.add_if_unique(example)
            if was_added:
                stats["approved"] += 1
                logger.info("Approved and injected trace %s into memory", trace_id)
            else:
                stats["rejected_duplicate"] += 1

        logger.info("Feedback processing complete: %s", stats)
        return stats

    def manually_add_example(
        self,
        question: str,
        sql_query: str,
        *,
        explanation: str = "",
        tables_used: list[str] | None = None,
        skip_validation: bool = False,
    ) -> tuple[bool, str]:
        """Manually add a vetted example (bypass Langfuse).

        Returns (success, message).
        """
        if not skip_validation:
            validation = self.validator.validate(sql_query)
            if not validation.is_valid:
                return False, f"Validation failed: {validation.reason}"

        example = VettedExample(
            question=question,
            sql_query=sql_query,
            explanation=explanation,
            tables_used=tables_used or [],
        )

        was_added = self.deduplicator.add_if_unique(example)
        if was_added:
            return True, "Example added to behavioral memory."
        return False, "Example too similar to existing entry (duplicate)."

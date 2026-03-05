"""Langfuse integration — tracing and feedback retrieval.

Handles two directions:
  1. Outbound: wrapping agent calls with Langfuse traces (observability)
  2. Inbound:  polling Langfuse for new positive annotations (feedback loop)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from langfuse import Langfuse

from semantic_sql.config import settings

logger = logging.getLogger(__name__)


def get_langfuse_client() -> Langfuse:
    return Langfuse(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_host,
    )


class LangfuseTracer:
    """Wraps agent invocations with Langfuse traces for full observability."""

    def __init__(self, client: Langfuse | None = None):
        self.client = client or get_langfuse_client()

    def create_trace(
        self,
        name: str = "text-to-sql",
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        return self.client.trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            tags=tags or ["semantic-sql"],
        )

    def score_trace(
        self,
        trace_id: str,
        *,
        name: str | None = None,
        value: float = 1.0,
        comment: str = "",
    ):
        self.client.score(
            trace_id=trace_id,
            name=name or settings.feedback_score_name,
            value=value,
            comment=comment,
        )

    def flush(self):
        self.client.flush()


class FeedbackPoller:
    """Polls Langfuse for traces that received positive human annotations.

    Designed to run periodically (e.g. via the feedback pipeline) to discover
    new expert-approved interactions and inject them into the example store.
    """

    def __init__(self, client: Langfuse | None = None):
        self.client = client or get_langfuse_client()

    def fetch_positive_traces(
        self,
        *,
        since: datetime | None = None,
        limit: int = 100,
        tag: str = "semantic-sql",
    ) -> list[dict[str, Any]]:
        """Fetch traces that have a positive quality score.

        Returns list of dicts with keys: trace_id, question, generated_sql,
        score_value, score_comment.
        """
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(days=7)

        traces_response = self.client.fetch_traces(
            limit=limit,
            tags=[tag],
        )
        traces = traces_response.data if hasattr(traces_response, "data") else traces_response

        approved: list[dict[str, Any]] = []
        for trace in traces:
            trace_id = trace.id
            scores = self.client.fetch_scores(trace_id=trace_id)
            score_list = scores.data if hasattr(scores, "data") else scores

            for score in score_list:
                if (
                    score.name == settings.feedback_score_name
                    and score.value is not None
                    and score.value >= settings.feedback_positive_threshold
                ):
                    metadata = trace.metadata or {}
                    approved.append(
                        {
                            "trace_id": trace_id,
                            "question": metadata.get("question", trace.input or ""),
                            "generated_sql": metadata.get("generated_sql", ""),
                            "score_value": score.value,
                            "score_comment": score.comment or "",
                        }
                    )
                    break

        logger.info("Found %d positively-scored traces since %s", len(approved), since)
        return approved

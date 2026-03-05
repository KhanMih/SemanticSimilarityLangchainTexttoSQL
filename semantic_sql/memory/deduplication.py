"""Semantic de-duplication — "Check-before-Add" logic.

Before adding a new vetted example to memory we check whether a near-
identical example already exists.  This keeps the store lean and prevents
wasting context-window tokens on redundant information.
"""

from __future__ import annotations

import logging

from semantic_sql.config import settings
from semantic_sql.memory.example_store import ExampleStore
from semantic_sql.models.schemas import VettedExample

logger = logging.getLogger(__name__)


class Deduplicator:
    """Gate that prevents near-duplicate examples from entering the store."""

    def __init__(
        self,
        store: ExampleStore,
        threshold: float | None = None,
    ):
        self.store = store
        self.threshold = threshold or settings.similarity_dedup_threshold

    def is_duplicate(self, example: VettedExample) -> tuple[bool, float]:
        """Check if an example is semantically too close to an existing one.

        Returns (is_dup, similarity_score).
        """
        docs = self.store.vectorstore.similarity_search_with_score(
            example.question, k=1
        )
        if not docs:
            return False, 0.0

        doc, score = docs[0]
        # pgvector returns *distance* (lower = more similar) for L2,
        # but cosine stores return similarity (higher = more similar).
        # We normalise: treat score > threshold as duplicate.
        is_dup = score >= self.threshold
        if is_dup:
            logger.info(
                "Duplicate detected (%.3f >= %.3f): '%s' ~ '%s'",
                score,
                self.threshold,
                example.question[:60],
                doc.page_content[:60],
            )
        return is_dup, float(score)

    def add_if_unique(self, example: VettedExample) -> bool:
        """Add example only if it passes the de-duplication gate.

        Returns True if the example was added, False if it was a duplicate.
        """
        is_dup, score = self.is_duplicate(example)
        if is_dup:
            logger.info("Skipping duplicate example (score=%.3f)", score)
            return False
        self.store.add_example(example)
        return True

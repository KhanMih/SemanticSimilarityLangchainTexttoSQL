"""Feedback Pipeline — continuous agent evolution loop.

Runs periodically (or on-demand) to:
  1. Poll Langfuse for new positive annotations
  2. Validate the SQL in each approved trace
  3. De-duplicate against existing memory
  4. Inject approved examples into the vector store
"""

from __future__ import annotations

import logging
import time

from semantic_sql.config import settings
from semantic_sql.feedback.annotation_handler import AnnotationHandler

logger = logging.getLogger(__name__)


class FeedbackPipeline:
    """Processes human feedback into behavioral memory."""

    def __init__(self, handler: AnnotationHandler | None = None):
        self.handler = handler or AnnotationHandler()

    def run_once(self) -> dict[str, int]:
        """Single pass: poll → validate → dedupe → inject."""
        logger.info("Running feedback pipeline (single pass)...")
        stats = self.handler.process_feedback()
        logger.info("Feedback pipeline complete: %s", stats)
        return stats

    def run_loop(self, *, max_iterations: int | None = None) -> None:
        """Continuous polling loop.  Runs until interrupted or max_iterations reached."""
        interval = settings.feedback_poll_interval
        iteration = 0
        logger.info(
            "Starting feedback loop (interval=%ds, max_iterations=%s)",
            interval,
            max_iterations or "∞",
        )

        try:
            while True:
                iteration += 1
                if max_iterations and iteration > max_iterations:
                    break

                try:
                    stats = self.run_once()
                    logger.info("Iteration %d: %s", iteration, stats)
                except Exception:
                    logger.exception("Error in feedback iteration %d", iteration)

                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Feedback loop stopped by user after %d iterations", iteration)

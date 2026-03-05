"""SQL validation — ensures only correct queries enter the behavioral memory.

Even if an SME tags a trace as "positive", we run automated checks before
allowing it into the example store.  This prevents accidental bad tags from
poisoning memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from semantic_sql.db.executor import SQLExecutor

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str = ""


class SQLValidator:
    """Multi-check validation gate for candidate examples."""

    def __init__(self, executor: SQLExecutor | None = None):
        self.executor = executor or SQLExecutor()

    def validate(self, sql: str) -> ValidationResult:
        """Run all validation checks in sequence. Fail fast on first error."""
        checks = [
            self._check_not_empty,
            self._check_no_writes,
            self._check_parseable,
        ]
        for check in checks:
            result = check(sql)
            if not result.is_valid:
                return result
        return ValidationResult(is_valid=True, reason="All checks passed")

    def _check_not_empty(self, sql: str) -> ValidationResult:
        if not sql or not sql.strip():
            return ValidationResult(is_valid=False, reason="SQL is empty")
        return ValidationResult(is_valid=True)

    def _check_no_writes(self, sql: str) -> ValidationResult:
        valid, err = self.executor.validate_sql(sql)
        if not valid:
            return ValidationResult(is_valid=False, reason=err or "Write operation detected")
        return ValidationResult(is_valid=True)

    def _check_parseable(self, sql: str) -> ValidationResult:
        """Use EXPLAIN to verify the query is syntactically valid."""
        result = self.executor.dry_run(sql)
        if not result.success:
            return ValidationResult(
                is_valid=False,
                reason=f"EXPLAIN failed: {result.error}",
            )
        return ValidationResult(is_valid=True)

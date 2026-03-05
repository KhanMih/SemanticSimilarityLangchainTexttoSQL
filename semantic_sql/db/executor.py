"""Safe, read-only SQL execution against the business database."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.engine import Engine

from semantic_sql.config import settings
from semantic_sql.db.connection import get_engine

_DANGEROUS_PATTERNS = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


@dataclass
class ExecutionResult:
    success: bool
    rows: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    error: str | None = None


class SQLExecutor:
    """Executes SQL in a read-only, time-limited context."""

    def __init__(self, engine: Engine | None = None, *, read_only: bool = True):
        self.engine = engine or get_engine()
        self.read_only = read_only
        self.timeout = settings.sql_execution_timeout

    def validate_sql(self, sql: str) -> tuple[bool, str | None]:
        """Reject obviously dangerous statements before execution."""
        stripped = sql.strip().rstrip(";")
        if self.read_only and _DANGEROUS_PATTERNS.search(stripped):
            return False, "Write operations are not allowed in read-only mode."
        return True, None

    def execute(self, sql: str) -> ExecutionResult:
        valid, err = self.validate_sql(sql)
        if not valid:
            return ExecutionResult(success=False, error=err)

        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(f"SET statement_timeout = '{self.timeout}s'")
                )
                result = conn.execute(text(sql))
                rows = [dict(r._mapping) for r in result]
                columns = list(result.keys()) if result.returns_rows else []
                return ExecutionResult(
                    success=True,
                    rows=rows,
                    columns=columns,
                    row_count=len(rows),
                )
        except Exception as exc:
            return ExecutionResult(success=False, error=str(exc))

    def dry_run(self, sql: str) -> ExecutionResult:
        """Validate SQL via EXPLAIN without returning data."""
        valid, err = self.validate_sql(sql)
        if not valid:
            return ExecutionResult(success=False, error=err)
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"EXPLAIN {sql}"))
                conn.rollback()
            return ExecutionResult(success=True)
        except Exception as exc:
            return ExecutionResult(success=False, error=str(exc))

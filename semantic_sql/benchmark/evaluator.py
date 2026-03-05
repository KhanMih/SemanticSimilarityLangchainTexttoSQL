"""Evaluation engine — computes metrics by comparing predicted vs gold SQL.

Metrics:
  1. Execution Accuracy (EX) — do both queries return the same result set?
  2. Valid SQL Rate (VR)     — does the predicted SQL execute without errors?
  3. Exact Match (EM)        — are the SQL strings identical after normalisation?
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class ExampleScore:
    question_id: int
    question: str
    gold_sql: str
    predicted_sql: str
    execution_match: bool = False
    valid_sql: bool = False
    exact_match: bool = False
    error: str | None = None
    difficulty: str = "unknown"


@dataclass
class BenchmarkScores:
    """Aggregated scores for one strategy across all examples."""

    strategy_name: str
    scores: list[ExampleScore] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.scores)

    @property
    def execution_accuracy(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.execution_match) / self.total

    @property
    def valid_sql_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.valid_sql) / self.total

    @property
    def exact_match_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.exact_match) / self.total

    def by_difficulty(self) -> dict[str, dict[str, float]]:
        """Break down execution accuracy by difficulty level."""
        groups: dict[str, list[ExampleScore]] = {}
        for s in self.scores:
            groups.setdefault(s.difficulty, []).append(s)

        result = {}
        for diff, items in sorted(groups.items()):
            n = len(items)
            ex = sum(1 for s in items if s.execution_match) / n if n else 0
            result[diff] = {"count": n, "execution_accuracy": round(ex, 4)}
        return result

    def summary(self) -> dict[str, float | int]:
        return {
            "strategy": self.strategy_name,
            "total": self.total,
            "execution_accuracy": round(self.execution_accuracy, 4),
            "valid_sql_rate": round(self.valid_sql_rate, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
        }


class SQLEvaluator:
    """Compares predicted SQL against gold SQL using multiple metrics."""

    def __init__(self, engine: Engine, timeout: int = 30):
        self.engine = engine
        self.timeout = timeout

    def evaluate(
        self,
        question_id: int,
        question: str,
        gold_sql: str,
        predicted_sql: str,
        difficulty: str = "unknown",
    ) -> ExampleScore:
        score = ExampleScore(
            question_id=question_id,
            question=question,
            gold_sql=gold_sql,
            predicted_sql=predicted_sql,
            difficulty=difficulty,
        )

        if not predicted_sql or not predicted_sql.strip():
            score.error = "Empty prediction"
            return score

        score.exact_match = _normalize_sql(gold_sql) == _normalize_sql(predicted_sql)

        gold_result = self._safe_execute(gold_sql)
        if gold_result is None:
            score.error = "Gold SQL failed to execute"
            return score

        pred_result = self._safe_execute(predicted_sql)
        if pred_result is None:
            score.valid_sql = False
            score.error = "Predicted SQL failed to execute"
            return score

        score.valid_sql = True
        score.execution_match = _results_match(gold_result, pred_result)

        return score

    def _safe_execute(self, sql: str) -> list[tuple] | None:
        """Execute SQL and return sorted result rows, or None on failure."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"SET statement_timeout = '{self.timeout}s'"))
                result = conn.execute(text(sql))
                rows = [tuple(row) for row in result.fetchall()]
                return rows
        except Exception as exc:
            logger.debug("SQL execution failed: %s — %s", sql[:80], exc)
            return None


def _normalize_sql(sql: str) -> str:
    """Normalise SQL for exact-match comparison."""
    s = sql.strip().rstrip(";").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s


def _results_match(gold: list[tuple], predicted: list[tuple]) -> bool:
    """Compare two result sets for equality (order-insensitive).

    Handles:
      - Different row ordering (sorts both)
      - Float precision (rounds to 4 decimal places)
    """
    def normalize_row(row: tuple) -> tuple:
        return tuple(
            round(v, 4) if isinstance(v, float) else v
            for v in row
        )

    gold_norm = sorted(normalize_row(r) for r in gold)
    pred_norm = sorted(normalize_row(r) for r in predicted)

    if gold_norm == pred_norm:
        return True

    # Also check if column count differs but values match
    # (e.g. gold returns (a, b) but pred returns (a, b, c))
    if len(gold_norm) > 0 and len(pred_norm) > 0:
        gold_cols = len(gold_norm[0]) if gold_norm else 0
        pred_cols = len(pred_norm[0]) if pred_norm else 0
        if gold_cols != pred_cols:
            return False

    return False

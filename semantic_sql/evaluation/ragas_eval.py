"""RAGAS-based evaluation for the semantic-sql agent.

Implements the evaluation framework from:
  https://docs.ragas.io/en/stable/howtos/applications/text2sql/

Two-phase comparison:
  Phase 1 — Baseline (zero-shot): No examples, agent relies only on schema.
  Phase 2 — With learning (dynamic few-shot): Agent retrieves semantically
            similar vetted examples from the vector store.

Evaluation method:
  For every question we evaluate the **final LLM response** — the natural-
  language answer the user would see — NOT the raw SQL or result sets.

  1. Generate SQL via the chosen strategy
  2. Execute both gold and predicted SQL
  3. Generate a natural-language answer from each result set
  4. An LLM judge compares the two answers for factual equivalence
  5. The judge verdict determines correctness

Usage:
  from semantic_sql.evaluation.ragas_eval import RAGASEvalRunner
  runner = RAGASEvalRunner()
  report = runner.run()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.table import Table
from sqlalchemy import text
from sqlalchemy.engine import Engine

from semantic_sql.benchmark.strategies import (
    DynamicFewShotStrategy,
    ZeroShotStrategy,
)
from semantic_sql.config import settings
from semantic_sql.db.connection import get_engine
from semantic_sql.db.schema_inspector import SchemaInspector
from semantic_sql.evaluation.ground_truth import EVALUATION_QUESTIONS
from semantic_sql.evaluation.learning_examples import get_learning_examples
from semantic_sql.memory.example_store import ExampleStore
from semantic_sql.models.schemas import TableSchema

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# RAGAS availability check (used for reporting only — we always use our own
# response-level evaluation)
# ---------------------------------------------------------------------------

try:
    from ragas.metrics import MetricResult, discrete_metric  # noqa: F401

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning(
        "ragas package not installed — install with: pip install ragas"
    )


# ---------------------------------------------------------------------------
# Safe SQL execution
# ---------------------------------------------------------------------------

def _safe_execute(engine: Engine, sql: str, timeout: int = 30) -> tuple[bool, list[tuple] | None]:
    """Execute SQL and return (success, rows)."""
    try:
        with engine.connect() as conn:
            conn.execute(text(f"SET statement_timeout = '{timeout}s'"))
            result = conn.execute(text(sql))
            rows = [tuple(row) for row in result.fetchall()]
            return True, rows
    except Exception as exc:
        logger.debug("SQL execution failed: %s — %s", sql[:80], exc)
        return False, None


# ---------------------------------------------------------------------------
# LLM-based answer generation & comparison
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM = (
    "You are a data analyst. Given a question and SQL query results, provide a "
    "clear, concise natural-language answer. Include specific numbers and values. "
    "If the query returned no results, say so explicitly."
)

_JUDGE_SYSTEM = (
    "You are an impartial judge evaluating two data analysis answers to the same "
    "question. Determine whether both answers convey the SAME factual information.\n\n"
    "Rules:\n"
    "- Minor differences in wording or formatting are OK\n"
    "- Rounding differences within 1% are OK (e.g., '$599.99' vs '$600.00')\n"
    "- Extra detail in one answer is OK as long as the core facts match\n"
    "- The answers must agree on the KEY facts: same entities, same quantities "
    "(within rounding), same rankings\n"
    "- If one answer has an error or 'no results' and the other has data, "
    "they are DIFFERENT\n\n"
    "Respond with EXACTLY one line:\n"
    "EQUIVALENT: <brief reason>\n"
    "or\n"
    "DIFFERENT: <brief reason>"
)


def _format_results_preview(rows: list[tuple] | None) -> str:
    if rows is None:
        return "Query failed to execute."
    if not rows:
        return "Query returned no results."
    lines = [str(row) for row in rows[:25]]
    out = "\n".join(lines)
    if len(rows) > 25:
        out += f"\n... and {len(rows) - 25} more rows"
    return out


# ---------------------------------------------------------------------------
# Per-question evaluation result
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    question_id: int
    question: str
    difficulty: str
    gold_sql: str
    predicted_sql: str
    execution_accuracy: str  # "correct" | "incorrect"
    reason: str
    valid_sql: bool
    gold_answer: str = ""
    predicted_answer: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Phase report (one phase = one strategy)
# ---------------------------------------------------------------------------

@dataclass
class PhaseReport:
    phase_name: str
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.execution_accuracy == "correct")

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def valid_sql_rate(self) -> float:
        return sum(1 for r in self.results if r.valid_sql) / self.total if self.total else 0.0

    def by_difficulty(self) -> dict[str, dict[str, float | int]]:
        groups: dict[str, list[QuestionResult]] = {}
        for r in self.results:
            groups.setdefault(r.difficulty, []).append(r)
        out = {}
        for diff, items in sorted(groups.items()):
            n = len(items)
            c = sum(1 for r in items if r.execution_accuracy == "correct")
            out[diff] = {"count": n, "correct": c, "accuracy": round(c / n, 4) if n else 0}
        return out


# ---------------------------------------------------------------------------
# Full evaluation report (baseline vs learned)
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    baseline: PhaseReport
    with_learning: PhaseReport
    elapsed_seconds: float = 0.0

    @property
    def improvement(self) -> float:
        return self.with_learning.accuracy - self.baseline.accuracy

    @property
    def improvement_pct_points(self) -> float:
        return round(self.improvement * 100, 1)

    def to_dict(self) -> dict:
        return {
            "evaluation_method": "response_comparison",
            "ragas_installed": RAGAS_AVAILABLE,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "summary": {
                "baseline_accuracy": round(self.baseline.accuracy, 4),
                "with_learning_accuracy": round(self.with_learning.accuracy, 4),
                "improvement_pct_points": self.improvement_pct_points,
                "total_questions": self.baseline.total,
            },
            "baseline": {
                "phase": self.baseline.phase_name,
                "accuracy": round(self.baseline.accuracy, 4),
                "valid_sql_rate": round(self.baseline.valid_sql_rate, 4),
                "by_difficulty": self.baseline.by_difficulty(),
                "details": [
                    {
                        "question_id": r.question_id,
                        "question": r.question,
                        "difficulty": r.difficulty,
                        "gold_sql": r.gold_sql,
                        "predicted_sql": r.predicted_sql,
                        "execution_accuracy": r.execution_accuracy,
                        "gold_answer": r.gold_answer,
                        "predicted_answer": r.predicted_answer,
                        "reason": r.reason,
                    }
                    for r in self.baseline.results
                ],
            },
            "with_learning": {
                "phase": self.with_learning.phase_name,
                "accuracy": round(self.with_learning.accuracy, 4),
                "valid_sql_rate": round(self.with_learning.valid_sql_rate, 4),
                "by_difficulty": self.with_learning.by_difficulty(),
                "details": [
                    {
                        "question_id": r.question_id,
                        "question": r.question,
                        "difficulty": r.difficulty,
                        "gold_sql": r.gold_sql,
                        "predicted_sql": r.predicted_sql,
                        "execution_accuracy": r.execution_accuracy,
                        "gold_answer": r.gold_answer,
                        "predicted_answer": r.predicted_answer,
                        "reason": r.reason,
                    }
                    for r in self.with_learning.results
                ],
            },
        }

    def to_json(self, path: str | None = None) -> str:
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=str)
        if path:
            Path(path).write_text(json_str)
            logger.info("Report saved to %s", path)
        return json_str


# ---------------------------------------------------------------------------
# RAGAS Evaluation Runner
# ---------------------------------------------------------------------------

EVAL_COLLECTION = "ragas_eval_examples"


class RAGASEvalRunner:
    """Runs the two-phase evaluation and produces a comparison report.

    Phase 1 — Baseline (zero-shot):
        Agent receives only the database schema, no few-shot examples.

    Phase 2 — With learning (dynamic few-shot):
        Vetted learning examples are loaded into a separate pgvector collection.
        The agent retrieves the most similar examples for each question.

    Evaluation compares the **final LLM-generated response** for every
    question — the natural-language answer the user would see — not the raw
    SQL query or result sets.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        timeout: int = 30,
        limit: int | None = None,
    ):
        self.engine = engine or get_engine()
        self.timeout = timeout
        self.limit = limit
        self.inspector = SchemaInspector(self.engine)
        self.llm = ChatGoogleGenerativeAI(
            model=settings.default_llm_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )

    def run(self, output_file: str | None = None) -> EvaluationReport:
        """Execute the full two-phase evaluation."""
        start = time.time()
        schemas = self.inspector.get_all_schemas()
        questions = EVALUATION_QUESTIONS
        if self.limit:
            questions = questions[: self.limit]

        console.print(
            f"\n[bold]RAGAS Text-to-SQL Evaluation[/bold]  "
            f"({len(questions)} questions, "
            f"{'RAGAS installed' if RAGAS_AVAILABLE else 'RAGAS not installed'})\n"
            f"[dim]Evaluation method: comparing final LLM responses (not SQL)[/dim]\n"
        )

        # ── Phase 1: Baseline ──────────────────────────────────
        console.print("[bold blue]Phase 1: Baseline (zero-shot — no examples)[/bold blue]")
        baseline = self._run_phase(
            phase_name="zero_shot_baseline",
            strategy=ZeroShotStrategy(),
            questions=questions,
            schemas=schemas,
        )
        console.print(
            f"  Response Accuracy: [{'green' if baseline.accuracy > 0.5 else 'yellow'}]"
            f"{baseline.accuracy:.1%}[/]  "
            f"({baseline.correct}/{baseline.total})\n"
        )

        # ── Phase 2: With learning ────────────────────────────
        console.print("[bold blue]Phase 2: With learning (dynamic few-shot)[/bold blue]")
        store = self._prepare_learning_store()
        with_learning = self._run_phase(
            phase_name="dynamic_few_shot_learned",
            strategy=DynamicFewShotStrategy(store=store, k=3),
            questions=questions,
            schemas=schemas,
        )
        console.print(
            f"  Response Accuracy: [{'green' if with_learning.accuracy > 0.5 else 'yellow'}]"
            f"{with_learning.accuracy:.1%}[/]  "
            f"({with_learning.correct}/{with_learning.total})\n"
        )

        elapsed = time.time() - start
        report = EvaluationReport(
            baseline=baseline,
            with_learning=with_learning,
            elapsed_seconds=elapsed,
        )

        self._print_comparison(report)

        if output_file:
            report.to_json(output_file)
            console.print(f"\n[dim]Report saved to {output_file}[/dim]")

        return report

    # ------------------------------------------------------------------
    # LLM answer generation & judging
    # ------------------------------------------------------------------

    def _generate_answer(
        self, question: str, sql: str, rows: list[tuple] | None, success: bool,
    ) -> str:
        """Generate a natural-language answer from SQL results.

        This simulates what the real TextToSQLAgent does after SQL execution.
        """
        if not success:
            return "The query failed to execute."
        if rows is not None and not rows:
            return "The query returned no results."

        results_text = _format_results_preview(rows)
        user_msg = (
            f"Question: {question}\n\n"
            f"SQL query:\n{sql}\n\n"
            f"Query results ({len(rows or [])} rows):\n{results_text}\n\n"
            f"Provide a clear, concise answer to the question based on these results."
        )
        try:
            resp = self.llm.invoke([
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user", "content": user_msg},
            ])
            return resp.content.strip()
        except Exception as exc:
            logger.warning("Answer generation failed: %s", exc)
            return f"[Answer generation failed: {exc}]"

    def _judge_answers(
        self, question: str, gold_answer: str, pred_answer: str,
    ) -> tuple[bool, str]:
        """Use LLM as judge to determine if two answers are equivalent."""
        user_msg = (
            f"Question: {question}\n\n"
            f"Reference answer:\n{gold_answer}\n\n"
            f"Predicted answer:\n{pred_answer}\n\n"
            f"Are these answers equivalent?"
        )
        try:
            resp = self.llm.invoke([
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ])
            verdict = resp.content.strip()
            is_equivalent = verdict.upper().startswith("EQUIVALENT")
            return is_equivalent, verdict
        except Exception as exc:
            logger.warning("Judge call failed: %s", exc)
            return False, f"Judge failed: {exc}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_learning_store(self) -> ExampleStore:
        """Populate a fresh vector store collection with learning examples."""
        store = ExampleStore(collection_name=EVAL_COLLECTION)
        examples = get_learning_examples()
        console.print(f"  Loading {len(examples)} learning examples into vector store...")
        for ex in examples:
            store.add_example(ex)
        console.print(f"  [green]✓[/green] {len(examples)} examples loaded\n")
        return store

    def _run_phase(
        self,
        phase_name: str,
        strategy: ZeroShotStrategy | DynamicFewShotStrategy,
        questions: list[dict],
        schemas: list[TableSchema],
    ) -> PhaseReport:
        """Run one evaluation phase against all questions.

        For EVERY question:
          1. Generate SQL via strategy
          2. Execute gold SQL and predicted SQL
          3. Generate NL response from each result set
          4. LLM judge compares the two responses
          5. Judge verdict determines correctness
        """
        phase = PhaseReport(phase_name=phase_name)

        for i, q in enumerate(questions):
            result = strategy.generate_sql(q["question"], schemas)
            predicted_sql = result.generated_sql

            gold_success, gold_result = _safe_execute(
                self.engine, q["gold_sql"], self.timeout,
            )
            pred_success, pred_result = _safe_execute(
                self.engine, predicted_sql, self.timeout,
            )

            # ── Generate NL responses (what the user would see) ──
            gold_answer = self._generate_answer(
                q["question"], q["gold_sql"], gold_result, gold_success,
            )
            pred_answer = self._generate_answer(
                q["question"], predicted_sql, pred_result, pred_success,
            )

            # ── Determine correctness via LLM judge ──
            if not pred_success:
                acc_value = "incorrect"
                acc_reason = "Predicted SQL failed to execute"
            elif not gold_success:
                acc_value = "incorrect"
                acc_reason = "Gold SQL failed to execute"
            else:
                is_equiv, verdict = self._judge_answers(
                    q["question"], gold_answer, pred_answer,
                )
                if is_equiv:
                    acc_value = "correct"
                    acc_reason = f"Responses equivalent: {verdict}"
                else:
                    acc_value = "incorrect"
                    acc_reason = f"Responses differ: {verdict}"

            qr = QuestionResult(
                question_id=i,
                question=q["question"],
                difficulty=q["difficulty"],
                gold_sql=q["gold_sql"],
                predicted_sql=predicted_sql,
                execution_accuracy=acc_value,
                reason=acc_reason,
                valid_sql=pred_success,
                gold_answer=gold_answer,
                predicted_answer=pred_answer,
                error=None if pred_success else "SQL execution failed",
            )
            phase.results.append(qr)

            if (i + 1) % 10 == 0 or (i + 1) == len(questions):
                running = phase.correct / (i + 1)
                console.print(
                    f"  [{i + 1}/{len(questions)}] running accuracy: {running:.1%}",
                    style="dim",
                )

        return phase

    def _print_comparison(self, report: EvaluationReport) -> None:
        """Print a rich comparison table."""
        console.print()

        # ── Summary table ──
        table = Table(
            title="Evaluation — Baseline vs With Learning (Response Comparison)",
            show_lines=True,
        )
        table.add_column("Phase", style="bold")
        table.add_column("Response Accuracy", justify="right")
        table.add_column("Valid SQL Rate", justify="right")
        table.add_column("Correct / Total", justify="right")

        baseline_acc = f"{report.baseline.accuracy:.1%}"
        learned_acc = f"{report.with_learning.accuracy:.1%}"
        if report.with_learning.accuracy > report.baseline.accuracy:
            learned_acc = f"[bold green]{learned_acc}[/bold green]"

        table.add_row(
            "Baseline (zero-shot)",
            baseline_acc,
            f"{report.baseline.valid_sql_rate:.1%}",
            f"{report.baseline.correct}/{report.baseline.total}",
        )
        table.add_row(
            "With Learning (dynamic few-shot)",
            learned_acc,
            f"{report.with_learning.valid_sql_rate:.1%}",
            f"{report.with_learning.correct}/{report.with_learning.total}",
        )
        console.print(table)

        # ── Improvement banner ──
        imp = report.improvement_pct_points
        color = "green" if imp > 0 else ("yellow" if imp == 0 else "red")
        console.print(
            f"\n  [bold {color}]Improvement: {'+' if imp > 0 else ''}{imp} percentage points[/bold {color}]"
        )

        # ── Difficulty breakdown ──
        diff_table = Table(title="Accuracy by Difficulty", show_lines=True)
        diff_table.add_column("Difficulty", style="bold")
        diff_table.add_column("Baseline", justify="right")
        diff_table.add_column("With Learning", justify="right")
        diff_table.add_column("Delta", justify="right")

        baseline_diff = report.baseline.by_difficulty()
        learned_diff = report.with_learning.by_difficulty()

        for diff in ["simple", "moderate", "challenging"]:
            b = baseline_diff.get(diff, {"accuracy": 0, "count": 0})
            l = learned_diff.get(diff, {"accuracy": 0, "count": 0})
            delta = l["accuracy"] - b["accuracy"]
            delta_str = f"{'+' if delta > 0 else ''}{delta * 100:.1f}pp"
            delta_color = "green" if delta > 0 else ("dim" if delta == 0 else "red")

            diff_table.add_row(
                f"{diff} ({b['count']})",
                f"{b['accuracy']:.1%}",
                f"{l['accuracy']:.1%}",
                f"[{delta_color}]{delta_str}[/{delta_color}]",
            )

        console.print(diff_table)

        # ── Questions that flipped from incorrect → correct ──
        flipped = []
        for b_r, l_r in zip(report.baseline.results, report.with_learning.results):
            if b_r.execution_accuracy == "incorrect" and l_r.execution_accuracy == "correct":
                flipped.append(l_r)

        if flipped:
            console.print(
                f"\n[bold green]Questions fixed by learning ({len(flipped)}):[/bold green]"
            )
            for r in flipped:
                console.print(f"  [green]✓[/green] [{r.difficulty}] {r.question}")

        # ── Questions that regressed (correct → incorrect) ──
        regressed = []
        for b_r, l_r in zip(report.baseline.results, report.with_learning.results):
            if b_r.execution_accuracy == "correct" and l_r.execution_accuracy == "incorrect":
                regressed.append(l_r)

        if regressed:
            console.print(
                f"\n[bold yellow]Regressions ({len(regressed)}):[/bold yellow]"
            )
            for r in regressed:
                console.print(f"  [yellow]✗[/yellow] [{r.difficulty}] {r.question}")

        console.print(f"\n[dim]Total evaluation time: {report.elapsed_seconds:.1f}s[/dim]")
        console.print("[dim]Evaluation: LLM response comparison (not SQL result sets)[/dim]")
        if RAGAS_AVAILABLE:
            console.print("[dim]RAGAS installed[/dim]")

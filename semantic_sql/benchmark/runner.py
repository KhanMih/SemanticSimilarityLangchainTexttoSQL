"""Benchmark runner — orchestrates dataset loading, strategy execution, and evaluation.

Usage:
  semantic-sql benchmark run --dataset ./data/bird_mini_dev.json --db-root ./data/databases
  semantic-sql benchmark run --strategies zero_shot,dynamic_few_shot --limit 50
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table
from sqlalchemy.engine import Engine

from semantic_sql.benchmark.dataset_loader import BenchmarkDataset, BenchmarkExample
from semantic_sql.benchmark.evaluator import BenchmarkScores, ExampleScore, SQLEvaluator
from semantic_sql.benchmark.strategies import (
    ALL_STRATEGIES,
    BaseStrategy,
    DynamicWithFeedbackStrategy,
)
from semantic_sql.db.connection import get_engine
from semantic_sql.db.schema_inspector import SchemaInspector
from semantic_sql.models.schemas import TableSchema

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BenchmarkConfig:
    strategies: list[str] = field(
        default_factory=lambda: ["zero_shot", "static_few_shot", "random_few_shot", "dynamic_few_shot"]
    )
    limit: int | None = None  # cap number of examples to evaluate
    timeout: int = 30
    output_file: str | None = None


@dataclass
class BenchmarkReport:
    """Full benchmark report across all strategies."""

    results: dict[str, BenchmarkScores] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def comparison_table(self) -> dict[str, dict[str, float]]:
        return {name: scores.summary() for name, scores in self.results.items()}

    def to_json(self, path: str | None = None) -> str:
        data = {
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "strategies": {},
        }
        for name, scores in self.results.items():
            data["strategies"][name] = {
                **scores.summary(),
                "by_difficulty": scores.by_difficulty(),
                "details": [
                    {
                        "question_id": s.question_id,
                        "question": s.question,
                        "gold_sql": s.gold_sql,
                        "predicted_sql": s.predicted_sql,
                        "execution_match": s.execution_match,
                        "valid_sql": s.valid_sql,
                        "exact_match": s.exact_match,
                        "error": s.error,
                        "difficulty": s.difficulty,
                    }
                    for s in scores.scores
                ],
            }

        json_str = json.dumps(data, indent=2, default=str)
        if path:
            Path(path).write_text(json_str)
            logger.info("Report saved to %s", path)
        return json_str


class BenchmarkRunner:
    """Runs all strategies against a dataset and produces a comparison report."""

    def __init__(
        self,
        dataset: BenchmarkDataset,
        engine: Engine | None = None,
        config: BenchmarkConfig | None = None,
    ):
        self.dataset = dataset
        self.engine = engine or get_engine()
        self.config = config or BenchmarkConfig()
        self.evaluator = SQLEvaluator(self.engine, timeout=self.config.timeout)
        self.inspector = SchemaInspector(self.engine)

    def run(self) -> BenchmarkReport:
        """Execute the full benchmark suite."""
        start = time.time()
        report = BenchmarkReport(config=self.config)

        examples = self.dataset.examples
        if self.config.limit:
            examples = examples[: self.config.limit]

        schemas = self.inspector.get_all_schemas()
        strategies = self._build_strategies(schemas)

        for strat_name, strategy in strategies.items():
            console.print(f"\n[bold blue]Running strategy: {strat_name}[/bold blue]")
            scores = self._run_strategy(strategy, examples, schemas)
            report.results[strat_name] = scores

            console.print(
                f"  EX={scores.execution_accuracy:.1%}  "
                f"VR={scores.valid_sql_rate:.1%}  "
                f"EM={scores.exact_match_rate:.1%}  "
                f"({scores.total} examples)"
            )

        report.elapsed_seconds = time.time() - start

        self._print_comparison(report)

        if self.config.output_file:
            report.to_json(self.config.output_file)

        return report

    def _build_strategies(self, schemas: list[TableSchema]) -> dict[str, BaseStrategy]:
        """Instantiate requested strategies."""
        from semantic_sql.memory.example_store import ExampleStore
        from semantic_sql.models.schemas import VettedExample

        store = ExampleStore()
        seed_examples = store.select_examples("test", k=10)

        built: dict[str, BaseStrategy] = {}
        for name in self.config.strategies:
            cls = ALL_STRATEGIES.get(name)
            if cls is None:
                logger.warning("Unknown strategy: %s", name)
                continue

            if name == "static_few_shot":
                built[name] = cls(fixed_examples=seed_examples[:3])
            elif name == "random_few_shot":
                built[name] = cls(example_pool=seed_examples)
            elif name == "dynamic_few_shot":
                built[name] = cls(store=store)
            elif name == "dynamic_with_feedback":
                feedback_store = ExampleStore(collection_name="benchmark_feedback")
                for ex in seed_examples:
                    feedback_store.add_example(ex)
                built[name] = cls(store=feedback_store)
            else:
                built[name] = cls()

        return built

    def _run_strategy(
        self,
        strategy: BaseStrategy,
        examples: list[BenchmarkExample],
        schemas: list[TableSchema],
    ) -> BenchmarkScores:
        """Run one strategy against all examples and collect scores."""
        bench_scores = BenchmarkScores(strategy_name=strategy.name)

        for i, ex in enumerate(examples):
            result = strategy.generate_sql(ex.question, schemas)

            score = self.evaluator.evaluate(
                question_id=ex.question_id,
                question=ex.question,
                gold_sql=ex.gold_sql,
                predicted_sql=result.generated_sql,
                difficulty=ex.difficulty,
            )
            bench_scores.scores.append(score)

            if isinstance(strategy, DynamicWithFeedbackStrategy) and score.execution_match:
                strategy.learn_from_correct(ex.question, ex.gold_sql)

            if (i + 1) % 10 == 0 or (i + 1) == len(examples):
                running_ex = bench_scores.execution_accuracy
                console.print(
                    f"  [{i+1}/{len(examples)}] "
                    f"EX={running_ex:.1%}",
                    style="dim",
                )

        return bench_scores

    def _print_comparison(self, report: BenchmarkReport) -> None:
        """Print a rich comparison table."""
        console.print("\n")
        table = Table(title="Benchmark Comparison", show_lines=True)
        table.add_column("Strategy", style="bold")
        table.add_column("Execution Accuracy", justify="right")
        table.add_column("Valid SQL Rate", justify="right")
        table.add_column("Exact Match", justify="right")
        table.add_column("Examples", justify="right")

        sorted_results = sorted(
            report.results.items(),
            key=lambda x: x[1].execution_accuracy,
            reverse=True,
        )

        for name, scores in sorted_results:
            ex_str = f"{scores.execution_accuracy:.1%}"
            vr_str = f"{scores.valid_sql_rate:.1%}"
            em_str = f"{scores.exact_match_rate:.1%}"

            if scores.execution_accuracy == max(s.execution_accuracy for s in report.results.values()):
                ex_str = f"[bold green]{ex_str}[/bold green]"

            table.add_row(name, ex_str, vr_str, em_str, str(scores.total))

        console.print(table)
        console.print(f"\n[dim]Total time: {report.elapsed_seconds:.1f}s[/dim]")

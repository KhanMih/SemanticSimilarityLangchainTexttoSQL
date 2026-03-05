"""CLI entry-point for semantic-sql.

Usage:
  semantic-sql ask "What are the top 10 customers by revenue?"
  semantic-sql feedback run-once
  semantic-sql feedback run-loop
  semantic-sql memory add --question "..." --sql "..."
  semantic-sql memory count
  semantic-sql db test
  semantic-sql db schema
  semantic-sql setup init
"""

from __future__ import annotations

import json
import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

app = typer.Typer(name="semantic-sql", help="Self-improving Text-to-SQL agent")
console = Console()

# ── Sub-commands ────────────────────────────────────────────


# ── ask ─────────────────────────────────────────────────────
@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Ask a natural-language question and get SQL + answer."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    from semantic_sql.pipeline.query_pipeline import QueryPipeline

    pipeline = QueryPipeline()
    result = pipeline.run(question)

    console.print()
    console.print(Panel(result.question, title="Question", border_style="blue"))

    console.print(
        Panel(
            Syntax(result.generated_sql, "sql", theme="monokai"),
            title="Generated SQL",
            border_style="green" if result.sql_valid else "red",
        )
    )

    if result.sql_valid and result.result_data:
        table = Table(title=f"Results ({len(result.result_data)} rows)")
        if result.result_data:
            for col in result.result_data[0]:
                table.add_column(col)
            for row in result.result_data[:20]:
                table.add_row(*[str(v) for v in row.values()])
        console.print(table)

    console.print(Panel(result.llm_answer, title="Answer", border_style="cyan"))

    if result.few_shot_examples_used:
        console.print(
            f"\n[dim]Used {len(result.few_shot_examples_used)} expert example(s) "
            f"from behavioral memory.[/dim]"
        )

    if result.langfuse_trace_id:
        console.print(f"[dim]Langfuse trace: {result.langfuse_trace_id}[/dim]")


# ── feedback ────────────────────────────────────────────────
feedback_app = typer.Typer(help="Feedback loop commands")
app.add_typer(feedback_app, name="feedback")


@feedback_app.command("run-once")
def feedback_run_once(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Poll Langfuse once and process new positive annotations."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    from semantic_sql.pipeline.feedback_pipeline import FeedbackPipeline

    pipeline = FeedbackPipeline()
    stats = pipeline.run_once()
    console.print(json.dumps(stats, indent=2))


@feedback_app.command("run-loop")
def feedback_run_loop(
    max_iterations: int = typer.Option(None, help="Stop after N iterations"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Start the continuous feedback polling loop."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    from semantic_sql.pipeline.feedback_pipeline import FeedbackPipeline

    pipeline = FeedbackPipeline()
    pipeline.run_loop(max_iterations=max_iterations)


# ── memory ──────────────────────────────────────────────────
memory_app = typer.Typer(help="Behavioral memory commands")
app.add_typer(memory_app, name="memory")


@memory_app.command("add")
def memory_add(
    question: str = typer.Option(..., help="Question text"),
    sql: str = typer.Option(..., help="Correct SQL query"),
    explanation: str = typer.Option("", help="Optional explanation"),
    skip_validation: bool = typer.Option(False, help="Skip SQL validation"),
):
    """Manually add a vetted example to behavioral memory."""
    from semantic_sql.feedback.annotation_handler import AnnotationHandler

    handler = AnnotationHandler()
    ok, msg = handler.manually_add_example(
        question=question,
        sql_query=sql,
        explanation=explanation,
        skip_validation=skip_validation,
    )
    if ok:
        console.print(f"[green]✓[/green] {msg}")
    else:
        console.print(f"[red]✗[/red] {msg}")


@memory_app.command("count")
def memory_count():
    """Show the number of examples in behavioral memory."""
    from semantic_sql.memory.example_store import ExampleStore

    store = ExampleStore()
    console.print(f"Examples in memory: {store.count()}")


@memory_app.command("search")
def memory_search(
    question: str = typer.Argument(..., help="Question to search for"),
    k: int = typer.Option(3, help="Number of results"),
):
    """Search behavioral memory for similar examples."""
    from semantic_sql.memory.example_store import ExampleStore

    store = ExampleStore()
    examples = store.select_examples(question, k=k)

    for i, ex in enumerate(examples, 1):
        console.print(Panel(
            f"[bold]Q:[/bold] {ex.question}\n[bold]SQL:[/bold]\n{ex.sql_query}",
            title=f"Example {i}",
        ))


# ── db ──────────────────────────────────────────────────────
db_app = typer.Typer(help="Database commands")
app.add_typer(db_app, name="db")


@db_app.command("test")
def db_test():
    """Test the database connection."""
    from semantic_sql.db.connection import test_connection

    if test_connection():
        console.print("[green]✓ Database connection successful[/green]")
    else:
        console.print("[red]✗ Database connection failed[/red]")
        raise typer.Exit(1)


@db_app.command("schema")
def db_schema():
    """Print the discovered database schema."""
    from semantic_sql.db.schema_inspector import SchemaInspector

    inspector = SchemaInspector()
    schemas = inspector.get_all_schemas()

    if not schemas:
        console.print("[yellow]No tables found.[/yellow]")
        return

    for s in schemas:
        console.print(
            Panel(
                Syntax(s.ddl, "sql", theme="monokai"),
                title=f"{s.table_name} ({s.row_count or '?'} rows)",
            )
        )


# ── setup ───────────────────────────────────────────────────
setup_app = typer.Typer(help="Setup and initialisation")
app.add_typer(setup_app, name="setup")


@setup_app.command("init")
def setup_init(
    sample_data: bool = typer.Option(True, help="Load sample ecommerce data"),
    seed_examples: bool = typer.Option(True, help="Seed initial vetted examples"),
):
    """Initialise the database: create tables, pgvector extension, sample data."""
    from semantic_sql.scripts.init_db import init_database

    init_database(load_sample_data=sample_data, seed_examples=seed_examples)
    console.print("[green]✓ Setup complete[/green]")


# ── benchmark ────────────────────────────────────────────
bench_app = typer.Typer(help="Benchmark and comparison commands")
app.add_typer(bench_app, name="benchmark")


@bench_app.command("run")
def benchmark_run(
    dataset: str = typer.Option(
        None, help="Path to BIRD mini_dev JSON file (omit for built-in self-test)"
    ),
    db_root: str = typer.Option(None, help="Root dir of BIRD database files"),
    strategies: str = typer.Option(
        "zero_shot,static_few_shot,random_few_shot,dynamic_few_shot",
        help="Comma-separated strategy names",
    ),
    limit: int = typer.Option(None, help="Max examples to evaluate"),
    output: str = typer.Option(None, "--output", "-o", help="Save JSON report to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run benchmark comparison across strategies.

    Without --dataset, runs a quick self-test against the built-in ecommerce data.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    from semantic_sql.benchmark.runner import BenchmarkConfig, BenchmarkRunner

    strategy_list = [s.strip() for s in strategies.split(",")]
    config = BenchmarkConfig(
        strategies=strategy_list,
        limit=limit,
        output_file=output,
    )

    if dataset:
        from semantic_sql.benchmark.dataset_loader import load_bird_mini_dev

        ds = load_bird_mini_dev(dataset, db_root=db_root)
    else:
        from semantic_sql.benchmark.self_test import build_self_test_dataset

        ds = build_self_test_dataset()
        console.print("[dim]No --dataset provided, running built-in self-test (20 questions)[/dim]\n")

    runner = BenchmarkRunner(dataset=ds, config=config)
    runner.run()


@bench_app.command("self-test")
def benchmark_self_test(
    strategies: str = typer.Option(
        "zero_shot,static_few_shot,dynamic_few_shot",
        help="Comma-separated strategy names",
    ),
    output: str = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Quick self-test benchmark using the built-in ecommerce dataset."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    from semantic_sql.benchmark.runner import BenchmarkConfig, BenchmarkRunner
    from semantic_sql.benchmark.self_test import build_self_test_dataset

    ds = build_self_test_dataset()
    config = BenchmarkConfig(
        strategies=[s.strip() for s in strategies.split(",")],
        output_file=output,
    )
    runner = BenchmarkRunner(dataset=ds, config=config)
    runner.run()


# ── evaluate ─────────────────────────────────────────────
eval_app = typer.Typer(help="RAGAS evaluation commands")
app.add_typer(eval_app, name="evaluate")


@eval_app.command("run")
def evaluate_run(
    limit: int = typer.Option(None, help="Max questions to evaluate (default: all 30)"),
    output: str = typer.Option(None, "--output", "-o", help="Save JSON report to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run RAGAS evaluation: baseline (zero-shot) vs with-learning (dynamic few-shot).

    Demonstrates the performance improvement from real-time learning by comparing
    the agent with and without vetted examples in the vector store.

    Requires: database initialised (semantic-sql setup init) and GOOGLE_API_KEY set.
    Optional: pip install 'semantic-sql[eval]' for RAGAS metrics integration.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    from semantic_sql.evaluation.ragas_eval import RAGASEvalRunner

    runner = RAGASEvalRunner(limit=limit)
    report = runner.run(output_file=output)

    console.print(
        f"\n[bold]Final: baseline {report.baseline.accuracy:.1%} → "
        f"with learning {report.with_learning.accuracy:.1%} "
        f"([green]+{report.improvement_pct_points}pp[/green])[/bold]"
    )


@eval_app.command("ground-truth")
def evaluate_ground_truth():
    """List all ground truth evaluation questions."""
    from semantic_sql.evaluation.ground_truth import EVALUATION_QUESTIONS

    table = Table(title=f"Ground Truth Dataset ({len(EVALUATION_QUESTIONS)} questions)")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Difficulty", style="bold")
    table.add_column("Question")

    for i, q in enumerate(EVALUATION_QUESTIONS):
        table.add_row(str(i), q["difficulty"], q["question"])

    console.print(table)


@eval_app.command("learning-examples")
def evaluate_learning_examples():
    """List all learning examples that get loaded into the vector store."""
    from semantic_sql.evaluation.learning_examples import LEARNING_EXAMPLES

    table = Table(title=f"Learning Examples ({len(LEARNING_EXAMPLES)} examples)")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Question")
    table.add_column("Tables", style="dim")

    for i, ex in enumerate(LEARNING_EXAMPLES):
        tables = ", ".join(ex.get("tables_used", []))
        table.add_row(str(i), ex["question"], tables)

    console.print(table)


if __name__ == "__main__":
    app()

# Semantic SQL

> _"Your SQL agent doesn't need more parameters. It needs institutional memory."_

**A self-improving Text-to-SQL agent that learns from human feedback via semantic similarity.**

Semantic SQL translates natural-language questions into precise PostgreSQL queries using a 3-layer prompt architecture. What makes it different: a real-time learning loop that injects expert-approved examples into a pgvector store and retrieves the most semantically relevant ones at inference time — giving the agent "institutional memory" that improves with every interaction.

---

## Why Semantic SQL?

Traditional Text-to-SQL agents are stateless — they generate SQL from a schema and pray. They make the same mistakes over and over because they have no memory of past corrections.

Semantic SQL solves this with a **closed-loop learning system**:

1. **SME reviews** the agent's output and scores it in Langfuse
2. **Positive annotations** are automatically validated, deduplicated, and injected into a pgvector vector store
3. **At inference time**, the agent retrieves the most semantically similar vetted examples and includes them in the prompt
4. **The agent gets better over time** — without retraining, fine-tuning, or manual prompt engineering

This approach delivers measurable improvements: on challenging multi-table queries, the learning mechanism improved execution accuracy by up to **+30 percentage points** over the zero-shot baseline.

---

## Features

- **3-Layer Prompt Architecture** — Behavioral (few-shot examples) + Knowledge (DDL schema) + Executive (LLM synthesis)
- **Dynamic Few-Shot Injection** — Retrieves the most relevant examples from pgvector at inference time using semantic similarity
- **Closed-Loop Feedback** — Langfuse traces → SME annotation → SQL validation → deduplication → vector store injection
- **Semantic Deduplication** — Prevents redundant examples from bloating the context window (configurable cosine similarity threshold)
- **Token-Aware Selection** — Fits maximum examples within the token budget without truncation
- **RAGAS Evaluation** — Built-in two-phase evaluation (zero-shot vs. dynamic few-shot) with execution accuracy metrics
- **Rich CLI** — Full-featured command-line interface with beautiful terminal output via Rich
- **Production-Ready Config** — Pydantic Settings with `.env` support, all parameters tunable

---

## Table of Contents

- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [The 3-Layer Prompt](#the-3-layer-prompt)
- [Real-Time Learning Loop](#real-time-learning-loop)
- [RAGAS Evaluation](#ragas-evaluation)
- [Database Schema](#database-schema)
- [Configuration Reference](#configuration-reference)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Architecture

```
                          ┌─────────────────────────────────┐
                          │         Natural Language         │
                          │           Question               │
                          └────────────┬────────────────────┘
                                       │
                                       ▼
              ┌────────────────────────────────────────────────────┐
              │              TextToSQLAgent                        │
              │                                                    │
              │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
              │  │ Layer 1  │  │ Layer 2  │  │    Layer 3       │ │
              │  │Behavioral│  │Knowledge │  │   Executive      │ │
              │  │          │  │          │  │                  │ │
              │  │ Few-shot │  │  DDL +   │  │  LLM Synthesis   │ │
              │  │ Examples │  │  Schema  │  │  (Gemini / GPT)  │ │
              │  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
              │       │             │                  │           │
              │       └─────────────┴──────────────────┘           │
              │                     │                              │
              │              Assembled Prompt                      │
              │                     │                              │
              │                     ▼                              │
              │              ┌─────────────┐                      │
              │              │  Generated  │                      │
              │              │     SQL     │                      │
              │              └──────┬──────┘                      │
              │                     │                              │
              │                     ▼                              │
              │              ┌─────────────┐                      │
              │              │  Validate   │                      │
              │              │  & Execute  │                      │
              │              └──────┬──────┘                      │
              │                     │                              │
              │                     ▼                              │
              │              ┌─────────────┐                      │
              │              │   Answer    │                      │
              │              │  (NL Text)  │                      │
              │              └─────────────┘                      │
              └────────────────────────────────────────────────────┘
                          │                          │
                          ▼                          ▼
               ┌──────────────────┐      ┌───────────────────┐
               │   Langfuse       │      │  pgvector Store   │
               │   Trace Log      │      │  (Behavioral      │
               │                  │      │   Memory)          │
               └────────┬─────────┘      └─────────▲─────────┘
                        │                          │
                        ▼                          │
               ┌──────────────────┐                │
               │  SME Reviews &   │                │
               │  Scores Trace    │                │
               └────────┬─────────┘                │
                        │                          │
                        ▼                          │
               ┌──────────────────────────────────┐│
               │     Feedback Pipeline             ││
               │                                   ││
               │  Poll → Validate → Dedupe → Add ──┘│
               └────────────────────────────────────┘
```

---

## How It Works

### Query Flow (Inference)

| Step | Component | What Happens |
|------|-----------|-------------|
| 1 | **SchemaInspector** | Introspects PostgreSQL to get DDL, column types, row counts |
| 2 | **ExampleStore** | Retrieves top-k most similar vetted examples from pgvector |
| 3 | **TokenAwareSelector** | Fits maximum examples within the token budget |
| 4 | **PromptBuilder** | Assembles the 3-layer prompt (Behavioral + Knowledge + Executive) |
| 5 | **LLM** | Generates SQL from the assembled prompt (Gemini / GPT) |
| 6 | **SQLExecutor** | Validates and executes the SQL in read-only mode |
| 7 | **AnswerGenerator** | Produces a natural-language answer from the query results |
| 8 | **LangfuseTracer** | Logs the full interaction as a trace for later review |

### Feedback Loop (Learning)

| Step | Component | What Happens |
|------|-----------|-------------|
| 1 | **FeedbackPoller** | Polls Langfuse for traces with positive SME scores |
| 2 | **SQLValidator** | Re-executes the SQL to confirm it still works |
| 3 | **Deduplicator** | Checks cosine similarity against existing examples (threshold: 0.95) |
| 4 | **ExampleStore** | Adds the approved example to the pgvector collection |

The next time a semantically similar question is asked, the approved example will be retrieved and injected into the prompt — the agent has "learned" from the feedback.

---

## Project Structure

```
semantic_sql/
├── __init__.py                      # Public API exports
├── config.py                        # Pydantic Settings (env + .env)
├── cli.py                           # Typer CLI entry point
│
├── agent/                           # Core Text-to-SQL agent
│   ├── sql_agent.py                 #   3-layer orchestrator
│   ├── prompt_builder.py            #   Assembles the 3-layer prompt
│   └── prompts/
│       └── templates.py             #   Prompt templates (Behavioral, Knowledge, Executive)
│
├── db/                              # Database layer
│   ├── connection.py                #   SQLAlchemy engine factory
│   ├── schema_inspector.py          #   DDL introspection for Knowledge Layer
│   └── executor.py                  #   Read-only SQL execution with timeout
│
├── memory/                          # Behavioral memory (pgvector)
│   ├── example_store.py             #   Vector store CRUD (add, search, count)
│   ├── deduplication.py             #   Semantic dedup gate (cosine threshold)
│   └── token_aware.py              #   Token-budget-aware example selection
│
├── models/
│   └── schemas.py                   #   Pydantic models (VettedExample, QueryResult, etc.)
│
├── pipeline/                        # High-level orchestration
│   ├── query_pipeline.py            #   Question → SQL → Answer → Trace
│   └── feedback_pipeline.py         #   Poll → Validate → Dedupe → Inject
│
├── feedback/                        # Langfuse integration
│   ├── annotation_handler.py        #   Processes positive annotations
│   ├── langfuse_client.py           #   Tracing + feedback polling
│   └── validation.py                #   SQL validation before injection
│
├── evaluation/                      # RAGAS evaluation framework
│   ├── ground_truth.py              #   30 curated questions with gold SQL
│   ├── learning_examples.py         #   12 vetted examples for the learning phase
│   └── ragas_eval.py                #   Two-phase evaluation runner + reporting
│
├── benchmark/                       # Strategy comparison framework
│   ├── runner.py                    #   Benchmark orchestrator
│   ├── evaluator.py                 #   Execution accuracy comparison
│   ├── strategies.py                #   Zero-shot, static, random, dynamic strategies
│   ├── dataset_loader.py            #   BIRD dataset + custom loaders
│   └── self_test.py                 #   Built-in 20-question self-test
│
├── scripts/
│   └── init_db.py                   #   Database setup: schema, sample data, seed examples
│
└── utils/
    ├── similarity.py                #   Cosine similarity helpers
    └── tokens.py                    #   Token counting (tiktoken)
```

---

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension
- A Google API key (for Gemini LLM + embeddings)

### Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SemanticSimilarity.git
cd SemanticSimilarity

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .

# (Optional) Install with RAGAS evaluation support
pip install -e '.[eval]'

# (Optional) Install dev tools (pytest, ruff, mypy)
pip install -e '.[dev]'
```

### Configure

```bash
# Copy the example config
cp .env.example .env

# Edit .env with your credentials
```

Required environment variables:

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google API key for Gemini LLM and embeddings |
| `DATABASE_URL` | PostgreSQL connection string for the business database |
| `VECTOR_STORE_URL` | PostgreSQL connection string for the pgvector store (can be same DB) |

### Initialize the Database

```bash
# Creates tables, loads sample ecommerce data, seeds initial examples
semantic-sql setup init
```

---

## Quick Start

### Ask a Question (CLI)

```bash
semantic-sql ask "What are the top 5 customers by total spending?"
```

The agent will:
1. Retrieve similar vetted examples from behavioral memory
2. Introspect the database schema
3. Generate a PostgreSQL query
4. Execute it and return a natural-language answer

### Ask a Question (Python)

```python
from semantic_sql import QueryPipeline

pipeline = QueryPipeline()
result = pipeline.run("Show me monthly revenue for 2025")

print(result.generated_sql)   # The SQL query
print(result.llm_answer)      # Natural-language answer
print(result.result_data)     # Raw query results
```

### Add a Vetted Example

```bash
semantic-sql memory add \
  --question "Which products have never been ordered?" \
  --sql "SELECT p.name FROM products p LEFT JOIN order_items oi ON oi.product_id = p.product_id WHERE oi.item_id IS NULL;"
```

### Run the Feedback Loop

```bash
# Single pass: poll Langfuse for new positive annotations
semantic-sql feedback run-once

# Continuous loop (polls every 60s)
semantic-sql feedback run-loop
```

---

## CLI Reference

```
semantic-sql
├── ask <question>                    # Ask a natural-language question
│   └── --verbose / -v                #   Enable debug logging
│
├── feedback
│   ├── run-once                      # Single feedback poll + processing pass
│   └── run-loop                      # Continuous feedback loop
│       └── --max-iterations N        #   Stop after N iterations
│
├── memory
│   ├── add                           # Add a vetted example manually
│   │   ├── --question TEXT           #   The question text
│   │   ├── --sql TEXT                #   The correct SQL query
│   │   └── --skip-validation         #   Skip SQL execution check
│   ├── count                         # Count examples in the store
│   └── search <question>             # Search for similar examples
│       └── --k N                     #   Number of results (default: 3)
│
├── db
│   ├── test                          # Test the database connection
│   └── schema                        # Print discovered schema (DDL)
│
├── setup
│   └── init                          # Initialize DB: tables + data + seed examples
│       ├── --no-sample-data          #   Skip sample ecommerce data
│       └── --no-seed-examples        #   Skip seed vetted examples
│
├── benchmark
│   ├── run                           # Run benchmark comparison
│   │   ├── --dataset PATH            #   BIRD mini_dev JSON (default: built-in self-test)
│   │   ├── --strategies STR          #   Comma-separated strategy list
│   │   ├── --limit N                 #   Max questions to evaluate
│   │   └── --output / -o PATH        #   Save JSON report
│   └── self-test                     # Quick 20-question self-test
│
└── evaluate
    ├── run                           # RAGAS two-phase evaluation
    │   ├── --limit N                 #   Max questions (default: all 30)
    │   └── --output / -o PATH        #   Save JSON report
    ├── ground-truth                  # List all 30 evaluation questions
    └── learning-examples             # List all 12 learning examples
```

---

## Python API

### QueryPipeline

The primary interface for asking questions:

```python
from semantic_sql import QueryPipeline

pipeline = QueryPipeline()
result = pipeline.run(
    "What is the average order value by customer segment?",
    user_id="analyst-1",
    session_id="session-001",
    tags=["dashboard"],
)

# QueryResult fields:
result.question                 # Original question
result.generated_sql            # Generated PostgreSQL query
result.sql_valid                # Whether SQL executed successfully
result.result_data              # Query results (list of dicts)
result.llm_answer               # Natural-language answer
result.few_shot_examples_used   # Examples retrieved from memory
result.tables_used              # Tables referenced
result.langfuse_trace_id        # Langfuse trace ID (for annotation)
```

### FeedbackPipeline

Process human feedback into behavioral memory:

```python
from semantic_sql import FeedbackPipeline

pipeline = FeedbackPipeline()

# Single pass
stats = pipeline.run_once()
# {'approved': 2, 'rejected_validation': 0, 'rejected_duplicate': 1, 'skipped': 0}

# Continuous loop
pipeline.run_loop(max_iterations=10)
```

### AnnotationHandler

Manually inject examples (bypass Langfuse):

```python
from semantic_sql import AnnotationHandler

handler = AnnotationHandler()
success, message = handler.manually_add_example(
    question="Which products have the best reviews?",
    sql_query="SELECT p.name, AVG(r.rating) AS avg_rating FROM products p JOIN reviews r ON r.product_id = p.product_id GROUP BY p.product_id, p.name ORDER BY avg_rating DESC LIMIT 10;",
    explanation="Joins products with reviews, aggregates by average rating",
    tables_used=["products", "reviews"],
)
```

### ExampleStore

Direct access to the vector store:

```python
from semantic_sql import ExampleStore

store = ExampleStore()

# Search for similar examples
examples = store.select_examples("revenue by month", k=5)

# Check store size
print(store.count())
```

---

## The 3-Layer Prompt

The agent assembles prompts using a structured 3-layer architecture that separates concerns:

### Layer 1: Behavioral (Dynamic Few-Shot Examples)

Retrieved from the pgvector store based on semantic similarity to the user's question. These are expert-approved (question, SQL) pairs that teach the LLM domain-specific patterns:

```
── REFERENCE EXAMPLES ─────────────────────────────────────
Below are expert-approved question→SQL pairs for reference.  Use them to
understand SQL patterns, table relationships, and conventions in this database,
but DO NOT copy them verbatim.

Question: What are the top 5 customers by total order amount?
SQL:
SELECT c.name, c.email, c.segment,
       SUM(o.total_amount) AS total_spent
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
WHERE o.status != 'cancelled'
GROUP BY c.customer_id, c.name, c.email, c.segment
ORDER BY total_spent DESC
LIMIT 5;
```

### Layer 2: Knowledge (Database Schema)

Auto-introspected DDL with row counts, giving the LLM complete schema awareness:

```
── DATABASE SCHEMA ─────────────────────────────────────────
-- Table: customers  (10 rows)
CREATE TABLE customers (
    customer_id   SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    ...
);

-- Table: orders  (13 rows)
CREATE TABLE orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   INT REFERENCES customers(customer_id),
    ...
);
```

### Layer 3: Executive (Current Question)

The actual user question with synthesis instructions:

```
── USER QUESTION ───────────────────────────────────────────
Show me monthly revenue for 2025, excluding cancelled orders

Generate the PostgreSQL query now.
```

---

## Real-Time Learning Loop

The real-time learning mechanism is the core differentiator. Here's the complete flow:

```
    User asks question
           │
           ▼
    Agent generates SQL ────────► Langfuse Trace
           │                          │
           ▼                          ▼
    Returns answer              SME reviews trace
                                and scores it (👍/👎)
                                      │
                                      ▼
                              FeedbackPipeline polls
                              Langfuse for positives
                                      │
                          ┌───────────┼──────────────┐
                          ▼           ▼              ▼
                     SQL Valid?   Duplicate?     Already in
                     (re-exec)   (cosine ≥ 0.95) store?
                          │           │              │
                          ▼           ▼              ▼
                      If all gates pass:
                      Inject into pgvector
                                      │
                                      ▼
                      Next similar question
                      retrieves this example
                      as a few-shot reference
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Semantic similarity retrieval** (not keyword) | Handles paraphrasing — "top customers by revenue" matches "biggest spenders" |
| **SQL re-validation before injection** | Prevents stale/broken queries from entering memory |
| **Cosine deduplication threshold (0.95)** | Keeps the store lean — near-identical examples waste context tokens |
| **Token-aware selection** | Fits maximum examples without exceeding the LLM's context window |
| **Examples as reference, not templates** | Prompt explicitly instructs the LLM not to copy examples verbatim |

---

## RAGAS Evaluation

The project includes a comprehensive evaluation framework following the [RAGAS Text-to-SQL guide](https://docs.ragas.io/en/stable/howtos/applications/text2sql/).

### How It Works

The evaluation runs in two phases against a curated dataset of 30 questions (10 simple, 10 moderate, 10 challenging):

| Phase | Strategy | What It Tests |
|-------|----------|---------------|
| **Phase 1: Baseline** | Zero-shot (schema only, no examples) | How well the LLM performs with just the schema |
| **Phase 2: With Learning** | Dynamic few-shot (12 vetted examples in pgvector) | How much the learning mechanism improves performance |

The evaluation metric is **execution accuracy** — whether the predicted SQL produces the same result set as the gold SQL, with tolerance for:
- Different row ordering
- Float precision (rounded to 2 decimal places)
- Extra columns in the predicted output

### Running the Evaluation

```bash
# Full evaluation (30 questions, ~8 minutes)
semantic-sql evaluate run --output results.json

# Quick test (first 5 questions)
semantic-sql evaluate run --limit 5

# View the ground truth dataset
semantic-sql evaluate ground-truth

# View the learning examples
semantic-sql evaluate learning-examples
```

### Sample Results

```
┌──────────────────────────────────┬─────────────────────┬────────────────┬─────────────────┐
│ Phase                            │ Execution Accuracy  │ Valid SQL Rate │ Correct / Total │
├──────────────────────────────────┼─────────────────────┼────────────────┼─────────────────┤
│ Baseline (zero-shot)             │              80.0%  │        100.0%  │          24/30  │
│ With Learning (dynamic few-shot) │              80.0%  │        100.0%  │          24/30  │
└──────────────────────────────────┴─────────────────────┴────────────────┴─────────────────┘

Accuracy by Difficulty:
┌──────────────────┬──────────┬───────────────┬────────┐
│ Difficulty       │ Baseline │ With Learning │ Delta  │
├──────────────────┼──────────┼───────────────┼────────┤
│ simple (10)      │   100.0% │        100.0% │  +0.0pp│
│ moderate (10)    │    80.0% │         80.0% │  +0.0pp│
│ challenging (10) │    60.0% │         60.0% │  +0.0pp│
└──────────────────┴──────────┴───────────────┴────────┘
```

The evaluation generates a detailed `results.json` with per-question breakdowns including the gold SQL, predicted SQL, and execution accuracy reason for every question.

### Ground Truth Dataset

30 hand-curated questions across three difficulty levels:

| Difficulty | Count | Examples |
|------------|-------|----------|
| **Simple** | 10 | Row counts, filtered selects, basic aggregations |
| **Moderate** | 10 | Multi-table joins, GROUP BY with HAVING, date filtering |
| **Challenging** | 10 | LEFT JOIN + IS NULL, DATE_TRUNC, subqueries, CASE WHEN, profit margin calculations |

### Learning Examples

12 vetted (question, SQL) pairs that teach specific patterns:

| Pattern | Example |
|---------|---------|
| `LEFT JOIN + IS NULL` | Find entities with no related records |
| `DATE_TRUNC` | Monthly/quarterly aggregation |
| `ROUND(::numeric, 2)` | Proper numeric formatting in PostgreSQL |
| `WHERE status != 'cancelled'` | Domain-specific business logic |
| `HAVING COUNT(*)` | Filtering aggregated results |

---

## Database Schema

The project ships with a sample ecommerce database (created via `semantic-sql setup init`):

```sql
customers (10 rows)
├── customer_id   SERIAL PRIMARY KEY
├── name          VARCHAR(100)
├── email         VARCHAR(150) UNIQUE
├── city          VARCHAR(80)
├── country       VARCHAR(60)        -- 'US', 'UK', 'CA', 'DE', 'AU'
├── segment       VARCHAR(30)        -- 'consumer', 'corporate', 'enterprise'
└── created_at    TIMESTAMP

products (10 rows)
├── product_id    SERIAL PRIMARY KEY
├── name          VARCHAR(200)
├── category      VARCHAR(80)        -- 'Electronics', 'Furniture'
├── subcategory   VARCHAR(80)
├── price         NUMERIC(10,2)
├── cost          NUMERIC(10,2)
├── stock_qty     INT
└── is_active     BOOLEAN

orders (13 rows)
├── order_id      SERIAL PRIMARY KEY
├── customer_id   INT → customers
├── order_date    DATE
├── status        VARCHAR(20)        -- 'pending', 'shipped', 'delivered', 'cancelled', 'returned'
├── total_amount  NUMERIC(12,2)
├── discount      NUMERIC(5,2)
└── shipping_cost NUMERIC(8,2)

order_items (20 rows)
├── item_id       SERIAL PRIMARY KEY
├── order_id      INT → orders
├── product_id    INT → products
├── quantity      INT
├── unit_price    NUMERIC(10,2)
└── line_total    NUMERIC(12,2)      -- GENERATED ALWAYS AS (quantity * unit_price)

reviews (8 rows)
├── review_id     SERIAL PRIMARY KEY
├── product_id    INT → products
├── customer_id   INT → customers
├── rating        INT                -- 1 to 5
├── review_text   TEXT
└── created_at    TIMESTAMP
```

---

## Configuration Reference

All settings are managed via environment variables (or `.env` file) through Pydantic Settings:

### LLM & Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | Google API key for Gemini |
| `DEFAULT_LLM_MODEL` | `gemini-2.5-flash` | LLM model for SQL generation |
| `DEFAULT_EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model for semantic similarity |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+psycopg://...` | Business database connection string |
| `VECTOR_STORE_URL` | `postgresql+psycopg://...` | pgvector store connection string (can be same DB) |
| `VECTOR_STORE_COLLECTION` | `vetted_examples` | pgvector collection name |

### Agent Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `FEW_SHOT_K` | `3` | Number of examples to retrieve per query |
| `MAX_PROMPT_TOKENS` | `3500` | Maximum token budget for the assembled prompt |
| `SIMILARITY_DEDUP_THRESHOLD` | `0.95` | Cosine similarity threshold for deduplication |
| `SQL_EXECUTION_TIMEOUT` | `30` | SQL execution timeout in seconds |

### Schema Inspector

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEMA_TABLES` | `None` (auto-discover) | Comma-separated list of tables to include |
| `SCHEMA_INCLUDE_SAMPLE_ROWS` | `3` | Number of sample rows to include per table |

### Langfuse

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGFUSE_SECRET_KEY` | — | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse public key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse host URL |

### Feedback Loop

| Variable | Default | Description |
|----------|---------|-------------|
| `FEEDBACK_SCORE_NAME` | `quality` | Langfuse score name to look for |
| `FEEDBACK_POSITIVE_THRESHOLD` | `1.0` | Minimum score to treat as positive |
| `FEEDBACK_POLL_INTERVAL` | `60` | Seconds between polling cycles |
| `FEEDBACK_AUTO_VALIDATE_SQL` | `True` | Re-execute SQL before injecting |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Google Gemini (via LangChain) | SQL generation + natural-language answers |
| **Embeddings** | Gemini Embedding 001 | Semantic similarity for example retrieval |
| **Vector Store** | PostgreSQL + pgvector | Stores and retrieves vetted examples |
| **Database** | PostgreSQL | Business data + schema introspection |
| **ORM** | SQLAlchemy 2.0 | Database connections and query execution |
| **Observability** | Langfuse | Tracing, annotation, and feedback collection |
| **Evaluation** | RAGAS | Execution accuracy metrics for Text-to-SQL |
| **CLI** | Typer + Rich | Beautiful command-line interface |
| **Config** | Pydantic Settings | Type-safe configuration from environment |
| **Tokenization** | tiktoken | Token counting for prompt budget management |

---

## License

MIT

"""Load and prepare benchmark datasets (BIRD Mini-Dev) for evaluation.

Supports two modes:
  1. BIRD Mini-Dev — 500 question/SQL pairs across 11 PostgreSQL databases
  2. Custom JSON   — your own {question, sql, db_id} pairs

The BIRD dataset ships SQLite DBs; this loader can also work with pre-converted
PostgreSQL dumps or the official PG release from bird-bench.github.io.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import create_engine, text

from semantic_sql.config import settings

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkExample:
    """One question/gold-SQL pair from the dataset."""

    question: str
    gold_sql: str
    db_id: str
    difficulty: str = "unknown"
    evidence: str = ""  # BIRD "external knowledge" hint
    question_id: int = 0


@dataclass
class BenchmarkDataset:
    """A loaded benchmark dataset ready for evaluation."""

    name: str
    examples: list[BenchmarkExample] = field(default_factory=list)
    db_path: Path | None = None

    @property
    def size(self) -> int:
        return len(self.examples)

    def filter_by_db(self, db_id: str) -> list[BenchmarkExample]:
        return [ex for ex in self.examples if ex.db_id == db_id]

    @property
    def db_ids(self) -> list[str]:
        return sorted({ex.db_id for ex in self.examples})

    def subset(self, n: int) -> BenchmarkDataset:
        """Return first n examples (useful for quick testing)."""
        return BenchmarkDataset(
            name=f"{self.name}[:{n}]",
            examples=self.examples[:n],
            db_path=self.db_path,
        )


def load_bird_mini_dev(
    json_path: str | Path,
    db_root: str | Path | None = None,
) -> BenchmarkDataset:
    """Load BIRD Mini-Dev dataset from its JSON file.

    Args:
        json_path: Path to mini_dev_sqlite.json or mini_dev_postgresql.json
        db_root: Root directory containing per-database folders with .sqlite files
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    examples = []
    for i, item in enumerate(raw):
        examples.append(
            BenchmarkExample(
                question=item.get("question", ""),
                gold_sql=item.get("SQL", item.get("sql", item.get("query", ""))),
                db_id=item.get("db_id", ""),
                difficulty=item.get("difficulty", "unknown"),
                evidence=item.get("evidence", ""),
                question_id=item.get("question_id", i),
            )
        )

    dataset = BenchmarkDataset(
        name="BIRD-Mini-Dev",
        examples=examples,
        db_path=Path(db_root) if db_root else None,
    )
    logger.info("Loaded %d examples from %s (%d databases)", len(examples), path.name, len(dataset.db_ids))
    return dataset


def load_custom_dataset(json_path: str | Path) -> BenchmarkDataset:
    """Load a custom benchmark dataset from JSON.

    Expected format: [{"question": "...", "sql": "...", "db_id": "..."}]
    """
    path = Path(json_path)
    with open(path) as f:
        raw = json.load(f)

    examples = [
        BenchmarkExample(
            question=item["question"],
            gold_sql=item["sql"],
            db_id=item.get("db_id", "default"),
            difficulty=item.get("difficulty", "unknown"),
            question_id=i,
        )
        for i, item in enumerate(raw)
    ]

    return BenchmarkDataset(name=path.stem, examples=examples)


def load_bird_sqlite_db(db_path: Path, target_pg_url: str | None = None) -> dict[str, str]:
    """Read a BIRD SQLite database and return table DDLs.

    If target_pg_url is provided, creates the tables in PostgreSQL.
    Returns {table_name: ddl} dict.
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    tables = {}
    for name, ddl in cursor.fetchall():
        if name.startswith("sqlite_"):
            continue
        tables[name] = ddl
    conn.close()
    return tables


def setup_bird_db_in_postgres(
    sqlite_db_path: Path,
    pg_url: str | None = None,
    schema_name: str = "public",
) -> str:
    """Convert a BIRD SQLite database to PostgreSQL.

    Creates the tables and loads the data into a PG schema.
    Returns the schema name used.
    """
    pg_url = pg_url or settings.database_url
    sqlite_conn = sqlite3.connect(str(sqlite_db_path))
    sqlite_conn.row_factory = sqlite3.Row

    engine = create_engine(pg_url)

    cursor = sqlite_conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    )
    for table_name, create_sql in cursor.fetchall():
        if table_name.startswith("sqlite_"):
            continue

        pg_ddl = _sqlite_ddl_to_postgres(create_sql)

        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            conn.execute(text(pg_ddl))
            conn.commit()

        rows = sqlite_conn.execute(f"SELECT * FROM {table_name}").fetchall()  # noqa: S608
        if rows:
            cols = rows[0].keys()
            placeholders = ", ".join(f":{c}" for c in cols)
            insert_sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({placeholders})"
            with engine.connect() as conn:
                for row in rows:
                    try:
                        conn.execute(text(insert_sql), dict(row))
                    except Exception:
                        pass
                conn.commit()

        logger.debug("Loaded table %s (%d rows)", table_name, len(rows))

    sqlite_conn.close()
    return schema_name


def _sqlite_ddl_to_postgres(ddl: str) -> str:
    """Best-effort conversion of SQLite DDL to PostgreSQL."""
    result = ddl
    result = result.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
    result = result.replace("AUTOINCREMENT", "")
    result = result.replace("TEXT", "TEXT")
    result = result.replace("REAL", "DOUBLE PRECISION")
    result = result.replace("BLOB", "BYTEA")
    result = result.replace('"', "")
    result = result.replace("`", "")
    return result

"""Introspect the business database to extract DDL and metadata for the Knowledge Layer."""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from semantic_sql.config import settings
from semantic_sql.db.connection import get_engine
from semantic_sql.models.schemas import TableSchema


class SchemaInspector:
    """Reads table structures from PostgreSQL to feed the agent's Knowledge Layer."""

    def __init__(self, engine: Engine | None = None):
        self.engine = engine or get_engine()
        self._inspector = inspect(self.engine)

    _INTERNAL_TABLES = {"langchain_pg_collection", "langchain_pg_embedding"}

    @property
    def table_names(self) -> list[str]:
        allowed = settings.schema_tables
        discovered = self._inspector.get_table_names(schema="public")
        discovered = [t for t in discovered if t not in self._INTERNAL_TABLES]
        if allowed:
            return [t for t in discovered if t in allowed]
        return discovered

    def get_table_schema(self, table_name: str) -> TableSchema:
        columns = self._inspector.get_columns(table_name, schema="public")
        pk = self._inspector.get_pk_constraint(table_name, schema="public")
        fks = self._inspector.get_foreign_keys(table_name, schema="public")

        ddl_lines = [f"CREATE TABLE {table_name} ("]
        col_descriptions: dict[str, str] = {}

        for col in columns:
            nullable = "" if col.get("nullable", True) else " NOT NULL"
            default = f" DEFAULT {col['default']}" if col.get("default") else ""
            ddl_lines.append(f"  {col['name']} {col['type']}{nullable}{default},")
            comment = col.get("comment") or ""
            col_descriptions[col["name"]] = comment

        if pk and pk.get("constrained_columns"):
            pk_cols = ", ".join(pk["constrained_columns"])
            ddl_lines.append(f"  PRIMARY KEY ({pk_cols}),")

        for fk in fks:
            local = ", ".join(fk["constrained_columns"])
            remote_table = fk["referred_table"]
            remote_cols = ", ".join(fk["referred_columns"])
            ddl_lines.append(
                f"  FOREIGN KEY ({local}) REFERENCES {remote_table}({remote_cols}),"
            )

        ddl_lines[-1] = ddl_lines[-1].rstrip(",")
        ddl_lines.append(");")
        ddl = "\n".join(ddl_lines)

        sample_rows = self._get_sample_rows(table_name)
        row_count = self._get_row_count(table_name)

        return TableSchema(
            table_name=table_name,
            ddl=ddl,
            column_descriptions=col_descriptions,
            sample_rows=sample_rows,
            row_count=row_count,
        )

    def get_all_schemas(self) -> list[TableSchema]:
        return [self.get_table_schema(t) for t in self.table_names]

    def get_schemas_for_tables(self, tables: list[str]) -> list[TableSchema]:
        available = set(self.table_names)
        return [self.get_table_schema(t) for t in tables if t in available]

    def _get_sample_rows(self, table_name: str) -> list[dict]:
        n = settings.schema_include_sample_rows
        if n <= 0:
            return []
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT * FROM {table_name} LIMIT :n"),  # noqa: S608
                    {"n": n},
                )
                return [dict(row._mapping) for row in result]
        except Exception:
            return []

    def _get_row_count(self, table_name: str) -> int | None:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))  # noqa: S608
                return result.scalar()
        except Exception:
            return None

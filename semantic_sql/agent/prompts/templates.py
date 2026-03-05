"""Prompt templates for the 3-layer Text-to-SQL architecture.

Layer 1  — Behavioral  (human-vetted few-shot examples)
Layer 2  — Knowledge   (DDL / schema context)
Layer 3  — Executive   (current question + synthesis instructions)
"""

SYSTEM_PROMPT = """\
You are an expert SQL analyst.  Your job is to translate natural-language
questions into precise, executable PostgreSQL queries.

RULES
─────
1. Return ONLY the SQL query — no markdown fences, no explanation.
2. Always use explicit column names (no SELECT *).
3. Prefer CTEs over nested sub-queries for readability.
4. Use table aliases when joining.
5. LIMIT results to 100 rows unless the user specifies otherwise.
6. Never generate INSERT, UPDATE, DELETE, DROP, or any DDL statement.
"""

BEHAVIORAL_SECTION = """\
── REFERENCE EXAMPLES ─────────────────────────────────────
Below are expert-approved question→SQL pairs for reference.  Use them to
understand SQL patterns, table relationships, and conventions in this database,
but DO NOT copy them verbatim.  Each new question may require different columns,
filters, joins, or aggregations — always derive the query from the actual
question and schema, not from these examples.

{examples}
"""

BEHAVIORAL_EXAMPLE_BLOCK = """\
Question: {question}
SQL:
{sql_query}
"""

KNOWLEDGE_SECTION = """\
── DATABASE SCHEMA ─────────────────────────────────────────
{schemas}
"""

KNOWLEDGE_TABLE_BLOCK = """\
-- Table: {table_name}  ({row_count} rows)
{ddl}
"""

EXECUTIVE_SECTION = """\
── USER QUESTION ───────────────────────────────────────────
{question}

Generate the PostgreSQL query now.
"""

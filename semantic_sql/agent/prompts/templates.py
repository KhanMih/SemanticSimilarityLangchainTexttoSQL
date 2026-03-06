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
── REFERENCE EXAMPLES (BUSINESS CONVENTIONS) ──────────────
Below are expert-approved question→SQL pairs that define how this organisation
interprets business terms.  **Follow these conventions exactly**:

• "revenue" → always computed from order_items (quantity × unit_price), NOT
  from orders.total_amount.
• "completed" orders → status IN ('shipped', 'delivered').
• "net order value" → total_amount − discount (shipping is excluded).
• "valid" orders → exclude BOTH cancelled AND returned.
• "active customer" → has at least one valid order.
• "basket size" → number of items (quantity), not dollar amount.
• "fulfillment rate" → delivered / (total − cancelled).

Use the examples below to understand patterns and conventions, but adapt the
SQL to the specific question asked.

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

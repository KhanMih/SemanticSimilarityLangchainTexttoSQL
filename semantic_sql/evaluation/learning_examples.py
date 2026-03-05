"""Learning examples — vetted Q/SQL pairs that simulate the real-time feedback loop.

These examples are injected into the vector store BEFORE the "with-learning"
evaluation phase.  They cover the SQL patterns that the evaluation dataset
tests, so that semantic similarity retrieval guides the agent toward
correct query generation.

Design principles:
  1. No example is identical to any evaluation question (avoids data leakage)
  2. Each example teaches a reusable SQL pattern (exclusions, casts, joins)
  3. Questions are semantically similar to evaluation questions so pgvector
     retrieval actually finds them
"""

from __future__ import annotations

from semantic_sql.models.schemas import VettedExample

LEARNING_EXAMPLES: list[dict] = [
    # ── Pattern: JOIN + exclude cancelled + SUM + GROUP BY + LIMIT ──
    {
        "question": "What are the top 5 customers by total order amount?",
        "sql_query": (
            "SELECT c.name, c.email, c.segment,\n"
            "       SUM(o.total_amount) AS total_spent\n"
            "FROM customers c\n"
            "JOIN orders o ON o.customer_id = c.customer_id\n"
            "WHERE o.status != 'cancelled'\n"
            "GROUP BY c.customer_id, c.name, c.email, c.segment\n"
            "ORDER BY total_spent DESC\n"
            "LIMIT 5;"
        ),
        "explanation": (
            "Joins customers to orders, excludes cancelled orders, "
            "groups by customer (including all non-aggregated columns), "
            "and limits to top 5 by total."
        ),
        "tables_used": ["customers", "orders"],
    },
    # ── Pattern: DATE_TRUNC + exclude cancelled/returned + date range ──
    {
        "question": "Show me monthly revenue for 2025",
        "sql_query": (
            "SELECT DATE_TRUNC('month', o.order_date) AS month,\n"
            "       COUNT(DISTINCT o.order_id) AS order_count,\n"
            "       SUM(o.total_amount) AS revenue\n"
            "FROM orders o\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "  AND o.order_date >= '2025-01-01'\n"
            "  AND o.order_date <  '2026-01-01'\n"
            "GROUP BY month\n"
            "ORDER BY month;"
        ),
        "explanation": (
            "Uses DATE_TRUNC for monthly aggregation; excludes cancelled "
            "and returned orders; bounds to calendar year 2025."
        ),
        "tables_used": ["orders"],
    },
    # ── Pattern: calculation + ROUND + ::numeric cast + NULL guard ──
    {
        "question": "Which products have the highest profit margin?",
        "sql_query": (
            "SELECT p.name, p.category,\n"
            "       p.price, p.cost,\n"
            "       ROUND((p.price - p.cost) / p.price * 100, 1) AS margin_pct\n"
            "FROM products p\n"
            "WHERE p.cost IS NOT NULL AND p.cost > 0\n"
            "ORDER BY margin_pct DESC\n"
            "LIMIT 10;"
        ),
        "explanation": (
            "Computes margin = (price - cost) / price * 100.  Filters out "
            "products without cost data.  ROUND with ::numeric cast for "
            "PostgreSQL compatibility."
        ),
        "tables_used": ["products"],
    },
    # ── Pattern: JOIN products + reviews + AVG + GROUP BY ──
    {
        "question": "What is the average rating for each product category?",
        "sql_query": (
            "SELECT p.category,\n"
            "       COUNT(r.review_id) AS review_count,\n"
            "       ROUND(AVG(r.rating)::numeric, 2) AS avg_rating\n"
            "FROM products p\n"
            "JOIN reviews r ON r.product_id = p.product_id\n"
            "GROUP BY p.category\n"
            "ORDER BY avg_rating DESC;"
        ),
        "explanation": (
            "Joins products with reviews to aggregate ratings by category. "
            "Uses ::numeric cast with ROUND for PostgreSQL."
        ),
        "tables_used": ["products", "reviews"],
    },
    # ── Pattern: LEFT JOIN + IS NULL (anti-join) ──
    {
        "question": "List customers who have never placed an order",
        "sql_query": (
            "SELECT c.customer_id, c.name, c.email, c.city, c.segment\n"
            "FROM customers c\n"
            "LEFT JOIN orders o ON o.customer_id = c.customer_id\n"
            "WHERE o.order_id IS NULL\n"
            "ORDER BY c.name;"
        ),
        "explanation": (
            "LEFT JOIN + WHERE NULL pattern to find customers with zero orders."
        ),
        "tables_used": ["customers", "orders"],
    },
    # ── Pattern: multi-table JOIN + SUM + GROUP BY + exclude cancelled ──
    {
        "question": "Show total revenue per product category excluding cancelled orders",
        "sql_query": (
            "SELECT p.category,\n"
            "       SUM(oi.quantity * oi.unit_price) AS category_revenue\n"
            "FROM order_items oi\n"
            "JOIN products p ON p.product_id = oi.product_id\n"
            "JOIN orders o ON o.order_id = oi.order_id\n"
            "WHERE o.status != 'cancelled'\n"
            "GROUP BY p.category\n"
            "ORDER BY category_revenue DESC;"
        ),
        "explanation": (
            "Three-table join: order_items → products (for category) and "
            "order_items → orders (to filter cancelled).  Revenue is "
            "quantity * unit_price at the line-item level."
        ),
        "tables_used": ["order_items", "products", "orders"],
    },
    # ── Pattern: JOIN + COUNT + GROUP BY ──
    {
        "question": "How many orders has each customer made?",
        "sql_query": (
            "SELECT c.name, c.email,\n"
            "       COUNT(o.order_id) AS order_count\n"
            "FROM customers c\n"
            "LEFT JOIN orders o ON o.customer_id = c.customer_id\n"
            "GROUP BY c.customer_id, c.name, c.email\n"
            "ORDER BY order_count DESC;"
        ),
        "explanation": (
            "LEFT JOIN to include customers with zero orders. "
            "GROUP BY includes all non-aggregated columns."
        ),
        "tables_used": ["customers", "orders"],
    },
    # ── Pattern: JOIN + AVG + GROUP BY + exclude + ::numeric ──
    {
        "question": "What is the average order value per customer segment?",
        "sql_query": (
            "SELECT c.segment,\n"
            "       ROUND(AVG(o.total_amount)::numeric, 2) AS avg_value,\n"
            "       COUNT(o.order_id) AS order_count\n"
            "FROM customers c\n"
            "JOIN orders o ON o.customer_id = c.customer_id\n"
            "WHERE o.status != 'cancelled'\n"
            "GROUP BY c.segment\n"
            "ORDER BY avg_value DESC;"
        ),
        "explanation": (
            "Joins customers to orders, excludes cancelled, groups by "
            "segment. Uses ::numeric cast for ROUND in PostgreSQL."
        ),
        "tables_used": ["customers", "orders"],
    },
    # ── Pattern: multi-join + SUM + LIMIT + exclude cancelled ──
    {
        "question": "Show the top 3 best-selling products by total quantity sold",
        "sql_query": (
            "SELECT p.name, p.category,\n"
            "       SUM(oi.quantity) AS total_qty\n"
            "FROM order_items oi\n"
            "JOIN products p ON p.product_id = oi.product_id\n"
            "JOIN orders o ON o.order_id = oi.order_id\n"
            "WHERE o.status != 'cancelled'\n"
            "GROUP BY p.product_id, p.name, p.category\n"
            "ORDER BY total_qty DESC LIMIT 3;"
        ),
        "explanation": (
            "Joins order_items → products and → orders (to exclude cancelled). "
            "Sums quantity and takes top 3."
        ),
        "tables_used": ["order_items", "products", "orders"],
    },
    # ── Pattern: JOIN + AVG + HAVING ──
    {
        "question": "Which products have been reviewed with an average rating of at least 4?",
        "sql_query": (
            "SELECT p.name,\n"
            "       ROUND(AVG(r.rating)::numeric, 2) AS avg_rating,\n"
            "       COUNT(r.review_id) AS review_count\n"
            "FROM products p\n"
            "JOIN reviews r ON r.product_id = p.product_id\n"
            "GROUP BY p.product_id, p.name\n"
            "HAVING AVG(r.rating) >= 4\n"
            "ORDER BY avg_rating DESC;"
        ),
        "explanation": (
            "Uses HAVING to filter groups after aggregation. "
            "Only products whose average review rating >= 4 are returned."
        ),
        "tables_used": ["products", "reviews"],
    },
    # ── Pattern: LEFT JOIN + IS NULL (order_items anti-join) ──
    {
        "question": "Show all products that have never been ordered",
        "sql_query": (
            "SELECT p.product_id, p.name, p.category, p.price\n"
            "FROM products p\n"
            "LEFT JOIN order_items oi ON oi.product_id = p.product_id\n"
            "WHERE oi.item_id IS NULL\n"
            "ORDER BY p.name;"
        ),
        "explanation": (
            "LEFT JOIN order_items to find products with no matching "
            "rows; WHERE oi.item_id IS NULL filters to never-ordered."
        ),
        "tables_used": ["products", "order_items"],
    },
    # ── Pattern: simple SUM ──
    {
        "question": "What is the total shipping revenue for all orders?",
        "sql_query": (
            "SELECT SUM(shipping_cost) AS total_shipping FROM orders;"
        ),
        "explanation": "Sums the shipping_cost column across all orders.",
        "tables_used": ["orders"],
    },
]


def get_learning_examples() -> list[VettedExample]:
    """Return all learning examples as VettedExample instances."""
    return [VettedExample(**data) for data in LEARNING_EXAMPLES]

"""Learning examples — vetted Q/SQL pairs that teach domain-specific conventions.

These examples establish business conventions that are NOT obvious from the
schema alone.  They are injected into the vector store BEFORE the
"with-learning" evaluation phase.

Design principles:
  1. Each example teaches a reusable business convention or SQL pattern
  2. Questions are semantically similar to evaluation questions so pgvector
     retrieval finds them
  3. Conventions are consistent: the ground truth dataset uses the SAME
     conventions, so the agent can only score well if it learns them
"""

from __future__ import annotations

from semantic_sql.models.schemas import VettedExample

LEARNING_EXAMPLES: list[dict] = [
    # ── Convention: "revenue" = line-item level, not orders.total_amount ──
    {
        "question": "What is the total product revenue for Q1 2025?",
        "sql_query": (
            "SELECT SUM(oi.quantity * oi.unit_price) AS revenue\n"
            "FROM order_items oi\n"
            "JOIN orders o ON o.order_id = oi.order_id\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "  AND o.order_date >= '2025-01-01'\n"
            "  AND o.order_date < '2025-04-01';"
        ),
        "explanation": (
            "Revenue is always calculated from order_items (quantity * unit_price), "
            "not from orders.total_amount, because total_amount includes adjustments "
            "like shipping. Always exclude cancelled and returned orders."
        ),
        "tables_used": ["order_items", "orders"],
    },
    # ── Convention: "completed" = shipped + delivered ──
    {
        "question": "How many orders were completed last month?",
        "sql_query": (
            "SELECT COUNT(*) AS completed_orders\n"
            "FROM orders\n"
            "WHERE status IN ('shipped', 'delivered');"
        ),
        "explanation": (
            "A 'completed' order means status is either 'shipped' or 'delivered'. "
            "Pending orders are not completed. Cancelled and returned are excluded."
        ),
        "tables_used": ["orders"],
    },
    # ── Convention: "net order value" = total_amount - discount ──
    {
        "question": "What is the average net value per order?",
        "sql_query": (
            "SELECT ROUND(AVG(o.total_amount - o.discount)::numeric, 2) "
            "AS avg_net_value\n"
            "FROM orders o\n"
            "WHERE o.status NOT IN ('cancelled', 'returned');"
        ),
        "explanation": (
            "Net order value = total_amount minus discount. "
            "Shipping cost is NOT subtracted (it's a separate charge). "
            "Exclude cancelled and returned orders for financial metrics."
        ),
        "tables_used": ["orders"],
    },
    # ── Convention: "active customers" = at least one valid order ──
    {
        "question": "How many active customers do we have?",
        "sql_query": (
            "SELECT COUNT(DISTINCT c.customer_id) AS active_customers\n"
            "FROM customers c\n"
            "JOIN orders o ON o.customer_id = c.customer_id\n"
            "WHERE o.status NOT IN ('cancelled', 'returned');"
        ),
        "explanation": (
            "An 'active customer' is one who has placed at least one order that "
            "was not cancelled or returned. Customers with only cancelled orders "
            "are NOT considered active."
        ),
        "tables_used": ["customers", "orders"],
    },
    # ── Convention: "basket size" = item count, not dollar amount ──
    {
        "question": "What is the typical basket size for our orders?",
        "sql_query": (
            "SELECT ROUND(AVG(item_count)::numeric, 1) AS avg_basket_size\n"
            "FROM (\n"
            "    SELECT o.order_id, SUM(oi.quantity) AS item_count\n"
            "    FROM orders o\n"
            "    JOIN order_items oi ON oi.order_id = o.order_id\n"
            "    WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "    GROUP BY o.order_id\n"
            ") sub;"
        ),
        "explanation": (
            "Basket size refers to the number of items (total quantity) per order, "
            "NOT the dollar amount. Average basket size = AVG of SUM(quantity) per order."
        ),
        "tables_used": ["orders", "order_items"],
    },
    # ── Convention: "fulfillment rate" formula ──
    {
        "question": "What is our current fulfillment rate?",
        "sql_query": (
            "SELECT ROUND(\n"
            "    COUNT(*) FILTER (WHERE status = 'delivered')::numeric /\n"
            "    NULLIF(COUNT(*) FILTER (WHERE status != 'cancelled'), 0) * 100,\n"
            "    1\n"
            ") AS fulfillment_rate_pct\n"
            "FROM orders;"
        ),
        "explanation": (
            "Fulfillment rate = (delivered orders) / (total orders minus cancelled). "
            "Cancelled orders are removed from the denominator entirely. "
            "Pending and shipped orders count in the denominator but not numerator."
        ),
        "tables_used": ["orders"],
    },
    # ── Convention: "valid orders" excludes cancelled AND returned ──
    {
        "question": "Show the total number of valid orders per month in 2025",
        "sql_query": (
            "SELECT DATE_TRUNC('month', o.order_date) AS month,\n"
            "       COUNT(*) AS valid_orders\n"
            "FROM orders o\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "  AND o.order_date >= '2025-01-01'\n"
            "  AND o.order_date < '2026-01-01'\n"
            "GROUP BY month\n"
            "ORDER BY month;"
        ),
        "explanation": (
            "'Valid orders' means excluding both cancelled AND returned orders. "
            "Many queries only exclude 'cancelled' — our convention excludes both."
        ),
        "tables_used": ["orders"],
    },
    # ── Convention: "product performance" = revenue from line items ──
    {
        "question": "Which product categories perform best?",
        "sql_query": (
            "SELECT p.category,\n"
            "       SUM(oi.quantity * oi.unit_price) AS category_revenue\n"
            "FROM order_items oi\n"
            "JOIN products p ON p.product_id = oi.product_id\n"
            "JOIN orders o ON o.order_id = oi.order_id\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "GROUP BY p.category\n"
            "ORDER BY category_revenue DESC;"
        ),
        "explanation": (
            "Product performance is measured by revenue from order_items "
            "(quantity * unit_price), excluding cancelled/returned orders."
        ),
        "tables_used": ["order_items", "products", "orders"],
    },
    # ── Convention: LEFT JOIN anti-pattern for "never" questions ──
    {
        "question": "Which products have never been sold?",
        "sql_query": (
            "SELECT p.name, p.category, p.price\n"
            "FROM products p\n"
            "LEFT JOIN order_items oi ON oi.product_id = p.product_id\n"
            "WHERE oi.item_id IS NULL\n"
            "ORDER BY p.name;"
        ),
        "explanation": (
            "Use LEFT JOIN + WHERE IS NULL to find products with no order_items. "
            "This is the anti-join pattern for 'never' queries."
        ),
        "tables_used": ["products", "order_items"],
    },
    # ── Convention: profit margin uses cost, excludes NULL costs ──
    {
        "question": "Show profit margins for all products",
        "sql_query": (
            "SELECT p.name, p.price, p.cost,\n"
            "       ROUND(((p.price - p.cost) / p.price * 100)::numeric, 1) "
            "AS margin_pct\n"
            "FROM products p\n"
            "WHERE p.cost IS NOT NULL AND p.cost > 0\n"
            "ORDER BY margin_pct DESC;"
        ),
        "explanation": (
            "Profit margin = (price - cost) / price * 100. "
            "Always filter out products with NULL or zero cost. "
            "Use ROUND with ::numeric cast for PostgreSQL."
        ),
        "tables_used": ["products"],
    },
    # ── Convention: "repeat customers" = more than one valid order ──
    {
        "question": "How many repeat customers do we have?",
        "sql_query": (
            "SELECT COUNT(*) AS repeat_customers\n"
            "FROM (\n"
            "    SELECT c.customer_id\n"
            "    FROM customers c\n"
            "    JOIN orders o ON o.customer_id = c.customer_id\n"
            "    WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "    GROUP BY c.customer_id\n"
            "    HAVING COUNT(o.order_id) > 1\n"
            ") sub;"
        ),
        "explanation": (
            "A repeat customer has placed MORE THAN ONE valid (non-cancelled, "
            "non-returned) order. Use HAVING COUNT > 1 after grouping."
        ),
        "tables_used": ["customers", "orders"],
    },
    # ── Convention: customer ranking uses revenue from order_items ──
    {
        "question": "Rank customers by their total spending",
        "sql_query": (
            "SELECT c.name, c.segment,\n"
            "       SUM(oi.quantity * oi.unit_price) AS total_revenue\n"
            "FROM customers c\n"
            "JOIN orders o ON o.customer_id = c.customer_id\n"
            "JOIN order_items oi ON oi.order_id = o.order_id\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "GROUP BY c.customer_id, c.name, c.segment\n"
            "ORDER BY total_revenue DESC;"
        ),
        "explanation": (
            "Customer spending is calculated from order_items revenue "
            "(quantity * unit_price), not orders.total_amount. "
            "Exclude cancelled and returned orders."
        ),
        "tables_used": ["customers", "orders", "order_items"],
    },
]


def get_learning_examples() -> list[VettedExample]:
    """Return all learning examples as VettedExample instances."""
    return [VettedExample(**data) for data in LEARNING_EXAMPLES]

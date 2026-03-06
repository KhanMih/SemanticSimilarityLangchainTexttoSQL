"""Ground truth evaluation dataset — 30 hand-curated questions with gold SQL.

Design rationale
────────────────
•  **Simple (10)**: Straightforward schema lookups. The LLM should get
   these right regardless of whether learning examples are available.

•  **Moderate (10)**: Use *ambiguous business terms* — "revenue", "completed
   orders", "net value", "active customers" — whose correct SQL interpretation
   follows conventions that are taught ONLY by the learning examples.

•  **Challenging (10)**: Multi-step queries that combine domain conventions
   with complex SQL patterns (window functions, subqueries, conditional
   aggregation).  Require both business knowledge AND SQL skill.

This split means:
  • Baseline (zero-shot): high on simple, low on moderate & challenging
  • With learning: high across all three → clear improvement
"""

from __future__ import annotations

EVALUATION_QUESTIONS: list[dict] = [
    # ═══════════════════════════════════════════════════════════════════
    #  SIMPLE — plain schema lookups (10)
    # ═══════════════════════════════════════════════════════════════════
    {
        "question": "How many customers are registered?",
        "gold_sql": "SELECT COUNT(*) AS customer_count FROM customers;",
        "difficulty": "simple",
    },
    {
        "question": "List all product names and their prices",
        "gold_sql": (
            "SELECT name, price FROM products ORDER BY name;"
        ),
        "difficulty": "simple",
    },
    {
        "question": "How many orders are in the system?",
        "gold_sql": "SELECT COUNT(*) AS order_count FROM orders;",
        "difficulty": "simple",
    },
    {
        "question": "What are the distinct product categories?",
        "gold_sql": "SELECT DISTINCT category FROM products ORDER BY category;",
        "difficulty": "simple",
    },
    {
        "question": "Show all customers from the United States",
        "gold_sql": (
            "SELECT name, email, country FROM customers "
            "WHERE country = 'US' ORDER BY name;"
        ),
        "difficulty": "simple",
    },
    {
        "question": "What is the most expensive product?",
        "gold_sql": (
            "SELECT name, price FROM products "
            "ORDER BY price DESC LIMIT 1;"
        ),
        "difficulty": "simple",
    },
    {
        "question": "How many products cost less than $100?",
        "gold_sql": (
            "SELECT COUNT(*) AS cheap_products FROM products WHERE price < 100;"
        ),
        "difficulty": "simple",
    },
    {
        "question": "How many orders have been delivered?",
        "gold_sql": (
            "SELECT COUNT(*) AS delivered_count "
            "FROM orders WHERE status = 'delivered';"
        ),
        "difficulty": "simple",
    },
    {
        "question": "What is the average product price?",
        "gold_sql": (
            "SELECT ROUND(AVG(price)::numeric, 2) AS avg_price FROM products;"
        ),
        "difficulty": "simple",
    },
    {
        "question": "How many reviews are in the database?",
        "gold_sql": "SELECT COUNT(*) AS review_count FROM reviews;",
        "difficulty": "simple",
    },

    # ═══════════════════════════════════════════════════════════════════
    #  MODERATE — ambiguous business terms (10)
    #  Correct interpretation requires conventions from learning examples
    # ═══════════════════════════════════════════════════════════════════
    {
        # Convention: "revenue" → order_items level, not orders.total_amount
        "question": "What is the total revenue?",
        "gold_sql": (
            "SELECT SUM(oi.quantity * oi.unit_price) AS total_revenue\n"
            "FROM order_items oi\n"
            "JOIN orders o ON o.order_id = oi.order_id\n"
            "WHERE o.status NOT IN ('cancelled', 'returned');"
        ),
        "difficulty": "moderate",
    },
    {
        # Convention: "completed" → shipped + delivered
        "question": "How many completed orders do we have?",
        "gold_sql": (
            "SELECT COUNT(*) AS completed_orders\n"
            "FROM orders\n"
            "WHERE status IN ('shipped', 'delivered');"
        ),
        "difficulty": "moderate",
    },
    {
        # Convention: "net order value" → total_amount - discount
        "question": "What is the average net order value?",
        "gold_sql": (
            "SELECT ROUND(AVG(o.total_amount - o.discount)::numeric, 2) "
            "AS avg_net_value\n"
            "FROM orders o\n"
            "WHERE o.status NOT IN ('cancelled', 'returned');"
        ),
        "difficulty": "moderate",
    },
    {
        # Convention: "active customers" → with non-cancelled/returned orders
        "question": "How many active customers do we have?",
        "gold_sql": (
            "SELECT COUNT(DISTINCT c.customer_id) AS active_customers\n"
            "FROM customers c\n"
            "JOIN orders o ON o.customer_id = c.customer_id\n"
            "WHERE o.status NOT IN ('cancelled', 'returned');"
        ),
        "difficulty": "moderate",
    },
    {
        # Convention: "basket size" → item count, not dollar amount
        "question": "What is the average basket size per order?",
        "gold_sql": (
            "SELECT ROUND(AVG(item_count)::numeric, 1) AS avg_basket_size\n"
            "FROM (\n"
            "    SELECT o.order_id, SUM(oi.quantity) AS item_count\n"
            "    FROM orders o\n"
            "    JOIN order_items oi ON oi.order_id = o.order_id\n"
            "    WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "    GROUP BY o.order_id\n"
            ") sub;"
        ),
        "difficulty": "moderate",
    },
    {
        # Convention: "valid orders" → exclude cancelled AND returned
        "question": "Show the number of valid orders per month in 2025",
        "gold_sql": (
            "SELECT DATE_TRUNC('month', o.order_date)::date AS month,\n"
            "       COUNT(*) AS valid_orders\n"
            "FROM orders o\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "  AND o.order_date >= '2025-01-01'\n"
            "  AND o.order_date < '2026-01-01'\n"
            "GROUP BY month\n"
            "ORDER BY month;"
        ),
        "difficulty": "moderate",
    },
    {
        # Convention: "revenue" by category → order_items level
        "question": "What is the revenue by product category?",
        "gold_sql": (
            "SELECT p.category,\n"
            "       SUM(oi.quantity * oi.unit_price) AS category_revenue\n"
            "FROM order_items oi\n"
            "JOIN products p ON p.product_id = oi.product_id\n"
            "JOIN orders o ON o.order_id = oi.order_id\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "GROUP BY p.category\n"
            "ORDER BY category_revenue DESC;"
        ),
        "difficulty": "moderate",
    },
    {
        # Convention: "repeat customers" → more than one valid order
        "question": "How many repeat customers are there?",
        "gold_sql": (
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
        "difficulty": "moderate",
    },
    {
        # Convention: customer "spending" → order_items revenue
        "question": "Who are the top 3 customers by spending?",
        "gold_sql": (
            "SELECT c.name,\n"
            "       SUM(oi.quantity * oi.unit_price) AS total_spending\n"
            "FROM customers c\n"
            "JOIN orders o ON o.customer_id = c.customer_id\n"
            "JOIN order_items oi ON oi.order_id = o.order_id\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "GROUP BY c.customer_id, c.name\n"
            "ORDER BY total_spending DESC\n"
            "LIMIT 3;"
        ),
        "difficulty": "moderate",
    },
    {
        # Convention: "fulfillment rate" → delivered / (total - cancelled)
        "question": "What is the order fulfillment rate?",
        "gold_sql": (
            "SELECT ROUND(\n"
            "    COUNT(*) FILTER (WHERE status = 'delivered')::numeric /\n"
            "    NULLIF(COUNT(*) FILTER (WHERE status != 'cancelled'), 0) * 100,\n"
            "    1\n"
            ") AS fulfillment_rate_pct\n"
            "FROM orders;"
        ),
        "difficulty": "moderate",
    },

    # ═══════════════════════════════════════════════════════════════════
    #  CHALLENGING — domain conventions + complex SQL patterns (10)
    # ═══════════════════════════════════════════════════════════════════
    {
        # revenue per customer (order_items level) + ranking
        "question": "Show customer revenue ranking with their segment",
        "gold_sql": (
            "SELECT c.name, c.segment,\n"
            "       SUM(oi.quantity * oi.unit_price) AS total_revenue,\n"
            "       RANK() OVER (ORDER BY SUM(oi.quantity * oi.unit_price) DESC) "
            "AS revenue_rank\n"
            "FROM customers c\n"
            "JOIN orders o ON o.customer_id = c.customer_id\n"
            "JOIN order_items oi ON oi.order_id = o.order_id\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "GROUP BY c.customer_id, c.name, c.segment\n"
            "ORDER BY total_revenue DESC;"
        ),
        "difficulty": "challenging",
    },
    {
        # revenue + net value combined
        "question": (
            "What is the total revenue and total net order value from "
            "completed orders?"
        ),
        "gold_sql": (
            "SELECT SUM(oi.quantity * oi.unit_price) AS total_revenue,\n"
            "       SUM(o.total_amount - o.discount) AS total_net_value\n"
            "FROM orders o\n"
            "JOIN order_items oi ON oi.order_id = o.order_id\n"
            "WHERE o.status IN ('shipped', 'delivered');"
        ),
        "difficulty": "challenging",
    },
    {
        # profit margin with cost filter
        "question": "What is the profit margin percentage for each product?",
        "gold_sql": (
            "SELECT p.name, p.price, p.cost,\n"
            "       ROUND(((p.price - p.cost) / p.price * 100)::numeric, 1) "
            "AS margin_pct\n"
            "FROM products p\n"
            "WHERE p.cost IS NOT NULL AND p.cost > 0\n"
            "ORDER BY margin_pct DESC;"
        ),
        "difficulty": "challenging",
    },
    {
        # revenue by category + percentage of total
        "question": (
            "Show each product category's revenue and its percentage of "
            "total revenue"
        ),
        "gold_sql": (
            "SELECT p.category,\n"
            "       SUM(oi.quantity * oi.unit_price) AS category_revenue,\n"
            "       ROUND(\n"
            "           SUM(oi.quantity * oi.unit_price) /\n"
            "           SUM(SUM(oi.quantity * oi.unit_price)) OVER () * 100,\n"
            "           1\n"
            "       ) AS pct_of_total\n"
            "FROM order_items oi\n"
            "JOIN products p ON p.product_id = oi.product_id\n"
            "JOIN orders o ON o.order_id = oi.order_id\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "GROUP BY p.category\n"
            "ORDER BY category_revenue DESC;"
        ),
        "difficulty": "challenging",
    },
    {
        # active customers per segment
        "question": "How many active customers are there in each segment?",
        "gold_sql": (
            "SELECT c.segment,\n"
            "       COUNT(DISTINCT c.customer_id) AS active_customers\n"
            "FROM customers c\n"
            "JOIN orders o ON o.customer_id = c.customer_id\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "GROUP BY c.segment\n"
            "ORDER BY active_customers DESC;"
        ),
        "difficulty": "challenging",
    },
    {
        # products never sold (anti-join)
        "question": "Which products have never been ordered?",
        "gold_sql": (
            "SELECT p.name, p.category, p.price\n"
            "FROM products p\n"
            "LEFT JOIN order_items oi ON oi.product_id = p.product_id\n"
            "WHERE oi.item_id IS NULL\n"
            "ORDER BY p.name;"
        ),
        "difficulty": "challenging",
    },
    {
        # average basket size by customer segment
        "question": "What is the average basket size by customer segment?",
        "gold_sql": (
            "SELECT c.segment,\n"
            "       ROUND(AVG(sub.item_count)::numeric, 1) AS avg_basket_size\n"
            "FROM (\n"
            "    SELECT o.order_id, o.customer_id,\n"
            "           SUM(oi.quantity) AS item_count\n"
            "    FROM orders o\n"
            "    JOIN order_items oi ON oi.order_id = o.order_id\n"
            "    WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "    GROUP BY o.order_id, o.customer_id\n"
            ") sub\n"
            "JOIN customers c ON c.customer_id = sub.customer_id\n"
            "GROUP BY c.segment\n"
            "ORDER BY avg_basket_size DESC;"
        ),
        "difficulty": "challenging",
    },
    {
        # net order value by country
        "question": (
            "What is the average net order value by customer country?"
        ),
        "gold_sql": (
            "SELECT c.country,\n"
            "       ROUND(AVG(o.total_amount - o.discount)::numeric, 2) "
            "AS avg_net_value\n"
            "FROM orders o\n"
            "JOIN customers c ON c.customer_id = o.customer_id\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "GROUP BY c.country\n"
            "ORDER BY avg_net_value DESC;"
        ),
        "difficulty": "challenging",
    },
    {
        # revenue by month + completed order filter
        "question": (
            "Show monthly revenue from completed orders in 2025"
        ),
        "gold_sql": (
            "SELECT DATE_TRUNC('month', o.order_date)::date AS month,\n"
            "       SUM(oi.quantity * oi.unit_price) AS monthly_revenue\n"
            "FROM orders o\n"
            "JOIN order_items oi ON oi.order_id = o.order_id\n"
            "WHERE o.status IN ('shipped', 'delivered')\n"
            "  AND o.order_date >= '2025-01-01'\n"
            "  AND o.order_date < '2026-01-01'\n"
            "GROUP BY month\n"
            "ORDER BY month;"
        ),
        "difficulty": "challenging",
    },
    {
        # conditional aggregation: valid vs total orders per customer
        "question": (
            "For each customer, show their total orders, valid orders, "
            "and valid order rate"
        ),
        "gold_sql": (
            "SELECT c.name,\n"
            "       COUNT(*) AS total_orders,\n"
            "       COUNT(*) FILTER (\n"
            "           WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "       ) AS valid_orders,\n"
            "       ROUND(\n"
            "           COUNT(*) FILTER (\n"
            "               WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "           )::numeric / COUNT(*) * 100, 1\n"
            "       ) AS valid_rate_pct\n"
            "FROM customers c\n"
            "JOIN orders o ON o.customer_id = c.customer_id\n"
            "GROUP BY c.customer_id, c.name\n"
            "ORDER BY valid_rate_pct ASC;"
        ),
        "difficulty": "challenging",
    },
]

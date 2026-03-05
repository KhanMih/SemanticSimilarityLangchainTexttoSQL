"""Built-in self-test dataset — 20 questions with gold SQL against the ecommerce sample data.

This lets you run a quick benchmark without downloading BIRD or any external dataset.
The gold SQL is hand-written and verified against the sample data from init_db.py.
"""

from __future__ import annotations

from semantic_sql.benchmark.dataset_loader import BenchmarkDataset, BenchmarkExample

SELF_TEST_EXAMPLES = [
    # ── Simple aggregations ──────────────────────────
    {
        "question": "How many customers are there?",
        "sql": "SELECT COUNT(*) AS customer_count FROM customers;",
        "difficulty": "simple",
    },
    {
        "question": "What is the total number of orders?",
        "sql": "SELECT COUNT(*) AS order_count FROM orders;",
        "difficulty": "simple",
    },
    {
        "question": "How many products are in the Electronics category?",
        "sql": "SELECT COUNT(*) AS product_count FROM products WHERE category = 'Electronics';",
        "difficulty": "simple",
    },
    {
        "question": "What is the average product price?",
        "sql": "SELECT ROUND(AVG(price)::numeric, 2) AS avg_price FROM products;",
        "difficulty": "simple",
    },
    # ── Filtering + sorting ──────────────────────────
    {
        "question": "List all customers from the US",
        "sql": "SELECT customer_id, name, email, city, segment FROM customers WHERE country = 'US' ORDER BY name;",
        "difficulty": "simple",
    },
    {
        "question": "Show all cancelled orders",
        "sql": "SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE status = 'cancelled';",
        "difficulty": "simple",
    },
    {
        "question": "Which products cost more than 100 dollars?",
        "sql": "SELECT product_id, name, category, price FROM products WHERE price > 100 ORDER BY price DESC;",
        "difficulty": "simple",
    },
    # ── Joins ────────────────────────────────────────
    {
        "question": "Who are the top 5 customers by total order amount?",
        "sql": (
            "SELECT c.name, c.email, SUM(o.total_amount) AS total_spent "
            "FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
            "WHERE o.status != 'cancelled' "
            "GROUP BY c.customer_id, c.name, c.email "
            "ORDER BY total_spent DESC LIMIT 5;"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "Show the total revenue per product category",
        "sql": (
            "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue "
            "FROM order_items oi JOIN products p ON p.product_id = oi.product_id "
            "JOIN orders o ON o.order_id = oi.order_id "
            "WHERE o.status != 'cancelled' "
            "GROUP BY p.category ORDER BY revenue DESC;"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "What is the average rating for each product?",
        "sql": (
            "SELECT p.name, ROUND(AVG(r.rating)::numeric, 2) AS avg_rating, COUNT(r.review_id) AS review_count "
            "FROM products p JOIN reviews r ON r.product_id = p.product_id "
            "GROUP BY p.product_id, p.name ORDER BY avg_rating DESC;"
        ),
        "difficulty": "moderate",
    },
    # ── LEFT JOIN / anti-join ────────────────────────
    {
        "question": "Which customers have never placed an order?",
        "sql": (
            "SELECT c.customer_id, c.name, c.email "
            "FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id "
            "WHERE o.order_id IS NULL ORDER BY c.name;"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "Which products have never been ordered?",
        "sql": (
            "SELECT p.product_id, p.name, p.category "
            "FROM products p LEFT JOIN order_items oi ON oi.product_id = p.product_id "
            "WHERE oi.item_id IS NULL ORDER BY p.name;"
        ),
        "difficulty": "moderate",
    },
    # ── Date / time queries ──────────────────────────
    {
        "question": "Show monthly revenue for 2025",
        "sql": (
            "SELECT DATE_TRUNC('month', o.order_date) AS month, "
            "SUM(o.total_amount) AS revenue "
            "FROM orders o WHERE o.status NOT IN ('cancelled', 'returned') "
            "AND o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01' "
            "GROUP BY month ORDER BY month;"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "How many orders were placed each month?",
        "sql": (
            "SELECT DATE_TRUNC('month', order_date) AS month, COUNT(*) AS order_count "
            "FROM orders GROUP BY month ORDER BY month;"
        ),
        "difficulty": "moderate",
    },
    # ── Calculations ─────────────────────────────────
    {
        "question": "Which products have the highest profit margin?",
        "sql": (
            "SELECT name, category, price, cost, "
            "ROUND(((price - cost) / price * 100)::numeric, 1) AS margin_pct "
            "FROM products WHERE cost IS NOT NULL AND cost > 0 "
            "ORDER BY margin_pct DESC LIMIT 10;"
        ),
        "difficulty": "challenging",
    },
    {
        "question": "What is the average order value by customer segment?",
        "sql": (
            "SELECT c.segment, ROUND(AVG(o.total_amount)::numeric, 2) AS avg_order_value, "
            "COUNT(o.order_id) AS order_count "
            "FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
            "WHERE o.status != 'cancelled' "
            "GROUP BY c.segment ORDER BY avg_order_value DESC;"
        ),
        "difficulty": "challenging",
    },
    # ── Multi-join / complex ─────────────────────────
    {
        "question": "Show the top 3 best-selling products by quantity sold",
        "sql": (
            "SELECT p.name, p.category, SUM(oi.quantity) AS total_qty "
            "FROM order_items oi JOIN products p ON p.product_id = oi.product_id "
            "JOIN orders o ON o.order_id = oi.order_id "
            "WHERE o.status != 'cancelled' "
            "GROUP BY p.product_id, p.name, p.category "
            "ORDER BY total_qty DESC LIMIT 3;"
        ),
        "difficulty": "challenging",
    },
    {
        "question": "Which country generates the most revenue?",
        "sql": (
            "SELECT c.country, SUM(o.total_amount) AS revenue "
            "FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
            "WHERE o.status != 'cancelled' "
            "GROUP BY c.country ORDER BY revenue DESC LIMIT 1;"
        ),
        "difficulty": "challenging",
    },
    {
        "question": "What is the total discount given across all orders?",
        "sql": "SELECT SUM(discount) AS total_discount FROM orders;",
        "difficulty": "simple",
    },
    {
        "question": "Show customers who placed more than one order",
        "sql": (
            "SELECT c.name, c.email, COUNT(o.order_id) AS order_count "
            "FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
            "GROUP BY c.customer_id, c.name, c.email "
            "HAVING COUNT(o.order_id) > 1 ORDER BY order_count DESC;"
        ),
        "difficulty": "challenging",
    },
]


def build_self_test_dataset() -> BenchmarkDataset:
    """Build the built-in self-test dataset (20 questions against ecommerce data)."""
    examples = [
        BenchmarkExample(
            question=item["question"],
            gold_sql=item["sql"],
            db_id="ecommerce",
            difficulty=item["difficulty"],
            question_id=i,
        )
        for i, item in enumerate(SELF_TEST_EXAMPLES)
    ]
    return BenchmarkDataset(name="self-test-ecommerce", examples=examples)

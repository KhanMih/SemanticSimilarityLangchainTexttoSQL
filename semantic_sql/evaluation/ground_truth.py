"""Ground truth evaluation dataset — 30 questions with verified gold SQL.

These questions are designed to test patterns where a zero-shot agent
(no few-shot examples) typically struggles, but a dynamic few-shot agent
(with learned examples in the vector store) performs significantly better.

Key difficulty patterns tested:
  - Excluding cancelled/returned orders in revenue calculations
  - PostgreSQL ::numeric casting for ROUND
  - DATE_TRUNC for temporal aggregation
  - LEFT JOIN + IS NULL anti-join pattern
  - Proper GROUP BY with all non-aggregated columns
  - Multi-table JOINs with correct column selection
  - HAVING clause for group-level filtering
  - Profit margin calculations with NULL guards
"""

from __future__ import annotations

from semantic_sql.benchmark.dataset_loader import BenchmarkDataset, BenchmarkExample

EVALUATION_QUESTIONS: list[dict[str, str]] = [
    # ── Simple (10) ─────────────────────────────────────────────
    {
        "question": "How many customers are registered in the system?",
        "gold_sql": "SELECT COUNT(*) AS customer_count FROM customers;",
        "difficulty": "simple",
    },
    {
        "question": "What is the total number of products available?",
        "gold_sql": "SELECT COUNT(*) AS product_count FROM products;",
        "difficulty": "simple",
    },
    {
        "question": "How many orders have been placed in total?",
        "gold_sql": "SELECT COUNT(*) AS order_count FROM orders;",
        "difficulty": "simple",
    },
    {
        "question": "List all unique product categories",
        "gold_sql": "SELECT DISTINCT category FROM products ORDER BY category;",
        "difficulty": "simple",
    },
    {
        "question": "What is the most expensive product?",
        "gold_sql": "SELECT name, price FROM products ORDER BY price DESC LIMIT 1;",
        "difficulty": "simple",
    },
    {
        "question": "Show all customers from the UK",
        "gold_sql": (
            "SELECT name, email, city FROM customers "
            "WHERE country = 'UK' ORDER BY name;"
        ),
        "difficulty": "simple",
    },
    {
        "question": "How many products are in the Furniture category?",
        "gold_sql": (
            "SELECT COUNT(*) AS furniture_count FROM products "
            "WHERE category = 'Furniture';"
        ),
        "difficulty": "simple",
    },
    {
        "question": "What is the total discount given across all orders?",
        "gold_sql": "SELECT SUM(discount) AS total_discount FROM orders;",
        "difficulty": "simple",
    },
    {
        "question": "How many orders have been delivered?",
        "gold_sql": (
            "SELECT COUNT(*) AS delivered_count FROM orders "
            "WHERE status = 'delivered';"
        ),
        "difficulty": "simple",
    },
    {
        "question": "List products priced under 50 dollars",
        "gold_sql": (
            "SELECT name, category, price FROM products "
            "WHERE price < 50 ORDER BY price;"
        ),
        "difficulty": "simple",
    },
    # ── Moderate (10) ───────────────────────────────────────────
    {
        "question": "What is the total revenue from non-cancelled orders?",
        "gold_sql": (
            "SELECT SUM(total_amount) AS total_revenue "
            "FROM orders WHERE status != 'cancelled';"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "What is the average order value excluding cancelled orders?",
        "gold_sql": (
            "SELECT ROUND(AVG(total_amount)::numeric, 2) AS avg_order_value "
            "FROM orders WHERE status != 'cancelled';"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "What is the average product price by category?",
        "gold_sql": (
            "SELECT category, ROUND(AVG(price)::numeric, 2) AS avg_price "
            "FROM products GROUP BY category ORDER BY avg_price DESC;"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "How many orders did each customer place?",
        "gold_sql": (
            "SELECT c.name, COUNT(o.order_id) AS order_count "
            "FROM customers c "
            "LEFT JOIN orders o ON o.customer_id = c.customer_id "
            "GROUP BY c.customer_id, c.name "
            "ORDER BY order_count DESC;"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "Show the total quantity sold for each product",
        "gold_sql": (
            "SELECT p.name, SUM(oi.quantity) AS total_sold "
            "FROM products p "
            "JOIN order_items oi ON oi.product_id = p.product_id "
            "GROUP BY p.product_id, p.name "
            "ORDER BY total_sold DESC;"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "What is the average review rating for each product?",
        "gold_sql": (
            "SELECT p.name, "
            "ROUND(AVG(r.rating)::numeric, 2) AS avg_rating, "
            "COUNT(r.review_id) AS review_count "
            "FROM products p "
            "JOIN reviews r ON r.product_id = p.product_id "
            "GROUP BY p.product_id, p.name "
            "ORDER BY avg_rating DESC;"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "Show the revenue by product category from non-cancelled orders",
        "gold_sql": (
            "SELECT p.category, "
            "SUM(oi.quantity * oi.unit_price) AS revenue "
            "FROM order_items oi "
            "JOIN products p ON p.product_id = oi.product_id "
            "JOIN orders o ON o.order_id = oi.order_id "
            "WHERE o.status != 'cancelled' "
            "GROUP BY p.category "
            "ORDER BY revenue DESC;"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "Which products have more than one review?",
        "gold_sql": (
            "SELECT p.name, COUNT(r.review_id) AS review_count "
            "FROM products p "
            "JOIN reviews r ON r.product_id = p.product_id "
            "GROUP BY p.product_id, p.name "
            "HAVING COUNT(r.review_id) > 1 "
            "ORDER BY review_count DESC;"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "Show orders placed in February 2025",
        "gold_sql": (
            "SELECT order_id, customer_id, order_date, status, total_amount "
            "FROM orders "
            "WHERE order_date >= '2025-02-01' AND order_date < '2025-03-01' "
            "ORDER BY order_date;"
        ),
        "difficulty": "moderate",
    },
    {
        "question": "What is the total revenue by customer country excluding cancelled orders?",
        "gold_sql": (
            "SELECT c.country, SUM(o.total_amount) AS revenue "
            "FROM customers c "
            "JOIN orders o ON o.customer_id = c.customer_id "
            "WHERE o.status != 'cancelled' "
            "GROUP BY c.country "
            "ORDER BY revenue DESC;"
        ),
        "difficulty": "moderate",
    },
    # ── Challenging (10) ────────────────────────────────────────
    {
        "question": "Which customers have never placed an order?",
        "gold_sql": (
            "SELECT c.name, c.email "
            "FROM customers c "
            "LEFT JOIN orders o ON o.customer_id = c.customer_id "
            "WHERE o.order_id IS NULL "
            "ORDER BY c.name;"
        ),
        "difficulty": "challenging",
    },
    {
        "question": "Which products have never been ordered?",
        "gold_sql": (
            "SELECT p.name, p.category "
            "FROM products p "
            "LEFT JOIN order_items oi ON oi.product_id = p.product_id "
            "WHERE oi.item_id IS NULL "
            "ORDER BY p.name;"
        ),
        "difficulty": "challenging",
    },
    {
        "question": "Which products have no reviews?",
        "gold_sql": (
            "SELECT p.name, p.category "
            "FROM products p "
            "LEFT JOIN reviews r ON r.product_id = p.product_id "
            "WHERE r.review_id IS NULL "
            "ORDER BY p.name;"
        ),
        "difficulty": "challenging",
    },
    {
        "question": "Show monthly order count and revenue for 2025 excluding cancelled and returned orders",
        "gold_sql": (
            "SELECT DATE_TRUNC('month', o.order_date) AS month, "
            "COUNT(*) AS order_count, "
            "SUM(o.total_amount) AS revenue "
            "FROM orders o "
            "WHERE o.status NOT IN ('cancelled', 'returned') "
            "AND o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01' "
            "GROUP BY month "
            "ORDER BY month;"
        ),
        "difficulty": "challenging",
    },
    {
        "question": "What is the profit margin percentage for each product?",
        "gold_sql": (
            "SELECT name, category, price, cost, "
            "ROUND(((price - cost) / price * 100)::numeric, 1) AS margin_pct "
            "FROM products "
            "WHERE cost IS NOT NULL AND cost > 0 "
            "ORDER BY margin_pct DESC;"
        ),
        "difficulty": "challenging",
    },
    {
        "question": "Who are the top 3 customers by total spending excluding cancelled orders?",
        "gold_sql": (
            "SELECT c.name, SUM(o.total_amount) AS total_spent "
            "FROM customers c "
            "JOIN orders o ON o.customer_id = c.customer_id "
            "WHERE o.status != 'cancelled' "
            "GROUP BY c.customer_id, c.name "
            "ORDER BY total_spent DESC LIMIT 3;"
        ),
        "difficulty": "challenging",
    },
    {
        "question": "Show the top 5 products by total revenue from non-cancelled orders",
        "gold_sql": (
            "SELECT p.name, SUM(oi.quantity * oi.unit_price) AS revenue "
            "FROM order_items oi "
            "JOIN products p ON p.product_id = oi.product_id "
            "JOIN orders o ON o.order_id = oi.order_id "
            "WHERE o.status != 'cancelled' "
            "GROUP BY p.product_id, p.name "
            "ORDER BY revenue DESC LIMIT 5;"
        ),
        "difficulty": "challenging",
    },
    {
        "question": "What is the average order value by customer segment excluding cancelled orders?",
        "gold_sql": (
            "SELECT c.segment, "
            "ROUND(AVG(o.total_amount)::numeric, 2) AS avg_order_value, "
            "COUNT(o.order_id) AS order_count "
            "FROM customers c "
            "JOIN orders o ON o.customer_id = c.customer_id "
            "WHERE o.status != 'cancelled' "
            "GROUP BY c.segment "
            "ORDER BY avg_order_value DESC;"
        ),
        "difficulty": "challenging",
    },
    {
        "question": "Show customers who placed more than one order",
        "gold_sql": (
            "SELECT c.name, c.email, COUNT(o.order_id) AS order_count "
            "FROM customers c "
            "JOIN orders o ON o.customer_id = c.customer_id "
            "GROUP BY c.customer_id, c.name, c.email "
            "HAVING COUNT(o.order_id) > 1 "
            "ORDER BY order_count DESC;"
        ),
        "difficulty": "challenging",
    },
    {
        "question": "What is the total shipping cost by customer country?",
        "gold_sql": (
            "SELECT c.country, SUM(o.shipping_cost) AS total_shipping "
            "FROM customers c "
            "JOIN orders o ON o.customer_id = c.customer_id "
            "GROUP BY c.country "
            "ORDER BY total_shipping DESC;"
        ),
        "difficulty": "challenging",
    },
]


def build_evaluation_dataset() -> BenchmarkDataset:
    """Build the RAGAS evaluation dataset (30 questions against ecommerce data)."""
    examples = [
        BenchmarkExample(
            question=item["question"],
            gold_sql=item["gold_sql"],
            db_id="ecommerce",
            difficulty=item["difficulty"],
            question_id=i,
        )
        for i, item in enumerate(EVALUATION_QUESTIONS)
    ]
    return BenchmarkDataset(name="ragas-evaluation", examples=examples)

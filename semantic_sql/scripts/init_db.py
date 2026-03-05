"""Database initialisation — creates tables, pgvector extension, sample data, and seed examples.

Run via CLI:  semantic-sql setup init
Or directly:  python -m semantic_sql.scripts.init_db
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from semantic_sql.config import settings
from semantic_sql.db.connection import get_engine

logger = logging.getLogger(__name__)

# ── Sample ecommerce schema ─────────────────────────────────

SAMPLE_DDL = """
-- Enable pgvector extension (required for the example store)
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Ecommerce sample tables ────────────────────────────────

CREATE TABLE IF NOT EXISTS customers (
    customer_id   SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    email         VARCHAR(150) UNIQUE NOT NULL,
    city          VARCHAR(80),
    country       VARCHAR(60) DEFAULT 'US',
    segment       VARCHAR(30) CHECK (segment IN ('consumer','corporate','enterprise')),
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS products (
    product_id    SERIAL PRIMARY KEY,
    name          VARCHAR(200) NOT NULL,
    category      VARCHAR(80),
    subcategory   VARCHAR(80),
    price         NUMERIC(10,2) NOT NULL,
    cost          NUMERIC(10,2),
    stock_qty     INT DEFAULT 0,
    is_active     BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   INT REFERENCES customers(customer_id),
    order_date    DATE NOT NULL,
    status        VARCHAR(20) DEFAULT 'pending'
                    CHECK (status IN ('pending','shipped','delivered','cancelled','returned')),
    total_amount  NUMERIC(12,2),
    discount      NUMERIC(5,2) DEFAULT 0,
    shipping_cost NUMERIC(8,2) DEFAULT 0
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id       SERIAL PRIMARY KEY,
    order_id      INT REFERENCES orders(order_id),
    product_id    INT REFERENCES products(product_id),
    quantity      INT NOT NULL CHECK (quantity > 0),
    unit_price    NUMERIC(10,2) NOT NULL,
    line_total    NUMERIC(12,2) GENERATED ALWAYS AS (quantity * unit_price) STORED
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id     SERIAL PRIMARY KEY,
    product_id    INT REFERENCES products(product_id),
    customer_id   INT REFERENCES customers(customer_id),
    rating        INT CHECK (rating BETWEEN 1 AND 5),
    review_text   TEXT,
    created_at    TIMESTAMP DEFAULT NOW()
);
"""

SAMPLE_DATA = """
-- Customers
INSERT INTO customers (name, email, city, country, segment) VALUES
    ('Alice Johnson',   'alice@example.com',   'New York',   'US', 'consumer'),
    ('Bob Smith',       'bob@example.com',     'London',     'UK', 'corporate'),
    ('Carol Williams',  'carol@example.com',   'San Francisco','US','enterprise'),
    ('David Brown',     'david@example.com',   'Toronto',    'CA', 'consumer'),
    ('Eve Davis',       'eve@example.com',     'Berlin',     'DE', 'corporate'),
    ('Frank Miller',    'frank@example.com',   'New York',   'US', 'enterprise'),
    ('Grace Wilson',    'grace@example.com',   'Sydney',     'AU', 'consumer'),
    ('Henry Taylor',    'henry@example.com',   'London',     'UK', 'consumer'),
    ('Ivy Anderson',    'ivy@example.com',     'Chicago',    'US', 'corporate'),
    ('Jack Thomas',     'jack@example.com',    'Vancouver',  'CA', 'enterprise')
ON CONFLICT (email) DO NOTHING;

-- Products
INSERT INTO products (name, category, subcategory, price, cost, stock_qty) VALUES
    ('Laptop Pro 15',        'Electronics', 'Laptops',     1299.99, 900.00, 45),
    ('Wireless Mouse',       'Electronics', 'Accessories',   29.99,  12.00, 200),
    ('USB-C Hub',            'Electronics', 'Accessories',   49.99,  22.00, 150),
    ('Standing Desk',        'Furniture',   'Desks',        599.99, 350.00, 30),
    ('Ergonomic Chair',      'Furniture',   'Chairs',       449.99, 280.00, 25),
    ('Monitor 27"',          'Electronics', 'Monitors',     399.99, 250.00, 60),
    ('Keyboard Mechanical',  'Electronics', 'Accessories',   89.99,  45.00, 120),
    ('Webcam HD',            'Electronics', 'Accessories',   69.99,  30.00, 80),
    ('Desk Lamp LED',        'Furniture',   'Lighting',      39.99,  18.00, 100),
    ('Cable Management Kit', 'Furniture',   'Accessories',   24.99,  10.00, 200)
ON CONFLICT DO NOTHING;

-- Orders
INSERT INTO orders (customer_id, order_date, status, total_amount, discount, shipping_cost) VALUES
    (1, '2025-01-15', 'delivered',  1329.98, 0,    15.00),
    (2, '2025-01-20', 'delivered',   599.99, 50.00, 0.00),
    (3, '2025-02-01', 'shipped',    1749.97, 0,    25.00),
    (1, '2025-02-10', 'delivered',    89.99, 0,     5.00),
    (4, '2025-02-14', 'delivered',   449.99, 25.00, 0.00),
    (5, '2025-03-01', 'pending',     129.97, 0,    10.00),
    (6, '2025-03-05', 'shipped',    1299.99, 100.00,0.00),
    (7, '2025-03-10', 'cancelled',   399.99, 0,    15.00),
    (2, '2025-03-15', 'delivered',   159.97, 10.00, 5.00),
    (8, '2025-03-20', 'pending',     599.99, 0,    20.00),
    (3, '2025-04-01', 'delivered',    69.99, 0,     5.00),
    (9, '2025-04-05', 'shipped',     949.98, 0,    15.00),
    (10,'2025-04-10', 'delivered',  1749.97, 150.00,0.00)
ON CONFLICT DO NOTHING;

-- Order Items
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
    (1, 1, 1, 1299.99),
    (1, 2, 1,   29.99),
    (2, 4, 1,  599.99),
    (3, 1, 1, 1299.99),
    (3, 5, 1,  449.99),
    (4, 7, 1,   89.99),
    (5, 5, 1,  449.99),
    (6, 2, 1,   29.99),
    (6, 3, 1,   49.99),
    (6, 8, 1,   69.99),
    (7, 1, 1, 1299.99),
    (8, 6, 1,  399.99),
    (9, 7, 1,   89.99),
    (9, 8, 1,   69.99),
    (10,4, 1,  599.99),
    (11,8, 1,   69.99),
    (12,6, 1,  399.99),
    (12,5, 1,  449.99),
    (13,1, 1, 1299.99),
    (13,5, 1,  449.99)
ON CONFLICT DO NOTHING;

-- Reviews
INSERT INTO reviews (product_id, customer_id, rating, review_text) VALUES
    (1, 1, 5, 'Amazing laptop, super fast and great display'),
    (1, 3, 4, 'Good performance but runs a bit hot'),
    (4, 2, 5, 'Best standing desk I have owned'),
    (5, 5, 4, 'Very comfortable, worth the price'),
    (2, 1, 3, 'Decent mouse, nothing special'),
    (6, 8, 5, 'Crystal clear display, perfect for coding'),
    (7, 4, 4, 'Great tactile feedback'),
    (5, 3, 5, 'Perfect ergonomic support for long hours')
ON CONFLICT DO NOTHING;
"""

# ── Seed vetted examples (bootstrap behavioral memory) ──────

SEED_EXAMPLES = [
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
        "explanation": "Excludes cancelled orders; groups by customer and sums total_amount.",
        "tables_used": ["customers", "orders"],
    },
    {
        "question": "Show me monthly revenue for 2025",
        "sql_query": (
            "SELECT DATE_TRUNC('month', o.order_date) AS month,\n"
            "       COUNT(DISTINCT o.order_id)         AS order_count,\n"
            "       SUM(o.total_amount)                AS revenue\n"
            "FROM orders o\n"
            "WHERE o.status NOT IN ('cancelled', 'returned')\n"
            "  AND o.order_date >= '2025-01-01'\n"
            "  AND o.order_date <  '2026-01-01'\n"
            "GROUP BY month\n"
            "ORDER BY month;"
        ),
        "explanation": "Uses DATE_TRUNC for monthly aggregation, excludes cancelled/returned.",
        "tables_used": ["orders"],
    },
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
        "explanation": "Margin = (price - cost) / price * 100. Filters out products without cost data.",
        "tables_used": ["products"],
    },
    {
        "question": "What is the average rating for each product category?",
        "sql_query": (
            "SELECT p.category,\n"
            "       COUNT(r.review_id)        AS review_count,\n"
            "       ROUND(AVG(r.rating), 2)   AS avg_rating\n"
            "FROM products p\n"
            "JOIN reviews r ON r.product_id = p.product_id\n"
            "GROUP BY p.category\n"
            "ORDER BY avg_rating DESC;"
        ),
        "explanation": "Joins products with reviews to aggregate ratings by category.",
        "tables_used": ["products", "reviews"],
    },
    {
        "question": "List customers who have never placed an order",
        "sql_query": (
            "SELECT c.customer_id, c.name, c.email, c.city, c.segment\n"
            "FROM customers c\n"
            "LEFT JOIN orders o ON o.customer_id = c.customer_id\n"
            "WHERE o.order_id IS NULL\n"
            "ORDER BY c.name;"
        ),
        "explanation": "LEFT JOIN + WHERE NULL pattern to find customers with zero orders.",
        "tables_used": ["customers", "orders"],
    },
]


def init_database(
    *,
    load_sample_data: bool = True,
    seed_examples: bool = True,
) -> None:
    """Create schema, load sample data, and seed behavioral memory."""
    engine = get_engine()

    # 1. Create schema + pgvector extension
    logger.info("Creating database schema...")
    with engine.connect() as conn:
        for statement in SAMPLE_DDL.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()
    logger.info("Schema created.")

    # 2. Load sample data
    if load_sample_data:
        logger.info("Loading sample ecommerce data...")
        with engine.connect() as conn:
            for statement in SAMPLE_DATA.split(";"):
                stmt = statement.strip()
                if stmt:
                    try:
                        conn.execute(text(stmt))
                    except Exception as exc:
                        logger.debug("Skipping (likely duplicate): %s", exc)
            conn.commit()
        logger.info("Sample data loaded.")

    # 3. Seed behavioral memory
    if seed_examples:
        logger.info("Seeding behavioral memory with %d examples...", len(SEED_EXAMPLES))
        from semantic_sql.memory.example_store import ExampleStore
        from semantic_sql.models.schemas import VettedExample

        store = ExampleStore()
        for ex_data in SEED_EXAMPLES:
            example = VettedExample(**ex_data)
            store.add_example(example)
        logger.info("Behavioral memory seeded.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_database()

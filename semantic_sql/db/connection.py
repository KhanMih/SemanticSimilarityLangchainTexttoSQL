"""PostgreSQL connection management via SQLAlchemy."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from semantic_sql.config import settings

_engine: Engine | None = None


def get_engine(url: str | None = None) -> Engine:
    """Return a singleton SQLAlchemy engine for the business database."""
    global _engine
    if _engine is None or url is not None:
        _engine = create_engine(
            url or settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def test_connection(url: str | None = None) -> bool:
    """Verify the database is reachable."""
    try:
        engine = get_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

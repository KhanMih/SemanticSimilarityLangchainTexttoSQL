"""Central configuration loaded from environment / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM / Embeddings ─────────────────────────────────
    google_api_key: str = ""
    default_llm_model: str = "gemini-2.5-flash"
    default_embedding_model: str = "models/gemini-embedding-001"

    # ── PostgreSQL (business data) ────────────────────────
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/semantic_sql"

    # ── pgvector store (few-shot examples) ────────────────
    vector_store_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/semantic_sql"
    vector_store_collection: str = "vetted_examples"

    # ── Langfuse ──────────────────────────────────────────
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── Agent tuning ──────────────────────────────────────
    few_shot_k: int = 3
    max_prompt_tokens: int = 3500
    similarity_dedup_threshold: float = 0.95
    sql_execution_timeout: int = 30

    # ── Schema inspector ──────────────────────────────────
    schema_tables: list[str] | None = None  # None = auto-discover all tables
    schema_include_sample_rows: int = 3

    # ── Feedback loop ─────────────────────────────────────
    feedback_score_name: str = "quality"
    feedback_positive_threshold: float = 1.0  # score >= this counts as "positive"
    feedback_poll_interval: int = 60  # seconds between Langfuse polling cycles
    feedback_auto_validate_sql: bool = True


settings = Settings()

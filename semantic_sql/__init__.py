"""semantic-sql — Self-Improving Text-to-SQL Agent via Dynamic Few-Shot Injection."""

from semantic_sql.agent.sql_agent import TextToSQLAgent
from semantic_sql.config import Settings, settings
from semantic_sql.feedback.annotation_handler import AnnotationHandler
from semantic_sql.memory.example_store import ExampleStore
from semantic_sql.pipeline.feedback_pipeline import FeedbackPipeline
from semantic_sql.pipeline.query_pipeline import QueryPipeline

__all__ = [
    "AnnotationHandler",
    "ExampleStore",
    "FeedbackPipeline",
    "QueryPipeline",
    "Settings",
    "TextToSQLAgent",
    "settings",
]

"""Behavioral Memory — wraps LangChain SemanticSimilarityExampleSelector with pgvector.

This is the heart of the dynamic few-shot injection system.  Vetted examples
live in a pgvector-backed vector store and are retrieved at inference time to
give the agent "Institutional Memory".
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from semantic_sql.config import settings
from semantic_sql.models.schemas import VettedExample

logger = logging.getLogger(__name__)


class ExampleStore:
    """Manages the lifecycle of vetted few-shot examples in pgvector."""

    def __init__(
        self,
        connection_url: str | None = None,
        collection_name: str | None = None,
        embeddings: Embeddings | None = None,
    ):
        self._connection_url = connection_url or settings.vector_store_url
        self._collection_name = collection_name or settings.vector_store_collection
        self._embeddings = embeddings or GoogleGenerativeAIEmbeddings(
            model=settings.default_embedding_model,
            google_api_key=settings.google_api_key,
        )
        self._selector: SemanticSimilarityExampleSelector | None = None
        self._vectorstore: PGVector | None = None

    @property
    def vectorstore(self) -> PGVector:
        if self._vectorstore is None:
            self._vectorstore = PGVector(
                collection_name=self._collection_name,
                connection=self._connection_url,
                embeddings=self._embeddings,
                use_jsonb=True,
            )
        return self._vectorstore

    @property
    def selector(self) -> SemanticSimilarityExampleSelector:
        if self._selector is None:
            self._selector = SemanticSimilarityExampleSelector(
                vectorstore=self.vectorstore,
                k=settings.few_shot_k,
                input_keys=["question"],
            )
        return self._selector

    def select_examples(self, question: str, k: int | None = None) -> list[VettedExample]:
        """Retrieve the top-k most similar vetted examples for a question."""
        sel = self.selector
        if k is not None and k != sel.k:
            sel.k = k

        raw_examples: list[dict[str, Any]] = sel.select_examples({"question": question})
        return [
            VettedExample(
                question=ex["question"],
                sql_query=ex["sql_query"],
                explanation=ex.get("explanation", ""),
                tables_used=ex.get("tables_used", "").split(",") if ex.get("tables_used") else [],
            )
            for ex in raw_examples
        ]

    def add_example(self, example: VettedExample) -> str:
        """Add a single vetted example to the vector store."""
        doc = {
            "question": example.question,
            "sql_query": example.sql_query,
            "explanation": example.explanation,
            "tables_used": ",".join(example.tables_used),
        }
        self.selector.add_example(doc)
        logger.info("Added vetted example: %s", example.question[:80])
        return example.id

    def add_examples_bulk(self, examples: list[VettedExample]) -> int:
        """Bulk-add a list of vetted examples. Returns count added."""
        added = 0
        for ex in examples:
            self.add_example(ex)
            added += 1
        return added

    def similarity_score(self, question: str) -> float:
        """Return the max similarity score for a question against the store."""
        docs = self.vectorstore.similarity_search_with_score(question, k=1)
        if not docs:
            return 0.0
        _doc, score = docs[0]
        return float(score)

    def count(self) -> int:
        """Approximate count of examples in the store."""
        try:
            docs = self.vectorstore.similarity_search("test", k=10000)
            return len(docs)
        except Exception:
            return 0

"""Benchmark strategies — the 5 approaches we compare.

Each strategy takes a question + schema context and returns generated SQL.
The only difference between them is HOW few-shot examples are selected.

Strategy 1: Zero-shot        — no examples at all
Strategy 2: Static few-shot  — same fixed examples for every question
Strategy 3: Random few-shot  — randomly sampled examples per question
Strategy 4: Dynamic few-shot — SemanticSimilarity-selected examples (our approach)
Strategy 5: Dynamic + feedback — same as #4 but with iteratively grown memory
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_google_genai import ChatGoogleGenerativeAI

from semantic_sql.agent.prompt_builder import build_prompt
from semantic_sql.agent.sql_agent import _clean_sql
from semantic_sql.config import settings
from semantic_sql.memory.example_store import ExampleStore
from semantic_sql.models.schemas import PromptPayload, TableSchema, VettedExample

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    strategy_name: str
    question: str
    generated_sql: str
    examples_used: list[VettedExample] = field(default_factory=list)
    error: str | None = None


class BaseStrategy(ABC):
    """Base class for all benchmark strategies."""

    name: str = "base"

    def __init__(self, llm: ChatGoogleGenerativeAI | None = None):
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.default_llm_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )

    @abstractmethod
    def get_examples(self, question: str) -> list[VettedExample]:
        ...

    def generate_sql(
        self,
        question: str,
        schemas: list[TableSchema],
    ) -> StrategyResult:
        try:
            examples = self.get_examples(question)
            payload = PromptPayload(
                behavioral_examples=examples,
                schema_context=schemas,
                question=question,
            )
            system_msg, user_msg = build_prompt(payload)

            raw = self.llm.invoke(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
            ).content
            sql = _clean_sql(raw)

            return StrategyResult(
                strategy_name=self.name,
                question=question,
                generated_sql=sql,
                examples_used=examples,
            )
        except Exception as exc:
            return StrategyResult(
                strategy_name=self.name,
                question=question,
                generated_sql="",
                error=str(exc),
            )


class ZeroShotStrategy(BaseStrategy):
    """Strategy 1: No examples. Just schema + question."""

    name = "zero_shot"

    def get_examples(self, question: str) -> list[VettedExample]:
        return []


class StaticFewShotStrategy(BaseStrategy):
    """Strategy 2: Same fixed examples for every question.

    Uses the first k seed examples regardless of the question.
    """

    name = "static_few_shot"

    def __init__(
        self,
        fixed_examples: list[VettedExample] | None = None,
        k: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fixed_examples = (fixed_examples or [])[:k]

    def get_examples(self, question: str) -> list[VettedExample]:
        return self.fixed_examples


class RandomFewShotStrategy(BaseStrategy):
    """Strategy 3: Randomly sampled examples per question.

    Draws k random examples from the pool for each question.
    """

    name = "random_few_shot"

    def __init__(
        self,
        example_pool: list[VettedExample] | None = None,
        k: int = 3,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.example_pool = example_pool or []
        self.k = k
        self.rng = random.Random(seed)

    def get_examples(self, question: str) -> list[VettedExample]:
        if len(self.example_pool) <= self.k:
            return self.example_pool
        return self.rng.sample(self.example_pool, self.k)


class DynamicFewShotStrategy(BaseStrategy):
    """Strategy 4: SemanticSimilarity-selected examples (OUR APPROACH).

    Uses the ExampleStore backed by pgvector to find the most relevant
    vetted examples for each question.
    """

    name = "dynamic_few_shot"

    def __init__(
        self,
        store: ExampleStore | None = None,
        k: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.store = store or ExampleStore()
        self.k = k

    def get_examples(self, question: str) -> list[VettedExample]:
        return self.store.select_examples(question, k=self.k)


class DynamicWithFeedbackStrategy(BaseStrategy):
    """Strategy 5: Dynamic few-shot + simulated feedback loop.

    Same as Strategy 4, but after each correct prediction the {question, SQL}
    pair is added to the store (simulating the SME approval workflow).
    This tests whether the memory improves over time.
    """

    name = "dynamic_with_feedback"

    def __init__(
        self,
        store: ExampleStore | None = None,
        k: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.store = store or ExampleStore()
        self.k = k

    def get_examples(self, question: str) -> list[VettedExample]:
        return self.store.select_examples(question, k=self.k)

    def learn_from_correct(self, question: str, gold_sql: str) -> None:
        """Simulate the feedback loop: inject a correct example into memory."""
        from semantic_sql.memory.deduplication import Deduplicator

        example = VettedExample(question=question, sql_query=gold_sql)
        dedup = Deduplicator(self.store)
        dedup.add_if_unique(example)


ALL_STRATEGIES = {
    "zero_shot": ZeroShotStrategy,
    "static_few_shot": StaticFewShotStrategy,
    "random_few_shot": RandomFewShotStrategy,
    "dynamic_few_shot": DynamicFewShotStrategy,
    "dynamic_with_feedback": DynamicWithFeedbackStrategy,
}

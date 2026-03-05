"""RAGAS-based evaluation module for the semantic-sql agent.

Demonstrates the value of real-time learning by comparing:
  Phase 1 (baseline): Zero-shot — no examples, agent relies only on schema
  Phase 2 (learned):  Dynamic few-shot — agent retrieves semantically similar
                      vetted examples from the vector store

Usage:
  semantic-sql evaluate run
  semantic-sql evaluate run --output results.json
"""

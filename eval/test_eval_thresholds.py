"""
Pytest wrapper around the same evaluation functions run_eval.py uses for the
human-readable report -- so the "is this good enough" threshold check and
the report are backed by one implementation, not two.

Thresholds are intentionally loose (this is a small, hand-written golden
set and an LLM-based pipeline, not a strict regression suite) -- the point
is to catch a genuinely broken pipeline (e.g. retrieval returning nothing,
routing sending everything to the wrong agent), not to chase a specific
score. Skips cleanly if the stack (Postgres/mocks/Gemini) isn't available --
see backend/tests/conftest.py's `services_available` fixture, reused here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))  # allow `import backend.app...` when run standalone

from backend.tests.conftest import services_available, settings  # noqa: E402,F401 - reused fixtures

from eval.run_eval import evaluate_agents, evaluate_retrieval  # noqa: E402


def test_retrieval_metrics_meet_minimum_bar(services_available):
    results = evaluate_retrieval()
    assert results["mean_recall_at_k"] >= 0.5, "retrieval is missing the expected source document for most queries"
    assert results["mrr"] >= 0.5, "the first relevant result usually isn't near the top of the ranking"


def test_agent_routing_meets_minimum_bar(services_available):
    results = evaluate_agents()
    assert results["routing_accuracy"] >= 0.7, "supervisor is misrouting most golden queries"
    assert results["task_completion_rate"] >= 0.6, "agent answers are missing expected content/behavior"

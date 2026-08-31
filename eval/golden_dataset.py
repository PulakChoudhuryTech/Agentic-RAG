"""Loads data/golden/golden_dataset.jsonl and splits it by `type` for the
three eval scripts (retrieval_metrics uses "rag", agent_metrics uses "agent"
and "agent_conditional", ragas_eval uses "rag")."""

from __future__ import annotations

import json
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent.parent / "data" / "golden" / "golden_dataset.jsonl"


def load_golden_dataset() -> list[dict]:
    rows = []
    with open(GOLDEN_DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_type(dataset: list[dict], type_name: str) -> list[dict]:
    return [row for row in dataset if row["type"] == type_name]

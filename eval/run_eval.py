"""
Eval CLI: `python -m eval.run_eval --suite {retrieval,agents,ragas,all}`
(or `make eval-retrieval` / `make eval-agents` / `make eval-ragas` / `make eval-all`).

Runs the requested suite(s) against whatever feature flags are currently set
in .env -- this is the intended way to "compare configurations": run
`make eval-retrieval`, flip a flag (e.g. ENABLE_RERANKING), restart, run it
again, and diff eval/results/results_<timestamp>.json.

Writes a timestamped JSON file to eval/results/ and regenerates the
human-readable docs/eval_report.md from the same numbers.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.app.config import get_settings
from backend.app.rag.pipeline import run_rag_pipeline
from backend.app.trace.trace import Trace
from eval.agent_metrics import run_agent_evaluation
from eval.golden_dataset import by_type, load_golden_dataset
from eval.retrieval_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

RESULTS_DIR = Path(__file__).parent / "results"
REPORT_PATH = Path(__file__).parent.parent / "docs" / "eval_report.md"
DEFAULT_K = 5


def evaluate_retrieval() -> dict:
    settings = get_settings()
    dataset = by_type(load_golden_dataset(), "rag")

    per_case = []
    for case in dataset:
        trace = Trace(enabled=False)
        result = run_rag_pipeline(case["query"], settings, trace)
        retrieved = [c.source_path for c in result.citations]
        relevant = set(case["expected_source_docs"])

        per_case.append(
            {
                "id": case["id"],
                "query": case["query"],
                "retrieved": retrieved,
                "relevant": list(relevant),
                "precision_at_k": precision_at_k(retrieved, relevant, DEFAULT_K),
                "recall_at_k": recall_at_k(retrieved, relevant, DEFAULT_K),
                "reciprocal_rank": reciprocal_rank(retrieved, relevant),
                "ndcg_at_k": ndcg_at_k(retrieved, relevant, DEFAULT_K),
            }
        )

    n = len(per_case) or 1
    return {
        "k": DEFAULT_K,
        "cases_evaluated": len(per_case),
        "mean_precision_at_k": sum(c["precision_at_k"] for c in per_case) / n,
        "mean_recall_at_k": sum(c["recall_at_k"] for c in per_case) / n,
        "mrr": sum(c["reciprocal_rank"] for c in per_case) / n,
        "mean_ndcg_at_k": sum(c["ndcg_at_k"] for c in per_case) / n,
        "per_case": per_case,
    }


def evaluate_agents() -> dict:
    results = run_agent_evaluation()
    n = len(results) or 1
    tool_cases = [r for r in results if r.tool_selection_correct is not None]

    return {
        "cases_evaluated": len(results),
        "routing_accuracy": sum(r.routing_correct for r in results) / n,
        "tool_selection_accuracy": (
            sum(r.tool_selection_correct for r in tool_cases) / len(tool_cases) if tool_cases else None
        ),
        "tool_execution_success_rate": (
            sum(bool(r.tool_execution_success) for r in tool_cases) / len(tool_cases) if tool_cases else None
        ),
        "task_completion_rate": sum(r.task_completed for r in results) / n,
        "per_case": [
            {
                "id": r.case_id,
                "routing_correct": r.routing_correct,
                "tool_selection_correct": r.tool_selection_correct,
                "tool_execution_success": r.tool_execution_success,
                "task_completed": r.task_completed,
                "node_sequence": r.node_sequence,
            }
            for r in results
        ],
    }


def evaluate_ragas() -> dict:
    from eval.ragas_eval import run_ragas_evaluation

    return run_ragas_evaluation()


def write_report(results: dict) -> None:
    lines = ["# Evaluation Report", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]

    if "retrieval" in results:
        r = results["retrieval"]
        lines += [
            "## Retrieval metrics",
            "",
            f"- Cases evaluated: {r['cases_evaluated']} (k={r['k']})",
            f"- Precision@k: {r['mean_precision_at_k']:.3f}",
            f"- Recall@k: {r['mean_recall_at_k']:.3f}",
            f"- MRR: {r['mrr']:.3f}",
            f"- NDCG@k: {r['mean_ndcg_at_k']:.3f}",
            "",
        ]

    if "agents" in results:
        a = results["agents"]
        lines += [
            "## Agent metrics",
            "",
            f"- Cases evaluated: {a['cases_evaluated']}",
            f"- Routing accuracy: {a['routing_accuracy']:.3f}",
            f"- Tool selection accuracy: {a['tool_selection_accuracy']}",
            f"- Tool execution success rate: {a['tool_execution_success_rate']}",
            f"- Task completion rate: {a['task_completion_rate']:.3f}",
            "",
        ]

    if "ragas" in results:
        g = results["ragas"]
        lines += [
            "## RAGAS metrics (LLM-judged, small subset)",
            "",
            f"- Cases evaluated: {g['cases_evaluated']}",
            f"- Mean faithfulness: {g['mean_faithfulness']:.3f}",
            f"- Mean answer relevancy: {g['mean_answer_relevancy']:.3f}",
            "",
        ]

    REPORT_PATH.write_text("\n".join(lines))
    print(f"wrote {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["retrieval", "agents", "ragas", "all"], default="all")
    args = parser.parse_args()

    results: dict = {}
    if args.suite in ("retrieval", "all"):
        print("running retrieval evaluation...")
        results["retrieval"] = evaluate_retrieval()
    if args.suite in ("agents", "all"):
        print("running agent evaluation...")
        results["agents"] = evaluate_agents()
    if args.suite == "ragas":  # deliberately excluded from "all" -- see module docstring
        print("running RAGAS evaluation (slower, costs LLM judge calls)...")
        results["ragas"] = evaluate_ragas()

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = RESULTS_DIR / f"results_{timestamp}.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {results_path}")

    write_report(results)


if __name__ == "__main__":
    main()

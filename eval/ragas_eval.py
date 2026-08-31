"""
RAGAS-based evaluation: `faithfulness` (does the answer only claim things
supported by the retrieved context?) and `answer_relevancy` (does the answer
actually address the question?). These two need an LLM-as-judge, which is
exactly what RAGAS provides off the shelf -- reimplementing an LLM-judge
metric from scratch would just be a worse version of RAGAS, so we use it
here instead of hand-rolling it (contrast with retrieval_metrics.py, where
the formulas are simple enough that hand-rolling is the better choice).

Kept to a SMALL subset of the golden "rag" cases (RAGAS_SUBSET_SIZE) and a
separate `make eval-ragas` target (not part of the default fast eval run):
each case costs several extra Gemini calls for the judge itself, on top of
the pipeline run being evaluated.

`ragas` is an optional dependency (`pip install -e ".[eval]"`) -- imports
are inside the function so `make eval-retrieval` / `make eval-agents` still
work without it installed.
"""

from __future__ import annotations

from backend.app.config import get_settings
from backend.app.rag.pipeline import run_rag_pipeline
from backend.app.trace.trace import Trace
from eval.golden_dataset import by_type, load_golden_dataset

RAGAS_SUBSET_SIZE = 6


def run_ragas_evaluation() -> dict:
    from datasets import Dataset
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    from backend.app.llm.gemini_client import get_chat_model

    settings = get_settings()
    dataset_rows = by_type(load_golden_dataset(), "rag")[:RAGAS_SUBSET_SIZE]

    questions, answers, contexts_list = [], [], []
    for row in dataset_rows:
        trace = Trace(enabled=False)
        result = run_rag_pipeline(row["query"], settings, trace)
        questions.append(row["query"])
        answers.append(result.answer)
        contexts_list.append([result.context] if result.context else [""])

    hf_dataset = Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts_list})

    judge_llm = LangchainLLMWrapper(get_chat_model(settings))
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=settings.gemini_embedding_model, google_api_key=settings.gemini_api_key)
    )

    result = evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = result.to_pandas()
    return {
        "cases_evaluated": len(dataset_rows),
        "mean_faithfulness": float(df["faithfulness"].mean()),
        "mean_answer_relevancy": float(df["answer_relevancy"].mean()),
        "per_case": df.to_dict(orient="records"),
    }

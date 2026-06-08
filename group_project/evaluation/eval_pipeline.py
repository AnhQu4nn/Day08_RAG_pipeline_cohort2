"""RAGAS-first RAG evaluation with local fallback metrics and A/B comparison."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.benchmark_latency import temporary_env
from src.local_index import tokenize
from src.task10_generation import generate_with_citation
from src.task9_retrieval_pipeline import retrieve

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
RAW_RESULTS_PATH = Path(__file__).parent / "ragas_results.json"

BASELINE_CONFIG = {
    "name": "baseline",
    "env": {
        "PAGEINDEX_FALLBACK_ENABLED": "0",
        "RAG_QUERY_MAX_VARIANTS": "3",
        "RAG_QUERY_MAX_WORDS": "48",
        "RAG_DISABLE_LLM_QUERY_VARIANTS": "0",
        "HYDE_ENABLED": "0",
        "RERANK_METHOD": "cross_encoder",
    },
}

OPTIMIZED_CONFIG = {
    "name": "disable_llm_query_variants",
    "env": {
        "PAGEINDEX_FALLBACK_ENABLED": "0",
        "RAG_QUERY_MAX_VARIANTS": "1",
        "RAG_QUERY_MAX_WORDS": "32",
        "RAG_DISABLE_LLM_QUERY_VARIANTS": "1",
        "HYDE_ENABLED": "0",
        "RERANK_METHOD": "cross_encoder",
    },
}

HYDE_CONFIG = {
    "name": "query_variants_and_hyde_enable",
    "env": {
        "PAGEINDEX_FALLBACK_ENABLED": "0",
        "RAG_QUERY_MAX_VARIANTS": "1",
        "RAG_QUERY_MAX_WORDS": "32",
        "RAG_DISABLE_LLM_QUERY_VARIANTS": "1",
        "HYDE_ENABLED": "1",
        "RERANK_METHOD": "cross_encoder",
    },
}

JINA_LATE_CHUNKING_CONFIG = {
    "name": "jina_late_chunking",
    "env": {
        "PAGEINDEX_FALLBACK_ENABLED": "0",
        "RAG_QUERY_MAX_VARIANTS": "3",
        "RAG_QUERY_MAX_WORDS": "48",
        "RAG_DISABLE_LLM_QUERY_VARIANTS": "0",
        "HYDE_ENABLED": "0",
        "RERANK_METHOD": "cross_encoder",
        "CHUNKING_METHOD": "jina_late",
        "JINA_EMBEDDING_MODEL": "jina-embeddings-v3",
    },
}


def load_golden_dataset() -> list[dict]:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def _overlap_score(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _local_score_case(item: dict, answer: str, sources: list[dict]) -> dict:
    contexts = "\n".join(source.get("content", "") for source in sources)
    expected_context = item.get("expected_context", "")
    expected_answer = item.get("expected_answer", "")
    scores = {
        "faithfulness": _overlap_score(answer, contexts),
        "answer_relevance": _overlap_score(expected_answer, answer),
        "context_recall": _overlap_score(expected_answer + " " + expected_context, contexts),
        "context_precision": _overlap_score(contexts, expected_answer + " " + expected_context),
    }
    scores["average"] = sum(scores.values()) / max(1, len(scores))
    return scores


def _run_pipeline_case(item: dict, top_k: int = 5, generate: bool = True) -> dict:
    started = time.perf_counter()
    if generate:
        result = generate_with_citation(item["question"], top_k=top_k)
        answer = result["answer"]
        sources = result["sources"]
        timings = result.get("timings", {})
    else:
        sources = retrieve(item["question"], top_k=top_k)
        answer = " ".join(source.get("content", "") for source in sources[:1])
        timings = sources[0].get("metadata", {}).get("timings", {}) if sources else {}
    latency_ms = (time.perf_counter() - started) * 1000
    return {
        "question": item["question"],
        "answer": answer,
        "contexts": [source.get("content", "") for source in sources],
        "ground_truth": item.get("expected_answer", ""),
        "expected_context": item.get("expected_context", ""),
        "sources": sources,
        "latency_ms": latency_ms,
        "timings": timings,
    }


def _ragas_available() -> bool:
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        return False
    try:
        import ragas  # noqa: F401
        import datasets  # noqa: F401
    except Exception:
        return False
    return True


def _evaluate_rows_with_ragas(rows: list[dict]) -> tuple[list[dict], str]:
    if not _ragas_available():
        return rows, "ragas_unavailable_or_missing_api_key"

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_list(
            [
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "contexts": row["contexts"],
                    "ground_truth": row["ground_truth"],
                }
                for row in rows
            ]
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        frame = result.to_pandas()
        for idx, row in enumerate(rows):
            if idx >= len(frame):
                continue
            scored = frame.iloc[idx].to_dict()
            row["faithfulness"] = float(scored.get("faithfulness", row.get("faithfulness", 0.0)) or 0.0)
            row["answer_relevance"] = float(scored.get("answer_relevancy", scored.get("answer_relevance", row.get("answer_relevance", 0.0))) or 0.0)
            row["context_precision"] = float(scored.get("context_precision", row.get("context_precision", 0.0)) or 0.0)
            row["context_recall"] = float(scored.get("context_recall", row.get("context_recall", 0.0)) or 0.0)
            row["average"] = statistics.mean([row["faithfulness"], row["answer_relevance"], row["context_precision"], row["context_recall"]])
        return rows, "ragas"
    except Exception as exc:
        for row in rows:
            row["ragas_error"] = type(exc).__name__
        return rows, f"ragas_error:{type(exc).__name__}"


def evaluate_config(golden_dataset: list[dict], config: dict, top_k: int = 5, use_ragas: bool = True) -> dict:
    rows = []
    with temporary_env(config.get("env", {})):
        for item in golden_dataset:
            row = _run_pipeline_case(item, top_k=top_k, generate=True)
            row.update(_local_score_case(item, row["answer"], row["sources"]))
            rows.append(row)

    evaluator = "local_overlap"
    if use_ragas:
        rows, evaluator = _evaluate_rows_with_ragas(rows)

    metrics = ["faithfulness", "answer_relevance", "context_recall", "context_precision", "average", "latency_ms"]
    summary = {metric: statistics.mean([float(row.get(metric, 0.0)) for row in rows]) if rows else 0.0 for metric in metrics}
    return {"name": config["name"], "env": config.get("env", {}), "evaluator": evaluator, "summary": summary, "rows": rows}


def compare_configs(
    rag_pipeline=None,
    golden_dataset: list[dict] | None = None,
    configs: list[dict] | None = None,
    use_ragas: bool = True,
) -> dict:
    dataset = golden_dataset or load_golden_dataset()
    selected_configs = configs or [BASELINE_CONFIG, OPTIMIZED_CONFIG, HYDE_CONFIG, JINA_LATE_CHUNKING_CONFIG]
    return {config["name"]: evaluate_config(dataset, config, use_ragas=use_ragas) for config in selected_configs}


def export_results(results: dict, comparison: dict | None = None):
    configs = comparison or results
    RAW_RESULTS_PATH.write_text(json.dumps(configs, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    names = list(configs)
    metrics = ["faithfulness", "answer_relevance", "context_recall", "context_precision", "average", "latency_ms"]

    lines = ["# RAG Evaluation Results", "", "## Overall Scores", ""]
    lines.append("| Metric | " + " | ".join(names) + " |")
    lines.append("|---" + "|---:" * len(names) + "|")
    for metric in metrics:
        lines.append("| " + metric + " | " + " | ".join(f"{configs[name]['summary'].get(metric, 0.0):.3f}" for name in names) + " |")

    baseline_name = names[0] if names else ""
    worst = sorted(configs.get(baseline_name, {}).get("rows", []), key=lambda row: row.get("average", 0.0))[:5]
    lines.extend(["", "## Evaluator", ""])
    for name in names:
        lines.append(f"- `{name}`: {configs[name].get('evaluator', 'unknown')}")

    lines.extend(["", "## Worst Performers", ""])
    lines.append("| # | Question | Faithfulness | Relevance | Recall | Precision | Latency ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for i, row in enumerate(worst, 1):
        lines.append(
            f"| {i} | {row['question']} | {row.get('faithfulness', 0.0):.3f} | "
            f"{row.get('answer_relevance', 0.0):.3f} | {row.get('context_recall', 0.0):.3f} | "
            f"{row.get('context_precision', 0.0):.3f} | {row.get('latency_ms', 0.0):.1f} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append("- PageIndex is not part of the default evaluation configs; keep it as a later last-option fallback only.")
    lines.append("- `jina_late_chunking` needs `JINA_API_KEY`; otherwise it falls back to local embeddings and the comparison is not a true late-chunking run.")
    lines.append("- If RAGAS is unavailable or judge credentials are missing, the script reports local overlap fallback metrics.")
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    return evaluate_config(golden_dataset, BASELINE_CONFIG, use_ragas=False)


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    return evaluate_config(golden_dataset, BASELINE_CONFIG, use_ragas=True)


def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    return evaluate_config(golden_dataset, BASELINE_CONFIG, use_ragas=False)


if __name__ == "__main__":
    dataset = load_golden_dataset()
    comparison = compare_configs(golden_dataset=dataset)
    export_results(comparison)
    print(f"Evaluated {len(dataset)} cases. Results: {RESULTS_PATH}")
# RAG Evaluation Results

## Overall Scores

| Metric | baseline | disable_llm_query_variants | query_variants_and_hyde_enable | jina_late_chunking |
|---|---:|---:|---:|---:|
| faithfulness | 0.708 | 0.798 | 0.758 | 0.763 |
| answer_relevance | 0.444 | 0.556 | 0.593 | 0.481 |
| context_recall | 0.676 | 0.676 | 0.676 | 0.824 |
| context_precision | 0.128 | 0.128 | 0.128 | 0.158 |
| average | 0.489 | 0.540 | 0.539 | 0.557 |
| latency_ms | 102423.318 | 82284.331 | 62363.384 | 21581.374 |

## Evaluator

- `baseline`: ragas_unavailable:ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
- `disable_llm_query_variants`: ragas_unavailable:ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
- `query_variants_and_hyde_enable`: ragas_unavailable:ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
- `jina_late_chunking`: ragas_unavailable:ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'

## Worst Performers

| # | Question | Faithfulness | Relevance | Recall | Precision | Latency ms |
|---|---|---:|---:|---:|---:|---:|
| 1 | Hình phạt cho tội tàng trữ trái phép chất ma túy theo Điều 249 Bộ luật Hình sự? | 0.708 | 0.444 | 0.676 | 0.128 | 102423.3 |

## Notes

- PageIndex is not part of the default evaluation configs; keep it as a later last-option fallback only.
- `jina_late_chunking` needs `JINA_API_KEY`; otherwise it falls back to local embeddings and the comparison is not a true late-chunking run.
- If RAGAS is unavailable or judge credentials are missing, the script reports local overlap fallback metrics.
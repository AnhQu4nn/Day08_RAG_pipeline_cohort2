"""
Task 7 - Reranking Module.

Uses Jina Reranker when JINA_API_KEY is available. Without an API key, it falls
back to a deterministic local lexical reranker so tests and demos can run.
"""

import math
import os
import re
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def cosine_sim(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def _local_relevance(query: str, content: str) -> float:
    query_counts = Counter(_tokenize(query))
    doc_counts = Counter(_tokenize(content))
    if not query_counts or not doc_counts:
        return 0.0
    overlap = sum(min(count, doc_counts.get(term, 0)) for term, count in query_counts.items())
    return overlap / sum(query_counts.values())


def _local_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    reranked = []
    for candidate in candidates:
        item = candidate.copy()
        base_score = float(item.get("score", 0.0))
        item["score"] = 0.7 * _local_relevance(query, item.get("content", "")) + 0.3 * base_score
        reranked.append(item)
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates using Jina when configured, otherwise use local scoring.
    """
    if not candidates:
        return []
    if not JINA_API_KEY:
        return _local_rerank(query, candidates, top_k)

    try:
        import requests

        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {JINA_API_KEY}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        reranked = response.json()["results"]
        return [
            {**candidates[item["index"]], "score": float(item["relevance_score"])}
            for item in reranked
        ]
    except Exception:
        return _local_rerank(query, candidates, top_k)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance: choose relevant but non-duplicate results.
    """
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            embedding = candidates[idx].get("embedding", [])
            relevance = cosine_sim(query_embedding, embedding)
            max_sim_to_selected = max(
                (
                    cosine_sim(embedding, candidates[sel_idx].get("embedding", []))
                    for sel_idx in selected
                ),
                default=0.0,
            )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion over multiple ranked result lists.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (k + rank)
            content_map[key] = item

    results = []
    for content, score in sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        return _local_rerank(query, candidates, top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Dieu 248: Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si bi bat vi su dung ma tuy", "score": 0.7, "metadata": {}},
        {"content": "Python programming", "score": 0.4, "metadata": {}},
    ]
    for result in rerank("hinh phat ma tuy", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")

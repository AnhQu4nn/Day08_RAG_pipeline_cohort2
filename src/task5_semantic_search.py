"""
Task 5 - Semantic Search Module.

Uses Weaviate when it is configured, and falls back to a lightweight local
token-cosine search over the markdown corpus so the pipeline remains runnable
without external services.
"""

import math
import os
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _cosine_counts(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _local_semantic_search(query: str, top_k: int) -> list[dict]:
    try:
        from .task4_chunking_indexing import chunk_documents, load_documents
    except ImportError:
        from task4_chunking_indexing import chunk_documents, load_documents

    chunks = chunk_documents(load_documents())
    query_counts = Counter(_tokenize(query))
    results = []

    for chunk in chunks:
        score = _cosine_counts(query_counts, Counter(_tokenize(chunk["content"])))
        if score > 0:
            results.append({
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _weaviate_semantic_search(query: str, top_k: int) -> list[dict]:
    import weaviate
    from weaviate.classes.init import Auth
    from weaviate.classes.query import MetadataQuery
    from openai import OpenAI

    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    if not weaviate_url or not weaviate_api_key:
        raise ValueError("Missing WEAVIATE_URL or WEAVIATE_API_KEY")

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API in .env")

    client_llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    query_embedding = client_llm.embeddings.create(input=[query], model="openai/text-embedding-3-small").data[0].embedding

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )
    try:
        collection = client.collections.get("DrugLawDocsOpenAI")
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            props = obj.properties or {}
            distance = getattr(obj.metadata, "distance", 1.0)
            results.append({
                "content": props.get("content", ""),
                "score": float(1 - distance),
                "metadata": {
                    "source": props.get("source", ""),
                    "source_path": props.get("source_path", ""),
                    "type": props.get("doc_type", ""),
                    "chunk_index": props.get("chunk_index"),
                },
            })
        return sorted(results, key=lambda item: item["score"], reverse=True)
    finally:
        client.close()


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Return semantic-style results sorted by score descending.
    """
    top_k = max(0, int(top_k))
    if top_k == 0:
        return []

    try:
        return _weaviate_semantic_search(query, top_k)
    except Exception:
        return _local_semantic_search(query, top_k)


if __name__ == "__main__":
    results = semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")

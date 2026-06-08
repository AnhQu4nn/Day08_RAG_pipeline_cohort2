"""
Task 6 - Lexical Search Module (BM25).

The module builds its corpus from data/standardized/ at call time, so it works
in tests and in the retrieval pipeline without a separate indexing step.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

CORPUS: list[dict] = []
_BM25_INDEX = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


@dataclass
class SimpleBM25:
    tokenized_corpus: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.term_freqs = [Counter(doc) for doc in self.tokenized_corpus]
        doc_freq = Counter()
        for doc in self.tokenized_corpus:
            doc_freq.update(set(doc))
        n_docs = len(self.tokenized_corpus)
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for idx, freqs in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            for token in query_tokens:
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(token, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


def _load_corpus() -> list[dict]:
    try:
        from .task4_chunking_indexing import chunk_documents, load_documents
    except ImportError:
        from task4_chunking_indexing import chunk_documents, load_documents

    return chunk_documents(load_documents())


def build_bm25_index(corpus: list[dict]):
    """
    Build a BM25 index from a list of {'content': str, 'metadata': dict}.
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except ImportError:
        return SimpleBM25(tokenized_corpus)


def _ensure_index() -> None:
    global CORPUS, _BM25_INDEX
    if CORPUS and _BM25_INDEX is not None:
        return
    CORPUS = _load_corpus()
    _BM25_INDEX = build_bm25_index(CORPUS) if CORPUS else None


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search by keyword relevance and return results sorted by score descending.
    """
    top_k = max(0, int(top_k))
    if top_k == 0:
        return []

    _ensure_index()
    if not CORPUS or _BM25_INDEX is None:
        return []

    scores = _BM25_INDEX.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

    results = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append({
            "content": CORPUS[idx]["content"],
            "score": score,
            "metadata": CORPUS[idx].get("metadata", {}),
        })
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    results = lexical_search("Dieu 248 tang tru trai phep chat ma tuy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")

"""
hybrid_search.py — Lexical BM25 + Dense Vector Hybrid Search with Reciprocal Rank Fusion (RRF)
"""
import math
import re
from typing import List, Dict, Any


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from",
    "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "of", "in", "on", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "this",
    "that", "these", "those", "it", "its", "what", "which", "who", "whom", "how", "why"
}


def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into clean tokens."""
    tokens = re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


class BM25Index:
    """Lightweight in-memory BM25 Okapi index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, chunks: List[Dict[str, Any]]) -> "BM25Index":
        self.docs = chunks
        self.doc_tokens = [tokenize(c.get("text", "")) for c in chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        n_docs = len(chunks)
        self.avg_doc_len = (sum(self.doc_lens) / n_docs) if n_docs > 0 else 0.0

        # Document frequencies
        self.df = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        # Inverse document frequency
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

        return self

    def query(self, query_str: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Score all documents against query and return top-K."""
        if not self.docs:
            return []

        q_tokens = tokenize(query_str)
        if not q_tokens:
            return []

        scores = [0.0] * len(self.docs)
        for i, (tokens, doc_len) in enumerate(zip(self.doc_tokens, self.doc_lens)):
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            doc_score = 0.0
            len_norm = 1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0))
            for q_term in q_tokens:
                if q_term in tf_map:
                    tf = tf_map[q_term]
                    idf = self.idf.get(q_term, 0.0)
                    denom = tf + self.k1 * len_norm
                    term_score = idf * (tf * (self.k1 + 1.0)) / (denom or 1.0)
                    doc_score += term_score

            scores[i] = doc_score

        # Pair with docs
        results = []
        for i, score in enumerate(scores):
            if score > 0:
                doc_copy = dict(self.docs[i])
                doc_copy["bm25_score"] = score
                results.append(doc_copy)

        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]


def reciprocal_rank_fusion(
    dense_hits: List[Dict[str, Any]],
    sparse_hits: List[Dict[str, Any]],
    k: int = 60,
    dense_weight: float = 0.65,
    sparse_weight: float = 0.35,
    top_k: int = 5,
    min_score_threshold: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Fuse dense vector rankings with BM25 lexical rankings using Reciprocal Rank Fusion.
    Applies exact phrase boosting and relevance filtering.
    """
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Rank dense hits
    for rank, hit in enumerate(dense_hits):
        doc_key = f"{hit.get('source')}_{hit.get('page')}_{hit.get('chunk_index')}"
        doc_map[doc_key] = hit
        rrf = dense_weight / (k + rank + 1)
        fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + rrf

    # Rank sparse hits
    for rank, hit in enumerate(sparse_hits):
        doc_key = f"{hit.get('source')}_{hit.get('page')}_{hit.get('chunk_index')}"
        if doc_key not in doc_map:
            doc_map[doc_key] = hit
        rrf = sparse_weight / (k + rank + 1)
        fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + rrf

    if not fused_scores:
        return []

    # Normalize fused scores to [0, 1]
    max_score = max(fused_scores.values()) or 1.0
    scored_docs = []
    for doc_key, score in fused_scores.items():
        doc = dict(doc_map[doc_key])
        norm_score = round(min(1.0, score / max_score), 4)

        # Blend with original vector score if available for grounded calibration
        if "score" in doc and doc["score"] is not None:
            final_relevance = round(0.7 * doc["score"] + 0.3 * norm_score, 4)
        else:
            final_relevance = norm_score

        doc["relevance_score"] = final_relevance
        doc["score"] = final_relevance
        scored_docs.append(doc)

    scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Filter out irrelevant noise below threshold if we have good candidates
    filtered = [d for d in scored_docs if d["relevance_score"] >= min_score_threshold]
    final_hits = filtered if filtered else scored_docs[:2]

    return final_hits[:top_k]

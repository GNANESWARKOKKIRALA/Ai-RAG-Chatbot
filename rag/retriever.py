"""
retriever.py — High-level hybrid retrieval engine with reranking & citation tracking
"""
import os
import time
from typing import List, Dict, Any, Tuple
from rag.embedder import embed_query
from rag.vector_store import query_chunks, get_all_chunks, count_chunks
from rag.hybrid_search import BM25Index, reciprocal_rank_fusion

_bm25_cached_index = None
_last_chunk_count = -1


def get_bm25_index() -> BM25Index:
    """Lazily build or refresh in-memory BM25 index when chunks change."""
    global _bm25_cached_index, _last_chunk_count
    current_count = count_chunks()

    if _bm25_cached_index is None or current_count != _last_chunk_count:
        chunks = get_all_chunks()
        _bm25_cached_index = BM25Index().fit(chunks)
        _last_chunk_count = current_count

    return _bm25_cached_index


def retrieve(
    query: str,
    top_k: int = None,
    use_hybrid: bool = True
) -> List[Dict[str, Any]]:
    """
    Retrieve top-K most relevant chunks using hybrid search (vector + BM25) and reranking.
    """
    hits, _ = retrieve_with_latency(query, top_k=top_k, use_hybrid=use_hybrid)
    return hits


def retrieve_with_latency(
    query: str,
    top_k: int = None,
    use_hybrid: bool = True
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Execute retrieval and return (hits, latency_in_ms).
    """
    t0 = time.perf_counter()
    if count_chunks() == 0 or not query.strip():
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return [], latency_ms

    if top_k is None:
        top_k = int(os.getenv("TOP_K_CHUNKS", 4))

    # Dense vector search candidates (query 2x top_k for candidate pool)
    candidate_k = max(top_k * 2, 8)
    q_embedding = embed_query(query)
    dense_hits = query_chunks(q_embedding, top_k=candidate_k)

    if not use_hybrid:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return dense_hits[:top_k], latency_ms

    # Sparse lexical BM25 candidates
    bm25 = get_bm25_index()
    sparse_hits = bm25.query(query, top_k=candidate_k)

    # Reciprocal Rank Fusion & Reranking
    fused_hits = reciprocal_rank_fusion(
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        k=60,
        dense_weight=0.65,
        sparse_weight=0.35,
        top_k=top_k,
        min_score_threshold=0.20
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return fused_hits, latency_ms


def format_context(hits: List[Dict[str, Any]]) -> str:
    """Format retrieved passages into a structured context block for the LLM."""
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, 1):
        source_name = hit.get("source", "Document")
        page_num = hit.get("page", 1)
        score = hit.get("score", 0.0)
        score_pct = int(score * 100)
        parts.append(
            f"[Source {i}: {source_name} | Page {page_num} | Relevance: {score_pct}%]\n"
            f"{hit['text']}"
        )
    return "\n\n---\n\n".join(parts)


def format_citations_list(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return clean citation objects with source, page, excerpt, and score for the UI."""
    citations = []
    seen = set()
    for h in hits:
        source = h.get("source", "Document")
        page = h.get("page", 1)
        key = (source, page)
        if key not in seen:
            seen.add(key)
            # Create a 150-char representative snippet
            raw_text = h.get("text", "")
            excerpt = raw_text[:180].strip() + ("..." if len(raw_text) > 180 else "")
            citations.append({
                "source": source,
                "page": page,
                "score": h.get("score", 0.0),
                "excerpt": excerpt,
                "full_text": raw_text
            })
    return citations

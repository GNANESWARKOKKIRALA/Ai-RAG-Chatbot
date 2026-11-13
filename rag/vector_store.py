"""
vector_store.py — ChromaDB operations: hybrid metadata storage, vector queries, chunk retrieval
"""
import os
from typing import List, Dict, Any, Union
import chromadb
from chromadb.config import Settings


_client = None


def get_client() -> chromadb.Client:
    global _client
    if _client is None:
        db_path = os.getenv("CHROMA_DB_PATH", "./chroma_store")
        _client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
    return _client


def get_collection(name: str = "rag_docs") -> chromadb.Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


def add_chunks(
    chunks: Union[List[str], List[Dict[str, Any]]],
    embeddings: List[List[float]],
    source: str,
    doc_id: str,
    pages: List[int] = None
) -> None:
    """
    Add text chunks with their embeddings to ChromaDB.
    Supports either plain strings or structured chunk dictionaries with page numbers.
    """
    if not chunks:
        return

    collection = get_collection()
    
    doc_texts = []
    ids = []
    metadatas = []

    for i, item in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        ids.append(chunk_id)

        if isinstance(item, dict):
            text = item.get("text", "")
            page_num = int(item.get("page", 1))
            section = str(item.get("section", f"Page {page_num}"))
            chunk_idx = int(item.get("chunk_index", i))
        else:
            text = str(item)
            page_num = pages[i] if pages and i < len(pages) else 1
            section = f"Page {page_num}"
            chunk_idx = i

        doc_texts.append(text)
        metadatas.append({
            "source": source,
            "doc_id": doc_id,
            "chunk_index": chunk_idx,
            "page": page_num,
            "section": section
        })

    collection.add(
        documents=doc_texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )


def query_chunks(
    query_embedding: List[float],
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """Retrieve top-K most relevant chunks with enriched metadata."""
    collection = get_collection()
    total_docs = collection.count()
    if total_docs == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total_docs),
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # Cosine distance to similarity score in [0, 1]
            similarity = max(0.0, min(1.0, 1.0 - dist))
            hits.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "doc_id": meta.get("doc_id", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "page": meta.get("page", 1),
                "section": meta.get("section", f"Page {meta.get('page', 1)}"),
                "score": round(similarity, 4),
                "distance": round(dist, 4)
            })
    return hits


def get_all_chunks() -> List[Dict[str, Any]]:
    """Return all stored chunks with metadata for hybrid lexical BM25 indexing."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    
    data = collection.get(include=["documents", "metadatas"])
    chunks = []
    if data and data.get("documents"):
        for doc_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            chunks.append({
                "id": doc_id,
                "text": text,
                "source": meta.get("source", "unknown"),
                "doc_id": meta.get("doc_id", ""),
                "page": meta.get("page", 1),
                "section": meta.get("section", f"Page {meta.get('page', 1)}"),
                "chunk_index": meta.get("chunk_index", 0)
            })
    return chunks


def delete_document(doc_id: str) -> None:
    """Remove all chunks belonging to a document from the collection."""
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])


def count_chunks() -> int:
    return get_collection().count()


def reset_store() -> None:
    """Delete and recreate the Chroma collection."""
    client = get_client()
    try:
        client.delete_collection("rag_docs")
    except Exception:
        pass

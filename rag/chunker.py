"""
chunker.py — Sentence-aware text chunking with page and section metadata preservation
"""
import re
from typing import List, Dict, Any


def split_into_sentences(text: str) -> List[str]:
    """Split text cleanly into sentences or paragraphs."""
    # Split on paragraph boundaries first, then sentence ends
    paragraphs = text.split("\n\n")
    sentences = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # Split on sentence boundaries (.!? followed by space or newline)
        raw_s = re.split(r'(?<=[.!?])\s+', p)
        for s in raw_s:
            s = s.strip()
            if s:
                sentences.append(s)
    return sentences


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 40) -> List[str]:
    """
    Split text into semantically coherent overlapping chunks.
    Maintains backward compatibility with original chunk_text signature.
    """
    if not text or not text.strip():
        return []
    
    sentences = split_into_sentences(text)
    if not sentences:
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), max(1, chunk_size - overlap))]
    
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        s_words = sentence.split()
        s_len = len(s_words)
        
        # If single sentence is bigger than chunk_size, split by words
        if s_len > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            for i in range(0, s_len, max(1, chunk_size - overlap)):
                chunks.append(" ".join(s_words[i:i + chunk_size]))
            continue

        if current_len + s_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlap sentences from the end
            overlap_chunk = []
            overlap_len = 0
            for s in reversed(current_chunk):
                words = s.split()
                if overlap_len + len(words) <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_len += len(words)
                else:
                    break
            current_chunk = overlap_chunk
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += s_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 250,
    overlap: int = 40
) -> List[Dict[str, Any]]:
    """
    Chunk page objects while preserving page numbers and section labels.
    Returns: [{"text": str, "page": int, "section": str, "chunk_index": int}]
    """
    all_chunks = []
    global_idx = 0

    for page in pages:
        page_num = page.get("page", 1)
        section = page.get("section", f"Page {page_num}")
        page_text = page.get("text", "")

        raw_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for chunk in raw_chunks:
            all_chunks.append({
                "text": chunk,
                "page": page_num,
                "section": section,
                "chunk_index": global_idx,
                "char_count": len(chunk),
                "word_count": len(chunk.split())
            })
            global_idx += 1

    return all_chunks

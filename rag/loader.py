"""
loader.py — Parse PDF, TXT, DOCX, CSV with page-level metadata extraction
"""
import os
from typing import List, Dict, Any
import pdfplumber
import docx
import pandas as pd


def load_file(filepath: str) -> str:
    """Load any supported file and return its complete text content (backward compatibility)."""
    pages = load_document_pages(filepath)
    return "\n\n".join(p["text"] for p in pages if p["text"].strip())


def load_document_pages(filepath: str) -> List[Dict[str, Any]]:
    """
    Load any supported file and return list of page-level dicts:
    [{"text": str, "page": int, "section": str}]
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return _load_pdf_pages(filepath)
    elif ext in (".txt", ".md"):
        return _load_txt_pages(filepath)
    elif ext == ".docx":
        return _load_docx_pages(filepath)
    elif ext == ".csv":
        return _load_csv_pages(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _load_pdf_pages(path: str) -> List[Dict[str, Any]]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            t = page.extract_text()
            if t and t.strip():
                pages.append({
                    "text": t.strip(),
                    "page": idx,
                    "section": f"Page {idx}"
                })
    return pages


def _load_txt_pages(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()
    if not content:
        return []
    return [{
        "text": content,
        "page": 1,
        "section": "Document"
    }]


def _load_docx_pages(path: str) -> List[Dict[str, Any]]:
    doc = docx.Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        return []
    
    # Group paragraphs into logical sections of ~5 paragraphs
    sections = []
    chunk_size = 5
    for i in range(0, len(paragraphs), chunk_size):
        sec_num = (i // chunk_size) + 1
        text = "\n\n".join(paragraphs[i:i + chunk_size])
        sections.append({
            "text": text,
            "page": sec_num,
            "section": f"Section {sec_num}"
        })
    return sections


def _load_csv_pages(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    # Convert chunks of 50 rows into pages to preserve tabular structure
    if df.empty:
        return []
    
    pages = []
    row_chunk = 50
    for i in range(0, len(df), row_chunk):
        page_num = (i // row_chunk) + 1
        chunk_df = df.iloc[i:i + row_chunk]
        text = chunk_df.to_string(index=False)
        pages.append({
            "text": text,
            "page": page_num,
            "section": f"Rows {i+1}-{min(i+row_chunk, len(df))}"
        })
    return pages

"""
verifier.py — Grounding verification between generated responses and retrieved context
"""
import re
from typing import List, Dict, Any


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "in", "on", "at",
    "to", "for", "with", "by", "about", "as", "of", "from", "that", "this", "these",
    "those", "it", "its", "they", "them", "their", "we", "us", "our", "you", "your",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
    "will", "would", "shall", "should", "may", "might", "must", "not", "also", "into",
    "such", "than", "more", "most", "some", "any", "which", "who", "whom", "whose"
}


def extract_key_terms(text: str) -> set:
    """Extract significant content terms (words, numbers, acronyms) for fact verification."""
    words = re.findall(r'\b[a-zA-Z0-9_\-\.]{3,}\b', text.lower())
    return {w for w in words if w not in _STOPWORDS and not w.isdigit()}


def verify_answer(
    answer: str,
    retrieved_hits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Verify whether the generated response is factually supported by retrieved passages.
    Returns:
        {
            "status": "grounded" | "partial" | "not_in_docs" | "no_context",
            "label": str,
            "badge_color": str,
            "supported": bool,
            "grounding_score": float,
            "explanation": str
        }
    """
    if not retrieved_hits:
        return {
            "status": "no_context",
            "label": "General Knowledge (No Documents)",
            "badge_color": "#64748b",
            "supported": False,
            "grounding_score": 0.0,
            "explanation": "No documents uploaded or retrieved. Answer was generated from general model knowledge."
        }

    lower_ans = answer.lower()
    if (
        "couldn't find" in lower_ans
        or "could not find" in lower_ans
        or "not mentioned in the context" in lower_ans
        or "no relevant information" in lower_ans
        or "not found in the uploaded" in lower_ans
    ):
        return {
            "status": "not_in_docs",
            "label": "Information Not in Documents",
            "badge_color": "#f59e0b",
            "supported": False,
            "grounding_score": 0.0,
            "explanation": "The assistant recognized that the requested topic is absent from the uploaded documents."
        }

    # Extract terms from answer and from combined retrieved passages
    combined_context = " ".join(h.get("text", "") for h in retrieved_hits)
    context_terms = extract_key_terms(combined_context)
    answer_terms = extract_key_terms(answer)

    if not answer_terms or not context_terms:
        return {
            "status": "partial",
            "label": "General Synthesis",
            "badge_color": "#6366f1",
            "supported": True,
            "grounding_score": 0.5,
            "explanation": "Answer provides a general overview based on retrieved topics."
        }

    overlap = answer_terms.intersection(context_terms)
    ratio = len(overlap) / len(answer_terms)

    if ratio >= 0.60:
        return {
            "status": "grounded",
            "label": "Verified Document Grounding",
            "badge_color": "#10b981",
            "supported": True,
            "grounding_score": round(ratio, 2),
            "explanation": "High factual overlap: Key terms and facts directly match retrieved document passages."
        }
    elif ratio >= 0.35:
        return {
            "status": "partial",
            "label": "Partially Grounded",
            "badge_color": "#3b82f6",
            "supported": True,
            "grounding_score": round(ratio, 2),
            "explanation": "Moderate factual overlap: Combines cited passages with broader explanatory context."
        }
    else:
        return {
            "status": "unverified",
            "label": "Uncorroborated Claims",
            "badge_color": "#f43f5e",
            "supported": False,
            "grounding_score": round(ratio, 2),
            "explanation": "Low passage overlap: Output may contain extrapolations beyond the exact text in the documents."
        }

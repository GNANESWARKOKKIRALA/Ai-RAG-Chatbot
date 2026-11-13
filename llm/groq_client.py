"""
groq_client.py — Groq API integration with selectable chat modes, follow-up generation & streaming
"""
import os
import re
import time
from typing import List, Dict, Generator, Tuple
from groq import Groq

client = None

MAX_HISTORY_MESSAGES = 20
MAX_TOKENS_RESPONSE = 1024

CHAT_MODES = {
    "Research": {
        "title": "🔬 Research Mode",
        "description": "Comprehensive, evidence-based responses with deep analysis and specific citations.",
        "instruction": (
            "Provide an in-depth, rigorous, evidence-based response. Cite specific sources and page numbers "
            "as [Source: filename, Page X]. Structure your analysis with clear markdown subheadings, "
            "nuanced explanations, and thorough evaluation of the evidence."
        )
    },
    "Simple": {
        "title": "💡 Simple Mode",
        "description": "Plain-English explanations, intuitive analogies, and approachable language.",
        "instruction": (
            "Explain concepts in simple, plain English that anyone can easily understand. Avoid heavy jargon, "
            "use everyday analogies where appropriate, and keep formatting clear with simple bullet points."
        )
    },
    "Technical": {
        "title": "⚙️ Technical Mode",
        "description": "Engineering-grade detail, architectural specifics, specifications, and data points.",
        "instruction": (
            "Provide precise technical explanations focusing on specifications, architectures, algorithms, "
            "data structures, technical constraints, and exact terminology mentioned in the documents."
        )
    },
    "Summary": {
        "title": "📋 Summary Mode",
        "description": "Concise executive briefing with key findings and bulleted takeaways.",
        "instruction": (
            "Provide a concise executive summary. Begin with a 2-sentence key conclusion, followed by "
            "3 to 5 clear, high-impact bulleted takeaways. Avoid fluff and keep the response succinct."
        )
    }
}


def get_client() -> Groq:
    global client
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env file")
        client = Groq(api_key=api_key)
    return client


def build_system_prompt(context: str, mode: str = "Research") -> str:
    mode_info = CHAT_MODES.get(mode, CHAT_MODES["Research"])
    mode_instruction = mode_info["instruction"]

    if context:
        return (
            "You are a professional AI Document Intelligence Assistant. "
            "Your objective is to deliver accurate, grounded answers strictly using the retrieved document context below.\n\n"
            f"RESPONSE STYLE DIRECTIVE ({mode.upper()} MODE):\n{mode_instruction}\n\n"
            "STRICT GROUNDING RULES:\n"
            "1. Answer ONLY using the facts present in the RETRIEVED CONTEXT below.\n"
            "2. If the user's question cannot be answered from the provided context, state clearly: "
            "'I couldn't find relevant information in the uploaded documents to answer this question.'\n"
            "3. Whenever citing specific facts or numbers, mention the source document name and page number.\n"
            "4. Never extrapolate unsupported speculations without explicitly flagging them as uncertain.\n\n"
            f"RETRIEVED CONTEXT:\n{context}"
        )
    return (
        "You are a professional AI Document Intelligence Assistant powered by Groq. "
        "No documents have been uploaded yet. Answer general questions politely from your foundational knowledge. "
        "If the user asks about specific uploaded documents, guide them to upload files (PDF, DOCX, TXT, CSV) "
        "via the Knowledge Base or sidebar.\n\n"
        f"RESPONSE STYLE DIRECTIVE ({mode.upper()} MODE):\n{mode_instruction}"
    )


def trim_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    trimmed = messages[-MAX_HISTORY_MESSAGES:]
    if trimmed and trimmed[0]["role"] == "assistant":
        trimmed = trimmed[1:]
    return trimmed


def chat(
    messages: List[Dict[str, str]],
    context: str = "",
    mode: str = "Research",
    stream: bool = True,
    temperature: float = 0.2
) -> Generator[str, None, None]:
    """
    Stream responses from Groq model with selected response mode and error resilience.
    """
    model = os.getenv("LLAMA_MODEL", "openai/gpt-oss-120b")
    system_prompt = build_system_prompt(context, mode=mode)
    safe_messages = trim_history(messages)

    full_messages = [{"role": "system", "content": system_prompt}] + safe_messages
    groq = get_client()

    try:
        if stream:
            response = groq.chat.completions.create(
                model=model,
                messages=full_messages,
                max_tokens=MAX_TOKENS_RESPONSE,
                temperature=temperature,
                stream=True
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        else:
            response = groq.chat.completions.create(
                model=model,
                messages=full_messages,
                max_tokens=MAX_TOKENS_RESPONSE,
                temperature=temperature,
            )
            yield response.choices[0].message.content or ""

    except Exception as e:
        err = str(e).lower()
        if "rate_limit" in err or "429" in err:
            yield "\n\n⚠️ **Rate limit reached.** Please wait a moment before sending another message."
        elif "token" in err or "context" in err or "length" in err:
            yield "\n\n⚠️ **Context limit reached.** Please click **New Chat** or clear history to reset context."
        elif "model_not_found" in err or "does not exist" in err:
            yield f"\n\n❌ **Model Configuration Error:** {str(e)}. Please check your model configuration."
        else:
            yield f"\n\n❌ **Groq API Error:** {str(e)}"


def generate_followup_questions(
    user_query: str,
    assistant_response: str,
    context: str = ""
) -> List[str]:
    """
    Generate 3 intelligent, highly relevant follow-up questions for the user.
    """
    if not user_query or not assistant_response:
        return []

    model = os.getenv("LLAMA_MODEL", "openai/gpt-oss-120b")
    groq = get_client()

    prompt = (
        f"User asked: '{user_query}'\n\n"
        f"Assistant replied: '{assistant_response[:600]}'\n\n"
        "Generate exactly 3 concise, highly relevant follow-up questions that the user might want to ask next "
        "to explore this document or topic deeper.\n"
        "Rules:\n"
        "- Exactly 3 questions.\n"
        "- Each under 15 words.\n"
        "- Format as a numbered list: 1., 2., 3.\n"
        "- Do not include any other conversational filler or intro text."
    )

    try:
        resp = groq.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful inquiry generator. Output exactly 3 numbered questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.4
        )
        raw_text = resp.choices[0].message.content or ""
        questions = []
        for line in raw_text.splitlines():
            line = line.strip()
            # Match 1. Question or 1) Question or - Question
            cleaned = re.sub(r'^[0-9]+[\.\)]\s*|^-\s*', '', line).strip()
            if cleaned and len(cleaned) > 5 and cleaned.endswith('?'):
                questions.append(cleaned)
            elif cleaned and len(cleaned) > 5 and not cleaned.endswith('?'):
                questions.append(cleaned + "?")
        return questions[:3]
    except Exception:
        # Fallback intelligent suggestions
        return [
            "Can you provide more details on this topic?",
            "What are the key implications mentioned in the document?",
            "Could you summarize the main conclusions?"
        ]
"""
groq_client.py — Groq API integration with LLaMA 3.3 70B
                 Token limit fix: trims history to avoid context overflow
"""
import os
from groq import Groq
from typing import List, Dict, Generator

client = None

# ── Safe limits for llama-3.3-70b-versatile (128k context) ──────────────────
# We keep the last N exchanges so we never approach the token ceiling.
# Each message ≈ 200-400 tokens on average; 20 messages ≈ 8k tokens max.
MAX_HISTORY_MESSAGES = 20   # keep last 20 messages (10 exchanges)
MAX_TOKENS_RESPONSE  = 1024


def get_client() -> Groq:
    global client
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env file")
        client = Groq(api_key=api_key)
    return client


def build_system_prompt(context: str) -> str:
    if context:
        return (
            "You are an intelligent RAG assistant. "
            "Answer the user's question strictly using the context below.\n"
            "If the answer is not in the context, say: "
            "'I couldn't find relevant information in the uploaded documents.'\n"
            "Always mention which source(s) you used in your answer.\n\n"
            f"RETRIEVED CONTEXT:\n{context}"
        )
    return (
        "You are a helpful AI chatbot assistant powered by LLaMA 3.3 70B via Groq. "
        "You help users by answering questions about documents they upload. "
        "No documents have been uploaded yet. "
        "Answer general questions from your own knowledge. "
        "If asked about documents, politely remind the user to upload files "
        "(PDF, DOCX, TXT, CSV) using the sidebar. "
        "Never interpret 'RAG' as 'Red, Amber, Green' — in this app RAG stands for "
        "Retrieval-Augmented Generation, a technique where answers are grounded in "
        "uploaded documents."
    )


def trim_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Keep only the last MAX_HISTORY_MESSAGES messages to avoid token-limit errors.
    Always preserves message pairs (user+assistant) so the conversation stays coherent.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    # Trim from the front, keep an even number so pairs stay together
    trimmed = messages[-MAX_HISTORY_MESSAGES:]
    # If first message is assistant (orphaned), drop it
    if trimmed and trimmed[0]["role"] == "assistant":
        trimmed = trimmed[1:]
    return trimmed


def chat(
    messages: List[Dict[str, str]],
    context: str = "",
    stream: bool = True
) -> Generator[str, None, None]:
    """
    Send messages to Groq LLaMA and yield response chunks.
    Automatically trims history to prevent token-limit errors.
    """
    model         = os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile")
    system_prompt = build_system_prompt(context)
    safe_messages = trim_history(messages)

    full_messages = [{"role": "system", "content": system_prompt}] + safe_messages

    groq = get_client()

    try:
        if stream:
            response = groq.chat.completions.create(
                model=model,
                messages=full_messages,
                max_tokens=MAX_TOKENS_RESPONSE,
                temperature=0.2,
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
                temperature=0.2,
            )
            yield response.choices[0].message.content

    except Exception as e:
        err = str(e).lower()
        if "rate_limit" in err or "429" in err:
            yield "\n\n⚠️ **Rate limit reached.** Please wait a moment and try again."
        elif "token" in err or "context" in err or "length" in err:
            yield "\n\n⚠️ **Conversation too long.** Click **🧹 Clear Chat** in the sidebar to start fresh."
        else:
            yield f"\n\n❌ **Error:** {str(e)}"
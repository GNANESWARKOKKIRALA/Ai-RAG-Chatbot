"""
helpers.py — SQLite setup, session tracking, document metadata, analytics, and chat export
"""
import sqlite3
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = "chat_history.db"


# ── SQLite Database Initialization & Migrations ──────────────────────────────

def init_db() -> None:
    """Create SQLite tables and run backward-compatible column migrations."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Documents table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id           TEXT PRIMARY KEY,
            filename     TEXT NOT NULL,
            file_type    TEXT NOT NULL,
            chunk_count  INTEGER DEFAULT 0,
            file_size_kb REAL DEFAULT 0.0,
            status       TEXT DEFAULT 'indexed',
            uploaded_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Chat history table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,
            message     TEXT NOT NULL,
            username    TEXT DEFAULT 'guest',
            timestamp   TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Chat sessions metadata table (for titles, renaming, searching)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id  TEXT PRIMARY KEY,
            username    TEXT NOT NULL,
            title       TEXT NOT NULL,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # RAG Analytics telemetry table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rag_analytics (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT,
            username         TEXT,
            query            TEXT,
            retrieval_ms     REAL,
            llm_ms           REAL,
            chunks_retrieved INTEGER,
            avg_relevance    REAL,
            mode             TEXT,
            timestamp        TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migrations for documents table columns if upgrading from older schema
    doc_cols = [r[1] for r in cur.execute("PRAGMA table_info(documents)").fetchall()]
    if "file_size_kb" not in doc_cols:
        cur.execute("ALTER TABLE documents ADD COLUMN file_size_kb REAL DEFAULT 0.0")
    if "status" not in doc_cols:
        cur.execute("ALTER TABLE documents ADD COLUMN status TEXT DEFAULT 'indexed'")

    # Migrations for chat_history table columns
    chat_cols = [r[1] for r in cur.execute("PRAGMA table_info(chat_history)").fetchall()]
    if "username" not in chat_cols:
        cur.execute("ALTER TABLE chat_history ADD COLUMN username TEXT DEFAULT 'guest'")

    conn.commit()
    conn.close()


# ── Document Tracking ─────────────────────────────────────────────────────────

def save_document(
    filename: str,
    file_type: str,
    chunk_count: int,
    file_size_kb: float = 0.0,
    status: str = "indexed"
) -> str:
    doc_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO documents (id, filename, file_type, chunk_count, file_size_kb, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (doc_id, filename, file_type, chunk_count, round(file_size_kb, 1), status)
    )
    conn.commit()
    conn.close()
    return doc_id


def get_documents() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_document_record(doc_id: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()


# ── Chat History & Sessions ───────────────────────────────────────────────────

def save_message(session_id: str, role: str, message: str, username: str = "guest") -> None:
    """Save a chat message and automatically register or update the session title."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO chat_history (session_id, role, message, username) VALUES (?, ?, ?, ?)",
        (session_id, role, message, username)
    )

    # If first user message in this session, initialize the session title
    if role == "user":
        cur = conn.cursor()
        existing = cur.execute(
            "SELECT session_id FROM chat_sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        # Derive a clean title from the first 35 chars of user message
        title = message.strip().replace("\n", " ")
        if len(title) > 35:
            title = title[:32] + "..."
        if not title:
            title = "New Conversation"

        if not existing:
            cur.execute(
                "INSERT INTO chat_sessions (session_id, username, title) VALUES (?, ?, ?)",
                (session_id, username, title)
            )
        else:
            cur.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )

    conn.commit()
    conn.close()


def get_history(session_id: str, username: str = None) -> List[Dict[str, str]]:
    """Return chat history for a session."""
    conn = sqlite3.connect(DB_PATH)
    if username:
        rows = conn.execute(
            "SELECT role, message FROM chat_history "
            "WHERE session_id = ? AND username = ? ORDER BY id",
            (session_id, username)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT role, message FROM chat_history WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


def get_user_sessions(username: str, search_query: str = "") -> List[Dict[str, Any]]:
    """Retrieve all conversations for a user with optional title or message search."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if search_query and search_query.strip():
        term = f"%{search_query.strip()}%"
        # Search in title or in message content
        rows = conn.execute("""
            SELECT DISTINCT s.session_id, s.title, s.created_at, s.updated_at
            FROM chat_sessions s
            LEFT JOIN chat_history h ON s.session_id = h.session_id
            WHERE s.username = ? AND (s.title LIKE ? OR h.message LIKE ?)
            ORDER BY s.updated_at DESC
        """, (username, term, term)).fetchall()
    else:
        rows = conn.execute("""
            SELECT session_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE username = ?
            ORDER BY updated_at DESC
        """, (username,)).fetchall()

    conn.close()
    return [dict(r) for r in rows]


def rename_session(session_id: str, new_title: str) -> None:
    if not new_title.strip():
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
        (new_title.strip(), session_id)
    )
    conn.commit()
    conn.close()


def delete_session(session_id: str, username: str = None) -> None:
    """Delete a conversation and all its messages."""
    conn = sqlite3.connect(DB_PATH)
    if username:
        conn.execute("DELETE FROM chat_history WHERE session_id = ? AND username = ?", (session_id, username))
        conn.execute("DELETE FROM chat_sessions WHERE session_id = ? AND username = ?", (session_id, username))
    else:
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def clear_history(session_id: str, username: str = None) -> None:
    delete_session(session_id, username=username)


def new_session_id() -> str:
    return str(uuid.uuid4())


# ── Chat Export ───────────────────────────────────────────────────────────────

def export_chat_as_txt(session_id: str, username: str = None) -> str:
    """Export the conversation as a clean text file."""
    messages = get_history(session_id, username=username)
    if not messages:
        return "No conversation messages to export."

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        "AI RAG CHATBOT — CONVERSATION EXPORT",
        f"Exported At: {now_str}",
        f"Session ID : {session_id}",
        f"User       : {username or 'Guest'}",
        "=" * 60,
        ""
    ]
    for msg in messages:
        role = "USER" if msg["role"] == "user" else "ASSISTANT"
        lines.append(f"[{role}]:")
        lines.append(msg["content"])
        lines.append("-" * 40)
        lines.append("")

    return "\n".join(lines)


def export_chat_as_markdown(session_id: str, username: str = None) -> str:
    """Export the conversation as a styled Markdown document."""
    messages = get_history(session_id, username=username)
    if not messages:
        return "# Conversation Export\n\nNo messages recorded."

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = [
        "# 🤖 AI Document Intelligence Assistant — Chat Transcript",
        f"**Date:** {now_str}  ",
        f"**User:** `{username or 'Guest'}`  ",
        f"**Session:** `{session_id}`",
        "\n---\n"
    ]
    for msg in messages:
        if msg["role"] == "user":
            md.append(f"### 🧑 User\n{msg['content']}\n")
        else:
            md.append(f"### 🤖 Assistant\n{msg['content']}\n")
        md.append("---\n")

    return "\n".join(md)


# ── RAG Analytics Telemetry ───────────────────────────────────────────────────

def log_analytics(
    session_id: str,
    username: str,
    query: str,
    retrieval_ms: float,
    llm_ms: float,
    chunks_retrieved: int,
    avg_relevance: float,
    mode: str
) -> None:
    """Log actual performance metrics for RAG telemetry."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO rag_analytics
           (session_id, username, query, retrieval_ms, llm_ms, chunks_retrieved, avg_relevance, mode)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            username,
            query[:100],
            round(retrieval_ms, 2),
            round(llm_ms, 2),
            chunks_retrieved,
            round(avg_relevance, 4),
            mode
        )
    )
    conn.commit()
    conn.close()


def get_analytics_summary() -> Dict[str, Any]:
    """Calculate real measured metrics across all queries."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    row = cur.execute("""
        SELECT
            COUNT(*) as total_queries,
            AVG(retrieval_ms) as avg_retrieval_ms,
            AVG(llm_ms) as avg_llm_ms,
            AVG(chunks_retrieved) as avg_chunks,
            AVG(avg_relevance) as avg_relevance
        FROM rag_analytics
    """).fetchone()

    recent_queries = cur.execute("""
        SELECT query, retrieval_ms, llm_ms, chunks_retrieved, avg_relevance, mode, timestamp
        FROM rag_analytics
        ORDER BY id DESC
        LIMIT 10
    """).fetchall()

    conn.close()

    total_q = row["total_queries"] if row and row["total_queries"] else 0
    return {
        "total_queries": total_q,
        "avg_retrieval_ms": round(row["avg_retrieval_ms"] or 0.0, 1) if total_q > 0 else 0.0,
        "avg_llm_ms": round(row["avg_llm_ms"] or 0.0, 1) if total_q > 0 else 0.0,
        "avg_chunks": round(row["avg_chunks"] or 0.0, 1) if total_q > 0 else 0.0,
        "avg_relevance": round((row["avg_relevance"] or 0.0) * 100, 1) if total_q > 0 else 0.0,
        "recent_queries": [dict(r) for r in recent_queries]
    }


def format_sources(hits: List[Dict]) -> str:
    """Format source badges string."""
    if not hits:
        return ""
    sources = []
    seen = set()
    for h in hits:
        name = h.get("source", "Document")
        page = h.get("page", 1)
        tag = f"`{name} (p.{page})`"
        if tag not in seen:
            seen.add(tag)
            sources.append(tag)
    return "📎 **Sources:** " + " · ".join(sources)
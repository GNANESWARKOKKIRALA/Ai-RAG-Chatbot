"""
helpers.py — Utility functions: DB init, session, formatting
"""
import sqlite3
import os
import uuid
from datetime import datetime
from typing import List, Dict

DB_PATH = "chat_history.db"


# ── SQLite Setup ────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create SQLite tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            filename    TEXT NOT NULL,
            file_type   TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

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

    # Migration: add username column to existing tables that pre-date auth
    existing_cols = [
        row[1] for row in cur.execute("PRAGMA table_info(chat_history)").fetchall()
    ]
    if "username" not in existing_cols:
        cur.execute(
            "ALTER TABLE chat_history ADD COLUMN username TEXT DEFAULT 'guest'"
        )

    conn.commit()
    conn.close()


# ── Document Tracking ───────────────────────────────────────────────────────

def save_document(filename: str, file_type: str, chunk_count: int) -> str:
    doc_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO documents (id, filename, file_type, chunk_count) VALUES (?, ?, ?, ?)",
        (doc_id, filename, file_type, chunk_count)
    )
    conn.commit()
    conn.close()
    return doc_id


def get_documents() -> List[Dict]:
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


# ── Chat History ─────────────────────────────────────────────────────────────

def save_message(session_id: str, role: str, message: str, username: str = "guest") -> None:
    """Save a chat message, optionally tied to a username."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO chat_history (session_id, role, message, username) VALUES (?, ?, ?, ?)",
        (session_id, role, message, username)
    )
    conn.commit()
    conn.close()


def get_history(session_id: str, username: str = None) -> List[Dict[str, str]]:
    """
    Return chat history for a session.
    If username is supplied, filter to that user's messages only.
    """
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


def clear_history(session_id: str, username: str = None) -> None:
    """Clear chat history. If username given, only clear that user's messages."""
    conn = sqlite3.connect(DB_PATH)
    if username:
        conn.execute(
            "DELETE FROM chat_history WHERE session_id = ? AND username = ?",
            (session_id, username)
        )
    else:
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


# ── Formatting ──────────────────────────────────────────────────────────────

def format_sources(hits: List[Dict]) -> str:
    if not hits:
        return ""
    sources = list({h["source"] for h in hits})
    return "📎 **Sources:** " + " · ".join(f"`{s}`" for s in sources)


def new_session_id() -> str:
    return str(uuid.uuid4())
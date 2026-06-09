"""
auth/db.py — SQLite setup for users table + chat_history username column migration
"""
import sqlite3

DB_PATH = "chat_history.db"


def init_users_db() -> None:
    """Create users table and ensure chat_history has a username column."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Create users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add username column to chat_history if it doesn't already exist
    existing_cols = [
        row[1]
        for row in cur.execute("PRAGMA table_info(chat_history)").fetchall()
    ]
    if "username" not in existing_cols:
        cur.execute(
            "ALTER TABLE chat_history ADD COLUMN username TEXT DEFAULT 'guest'"
        )

    conn.commit()
    conn.close()
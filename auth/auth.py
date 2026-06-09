"""
auth/auth.py — Authentication logic using bcrypt + SQLite
"""
import sqlite3
import bcrypt
from typing import Optional, Dict

DB_PATH = "chat_history.db"


# ── Password Helpers ────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt. Returns a str."""
    hashed = bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── User Management ─────────────────────────────────────────────────────────

def create_user(username: str, password: str, email: str = None) -> Dict:
    """
    Create a new user.
    Returns {"ok": True} on success or {"ok": False, "error": "..."} on failure.
    """
    username = username.strip().lower()
    if not username or not password:
        return {"ok": False, "error": "Username and password are required."}
    if len(password) < 6:
        return {"ok": False, "error": "Password must be at least 6 characters."}

    pw_hash = hash_password(password)
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email.strip().lower() if email else None, pw_hash),
        )
        conn.commit()
        conn.close()
        return {"ok": True}
    except sqlite3.IntegrityError as e:
        err = str(e)
        if "username" in err:
            return {"ok": False, "error": "Username already taken."}
        if "email" in err:
            return {"ok": False, "error": "Email already registered."}
        return {"ok": False, "error": "Registration failed. Please try again."}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """
    Authenticate a user by username + password.
    Returns user dict on success, None on failure.
    """
    username = username.strip().lower()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, username, email, password_hash, created_at "
        "FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()

    if row is None:
        return None
    if not verify_password(password, row["password_hash"]):
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "email": row["email"],
        "created_at": row["created_at"],
    }
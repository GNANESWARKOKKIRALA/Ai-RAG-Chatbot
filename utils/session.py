"""
utils/session.py — Streamlit session_state helpers for auth
"""
import streamlit as st


def init_session() -> None:
    """Initialise auth-related session keys if absent."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None
    # which tab is shown on the auth page: "login" | "signup"
    if "auth_tab" not in st.session_state:
        st.session_state["auth_tab"] = "login"


def login_user(user: dict) -> None:
    """Mark the session as authenticated and store the user dict."""
    st.session_state["authenticated"] = True
    st.session_state["user"] = user
    # Clear any leftover chat state so history reloads per-user
    st.session_state.pop("messages", None)
    st.session_state.pop("session_id", None)


def logout_user() -> None:
    """Clear all session state and return to the login screen."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Re-initialise so the auth page renders cleanly
    init_session()


def is_authenticated() -> bool:
    return st.session_state.get("authenticated", False)


def current_user() -> dict | None:
    return st.session_state.get("user", None)


def current_username() -> str:
    user = current_user()
    return user["username"] if user else "guest"
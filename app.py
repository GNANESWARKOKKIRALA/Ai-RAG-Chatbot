"""
app.py — Streamlit UI for  AI RAG Chatbot (Groq + LLaMA 3.3 70B + ChromaDB + SQLite)
         Auth layer added: login / signup / logout, per-user chat history.
"""
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag.loader import load_file
from rag.chunker import chunk_text
from rag.embedder import embed_texts
from rag.vector_store import add_chunks, reset_store, count_chunks
from rag.retriever import retrieve, format_context
from llm.groq_client import chat
from utils.helpers import (
    init_db, save_document, get_documents, delete_document_record,
    save_message, get_history, clear_history, format_sources, new_session_id
)
from auth.db import init_users_db
from auth.auth import create_user, authenticate_user
from utils.session import (
    init_session, login_user, logout_user,
    is_authenticated, current_user, current_username
)

# ── Init ─────────────────────────────────────────────────────────────────────
init_db()
init_users_db()
init_session()

st.set_page_config(
    page_title=" AI RAG Chatbot — LLaMA 3.3 via Groq",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme Configuration ───────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "system"

# Generate theme CSS based on session state
theme_css = ""
if st.session_state.theme == "light":
    theme_css = """
    :root {
        --chat-bg: #FFFFFF !important;
        --sidebar-bg: #F9F9F9 !important;
        --input-bg: #F4F4F4 !important;
        --text-color: #111111 !important;
        --secondary-text: #6B7280 !important;
        --border-color: #E5E7EB !important;
        --accent-color: #111111 !important;
        --msg-user-bg: #F4F4F4 !important;
        --msg-assistant-bg: #FFFFFF !important;
    }
    """
elif st.session_state.theme == "dark":
    theme_css = """
    :root {
        --chat-bg: #212121 !important;
        --sidebar-bg: #171717 !important;
        --input-bg: #303030 !important;
        --text-color: #FFFFFF !important;
        --secondary-text: #A1A1AA !important;
        --border-color: #303030 !important;
        --accent-color: #FFFFFF !important;
        --msg-user-bg: #303030 !important;
        --msg-assistant-bg: #212121 !important;
    }
    """
else: # system default
    theme_css = """
    :root {
        --chat-bg: #FFFFFF;
        --sidebar-bg: #F9F9F9;
        --input-bg: #F4F4F4;
        --text-color: #111111;
        --secondary-text: #6B7280;
        --border-color: #E5E7EB;
        --accent-color: #111111;
        --msg-user-bg: #F4F4F4;
        --msg-assistant-bg: #FFFFFF;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --chat-bg: #212121;
            --sidebar-bg: #171717;
            --input-bg: #303030;
            --text-color: #FFFFFF;
            --secondary-text: #A1A1AA;
            --border-color: #303030;
            --accent-color: #FFFFFF;
            --msg-user-bg: #303030;
            --msg-assistant-bg: #212121;
        }
    }
    """

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

{theme_css}

html, body, [class*="css"] {{ 
    font-family: 'Inter', sans-serif; 
}}

/* Hide Streamlit default header and footer */
header[data-testid="stHeader"] {{
    display: none !important;
}}
footer {{
    display: none !important;
}}

.stApp {{ 
    background-color: var(--chat-bg) !important; 
    color: var(--text-color) !important; 
}}

/* Sidebar styling (fixed to 260px) */
section[data-testid="stSidebar"] {{
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border-color) !important;
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
    transition: width 200ms ease, min-width 200ms ease, max-width 200ms ease !important;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
    padding-top: 1rem !important;
}}

/* Scrollbars */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: transparent;
}}
::-webkit-scrollbar-thumb {{
    background: var(--border-color);
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: var(--secondary-text);
}}

/* Minimalist Brand header */
.brand-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 4px;
    margin-bottom: 16px;
}}
.brand-logo {{
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--chat-bg) !important;
    background-color: var(--accent-color);
    font-weight: 800;
    font-size: 1rem;
}}
.brand-text {{
    color: var(--text-color) !important;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: -0.3px;
}}

/* Left Sidebar Buttons (e.g. New Chat, Reset, Logout) */
.stButton > button {{
    background-color: transparent !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-color) !important;
    border-radius: 20px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    width: 100% !important;
}}
.stButton > button:hover {{
    background-color: var(--input-bg) !important;
    border-color: var(--accent-color) !important;
}}

/* Primary button pop (New Chat / Submit) */
.stButton > button[kind="primary"] {{
    background-color: var(--accent-color) !important;
    color: var(--chat-bg) !important;
    border: none !important;
    font-weight: 600 !important;
}}
.stButton > button[kind="primary"]:hover {{
    opacity: 0.9 !important;
}}

/* User Profile card pinned at bottom of sidebar */
.user-profile-card {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: transparent;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 12px;
    margin-top: 24px;
}}
.user-profile-avatar {{
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background-color: var(--border-color);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-color);
    font-size: 0.95rem;
}}
.user-profile-info {{
    display: flex;
    flex-direction: column;
}}
.user-profile-name {{
    color: var(--text-color);
    font-weight: 600;
    font-size: 0.85rem;
}}
.user-profile-role {{
    color: var(--secondary-text);
    font-size: 0.72rem;
}}

/* Document chips in sidebar */
.doc-chip-sidebar {{
    background-color: var(--input-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    font-size: 0.8rem !important;
    color: var(--text-color) !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-bottom: 4px;
}}

/* ChatGPT Style Welcome Page (Vertically Centered) */
.chatgpt-welcome-container {{
    text-align: center;
    padding-bottom: 24px;
}}
.chatgpt-welcome-icon {{
    font-size: 2.8rem;
    margin-bottom: 12px;
    display: inline-block;
    color: var(--text-color);
}}
.chatgpt-welcome-title {{
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: var(--text-color) !important;
    letter-spacing: -0.5px !important;
}}

/* Center the main container vertically when welcome page is active */
.main .block-container:has(.chatgpt-welcome-container) {{
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    min-height: 85vh !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}}

/* Suggestion Buttons in Main Chat area (styled like boxes) */
.main .stButton > button {{
    border: 1px solid var(--border-color) !important;
    background-color: transparent !important;
    color: var(--text-color) !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    text-align: left !important;
    font-size: 0.85rem !important;
    display: block !important;
    width: 100% !important;
    min-height: 48px !important;
    line-height: 1.3 !important;
}}
.main .stButton > button:hover {{
    background-color: var(--input-bg) !important;
    border-color: var(--accent-color) !important;
}}

/* Center Chat Area Width constraint */
.main .block-container {{
    max-width: 768px !important;
    margin: 0 auto !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    padding-top: 3rem !important;
    padding-bottom: 7rem !important;
}}

/* ChatGPT Style Messages */
div[data-testid="stChatMessage"] {{
    background-color: var(--chat-bg) !important;
    border: none !important;
    padding: 1rem 0.5rem !important;
    margin-bottom: 1rem !important;
    border-bottom: 1px solid var(--border-color) !important;
    border-radius: 0px !important;
}}

/* User messages get a very subtle background tint to distinguish them, or flat */
div[data-testid="stChatMessage"]:has(img[src*="user"]),
div[data-testid="stChatMessage"]:has(span:contains("🧑")),
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatar"] span:contains("🧑")) {{
    background-color: var(--msg-user-bg) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    border: none !important;
}}

/* Hide default chat message avatars if we want extreme minimalism or keep them clean */
div[data-testid="stChatMessageAvatar"] {{
    background-color: var(--input-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}}

div[data-testid="stChatMessageContent"] {{
    color: var(--text-color) !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
}}

/* ChatGPT Styled Input Bar with internal icons */
div[data-testid="stChatInput"] {{
    background-color: var(--chat-bg) !important;
    border: none !important;
    padding-bottom: 24px !important;
    padding-top: 10px !important;
    max-width: 768px !important;
    margin: 0 auto !important;
}}
div[data-testid="stChatInput"] > div {{
    background-color: var(--input-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 26px !important;
    padding: 6px 12px !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03) !important;
    position: relative !important;
}}
div[data-testid="stChatInput"] > div:focus-within {{
    border-color: var(--accent-color) !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06) !important;
}}

/* Inject paperclip attach icon on left */
div[data-testid="stChatInput"] > div::before {{
    content: "📎" !important;
    position: absolute !important;
    left: 16px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    font-size: 1.05rem !important;
    color: var(--secondary-text) !important;
    opacity: 0.65 !important;
    cursor: pointer !important;
    pointer-events: none !important;
    z-index: 10 !important;
}}

/* Inject microphone voice icon on right */
div[data-testid="stChatInput"] > div::after {{
    content: "🎤" !important;
    position: absolute !important;
    right: 56px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    font-size: 1.05rem !important;
    color: var(--secondary-text) !important;
    opacity: 0.65 !important;
    cursor: pointer !important;
    pointer-events: none !important;
    z-index: 10 !important;
}}

div[data-testid="stChatInput"] textarea {{
    background-color: transparent !important;
    color: var(--text-color) !important;
    font-size: 1rem !important;
    line-height: 1.5 !important;
    padding-left: 42px !important;
    padding-right: 80px !important;
    border: none !important;
}}

/* Tables */
table {{
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1rem 0 !important;
}}
th, td {{
    border: 1px solid var(--border-color) !important;
    padding: 10px 14px !important;
    text-align: left !important;
    font-size: 0.92rem !important;
}}
th {{
    background-color: var(--input-bg) !important;
    color: var(--text-color) !important;
    font-weight: 600 !important;
}}
td {{
    color: var(--text-color) !important;
}}

/* Code block adjustments */
code {{
    color: var(--text-color) !important;
    background-color: var(--input-bg) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.9rem !important;
}}
pre code {{
    padding: 0 !important;
    background-color: transparent !important;
}}

/* Collapsible File Uploader Adjustments */
div[data-testid="stFileUploader"] {{
    background: var(--input-bg) !important;
    border: 1px dashed var(--border-color) !important;
    border-radius: 12px !important;
    padding: 8px !important;
    font-size: 0.8rem !important;
}}
div[data-testid="stFileUploader"] section {{
    padding: 8px !important;
}}

/* Source badges */
.source-badge {{
    display: inline-block;
    background-color: var(--input-bg);
    color: var(--secondary-text);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.78rem;
    margin: 4px 2px;
}}
.source-badge:hover {{
    border-color: var(--accent-color);
    color: var(--text-color);
}}

/* Auth branding custom overrides */
.auth-brand {{
    text-align: center;
    margin-bottom: 28px;
    padding-top: 40px;
}}
.auth-brand-icon {{
    width: 80px;
    height: 80px;
    background-color: var(--accent-color) !important;
    color: var(--chat-bg) !important;
    border-radius: 24px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 2.2rem;
    margin-bottom: 16px;
    box-shadow: none !important;
}}
.auth-brand-title {{
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--text-color) !important;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
}}
.auth-brand-sub {{
    color: var(--secondary-text) !important;
    font-size: 0.95rem;
    margin: 0 0 12px 0;
}}
.auth-pills {{
    display: flex;
    gap: 6px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 10px;
}}
.auth-pill {{
    background-color: var(--input-bg) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-color) !important;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
}}
.auth-form-title {{
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-color) !important;
    margin-bottom: 6px;
}}
.auth-form-sub {{
    color: var(--secondary-text) !important;
    font-size: 0.85rem;
    margin-bottom: 24px;
}}
.auth-divider {{
    text-align: center;
    margin-top: 20px;
    padding-top: 18px;
    border-top: 1px solid var(--border-color) !important;
    color: var(--secondary-text) !important;
    font-size: 0.85rem;
}}
.auth-divider span {{
    color: var(--accent-color) !important;
    font-weight: 600;
}}

/* Smooth transitions */
.stButton > button, div[data-testid="stChatInput"] > div, .doc-chip-sidebar, .source-badge, section[data-testid="stSidebar"] {{
    transition: background-color 200ms ease, border-color 200ms ease, color 200ms ease, box-shadow 200ms ease !important;
}}

/* Mobile responsiveness */
@media (max-width: 768px) {{
    .chatgpt-welcome-title {{ font-size: 1.6rem !important; }}
    div[data-testid="stChatInput"] {{ width: 100% !important; left: 0 !important; }}
}}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# AUTH PAGE  (shown when not logged in)
# ════════════════════════════════════════════════════════════════════════════

def show_auth_page() -> None:
    """Render the login / signup screen."""

    _, col, _ = st.columns([1, 2, 1])

    with col:
        # ── Brand / Hero ─────────────────────────────────────────────────────
        st.markdown("""
        <div class="auth-brand">
          <div class="auth-brand-icon">🤖</div>
          <div class="auth-brand-title">AI RAG Chatbot</div>
          <div class="auth-brand-sub">Your intelligent document assistant</div>
          <div class="auth-pills">
            <span class="auth-pill">⚡ LLaMA 3.3 70B</span>
            <span class="auth-pill">🚀 Groq</span>
            <span class="auth-pill">🗄️ ChromaDB</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── LOGIN ────────────────────────────────────────────────────────────
        if st.session_state.auth_tab == "login":
            st.markdown('<div class="auth-form-title">Welcome back 👋</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-form-sub">Sign in to access your chat history</div>', unsafe_allow_html=True)

            username = st.text_input("Username", placeholder="Enter your username", key="login_user")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")
            st.write("")
            if st.button("🔑  Login", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        login_user(user)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")

            st.markdown("""
            <div class="auth-divider">
              Don't have an account? &nbsp;<span>👇 Click Sign Up below</span>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            if st.button("📝  Sign Up instead", use_container_width=True, type="secondary"):
                st.session_state.auth_tab = "signup"
                st.rerun()

        # ── SIGN UP ──────────────────────────────────────────────────────────
        else:
            st.markdown('<div class="auth-form-title">Create account ✨</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-form-sub">Join and start chatting with your documents</div>', unsafe_allow_html=True)

            new_username = st.text_input("Username", placeholder="Choose a username", key="su_user")
            new_email    = st.text_input("Email (optional)", placeholder="you@example.com", key="su_email")
            new_pass     = st.text_input("Password", type="password", placeholder="Min. 6 characters", key="su_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="su_pass2")
            st.write("")
            if st.button("🚀  Create Account", use_container_width=True, type="primary"):
                if not new_username or not new_pass:
                    st.error("Username and password are required.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    result = create_user(new_username, new_pass, email=new_email or None)
                    if result["ok"]:
                        st.success("✅ Account created! You can now log in.")
                        st.session_state.auth_tab = "login"
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")

            st.markdown("""
            <div class="auth-divider">
              Already have an account? &nbsp;<span>👇 Click Login below</span>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            if st.button("🔑  Login instead", use_container_width=True, type="secondary"):
                st.session_state.auth_tab = "login"
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP  (shown when logged in)
# ════════════════════════════════════════════════════════════════════════════

def show_chatbot() -> None:
    username = current_username()

    # ── Session State ────────────────────────────────────────────────────────
    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    session_id = st.session_state.session_id

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        # Minimalist Brand Header
        st.markdown("""
        <div class="brand-header">
            <div class="brand-logo">🤖</div>
            <div class="brand-text">AI Assistant</div>
        </div>
        """, unsafe_allow_html=True)

        # Primary Action: New Chat (clears conversation history)
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            clear_history(session_id, username=username)
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # Collapsible Document upload panel
        with st.expander("📂 Upload Documents", expanded=False):
            uploaded_files = st.file_uploader(
                "Drag & drop or browse",
                type=["pdf", "txt", "md", "docx", "csv"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

            if uploaded_files:
                for uf in uploaded_files:
                    existing = [d["filename"] for d in get_documents()]
                    if uf.name in existing:
                        st.warning(f"'{uf.name}' already uploaded.")
                        continue

                    with st.spinner(f"Processing {uf.name}…"):
                        try:
                            suffix = os.path.splitext(uf.name)[1]
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(uf.read())
                                tmp_path = tmp.name

                            text        = load_file(tmp_path)
                            os.unlink(tmp_path)
                            chunks_list = chunk_text(text)
                            embeddings  = embed_texts(chunks_list)
                            doc_id      = save_document(uf.name, suffix.lstrip("."), len(chunks_list))
                            add_chunks(chunks_list, embeddings, uf.name, doc_id)
                            st.success(f"✅ {uf.name} — {len(chunks_list)} chunks")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

        st.divider()

        # Document list in sidebar
        docs = get_documents()
        if docs:
            st.markdown("### 📄 Uploaded Documents")
            for doc in docs:
                col_n, col_x = st.columns([5, 1])
                with col_n:
                    ext = doc["filename"].split(".")[-1].lower()
                    icon = "📄"
                    if ext == "pdf":
                        icon = "📕"
                    elif ext in ["doc", "docx"]:
                        icon = "📘"
                    elif ext == "csv":
                        icon = "📊"
                    elif ext in ["txt", "md"]:
                        icon = "📝"
                    st.markdown(
                        f'<div class="doc-chip-sidebar">'
                        f'  <span>{icon} {doc["filename"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col_x:
                    if st.button("🗑️", key=f"del_{doc['id']}", help="Remove document"):
                        from rag.vector_store import delete_document
                        delete_document(doc["id"])
                        delete_document_record(doc["id"])
                        st.rerun()

        st.divider()

        # Theme Manual Toggle Mode
        theme_mode = st.selectbox(
            "🌓 Theme Mode",
            options=["System Default", "Light", "Dark"],
            index=0 if st.session_state.theme == "system" else (1 if st.session_state.theme == "light" else 2)
        )
        new_theme = "system" if theme_mode == "System Default" else theme_mode.lower()
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

        # Database utilities
        if st.button("🗑️ Reset Database", use_container_width=True):
            reset_store()
            clear_history(session_id, username=username)
            st.session_state.messages = []
            conn = __import__("sqlite3").connect("chat_history.db")
            conn.execute("DELETE FROM documents")
            conn.commit()
            conn.close()
            st.rerun()

        st.divider()

        st.markdown(
            '<div style="font-size:0.72rem;color:var(--secondary-text);text-align:center;">'
            'LLaMA 3.3 · Groq · ChromaDB · SQLite'
            '</div>',
            unsafe_allow_html=True
        )

        st.divider()

        # Logged-in user info + logout — pinned to bottom
        st.markdown(
            f'<div class="user-profile-card">'
            f'  <div class="user-profile-avatar">👤</div>'
            f'  <div class="user-profile-info">'
            f'    <div class="user-profile-name">{username}</div>'
            f'    <div class="user-profile-role">Active Account</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            logout_user()
            st.rerun()

    # ── Main Chat Area ────────────────────────────────────────────────────────
    # Load per-user history from DB on first load
    if not st.session_state.messages:
        st.session_state.messages = get_history(session_id, username=username)

    # Welcome / Intro Screen if no messages
    if not st.session_state.messages:
        st.markdown("""
        <div class="chatgpt-welcome-container">
          <div class="chatgpt-welcome-icon">🤖</div>
          <h1 class="chatgpt-welcome-title">How can I help you today?</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Suggested prompts in a 2x2 layout
        sp_col1, sp_col2 = st.columns(2)
        with sp_col1:
            if st.button("📝 Summarize Project Report", use_container_width=True, key="sp1"):
                st.session_state.suggested_prompt = "Summarize Project Report"
                st.rerun()
            if st.button("📊 Explain Q3 Financials", use_container_width=True, key="sp2"):
                st.session_state.suggested_prompt = "Explain Q3 Financials"
                st.rerun()
        with sp_col2:
            if st.button("✍️ Draft Marketing Email", use_container_width=True, key="sp3"):
                st.session_state.suggested_prompt = "Draft Marketing Email"
                st.rerun()
            if st.button("💡 Brainstorm new ideas", use_container_width=True, key="sp4"):
                st.session_state.suggested_prompt = "Brainstorm new marketing ideas"
                st.rerun()

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑"):
            st.markdown(msg["content"])

    # ── Chat Input & Processing ───────────────────────────────────────────────
    clicked_prompt = None
    if "suggested_prompt" in st.session_state and st.session_state.suggested_prompt:
        clicked_prompt = st.session_state.suggested_prompt
        del st.session_state["suggested_prompt"]

    prompt = st.chat_input("Message AI Assistant...")
    if clicked_prompt:
        prompt = clicked_prompt

    if prompt:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(session_id, "user", prompt, username=username)

        # Retrieve context
        hits    = retrieve(prompt) if count_chunks() > 0 else []
        context = format_context(hits)
        sources_text = format_sources(hits)

        # Stream response
        with st.chat_message("assistant", avatar="🤖"):
            placeholder   = st.empty()
            full_response = ""

            history_for_llm = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            for chunk in chat(history_for_llm, context=context, stream=True):
                full_response += chunk
                placeholder.markdown(full_response + "▌")

            if sources_text:
                full_response += f"\n\n{sources_text}"

            placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        save_message(session_id, "assistant", full_response, username=username)
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if is_authenticated():
    show_chatbot()
else:
    show_auth_page()
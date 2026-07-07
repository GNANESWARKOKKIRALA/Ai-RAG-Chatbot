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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
}

.stApp { 
    background-color: #090b15 !important; 
    color: #e2e8f0; 
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #06070d !important;
    border-right: 1px solid #1f244a !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 1.5rem !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #090b15;
}
::-webkit-scrollbar-thumb {
    background: #1f244a;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #3b427b;
}

/* Brand header */
.brand-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 4px;
    margin-bottom: 24px;
}
.brand-logo {
    background: linear-gradient(135deg, #00f2fe 0%, #7c3aed 100%);
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 800;
    font-size: 1.15rem;
    box-shadow: 0 4px 15px rgba(0, 242, 254, 0.35);
}
.brand-text {
    color: #f1f5f9;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: -0.5px;
}

/* New chat buttons & standard buttons */
.stButton > button {
    background: #111326 !important;
    border: 1px solid #1f244a !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton > button:hover {
    background: #1f244a !important;
    border-color: #3b427b !important;
    box-shadow: 0 0 10px rgba(124, 58, 237, 0.15) !important;
}

/* Make primary buttons pop with gradient */
.stButton > button[kind="primary"], 
div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00f2fe 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(0, 242, 254, 0.25) !important;
}
.stButton > button[kind="primary"]:hover,
div[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    opacity: 0.92 !important;
    box-shadow: 0 4px 20px rgba(0, 242, 254, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* User profile card */
.user-profile-card {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #111326;
    border: 1px solid #1f244a;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 12px;
    margin-top: 24px;
}
.user-profile-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #00f2fe, #7c3aed);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.1rem;
    box-shadow: 0 2px 10px rgba(124, 58, 237, 0.25);
}
.user-profile-info {
    display: flex;
    flex-direction: column;
}
.user-profile-name {
    color: #f1f5f9;
    font-weight: 600;
    font-size: 0.9rem;
}
.user-profile-role {
    color: #64748b;
    font-size: 0.75rem;
}

/* Document Chip Styles */
.doc-chip-v2 {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #111326;
    border: 1px solid #1f244a;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}
.doc-chip-v2:hover {
    border-color: #3b427b;
    background: #161933;
}
.doc-icon {
    font-size: 1.25rem;
}
.doc-details {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
    overflow: hidden;
}
.doc-name {
    color: #c4b5fd;
    font-weight: 600;
    font-size: 0.82rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.doc-meta {
    color: #64748b;
    font-size: 0.72rem;
}

/* Meta Card styling */
.meta-card {
    background: #0f1225;
    border: 1px solid #1f244a;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    transition: all 0.3s ease;
}
.meta-card:hover {
    border-color: rgba(124, 92, 250, 0.2);
}
.meta-card-title {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 12px;
}

/* Confidence Bar */
.confidence-bar-container {
    background: #161933;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin-bottom: 8px;
}
.confidence-bar-fill {
    background: linear-gradient(90deg, #00f2fe, #7c3aed);
    height: 100%;
    width: 95%;
    border-radius: 8px;
}
.confidence-text {
    color: #00f2fe;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Status Indicator */
.status-indicator-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
    display: inline-block;
}
.status-text {
    color: #10b981;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Welcome Page Styling */
.welcome-container {
    text-align: center;
    padding: 40px 20px;
    margin-bottom: 30px;
}
.welcome-title {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #f1f5f9 !important;
    margin-bottom: 8px !important;
    letter-spacing: -0.5px !important;
}
.welcome-subtitle {
    color: #64748b !important;
    font-size: 1rem !important;
}
.suggested-prompts-label {
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 14px;
    text-align: center;
}

/* Chat Header V2 */
.chat-header-v2 {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.15) 0%, rgba(37, 99, 235, 0.15) 100%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 16px 24px;
    border-radius: 16px;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
}
.chat-header-title {
    color: white;
    font-size: 1.3rem;
    font-weight: 700;
}
.chat-header-subtitle {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-left: 6px;
}

/* Chat Input Styling */
div[data-testid="stChatInput"] {
    background: #0f1225 !important;
    border: 1px solid #1f244a !important;
    border-radius: 16px !important;
    padding: 8px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3) !important;
}
div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #f1f5f9 !important;
    border: none !important;
    font-size: 0.95rem !important;
}

/* Drag & Drop File Uploader */
div[data-testid="stFileUploader"] {
    background: #0f1225 !important;
    border: 2px dashed #1f244a !important;
    border-radius: 14px !important;
    padding: 14px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #3b427b !important;
}

/* Chat Messages */
div[data-testid="stChatMessage"] {
    background: rgba(19, 22, 43, 0.3) !important;
    border: 1px solid rgba(31, 36, 74, 0.4) !important;
    border-radius: 16px !important;
    padding: 18px !important;
    margin-bottom: 12px !important;
    backdrop-filter: blur(12px) !important;
}
div[data-testid="stChatMessage"]:hover {
    border-color: rgba(124, 92, 250, 0.25) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
}

/* Source badge custom styling */
.source-badge {
    display: inline-block;
    background: #111326;
    color: #a78bfa;
    border: 1px solid #1f244a;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.78rem;
    margin: 4px;
    transition: all 0.2s ease;
}
.source-badge:hover {
    border-color: #7c3aed;
    background: #161933;
}

/* Auth Brand */
.auth-brand {
    text-align: center;
    margin-bottom: 28px;
    padding-top: 40px;
}
.auth-brand-icon {
    width: 80px;
    height: 80px;
    background: linear-gradient(135deg, #00f2fe 0%, #7c3aed 100%);
    border-radius: 24px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 2.2rem;
    margin-bottom: 16px;
    box-shadow: 0 8px 40px rgba(0,242,254,0.3);
}
.auth-brand-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
}
.auth-brand-sub {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0 0 12px 0;
}
.auth-pills {
    display: flex;
    gap: 6px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 10px;
}
.auth-pill {
    background: rgba(0, 242, 254, 0.08);
    border: 1px solid rgba(0, 242, 254, 0.2);
    color: #00f2fe;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
}
.auth-form-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 6px;
}
.auth-form-sub {
    color: #64748b;
    font-size: 0.85rem;
    margin-bottom: 24px;
}
.auth-divider {
    text-align: center;
    margin-top: 20px;
    padding-top: 18px;
    border-top: 1px solid #1f244a;
    color: #475569;
    font-size: 0.85rem;
}
.auth-divider span {
    color: #00f2fe;
    font-weight: 600;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .welcome-title { font-size: 1.6rem !important; }
}
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
        # Brand Header matching UI reference
        st.markdown("""
        <div class="brand-header">
            <div class="brand-logo">AI</div>
            <div class="brand-text">RAG Assistant</div>
        </div>
        """, unsafe_allow_html=True)

        # Primary Action: New Chat (clears conversation history)
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            clear_history(session_id, username=username)
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # Document upload panel
        st.markdown("### 📂 Upload Documents")
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
            '<div style="font-size:0.72rem;color:#475569;text-align:center;">'
            'Built with 🤖 LLaMA 3.3 · Groq · ChromaDB · SQLite'
            '</div>',
            unsafe_allow_html=True
        )

        st.divider()

        # Logged-in user info + logout — pinned to bottom with the premium design
        st.markdown(
            f'<div class="user-profile-card">'
            f'  <div class="user-profile-avatar">👤</div>'
            f'  <div class="user-profile-info">'
            f'    <div class="user-profile-name">{username}</div>'
            f'    <div class="user-profile-role">Active User</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            logout_user()
            st.rerun()

    # ── Main Layout (Chat on Left, Metadata on Right) ─────────────────────────
    chat_col, right_col = st.columns([3.2, 1.3], gap="medium")

    with chat_col:
        # Chat Header V2
        st.markdown("""
        <div class="chat-header-v2">
          <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
              <span class="chat-header-title">AI RAG Assistant</span>
              <span class="chat-header-subtitle">- Workspace</span>
            </div>
            <div style="font-size:0.75rem;color:#64748b;background:#111326;padding:4px 10px;border-radius:8px;border:1px solid #1f244a;">
              Groq LLaMA 3.3 Active
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Load per-user history from DB on first load
        if not st.session_state.messages:
            st.session_state.messages = get_history(session_id, username=username)

        # Welcome message if chat history is empty
        if not st.session_state.messages:
            st.markdown("""
            <div class="welcome-container">
              <h1 class="welcome-title">How can I help you today?</h1>
              <p class="welcome-subtitle">Ask questions about your uploaded documents, or start with a suggestion.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="suggested-prompts-label">Suggested Prompts</div>', unsafe_allow_html=True)
            sp_col1, sp_col2, sp_col3 = st.columns(3)
            with sp_col1:
                if st.button("📝 Summarize Report", use_container_width=True, key="sp1"):
                    st.session_state.suggested_prompt = "Summarize Project Report"
                    st.rerun()
            with sp_col2:
                if st.button("✍️ Draft Marketing", use_container_width=True, key="sp2"):
                    st.session_state.suggested_prompt = "Draft Marketing Email"
                    st.rerun()
            with sp_col3:
                if st.button("📊 Explain Financials", use_container_width=True, key="sp3"):
                    st.session_state.suggested_prompt = "Explain Q3 Financials"
                    st.rerun()
            st.write("")

        # Display messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑"):
                st.markdown(msg["content"])

        # ── Chat Input & Processing ───────────────────────────────────────────
        clicked_prompt = None
        if "suggested_prompt" in st.session_state and st.session_state.suggested_prompt:
            clicked_prompt = st.session_state.suggested_prompt
            del st.session_state["suggested_prompt"]

        prompt = st.chat_input("Ask a question about your documents…")
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

    with right_col:
        # Confidence Score Card
        st.markdown("""
        <div class="meta-card">
          <div class="meta-card-title">Confidence Score</div>
          <div class="confidence-bar-container">
            <div class="confidence-bar-fill"></div>
          </div>
          <div class="confidence-text">95% (Highly Relevant Context)</div>
        </div>
        """, unsafe_allow_html=True)

        # Processing Status
        st.markdown("""
        <div class="meta-card">
          <div class="meta-card-title">System Status</div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span class="status-indicator-dot"></span>
            <span class="status-text">Complete & Indexed</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Indexed chunks count
        chunks = count_chunks()
        st.markdown(f"""
        <div class="meta-card">
          <div class="meta-card-title">Indexed Knowledge Chunks</div>
          <div style="font-size: 1.8rem; font-weight: 700; color: #a78bfa;">{chunks}</div>
          <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px;">Total vector embeddings in ChromaDB</div>
        </div>
        """, unsafe_allow_html=True)

        # Document list
        docs = get_documents()
        if docs:
            st.markdown('<div class="meta-card-title" style="margin-top: 15px; margin-bottom: 10px;">Retrieved Documents</div>', unsafe_allow_html=True)
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
                        f'<div class="doc-chip-v2">'
                        f'  <span class="doc-icon">{icon}</span>'
                        f'  <div class="doc-details">'
                        f'    <div class="doc-name">{doc["filename"]}</div>'
                        f'    <div class="doc-meta">{doc["chunk_count"]} chunks</div>'
                        f'  </div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col_x:
                    if st.button("🗑️", key=f"del_{doc['id']}", help="Remove document"):
                        from rag.vector_store import delete_document
                        delete_document(doc["id"])
                        delete_document_record(doc["id"])
                        st.rerun()
        else:
            st.markdown("""
            <div class="meta-card" style="text-align: center; padding: 24px 10px;">
              <div style="font-size: 1.5rem; margin-bottom: 8px;">📂</div>
              <div style="font-size: 0.8rem; color: #64748b;">No documents uploaded. Upload files in the sidebar to begin RAG.</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Pinned decorative sparkle
        st.markdown("""
        <div style="text-align: center; margin-top: 30px; opacity: 0.15;">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2">
            <path d="M12 3v18M3 12h18M12 3l4 4M12 21l-4-4M3 12l4-4M21 12l-4 4"/>
          </svg>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if is_authenticated():
    show_chatbot()
else:
    show_auth_page()
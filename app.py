"""
app.py — Production-Grade AI Document Intelligence Assistant
Featuring: Hybrid RAG (Vector + BM25), Grounding Verification, Source Citations,
Smart Follow-ups, 4 Chat Modes, Telemetry Analytics, and Professional SaaS UI.
"""
import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# RAG & LLM modules
from rag.loader import load_document_pages
from rag.chunker import chunk_document_pages
from rag.embedder import embed_texts
from rag.vector_store import add_chunks, reset_store, count_chunks, delete_document
from rag.retriever import retrieve_with_latency, format_context, format_citations_list
from rag.verifier import verify_answer
from llm.groq_client import chat, generate_followup_questions, CHAT_MODES

# Utilities & Database
from utils.helpers import (
    init_db, save_document, get_documents, delete_document_record,
    save_message, get_history, clear_history, new_session_id,
    get_user_sessions, rename_session, delete_session,
    export_chat_as_txt, export_chat_as_markdown,
    log_analytics, get_analytics_summary
)
from auth.db import init_users_db
from auth.auth import create_user, authenticate_user
from utils.session import (
    init_session, login_user, logout_user,
    is_authenticated, current_user, current_username
)

# ── Initialization ────────────────────────────────────────────────────────────
init_db()
init_users_db()
init_session()

st.set_page_config(
    page_title="NexusRAG — AI Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Modern Professional SaaS Dark Theme CSS ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}

/* Background & Core Colors */
.stApp {
    background: #090a0f;
    color: #e2e8f0;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0f1118 !important;
    border-right: 1px solid #1e2230 !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* Main Container max width */
.main .block-container {
    max-width: 1050px;
    padding-top: 1.8rem;
    padding-bottom: 5rem;
}

/* Scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #262b3d; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3d445e; }

/* Brand Badge */
.brand-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 2px;
    margin-bottom: 18px;
}
.brand-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
}
.brand-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.3px;
    margin: 0;
}
.brand-sub {
    font-size: 0.72rem;
    color: #94a3b8;
    margin: 0;
}

/* SaaS Metric Card */
.metric-box {
    background: #131722;
    border: 1px solid #202637;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.metric-box:hover {
    border-color: #313a52;
    transform: translateY(-1px);
}
.metric-num {
    font-size: 1.7rem;
    font-weight: 800;
    color: #818cf8;
    letter-spacing: -0.5px;
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #94a3b8;
    margin-top: 4px;
}

/* Sidebar Document Chips & Metrics */
.side-metric {
    background: #141824;
    border: 1px solid #1f2638;
    border-radius: 8px;
    padding: 8px 10px;
    text-align: center;
    margin-bottom: 8px;
}
.side-metric .num {
    font-size: 1.25rem;
    font-weight: 700;
    color: #818cf8;
}
.side-metric .lbl {
    font-size: 0.7rem;
    color: #94a3b8;
}
.doc-chip {
    background: #141824;
    border: 1px solid #1f2638;
    border-radius: 8px;
    padding: 6px 10px;
    margin-bottom: 6px;
    font-size: 0.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.doc-chip .name {
    color: #f1f5f9;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 140px;
}
.doc-chip .meta {
    color: #64748b;
    font-size: 0.72rem;
}

/* Chat Message Containers */
div[data-testid="stChatMessage"] {
    background: #11141e !important;
    border: 1px solid #1d2232 !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.3rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}

div[data-testid="stChatMessage"]:has(span:contains("🧑")),
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatar"] span:contains("🧑")) {
    background: #161b28 !important;
    border-color: #242c40 !important;
}

/* Citation & Verification Cards */
.citation-box {
    background: #141824;
    border: 1px solid #232a3d;
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 10px;
    font-size: 0.83rem;
}
.citation-header {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #a5b4fc;
    font-weight: 600;
    margin-bottom: 6px;
}
.citation-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1b2030;
    border: 1px solid #2a334c;
    border-radius: 6px;
    padding: 4px 10px;
    margin-right: 6px;
    margin-top: 4px;
    font-size: 0.78rem;
    color: #cbd5e1;
}

.verification-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 10px;
}

/* Follow-up question chip */
.followup-header {
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    margin-top: 14px;
    margin-bottom: 6px;
}

/* Custom Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
    border: 1px solid #272f42 !important;
    background: #161a26 !important;
    color: #e2e8f0 !important;
}
.stButton > button:hover {
    border-color: #6366f1 !important;
    color: #ffffff !important;
    background: #1e2334 !important;
    transform: translateY(-1px) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.45) !important;
    filter: brightness(1.1) !important;
}

/* Form text input */
.stTextInput > div > div > input {
    background: #121520 !important;
    border: 1px solid #23293a !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.25) !important;
}

/* Chat Input Bar */
div[data-testid="stChatInput"] {
    background: transparent !important;
}
div[data-testid="stChatInput"] > div {
    background: #141723 !important;
    border: 1.5px solid #252c3e !important;
    border-radius: 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
}
div[data-testid="stChatInput"] > div:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.2) !important;
}

/* User Profile card pinned in sidebar */
.user-profile-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #131722;
    border: 1px solid #222838;
    border-radius: 10px;
    padding: 10px 12px;
    margin-top: 14px;
}
.user-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #252d42;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
}
.user-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: #f1f5f9;
}
.user-role {
    font-size: 0.72rem;
    color: #94a3b8;
}

/* Auth Brand container */
.auth-hero {
    text-align: center;
    padding: 30px 10px 20px 10px;
}
.auth-logo {
    width: 64px;
    height: 64px;
    border-radius: 18px;
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    margin-bottom: 16px;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4);
}
.auth-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
}
.auth-sub {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 18px;
}
.auth-pills {
    display: flex;
    gap: 6px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 24px;
}
.auth-pill {
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.25);
    color: #a5b4fc;
    border-radius: 16px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION VIEW (When not logged in)
# ════════════════════════════════════════════════════════════════════════════

def show_auth_page() -> None:
    _, col, _ = st.columns([1, 1.8, 1])

    with col:
        st.markdown("""
        <div class="auth-hero">
          <div class="auth-logo">🧠</div>
          <div class="auth-title">NexusRAG</div>
          <div class="auth-sub">Enterprise AI Document Intelligence Platform</div>
          <div class="auth-pills">
            <span class="auth-pill">⚡ GPT OSS 120B / Groq</span>
            <span class="auth-pill">🔍 Hybrid BM25 + Vector</span>
            <span class="auth-pill">🛡️ Factual Verification</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.auth_tab == "login":
            st.markdown("#### Sign in to your workspace")
            username = st.text_input("Username", placeholder="Enter username", key="auth_user")
            password = st.text_input("Password", type="password", placeholder="Enter password", key="auth_pass")
            st.write("")
            if st.button("🔑 Sign In", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        login_user(user)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")

            st.write("")
            st.markdown(
                "<div style='text-align:center;font-size:0.83rem;color:#94a3b8;'>Don't have an account?</div>",
                unsafe_allow_html=True
            )
            if st.button("Create an Account", use_container_width=True):
                st.session_state.auth_tab = "signup"
                st.rerun()

        else:
            st.markdown("#### Create your workspace account")
            new_user = st.text_input("Username", placeholder="Choose username", key="su_user")
            new_email = st.text_input("Email (optional)", placeholder="you@domain.com", key="su_email")
            new_pass = st.text_input("Password", type="password", placeholder="Minimum 6 characters", key="su_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="su_pass2")
            st.write("")
            if st.button("🚀 Create Account", use_container_width=True, type="primary"):
                if not new_user or not new_pass:
                    st.error("Username and password are required.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    res = create_user(new_user, new_pass, email=new_email or None)
                    if res["ok"]:
                        st.success("✅ Account created successfully! Please sign in.")
                        st.session_state.auth_tab = "login"
                        st.rerun()
                    else:
                        st.error(f"❌ {res['error']}")

            st.write("")
            if st.button("Already have an account? Sign In", use_container_width=True):
                st.session_state.auth_tab = "login"
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION (When logged in)
# ════════════════════════════════════════════════════════════════════════════

def show_chatbot() -> None:
    username = current_username()

    # Session State defaults
    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "💬 Chat Workspace"
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "Research"
    if "top_k" not in st.session_state:
        st.session_state.top_k = 4
    if "use_hybrid" not in st.session_state:
        st.session_state.use_hybrid = True
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2
    if "queued_prompt" not in st.session_state:
        st.session_state.queued_prompt = None

    session_id = st.session_state.session_id

    # ── Sidebar Navigation & Management ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="brand-container">
          <div class="brand-icon">🧠</div>
          <div>
            <div class="brand-title">NexusRAG</div>
            <div class="brand-sub">Document Intelligence</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Primary Action: Start New Chat
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.session_id = new_session_id()
            st.session_state.messages = []
            st.rerun()

        # ── Document Upload in Sidebar ──────────────────────────────────────────
        st.divider()
        st.markdown("##### 📂 Upload Documents")
        sidebar_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "md", "docx", "csv"],
            accept_multiple_files=True,
            key="sidebar_uploader",
            label_visibility="collapsed"
        )

        if sidebar_files:
            existing_docs = [d["filename"] for d in get_documents()]
            for uf in sidebar_files:
                if uf.name in existing_docs:
                    st.warning(f"'{uf.name}' already indexed.")
                    continue

                with st.spinner(f"Indexing {uf.name}…"):
                    try:
                        suffix = os.path.splitext(uf.name)[1].lower()
                        file_bytes = uf.read()
                        size_kb = len(file_bytes) / 1024.0

                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name

                        pages = load_document_pages(tmp_path)
                        os.unlink(tmp_path)

                        chunks_list = chunk_document_pages(pages, chunk_size=250, overlap=40)
                        if not chunks_list:
                            st.warning(f"No text extracted from '{uf.name}'.")
                            continue

                        chunk_texts = [c["text"] for c in chunks_list]
                        embeddings = embed_texts(chunk_texts)
                        doc_id = save_document(
                            filename=uf.name,
                            file_type=suffix.lstrip("."),
                            chunk_count=len(chunks_list),
                            file_size_kb=size_kb,
                            status="indexed"
                        )
                        add_chunks(chunks_list, embeddings, source=uf.name, doc_id=doc_id)
                        st.success(f"✅ {uf.name} ({len(chunks_list)} chunks)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        # Quick Metrics & Uploaded Document List
        docs = get_documents()
        total_chunks = count_chunks()
        sm_col1, sm_col2 = st.columns(2)
        with sm_col1:
            st.markdown(
                f'<div class="side-metric"><div class="num">{len(docs)}</div><div class="lbl">Documents</div></div>',
                unsafe_allow_html=True
            )
        with sm_col2:
            st.markdown(
                f'<div class="side-metric"><div class="num">{total_chunks}</div><div class="lbl">Chunks</div></div>',
                unsafe_allow_html=True
            )

        if docs:
            with st.expander(f"📄 Uploaded Documents ({len(docs)})", expanded=False):
                for doc in docs:
                    col_n, col_x = st.columns([4, 1])
                    with col_n:
                        st.markdown(
                            f'<div class="doc-chip">'
                            f'  <div><div class="name">📄 {doc["filename"]}</div>'
                            f'  <div class="meta">{doc["chunk_count"]} chunks</div></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with col_x:
                        if st.button("🗑️", key=f"s_del_{doc['id']}", help="Delete document"):
                            delete_document(doc["id"])
                            delete_document_record(doc["id"])
                            st.rerun()

        st.divider()

        # Navigation
        nav_options = ["💬 Chat Workspace", "📚 Knowledge Base", "📊 RAG Analytics", "⚙️ Settings"]
        active_tab = st.radio(
            "Navigation",
            options=nav_options,
            index=nav_options.index(st.session_state.active_tab) if st.session_state.active_tab in nav_options else 0,
            label_visibility="collapsed"
        )
        if active_tab != st.session_state.active_tab:
            st.session_state.active_tab = active_tab
            st.rerun()

        st.divider()

        # Chat History & Search
        st.markdown("##### 🕒 Conversations")
        search_query = st.text_input("Search chats…", placeholder="Filter by keyword", label_visibility="collapsed")
        user_sessions = get_user_sessions(username, search_query=search_query)

        if user_sessions:
            for s in user_sessions[:8]:
                sid = s["session_id"]
                title = s["title"]
                is_active = (sid == session_id)
                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    prefix = "👉 " if is_active else "💬 "
                    if st.button(f"{prefix}{title}", key=f"sess_{sid}", use_container_width=True):
                        st.session_state.session_id = sid
                        st.session_state.messages = get_history(sid, username=username)
                        st.session_state.active_tab = "💬 Chat Workspace"
                        st.rerun()
                with col_del:
                    if st.button("✖", key=f"delsess_{sid}", help="Delete chat"):
                        delete_session(sid, username=username)
                        if is_active:
                            st.session_state.session_id = new_session_id()
                            st.session_state.messages = []
                        st.rerun()
        else:
            st.caption("No previous sessions found.")

        st.divider()

        # Quick Export Options
        st.markdown("##### 📥 Export Chat")
        txt_data = export_chat_as_txt(session_id, username=username)
        md_data = export_chat_as_markdown(session_id, username=username)
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.download_button(
                label="📄 TXT",
                data=txt_data,
                file_name=f"chat_export_{session_id[:8]}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_ex2:
            st.download_button(
                label="📝 Markdown",
                data=md_data,
                file_name=f"chat_export_{session_id[:8]}.md",
                mime="text/markdown",
                use_container_width=True
            )

        st.divider()

        # User Profile & Logout
        st.markdown(f"""
        <div class="user-profile-badge">
          <div class="user-avatar">👤</div>
          <div>
            <div class="user-name">{username}</div>
            <div class="user-role">Authenticated Workspace</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()

    # ════════════════════════════════════════════════════════════════════════════
    # VIEW 1: 💬 CHAT WORKSPACE
    # ════════════════════════════════════════════════════════════════════════════
    if st.session_state.active_tab == "💬 Chat Workspace":
        # Header Toolbar: Model badge + Response Mode selector
        tb_col1, tb_col2 = st.columns([3, 2])
        with tb_col1:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:8px;padding-top:6px;">'
                '<span style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:3px 8px;font-size:0.75rem;font-weight:600;color:#38bdf8;">⚡ GPT OSS 120B (Groq)</span>'
                '<span style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:3px 8px;font-size:0.75rem;font-weight:600;color:#a5b4fc;">🔍 Hybrid BM25+Vector</span>'
                '</div>',
                unsafe_allow_html=True
            )
        with tb_col2:
            selected_mode = st.selectbox(
                "Response Mode",
                options=list(CHAT_MODES.keys()),
                format_func=lambda m: CHAT_MODES[m]["title"],
                index=list(CHAT_MODES.keys()).index(st.session_state.chat_mode),
                label_visibility="collapsed"
            )
            if selected_mode != st.session_state.chat_mode:
                st.session_state.chat_mode = selected_mode
                st.rerun()

        st.write("")

        # Load DB history on initial load
        if not st.session_state.messages:
            st.session_state.messages = get_history(session_id, username=username)

        # Welcome screen if no messages
        if not st.session_state.messages:
            doc_count = len(get_documents())
            st.markdown(f"""
            <div style="background:#10131d;border:1px solid #1f2537;border-radius:14px;padding:30px 24px;margin-bottom:24px;text-align:center;">
              <div style="font-size:2.4rem;margin-bottom:8px;">🧠</div>
              <h2 style="font-size:1.6rem;font-weight:700;color:#f8fafc;margin:0 0 6px 0;">How can I assist your research today?</h2>
              <p style="color:#94a3b8;font-size:0.92rem;max-width:620px;margin:0 auto 16px auto;">
                Ask questions about your uploaded documents. Powered by hybrid retrieval, evidence verification, and precision citations.
              </p>
              <div style="display:inline-flex;gap:12px;background:#171c2b;border:1px solid #273046;border-radius:20px;padding:6px 16px;font-size:0.8rem;color:#cbd5e1;">
                <span>📚 <b>{doc_count}</b> Documents Indexed</span>
                <span>•</span>
                <span>🧩 <b>{count_chunks()}</b> Chunks Ready</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # 4 starter suggested questions
            st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#94a3b8;margin-bottom:8px;'>SUGGESTED STARTER INQUIRIES:</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Summarize the key findings of the uploaded documents", use_container_width=True):
                    st.session_state.queued_prompt = "Summarize the key findings of the uploaded documents."
                    st.rerun()
                if st.button("🔍 What are the major methodologies and specifications mentioned?", use_container_width=True):
                    st.session_state.queued_prompt = "What are the major methodologies and specifications mentioned?"
                    st.rerun()
            with col2:
                if st.button("📊 Explain the core metrics, data points, and results", use_container_width=True):
                    st.session_state.queued_prompt = "Explain the core metrics, data points, and results."
                    st.rerun()
                if st.button("⚠️ Identify potential risks, constraints, or limitations", use_container_width=True):
                    st.session_state.queued_prompt = "Identify potential risks, constraints, or limitations discussed in the documents."
                    st.rerun()

        # Render conversation messages
        for idx, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            avatar = "🤖" if role == "assistant" else "🧑"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])

                # Render metadata cards for assistant messages if present
                if role == "assistant":
                    citations = msg.get("citations", [])
                    verification = msg.get("verification", None)
                    passages = msg.get("passages", [])
                    followups = msg.get("followups", [])

                    # Citations Badge List
                    if citations:
                        cit_html = '<div class="citation-box"><div class="citation-header">📌 Source Citations:</div><div>'
                        for c in citations:
                            src = c["source"]
                            pg = c["page"]
                            score_pct = int(c.get("score", 0.0) * 100)
                            cit_html += f'<span class="citation-item">📄 <b>{src}</b> (p.{pg}) • {score_pct}% relevance</span>'
                        cit_html += '</div></div>'
                        st.markdown(cit_html, unsafe_allow_html=True)

                    # Verification Status Badge
                    if verification:
                        v_color = verification.get("badge_color", "#10b981")
                        v_label = verification.get("label", "Verified Grounding")
                        v_expl = verification.get("explanation", "")
                        st.markdown(
                            f'<div class="verification-badge" style="background:{v_color}20;border:1px solid {v_color}60;color:{v_color};">'
                            f'🛡️ {v_label} — <span style="font-weight:400;opacity:0.9;">{v_expl}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    # Expandable Context Inspector
                    if passages:
                        with st.expander("🔍 Inspect Retrieved Passages"):
                            for p_idx, p in enumerate(passages, 1):
                                st.markdown(f"**Passage {p_idx} — `{p.get('source', 'Doc')}` (Page {p.get('page', 1)}) | Relevance: {int(p.get('score', 0)*100)}%**")
                                st.info(p.get("text", ""))

                    # Interactive Follow-up Question Chips
                    if followups:
                        st.markdown('<div class="followup-header">💡 Suggested Follow-ups:</div>', unsafe_allow_html=True)
                        f_cols = st.columns(len(followups))
                        for f_idx, (q_text, f_col) in enumerate(zip(followups, f_cols)):
                            with f_col:
                                if st.button(f"👉 {q_text}", key=f"fup_{idx}_{f_idx}", use_container_width=True):
                                    st.session_state.queued_prompt = q_text
                                    st.rerun()

        # Handle Chat Input (either typed or queued from follow-up)
        prompt = st.chat_input("Ask a question about your documents…")
        if st.session_state.queued_prompt:
            prompt = st.session_state.queued_prompt
            st.session_state.queued_prompt = None

        if prompt:
            # Display user message
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)

            user_msg_entry = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_msg_entry)
            save_message(session_id, "user", prompt, username=username)

            # Retrieve context with latency measurement
            with st.spinner("Retrieving relevant passages…"):
                hits, retrieval_ms = retrieve_with_latency(
                    prompt,
                    top_k=st.session_state.top_k,
                    use_hybrid=st.session_state.use_hybrid
                )
                context = format_context(hits)
                citations = format_citations_list(hits)

            # Stream Assistant Response
            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                full_response = ""
                history_for_llm = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]

                t_llm_start = time.perf_counter()
                for chunk in chat(
                    history_for_llm,
                    context=context,
                    mode=st.session_state.chat_mode,
                    stream=True,
                    temperature=st.session_state.temperature
                ):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)
                llm_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)

                # Grounding Verification
                verification = verify_answer(full_response, hits)
                v_color = verification.get("badge_color", "#10b981")
                v_label = verification.get("label", "Verified Grounding")
                v_expl = verification.get("explanation", "")

                # Render Citations
                if citations:
                    cit_html = '<div class="citation-box"><div class="citation-header">📌 Source Citations:</div><div>'
                    for c in citations:
                        src = c["source"]
                        pg = c["page"]
                        score_pct = int(c.get("score", 0.0) * 100)
                        cit_html += f'<span class="citation-item">📄 <b>{src}</b> (p.{pg}) • {score_pct}% relevance</span>'
                    cit_html += '</div></div>'
                    st.markdown(cit_html, unsafe_allow_html=True)

                # Render Verification Badge
                st.markdown(
                    f'<div class="verification-badge" style="background:{v_color}20;border:1px solid {v_color}60;color:{v_color};">'
                    f'🛡️ {v_label} — <span style="font-weight:400;opacity:0.9;">{v_expl}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Expandable Inspector
                if hits:
                    with st.expander("🔍 Inspect Retrieved Passages"):
                        for p_idx, p in enumerate(hits, 1):
                            st.markdown(f"**Passage {p_idx} — `{p.get('source', 'Doc')}` (Page {p.get('page', 1)}) | Relevance: {int(p.get('score', 0)*100)}%**")
                            st.info(p.get("text", ""))

                # Generate 3 Follow-up Questions
                followups = generate_followup_questions(prompt, full_response, context=context)

                # Log Telemetry Analytics
                avg_rel = (sum(h.get("score", 0.0) for h in hits) / len(hits)) if hits else 0.0
                log_analytics(
                    session_id=session_id,
                    username=username,
                    query=prompt,
                    retrieval_ms=retrieval_ms,
                    llm_ms=llm_ms,
                    chunks_retrieved=len(hits),
                    avg_relevance=avg_rel,
                    mode=st.session_state.chat_mode
                )

            # Persist assistant message with rich metadata in session state & DB
            assistant_msg_entry = {
                "role": "assistant",
                "content": full_response,
                "citations": citations,
                "verification": verification,
                "passages": hits,
                "followups": followups
            }
            st.session_state.messages.append(assistant_msg_entry)
            save_message(session_id, "assistant", full_response, username=username)
            st.rerun()

    # ════════════════════════════════════════════════════════════════════════════
    # VIEW 2: 📚 KNOWLEDGE BASE
    # ════════════════════════════════════════════════════════════════════════════
    elif st.session_state.active_tab == "📚 Knowledge Base":
        st.markdown("### 📚 Knowledge Base Management")
        st.markdown("Upload, index, and organize your document collection.")

        # Upload Area
        st.markdown("##### 📤 Index New Documents")
        uploaded_files = st.file_uploader(
            "Upload files (PDF, DOCX, TXT, MD, CSV)",
            type=["pdf", "txt", "md", "docx", "csv"],
            accept_multiple_files=True
        )

        if uploaded_files:
            existing_docs = [d["filename"] for d in get_documents()]
            for uf in uploaded_files:
                if uf.name in existing_docs:
                    st.warning(f"⚠️ '{uf.name}' has already been indexed.")
                    continue

                with st.spinner(f"Parsing & Embedding {uf.name}…"):
                    try:
                        suffix = os.path.splitext(uf.name)[1].lower()
                        file_bytes = uf.read()
                        size_kb = len(file_bytes) / 1024.0

                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name

                        pages = load_document_pages(tmp_path)
                        os.unlink(tmp_path)

                        chunks_list = chunk_document_pages(pages, chunk_size=250, overlap=40)
                        if not chunks_list:
                            st.warning(f"No extractable text found in '{uf.name}'.")
                            continue

                        chunk_texts = [c["text"] for c in chunks_list]
                        embeddings = embed_texts(chunk_texts)
                        doc_id = save_document(
                            filename=uf.name,
                            file_type=suffix.lstrip("."),
                            chunk_count=len(chunks_list),
                            file_size_kb=size_kb,
                            status="indexed"
                        )
                        add_chunks(chunks_list, embeddings, source=uf.name, doc_id=doc_id)
                        st.success(f"✅ Successfully indexed **{uf.name}** ({len(chunks_list)} chunks, {round(size_kb, 1)} KB)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error indexing '{uf.name}': {e}")

        st.divider()

        # Indexed Documents Table / Inventory
        st.markdown("##### 📄 Indexed Document Library")
        docs = get_documents()
        if docs:
            for doc in docs:
                with st.container():
                    col_info, col_del = st.columns([5, 1])
                    with col_info:
                        size_str = f"{doc.get('file_size_kb', 0.0)} KB" if doc.get('file_size_kb') else "N/A"
                        st.markdown(
                            f'<div style="background:#131722;border:1px solid #202738;border-radius:10px;padding:12px 16px;margin-bottom:8px;">'
                            f'<div style="display:flex;align-items:center;gap:10px;">'
                            f'<span style="font-size:1.2rem;">📄</span>'
                            f'<div>'
                            f'<div style="font-weight:600;color:#f1f5f9;font-size:0.92rem;">{doc["filename"]}</div>'
                            f'<div style="font-size:0.75rem;color:#94a3b8;margin-top:2px;">'
                            f'Format: <span style="color:#a5b4fc;">{doc["file_type"].upper()}</span> • '
                            f'Size: <span style="color:#a5b4fc;">{size_str}</span> • '
                            f'Chunks: <span style="color:#a5b4fc;">{doc["chunk_count"]}</span> • '
                            f'Indexed: {doc["uploaded_at"]}'
                            f'</div></div></div></div>',
                            unsafe_allow_html=True
                        )
                    with col_del:
                        st.write("")
                        if st.button("🗑️ Delete", key=f"kb_del_{doc['id']}", use_container_width=True):
                            delete_document(doc["id"])
                            delete_document_record(doc["id"])
                            st.success(f"Removed '{doc['filename']}'")
                            st.rerun()
        else:
            st.info("No documents currently indexed. Upload files above to populate your knowledge base.")

    # ════════════════════════════════════════════════════════════════════════════
    # VIEW 3: 📊 RAG ANALYTICS DASHBOARD
    # ════════════════════════════════════════════════════════════════════════════
    elif st.session_state.active_tab == "📊 RAG Analytics":
        st.markdown("### 📊 RAG Performance & Telemetry Analytics")
        st.markdown("Real-time measured latency, relevance scoring, and indexing capacity metrics.")

        summary = get_analytics_summary()
        docs = get_documents()
        total_chunks = count_chunks()

        # KPI Metric Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-num">{len(docs)}</div>
              <div class="metric-label">Indexed Documents</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-num">{total_chunks}</div>
              <div class="metric-label">Total Vector Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-num">{summary['avg_retrieval_ms']} <span style="font-size:1rem;color:#94a3b8;">ms</span></div>
              <div class="metric-label">Avg Retrieval Latency</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-num">{summary['avg_llm_ms']} <span style="font-size:1rem;color:#94a3b8;">ms</span></div>
              <div class="metric-label">Avg Response Latency</div>
            </div>
            """, unsafe_allow_html=True)

        kpi5, kpi6, kpi7, kpi8 = st.columns(4)
        with kpi5:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-num">{summary['total_queries']}</div>
              <div class="metric-label">Total Inquiries Logged</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi6:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-num">{summary['avg_chunks']}</div>
              <div class="metric-label">Avg Chunks / Query</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi7:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-num">{summary['avg_relevance']}%</div>
              <div class="metric-label">Avg Relevance Score</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi8:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-num" style="color:#10b981;">100%</div>
              <div class="metric-label">System Health</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Recent Query Log
        st.markdown("##### 📜 Recent Query Telemetry Log")
        recent = summary.get("recent_queries", [])
        if recent:
            log_data = []
            for r in recent:
                log_data.append({
                    "Query Excerpt": r["query"],
                    "Retrieval Latency": f"{r['retrieval_ms']} ms",
                    "LLM Latency": f"{r['llm_ms']} ms",
                    "Chunks": r["chunks_retrieved"],
                    "Relevance": f"{int(r['avg_relevance'] * 100)}%",
                    "Mode": r["mode"],
                    "Timestamp": r["timestamp"]
                })
            st.dataframe(log_data, use_container_width=True)
        else:
            st.info("No query telemetry recorded yet. Ask questions in the Chat Workspace to see real benchmarks.")

    # ════════════════════════════════════════════════════════════════════════════
    # VIEW 4: ⚙️ SETTINGS
    # ════════════════════════════════════════════════════════════════════════════
    elif st.session_state.active_tab == "⚙️ Settings":
        st.markdown("### ⚙️ Engine Settings & Calibration")
        st.markdown("Tune retrieval thresholds, hybrid search weights, and model parameters.")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("##### 🔍 RAG Retrieval Tuning")
            top_k_val = st.slider("Top-K Chunks Retrieved", min_value=2, max_value=8, value=st.session_state.top_k)
            st.session_state.top_k = top_k_val

            hybrid_toggle = st.toggle("Enable Hybrid Search (Dense Vector + BM25 Lexical RRF)", value=st.session_state.use_hybrid)
            st.session_state.use_hybrid = hybrid_toggle

            st.caption("Hybrid search fuses cosine embeddings with BM25 lexical term matching for superior keyword and acronym accuracy.")

        with col_s2:
            st.markdown("##### 🤖 Model Parameters")
            temp_val = st.slider("Temperature", min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.05)
            st.session_state.temperature = temp_val

            st.text_input("Active LLM Model", value=os.getenv("LLAMA_MODEL", "openai/gpt-oss-120b"), disabled=True)
            st.text_input("Embedding Model", value=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"), disabled=True)

        st.divider()

        # Database Management
        st.markdown("##### 🗄️ Database Operations")
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            if st.button("🧹 Clear Current Conversation", use_container_width=True):
                clear_history(session_id, username=username)
                st.session_state.messages = []
                st.success("Conversation cleared.")
                st.rerun()
        with col_db2:
            if st.button("🗑️ Reset Entire Vector Store & Documents", use_container_width=True):
                reset_store()
                clear_history(session_id, username=username)
                st.session_state.messages = []
                conn = __import__("sqlite3").connect("chat_history.db")
                conn.execute("DELETE FROM documents")
                conn.execute("DELETE FROM rag_analytics")
                conn.commit()
                conn.close()
                st.success("Vector store and document database successfully reset.")
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if is_authenticated():
    show_chatbot()
else:
    show_auth_page()
"""
app.py — GAPX-AI: Intelligent Document Assistant
Ask · Discover · Get Trusted Answers
High accuracy hybrid RAG with clean, professional dark SaaS interface.
"""
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# RAG & LLM modules
from rag.loader import load_document_pages
from rag.chunker import chunk_document_pages
from rag.embedder import embed_texts
from rag.vector_store import add_chunks, reset_store, count_chunks, delete_document
from rag.retriever import retrieve, format_context, format_sources
from llm.groq_client import chat

# Utilities & Database
from utils.helpers import (
    init_db, save_document, get_documents, delete_document_record,
    save_message, get_history, clear_history, new_session_id
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
    page_title="GAPX-AI — Intelligent Document Assistant",
    page_icon="assets/gapx_banner.png" if os.path.exists("assets/gapx_banner.png") else "🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Professional Clean SaaS Dark Theme CSS ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}

.stApp {
    background: #08090e;
    color: #e2e8f0;
}

/* Fix: Top Streamlit Header & Toolbar Visibility (No Clipping) */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.5rem !important;
}

.main .block-container {
    max-width: 950px;
    padding-top: 4.5rem !important;
    padding-bottom: 6rem !important;
}

/* Sidebar styling & top spacing */
section[data-testid="stSidebar"] {
    background: #0d0f17 !important;
    border-right: 1px solid #1a1e2d !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 3.2rem !important;
    padding-bottom: 2rem !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #222738; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #353d54; }

/* GAPX-AI Header Banner in Chat */
.chat-header {
    background: linear-gradient(135deg, #6d28d9 0%, #2563eb 100%);
    padding: 22px 28px;
    border-radius: 16px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 6px 24px rgba(109, 40, 217, 0.3);
}
.chat-header-logo {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.18);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    font-weight: 800;
    color: #ffffff;
    box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.3);
}
.chat-header h1 {
    color: #ffffff;
    font-size: 1.55rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.4px;
}
.chat-header p {
    color: rgba(255, 255, 255, 0.88);
    margin: 3px 0 0 0;
    font-size: 0.88rem;
    font-weight: 400;
}

/* Sidebar Brand Header */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 2px;
    margin-bottom: 16px;
}
.sidebar-logo {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    font-weight: 800;
    color: #ffffff;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.4);
}
.sidebar-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.3px;
    margin: 0;
}
.sidebar-sub {
    font-size: 0.72rem;
    color: #a5b4fc;
    font-weight: 500;
    margin: 0;
}

/* Sidebar Metric Cards */
.side-metric {
    background: #111420;
    border: 1px solid #1e2434;
    border-radius: 10px;
    padding: 10px 12px;
    text-align: center;
    margin-bottom: 12px;
}
.side-metric .num {
    font-size: 1.4rem;
    font-weight: 800;
    color: #a78bfa;
}
.side-metric .lbl {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* Document Chips */
.doc-chip {
    background: #111420;
    border: 1px solid #1e2434;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 0.82rem;
}
.doc-chip .name {
    color: #f1f5f9;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.doc-chip .meta {
    color: #64748b;
    font-size: 0.72rem;
    margin-top: 2px;
}

/* Chat Messages */
div[data-testid="stChatMessage"] {
    background: #10131d !important;
    border: 1px solid #1c2130 !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.3rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}

div[data-testid="stChatMessage"]:has(span:contains("🧑")),
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatar"] span:contains("🧑")) {
    background: #141825 !important;
    border-color: #21293c !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    border: 1px solid #232a3d !important;
    background: #131724 !important;
    color: #e2e8f0 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #7c3aed !important;
    background: #1b2031 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
    filter: brightness(1.1) !important;
}

/* Chat Input Bar */
div[data-testid="stChatInput"] {
    background: transparent !important;
}
div[data-testid="stChatInput"] > div {
    background: #111420 !important;
    border: 1.5px solid #22293c !important;
    border-radius: 24px !important;
    box-shadow: 0 4px 22px rgba(0, 0, 0, 0.4) !important;
}
div[data-testid="stChatInput"] > div:focus-within {
    border-color: #7c3aed !important;
    box-shadow: 0 4px 26px rgba(124, 58, 237, 0.25) !important;
}

/* User Profile Badge */
.user-profile-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #111420;
    border: 1px solid #1e2435;
    border-radius: 10px;
    padding: 10px 12px;
    margin-top: 14px;
}
.user-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #232a3d;
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

/* Auth Pages Styling (Simple & Professional) */
.auth-hero {
    text-align: center;
    padding: 36px 10px 20px 10px;
}
.auth-logo {
    width: 68px;
    height: 68px;
    border-radius: 18px;
    background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 14px;
    box-shadow: 0 8px 32px rgba(124, 58, 237, 0.45);
}
.auth-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.auth-sub {
    color: #a5b4fc;
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 4px;
}
.auth-tagline {
    color: #94a3b8;
    font-size: 0.82rem;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION PAGE (Professional, Clean & Jargon-Free)
# ════════════════════════════════════════════════════════════════════════════

def show_auth_page() -> None:
    _, col, _ = st.columns([1, 1.8, 1])

    with col:
        st.markdown("""
        <div class="auth-hero">
          <div class="auth-logo">G</div>
          <div class="auth-title">GAPX-AI</div>
          <div class="auth-sub">Intelligent Document Assistant</div>
          <div class="auth-tagline">Ask · Discover · Get Trusted Answers</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.auth_tab == "login":
            st.markdown("#### Welcome Back")
            st.markdown("<p style='font-size:0.85rem;color:#94a3b8;margin-bottom:16px;'>Sign in to access your document assistant</p>", unsafe_allow_html=True)

            username = st.text_input("Username or Email", placeholder="Enter your username or email", key="auth_user")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="auth_pass")
            st.write("")
            if st.button("Sign In", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        login_user(user)
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please check your username and password.")

            st.write("")
            st.markdown(
                "<div style='text-align:center;font-size:0.83rem;color:#94a3b8;'>Don't have an account?</div>",
                unsafe_allow_html=True
            )
            if st.button("Create an Account", use_container_width=True):
                st.session_state.auth_tab = "signup"
                st.rerun()

        else:
            st.markdown("#### Create Your Account")
            st.markdown("<p style='font-size:0.85rem;color:#94a3b8;margin-bottom:16px;'>Join and start chatting with your documents</p>", unsafe_allow_html=True)

            new_user = st.text_input("Username", placeholder="Choose a username", key="su_user")
            new_email = st.text_input("Email", placeholder="your.name@example.com", key="su_email")
            new_pass = st.text_input("Password", type="password", placeholder="Create a password (minimum 6 characters)", key="su_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="su_pass2")
            st.write("")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if not new_user or not new_pass:
                    st.error("Username and password are required.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    res = create_user(new_user, new_pass, email=new_email or None)
                    if res["ok"]:
                        st.success("Account created successfully! You can now sign in.")
                        st.session_state.auth_tab = "login"
                        st.rerun()
                    else:
                        st.error(f"{res['error']}")

            st.write("")
            if st.button("Already have an account? Sign In", use_container_width=True):
                st.session_state.auth_tab = "login"
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# MAIN CHATBOT APPLICATION (Clean, Single-Screen Layout)
# ════════════════════════════════════════════════════════════════════════════

def show_chatbot() -> None:
    username = current_username()

    # Session State defaults
    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    session_id = st.session_state.session_id

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
          <div class="sidebar-logo">G</div>
          <div>
            <div class="sidebar-title">GAPX-AI</div>
            <div class="sidebar-sub">Intelligent Document Assistant</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Document and Chunk Statistics
        docs = get_documents()
        chunks = count_chunks()
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown(
                f'<div class="side-metric"><div class="num">{len(docs)}</div>'
                f'<div class="lbl">Documents</div></div>',
                unsafe_allow_html=True
            )
        with mcol2:
            st.markdown(
                f'<div class="side-metric"><div class="num">{chunks}</div>'
                f'<div class="lbl">Chunks</div></div>',
                unsafe_allow_html=True
            )

        # Document Uploader in Sidebar
        st.markdown("### 📂 Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag & drop or browse",
            type=["pdf", "txt", "md", "docx", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            existing_docs = [d["filename"] for d in get_documents()]
            for uf in uploaded_files:
                if uf.name in existing_docs:
                    st.warning(f"'{uf.name}' already uploaded.")
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

                        chunks_list = chunk_document_pages(pages, chunk_size=500, overlap=50)
                        if not chunks_list:
                            st.warning(f"No readable text found in '{uf.name}'.")
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
                        st.success(f"✅ {uf.name} — {len(chunks_list)} chunks")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing {uf.name}: {e}")

        # Document list in sidebar
        docs = get_documents()
        if docs:
            st.markdown("### 📄 Uploaded Documents")
            for doc in docs:
                col_n, col_x = st.columns([4, 1])
                with col_n:
                    st.markdown(
                        f'<div class="doc-chip">'
                        f'<div class="name">📄 {doc["filename"]}</div>'
                        f'<div class="meta">{doc["chunk_count"]} chunks</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col_x:
                    if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                        delete_document(doc["id"])
                        delete_document_record(doc["id"])
                        st.rerun()

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🧹 Clear Chat", use_container_width=True):
                clear_history(session_id, username=username)
                st.session_state.messages = []
                st.rerun()
        with col_b:
            if st.button("🗑️ Reset All", use_container_width=True):
                reset_store()
                clear_history(session_id, username=username)
                st.session_state.messages = []
                conn = __import__("sqlite3").connect("chat_history.db")
                conn.execute("DELETE FROM documents")
                conn.commit()
                conn.close()
                st.rerun()

        st.divider()

        # User profile & Logout
        st.markdown(f"""
        <div class="user-profile-badge">
          <div class="user-avatar">👤</div>
          <div>
            <div class="user-name">{username}</div>
            <div class="user-role">Document Workspace</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()

    # ── Main Chat Area ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="chat-header">
      <div class="chat-header-logo">G</div>
      <div>
        <h1>GAPX-AI</h1>
        <p>Intelligent Document Assistant · Ask · Discover · Get Trusted Answers</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Load per-user history from DB on first load
    if not st.session_state.messages:
        st.session_state.messages = get_history(session_id, username=username)

    # Display conversation messages
    for msg in st.session_state.messages:
        avatar = "🔮" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Welcome message when conversation is empty
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🔮"):
            st.markdown(
                f"👋 **Welcome back, {username}!** I'm **GAPX-AI**, your Intelligent Document Assistant.\n\n"
                "📂 **Upload documents** in the sidebar (PDF, DOCX, TXT, CSV) and ask me anything about them.\n\n"
                "✨ Answers are strictly verified and grounded in your uploaded files with source citations."
            )

    # ── Chat Input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your documents…"):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(session_id, "user", prompt, username=username)

        # Retrieve context using improved Hybrid Search (Vector + BM25)
        hits = retrieve(prompt) if count_chunks() > 0 else []
        context = format_context(hits)
        sources_text = format_sources(hits)

        # Stream high-accuracy response
        with st.chat_message("assistant", avatar="🔮"):
            placeholder = st.empty()
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
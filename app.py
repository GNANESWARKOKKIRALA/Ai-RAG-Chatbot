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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0f0f14; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background: #13131a !important;
    border-right: 1px solid #2d2d3d;
}

.chat-header {
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
    padding: 20px 28px;
    border-radius: 14px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.chat-header h1 { color: white; font-size: 1.5rem; font-weight: 700; margin: 0; }
.chat-header p  { color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem; }

.metric-card {
    background: #1a1a26;
    border: 1px solid #2d2d3d;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-card .num { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.metric-card .lbl { font-size: 0.75rem; color: #94a3b8; margin-top: 2px; }

.doc-chip {
    background: #1e1e2e;
    border: 1px solid #3b3b52;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
}
.doc-chip .name { color: #c4b5fd; font-weight: 500; }
.doc-chip .meta { color: #64748b; }

.source-badge {
    display: inline-block;
    background: #1e1b4b;
    color: #a5b4fc;
    border: 1px solid #3730a3;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    margin: 2px;
}

[data-testid="stChatMessage"] {
    background: #16161f !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}

.stTextInput > div > div > input {
    background: #1a1a26 !important;
    border: 1px solid #3b3b52 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

.stButton > button {
    border-radius: 20px 4px 20px 4px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: none !important;
}

/* Primary Button Styling - Login, Create Account, Send, etc. */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    border-radius: 4px 20px 4px 20px !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.55) !important;
    transform: translateY(-2px) !important;
    filter: brightness(1.1) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(1px) !important;
    box-shadow: 0 2px 8px rgba(139, 92, 246, 0.35) !important;
}

/* Secondary Button Styling - Sign Up instead, Login instead, Reset Database, etc. */
.stButton > button[kind="secondary"] {
    background: linear-gradient(#1a1a26, #1a1a26) padding-box, linear-gradient(135deg, #8b5cf6, #3b82f6) border-box !important;
    color: #e2e8f0 !important;
    border: 1.5px solid transparent !important;
    box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.05) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-radius: 4px 20px 4px 20px !important;
    background: linear-gradient(rgba(139, 92, 246, 0.15), rgba(139, 92, 246, 0.15)) padding-box, linear-gradient(135deg, #8b5cf6, #3b82f6) border-box !important;
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.25) !important;
    color: white !important;
    transform: translateY(-2px) !important;
}
.stButton > button[kind="secondary"]:active {
    transform: translateY(1px) !important;
}

/* Chat input submit button styling */
button[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important;
    color: white !important;
    border-radius: 50% !important;
    transition: all 0.3s ease !important;
}
button[data-testid="stChatInputSubmitButton"]:hover {
    transform: scale(1.1) !important;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.5) !important;
}

div[data-testid="stFileUploader"] {
    background: #1a1a26;
    border: 2px dashed #3b3b52;
    border-radius: 12px;
    padding: 10px;
}

/* ── Auth page ── */
.auth-brand {
    text-align: center;
    margin-bottom: 28px;
    padding-top: 40px;
}
.auth-brand-icon {
    width: 80px;
    height: 80px;
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
    border-radius: 24px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 2.2rem;
    margin-bottom: 16px;
    box-shadow: 0 8px 40px rgba(124,58,237,0.4);
}
.auth-brand-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
}
.auth-brand-sub {
    color: #64748b;
    font-size: 0.92rem;
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
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.25);
    color: #a78bfa;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 500;
}
.auth-form-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.auth-form-sub {
    color: #64748b;
    font-size: 0.82rem;
    margin-bottom: 20px;
}
.auth-divider {
    text-align: center;
    margin-top: 18px;
    padding-top: 16px;
    border-top: 1px solid #1e1e2e;
    color: #475569;
    font-size: 0.82rem;
}
.auth-divider span {
    color: #a78bfa;
    font-weight: 600;
}

/* Mobile responsiveness */
@media (max-width: 640px) {
    .chat-header { padding: 14px 16px; }
    .chat-header h1 { font-size: 1.1rem; }
    .auth-card { padding: 24px 16px; margin: 0 8px 24px 8px; }
    .auth-hero { padding: 32px 16px 20px 16px; }
    .auth-hero h1 { font-size: 1.4rem; }
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
        st.markdown("## 🤖  AI RAG Chatbot")
        st.markdown("**Powered by LLaMA 3.3 70B + Groq**")
        st.divider()

        # Metrics
        docs   = get_documents()
        chunks = count_chunks()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="metric-card"><div class="num">{len(docs)}</div>'
                f'<div class="lbl">Documents</div></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div class="metric-card"><div class="num">{chunks}</div>'
                f'<div class="lbl">Chunks</div></div>',
                unsafe_allow_html=True
            )

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
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        # Document list
        docs = get_documents()
        if docs:
            st.markdown("### 📄 Uploaded Documents")
            for doc in docs:
                col_n, col_x = st.columns([4, 1])
                with col_n:
                    st.markdown(
                        f'<div class="doc-chip">'
                        f'<span class="name">📄 {doc["filename"]}</span>'
                        f'<span class="meta">{doc["chunk_count"]} chunks</span>'
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
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🧹 Clear Chat"):
                clear_history(session_id, username=username)
                st.session_state.messages = []
                st.rerun()
        with col_b:
            if st.button("🗑️ Reset All"):
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
            'Built with 🤖LaMA 3.3 70B · Groq · ChromaDB · SQLite · Streamlit'
            '</div>',
            unsafe_allow_html=True
        )
        st.divider()

        # Logged-in user info + logout — pinned to bottom
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
            f'<span style="background:#1e1e2e;border:1px solid #3b3b52;border-radius:50%;'
            f'width:32px;height:32px;display:inline-flex;align-items:center;justify-content:center;font-size:1rem;">👤</span>'
            f'<span style="color:#c4b5fd;font-weight:600;">{username}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()

    # ── Main Chat Area ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="chat-header">
      <div style="font-size:2rem;"></div>
      <div>
        <h1>🤖AI RAG Chatbot</h1>
        <p>Ask questions about your documents · Powered by LLaMA 3.3 70B via Groq API</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Load per-user history from DB on first load
    if not st.session_state.messages:
        st.session_state.messages = get_history(session_id, username=username)

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑"):
            st.markdown(msg["content"])

    # Welcome message
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="😎"):
            st.markdown(
                f"👋 **Welcome back, {username}!** I'm your AI RAG Chatbot powered by **LLaMA 3.3 70B via Groq**.\n\n"
                "📂 Upload documents in the sidebar (PDF, DOCX, TXT, CSV) and ask me anything about them.\n\n"
                "⚡ Groq gives **ultra-fast** responses — try it!"
            )

    # ── Chat Input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your documents…"):
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


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if is_authenticated():
    show_chatbot()
else:
    show_auth_page()
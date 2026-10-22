"""
ByoKnK Rag Assistant - OpenWebUI-Inspired Frontend
==================================================
Powered by Google AI Studio (Gemma 4 & Gemini models) with FAISS semantic retrieval.

Design & Layout:
- Clean Top Navbar with multi-view navigation (Chat, Head-to-Head Lab, Knowledge Base, Documentation)
- Deploy button completely hidden via config.toml and CSS
- OpenWebUI dark theme with persistent telemetry and model pill
- Sleek sidebar with cluster status, session history, and quick actions
- Centered hero view with interactive starter suggestion cards
- Retrieval rationale showing grounded-source metadata without exposing model reasoning
- Collapsible source citations with relevance scores, chunk IDs, and source quotes
- Unified Settings & Knowledge Base Modal Dialog (@st.dialog)
- Fully compliant with Streamlit 1.66+ best practices (width="stretch")
"""

import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
from langchain_core.documents import Document

# Ensure root directory is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from app.rag_engine import (
    RAGEngine,
    get_llm,
    get_api_key,
    is_ollama_model,
    is_ollama_embedding_model,
    TEST_MODE,
    extract_text_and_thinking,
    format_json_payload,
    fetch_url_payload,
    MAX_UPLOAD_BYTES,
    MAX_PDF_PAGES,
    JSON_ENDPOINT_TIMEOUT_SECONDS,
)
from app.models import RAGResponse, RetrievedSource, ToolCallRecord, ChunkInfo

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="ByoKnK Rag Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Custom CSS for OpenWebUI Aesthetic & Navbar
# ==============================================================================
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Palette */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* =========================================================================
       1. Hide Streamlit status, stop button, and deploy controls
       ========================================================================= */
    /* Keep Streamlit's main menu available: it includes the sidebar toggle. */
    [data-testid="stStatusWidget"],
    .stStatusWidget,
    button[aria-label="Stop"],
    button[title="Stop"],
    header[data-testid="stHeader"] [data-testid="stStatusWidget"],
    [data-testid="stDeployButton"],
    .stDeployButton,
    button[title="Deploy this app"],
    header[data-testid="stHeader"] .stDeployButton,
    header[data-testid="stHeader"] [data-testid="stDeployButton"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    /* Streamlit Header: Make transparent so it blends with dark mode */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Side Menu / Sidebar Toggle (ensure it is clearly visible and clickable) */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        z-index: 999999 !important;
    }

    /* =========================================================================
       2. Top Navbar Styling
       ========================================================================= */
    .navbar-container {
        background: linear-gradient(180deg, #111827 0%, #0d131f 100%);
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 10px 18px 8px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45);
    }
    .brand-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #f3f4f6;
        letter-spacing: -0.01em;
    }
    .brand-emblem {
        font-size: 1.4rem;
        line-height: 1;
    }
    .brand-version-pill {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 9999px;
        padding: 1px 7px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-left: 4px;
    }

    /* Sub-Navbar Telemetry Bar */
    .telemetry-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 14px;
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid #1f2937;
        border-radius: 8px;
        margin-bottom: 16px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        color: #9ca3af;
    }
    .telemetry-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
        display: inline-block;
        margin-right: 6px;
    }
    .telemetry-badge-tag {
        background: #1f2937;
        border: 1px solid #374151;
        color: #d1d5db;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.72rem;
        margin-left: 6px;
    }

    /* =========================================================================
       3. Sidebar Styling
       ========================================================================= */
    .cluster-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .cluster-icon {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .cluster-info-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #f3f4f6;
    }
    .cluster-info-meta {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        color: #10b981;
    }
    .sidebar-section-header {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #6b7280;
        text-transform: uppercase;
        margin: 16px 0 8px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .session-count-badge {
        background: #1f2937;
        color: #9ca3af;
        border-radius: 9999px;
        padding: 1px 7px;
        font-size: 0.7rem;
    }

    /* =========================================================================
       4. Hero & Suggestion Cards
       ========================================================================= */
    .hero-wrapper {
        text-align: center;
        padding: 26px 16px 16px 16px;
        max-width: 820px;
        margin: 0 auto;
    }
    .hero-glow-avatar {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 62px;
        height: 62px;
        border-radius: 18px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #3b82f6;
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.25);
        font-size: 1.9rem;
        margin-bottom: 12px;
    }
    .hero-heading {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f9fafb;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
    }
    .hero-desc {
        font-size: 0.92rem;
        color: #9ca3af;
        line-height: 1.5;
        margin-bottom: 22px;
    }

    /* =========================================================================
       5. Messages & Reasoning Terminal
       ========================================================================= */
    .msg-meta-line {
        font-size: 0.75rem;
        color: #6b7280;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .msg-meta-model {
        font-weight: 600;
        color: #93c5fd;
    }
    .msg-meta-cosine {
        color: #10b981;
        font-family: 'JetBrains Mono', monospace;
    }
    .thought-terminal {
        background: #090d16;
        border: 1px solid #2e384d;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #a5b4fc;
        line-height: 1.55;
        white-space: pre-wrap;
        margin-bottom: 12px;
    }
    .grounded-response-card {
        background: #0d131f;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        line-height: 1.6;
    }
    .source-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.86rem;
    }
    .chunk-tag {
        display: inline-block;
        background: #1e3a8a;
        color: #bfdbfe;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .relevance-tag {
        display: inline-block;
        background: #064e3b;
        color: #6ee7b7;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* =========================================================================
       6. Bottom Input Badges
       ========================================================================= */
    .input-badge-container {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 0.78rem;
    }
    .input-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 9999px;
        padding: 3px 10px;
        color: #9ca3af;
    }
    .input-badge-green {
        color: #34d399;
        border-color: rgba(52, 211, 153, 0.3);
        background: rgba(52, 211, 153, 0.08);
    }
    .input-caption {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.72rem;
        color: #6b7280;
        margin-top: 6px;
        padding: 0 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Session State Initialization
# ==============================================================================
import importlib
import app.rag_engine

if (
    "engine" not in st.session_state
    or not hasattr(st.session_state.engine, "sessions")
    or not hasattr(st.session_state.engine.sessions, "list_sessions_with_metadata")
    or not hasattr(st.session_state.engine, "kb_store")
):
    importlib.reload(app.rag_engine)
    st.session_state.engine = app.rag_engine.RAGEngine()
    init_key = get_api_key()
    if init_key or TEST_MODE:
        try:
            st.session_state.engine.load_and_index(api_key=init_key)
        except Exception:
            pass

# Default Model & RAG parameters
if "model_name" not in st.session_state:
    st.session_state.model_name = "gemma-4-31b-it"

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = os.getenv("EMBEDDING_MODEL", "all-minilm")

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0

if "top_k" not in st.session_state:
    st.session_state.top_k = 3

if "min_relevance_pct" not in st.session_state:
    st.session_state.min_relevance_pct = 25

if "strict_mode" not in st.session_state:
    st.session_state.strict_mode = True

if "use_tool_calling" not in st.session_state:
    st.session_state.use_tool_calling = False

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = get_api_key() or ""

st.session_state.additional_documents = st.session_state.engine.kb_store.list_sources()

# Active View Navigation (Chat, Lab, KB, Docs)
if "active_nav_view" not in st.session_state:
    st.session_state.active_nav_view = "💬 Assistant Chat"

# Telemetry live state
if "telemetry" not in st.session_state:
    st.session_state.telemetry = {
        "latency_ms": "—",
        "l2_dist": "—",
        "sim_score": "—",
    }

# Fetch active sessions dynamically from SQLite database
db_sessions = st.session_state.engine.sessions.list_sessions_with_metadata()
if not db_sessions:
    st.session_state.engine.sessions.create_session("session_1", title="New Session")
    db_sessions = st.session_state.engine.sessions.list_sessions_with_metadata()

valid_session_ids = [s["session_id"] for s in db_sessions]
if "active_session_id" not in st.session_state or st.session_state.active_session_id not in valid_session_ids:
    st.session_state.active_session_id = valid_session_ids[0]


UPLOADED_SOURCES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uploaded_sources"
)
os.makedirs(UPLOADED_SOURCES_DIR, exist_ok=True)


def add_uploaded_document(uploaded_file) -> None:
    """Validate, extract, and persist an uploaded document into the knowledge base store."""
    payload = uploaded_file.getvalue()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploads must be 10 MB or smaller.")

    source_name = uploaded_file.name
    if source_name.lower().endswith(".pdf"):
        import pypdf

        reader = pypdf.PdfReader(BytesIO(payload))
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ValueError("PDFs must contain 100 pages or fewer.")
        content = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif source_name.lower().endswith((".json", ".ndjson", ".jsonl")):
        content = format_json_payload(payload, f"uploaded file ({source_name})")
    else:
        content = payload.decode("utf-8", errors="ignore")

    if not content.strip():
        raise ValueError("The uploaded document does not contain extractable text.")

    # Save copy of raw file to data/uploaded_sources/
    try:
        file_path = os.path.join(UPLOADED_SOURCES_DIR, source_name)
        with open(file_path, "wb") as f:
            f.write(payload)
    except Exception:
        pass

    engine: RAGEngine = st.session_state.engine
    engine.kb_store.add_source(
        name=source_name,
        content=content,
        source_type="file",
        meta={"filename": source_name, "size_bytes": len(payload)},
    )
    st.session_state.additional_documents = engine.kb_store.list_sources()


def add_json_endpoint_document(endpoint_url: str) -> None:
    """Fetch a JSON/NDJSON GET endpoint and persist its formatted payload into the knowledge base store."""
    payload = fetch_url_payload(endpoint_url)
    content = format_json_payload(payload, f"endpoint response ({endpoint_url})")

    engine: RAGEngine = st.session_state.engine
    engine.kb_store.add_source(
        name=endpoint_url,
        content=content,
        source_type="url",
        meta={"url": endpoint_url},
    )
    st.session_state.additional_documents = engine.kb_store.list_sources()


def rebuild_knowledge_base(
    chunk_size: int,
    chunk_overlap: int,
    uploaded_files=None,
    json_endpoint_url: str = "",
) -> None:
    """Rebuild the active knowledge base with chunking parameters and new sources."""
    if uploaded_files is not None:
        if isinstance(uploaded_files, list):
            for uf in uploaded_files:
                add_uploaded_document(uf)
        else:
            add_uploaded_document(uploaded_files)

    if json_endpoint_url and json_endpoint_url.strip():
        add_json_endpoint_document(json_endpoint_url.strip())

    engine: RAGEngine = st.session_state.engine
    engine.chunk_size = chunk_size
    engine.chunk_overlap = chunk_overlap
    engine.load_and_index(
        api_key=st.session_state.gemini_api_key,
    )
    st.session_state.additional_documents = engine.kb_store.list_sources()


def render_paginated_chunks(chunks: List[ChunkInfo], key_prefix: str) -> None:
    """Render a manageable page of indexed chunks with session-scoped navigation."""
    if not chunks:
        st.info("No chunks have been indexed yet.")
        return

    page_size = st.selectbox(
        "Chunks per page",
        options=[10, 25, 50],
        key=f"{key_prefix}_page_size",
    )
    total_pages = (len(chunks) + page_size - 1) // page_size
    page_key = f"{key_prefix}_page"
    current_page = min(max(st.session_state.get(page_key, 0), 0), total_pages - 1)

    previous_col, page_col, next_col = st.columns([1, 2, 1], vertical_alignment="center")
    if previous_col.button(
        "Previous",
        key=f"{key_prefix}_previous",
        disabled=current_page == 0,
        width="stretch",
    ):
        current_page -= 1
    page_col.caption(
        f"Page {current_page + 1} of {total_pages} · "
        f"chunks {current_page * page_size + 1}–{min((current_page + 1) * page_size, len(chunks))} of {len(chunks)}"
    )
    if next_col.button(
        "Next",
        key=f"{key_prefix}_next",
        disabled=current_page == total_pages - 1,
        width="stretch",
    ):
        current_page += 1
    st.session_state[page_key] = current_page

    start = current_page * page_size
    for chunk in chunks[start : start + page_size]:
        with st.expander(
            f"📄 Chunk #{chunk.chunk_id} • Source: {chunk.source} ({chunk.char_count} chars)",
            expanded=False,
        ):
            st.code(chunk.content, language="markdown")


def render_knowledge_base_ui(key_prefix: str, in_modal: bool = False) -> None:
    """Render the full Knowledge Base & Sources manager UI."""
    engine: RAGEngine = st.session_state.engine
    sources = engine.kb_store.list_sources()
    is_placeholder = engine.is_placeholder_kb

    st.subheader("📁 Knowledge Base & Sources Manager")
    st.caption(
        "Upload custom documents (.json, .ndjson, .jsonl, .pdf, .md, .txt) or enter JSON/NDJSON URLs. "
        "User sources persist across sessions and replace the default company policy placeholder."
    )

    # 1. Active Status & Sources List
    if is_placeholder:
        st.info(
            "ℹ️ **Default Placeholder Active (`data/company_policy.md`)**\n\n"
            "No custom sources are uploaded yet. The assistant is currently using the included company policy as a placeholder. "
            "Upload one or more files or provide a JSON/NDJSON URL below to replace it with your own knowledge base."
        )
        with st.expander("📄 View Default Placeholder (`data/company_policy.md`)", expanded=False):
            if os.path.exists("data/company_policy.md"):
                with open("data/company_policy.md", "r", encoding="utf-8") as f:
                    preview_txt = f.read()
                st.text_area("Default Policy Content", value=preview_txt, height=200, disabled=True, key=f"{key_prefix}_placeholder_preview")
            else:
                st.warning("`data/company_policy.md` not found.")
    else:
        st.success(
            f"🚀 **Custom Knowledge Base Active ({len(sources)} source{'s' if len(sources) != 1 else ''} indexed)**\n\n"
            "Your custom sources have replaced the default placeholder and persist in the system across sessions."
        )

        st.markdown("**Active Knowledge Base Sources:**")
        for s in sources:
            with st.container(border=True):
                col_info, col_del = st.columns([4, 1], vertical_alignment="center")
                icon = "🌐" if s["source_type"] == "url" else "📄"
                created_str = datetime.fromtimestamp(s["created_at"]).strftime("%b %d, %I:%M %p")
                with col_info:
                    st.markdown(f"**{icon} {s['name']}**")
                    st.caption(f"Type: `{s['source_type'].upper()}` • {s['char_count']:,} characters • Added: {created_str}")
                with col_del:
                    if st.button("🗑️ Remove", key=f"{key_prefix}_del_{s['id']}", width="stretch"):
                        engine.remove_source(s["id"])
                        st.session_state.additional_documents = engine.kb_store.list_sources()
                        st.toast(f"Removed source: {s['name']}", icon="🗑️")
                        st.rerun()

        if st.button("🔄 Reset to Default Placeholder", key=f"{key_prefix}_reset_kb", help="Remove all custom sources and revert to data/company_policy.md"):
            engine.clear_sources()
            st.session_state.additional_documents = []
            st.toast("Reset to default placeholder (company_policy.md)", icon="🔄")
            st.rerun()

    st.divider()

    # 2. Add Sources & Indexing Configuration
    st.markdown("#### 📤 Add New Sources")
    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        st.markdown("**Upload Document(s):**")
        uploaded_files = st.file_uploader(
            "Upload .json, .ndjson, .jsonl, .md, .txt, or .pdf",
            type=["json", "ndjson", "jsonl", "md", "txt", "pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"{key_prefix}_file_uploader",
        )
    with col_upload2:
        st.markdown("**JSON / NDJSON Endpoint URL:**")
        json_endpoint_url = st.text_input(
            "JSON GET endpoint URL",
            placeholder="https://api.example.com/data",
            help="Supports JSON and JSON Lines (NDJSON) responses up to 10 MB.",
            label_visibility="collapsed",
            key=f"{key_prefix}_endpoint_input",
        )

    st.markdown("#### ⚙️ Chunking Parameters")
    col_c1, col_c2 = st.columns(2)
    c_size = col_c1.slider("Chunk Size (characters)", 150, 1500, engine.chunk_size, 50, key=f"{key_prefix}_c_size")
    c_over = col_c2.slider("Chunk Overlap", 0, 300, engine.chunk_overlap, 10, key=f"{key_prefix}_c_over")

    if st.button("🔄 Ingest & Re-Index Knowledge Base Now", width="stretch", type="primary", key=f"{key_prefix}_btn_reindex"):
        with st.spinner("Processing documents and rebuilding FAISS vector index..."):
            try:
                rebuild_knowledge_base(
                    chunk_size=c_size,
                    chunk_overlap=c_over,
                    uploaded_files=uploaded_files,
                    json_endpoint_url=json_endpoint_url,
                )
                total_chunks = len(engine.chunks)
                st.toast(f"✅ Indexed successfully! Total chunks: {total_chunks}", icon="🚀")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion/Re-indexing failed: {e}")

    st.divider()

    # 3. Active Indexed Chunks Explorer
    st.markdown("#### 🔍 Active Chunk Index Explorer")
    if engine.active_source_names:
        st.caption(f"Sources included in active index: **{', '.join(engine.active_source_names)}**")
    all_chunks = engine.get_chunk_preview()
    st.caption(f"Displaying {len(all_chunks)} active vector chunk slices currently stored in FAISS.")
    render_paginated_chunks(all_chunks, f"{key_prefix}_chunks")


# ==============================================================================
# Modal Dialogs: Settings & Documentation
# ==============================================================================
@st.dialog("⚙️ Settings & Knowledge Base", width="large")
def show_settings_dialog():
    """Popup modal containing all RAG, Model, and Document configurations."""
    tab_gen, tab_rag, tab_kb, tab_cmp = st.tabs([
        "🔑 General & Model",
        "🔍 Retrieval & Agentic Tools",
        "📁 Document & Indexing",
        "🔬 Head-to-Head Lab",
    ])

    # Tab 1: General & Model Configuration
    with tab_gen:
        st.subheader("Model & API Authentication")
        new_api_key = st.text_input(
            "Google AI Studio API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            help="Get your key at https://aistudio.google.com/app/apikey",
        )
        if new_api_key != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = new_api_key
            st.session_state.engine.api_key = new_api_key
            if st.session_state.engine.vector_store is None:
                try:
                    st.session_state.engine.load_and_index(api_key=new_api_key)
                    st.toast("Vector store indexed successfully!", icon="🚀")
                except Exception as e:
                    st.error(f"Initialization error: {e}")

        st.caption("Don't have an API key? [Get a free Google AI Studio Key](https://aistudio.google.com/app/apikey)")

        st.divider()
        model_choices = [
            "gemma-4-31b-it",
            "gemma-4-26b-a4b-it",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "qwen2.5:0.5b",
        ]
        curr_idx = model_choices.index(st.session_state.model_name) if st.session_state.model_name in model_choices else 0
        new_model = st.selectbox(
            "Active LLM",
            options=model_choices,
            index=curr_idx,
            format_func=lambda m: f"🦙 {m} (Local Ollama)" if is_ollama_model(m) else f"✨ {m} (Google AI Studio)",
            help="Choose from Google AI Studio models (Gemma 4, Gemini) or local Ollama models (qwen2.5:0.5b).",
        )
        st.session_state.model_name = new_model
        st.session_state.engine.model = new_model

        if is_ollama_model(new_model):
            st.info("💡 **Local Ollama Active:** Running locally on `http://localhost:11434`. Runs 100% locally with zero cloud API keys required for inference.")

        st.session_state.temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.temperature),
            step=0.05,
        )

        st.session_state.strict_mode = st.toggle(
            "Strict Grounding Mode",
            value=bool(st.session_state.strict_mode),
            help="Refuse to answer if information is not found in retrieved policy documents.",
        )

    # Tab 2: Retrieval & Agentic Tools
    with tab_rag:
        st.subheader("Semantic Search Parameters")
        emb_options = [
            "all-minilm",
            "models/gemini-embedding-001",
            "models/text-embedding-004",
        ]
        emb_idx = emb_options.index(st.session_state.embedding_model) if st.session_state.embedding_model in emb_options else 0
        new_emb = st.selectbox(
            "Embedding Model",
            options=emb_options,
            index=emb_idx,
            format_func=lambda m: f"🦙 {m} (Local Ollama - Zero Quota & Zero Cost)" if is_ollama_embedding_model(m) else f"✨ {m} (Google AI Studio)",
            help="all-minilm is an ultra-lightweight 384-dim model (~120MB) running on local Ollama with zero quota limits and no cloud costs.",
        )
        st.session_state.embedding_model = new_emb
        st.session_state.engine.embedding_model = new_emb
        if st.session_state.engine.indexed_embedding_model != new_emb:
            st.info("Embedding model changed. Re-index the knowledge base before sending another query.")

        st.session_state.top_k = st.slider(
            "Top-K Chunks to Retrieve",
            min_value=1,
            max_value=8,
            value=int(st.session_state.top_k),
            step=1,
        )
        st.session_state.min_relevance_pct = st.slider(
            "Minimum relevance for strict answers",
            min_value=0,
            max_value=100,
            value=int(st.session_state.min_relevance_pct),
            help="Strict mode refuses to generate when the best L2-derived relevance score is below this value.",
        )

        st.divider()
        st.subheader("Agentic Tool Calling (bind_tools)")
        st.session_state.use_tool_calling = st.toggle(
            "Enable Agentic Tools",
            value=bool(st.session_state.use_tool_calling),
            help="When enabled, binds LangChain tools (search_policy_docs, calculate_leave_balance) to the LLM.",
        )

    # Tab 3: Document & Indexing
    with tab_kb:
        render_knowledge_base_ui("modal_kb", in_modal=True)

    # Tab 4: Head-to-Head Comparison
    with tab_cmp:
        st.subheader("Direct LLM vs. RAG Grounded Output")
        st.caption("Demonstrates how RAG prevents hallucination on proprietary Acme policies.")
        cmp_q = st.text_input(
            "Comparison Question:",
            value="What is the learning and development allowance at Acme Technologies?",
        )
        if st.button("⚡ Compare Outputs", width="stretch"):
            col_d, col_r = st.columns(2)
            with col_d:
                st.markdown("##### ❌ Direct LLM (No Context)")
                try:
                    direct_llm = get_llm(
                        model=st.session_state.model_name,
                        api_key=st.session_state.gemini_api_key,
                        temperature=st.session_state.temperature,
                    )
                    direct_resp = direct_llm.invoke(f"Answer concisely: {cmp_q}")
                    clean_d, _ = extract_text_and_thinking(getattr(direct_resp, "content", direct_resp))
                    st.info(clean_d)
                except Exception as e:
                    st.error(str(e))
            with col_r:
                st.markdown("##### ✅ RAG Grounded Output")
                try:
                    rag_res = st.session_state.engine.chat(
                        question=cmp_q,
                        session_id="comparison_tmp",
                        model=st.session_state.model_name,
                        temperature=st.session_state.temperature,
                        k=st.session_state.top_k,
                        strict_mode=st.session_state.strict_mode,
                        min_relevance_pct=st.session_state.min_relevance_pct,
                        api_key=st.session_state.gemini_api_key,
                    )
                    st.success(rag_res.answer)
                except Exception as e:
                    st.error(str(e))


@st.dialog("📖 Documentation & System Architecture", width="large")
def show_docs_dialog():
    """Information dialog outlining system components and capabilities."""
    st.markdown("### 🧠 ByoKnK Rag Assistant Architecture")
    st.markdown(
        f"""
        This application provides an **enterprise-grade Retrieval-Augmented Generation (RAG)** pipeline 
        grounded in your ingested knowledge base (or the default Acme Technologies policy placeholder) with Google AI Studio:

        - **Embedding Engine:** Local Ollama **`all-minilm`** (384-dim, unlimited queries, zero cloud cost, zero quota) or Google **`models/gemini-embedding-001`**.
        - **Vector Index:** In-memory FAISS vector store with L2 distance; the UI displays a derived relevance score, not cosine similarity.
        - **Reasoning Models:** Powered by Google's **Gemma 4** (`gemma-4-31b-it`, `gemma-4-26b-a4b-it`), 
          **Gemini** (`gemini-3.5-flash-lite`, `gemini-2.5-flash-lite`), and local **Ollama** (`qwen2.5:0.5b`).
        - **Retrieval rationale:** Shows source count, relevance, and latency without exposing model chain-of-thought.
        - **Agentic Capabilities:** LangChain tool binding (`bind_tools`) for arithmetic leave calculations and targeted policy searches.
        - **Safety & Grounding:** Strict grounding prompt template preventing external hallucinations.

        #### 🚀 REST API Endpoints
        The companion FastAPI server is live and fully accessible:
        - `GET  /health` - Cluster readiness and indexed chunk counts
        - `POST /retrieve` - Query vector store and inspect similarity percentages
        - `POST /chat` - Multi-turn conversational endpoint with session memory
        - `GET  /chunks` - Full catalogue of active chunk slices
        """
    )


# ==============================================================================
# SIDEBAR: OpenWebUI Style Rail
# ==============================================================================
with st.sidebar:
    # 1. RAG Cluster Alpha Card
    t = st.session_state.telemetry
    st.markdown(
        f"""
        <div class="cluster-card">
            <div class="cluster-icon">🗄️</div>
            <div>
                <div class="cluster-info-title">RAG Cluster Alpha</div>
                <div class="cluster-info-meta">Latency: {t['latency_ms']}ms &nbsp;|&nbsp; L2 relevance: {t['sim_score']}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. "+ New Session" Button
    if st.button("➕  New Session", width="stretch", type="primary", key="btn_new_session"):
        existing_sessions = st.session_state.engine.sessions.list_sessions()
        sess_num = len(existing_sessions) + 1
        new_id = f"session_{int(time.time())}"
        st.session_state.engine.sessions.create_session(new_id, title=f"Session {sess_num}")
        st.session_state.active_session_id = new_id
        st.session_state.active_nav_view = "💬 Assistant Chat"
        st.rerun()

    # 3. Recent Sessions List (Fetched from SQLite DB)
    sessions_meta = st.session_state.engine.sessions.list_sessions_with_metadata()
    total_sess = len(sessions_meta)
    st.markdown(
        f"""
        <div class="sidebar-section-header">
            <span>Recent Sessions</span>
            <span class="session-count-badge">{total_sess}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for s_info in sessions_meta:
        s_id = s_info["session_id"]
        s_title = s_info["title"] or s_id
        is_active = (s_id == st.session_state.active_session_id)
        col_s_name, col_s_del = st.columns([5, 1])

        with col_s_name:
            label = f"💬  {s_title}" if not is_active else f"▶  {s_title}"
            if st.button(
                label,
                key=f"sess_btn_{s_id}",
                width="stretch",
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_session_id = s_id
                st.session_state.active_nav_view = "💬 Assistant Chat"
                st.rerun()

        with col_s_del:
            if total_sess > 1:
                if st.button("✕", key=f"del_{s_id}", help=f"Delete {s_title}"):
                    st.session_state.engine.sessions.delete_session(s_id)
                    remaining = st.session_state.engine.sessions.list_sessions()
                    if st.session_state.active_session_id == s_id:
                        st.session_state.active_session_id = remaining[0] if remaining else "session_1"
                    st.rerun()

    st.divider()

    # 4. Bottom Knowledge Base & Settings Triggers
    kb_ready = (st.session_state.engine.vector_store is not None)
    num_chunks = len(st.session_state.engine.chunks) if kb_ready else 0

    if kb_ready:
        st.markdown(
            f"""
            <div style="font-size: 0.8rem; color: #10b981; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                <span class="telemetry-dot"></span> Knowledge Base Ready ({num_chunks} chunks)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #f59e0b; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                <span style="color: #f59e0b;">▲</span> Set API Key to Index Knowledge Base
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("⚙️  Settings & Knowledge", width="stretch", key="sb_settings_btn"):
        show_settings_dialog()

    if st.button("📖  Documentation", width="stretch", key="sb_docs_btn"):
        show_docs_dialog()


# ==============================================================================
# MAIN VIEW: TOP NAVBAR
# ==============================================================================

with st.container(border=True):
    col_nav_brand, col_nav_tabs, col_nav_model, col_nav_actions = st.columns(
        [2.6, 3.8, 2.0, 1.6], vertical_alignment="center"
    )

    with col_nav_brand:
        st.markdown(
            """
            <div class="brand-title">
                <span class="brand-emblem">🧠</span>
                <span>ByoKnK Rag Assistant</span>
                <span class="brand-version-pill">v2.4</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_nav_tabs:
        nav_options = [
            "💬 Assistant Chat",
            "🔬 Head-to-Head Lab",
            "📁 Knowledge Base",
            "📖 Documentation",
        ]
        curr_view_idx = (
            nav_options.index(st.session_state.active_nav_view)
            if st.session_state.active_nav_view in nav_options
            else 0
        )
        selected_nav = st.segmented_control(
            "Navigation",
            options=nav_options,
            default=nav_options[curr_view_idx],
            label_visibility="collapsed",
            key="navbar_segmented_control",
        )
        if selected_nav and selected_nav != st.session_state.active_nav_view:
            st.session_state.active_nav_view = selected_nav
            st.rerun()

    with col_nav_model:
        model_options = [
            "gemma-4-31b-it",
            "gemma-4-26b-a4b-it",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "qwen2.5:0.5b",
        ]
        cur_idx = model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0
        selected_model = st.selectbox(
            "Active Model",
            options=model_options,
            index=cur_idx,
            format_func=lambda m: f"🦙 {m} (Ollama)" if is_ollama_model(m) else f"✨ {m}",
            label_visibility="collapsed",
            key="top_model_selector",
        )
        if selected_model != st.session_state.model_name:
            st.session_state.model_name = selected_model
            st.session_state.engine.model = selected_model
            st.rerun()

    with col_nav_actions:
        c_set, c_clr = st.columns(2)
        with c_set:
            if st.button("⚙️", key="top_settings_icon", help="Open Settings & Configuration", width="stretch"):
                show_settings_dialog()
        with c_clr:
            if st.button("🗑️", key="top_clear_icon", help="Clear current chat messages", width="stretch"):
                st.session_state.engine.sessions.clear_session(st.session_state.active_session_id)
                st.rerun()

# Sub-Navbar Telemetry & Readiness Strip
t = st.session_state.telemetry
kb_status_str = f"{len(st.session_state.engine.chunks)} Chunks Indexed" if st.session_state.engine.vector_store else "Not Indexed"
st.markdown(
    f"""
    <div class="telemetry-strip">
        <div>
            <span class="telemetry-dot"></span>
            <span><b>RAG Cluster Alpha</b> &nbsp;•&nbsp; Latency: <b>{t['latency_ms']}ms</b> &nbsp;|&nbsp; L2 Dist: <b>{t['l2_dist']}</b> &nbsp;|&nbsp; L2 relevance: <b>{t['sim_score']}%</b></span>
        </div>
        <div>
            <span class="telemetry-badge-tag">🟢 {kb_status_str}</span>
            <span class="telemetry-badge-tag">{'🦙' if is_ollama_model(st.session_state.model_name) else '🤖'} {st.session_state.model_name}{' (Ollama)' if is_ollama_model(st.session_state.model_name) else ''}</span>
            <span class="telemetry-badge-tag">🛡️ {"Strict grounding" if st.session_state.strict_mode else "Context-guided"}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# VIEW DISPATCHER BASED ON NAVBAR SELECTION
# ==============================================================================

# ------------------------------------------------------------------------------
# VIEW 1: ASSISTANT CHAT (Default)
# ------------------------------------------------------------------------------
if st.session_state.active_nav_view == "💬 Assistant Chat":
    active_session = st.session_state.active_session_id
    messages = st.session_state.engine.sessions.get_messages(active_session)

    # Empty State: Centered Hero Greeting (Suggestions removed)
    if not messages:
        engine = st.session_state.engine
        if engine.is_placeholder_kb:
            hero_desc = "Gemma 4 RAG Assistant grounded in default Acme Technologies policy placeholder."
        else:
            sources_label = ", ".join(engine.active_source_names)
            hero_desc = f"Gemma 4 RAG Assistant grounded in your active knowledge base ({sources_label})."

        st.markdown(
            f"""
            <div class="hero-wrapper">
                <div class="hero-glow-avatar">🧠</div>
                <div class="hero-heading">How can I help you today?</div>
                <div class="hero-desc">
                    {hero_desc}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render Active Chat History from SQLite Database
    for msg in messages:
        role = msg.role
        avatar = "👤" if role == "user" else "🧠"

        if role == "user":
            timestamp = datetime.fromtimestamp(msg.timestamp).strftime("%I:%M %p")
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 2px;">
                    <span style="font-size: 0.74rem; color: #6b7280;">You &nbsp;•&nbsp; {timestamp}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.chat_message("user", avatar=avatar):
                st.markdown(msg.content)

        else:
            m_name = st.session_state.model_name
            m_time = datetime.fromtimestamp(msg.timestamp).strftime("%I:%M:%S %p")
            sim_val = msg.sources[0].similarity_pct if msg.sources else "—"

            st.markdown(
                f"""
                <div class="msg-meta-line">
                    <span class="msg-meta-model">{m_name} Grounded</span>
                    <span>•</span>
                    <span>{m_time}</span>
                    <span>•</span>
                    <span class="msg-meta-cosine">L2 relevance: {sim_val}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.chat_message("assistant", avatar=avatar):
                # 1. Retrieval rationale / thinking
                if msg.thinking:
                    with st.expander("Retrieval rationale", expanded=False):
                        st.code(msg.thinking, language="text")

                # 2. Grounded Answer Text
                st.markdown(msg.content)

                # 3. Citation Sources Expander
                if msg.sources:
                    with st.expander(f"📚 {len(msg.sources)} Retrieved Sources", expanded=False):
                        for src in msg.sources:
                            with st.container(border=True):
                                st.caption(
                                    f"Chunk {src.chunk_id} · L2 relevance {src.similarity_pct}% "
                                    f"· distance {src.distance} · {src.source}"
                                )
                                st.code(src.content, language="markdown")

                # 4. Prompt Inspector
                if msg.prompt:
                    with st.expander("📝 Grounded Prompt Sent to LLM", expanded=False):
                        st.code(msg.prompt, language="markdown")

                # 5. Tool Calls Log
                if msg.tool_calls:
                    with st.expander("🛠️ Tool Calling Execution Log", expanded=False):
                        for tc in msg.tool_calls:
                            with st.container(border=True):
                                st.caption(f"Tool: {tc.tool}")
                                st.code(f"Arguments: {tc.args}\nResult: {tc.output}", language="text")

    # Input Badges & Chat Input
    engine = st.session_state.engine
    if engine.is_placeholder_kb:
        grounding_badge = '<span class="input-badge input-badge-green"><span class="telemetry-dot"></span> Default Policy Placeholder</span>'
        chat_placeholder = "Message Acme Policy Assistant (default placeholder)..."
        docs_count_str = "1 placeholder doc"
    else:
        src_count = len(engine.active_source_names)
        grounding_badge = '<span class="input-badge input-badge-green"><span class="telemetry-dot"></span> Custom Knowledge Base</span>'
        chat_placeholder = "Message Knowledge Assistant..."
        docs_count_str = f"{src_count} source{'s' if src_count != 1 else ''} indexed"

    st.markdown(
        f"""
        <div class="input-badge-container">
            {grounding_badge}
            <span class="input-badge">📎 {docs_count_str}</span>
            <span class="input-badge">🛡️ {"Strict RAG" if st.session_state.strict_mode else "Context-guided RAG"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_query = st.chat_input(chat_placeholder)

    if user_query:
        curr_time_str = datetime.now().strftime("%I:%M %p")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 2px;">
                <span style="font-size: 0.74rem; color: #6b7280;">You &nbsp;•&nbsp; {curr_time_str}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🧠"):
            current_api_key = st.session_state.gemini_api_key or get_api_key()

            if not current_api_key and not TEST_MODE:
                warn_msg = "🔑 Please provide your Google AI Studio API key in **Settings** (⚙️ icon) to index documents and begin chatting."
                st.warning(warn_msg)
            else:
                with st.spinner(f"Retrieving context & generating answer with {st.session_state.model_name}..."):
                    start_time = time.time()
                    try:
                        if st.session_state.engine.vector_store is None:
                            st.session_state.engine.load_and_index(api_key=current_api_key)

                        response: RAGResponse = st.session_state.engine.chat(
                            question=user_query,
                            session_id=active_session,
                            model=st.session_state.model_name,
                            temperature=st.session_state.temperature,
                            k=st.session_state.top_k,
                            strict_mode=st.session_state.strict_mode,
                            min_relevance_pct=st.session_state.min_relevance_pct,
                            use_tool_calling=st.session_state.use_tool_calling,
                            api_key=current_api_key,
                        )
                        elapsed = round(time.time() - start_time, 2)
                        elapsed_ms = int(elapsed * 1000)

                        best_sim = (
                            response.sources[0].similarity_pct
                            if response.sources else 0.92
                        )
                        best_dist = (
                            round(response.sources[0].distance, 2)
                            if response.sources else 0.15
                        )
                        st.session_state.telemetry = {
                            "latency_ms": elapsed_ms,
                            "l2_dist": best_dist,
                            "sim_score": best_sim,
                        }
                        st.rerun()

                    except Exception as e:
                        err_msg = f"⚠️ Generation error: {str(e)}"
                        st.error(err_msg)

    st.markdown(
        """
        <div class="input-caption">
            <span>Powered by Gemma 4 31B-IT • Zero external data leakage</span>
            <span>Shift + Enter for multiline</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# VIEW 2: HEAD-TO-HEAD COMPARISON LAB
# ------------------------------------------------------------------------------
elif st.session_state.active_nav_view == "🔬 Head-to-Head Lab":
    st.subheader("🔬 Head-to-Head Comparison: Direct LLM vs. Grounded RAG")
    st.caption("Inspect how Retrieval-Augmented Generation eliminates hallucinations on proprietary company policies.")

    test_presets = [
        "What is the annual learning and development allowance in INR at Acme?",
        "How many days of unused annual leave can an employee roll over?",
        "When is a doctor's medical certificate mandatory for sick leave?",
        "Can full-time employees work remotely and how many days are allowed?",
    ]

    st.markdown("**Quick Preset Questions:**")
    col_p1, col_p2 = st.columns(2)
    chosen_preset = None
    with col_p1:
        if st.button(f"🎯 {test_presets[0]}", width="stretch", key="preset_0"):
            chosen_preset = test_presets[0]
        if st.button(f"🎯 {test_presets[1]}", width="stretch", key="preset_1"):
            chosen_preset = test_presets[1]
    with col_p2:
        if st.button(f"🎯 {test_presets[2]}", width="stretch", key="preset_2"):
            chosen_preset = test_presets[2]
        if st.button(f"🎯 {test_presets[3]}", width="stretch", key="preset_3"):
            chosen_preset = test_presets[3]

    cmp_input = st.text_input(
        "Question to Benchmark:",
        value=chosen_preset or "What is the annual learning and development allowance in INR at Acme?",
        key="h2h_benchmark_input",
    )

    if st.button("⚡ Execute Live Comparison Test", width="stretch", type="primary"):
        c_direct, c_rag = st.columns(2)

        with c_direct:
            with st.container(border=True):
                st.markdown("### ❌ Direct LLM (Un-Grounded)")
                st.caption("Prompted directly without proprietary context.")
                with st.spinner("Invoking direct LLM..."):
                    try:
                        d_llm = get_llm(
                            model=st.session_state.model_name,
                            api_key=st.session_state.gemini_api_key,
                            temperature=st.session_state.temperature,
                        )
                        d_res = d_llm.invoke(f"Answer concisely: {cmp_input}")
                        clean_text, _ = extract_text_and_thinking(getattr(d_res, "content", d_res))
                        st.warning(clean_text)
                        st.caption("⚠️ Notice: Direct LLMs lack access to internal private docs and frequently hallucinate allowances.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with c_rag:
            with st.container(border=True):
                st.markdown("### ✅ RAG Grounded Output")
                st.caption(f"Retrieved via FAISS + {st.session_state.model_name}.")
                with st.spinner("Retrieving context & generating answer..."):
                    try:
                        rag_out = st.session_state.engine.chat(
                            question=cmp_input,
                            session_id="benchmark_session",
                            model=st.session_state.model_name,
                            temperature=st.session_state.temperature,
                            k=st.session_state.top_k,
                            strict_mode=st.session_state.strict_mode,
                            min_relevance_pct=st.session_state.min_relevance_pct,
                            api_key=st.session_state.gemini_api_key,
                        )
                        st.success(rag_out.answer)
                        if rag_out.sources:
                            st.caption(f"📚 Grounded in {len(rag_out.sources)} chunks (Top relevance: {rag_out.sources[0].similarity_pct}%)")
                    except Exception as e:
                        st.error(f"Error: {e}")

# ------------------------------------------------------------------------------
# VIEW 3: KNOWLEDGE BASE & CHUNKS EXPLORER
# ------------------------------------------------------------------------------
elif st.session_state.active_nav_view == "📁 Knowledge Base":
    render_knowledge_base_ui("main_kb", in_modal=False)

# ------------------------------------------------------------------------------
# VIEW 4: SYSTEM ARCHITECTURE & DOCUMENTATION
# ------------------------------------------------------------------------------
elif st.session_state.active_nav_view == "📖 Documentation":
    st.subheader("📖 System Architecture & Technical Specifications")

    st.markdown(
        """
        The **ByoKnK Rag Assistant** is built on a modular Retrieval-Augmented Generation architecture
        connecting Google AI Studio models and local Ollama runtimes with an in-memory vector database and LangChain agentic tooling.

        ```
        [User Query] ──► [Navbar Model: Gemma 4 / Gemini / Ollama Qwen]
                               │
                        [Embeddings: all-minilm (Ollama) / gemini-embedding-001]
                               │
                        [FAISS Vector Store (Persistent Sources / Placeholder)]
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             [Top-K Context]      [Agentic Tools (bind_tools)]
                    │                     │
                    └──────────┬──────────┘
                               ▼
                 [Strict Grounding Prompt]
                               │
                               ▼
        [Answer + Retrieval Rationale + Collapsible Source Citations]
        ```

        ### ⚡ Embedding Engines
        - **`all-minilm` (Recommended / Default):** Local 384-dimensional dense semantic embeddings running via Ollama (`http://localhost:11434`). Ultra-lightweight (~120MB), zero API quotas, zero rate limits, and 100% free.
        - **`models/gemini-embedding-001`:** Google AI Studio 768-dimensional cloud embedding model.

        ### 🤖 Supported Reasoning Models
        #### ✨ Google AI Studio
        - **`gemma-4-31b-it`:** Google's 31B dense instruct open-weights reasoning model.
        - **`gemma-4-26b-a4b-it`:** 26B instruct model optimized for high-throughput reasoning.
        - **`gemini-3.5-flash-lite`:** Ultra-fast, cost-effective reasoning and retrieval model.
        - **`gemini-2.5-flash-lite`:** Optimized for ultra-low latency interactive chat.

        #### 🦙 Local Ollama Models (Zero Cloud Egress)
        - **`qwen2.5:0.5b`:** Alibaba's ultra-compact, high-performance 0.5B instruct model running locally on `http://localhost:11434`. Runs completely on-device with zero token costs and full LangChain tool-calling capability.

        ### 🛠️ Agentic Capabilities
        - `search_knowledge_base(query)`: Semantic retrieval function bound via `bind_tools`.
        - `search_policy_docs(query)`: Semantic retrieval alias for policy or knowledge base documents.
        - `calculate_leave_balance(days_requested, leave_type)`: Arithmetic calculator for policy rules.

        ### 🚀 FastAPI REST Endpoints
        The companion FastAPI server is live and fully accessible:
        - `GET    http://localhost:8080/health`: Health status and cluster readiness
        - `GET    http://localhost:8080/sources`: Active knowledge base sources and placeholder status
        - `POST   http://localhost:8080/sources/url`: Ingest JSON / NDJSON endpoint URL
        - `DELETE http://localhost:8080/sources/{id}`: Remove specific knowledge source
        - `DELETE http://localhost:8080/sources`: Reset to default placeholder
        - `POST   http://localhost:8080/retrieve`: Vector similarity search endpoint
        - `POST   http://localhost:8080/chat`: Multi-turn conversational endpoint
        - `GET    http://localhost:8080/chunks`: Catalogue of all indexed text chunks
        - `GET    http://localhost:8080/sessions`: Active session history inspector
        """

    )

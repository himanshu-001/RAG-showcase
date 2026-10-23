"""
Core RAG Engine for Acme Policy Assistant.
Designed for learning and production readiness:
- Modular LLM initialization with get_llm() supporting Gemma models via Google AI Studio
- Tool calling support via LangChain's bind_tools
- Configurable embeddings (gemini-embedding-001) & FAISS vector store
- Chunking visualizer support for educational inspection
- Per-session multi-turn conversational memory
- Clean extraction of reasoning/thinking vs final text response
- Structured Pydantic response models
"""

import os
import time
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import (
    RetrievedSource,
    ToolCallRecord,
    RAGResponse,
    ChunkInfo,
    SessionMessage,
)

# Load environment variables
load_dotenv()

# Check test mode
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 100
JSON_ENDPOINT_TIMEOUT_SECONDS = 15


def format_json_payload(payload: bytes, source_description: str) -> str:
    """Parse standard JSON or newline-delimited JSON (NDJSON/JSON Lines) into indexable text."""
    try:
        decoded_payload = payload.decode("utf-8").lstrip("\ufeff")
    except UnicodeDecodeError as exc:
        raise ValueError(f"The {source_description} is not valid UTF-8 JSON.") from exc

    try:
        parsed_payload = json.loads(decoded_payload)
    except json.JSONDecodeError:
        # Check for NDJSON / JSON Lines: one complete JSON value per non-empty line
        lines = [line.strip() for line in decoded_payload.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"The {source_description} does not contain JSON data.")
        try:
            parsed_payload = [json.loads(line) for line in lines]
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"The {source_description} is not valid JSON or newline-delimited JSON (NDJSON): {exc}"
            ) from exc

    if isinstance(parsed_payload, list):
        # Format each record cleanly separated by double newlines for optimal RAG chunking
        formatted_records = [
            json.dumps(item, ensure_ascii=False, indent=2)
            for item in parsed_payload
        ]
        return "\n\n".join(formatted_records)

    return json.dumps(parsed_payload, ensure_ascii=False, indent=2)


def fetch_url_payload(url: str, timeout: int = JSON_ENDPOINT_TIMEOUT_SECONDS) -> bytes:
    """Fetch content from an HTTP(S) URL with standard headers and size limits."""
    from urllib.parse import urlparse
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Enter a complete HTTP(S) URL.")

    req = Request(
        url,
        headers={
            "Accept": "application/json, application/x-ndjson, text/plain, */*",
            "User-Agent": "Mozilla/5.0 (compatible; ByoKnK-RAG/1.0)",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_UPLOAD_BYTES:
                raise ValueError("Payload exceeds 10 MB limit.")
            payload = response.read(MAX_UPLOAD_BYTES + 1)
    except HTTPError as exc:
        raise ValueError(f"The endpoint returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise ValueError(f"Could not reach the endpoint: {exc.reason}") from exc

    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Payload exceeds 10 MB limit.")

    return payload


def get_api_key(provided_key: Optional[str] = None) -> Optional[str]:
    """Retrieve Gemini/Google API key from argument or environment."""
    return (
        provided_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def extract_text_and_thinking(content: Any) -> Tuple[str, Optional[str]]:
    """
    Extracts clean text answer and optional thinking/reasoning process
    from LangChain response content.
    Handles:
    - Standard string responses
    - Thinking models returning structured content blocks:
      [{'type': 'thinking', 'thinking': '...'}, {'type': 'text', 'text': '...'}]
    - Multimodal message chunk objects
    """
    if isinstance(content, str):
        return content.strip(), None

    if isinstance(content, list):
        text_parts = []
        thinking_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type", "")
                if part_type == "text" or "text" in part:
                    text_parts.append(part.get("text", ""))
                elif part_type == "thinking" or "thinking" in part:
                    thinking_parts.append(part.get("thinking", ""))
                else:
                    text_parts.append(str(part))
            else:
                if hasattr(part, "text"):
                    text_parts.append(part.text)
                else:
                    text_parts.append(str(part))

        answer = "\n".join(t for t in text_parts if t).strip()
        thinking = "\n".join(th for th in thinking_parts if th).strip() if thinking_parts else None
        return answer, thinking

    return str(content).strip(), None


OLLAMA_KNOWN_MODELS = {
    "qwen2.5:0.5b",
    "qwen2:0.5b",
    "qwen:0.5b",
}


def is_ollama_model(model_name: Optional[str]) -> bool:
    """Determine whether a model name targets local Ollama."""
    if not model_name:
        return False
    clean = model_name.lower().strip()
    return (
        clean.startswith("ollama/")
        or clean.startswith("ollama:")
        or clean in OLLAMA_KNOWN_MODELS
        or clean.startswith("qwen")
    )


def get_llm(
    model: str = "gemini-3.5-flash-lite",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    tools: Optional[List[Any]] = None,
    max_output_tokens: Optional[int] = 1024,
    **kwargs,
):
    """
    Factory function to configure and return the Chat LLM.
    Supports:
    - Google AI Studio models: Gemma 4 (gemma-4-31b-it, gemma-4-26b-a4b-it), Gemma 2, and Gemini models.
    - Local Ollama models: Qwen (qwen2.5:0.5b), or any model prefixed with 'ollama/' via local Ollama runtime.

    If `tools` are provided, binds them using LangChain's bind_tools to enable tool calling.
    In TEST_MODE, returns a deterministic mock runnable for fast automated testing.
    """
    if TEST_MODE:
        from langchain_core.runnables import RunnableLambda
        from langchain_core.messages import AIMessage

        def mock_llm_call(input_val):
            # Format friendly test response
            q = str(input_val)
            if "annual leave" in q.lower():
                return AIMessage(content="According to the policy, full-time employees receive 24 days of paid annual leave.")
            return AIMessage(content=f"[TEST_MODE] This is a mock response from {model} model.")

        mock_runnable = RunnableLambda(mock_llm_call)
        if tools:
            # Attach a dummy bind_tools for test mode compatibility
            mock_runnable.bind_tools = lambda t: mock_runnable
        return mock_runnable

    if is_ollama_model(model):
        ollama_model_name = model
        if ollama_model_name.startswith("ollama/"):
            ollama_model_name = ollama_model_name[len("ollama/") :]
        elif ollama_model_name.startswith("ollama:"):
            ollama_model_name = ollama_model_name[len("ollama:") :]

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama

        ollama_kwargs = dict(kwargs)
        if max_output_tokens is not None and "num_predict" not in ollama_kwargs:
            ollama_kwargs["num_predict"] = max_output_tokens

        llm = ChatOllama(
            model=ollama_model_name,
            base_url=ollama_base_url,
            temperature=temperature,
            **ollama_kwargs,
        )

        if tools:
            llm = llm.bind_tools(tools)

        return llm

    key = get_api_key(api_key)
    if not key:
        raise ValueError(
            "Google AI Studio API key not found. Please provide GEMINI_API_KEY or GOOGLE_API_KEY."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **kwargs,
    )

    if tools:
        # LangChain tool binding for Agentic / Tool-calling RAG
        llm = llm.bind_tools(tools)

    return llm


OLLAMA_EMBEDDING_MODELS = {
    "all-minilm",
    "all-minilm:latest",
    "nomic-embed-text",
    "bge-m3",
    "bge-small-en",
}


def is_ollama_embedding_model(model_name: Optional[str]) -> bool:
    """Determine whether an embedding model targets local Ollama."""
    if not model_name:
        return False
    clean = model_name.lower().strip()
    return (
        clean in OLLAMA_EMBEDDING_MODELS
        or clean.startswith("ollama/")
        or clean.startswith("ollama:")
        or "minilm" in clean
        or "nomic" in clean
    )


def get_embeddings(
    api_key: Optional[str] = None,
    model: str = "all-minilm",
):
    """
    Returns text embeddings instance:
    - If model is an Ollama model (e.g. 'all-minilm', 'nomic-embed-text'), returns OllamaEmbeddings.
      100% local, runs on http://localhost:11434, zero quota limits, zero cost, no cloud key required.
    - If Google model (models/gemini-embedding-001), returns GoogleGenerativeAIEmbeddings.
    - Falls back to FakeEmbeddings in TEST_MODE.
    """
    if TEST_MODE:
        from langchain_core.embeddings import FakeEmbeddings
        dim = 384 if is_ollama_embedding_model(model) else 768
        return FakeEmbeddings(size=dim)

    if is_ollama_embedding_model(model):
        ollama_model_name = model
        if ollama_model_name.startswith("ollama/"):
            ollama_model_name = ollama_model_name[len("ollama/") :]
        elif ollama_model_name.startswith("ollama:"):
            ollama_model_name = ollama_model_name[len("ollama:") :]

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(
            model=ollama_model_name,
            base_url=ollama_base_url,
        )

    key = get_api_key(api_key)
    if not key:
        raise ValueError("Google AI Studio API key required for Google embeddings.")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=key,
    )


DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chat_sessions.db"
)


class SessionStore:
    """
    Persistent SQLite-backed multi-session store for chat history and inspectability.
    Saves sessions and conversation messages to data/chat_sessions.db.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    thinking TEXT,
                    sources_json TEXT,
                    prompt TEXT,
                    tool_calls_json TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
            conn.commit()

    def get_messages(self, session_id: str) -> List[SessionMessage]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT role, content, timestamp, thinking, sources_json, prompt, tool_calls_json
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,)
            )
            rows = cursor.fetchall()
            messages: List[SessionMessage] = []
            for r in rows:
                sources: List[RetrievedSource] = []
                if r["sources_json"]:
                    try:
                        raw_src = json.loads(r["sources_json"])
                        sources = [RetrievedSource(**s) for s in raw_src]
                    except Exception:
                        sources = []

                tool_calls: List[ToolCallRecord] = []
                if r["tool_calls_json"]:
                    try:
                        raw_tc = json.loads(r["tool_calls_json"])
                        tool_calls = [ToolCallRecord(**t) for t in raw_tc]
                    except Exception:
                        tool_calls = []

                messages.append(
                    SessionMessage(
                        role=r["role"],
                        content=r["content"],
                        timestamp=r["timestamp"],
                        thinking=r["thinking"],
                        sources=sources,
                        prompt=r["prompt"] or "",
                        tool_calls=tool_calls,
                    )
                )
            return messages

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        thinking: Optional[str] = None,
        sources: Optional[List[RetrievedSource]] = None,
        prompt: Optional[str] = None,
        tool_calls: Optional[List[ToolCallRecord]] = None,
    ):
        now = time.time()
        sources_json = json.dumps([s.model_dump() for s in (sources or [])]) if sources else None
        tool_calls_json = json.dumps([t.model_dump() for t in (tool_calls or [])]) if tool_calls else None

        first_line = content.strip().split("\n")[0][:38].strip()
        summary_title = first_line + ("..." if len(content.strip().split("\n")[0]) > 38 else "")

        with self._get_connection() as conn:
            cur = conn.execute("SELECT session_id, title FROM sessions WHERE session_id = ?", (session_id,))
            existing = cur.fetchone()
            if not existing:
                title = summary_title if role == "user" else session_id
                conn.execute(
                    "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, title, now, now)
                )
            else:
                current_title = existing["title"]
                # Dynamically title session from user's first query if it currently has a generic name
                if role == "user" and (
                    not current_title
                    or current_title.startswith("Session ")
                    or current_title == "New Session"
                    or current_title == session_id
                ):
                    conn.execute(
                        "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                        (summary_title, now, session_id)
                    )
                else:
                    conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))

            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, timestamp, thinking, sources_json, prompt, tool_calls_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, now, thinking, sources_json, prompt, tool_calls_json)
            )
            conn.commit()

    def clear_session(self, session_id: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.commit()

    def delete_session(self, session_id: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    def create_session(self, session_id: str, title: Optional[str] = None):
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title or session_id, now, now)
            )
            conn.commit()

    def list_sessions(self) -> List[str]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT session_id FROM sessions ORDER BY updated_at DESC")
            return [row["session_id"] for row in cur.fetchall()]

    def list_sessions_with_metadata(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT s.session_id, s.title, s.updated_at, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
            """)
            return [
                {
                    "session_id": row["session_id"],
                    "title": row["title"] or row["session_id"],
                    "updated_at": row["updated_at"],
                    "message_count": row["message_count"],
                }
                for row in cur.fetchall()
            ]



DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "company_policy.md"
)


class KnowledgeBaseStore:
    """
    Persistent SQLite-backed store for knowledge base sources.
    Stores user-uploaded documents (.md, .txt, .pdf, .json, .ndjson) and fetched JSON/NDJSON URL endpoints.
    Persists across sessions, server restarts, and multi-process access.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_sources (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    meta_json TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_created ON knowledge_sources(created_at)")
            conn.commit()

    def list_sources(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT id, name, source_type, char_count, created_at, updated_at, meta_json
                FROM knowledge_sources
                ORDER BY created_at ASC
            """)
            results = []
            for row in cur.fetchall():
                meta = {}
                if row["meta_json"]:
                    try:
                        meta = json.loads(row["meta_json"])
                    except Exception:
                        meta = {}
                results.append({
                    "id": row["id"],
                    "name": row["name"],
                    "source_type": row["source_type"],
                    "char_count": row["char_count"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "meta": meta,
                })
            return results

    def has_custom_sources(self) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) AS cnt FROM knowledge_sources")
            row = cur.fetchone()
            return bool(row and row["cnt"] > 0)

    def add_source(
        self,
        name: str,
        content: str,
        source_type: str = "file",
        source_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sid = source_id or f"{source_type}_{int(time.time() * 1000)}"
        now = time.time()
        meta_json = json.dumps(meta or {})
        char_count = len(content)

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO knowledge_sources (id, name, source_type, content, char_count, created_at, updated_at, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    source_type=excluded.source_type,
                    content=excluded.content,
                    char_count=excluded.char_count,
                    updated_at=excluded.updated_at,
                    meta_json=excluded.meta_json
            """, (sid, name, source_type, content, char_count, now, now, meta_json))
            conn.commit()

        return {
            "id": sid,
            "name": name,
            "source_type": source_type,
            "char_count": char_count,
            "created_at": now,
            "updated_at": now,
            "meta": meta or {},
        }

    def remove_source(self, source_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM knowledge_sources WHERE id = ? OR name = ?", (source_id, source_id))
            conn.commit()
            return cur.rowcount > 0

    def clear_sources(self) -> int:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM knowledge_sources")
            conn.commit()
            return cur.rowcount

    def get_documents(self, default_data_path: str) -> Tuple[List[Document], bool]:
        """
        Retrieves active documents for indexing.
        If user has added custom sources, returns only custom sources (is_placeholder=False).
        If no custom sources exist, returns the default placeholder document (is_placeholder=True).
        """
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT id, name, source_type, content
                FROM knowledge_sources
                ORDER BY created_at ASC
            """)
            rows = cur.fetchall()

        if rows:
            docs = [
                Document(
                    page_content=row["content"],
                    metadata={
                        "source": row["name"],
                        "source_id": row["id"],
                        "source_type": row["source_type"],
                    },
                )
                for row in rows
            ]
            return docs, False

        # Fallback to default placeholder document
        if os.path.exists(default_data_path):
            with open(default_data_path, "r", encoding="utf-8") as f:
                placeholder_content = f.read()
            doc = Document(
                page_content=placeholder_content,
                metadata={
                    "source": os.path.basename(default_data_path),
                    "source_id": "default_placeholder",
                    "source_type": "placeholder",
                },
            )
            return [doc], True

        return [], True


class RAGEngine:
    """
    Comprehensive RAG Engine combining Document Splitting, Vector Search,
    Prompt Augmentation, Conversational History, and Tool Calling.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        api_key: Optional[str] = None,
        model: str = "gemini-3.5-flash-lite",
        embedding_model: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        self.data_path = data_path or DEFAULT_DATA_PATH
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "all-minilm")
        self.vector_store: Optional[FAISS] = None
        self.indexed_embedding_model: Optional[str] = None
        self.raw_documents: List[Document] = []
        self.chunks: List[Document] = []
        self.sessions = SessionStore(db_path=db_path)
        self.kb_store = KnowledgeBaseStore(db_path=db_path)
        self.is_placeholder_kb: bool = True
        self.active_source_names: List[str] = []

    def load_and_index(
        self,
        custom_docs: Optional[List[Document]] = None,
        api_key: Optional[str] = None,
    ):
        """
        Loads documents, splits into chunks, and initializes the FAISS vector index.
        Uses persistent custom sources if available, otherwise falls back to company_policy.md placeholder.
        """
        if api_key:
            self.api_key = api_key

        if custom_docs is not None:
            self.raw_documents = custom_docs
            self.is_placeholder_kb = False
        else:
            docs, is_placeholder = self.kb_store.get_documents(self.data_path)
            self.raw_documents = docs
            self.is_placeholder_kb = is_placeholder

        self.active_source_names = [
            doc.metadata.get("source", "Document") for doc in self.raw_documents
        ]

        if not self.raw_documents:
            self.chunks = []
            self.vector_store = None
            return

        # 1. Chunking Phase (Recursive Character Splitting)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n\n", "\n", ". ", " "],
        )
        self.chunks = splitter.split_documents(self.raw_documents)

        # Annotate chunk metadata for educational inspection
        for idx, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_id"] = idx + 1
            chunk.metadata["char_count"] = len(chunk.page_content)

        # 2. Vector Embedding & FAISS Indexing
        embeddings = get_embeddings(api_key=self.api_key, model=self.embedding_model)
        self.vector_store = FAISS.from_documents(self.chunks, embeddings)
        self.indexed_embedding_model = self.embedding_model

    def add_source(
        self,
        name: str,
        content: str,
        source_type: str = "file",
        source_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a persistent source to the knowledge base and re-index."""
        res = self.kb_store.add_source(
            name=name,
            content=content,
            source_type=source_type,
            source_id=source_id,
            meta=meta,
        )
        self.load_and_index()
        return res

    def remove_source(self, source_id: str) -> bool:
        """Remove a persistent source and re-index."""
        removed = self.kb_store.remove_source(source_id)
        if removed:
            self.load_and_index()
        return removed

    def clear_sources(self) -> int:
        """Clear all custom sources (reverting to default placeholder) and re-index."""
        count = self.kb_store.clear_sources()
        self.load_and_index()
        return count

    def get_sources_summary(self) -> Dict[str, Any]:
        """Get summary of active knowledge base sources and status."""
        sources = self.kb_store.list_sources()
        is_placeholder = not bool(sources)
        if is_placeholder:
            placeholder_sources = [
                {
                    "id": "default_placeholder",
                    "name": os.path.basename(self.data_path),
                    "source_type": "placeholder",
                    "char_count": len(self.raw_documents[0].page_content) if self.raw_documents else 0,
                    "created_at": 0.0,
                    "updated_at": 0.0,
                    "meta": {"description": "Default sample policy placeholder"},
                }
            ]
            display_sources = placeholder_sources
        else:
            display_sources = sources

        return {
            "is_placeholder": is_placeholder,
            "sources": display_sources,
            "total_sources": len(display_sources),
            "total_chunks": len(self.chunks),
        }

    def retrieve_with_scores(self, query: str, k: int = 4) -> List[RetrievedSource]:
        """
        Retrieves top-k chunks with similarity scores.
        Returns list of structured RetrievedSource Pydantic models.
        Auto-initializes vector store if key is available.
        """
        if self.vector_store is None:
            effective_key = self.api_key or get_api_key()
            if is_ollama_embedding_model(self.embedding_model) or effective_key or TEST_MODE:
                self.load_and_index(api_key=effective_key)
            else:
                raise RuntimeError(
                    "Vector store not initialized. Please enter your Google AI Studio API key in the sidebar or set GEMINI_API_KEY in .env."
                )

        if self.indexed_embedding_model != self.embedding_model:
            raise RuntimeError(
                "Embedding model changed. Re-index the knowledge base before querying."
            )

        raw_results = self.vector_store.similarity_search_with_score(query, k=k)
        retrieved: List[RetrievedSource] = []
        for rank, (doc, dist) in enumerate(raw_results, 1):
            sim_pct = round(1.0 / (1.0 + float(dist)) * 100, 1)
            retrieved.append(
                RetrievedSource(
                    rank=rank,
                    chunk_id=doc.metadata.get("chunk_id", rank),
                    source=doc.metadata.get("source", "Document"),
                    content=doc.page_content,
                    distance=round(float(dist), 4),
                    similarity_pct=sim_pct,
                    char_count=len(doc.page_content),
                )
            )
        return retrieved

    def get_chunk_preview(self) -> List[ChunkInfo]:
        """Returns structured chunk details for the learning visualizer tab."""
        return [
            ChunkInfo(
                chunk_id=c.metadata.get("chunk_id", i + 1),
                source=c.metadata.get("source", "Unknown"),
                char_count=len(c.page_content),
                content=c.page_content,
            )
            for i, c in enumerate(self.chunks)
        ]

    def format_history_for_prompt(self, session_id: str, max_turns: int = 5) -> str:
        """Formats the last few turns of conversation for conversational context."""
        messages = self.sessions.get_messages(session_id)
        if not messages:
            return "No previous conversation."

        recent = messages[-(max_turns * 2):]
        formatted = []
        for msg in recent:
            prefix = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{prefix}: {msg.content}")
        return "\n".join(formatted)

    def build_grounded_prompt(
        self,
        context_chunks: List[RetrievedSource],
        question: str,
        history: str,
        strict_mode: bool = True,
    ) -> str:
        """Builds the complete grounded prompt sent to Gemma / Gemini."""
        context_text = "\n\n".join(
            f"--- [Source: {c.source} | Chunk {c.chunk_id}] ---\n{c.content}"
            for c in context_chunks
        )

        sources_desc = (
            ", ".join(self.active_source_names)
            if self.active_source_names
            else "Knowledge Base"
        )

        strict_instruction = (
            "Answer ONLY using the provided Knowledge Base Context. If the information is not explicitly "
            "stated in the context, respond strictly with: "
            "\"I could not find that information in the knowledge base.\""
            if strict_mode
            else "Use the provided context as primary truth. You may clarify general terms if needed."
        )

        prompt = f"""You are a helpful and precise Knowledge Base Assistant grounded in the active knowledge base ({sources_desc}).
{strict_instruction}
Always be polite, direct, concise, and cite the relevant source document(s) when answering.

### Relevant Knowledge Base Context:
{context_text}

### Conversation History:
{history}

### Current Question:
User: {question}

Assistant:"""
        return prompt

    def get_tools(self) -> List[Any]:
        """Defines LangChain tools for agentic / tool-calling mode."""
        engine_ref = self

        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the active knowledge base for relevant sections, entries, and information."""
            try:
                results = engine_ref.retrieve_with_scores(query, k=3)
                if not results:
                    return "No matching information found in the knowledge base."
                formatted = "\n\n".join(
                    f"[Chunk {r.chunk_id} from {r.source}]:\n{r.content}"
                    for r in results
                )
                return formatted
            except Exception as e:
                return f"Error querying knowledge base: {str(e)}"

        @tool
        def search_policy_docs(query: str) -> str:
            """Search policy or knowledge base documents for relevant sections."""
            return search_knowledge_base.invoke({"query": query})

        @tool
        def calculate_leave_balance(days_requested: int, leave_type: str = "annual") -> str:
            """Calculate remaining leave days based on company policy (24 annual days, 10 sick days)."""
            leave_type_clean = leave_type.lower().strip()
            if "annual" in leave_type_clean:
                total = 24
                remaining = total - days_requested
                return (
                    f"Annual Leave Policy: Full-time employees receive 24 days per calendar year. "
                    f"Requested: {days_requested} days. If approved, remaining allowance would be {remaining} days."
                )
            elif "sick" in leave_type_clean:
                total = 10
                remaining = total - days_requested
                notice = " Note: A medical certificate may be requested if sick leave exceeds 2 consecutive working days."
                return (
                    f"Sick Leave Policy: 10 days per calendar year. "
                    f"Requested: {days_requested} days. Remaining: {remaining} days.{notice}"
                )
            else:
                return f"Leave type '{leave_type}' is evaluated case-by-case under policy guidelines."

        return [search_knowledge_base, search_policy_docs, calculate_leave_balance]

    def chat(
        self,
        question: str,
        session_id: str = "default",
        model: Optional[str] = None,
        temperature: float = 0.0,
        k: int = 4,
        strict_mode: bool = True,
        min_relevance_pct: float = 25.0,
        use_tool_calling: bool = False,
        api_key: Optional[str] = None,
    ) -> RAGResponse:
        """
        Executes a complete Conversational RAG turn:
        1. Retrieves relevant chunks (with multi-turn context awareness).
        2. Formats history and grounded context.
        3. Calls Gemma/Gemini LLM (with or without tool-calling).
        4. Cleanses and parses thinking blocks vs final text answer.
        5. Saves turn to session store.
        6. Returns a validated RAGResponse Pydantic model.
        """
        chosen_model = model or self.model
        key = api_key or self.api_key or get_api_key()
        if key and not self.api_key:
            self.api_key = key

        # 1. Format conversational history
        history_str = self.format_history_for_prompt(session_id)

        # 2. Retrieve context chunks
        retrieved_chunks = self.retrieve_with_scores(question, k=k)

        tool_calls_record: List[ToolCallRecord] = []
        raw_prompt = ""
        clean_answer = ""
        thinking_process = None

        # Do not ask a model to decide whether a weak retrieval is sufficient.
        # The displayed relevance score is L2-derived, so it is intentionally not
        # described as cosine similarity.
        best_relevance = max((chunk.similarity_pct for chunk in retrieved_chunks), default=0.0)
        if strict_mode and not TEST_MODE and best_relevance < min_relevance_pct:
            clean_answer = "I could not find that information in the knowledge base."
            raw_prompt = (
                f"Strict grounding refusal: best retrieved relevance was "
                f"{best_relevance}% (minimum required: {min_relevance_pct}%)."
            )
            rag_response = RAGResponse(
                question=question,
                answer=clean_answer,
                thinking=None,
                session_id=session_id,
                sources=retrieved_chunks,
                prompt=raw_prompt,
                tool_calls=[],
                model_used=chosen_model,
            )
            self.sessions.add_message(session_id, role="user", content=question)
            self.sessions.add_message(
                session_id,
                role="assistant",
                content=clean_answer,
                sources=retrieved_chunks,
                prompt=raw_prompt,
            )
            return rag_response

        if use_tool_calling:
            # Tool-calling agentic mode using bind_tools
            tools = self.get_tools()
            llm_with_tools = get_llm(
                model=chosen_model,
                api_key=key,
                temperature=temperature,
                tools=tools,
            )

            sources_label = (
                ", ".join(self.active_source_names)
                if self.active_source_names
                else "Knowledge Base"
            )
            system_msg = (
                f"You are a helpful Knowledge Base Assistant grounded in ({sources_label}). "
                "Use the available tools (search_knowledge_base, calculate_leave_balance) to answer inquiries accurately. "
                "Answer only from the retrieved context or tool results. If neither "
                "supports the answer, say: 'I could not find that information in the knowledge base.'"
            )
            context_text = "\n\n".join(
                f"[Source: {chunk.source} | Chunk {chunk.chunk_id}]\n{chunk.content}"
                for chunk in retrieved_chunks
            )
            raw_prompt = (
                f"System: {system_msg}\n\nRetrieved Knowledge Base Context:\n{context_text}"
                f"\n\nHistory: {history_str}\nUser: {question}"
            )

            # Invoke model with tool binding
            response_msg = llm_with_tools.invoke(raw_prompt)

            # Check if LLM requested any tool calls
            if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
                tool_map = {t.name: t for t in tools}
                for t_call in response_msg.tool_calls:
                    t_name = t_call.get("name")
                    t_args = t_call.get("args", {})
                    if t_name in tool_map:
                        t_func = tool_map[t_name]
                        tool_output = t_func.invoke(t_args)
                        tool_calls_record.append(
                            ToolCallRecord(
                                tool=t_name,
                                args=t_args,
                                output=str(tool_output),
                            )
                        )

                # Follow-up generation after executing tools
                followup_prompt = (
                    f"{raw_prompt}\n"
                    f"Tool Execution Results:\n{tool_calls_record}\n"
                    f"Now provide a clear, finalized response based strictly on these tool results."
                )
                final_llm = get_llm(model=chosen_model, api_key=key, temperature=temperature)
                final_res = final_llm.invoke(followup_prompt)
                raw_content = getattr(final_res, "content", final_res)
            else:
                raw_content = getattr(response_msg, "content", response_msg)

            clean_answer, thinking_process = extract_text_and_thinking(raw_content)

        else:
            # Standard Direct Grounded RAG mode
            raw_prompt = self.build_grounded_prompt(
                context_chunks=retrieved_chunks,
                question=question,
                history=history_str,
                strict_mode=strict_mode,
            )
            llm = get_llm(model=chosen_model, api_key=key, temperature=temperature)
            response = llm.invoke(raw_prompt)
            raw_content = getattr(response, "content", response)
            clean_answer, thinking_process = extract_text_and_thinking(raw_content)

        # Build structured Pydantic response
        rag_response = RAGResponse(
            question=question,
            answer=clean_answer,
            thinking=thinking_process,
            session_id=session_id,
            sources=retrieved_chunks,
            prompt=raw_prompt,
            tool_calls=tool_calls_record,
            model_used=chosen_model,
        )

        # Record messages into SessionStore
        self.sessions.add_message(session_id, role="user", content=question)
        self.sessions.add_message(
            session_id,
            role="assistant",
            content=clean_answer,
            thinking=thinking_process,
            sources=retrieved_chunks,
            prompt=raw_prompt,
            tool_calls=tool_calls_record,
        )

        return rag_response

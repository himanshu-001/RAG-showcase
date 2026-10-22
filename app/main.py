import os
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query

from app.rag_engine import (
    RAGEngine,
    TEST_MODE,
    get_api_key,
    format_json_payload,
    fetch_url_payload,
)
from app.models import (
    ChatRequest,
    RAGResponse,
    RetrieveRequest,
    RetrieveResponse,
    ChunksResponse,
    HealthResponse,
    SessionHistoryResponse,
    SessionsListResponse,
    KnowledgeSourceInfo,
    KnowledgeSourcesResponse,
    AddUrlSourceRequest,
)

# Global RAG Engine instance
rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine

    if TEST_MODE:
        print("Running in TEST_MODE - initializing RAG engine with mock models.")
        rag_engine = RAGEngine()
        rag_engine.load_and_index()
        yield
        return

    api_key = get_api_key()
    if not api_key:
        print("WARNING: GEMINI_API_KEY or GOOGLE_API_KEY is not configured. Setting RAG engine in pending mode.")
        rag_engine = RAGEngine()
        yield
        return

    try:
        print("Initializing RAG Engine with Google AI Studio / Gemma 4 / Ollama models...")
        rag_engine = RAGEngine()
        rag_engine.load_and_index()
        print(f"RAG Engine successfully initialized! Indexed {len(rag_engine.chunks)} chunks.")
    except Exception as exc:
        print(f"WARNING: Initial index failed ({exc}). RAG engine starting in pending mode.")

    yield


app = FastAPI(
    title="Gemma 4 & Gemini Showcase RAG Assistant (with Ollama)",
    description="Educational RAG showcasing document chunking, FAISS vector retrieval, Google Gemma 4/Gemini models, local Ollama (Qwen 0.5B), tool-calling, and per-session conversational history.",
    version="2.2.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "Gemma 4 & Gemini Showcase RAG Assistant (with Ollama)",
        "version": "2.2.0",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "POST /chat",
            "retrieve": "POST /retrieve",
            "chunks": "GET /chunks",
            "sessions": "GET /sessions",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health():
    ready = rag_engine is not None and rag_engine.vector_store is not None
    return HealthResponse(
        status="healthy",
        rag_ready=ready,
        test_mode=TEST_MODE,
        indexed_chunks=len(rag_engine.chunks) if ready else 0,
    )


@app.get("/ready")
def ready():
    return {"status": "ready"}


@app.post("/chat", response_model=RAGResponse)
def chat(request: ChatRequest):
    # Auto-initialize if vector store is pending but key is available
    if rag_engine is not None and rag_engine.vector_store is None:
        key = get_api_key()
        if key or TEST_MODE:
            try:
                rag_engine.load_and_index(api_key=key)
            except Exception as e:
                print(f"Auto-index failed: {e}")

    if rag_engine is None or rag_engine.vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not ready. Ensure GEMINI_API_KEY is configured and engine is indexed.",
        )

    try:
        response = rag_engine.chat(
            question=request.question,
            session_id=request.session_id or "default",
            model=request.model or "gemma-4-31b-it",
            temperature=request.temperature or 0.0,
            k=request.k or 4,
            strict_mode=request.strict_mode if request.strict_mode is not None else True,
            min_relevance_pct=(
                request.min_relevance_pct
                if request.min_relevance_pct is not None
                else 25.0
            ),
            use_tool_calling=request.use_tool_calling or False,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest):
    # Auto-initialize if vector store is pending but key is available
    if rag_engine is not None and rag_engine.vector_store is None:
        key = get_api_key()
        if key or TEST_MODE:
            try:
                rag_engine.load_and_index(api_key=key)
            except Exception as e:
                print(f"Auto-index failed: {e}")

    if rag_engine is None or rag_engine.vector_store is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready")
    try:
        results = rag_engine.retrieve_with_scores(request.query, k=request.k or 4)
        return RetrieveResponse(query=request.query, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks", response_model=ChunksResponse)
def get_chunks():
    if rag_engine is None:
        return ChunksResponse(chunks=[], total=0)
    preview = rag_engine.get_chunk_preview()
    return ChunksResponse(chunks=preview, total=len(preview))


@app.get("/sessions", response_model=SessionsListResponse)
def list_sessions():
    if rag_engine is None:
        return SessionsListResponse(sessions=[])
    return SessionsListResponse(sessions=rag_engine.sessions.list_sessions())


@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(session_id: str):
    if rag_engine is None:
        return SessionHistoryResponse(session_id=session_id, messages=[])
    messages = rag_engine.sessions.get_messages(session_id)
    return SessionHistoryResponse(session_id=session_id, messages=messages)


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    if rag_engine is None:
        return {"status": "cleared"}
    rag_engine.sessions.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/sources", response_model=KnowledgeSourcesResponse)
def get_sources():
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    summary = rag_engine.get_sources_summary()
    return KnowledgeSourcesResponse(
        is_placeholder=summary["is_placeholder"],
        sources=[KnowledgeSourceInfo(**s) for s in summary["sources"]],
        total_sources=summary["total_sources"],
        total_chunks=summary["total_chunks"],
    )


@app.post("/sources/url", response_model=KnowledgeSourcesResponse)
def add_url_source(req: AddUrlSourceRequest):
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        payload = fetch_url_payload(req.url)
        content = format_json_payload(payload, f"URL endpoint ({req.url})")
        rag_engine.add_source(
            name=req.url,
            content=content,
            source_type="url",
            meta={"url": req.url},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    summary = rag_engine.get_sources_summary()
    return KnowledgeSourcesResponse(
        is_placeholder=summary["is_placeholder"],
        sources=[KnowledgeSourceInfo(**s) for s in summary["sources"]],
        total_sources=summary["total_sources"],
        total_chunks=summary["total_chunks"],
    )


@app.delete("/sources/{source_id}", response_model=KnowledgeSourcesResponse)
def delete_source(source_id: str):
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    rag_engine.remove_source(source_id)
    summary = rag_engine.get_sources_summary()
    return KnowledgeSourcesResponse(
        is_placeholder=summary["is_placeholder"],
        sources=[KnowledgeSourceInfo(**s) for s in summary["sources"]],
        total_sources=summary["total_sources"],
        total_chunks=summary["total_chunks"],
    )


@app.delete("/sources", response_model=KnowledgeSourcesResponse)
def reset_sources():
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    rag_engine.clear_sources()
    summary = rag_engine.get_sources_summary()
    return KnowledgeSourcesResponse(
        is_placeholder=summary["is_placeholder"],
        sources=[KnowledgeSourceInfo(**s) for s in summary["sources"]],
        total_sources=summary["total_sources"],
        total_chunks=summary["total_chunks"],
    )

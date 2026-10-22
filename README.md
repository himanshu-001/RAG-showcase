# 🧠 Showcase RAG Learning Project: Gemma 4, Gemini & Ollama Assistant

A complete, educational **Retrieval-Augmented Generation (RAG)** showcase built with Google's flagship **Gemma 4** (`gemma-4-31b-it`, `gemma-4-26b-a4b-it`) and **Gemini** models alongside local **Ollama** models (`qwen2.5:0.5b`). 

This repository demonstrates the entire lifecycle of a production-style RAG pipeline, complete with a **Streamlit Interactive Learning UI**, a **FastAPI backend**, per-session chat history, inspectable chunk retrieval, and LangChain tool calling (`bind_tools`).

---

## 🌟 Key Features

1. **Multi-Model Support (Google AI Studio & Local Ollama)**:
   - **Google AI Studio**: Access Google's state-of-the-art open-weights Gemma 4 models (`gemma-4-31b-it`, `gemma-4-26b-a4b-it`) alongside Gemini 3.5 (`gemini-3.5-flash-lite`), Gemini 2.5 (`gemini-2.5-flash-lite`), and Gemini 2.0. Powered by `gemini-embedding-001` high-dimensional embedding model.
   - **Local Ollama Integration**: Native, zero-cloud-egress support for `qwen2.5:0.5b` (and any local Ollama model) running on `http://localhost:11434`. Runs completely locally without requiring cloud API keys for LLM generation, with full LangChain function tool-calling support.
   - Dedicated `get_llm()` factory function for seamless multi-provider switching and parameter tuning.

2. **Per-Session Conversational Chat & History**:
   - Multi-turn conversation awareness with dedicated sessions (`Session 1`, `Session 2`, etc.).
   - Follow-up context handling: follow-up questions retain context from earlier turns.
   - Switch between conversations or start new ones at any time.

3. **Inspectable Educational RAG Pipeline**:
   - **🔍 Retrieved Chunk Inspection**: View exact chunks retrieved for each query, with L2 distance and similarity percentage.
   - **📝 Grounded Prompt Viewer**: Inspect the exact prompt template and context sent to the LLM.
   - **🔬 RAG Learning Lab**: Interactive visualizer for document chunking, vector similarity scoring, and head-to-head comparison (*Direct LLM vs. RAG Grounded LLM*).

4. **Agentic Tool Calling via `bind_tools`**:
   - Toggle between standard grounded RAG and agentic tool-calling RAG across both cloud and local models.
   - In tool-calling mode, the LLM dynamically invokes tools like `search_policy_docs` and `calculate_leave_balance`.

5. **Streamlit UI + FastAPI Backend**:
   - **Streamlit**: Beautiful visual playground for exploring and testing with quick model-switching pill.
   - **FastAPI**: REST endpoints for health checks, retrieval inspection, and session chat.

---

## 📐 How RAG Works

```
┌────────────────┐     1. Ingestion & Split     ┌────────────────────────┐
│ Document (.md) │ ───────────────────────────> │ Recursive Splitter     │
└────────────────┘                              │ (Chunk Size & Overlap) │
                                                └───────────┬────────────┘
                                                            │ 2. Embed
                                                            v
┌────────────────┐     3. Query Embedding       ┌────────────────────────┐
│ User Query     │ ───────────────────────────> │ FAISS Vector Store     │
└───────┬────────┘                              └───────────┬────────────┘
        │                                                   │ 4. Top-K Chunks
        │                                                   v
        │         5. Augmented Prompt           ┌────────────────────────┐
        └─────────────────────────────────────> │ Gemma / Gemini / Ollama│
          (Context + History + Question)        │ (Qwen 0.5B) LLM        │
                                                └───────────┬────────────┘
                                                            │ 6. Response
                                                            v
                                                ┌────────────────────────┐
                                                │ Grounded Answer        │
                                                └────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites & Virtual Environment

Using `uv` (recommended) or standard `pip`:

```bash
# Clone the repository and navigate into the directory
cd simple_rag/Deployment_LIVE

# Activate existing virtual environment
source .venv/bin/activate

# Or install dependencies
uv pip install -r requirements.txt
```

### 2. Configure Your Google AI Studio API Key

Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

Set it in your `.env` file or export it:
```bash
cp .env .env.local
export GEMINI_API_KEY="your-google-ai-studio-key-here"
```
*(Alternatively, enter it directly into the Streamlit settings dialog).*

### 3. Local Ollama Setup (Qwen 2.5 0.5B)

For completely private, local inference with zero cloud costs:
1. Ensure the Ollama daemon is running:
   ```bash
   ollama serve
   ```
2. Pull the lightweight chat model and quota-free embedding model:
   ```bash
   ollama pull qwen2.5:0.5b
   ollama pull all-minilm
   ```
3. Quick verification with `curl`:
   ```bash
   curl -s http://localhost:11434/api/generate -d '{"model": "qwen2.5:0.5b", "prompt": "Hi", "stream": false}'
   curl -s http://localhost:11434/api/embed -d '{"model": "all-minilm", "input": "Hello world"}'
   ```
4. **Selecting the Models**:
   - In the **Streamlit UI**, choose `🦙 qwen2.5:0.5b (Ollama)` from the navbar and `🦙 all-minilm` in the Settings modal (`⚙️`).
   - In the **FastAPI REST API**, send `{"model": "qwen2.5:0.5b"}` in the `POST /chat` body.

---

## 🖥️ Running the Application

### Option A: Launch the Streamlit Interactive UI

```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

**What you can do in Streamlit:**
- **Tab 1 (Conversational RAG)**: Chat with the Acme policy assistant. Inspect retrieved chunks, similarity percentages, full augmented prompts, and tool calls for every message.
- **Tab 2 (Learning Lab)**: Explore document chunks, run vector search simulations, and test Direct LLM vs. RAG side-by-side.
- **Tab 3 (Knowledge Base Manager)**: Upload your own `.txt`, `.md`, `.pdf`, or `.json` file, or index a JSON/JSON Lines GET endpoint; then adjust chunk sizes and overlap and re-index on the fly.

---

### Option B: Run the FastAPI Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```
Interactive API documentation available at `http://localhost:8080/docs`.

**Key Endpoints:**
- `GET /health`: Pipeline readiness and chunk count.
- `POST /chat`: Multi-turn conversational chat with session ID and source citations.
- `POST /retrieve`: Raw vector similarity retrieval for testing queries.
- `GET /chunks`: All chunk metadata in the active FAISS index.
- `GET /sessions`: Active chat sessions list.

---

## 🧪 Running Automated Tests

Run the test suite in offline `TEST_MODE`:
```bash
python -c "import test.test_api as t; t.test_health(); t.test_root(); t.test_retrieve(); t.test_chat_session(); print('All tests passed successfully!')"
```

---

## 📁 Repository Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI REST service
│   ├── rag_engine.py        # Core RAG engine: get_llm(), bind_tools, FAISS, session memory
│   └── streamlit_app.py     # Streamlit UI with learning lab & per-session chat
├── data/
│   └── company_policy.md    # Acme Technologies sample policy knowledge base
├── test/
│   └── test_api.py          # Automated test suite
├── .env                     # Environment variables & API keys
├── Dockerfile               # Production Docker container configuration
└── requirements.txt         # Pinned python dependencies
```

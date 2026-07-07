

https://github.com/user-attachments/assets/6021c852-1dca-426f-8b86-1d276f134542


# 📖 Quran Companion — Arabic-Grounded RAG Agent

A retrieval-augmented ReAct agent that answers questions about the Quran and Sahih al-Bukhari hadiths **using only retrieved, citable source text** — never from the model's own memory. Built with LangGraph, hybrid dense+sparse retrieval, and served via FastAPI with a themed chat frontend.

> Every answer is grounded in a verse (`surah:ayah`) or hadith reference — not generated freely.

---

## ✨ Key Features

### 🕌 Grounded Quran & Hadith Retrieval
- **Hybrid retrieval (dense + sparse)** — combines a FAISS vector index (`BAAI/bge-m3` embeddings) with a BM25 keyword retriever via an `EnsembleRetriever` (weighted 40% dense / 60% sparse) for robust matching across semantic and exact-term queries.
- **Arabic-aware preprocessing** — a custom `normalize_arabic()` step strips diacritics (tashkeel), tatweel, and unifies letter variants (أ/إ/آ → ا, ى/ي → ي, etc.) so queries match text regardless of orthographic variation.
- **Smart verse chunking** — a sliding-window chunker (5 ayahs, 1 overlap) groups verses per surah with proper `surah:ayah` reference tracking, then a de-duplication pass drops near-identical overlapping windows before reranking.
- **LLM-based reranking** — retrieved candidates are re-ranked by a dedicated LLM call (`RERANK_MODEL`) against the original question before the top 5 are handed to the answer model.
- **Hadith search tool** — keyword/fallback search over the Sahih al-Bukhari (English) collection, auto-downloaded and cached on first run.
- **Answer prompt enforces citation discipline** — the agent is instructed to answer *only* from retrieved verses and cite `[surah:ayah]` for every claim, explicitly refusing to guess when context is insufficient.
- **Automatic religious disclaimer** — any response that used `quran_search` or `hadith_search` gets an appended note clarifying this is an AI-assisted retrieval tool, not a scholarly ruling.

### 🤖 Agentic Orchestration (LangGraph)
- **ReAct-style tool-calling agent** built on `langgraph`, with a `StateGraph` that loops between an LLM node and a `ToolNode` until the model decides it has enough information to answer.
- **Six tools available to the agent**:
  | Tool | Purpose |
  |---|---|
  | `quran_search` | Hybrid-retrieval + reranked Quran verse search |
  | `hadith_search` | Sahih al-Bukhari hadith lookup |
  | `wikipedia` | General background/historical context |
  | `arxiv` | Research paper lookup |
  | `tavily_search` | Current events / general web search (primary) |
  | `ddg_search` | Fallback web search if Tavily is unavailable |
- **Strict system prompt guardrails** — the agent is instructed to never answer Quran- or hadith-related questions from its own knowledge without calling the relevant tool first.
- **Per-thread conversational memory** via LangGraph's `MemorySaver` checkpointer — pass the same `thread_id` to continue a conversation (in-memory only; resets on restart).
- **Automatic retry with backoff** — both the main agent LLM and the reranker LLM wrap calls in `with_retry(wait_exponential_jitter=True, stop_after_attempt=5)` to smooth over transient Groq rate limits/errors.

### ⚡ Swappable, Fast LLMs via Groq
- All three LLM roles (`AGENT_MODEL`, `RERANK_MODEL`, `ANSWER_MODEL`) are environment-configurable in `config.py` — swap models without touching pipeline code.
- Runs on [Groq](https://groq.com)'s LPU inference for low-latency responses, with multilingual models (e.g. Qwen3) selected for strong Arabic comprehension.

### 🚀 Production-Ready API & Ops
- **FastAPI backend** with a single `/chat` endpoint (message in, cited answer + tool trace + thread ID out) and a `/health` check.
- **Build-once startup** — the entire pipeline (embeddings, FAISS, BM25, hadith data, tool bindings, graph compilation) is built once at process startup and cached via `lru_cache`, not rebuilt per request.
- **Pre-built artifact caching** — `chunks_cache.pkl`, FAISS index files, and `sparse_retriever_cache.pkl` let the container skip expensive CSV parsing/embedding/BM25-building on every start.
- **Dockerized** — a slim `python:3.11` image that bakes in app code and pre-built `data/` artifacts for reproducible deployment.
- **LangSmith tracing built in** — every agent run (tool calls, rerank step, final answer) is traced for debugging when `LANGCHAIN_TRACING_V2=true`.
- **Tunable retrieval knobs** — `RETRIEVER_K`, `MAX_RERANK_CANDIDATES`, and `RERANK_TRUNCATE_CHARS` let you trade off recall, latency, and Groq token usage.

### 💬 Themed Chat Frontend
- Single-page, dependency-free HTML/CSS/JS chat UI (served as a static file by FastAPI) with:
  - Arabic-inspired dark theme (gold/sage/slate accents, Amiri + Inter typefaces)
  - Suggested starter prompts for empty state
  - Visual tool-source badges (Quran verse / Hadith / other) on each response
  - Persistent thread ID via `localStorage` so conversations survive page reloads
  - Animated "thinking" indicator and graceful error states

---

## 🏗️ Architecture

```
User query
    │
    ▼
FastAPI /chat ──► LangGraph agent (tool_calling_llm ⇄ tools loop)
                        │
        ┌───────────────┼───────────────────────┬─────────────┐
        ▼               ▼                       ▼             ▼
  quran_search     hadith_search           wikipedia /     tavily_search /
  (hybrid FAISS+   (Bukhari keyword         arxiv           ddg_search
   BM25 + rerank)   search)                (context)        (current events)
        │
        ▼
  Reranked, cited verses ──► Answer LLM ──► Final response (+ disclaimer if religious)
```

---

## 📦 Project Structure

```
app/
├── __init__.py
├── config.py        # env vars, model choices, retrieval tuning
├── main.py           # FastAPI app: /chat, /health, static frontend
├── pipeline.py        # RAG build steps, tools, LangGraph agent
└── static/
    └── index.html     # chat frontend
data/                  # pre-built artifacts (see setup below)
Dockerfile
langgraph.json
requirements.txt / pyproject.toml
RAG.ipynb              # original notebook the pipeline was built from
```

---

## 🛠️ Setup

### 1. Populate `data/`
Copy these into `data/` (built from `RAG.ipynb`):

| File | Required? |
|---|---|
| `Quran_RAG_Ready.csv` | Only if `chunks_cache.pkl` doesn't exist yet |
| `chunks_cache.pkl` | Recommended — skips CSV parsing/chunking on startup |
| `index.faiss` + `index.pkl` | Required — the FAISS vector index |
| `sparse_retriever_cache.pkl` | Recommended — skips rebuilding BM25 on startup |
| `hadith_bukhari_eng.json` | Optional — auto-downloaded on first run if absent |

### 2. Configure environment
```bash
cp .env.example .env
```
Fill in:
```
GROQ_API_KEY=...
TAVILY_API_KEY=...
LANGCHAIN_API_KEY=...        # optional, for LangSmith tracing
AGENT_MODEL=qwen/qwen3-32b
RERANK_MODEL=openai/gpt-oss-20b
ANSWER_MODEL=qwen/qwen3-32b
```

### 3. Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 4. Run with Docker
```bash
docker build -t quran-rag-agent .
docker run --env-file .env -p 8000:8000 quran-rag-agent
```

### 5. Call the API
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does the Quran say about patience during hardship?"}'
```
```json
{
  "answer": "...",
  "thread_id": "generated-or-provided-uuid",
  "tool_calls": ["quran_search"]
}
```
Pass the same `thread_id` back to continue the conversation.

---

## ⚙️ Configuration Reference

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `data` | Location of pre-built artifacts |
| `GROQ_API_KEY` | — | Groq API key for all LLM calls |
| `TAVILY_API_KEY` | — | Tavily web search API key |
| `LANGCHAIN_TRACING_V2` | `true` | Toggle LangSmith tracing |
| `AGENT_MODEL` | `qwen/qwen3-32b` | Main tool-calling agent model |
| `RERANK_MODEL` | `openai/gpt-oss-20b` | Reranks retrieved verse candidates |
| `ANSWER_MODEL` | `qwen/qwen3-32b` | Generates the final cited answer |
| `RETRIEVER_K` | `8` | Top-k results per retriever (dense & sparse) |
| `MAX_RERANK_CANDIDATES` | `12` | Max deduped candidates sent to reranker |
| `RERANK_TRUNCATE_CHARS` | `200` | Chars per candidate shown to reranker |

---

## ⚠️ Notes & Limitations
- `MemorySaver` is in-memory only — conversation history resets if the container restarts. Swap in a persistent LangGraph checkpointer (e.g. Postgres) for durable memory.
- Watch Groq free-tier TPM limits under load — `RETRIEVER_K` and `MAX_RERANK_CANDIDATES` are your main levers if you hit 429s.
- This tool provides AI-assisted retrieval, not scholarly rulings — always consult a qualified scholar for religious guidance.

---

## 🧰 Tech Stack
`FastAPI` · `LangGraph` · `LangChain` (+ `langchain-groq`, `langchain-huggingface`) · `Groq` (LPU inference) · `FAISS` · `BM25` (`rank-bm25`) · `sentence-transformers` (`BAAI/bge-m3`) · `Tavily` / `DuckDuckGo` / `Wikipedia` / `Arxiv` tools · `Docker` · `LangSmith`

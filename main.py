import uuid
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from . import config  # noqa: F401  (sets env vars / LangSmith tracing on import)
from .pipeline import get_graph, DISCLAIMER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quran-rag-agent")

app = FastAPI(title="Quran RAG Agent", version="1.0.0")

# Allows the frontend to be opened from a different origin/port too (e.g. a
# separate dev server) -- not required when served from this same app, but
# harmless to leave on for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_RELIGIOUS_TOOLS = {"quran_search", "hadith_search"}


@app.on_event("startup")
def startup_event():
    # Building the graph loads the embedding model, FAISS index, BM25 index,
    # hadith collection, and binds all tools -- do this once here, not per-request.
    logger.info("Building pipeline (embeddings, FAISS, BM25, tools, graph)...")
    get_graph()
    logger.info("Pipeline ready.")


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None  # pass the same thread_id back to continue a conversation


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    tool_calls: list[str]


@app.get("/")
def frontend():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    graph = get_graph()
    thread_id = req.thread_id or str(uuid.uuid4())
    config_dict = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config_dict,
    )

    messages = result["messages"]

    tool_calls_used = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            tool_calls_used.extend(call["name"] for call in m.tool_calls)

    final_answer = messages[-1].content

    if any(name in _RELIGIOUS_TOOLS for name in tool_calls_used):
        final_answer = final_answer + DISCLAIMER

    return ChatResponse(
        answer=final_answer,
        thread_id=thread_id,
        tool_calls=tool_calls_used,
    )

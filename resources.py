"""Process-wide resources shared by every browser session.

The agent (embedding model, FAISS indexes, hadith data, compiled graph) loads ONCE per server start,
in a background thread, while `get_status()` reports its progress to the UI.
"""
import threading
import time

import streamlit as st


@st.cache_resource(show_spinner=False)
def get_status() -> dict:
    return {"stage": "Starting", "fraction": 0.0, "detail": "", "ready": False, "error": None}


@st.cache_resource(show_spinner=False)
def get_saver():
    """Conversation memory (SQLite checkpointer in data/), shared by the agent and the sidebar."""
    from quran_agent.memory import get_checkpointer
    return get_checkpointer()


@st.cache_resource(show_spinner=False)
def get_agent():
    from quran_agent.graph import load_agent

    status = get_status()

    def report(stage: str, fraction: float, detail: str = ""):
        status.update(stage=stage, fraction=fraction, detail=detail)

    try:
        agent = load_agent(progress=report, checkpointer=get_saver())
    except Exception as e:
        status["error"] = f"{type(e).__name__}: {e}"
        raise
    status.update(stage="Ready", fraction=1.0, detail="", ready=True, error=None)
    return agent


def _load_in_background():
    start = time.time()
    try:
        get_agent()
        print(f"Agent ready in {time.time() - start:.0f}s", flush=True)
    except Exception as e:   # the error is kept in get_status() and shown by the UI
        print(f"Agent failed to load: {e}", flush=True)


@st.cache_resource(show_spinner=False)
def start_loading() -> threading.Thread:
    """Start loading the agent in a background thread (once per server process)."""
    thread = threading.Thread(target=_load_in_background, name="agent-loader", daemon=True)
    thread.start()
    return thread


def retry_loading():
    get_status().update(stage="Starting", fraction=0.0, detail="", error=None)
    start_loading.clear()
    start_loading()


@st.cache_resource(show_spinner=False)
def get_transcriber():
    from quran_agent.stt import Transcriber
    return Transcriber()

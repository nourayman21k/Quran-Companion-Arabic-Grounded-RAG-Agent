"""Quran companion -- Streamlit chat UI for the Quran & Hadith research agent.

Run:  streamlit run server.py   (preloads the agent at server start)
  or  streamlit run app.py      (loads it when the first visitor opens the page)
"""
import os
import re
import time
from collections import Counter

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from quran_agent import config
from quran_agent.graph import DISCLAIMER, reference_label, strip_number
from quran_agent.stt import LANGUAGES
from resources import get_agent, get_status, get_transcriber, retry_loading, start_loading

st.set_page_config(page_title="Quran companion", page_icon=":material/menu_book:")

TOOL_LABELS = {
    "quran_search_tool": "Quran search",
    "hadith_search": "Hadith search",
    "duck_duck_search": "Web search",
}
TOOL_ICONS = {
    "quran_search_tool": ":material/menu_book:",
    "hadith_search": ":material/history_edu:",
    "duck_duck_search": ":material/travel_explore:",
}
SUGGESTIONS = {
    ":material/family_restroom: بر الوالدين": "ماذا يقول القرآن والسنة عن بر الوالدين؟",
    ":material/pets: The cow story": "What is the story of the cow in Surah Al-Baqarah?",
    ":material/self_improvement: Patience": "What does Islam teach about patience during hardship?",
    ":material/favorite: إنما الأعمال بالنيات": "اشرح حديث إنما الأعمال بالنيات",
}
ARABIC_CHARS = re.compile(r"[\u0600-\u06FF]")


# ---------- helpers ----------
def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(bool(ARABIC_CHARS.match(c)) for c in letters) / len(letters) > 0.3


def write_text(text: str, key: str):
    """Markdown, right-to-left when the text is mostly Arabic."""
    with st.container(key=f"rtl-{key}" if is_arabic(text) else f"ltr-{key}"):
        st.markdown(text)


def chat_history() -> list:
    """Earlier turns for the query enhancer (lets follow-ups like 'what about his brother?' work)."""
    history = []
    for msg in st.session_state.messages:
        history.append(HumanMessage(content=msg["content"]) if msg["role"] == "user"
                       else AIMessage(content=msg["content"]))
    return history


def render_trace(trace: list[dict]):
    for step in trace:
        with st.status(step["label"], type="step", state="complete"):
            if step.get("detail"):
                st.markdown(step["detail"])


def render_sources(sources: list[dict], key: str):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/library_books:"):
        for n, s in enumerate(sources, start=1):
            st.markdown(f"{TOOL_ICONS.get(s['tool'], '')} **[{n}] {reference_label(s)}**")
            body = strip_number(s["text"]).split("\n", 1)
            text = body[1] if len(body) > 1 else body[0]
            write_text(text.replace("text_ar: ", "")[:1500], key=f"{key}-src{n}")


def render_assistant(msg: dict, key: str):
    if msg.get("trace"):
        with st.expander(msg.get("trace_label", "Research steps"), type="compact"):
            render_trace(msg["trace"])
    write_text(msg["content"], key=key)
    render_sources(msg.get("sources", []), key=key)
    st.caption(DISCLAIMER.replace("---", "").strip())


def describe(node: str, update: dict) -> dict | None:
    """Main-graph updates (enhancer, synthesizer, reflector) -> timeline steps."""
    if node == "enhancer":
        return {"label": "Rewrote the question for search", "detail": update["question"]}

    if node == "synthesizer":
        return {"label": "Wrote the answer", "detail": f"From {len(update.get('sources', []))} numbered sources"}

    if node == "reflector":
        decision = update["decision"]
        detail = update.get("feedback") or ", ".join(update.get("follow_ups", [])) or ""
        labels = {"accept": "Answer checked against the sources",
                  "revise": "Asked for a revision",
                  "research_more": "Asked for more research"}
        return {"label": labels.get(decision, decision), "detail": detail}
    return None


def describe_event(event: dict) -> dict | None:
    """Researcher / grader progress events (sent by the graph with stream_mode="custom")."""
    tool = TOOL_LABELS.get(event.get("tool"), event.get("tool") or "Search")
    if event["step"] == "search":
        queries = " · ".join(f"`{q}`" for q in event["queries"])
        return {"label": tool, "detail": f"Searching {queries}"}
    if event["step"] == "graded":
        if not event["total"]:
            return {"label": f"{tool}: no results", "detail": "Retrying with different keywords"}
        return {"label": f"{tool}: kept {event['kept']} of {event['total']} results",
                "detail": f"Still missing: {event['missing'] or 'nothing'}"}
    if event["step"] == "done":
        return {"label": f"{tool} finished", "detail": event.get("note", "")}
    return None


def run_agent(question: str, history: list) -> tuple[dict, list[dict]]:
    """Run the graph, streaming each step into the research timeline. Returns (final state, trace)."""
    agent = get_agent()
    final, trace = {}, []
    for namespace, mode, data in agent.stream(
        {"question": question, "history": history},
        stream_mode=["updates", "values", "custom"],
        subgraphs=True,
    ):
        if mode == "values":
            if not namespace:
                final = data
            continue
        if mode == "custom":
            steps = [describe_event(data)]
        elif not namespace:   # main-graph nodes; researcher progress comes through "custom"
            steps = [describe(node, update) for node, update in data.items() if update]
        else:
            continue
        for step in filter(None, steps):
            trace.append(step)
            render_trace([step])
    return final, trace


# ---------- state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    voice_language = st.segmented_control(
        "Voice language", list(LANGUAGES), default="Auto-detect", key="voice_language",
        help="Language Whisper large-v3 (Groq) should expect. Auto-detect works for Arabic and English.",
    )
    show_steps = st.toggle("Show research steps", value=True, key="show_steps",
                           help="Open the step-by-step research timeline while the agent works.")
    if st.button("New conversation", icon=":material/add_comment:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "A query enhancer rewrites your question, three researchers search the **Quran**, "
        "**Sahih al-Bukhari** and the **web** in parallel, a grader filters every result, "
        "and a reflector checks the cited answer before you see it."
    )
    if config.LLM_PROVIDER == "ollama":
        st.caption(f"LLM: `{config.OLLAMA_MODEL}` (local Ollama) · speech-to-text: `{config.WHISPER_MODEL}` (Groq)")
    else:
        st.caption(f"Search agents: `{config.FAST_LLM_MODEL}` · answers: `{config.LLM_MODEL}` "
                   f"(Groq, auto-fallback on rate limits) · speech-to-text: `{config.WHISPER_MODEL}`")

# ---------- header ----------
st.title("Quran companion")
st.caption("Ask about the Quran and Sahih al-Bukhari by text or voice, in Arabic or English. "
           "Every answer is grounded in cited verses, hadiths and sources.")

# right-to-left layout for Arabic text blocks (containers keyed rtl-*)
st.html("<style>[class*='st-key-rtl-'] { direction: rtl; text-align: right; }</style>")

if not os.getenv("GROQ_API_KEY"):
    st.error("`GROQ_API_KEY` is not set. Add it to a `.env` file next to `app.py` and restart.",
             icon=":material/key_off:")
    st.stop()

# ---------- loading progress (once per server start) ----------
if not get_status()["ready"]:
    start_loading()   # no-op if server.py already started it

    @st.fragment(run_every=1)
    def loading_progress():
        status = get_status()
        if status["ready"]:
            st.rerun()   # everything is loaded -> show the chat
        if status["error"]:
            st.error(f"Loading failed: {status['error']}", icon=":material/error:")
            if st.button("Retry", icon=":material/refresh:"):
                retry_loading()
            return
        label = f"**{status['stage']}**" + (f" · {status['detail']}" if status["detail"] else "")
        st.progress(min(max(status["fraction"], 0.0), 1.0), text=label)
        st.caption("This happens once per server start (about 20 s). The very first start also downloads "
                   "the bge-m3 embedding model (~2.3 GB).")

    loading_progress()
    st.stop()

# ---------- chat history ----------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            if msg.get("audio"):
                st.audio(msg["audio"])
                st.caption(":material/mic: Transcribed with Whisper large-v3 (Groq)")
            write_text(msg["content"], key=f"msg{i}")
        else:
            render_assistant(msg, key=f"msg{i}")

# ---------- input ----------
question, audio_bytes, audio_name = None, None, "audio.wav"

if not st.session_state.messages:
    suggestion = st.pills("Try asking", list(SUGGESTIONS), label_visibility="collapsed", key="suggestion")
    if suggestion:
        question = SUGGESTIONS[suggestion]

prompt = st.chat_input(
    "Ask a question, record your voice, or attach an audio file",
    accept_audio=True,
    accept_file=True,
    file_type=["wav", "mp3", "m4a", "flac", "ogg", "webm"],
    submit_mode="disable",
)

if prompt:
    audio_file = prompt.audio or (prompt.files[0] if prompt.files else None)
    question = (prompt.text or "").strip()
    if audio_file:
        audio_bytes = audio_file.getvalue()
        audio_name = audio_file.name or audio_name

if question or audio_bytes:
    turn = len(st.session_state.messages)
    history = chat_history()

    # --- user turn (voice -> Whisper large-v3 on Groq -> text) ---
    with st.chat_message("user"):
        if audio_bytes:
            st.audio(audio_bytes)
            with st.spinner("Transcribing with Whisper large-v3..."):
                try:
                    transcript = get_transcriber().transcribe(
                        audio_bytes, LANGUAGES[voice_language or "Auto-detect"], filename=audio_name)
                except Exception as e:
                    st.error(f"Transcription failed: {e}", icon=":material/mic_off:")
                    st.stop()
            st.caption(":material/mic: Transcribed with Whisper large-v3 (Groq)")
            question = f"{question}\n{transcript}".strip() if question else transcript
        if question:
            write_text(question, key=f"msg{turn}")

    if not question:
        st.warning("I couldn't hear any speech in that recording. Please try again.", icon=":material/mic_off:")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question, "audio": audio_bytes})

    # --- assistant turn ---
    with st.chat_message("assistant"):
        start = time.time()
        try:
            with st.status(":shimmer[Researching the Quran, hadith and the web]",
                           type="compact", expanded=show_steps) as status:
                final, trace = run_agent(question, history)
                elapsed = time.time() - start
                counts = Counter(e["tool"] for e in final.get("evidence", []))
                trace_label = (f"Researched for {elapsed:.0f}s · "
                               + ", ".join(f"{TOOL_LABELS[t]}: {n}" for t, n in counts.items()))
                status.update(label=trace_label, state="complete", expanded=False)
        except Exception as e:   # e.g. Groq rate limit still exceeded after all retries
            st.error(f"The agent failed: {e}", icon=":material/error:")
            st.session_state.messages.pop()
            st.stop()

        reply = {
            "role": "assistant",
            "content": final.get("answer") or "I couldn't find relevant sources for this question.",
            "sources": final.get("sources", []),
            "trace": trace,
            "trace_label": trace_label,
        }
        write_text(reply["content"], key=f"msg{turn + 1}")
        render_sources(reply["sources"], key=f"msg{turn + 1}")
        st.caption(DISCLAIMER.replace("---", "").strip())

    st.session_state.messages.append(reply)

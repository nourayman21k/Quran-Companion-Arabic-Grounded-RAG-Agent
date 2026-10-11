"""Conversation memory: a SQLite checkpointer that saves every conversation (thread) in data/,
so chats survive app and container restarts."""
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from . import config


def get_checkpointer(path=None) -> SqliteSaver:
    path = path or config.CHECKPOINT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    # Streamlit serves sessions from several threads; SqliteSaver serializes access with its own lock
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def list_threads(saver: SqliteSaver, limit: int = 20) -> list[str]:
    """Most recently updated conversation ids (checkpoint ids are time-ordered)."""
    with saver.lock:
        rows = saver.conn.execute(
            "SELECT thread_id, MAX(checkpoint_id) AS last FROM checkpoints "
            "WHERE checkpoint_ns = '' GROUP BY thread_id ORDER BY last DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [r[0] for r in rows]


def thread_messages(saver: SqliteSaver, thread_id: str) -> list:
    """The saved conversation (HumanMessage / AIMessage list) of a thread, read straight from the checkpoint."""
    saved = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    if not saved:
        return []
    return list(saved.checkpoint.get("channel_values", {}).get("messages", []))


def thread_title(saver: SqliteSaver, thread_id: str, max_chars: int = 40) -> str:
    messages = thread_messages(saver, thread_id)
    first = next((m.content for m in messages if m.type == "human"), "") or "New conversation"
    first = " ".join(first.split())
    return first if len(first) <= max_chars else first[:max_chars - 1] + "…"


def delete_thread(saver: SqliteSaver, thread_id: str):
    saver.delete_thread(thread_id)

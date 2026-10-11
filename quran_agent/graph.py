"""LangGraph multi-agent pipeline:

query enhancer + router --research--> 3 researchers in parallel (Quran / Hadith / Web, each with a grader)
                        |             -> synthesizer <-> reflector (revise, or research more) -> cited answer
                        +--chat-----> general chat (answers from the conversation memory, no tools)

The conversation (`messages`) is saved per thread by a checkpointer, so follow-ups and
"summarize your last answer" work, and chats survive restarts.
"""
import operator
import re
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Send
from pydantic import BaseModel, Field

from . import config

# ======================================================================
# States
# ======================================================================
ToolName = Literal["quran_search_tool", "hadith_search", "duck_duck_search"]


def add_or_reset(left: list, right: list | None) -> list:
    """Evidence accumulates within one question; a node returns None to clear it for the next question
    (the checkpointer keeps the whole state between turns)."""
    return [] if right is None else left + right


class MainState(BaseModel):
    question: str                                        # enhanced question (the enhancer overwrites the user's text)
    messages: Annotated[list[AnyMessage], add_messages] = []   # the conversation (saved by the checkpointer)
    route: str = ""                                      # "research" or "chat" (set by the enhancer)
    evidence: Annotated[list[dict], add_or_reset] = []   # merged from all researchers running in parallel
    answer: str = ""                                     # latest draft (inline [n] citations)
    sources: list[dict] = []                             # numbered sources the draft cites
    final_answer: str = ""                               # draft + references + disclaimer
    decision: str = ""                                   # reflector: accept / revise / research_more
    feedback: str = ""                                   # reflector: what to fix (for revise)
    follow_ups: list[str] = []                           # reflector: new questions (for research_more)
    reflections: int = 0


class ResearcherState(BaseModel):
    question: str
    tool_name: ToolName                                  # the ONE tool this researcher copy may use
    messages: Annotated[list[AnyMessage], add_messages] = []
    evidence: Annotated[list[dict], operator.add] = []   # {"question", "tool", "text"}
    rounds: int = 0


class ResearcherOutput(BaseModel):
    # only evidence flows back to MainState; each researcher's messages stay private
    evidence: Annotated[list[dict], operator.add] = []


# ======================================================================
# Prompts
# ======================================================================
class Enhancer(BaseModel):
    route: Literal["research", "chat"] = Field(description=(
        "'research' if answering needs the Quran, hadith or Islamic sources (verses, hadith, prophets, Islamic "
        "teachings, rulings, worship, Islamic history). 'chat' for everything else: greetings, thanks, questions "
        "about the assistant, general knowledge unrelated to Islam, or requests about the earlier conversation "
        "that need no new sources (summarize / translate / shorten / explain your last answer)."
    ))
    new_question: str = Field(description=(
        "The user's question rewritten as a clear, self-contained search question for the Quran, "
        "hadith, and web search tools. Fix spelling mistakes, colloquial or vague wording, and "
        "missing context (e.g. 'the cow story' -> 'the story of Musa commanding his people to "
        "slaughter a cow'), but keep the original meaning and do not add new topics or answer it. "
        "Write it in the same language as the user's question. "
        "If the question is already clear, return it unchanged."
    ))


ENHANCER_PROMPT = SystemMessage(content="""You are the Query Enhancer and router for a Quran and Hadith research assistant.
First choose the route:
- research: the question needs the Quran, hadith or Islamic sources -- even a follow-up like "what about his brother?"
  after a question about a prophet.
- chat: greetings, thanks, questions about you, general questions unrelated to Islam (e.g. coding, science, daily life),
  or requests about the earlier conversation that need no new sources ("summarize that", "translate your answer").
For chat, return the user's message as new_question (only resolve references); the rest of these rules are for research.

Rewrite the user's question so it can be searched well. The SAME rewritten question is sent in parallel to three researchers:
one searches the Arabic Quran, one searches Sahih al-Bukhari hadiths, and one searches the web (tafsir, context).
Each researcher sees ONLY your rewritten question, not the original one.
Rules:
- Keep the exact meaning and intent of the user's question. Do not add new topics or narrow it down.
- Fix spelling mistakes and turn slang or dialect (e.g. Egyptian Arabic) into clear standard wording,
  keeping the religious term it refers to (e.g. "اللي بيعق ابوه وامه" -> "عقوق الوالدين", NOT punishing the parents).
- Make it self-contained: name the prophet, surah, person or concept explicitly instead of "he", "it" or "that story".
- Expand vague references into the well-known topic (e.g. "the cow story" -> "the story of Musa commanding his people to slaughter a cow").
- You may receive the earlier conversation. Use it ONLY to resolve references in the latest question
  (e.g. after a question about Musa, "what about his brother?" -> "Who was Harun, the brother of Musa?").
  If the latest question is a new topic, ignore the earlier conversation.
- Write in the same language as the user (Arabic stays Arabic, English stays English).
- If the question is already clear, return it unchanged.
- Do not answer the question.""")

RESEARCHER_PROMPT = """You are a Researcher in a multi-agent Quran and Hadith research team. Your job is to gather evidence for the question, not to write the final answer.

## The team
- Query Enhancer: rewrote the user's question into the clear question below.
- Researchers: three researchers work on this SAME question in parallel, each with a different tool
  (Quran search, hadith search, web search). You only cover your own tool; the others cover theirs.
- Grader: reviews each of your tool results before you see it. It removes irrelevant results and adds a note
  starting with "GRADER:" that says what is still missing.
- Synthesizer: receives the evidence from all three researchers, picks the best context, and writes the final answer.

## The question
{question}

## Your tool
{tool_name}: {tool_description}

## How to work
1. Turn the question into a focused search query that suits your tool. {tool_tip}
2. Read the GRADER notes. If one says "nothing relevant" or that something is still missing,
   search again with different keywords (synonyms, the Arabic or English form, a narrower or broader topic).
3. Use specific search queries. Do not answer from your own memory.
4. Stop calling the tool once you have enough evidence, or after 2-3 searches that found nothing new.
   If your tool simply doesn't fit the question (e.g. no hadith exists on a purely Quranic detail), stop early.

## When you are done
Do not write the final answer. Reply with one line: "covered", or "not found: <what you tried>"."""

TOOL_TIPS = {
    "quran_search_tool": "The Quran index is Arabic, so write the query in Arabic (e.g. 'عقوق الوالدين', 'قصة البقرة مع موسى').",
    "hadith_search": "Describe the topic or the hadith wording in Arabic or English (e.g. 'بر الوالدين', 'honesty and lying').",
    "duck_duck_search": "Search for tafsir, quran verses, scholarly explanation, or historical context (e.g. 'تفسير ابن كثير سورة البقرة 67').",
}


class GraderOutput(BaseModel):
    relevant: list[int] = Field(description="Numbers of the results that help answer the question")
    still_missing: str = Field(description="One sentence on what is still missing, or empty if nothing")


GRADER_PROMPT = SystemMessage(content="""You are a relevance grader for a Quran and Hadith research assistant.
You will receive a question and a numbered list of search results (Quran passages, hadiths, or web pages).
1. Return the numbers of the results that directly help answer the question, even partially. Ignore results that are only loosely related
   (e.g. a hadith that only mentions the same word but is about a different topic).
2. In one sentence, say what information is still missing. Leave it empty if nothing is missing.
If no result is relevant, return an empty list. Do not answer the question yourself.""")

SYNTHESIZER_PROMPT = SystemMessage(content="""You are the Synthesizer for a Quran and Hadith research assistant.
Researchers have gathered numbered sources: Quran passages (Arabic), Sahih al-Bukhari hadiths, and web pages (tafsir, context).
Write the final answer to the user's question using ONLY these sources.

Rules:
- Base every statement on the sources. Do not use outside knowledge, and never quote a verse or hadith that is not in the sources.
- Cite every fact inline with its source number in square brackets, like [1] or [2][5]. Only use numbers from the list.
- When you refer to a Quran passage, name the surah and ayah range exactly as the source gives it (e.g. Al-Baqara 2:65-69).
- Use Quran and hadith sources for the religious content, and web sources for explanation (tafsir) and context.
- Cover every part of the question. If the sources don't answer a part, say so honestly.
- If sources disagree, say so and explain the difference.
- Answer in the same language as the question. Answer directly first, then add supporting details. Be concise and respectful.
- Do NOT write a references list or a disclaimer; they are added automatically.""")

DISCLAIMER = ("\n\n---\nThis is an AI-assisted retrieval tool, not a scholarly ruling. "
              "For religious guidance, please consult a qualified scholar.")


CHAT_PROMPT = SystemMessage(content="""You are Quran Companion, a friendly assistant. Your main skill is researching the Quran and
Sahih al-Bukhari with cited sources; that happens in a separate research mode. Right now you are in general chat mode.

- Answer greetings, questions about yourself, general-knowledge questions, and requests about the earlier conversation
  (summarize, translate, shorten, explain your previous answer) directly and helpfully.
- Use the earlier conversation as memory: refer back to what the user said and to your previous answers.
- When you restate a previous answer, keep its [n] citations; never invent new verses, hadiths or references.
- Do not give religious rulings or quote verses/hadith from memory -- suggest asking it as a question so the research
  mode can find cited sources.
- Answer in the same language as the user. Be concise.""")


class Reflection(BaseModel):
    decision: Literal["accept", "revise", "research_more"] = Field(
        description="accept = supported and complete; revise = fix the writing; research_more = information is missing")
    feedback: str = Field(
        description="For 'revise': which claims are unsupported or wrongly cited, and how to fix them. Empty otherwise.")
    follow_up_questions: list[str] = Field(
        description="For 'research_more': 1-2 self-contained questions covering only what is missing. Empty otherwise.")


REFLECTOR_PROMPT = SystemMessage(content="""You are the Reflector for a Quran and Hadith research assistant.
You review a draft answer against the numbered sources it was written from.

Check, in this order:
1. Support: every fact must have a citation [n], and source n must actually say it. Be strictest about religious content:
   a quoted or paraphrased verse/hadith that is not in its cited source, or a wrong surah:ayah or hadith number, is unsupported.
2. Coverage: the answer must cover every part of the user's question.

Then decide:
- accept: every fact is supported and every part of the question is covered. Minor style issues are fine.
- revise: the sources contain what is needed, but the answer has unsupported claims, wrong citations,
  or leaves out something the sources do say. Give specific feedback.
- research_more: the sources do NOT contain information needed for part of the question. Write 1-2 self-contained
  follow-up questions for only the missing parts, in the same language as the user's question
  (name the prophet, surah or concept instead of "he", "it" or "this"). Each one is sent to the Quran, hadith
  and web researchers.

Be strict about made-up facts and lenient about style.""")


# ======================================================================
# Source helpers
# ======================================================================
NO_RESULT_PREFIXES = ("No matching", "No web results", "Search failed")
# Quran / web results start with "[1] ...", hadith results start with "Hadith #..."
RESULT_SPLIT = re.compile(r"\n\n(?=\[\d+\]|Hadith #)")
SOURCE_LABELS = {"quran_search_tool": "Quran", "hadith_search": "Hadith", "duck_duck_search": "Web"}


def short_view(result: str, max_chars: int) -> str:
    """Hadith: keep the header + English text (drop the long Arabic narrator chain); cap the length."""
    if result.startswith("Hadith #") and "\nEN: " in result:
        header = result.split("\n", 1)[0]
        result = header + "\n" + result.split("\nEN: ", 1)[1]
    return result[:max_chars]


def strip_number(text: str) -> str:
    """'[2] Surah Al-Baqara (البقرة), 2:65-69\\n...' -> 'Surah Al-Baqara (البقرة), 2:65-69\\n...'"""
    return re.sub(r"^\[\d+\]\s*", "", text)


def unique_sources(evidence: list[dict]) -> list[dict]:
    """Drop duplicates: the same passage can be found by several searches, each with a different [n]."""
    seen, sources = set(), []
    for e in evidence:
        key = strip_number(e["text"])
        if key not in seen:
            seen.add(key)
            sources.append(e)
    return sources


def select_sources(evidence: list[dict]) -> list[dict]:
    """Unique sources, picked round-robin from each (question, tool) group, capped at MAX_SOURCES.
    Deterministic, so the synthesizer and reflector see the same [n] numbers."""
    groups = {}
    for e in unique_sources(evidence):
        groups.setdefault((e["question"], e["tool"]), []).append(e)
    picked = []
    for i in range(max((len(g) for g in groups.values()), default=0)):
        for g in groups.values():
            if i < len(g) and len(picked) < config.MAX_SOURCES:
                picked.append(g[i])
    return picked


def format_sources(sources: list[dict]) -> str:
    """One global numbering [1] ... [n] across all researchers, tagged with the source type."""
    return "\n\n".join(
        f"[{n}] ({SOURCE_LABELS.get(s['tool'], s['tool'])})\n"
        f"{short_view(strip_number(s['text']), config.SOURCE_CHARS).replace('text_ar: ', '')}"
        for n, s in enumerate(sources, start=1)
    )


def reference_label(source: dict) -> str:
    first_line = strip_number(source["text"]).splitlines()[0]
    if source["tool"] == "hadith_search":
        m = re.match(r"Hadith #(\d+)", first_line)
        return f"Sahih al-Bukhari, Hadith #{m.group(1)}" if m else first_line
    if source["tool"] == "quran_search_tool":
        return f"Quran: {first_line}"
    return f"Web: {first_line}"


def normalize_citations(answer: str) -> str:
    """gpt-oss sometimes cites as 【3】, ［3］ or 【3†source】 -- turn them all into [3]."""
    return re.sub(r"[【［]\s*(\d+)[^】］\]]*[】］]", r"[\1]", answer)


def build_references(answer: str, sources: list[dict]) -> str:
    """References list for the [n] numbers the answer actually cites."""
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(sources)})
    if not cited:
        return ""
    lines = [f"[{n}] {reference_label(sources[n - 1])}" for n in cited]
    return "\n\n## References\n" + "\n".join(lines)


def recent_history(messages: list[AnyMessage], turns: int, answer_chars: int) -> list[AnyMessage]:
    """The last `turns` messages before the current question, with long answers shortened
    (and the stored sources dropped -- the models only need the text)."""
    earlier = messages[:-1][-turns:]
    return [AIMessage(content=m.content[:answer_chars]) if isinstance(m, AIMessage)
            else HumanMessage(content=m.content) for m in earlier]


def _turn_reset() -> dict:
    """Per-question fields to clear at the start of every turn (the checkpointer keeps the old values)."""
    return {"evidence": None, "answer": "", "sources": [], "final_answer": "", "decision": "",
            "feedback": "", "follow_ups": [], "reflections": 0}


# ======================================================================
# Graph
# ======================================================================
def build_graph(models, tools: list, checkpointer=None):
    """Compile the full pipeline for `models` (llm.Models: fast + smart fallback chains) and the three tools.
    With a `checkpointer`, invoke with config={"configurable": {"thread_id": ...}} to keep a conversation."""
    tools_by_name = {t.name: t for t in tools}

    # many small calls -> fast model; the answer and its review -> smart model (each falls back on limits)
    enhancer_llm = models.fast.structured(Enhancer)
    grader_llm = models.fast.structured(GraderOutput)
    researcher_llms = {t.name: models.fast.with_tools([t]) for t in tools}   # each researcher sees ONLY its tool
    synthesizer_llm = models.smart.plain()
    reflector_llm = models.smart.structured(Reflection)
    chat_llm = models.smart.plain()

    # ---------- enhancer + router ----------
    def enhancer_node(state: MainState) -> dict:
        history = recent_history(state.messages, config.HISTORY_TURNS, answer_chars=800)
        out = enhancer_llm.invoke([ENHANCER_PROMPT, *history, HumanMessage(content=state.question)])
        return {"question": out.new_question, "route": out.route, **_turn_reset()}

    def route_after_enhancer(state: MainState):
        if state.route == "chat":
            return "chat"
        return [Send("graph_researcher", {"question": state.question, "tool_name": t.name}) for t in tools]

    # ---------- general chat (no tools, answers from the conversation memory) ----------
    def chat_node(state: MainState) -> dict:
        history = recent_history(state.messages, config.CHAT_HISTORY_TURNS, answer_chars=3000)
        user_text = state.messages[-1].content if state.messages else state.question
        answer = chat_llm.invoke([CHAT_PROMPT, *history, HumanMessage(content=user_text)]).content
        return {"answer": answer, "final_answer": answer, "messages": [AIMessage(content=answer)]}

    # ---------- researcher sub-graph ----------
    def researcher_agent_node(state: ResearcherState) -> dict:
        tool = tools_by_name[state.tool_name]
        sys_msg = SystemMessage(content=RESEARCHER_PROMPT.format(
            question=state.question,
            tool_name=tool.name,
            tool_description=tool.description,
            tool_tip=TOOL_TIPS[tool.name],
        ))
        response = researcher_llms[tool.name].invoke([sys_msg, HumanMessage(content=state.question)] + state.messages)

        # progress event for the UI (stream_mode="custom"); the sub-graph's own updates only carry evidence
        write = get_stream_writer()
        if response.tool_calls:
            write({"step": "search", "tool": tool.name,
                   "queries": [c["args"].get("query", "") for c in response.tool_calls]})
        else:
            write({"step": "done", "tool": tool.name, "note": str(response.content)[:300]})
        return {"messages": [response]}

    def research_grader_node(state: ResearcherState) -> dict:
        # this round's tool results = the ToolMessages at the end of the conversation
        tool_msgs = []
        for m in reversed(state.messages):
            if not isinstance(m, ToolMessage):
                break
            tool_msgs.append(m)

        write = get_stream_writer()
        cleaned, evidence = [], []
        for m in tool_msgs:
            content = m.content.strip()
            if content.startswith(NO_RESULT_PREFIXES):
                note = "GRADER: nothing found. Rewrite the query with different keywords."
                cleaned.append(m.model_copy(update={"content": f"{content}\n\n{note}"}))
                write({"step": "graded", "tool": m.name, "kept": 0, "total": 0, "missing": "no results"})
                continue

            results = RESULT_SPLIT.split(content)
            numbered = "\n\n".join(f"Result {i}:\n{short_view(r, config.GRADER_CHARS)}"
                                   for i, r in enumerate(results, start=1))
            out = grader_llm.invoke([GRADER_PROMPT, HumanMessage(content=f"Question: {state.question}\n\n{numbered}")])
            kept = [results[i - 1] for i in dict.fromkeys(out.relevant) if 1 <= i <= len(results)]
            # evidence keeps the FULL text (Arabic + English) for the final answer
            evidence += [{"question": state.question, "tool": m.name, "text": r} for r in kept]

            if kept:
                note = f"GRADER: kept {len(kept)} of {len(results)}. Still missing: {out.still_missing or 'nothing'}"
            else:
                note = "GRADER: nothing relevant. Rewrite the query with different keywords."
            write({"step": "graded", "tool": m.name, "kept": len(kept), "total": len(results),
                   "missing": out.still_missing})
            # the researcher only needs a short view to decide whether to search again
            short_kept = [short_view(r, config.RESEARCHER_CHARS) for r in kept]
            cleaned.append(m.model_copy(update={"content": "\n\n".join(short_kept + [note])}))

        # same message ids -> add_messages replaces the raw tool results with the graded ones
        return {"messages": cleaned, "evidence": evidence, "rounds": state.rounds + 1}

    def researcher_route(state: ResearcherState):
        if state.rounds >= config.MAX_ROUNDS:
            return END
        return tools_condition(state)

    researcher = StateGraph(ResearcherState, output_schema=ResearcherOutput)
    researcher.add_node("searcher", researcher_agent_node)
    researcher.add_node("tools", ToolNode(tools))
    researcher.add_node("grader", research_grader_node)
    researcher.add_edge(START, "searcher")
    researcher.add_conditional_edges("searcher", researcher_route, {"tools": "tools", END: END})
    researcher.add_edge("tools", "grader")
    researcher.add_edge("grader", "searcher")
    graph_researcher = researcher.compile(checkpointer=False)   # scratch work: not saved to the conversation memory

    # ---------- synthesizer + reflector ----------
    def synthesizer_node(state: MainState) -> dict:
        sources = select_sources(state.evidence)
        request = f"Question: {state.question}\n\nSources:\n{format_sources(sources)}"
        if state.feedback:   # revise loop: show the previous draft and what to fix
            request += (f"\n\nYour previous draft:\n{state.answer}"
                        f"\n\nThe Reflector asked for these fixes:\n{state.feedback}"
                        "\n\nWrite the corrected answer.")

        answer = synthesizer_llm.invoke([SYNTHESIZER_PROMPT, HumanMessage(content=request)]).content
        answer = normalize_citations(answer)
        return {
            "answer": answer,
            "sources": sources,
            "final_answer": answer + build_references(answer, sources) + DISCLAIMER,
        }

    def reflector_node(state: MainState) -> dict:
        review = reflector_llm.invoke([REFLECTOR_PROMPT, HumanMessage(content=
            f"Question: {state.question}\n\n"
            f"Sources:\n{format_sources(state.sources)}\n\n"
            f"Draft answer:\n{state.answer}"
        )])
        return {
            "decision": review.decision,
            "feedback": review.feedback if review.decision == "revise" else "",
            "follow_ups": review.follow_up_questions[:2] if review.decision == "research_more" else [],
            "reflections": state.reflections + 1,
        }

    def router_after_reflection(state: MainState):
        if state.decision == "accept" or state.reflections >= config.LOOP_MAX:
            return "finalize"
        if state.decision == "revise":
            return "synthesizer"
        if state.follow_ups:
            # every follow-up question goes to all 3 researchers (one per tool), in parallel
            return [Send("graph_researcher", {"question": q, "tool_name": t.name})
                    for q in state.follow_ups for t in tools]
        return "finalize"

    def finalize_node(state: MainState) -> dict:
        # only the accepted answer goes into the conversation memory (with its sources, for the UI)
        return {"messages": [AIMessage(content=state.answer, additional_kwargs={"sources": state.sources})]}

    graph = StateGraph(MainState)
    graph.add_node("enhancer", enhancer_node)
    graph.add_node("chat", chat_node)
    graph.add_node("graph_researcher", graph_researcher)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "enhancer")
    graph.add_conditional_edges("enhancer", route_after_enhancer, ["chat", "graph_researcher"])
    graph.add_edge("chat", END)
    graph.add_edge("graph_researcher", "synthesizer")   # waits for all researchers
    graph.add_edge("synthesizer", "reflector")
    graph.add_conditional_edges("reflector", router_after_reflection, ["synthesizer", "graph_researcher", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def load_agent(stores=None, progress=None, checkpointer=None):
    """Embeddings + indexes + tools + LLMs -> compiled graph (build once and reuse).
    `progress(stage, fraction, detail="")` is called as loading advances (fraction 0..1).
    `checkpointer`: conversation memory; defaults to the SQLite file in data/ (pass False to disable)."""
    from .data import _no_progress, load_stores
    from .llm import build_models
    from .memory import get_checkpointer
    from .tools import build_tools
    progress = progress or _no_progress
    stores = stores or load_stores(progress=progress)
    progress("Compiling the agent", 0.97)
    if checkpointer is None:
        checkpointer = get_checkpointer()
    return build_graph(build_models(), build_tools(stores), checkpointer=checkpointer or None)

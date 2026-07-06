"""
Builds the Quran RAG + ReAct agent pipeline exactly as developed in the notebook,
packaged so it can be constructed once at FastAPI startup instead of per-request.

Call get_graph() to get the compiled, memory-checkpointed LangGraph agent.
"""

import os
import re
import json
import pickle
from itertools import groupby
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, AnyMessage

from . import config

# ---------------------------------------------------------------------------
# 1. Arabic normalization (unchanged from the notebook)
# ---------------------------------------------------------------------------

def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D4-\u08FF]', '', text)
    text = text.replace("\u0640", "")
    text = re.sub(r'[\u0622\u0623\u0625\u0671]', '\u0627', text)
    text = re.sub(r'[\u0649\u06CC]', '\u064A', text)
    text = text.replace("\u0624", "\u0648")
    text = text.replace("\u0626", "\u064A")
    text = text.replace("\u0629", "\u0647")
    text = re.sub(r'[\u06DD\u06DE\u06E9\u08E2\u0660-\u0669\u06F0-\u06F9]', '', text)
    text = re.sub(r'[\ufd3e\ufd3f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_quran_docs(docs, window_size=5, overlap=1):
    docs = sorted(docs, key=lambda d: (int(d.metadata["surah_no"]), int(d.metadata["ayah_no"])))
    chunks = []
    step = window_size - overlap

    for surah_no, group_iter in groupby(docs, key=lambda d: d.metadata["surah_no"]):
        group = list(group_iter)
        for start in range(0, len(group), step):
            window = group[start:start + window_size]
            if not window:
                break
            first, last = window[0], window[-1]
            text = " ".join(d.page_content for d in window)
            f_meta, l_meta = first.metadata, last.metadata
            ref = (f"{surah_no}:{f_meta['ayah_no']}"
                   if f_meta["ayah_no"] == l_meta["ayah_no"]
                   else f"{surah_no}:{f_meta['ayah_no']}-{l_meta['ayah_no']}")
            metadata = {
                "chunk_id": f"{int(surah_no):03d}_{int(f_meta['ayah_no']):03d}_{int(l_meta['ayah_no']):03d}",
                "surah_no": int(surah_no),
                "surah_name_ar": f_meta["surah_name_ar"],
                "surah_name_en": f_meta["surah_name_en"],
                "revelation_place": f_meta["revelation_place"],
                "start_ayah": int(f_meta["ayah_no"]),
                "end_ayah": int(l_meta["ayah_no"]),
                "reference": ref,
            }
            chunks.append(Document(page_content=text, metadata=metadata))
            if start + window_size >= len(group):
                break
    return chunks


def dedupe_overlapping_chunks(docs, overlap_threshold=0.5):
    """Drop chunks that heavily overlap (same surah, overlapping ayah range)
    with a chunk already kept, since window_size/overlap chunking produces
    near-duplicate neighboring windows that dilute the reranker's view."""
    selected = []
    for doc in docs:
        s = doc.metadata.get("surah_no")
        a = doc.metadata.get("start_ayah")
        b = doc.metadata.get("end_ayah")
        is_dup = False
        for kept in selected:
            s2 = kept.metadata.get("surah_no")
            a2 = kept.metadata.get("start_ayah")
            b2 = kept.metadata.get("end_ayah")
            if s == s2 and a is not None and b is not None and a2 is not None and b2 is not None:
                overlap = max(0, min(b, b2) - max(a, a2) + 1)
                span = min(b - a + 1, b2 - a2 + 1)
                if span > 0 and overlap / span >= overlap_threshold:
                    is_dup = True
                    break
        if not is_dup:
            selected.append(doc)
    return selected


# ---------------------------------------------------------------------------
# 2. Build (once) -- chunks, embeddings, FAISS, BM25, hybrid retriever
# ---------------------------------------------------------------------------

def _path(*parts):
    return os.path.join(config.DATA_DIR, *parts)


def build_chunks():
    from langchain_community.document_loaders.csv_loader import CSVLoader

    cache_path = _path("chunks_cache.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    loader = CSVLoader(
        file_path=_path("Quran_RAG_Ready.csv"),
        encoding="utf-8-sig",
        content_columns=["text_ar"],
        metadata_columns=["chunk_id", "surah_no", "surah_name_ar",
                           "surah_name_en", "revelation_place",
                           "ayah_no", "ayah_no_global", "reference"],
    )
    csv_docs = loader.load()
    for doc in csv_docs:
        doc.page_content = normalize_arabic(doc.page_content)

    chunks = chunk_quran_docs(csv_docs, window_size=5, overlap=1)
    with open(cache_path, "wb") as f:
        pickle.dump(chunks, f)
    return chunks


def build_retrievers(chunks):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        config.DATA_DIR,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    bm25_cache_path = _path("sparse_retriever_cache.pkl")
    if os.path.exists(bm25_cache_path):
        with open(bm25_cache_path, "rb") as f:
            sparse_retriever = pickle.load(f)
    else:
        sparse_retriever = BM25Retriever.from_documents(chunks)
        with open(bm25_cache_path, "wb") as f:
            pickle.dump(sparse_retriever, f)
    sparse_retriever.k = config.RETRIEVER_K

    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.4, 0.6],
    )
    return hybrid_retriever


def build_hadith_collection():
    import requests

    cache_path = _path("hadith_bukhari_eng.json")
    edition = "eng-bukhari"
    url = f"https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{edition}.json"

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    return data.get("hadiths", [])


# ---------------------------------------------------------------------------
# 3. Tools
# ---------------------------------------------------------------------------

reranker_prompt = PromptTemplate.from_template("""
You are an expert retrieval reranker.

Your job is NOT to answer the question.

Your job is to rank the retrieved documents according to how useful they are for answering the user's question.

Question:
{question}

Retrieved Documents:
{documents}

Instructions:
1. Rank the documents from MOST relevant to LEAST relevant.
2. Prefer documents that directly answer the user's question.
3. Prefer exact matches over loosely related topics.
4. If multiple documents are related, rank the one containing the most complete answer first.
5. Ignore duplicate information.
6. Do NOT explain your reasoning.
7. Return ONLY the document indices.

Output format:
0,3,1,2,4
""")

answer_prompt = PromptTemplate.from_template("""
You are a knowledgeable assistant answering questions about the Quran using only the verses provided as context.

Question:
{question}

Relevant verses (each prefixed with its surah:ayah reference):
{context}

Instructions:
1. Answer using only the information in the verses above — do not use outside knowledge.
2. If the verses don't contain enough information to answer, say so honestly instead of guessing.
3. Cite the surah:ayah reference(s) shown in brackets for every claim you make — do not invent references not present in context.
4. Keep the answer concise and respectful of the religious nature of the text.

Answer:
""")

DISCLAIMER = (
    "\n\n---\n"
    "This is an AI-assisted retrieval tool, not a scholarly ruling. "
    "For religious guidance, please consult a qualified scholar."
)


def make_quran_search_tool(hybrid_retriever, rerank_chain):
    @tool
    def quran_search(query: str) -> str:
        """Search the Quran for verses relevant to a question or topic.
        Use this whenever the user asks about Quranic verses, Islamic teachings,
        stories from the Quran (e.g. Musa, Ibrahim), or wants a Quran-grounded answer.
        Returns the most relevant verses along with their surah:ayah references.
        """
        normalized_query = normalize_arabic(query)
        raw_results = hybrid_retriever.invoke(normalized_query)
        deduped = dedupe_overlapping_chunks(raw_results)[: config.MAX_RERANK_CANDIDATES]

        doc_lines = [
            f"{i}. [{doc.metadata.get('reference', '')}] {doc.page_content[: config.RERANK_TRUNCATE_CHARS]}"
            for i, doc in enumerate(deduped)
        ]
        formatted_docs = "\n".join(doc_lines)

        ranked_output = rerank_chain.invoke({"question": query, "documents": formatted_docs})
        order = [int(i.strip()) for i in ranked_output.split(",") if i.strip().isdigit()]
        reranked = [deduped[i] for i in order if i < len(deduped)]

        top_docs = reranked[:5] if reranked else deduped[:5]
        return "\n".join(f"[{d.metadata.get('reference', '')}] {d.page_content}" for d in top_docs)

    return quran_search


def make_hadith_search_tool(hadith_list):
    @tool
    def hadith_search(query: str, max_results: int = 5) -> str:
        """Search Sahih al-Bukhari hadiths (English translation) for a keyword or topic.
        Use this when the user asks about a hadith, or a saying/action of the Prophet
        Muhammad (peace be upon him), or wants a hadith to complement a Quranic answer.
        Returns matching hadiths with their hadith numbers.
        """
        query_lower = query.lower()
        matches = [h for h in hadith_list if query_lower in h.get("text", "").lower()]

        if not matches:
            words = [w for w in query_lower.split() if len(w) > 3]
            matches = [h for h in hadith_list if any(w in h.get("text", "").lower() for w in words)]

        top = matches[:max_results]
        if not top:
            return "No matching hadiths found."

        return "\n\n".join(f"Hadith #{h.get('hadithnumber')}: {h.get('text')}" for h in top)

    return hadith_search


def build_search_tools():
    import wikipedia
    from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
    from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
    from langchain_community.tools.tavily_search import TavilySearchResults
    import arxiv as arxiv_pkg

    wikipedia.set_user_agent("AgenticAI/1.0 (https://example.com)")

    def _fetch_results_compat(self, query: str):
        if self.is_arxiv_identifier(query):
            search = self.arxiv_search(id_list=query.split(), max_results=self.top_k_results)
        else:
            search = self.arxiv_search(query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results)
        return list(arxiv_pkg.Client().results(search))

    ArxivAPIWrapper._fetch_results = _fetch_results_compat

    arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500))
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500))
    tavily_tool = TavilySearchResults()
    ddg_tool = DuckDuckGoSearchRun()

    return arxiv_tool, wiki_tool, tavily_tool, ddg_tool


SYSTEM_PROMPT = SystemMessage(content="""You are an assistant that answers questions about the Quran and Islamic teachings.
Always prefer the quran_search tool for any question about Quranic verses, stories, or teachings — it returns grounded, citable verses.
Use hadith_search when the user asks about a hadith or a saying/action of the Prophet Muhammad (peace be upon him), or to complement a Quran answer with a related hadith.
Use wikipedia for general background/historical context.
Use arxiv for questions about research papers.
Use tavily_search or ddg_search only for current events or information not related to religious text — prefer tavily_search first, and use ddg_search as a fallback if tavily is unavailable.
Never answer Quran-related or hadith-related questions from your own knowledge without calling quran_search or hadith_search first.
""")


# ---------------------------------------------------------------------------
# 4. Graph
# ---------------------------------------------------------------------------

from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


@lru_cache(maxsize=1)
def get_graph():
    """Builds everything once (embeddings, FAISS, BM25, tools, LLMs, graph) and
    caches it for the lifetime of the process. Call this from FastAPI startup,
    not per-request -- it's too expensive to rebuild on every call.
    """
    from langchain_groq import ChatGroq
    from langgraph.graph import StateGraph, START
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.checkpoint.memory import MemorySaver

    chunks = build_chunks()
    hybrid_retriever = build_retrievers(chunks)
    hadith_list = build_hadith_collection()
    arxiv_tool, wiki_tool, tavily_tool, ddg_tool = build_search_tools()

    rerank_llm = ChatGroq(model=config.RERANK_MODEL).with_retry(
        wait_exponential_jitter=True, stop_after_attempt=5
    )
    rerank_chain = reranker_prompt | rerank_llm | StrOutputParser()

    quran_search = make_quran_search_tool(hybrid_retriever, rerank_chain)
    hadith_search = make_hadith_search_tool(hadith_list)

    tools = [arxiv_tool, wiki_tool, tavily_tool, ddg_tool, quran_search, hadith_search]

    llm = ChatGroq(model=config.AGENT_MODEL)
    llm_with_tools = llm.bind_tools(tools).with_retry(
        wait_exponential_jitter=True, stop_after_attempt=5
    )

    def tool_calling_llm(state: State):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SYSTEM_PROMPT] + messages
        return {"messages": [llm_with_tools.invoke(messages)]}

    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph

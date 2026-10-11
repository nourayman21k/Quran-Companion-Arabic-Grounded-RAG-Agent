"""Evaluate the Quran & Hadith agent.

Stages (run all, or one with --stage):
  agent   run every research question through the full graph (Groq models) and save answers,
          sources, latency and citation stats; measure router accuracy on the labeled routing set
  ragas   score the saved answers with Ragas (faithfulness, answer relevancy, context precision)
          using a LOCAL judge (Ollama, default qwen3:14b) so scoring doesn't spend the Groq quota
  report  write eval/RESULTS.md

Run:  python eval/run_eval.py            (or --stage agent|ragas|report)
"""
import argparse
import json
import re
import statistics
import sys
import time
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EVAL_DIR = ROOT / "eval"
QUESTIONS = EVAL_DIR / "questions.json"
AGENT_OUT = EVAL_DIR / "agent_outputs.json"
RAGAS_OUT = EVAL_DIR / "ragas_scores.json"
REPORT = EVAL_DIR / "RESULTS.md"


# ---------------------------------------------------------------- agent stage
def contexts_for(sources: list[dict]) -> list[str]:
    """The source texts exactly as the synthesizer saw them (what the answer must be faithful to)."""
    from quran_agent import config
    from quran_agent.graph import short_view, strip_number
    return [short_view(strip_number(s["text"]), config.SOURCE_CHARS).replace("text_ar: ", "") for s in sources]


def stage_agent():
    from langchain_core.messages import AIMessage, HumanMessage
    from quran_agent.graph import ENHANCER_PROMPT, Enhancer, load_agent
    from quran_agent.llm import build_models

    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    agent = load_agent(checkpointer=False)
    results = []
    for i, q in enumerate(data["research"], 1):
        start = time.time()
        try:
            state = agent.invoke({"question": q, "messages": [HumanMessage(content=q)]})
            error = None
        except Exception as e:   # keep going; count it as a failure
            state, error = {}, f"{type(e).__name__}: {e}"
        latency = time.time() - start
        answer, sources = state.get("answer", ""), state.get("sources", [])
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
        results.append({
            "question": q,
            "enhanced_question": state.get("question", ""),
            "route": state.get("route", ""),
            "answer": answer,
            "contexts": contexts_for(sources),
            "source_tools": [s["tool"] for s in sources],
            "evidence_per_tool": dict(Counter(e["tool"] for e in state.get("evidence", []))),
            "citations": len(cited),
            "valid_citations": sum(1 for n in cited if 1 <= n <= len(sources)),
            "reviews": state.get("reflections", 0),
            "decision": state.get("decision", ""),
            "latency_s": round(latency, 1),
            "error": error,
        })
        print(f"[{i}/{len(data['research'])}] {latency:5.0f}s  sources={len(sources)}  "
              f"citations={len(cited)}  {q[:50]}", flush=True)

    # router accuracy (enhancer call only)
    router = build_models().fast.structured(Enhancer)
    routing = []
    for item in data["routing"]:
        history = [HumanMessage(content=t) if role == "user" else AIMessage(content=t)
                   for role, t in item.get("history", [])]
        try:
            predicted = router.invoke([ENHANCER_PROMPT, *history, HumanMessage(content=item["text"])]).route
        except Exception as e:
            predicted = f"error: {e}"
        routing.append({**item, "predicted": predicted})
        print(f"route {item['label']:8} -> {predicted:8}  {item['text'][:50]}", flush=True)

    AGENT_OUT.write_text(json.dumps({"research": results, "routing": routing}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"saved {AGENT_OUT}")


# ---------------------------------------------------------------- ragas stage
def import_ragas():
    # ragas 0.4.3 imports ChatVertexAI, which langchain-community 0.4 removed; it isn't used here
    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = stub
    import ragas
    return ragas


def stage_ragas(judge_model: str):
    import_ragas()
    from langchain_ollama import ChatOllama
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy

    from quran_agent.data import load_embeddings

    outputs = json.loads(AGENT_OUT.read_text(encoding="utf-8"))["research"]
    rows = [o for o in outputs if o["answer"] and o["contexts"]]
    dataset = EvaluationDataset.from_list([
        {"user_input": o["question"], "response": o["answer"], "retrieved_contexts": o["contexts"]} for o in rows
    ])
    judge = LangchainLLMWrapper(ChatOllama(model=judge_model, temperature=0, reasoning=False, num_ctx=16384))
    embeddings = LangchainEmbeddingsWrapper(load_embeddings())
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(timeout=900, max_workers=2, max_retries=3),
    )
    df = result.to_pandas()
    scores = []
    for o, (_, row) in zip(rows, df.iterrows()):
        scores.append({
            "question": o["question"],
            "faithfulness": _num(row.get("faithfulness")),
            "answer_relevancy": _num(row.get("answer_relevancy")),
            "context_precision": _num(row.get("llm_context_precision_without_reference")),
        })
    RAGAS_OUT.write_text(json.dumps({"judge": judge_model, "scores": scores}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"saved {RAGAS_OUT}")


def _num(v):
    try:
        v = float(v)
        return None if v != v else round(v, 4)   # NaN -> None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- report stage
def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else float("nan")


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else float("nan")


def stage_report():
    from quran_agent import config
    agent = json.loads(AGENT_OUT.read_text(encoding="utf-8"))
    research, routing = agent["research"], agent["routing"]
    ragas = json.loads(RAGAS_OUT.read_text(encoding="utf-8")) if RAGAS_OUT.exists() else None

    ok = [r for r in research if r["answer"] and not r["error"]]
    cites = sum(r["citations"] for r in ok)
    valid = sum(r["valid_citations"] for r in ok)
    lat = [r["latency_s"] for r in ok]
    route_hits = sum(1 for r in routing if r["predicted"] == r["label"])
    research_items = [r for r in routing if r["label"] == "research"]
    chat_items = [r for r in routing if r["label"] == "chat"]
    tool_cov = Counter(t for r in ok for t in set(r["source_tools"]))

    lines = [
        "# Evaluation results",
        "",
        f"Agent: `{config.FAST_LLM_MODEL}` (search agents) + `{config.LLM_MODEL}` (answers), Groq free tier with fallbacks. "
        f"Retrieval: `{config.EMBEDDING_MODEL}` + FAISS over 1,573 Quran passages and 15,160 hadith documents.",
        "",
        "## Answer quality (Ragas)",
        "",
    ]
    if ragas:
        s = ragas["scores"]
        lines += [
            f"Judge: local `{ragas['judge']}` (Ollama). {len(s)} research questions (Arabic, English, Egyptian dialect).",
            "",
            "| Metric | Mean | Median |",
            "|---|---|---|",
            f"| Faithfulness (claims supported by the retrieved sources) | {mean([x['faithfulness'] for x in s]):.2f} | {median([x['faithfulness'] for x in s]):.2f} |",
            f"| Answer relevancy | {mean([x['answer_relevancy'] for x in s]):.2f} | {median([x['answer_relevancy'] for x in s]):.2f} |",
            f"| Context precision (retrieved sources useful for the answer) | {mean([x['context_precision'] for x in s]):.2f} | {median([x['context_precision'] for x in s]):.2f} |",
            "",
        ]
    lines += [
        "## Pipeline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Research questions answered | {len(ok)}/{len(research)} |",
        f"| Mean / median latency per question | {statistics.mean(lat):.0f} s / {statistics.median(lat):.0f} s |",
        f"| Sources per answer (mean) | {mean([len(r['contexts']) for r in ok]):.1f} |",
        f"| Inline citations per answer (mean) | {cites / max(len(ok), 1):.1f} |",
        f"| Citations pointing to a real source | {valid}/{cites} ({100 * valid / max(cites, 1):.0f}%) |",
        f"| Answers citing a Quran / hadith / web source | {tool_cov['quran_search_tool']} / {tool_cov['hadith_search']} / {tool_cov['duck_duck_search']} of {len(ok)} |",
        f"| Mean reflector reviews per answer | {mean([r['reviews'] for r in ok]):.1f} |",
        "",
        "## Router (research vs. general chat)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {route_hits}/{len(routing)} ({100 * route_hits / len(routing):.0f}%) |",
        f"| Research messages routed to research | {sum(r['predicted'] == 'research' for r in research_items)}/{len(research_items)} |",
        f"| Chat messages routed to chat | {sum(r['predicted'] == 'chat' for r in chat_items)}/{len(chat_items)} |",
        "",
        "Notes: small benchmark (10 research questions, 30 routing messages) scored by a 14B local judge, "
        "so treat the Ragas means as indicative. Per-question scores: `ragas_scores.json`; answers, sources "
        "and latencies: `agent_outputs.json`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["agent", "ragas", "report", "all"], default="all")
    p.add_argument("--judge", default="qwen3:14b", help="local Ollama model used as the Ragas judge")
    args = p.parse_args()
    if args.stage in ("agent", "all"):
        stage_agent()
    if args.stage in ("ragas", "all"):
        stage_ragas(args.judge)
    if args.stage in ("report", "all"):
        stage_report()

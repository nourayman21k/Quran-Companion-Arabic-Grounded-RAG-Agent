# Evaluation results

Agent: `openai/gpt-oss-20b` (search agents) + `openai/gpt-oss-120b` (answers), Groq free tier with fallbacks. Retrieval: `BAAI/bge-m3` + FAISS over 1,573 Quran passages and 15,160 hadith documents.

## Answer quality (Ragas)

Judge: local `qwen3:14b` (Ollama). 10 research questions (Arabic, English, Egyptian dialect).

| Metric | Mean | Median |
|---|---|---|
| Faithfulness (claims supported by the retrieved sources) | 0.87 | 1.00 |
| Answer relevancy | 0.76 | 0.86 |
| Context precision (retrieved sources useful for the answer) | 0.76 | 0.98 |

## Pipeline

| Metric | Value |
|---|---|
| Research questions answered | 10/10 |
| Mean / median latency per question | 94 s / 82 s |
| Sources per answer (mean) | 7.6 |
| Inline citations per answer (mean) | 10.7 |
| Citations pointing to a real source | 107/107 (100%) |
| Answers citing a Quran / hadith / web source | 8 / 4 / 10 of 10 |
| Mean reflector reviews per answer | 1.5 |

## Router (research vs. general chat)

| Metric | Value |
|---|---|
| Accuracy | 30/30 (100%) |
| Research messages routed to research | 15/15 |
| Chat messages routed to chat | 15/15 |

Notes: small benchmark (10 research questions, 30 routing messages) scored by a 14B local judge, so treat the Ragas means as indicative. Per-question scores: `ragas_scores.json`; answers, sources and latencies: `agent_outputs.json`.

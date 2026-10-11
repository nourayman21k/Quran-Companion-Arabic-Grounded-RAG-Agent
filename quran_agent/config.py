"""Settings for the Quran & Hadith research agent. Every value can be overridden with an environment variable (or .env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))

# --- data files (built by scripts/build_indexes.py) ---
QURAN_CSV = DATA_DIR / "Quran_RAG_Ready.csv"
QURAN_INDEX_DIR = DATA_DIR / "quran_faiss_index"
HADITH_INDEX_DIR = DATA_DIR / "hadith_faiss_index"
HADITH_JSON = {
    "eng": DATA_DIR / "hadith_bukhari_eng.json",   # English translation of Sahih al-Bukhari
    "ara": DATA_DIR / "hadith_bukhari_ara.json",   # original Arabic text of Sahih al-Bukhari
}
HADITH_API_URL = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions"

# --- models ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")                    # groq | ollama (everything local)
# Groq rate limits are PER MODEL (free tier: 8k tokens/min, 200k tokens/day each), so the work is split:
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")          # synthesizer + reflector (few, important calls)
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "openai/gpt-oss-20b")  # enhancer, researchers, graders (many calls)
# when a model hits its limit (or fails), the next one takes over
FALLBACK_LLM_MODELS = [m.strip() for m in os.getenv(
    "FALLBACK_LLM_MODELS", "qwen/qwen3.8-27b,openai/gpt-oss-20b,openai/gpt-oss-120b").split(",") if m.strip()]
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")            # gpt-oss hidden reasoning (counts toward limits)
# last-resort local fallback via Ollama (used only if Ollama is running and has this model; "" disables it)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")      # Groq speech-to-text
DEVICE = os.getenv("DEVICE", "auto")                               # auto | cuda | cpu

# --- retrieval caps (small on purpose: Groq free tier = 8k tokens/min) ---
QURAN_TOP_K = int(os.getenv("QURAN_TOP_K", 3))          # passages per Quran search
HADITH_TOP_K = int(os.getenv("HADITH_TOP_K", 3))        # hadiths per hadith search
WEB_TOP_K = int(os.getenv("WEB_TOP_K", 3))              # results per web search
HADITH_MAX_CHARS = 2000                                 # hadith text embedded per document

# --- agent loop ---
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", 2))            # searches per researcher
LOOP_MAX = int(os.getenv("LOOP_MAX", 3))                # reflector reviews before returning the draft
MAX_SOURCES = int(os.getenv("MAX_SOURCES", 9))          # sources given to the synthesizer / reflector
SOURCE_CHARS = int(os.getenv("SOURCE_CHARS", 600))      # characters per source
GRADER_CHARS = int(os.getenv("GRADER_CHARS", 600))      # characters of each result the grader sees
RESEARCHER_CHARS = int(os.getenv("RESEARCHER_CHARS", 400))  # characters of each kept result the researcher sees
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", 6))      # earlier chat messages the enhancer sees

# Groq SDK retries short per-minute waits itself; anything longer switches to the next model
LLM_MAX_RETRIES = 2
RETRY = dict(wait_exponential_jitter=True, stop_after_attempt=2)   # whole fallback chain, e.g. bad JSON


def resolve_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

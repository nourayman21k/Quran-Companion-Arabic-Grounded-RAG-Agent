import os
from dotenv import load_dotenv

load_dotenv()

# Where the pre-built artifacts live (FAISS index, pickled chunks/BM25, CSV, hadith json).
# Mounted as a volume or baked into the image at /app/data in Docker.
DATA_DIR = os.getenv("DATA_DIR", "data")

# --- API keys used by the various tools ---
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# --- LangSmith tracing (debugging) ---
# Set LANGCHAIN_TRACING_V2=false in .env to turn this off, e.g. in prod.
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "quran-rag-agent")

# --- Model choices (kept as config so you can swap without touching pipeline.py) ---
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen/qwen3-32b")
RERANK_MODEL = os.getenv("RERANK_MODEL", "openai/gpt-oss-20b")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "qwen/qwen3-32b")

# --- Retrieval tuning ---
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "8"))
MAX_RERANK_CANDIDATES = int(os.getenv("MAX_RERANK_CANDIDATES", "12"))
RERANK_TRUNCATE_CHARS = int(os.getenv("RERANK_TRUNCATE_CHARS", "200"))

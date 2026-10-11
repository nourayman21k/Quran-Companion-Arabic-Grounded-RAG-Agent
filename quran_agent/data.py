"""Embeddings, the Quran and hadith FAISS indexes, and the merged Arabic + English hadith collection.

Loading functions take an optional `progress(stage, fraction, detail="")` callback so a UI can show
a progress bar (fraction is 0..1 for the whole load).
"""
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import requests
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from . import config
from .arabic import chunk_quran_docs, normalize_arabic


@dataclass
class Stores:
    quran: FAISS
    hadith: FAISS
    hadith_by_number: dict


def _no_progress(stage: str, fraction: float, detail: str = ""):
    pass


# files sentence-transformers doesn't need (bge-m3 also ships an ONNX copy, ColBERT/sparse heads and images)
EMBEDDING_IGNORE = ["onnx/*", "imgs/*", "*.jpg", "*.webp", "*.pt", "*.onnx*", "*.h5", "*.msgpack", "openvino/*"]


def _model_cache_dir(repo_id: str) -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"


def embedding_model_cached(repo_id: str = config.EMBEDDING_MODEL) -> bool:
    # weights only appear in a snapshot folder once their download has finished
    repo_dir = _model_cache_dir(repo_id)
    return any(repo_dir.glob("snapshots/*/pytorch_model.bin")) or any(repo_dir.glob("snapshots/*/model.safetensors"))


def download_embedding_model(progress=_no_progress, repo_id: str = config.EMBEDDING_MODEL):
    """Download the embedding model once (first start), reporting bytes received as 0..60% of the load."""
    if embedding_model_cached(repo_id):
        return
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import filter_repo_objects

    info = HfApi().model_info(repo_id, files_metadata=True)
    wanted = set(filter_repo_objects([s.rfilename for s in info.siblings], ignore_patterns=EMBEDDING_IGNORE))
    big_files = [s for s in info.siblings if s.rfilename in wanted and s.lfs]   # the weights etc.
    total = sum(s.size for s in big_files) or 1
    blobs = _model_cache_dir(repo_id) / "blobs"

    def received() -> int:
        n = 0
        for s in big_files:
            blob = blobs / s.lfs.sha256
            if blob.exists():
                n += blob.stat().st_size
            else:   # still downloading: <sha256>.<id>.incomplete
                n += sum(p.stat().st_size for p in blobs.glob(f"{s.lfs.sha256}*.incomplete"))
        return n

    done = threading.Event()

    def report_loop():
        while not done.wait(1.0):
            got = received()
            progress("Downloading the embedding model (first start only)", 0.6 * min(got / total, 1.0),
                     f"{got / 1e9:.2f} / {total / 1e9:.2f} GB")

    progress("Downloading the embedding model (first start only)", 0.0, f"0.00 / {total / 1e9:.2f} GB")
    threading.Thread(target=report_loop, daemon=True).start()
    try:
        snapshot_download(repo_id, ignore_patterns=EMBEDDING_IGNORE)
    finally:
        done.set()


def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.resolve_device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


# ---------- Quran ----------
def build_quran_index(embeddings) -> FAISS:
    loader = CSVLoader(
        file_path=str(config.QURAN_CSV),
        encoding="utf-8-sig",
        content_columns=["text_ar"],
        metadata_columns=["chunk_id", "surah_no", "surah_name_ar",
                          "surah_name_en", "revelation_place",
                          "ayah_no", "ayah_no_global", "reference"],
    )
    docs = loader.load()
    for doc in docs:
        doc.page_content = normalize_arabic(doc.page_content)
    chunks = chunk_quran_docs(docs, window_size=5, overlap=1)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(config.QURAN_INDEX_DIR))
    return store


def load_quran_index(embeddings) -> FAISS:
    if not (config.QURAN_INDEX_DIR / "index.faiss").exists():
        return build_quran_index(embeddings)
    return FAISS.load_local(str(config.QURAN_INDEX_DIR), embeddings=embeddings,
                            allow_dangerous_deserialization=True)


# ---------- Hadith (fawazahmed0/hadith-api, free, no key required) ----------
def load_hadith_edition(lang: str) -> list[dict]:
    path = config.HADITH_JSON[lang]
    if not path.exists():
        edition = "eng-bukhari" if lang == "eng" else "ara-bukhari"
        resp = requests.get(f"{config.HADITH_API_URL}/{edition}.json", timeout=60)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(resp.json(), ensure_ascii=False), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8")).get("hadiths", [])


def load_hadith_records() -> list[dict]:
    """Merge both editions by hadith number so every record has Arabic + English text."""
    ara_by_number = {h["hadithnumber"]: h.get("text", "") for h in load_hadith_edition("ara")}
    records = []
    for h in load_hadith_edition("eng"):
        text_ar = ara_by_number.get(h["hadithnumber"], "")
        records.append({
            "hadithnumber": h["hadithnumber"],
            "text_en": h.get("text", ""),
            "text_ar": text_ar,
            "text_ar_norm": normalize_arabic(text_ar),   # what Arabic queries are matched against
        })
    return records


def build_hadith_index(embeddings, records: list[dict], batch_size: int = 512, progress=_no_progress) -> FAISS:
    # Two documents per hadith (normalized Arabic + English) so both Arabic and English queries
    # match well; bge-m3 is multilingual, so cross-language queries work too.
    docs = []
    for h in records:
        if h["text_ar_norm"]:
            docs.append(Document(page_content=h["text_ar_norm"][:config.HADITH_MAX_CHARS],
                                 metadata={"hadithnumber": h["hadithnumber"], "lang": "ar"}))
        if h["text_en"]:
            docs.append(Document(page_content=h["text_en"][:config.HADITH_MAX_CHARS],
                                 metadata={"hadithnumber": h["hadithnumber"], "lang": "en"}))

    store = None
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        if store is None:
            store = FAISS.from_documents(batch, embeddings)
        else:
            store.add_documents(batch)
        done = min(i + batch_size, len(docs))
        progress("Building the hadith index (first start only)", 0.86 + 0.1 * done / len(docs),
                 f"{done:,} / {len(docs):,} documents")
    store.save_local(str(config.HADITH_INDEX_DIR))
    return store


def load_hadith_index(embeddings, records: list[dict], progress=_no_progress) -> FAISS:
    if not (config.HADITH_INDEX_DIR / "index.faiss").exists():
        return build_hadith_index(embeddings, records, progress=progress)
    return FAISS.load_local(str(config.HADITH_INDEX_DIR), embeddings=embeddings,
                            allow_dangerous_deserialization=True)


def load_stores(embeddings=None, progress=_no_progress) -> Stores:
    """Load everything the tools need, building any missing index first."""
    if embeddings is None:
        download_embedding_model(progress)
        progress("Loading the embedding model", 0.62, f"{config.EMBEDDING_MODEL} on {config.resolve_device()}")
        embeddings = load_embeddings()

    progress("Loading the hadith collection", 0.72, "Sahih al-Bukhari, Arabic + English")
    records = load_hadith_records()

    quran_ready = (config.QURAN_INDEX_DIR / "index.faiss").exists()
    progress("Loading the Quran index" if quran_ready else "Building the Quran index (first start only)", 0.8)
    quran = load_quran_index(embeddings)

    progress("Loading the hadith index", 0.86)
    hadith = load_hadith_index(embeddings, records, progress)

    return Stores(quran=quran, hadith=hadith, hadith_by_number={h["hadithnumber"]: h for h in records})

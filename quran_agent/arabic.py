"""Arabic text normalization and Quran verse chunking."""
import re
from itertools import groupby

from langchain_core.documents import Document


def normalize_arabic(text: str) -> str:
    """Remove diacritics/tatweel and unify letter variants so queries match the indexed text."""
    if not text:
        return ""
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D4-\u08FF]', '', text)
    text = text.replace("\u0640", "")
    text = re.sub(r'[\u0622\u0623\u0625\u0671]', '\u0627', text)
    text = re.sub(r'[\u0649\u06CC]', '\u064A', text)
    text = text.replace("\u0624", "\u0648")  # ؤ -> و
    text = text.replace("\u0626", "\u064A")  # ئ -> ي
    text = text.replace("\u0629", "\u0647")
    text = re.sub(r'[\u06DD\u06DE\u06E9\u08E2\u0660-\u0669\u06F0-\u06F9]', '', text)
    text = re.sub(r'[\uFD3E\uFD3F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_quran_docs(docs, window_size=5, overlap=1):
    """Group consecutive ayahs of the same surah into windows of `window_size` with `overlap`."""
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

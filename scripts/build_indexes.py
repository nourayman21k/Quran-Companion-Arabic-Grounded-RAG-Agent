"""Build any missing data artifacts in data/:

- data/quran_faiss_index/   from data/Quran_RAG_Ready.csv (5-ayah windows, bge-m3)
- data/hadith_bukhari_*.json  downloaded from the free hadith API
- data/hadith_faiss_index/  Arabic + English document per hadith (bge-m3)

Existing artifacts are kept, so this is safe to run on every start (the Docker entrypoint does).
Use --rebuild to rebuild both indexes from scratch.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quran_agent import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="delete and rebuild both FAISS indexes")
    args = parser.parse_args()

    if args.rebuild:
        shutil.rmtree(config.QURAN_INDEX_DIR, ignore_errors=True)
        shutil.rmtree(config.HADITH_INDEX_DIR, ignore_errors=True)

    quran_ready = (config.QURAN_INDEX_DIR / "index.faiss").exists()
    hadith_ready = (config.HADITH_INDEX_DIR / "index.faiss").exists() and all(p.exists() for p in config.HADITH_JSON.values())
    if quran_ready and hadith_ready:
        print(f"All indexes present in {config.DATA_DIR} -- nothing to build.")
        return
    if not quran_ready and not config.QURAN_CSV.exists():
        sys.exit(f"Missing {config.QURAN_CSV} -- it is needed to build the Quran index.")

    from quran_agent.data import load_stores

    start = time.time()
    print(f"Building missing indexes in {config.DATA_DIR} on {config.resolve_device()} "
          "(Quran: ~1 min on GPU; hadith: ~3 min on GPU, ~45 min on CPU)...", flush=True)
    last = {"line": ""}

    def progress(stage, fraction, detail=""):
        line = f"[{fraction:4.0%}] {stage}" + (f" - {detail}" if detail else "")
        if line != last["line"]:   # the download reports every second; skip repeats
            print(line, flush=True)
            last["line"] = line

    stores = load_stores(progress=progress)
    print(f"Done in {time.time() - start:.0f}s -- Quran vectors: {stores.quran.index.ntotal}, "
          f"hadith vectors: {stores.hadith.index.ntotal}")


if __name__ == "__main__":
    main()

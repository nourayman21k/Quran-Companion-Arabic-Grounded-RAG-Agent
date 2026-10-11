#!/bin/sh
set -e

# Build any missing FAISS index / hadith data in the mounted data/ folder (no-op when present).
python scripts/build_indexes.py

# server.py starts loading the agent at boot; the page shows a progress bar until it is ready
exec streamlit run server.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true

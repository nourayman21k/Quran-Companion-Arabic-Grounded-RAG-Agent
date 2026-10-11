FROM python:3.12-slim

# CUDA build of torch (RTX 50-series needs CUDA >= 12.8). For a CPU-only image:
#   docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    DATA_DIR=/app/data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "torch>=2.12.1" --index-url ${TORCH_INDEX_URL}
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY quran_agent ./quran_agent
COPY scripts ./scripts
COPY .streamlit ./.streamlit
COPY app.py server.py resources.py docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD curl -fs http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]

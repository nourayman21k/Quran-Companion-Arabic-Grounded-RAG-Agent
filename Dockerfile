FROM python:3.11-slim

WORKDIR /app

# Build deps needed for faiss-cpu / sentence-transformers wheels on some platforms
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# Bake your pre-built artifacts (FAISS index, chunks_cache.pkl, CSV, etc.) into the image.
# See the README for what belongs in ./data before building.
COPY data ./data

ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

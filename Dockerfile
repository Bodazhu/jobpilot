FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time so first request is fast
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Copy application code
COPY app.py config.py ./
COPY pipeline/ ./pipeline/
COPY models/ ./models/
COPY templates/ ./templates/
COPY static/ ./static/
COPY scripts/ ./scripts/

# Copy pre-built indexes and database (baked into image, no rebuild needed)
COPY data/jobpilot.db data/jobs.index data/faiss_job_ids.npy data/bm25_corpus.pkl ./data/

EXPOSE 8080
ENV PYTHONUNBUFFERED=1

# Use $PORT (Render injects this). Single worker to stay within 512MB free tier.
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120

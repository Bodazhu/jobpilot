FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so container starts instantly
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Copy application code
COPY app.py config.py ./
COPY pipeline/ ./pipeline/
COPY models/ ./models/
COPY templates/ ./templates/
COPY static/ ./static/
COPY scripts/ ./scripts/

# Copy pre-built data (DB + indexes baked into image)
# Run `python scripts/load_kaggle_data.py` locally first,
# then `docker build` to embed the indexes into the image.
COPY data/jobpilot.db data/jobs.index data/faiss_job_ids.npy data/bm25_corpus.pkl ./data/

EXPOSE 8080
ENV PORT=8080 PYTHONUNBUFFERED=1

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]

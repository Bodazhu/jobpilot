import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "jobpilot-dev-secret-2026")
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/jobpilot.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Embedding model
    EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
    EMBED_DIM = 384

    # FAISS index path
    FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", str(BASE_DIR / "data" / "jobs.index"))
    JOB_IDS_PATH = os.environ.get("JOB_IDS_PATH", str(BASE_DIR / "data" / "faiss_job_ids.npy"))

    # BM25 cache
    BM25_CACHE_PATH = str(BASE_DIR / "data" / "bm25_corpus.pkl")

    # Adzuna API (optional — for live streaming)
    ADZUNA_APP_ID = os.environ.get("ADZUNA_APP_ID", "")
    ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")

    # Anthropic API (for resume generation)
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

    # Ranking
    CANDIDATE_POOL = 500   # ANN returns this many before re-ranking
    TOP_K = 20             # Final results shown to user
    BM25_WEIGHT = 0.3
    DENSE_WEIGHT = 0.7

    # Streaming interval (seconds)
    STREAM_INTERVAL = 300  # 5 minutes

    # Data directory
    DATA_DIR = str(BASE_DIR / "data")

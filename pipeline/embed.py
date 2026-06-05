"""
Embedding engine.

BAX-423 Technique 1: BM25 (sparse retrieval)
BAX-423 Technique 2: Dense embeddings (BAAI/bge-small-en-v1.5 via fastembed) + FAISS ANN

fastembed uses ONNX runtime — same quality as sentence-transformers, no PyTorch needed.
"""

import json
import logging
import os
import pickle

import faiss
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from models.database import Job, db

logger = logging.getLogger(__name__)

# EMBED_DIM for BAAI/bge-small-en-v1.5 = 384
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

_encoder: TextEmbedding | None = None
_faiss_index: faiss.IndexFlatIP | None = None
_faiss_job_ids: list[int] = []
_bm25: BM25Okapi | None = None
_bm25_job_ids: list[int] = []


def get_encoder() -> TextEmbedding:
    global _encoder
    if _encoder is None:
        logger.info(f"Loading fastembed model: {EMBED_MODEL_NAME}")
        _encoder = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _encoder


def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of texts to L2-normalised float32 vectors."""
    enc = get_encoder()
    vecs = list(enc.embed(texts))
    arr = np.array(vecs, dtype=np.float32)
    # Normalise for cosine similarity via inner product
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (arr / norms).astype(np.float32)


def _job_text(job) -> str:
    skills = ""
    try:
        skills = " ".join(json.loads(job.skills or "[]"))
    except Exception:
        skills = str(job.skills or "")
    desc = (job.description or "")[:400]
    return f"{job.title} {job.company} {skills} {desc}".strip()


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(index_path: str, ids_path: str, batch_size: int = 256) -> int:
    index = faiss.IndexFlatIP(EMBED_DIM)

    all_jobs = db.session.query(
        Job.id, Job.title, Job.company, Job.skills, Job.description
    ).all()
    if not all_jobs:
        logger.warning("No jobs in database — FAISS index empty")
        return 0

    job_ids = [j.id for j in all_jobs]
    texts = []
    for j in all_jobs:
        skill_str = ""
        try:
            skill_str = " ".join(json.loads(j.skills or "[]"))
        except Exception:
            pass
        texts.append(f"{j.title} {j.company} {skill_str} {(j.description or '')[:400]}")

    logger.info(f"Encoding {len(texts)} jobs for FAISS index…")
    enc = get_encoder()

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs = list(enc.embed(batch))
        all_vecs.extend(vecs)
        if i % (batch_size * 10) == 0:
            logger.info(f"  encoded {i}/{len(texts)}")

    arr = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    arr = (arr / norms).astype(np.float32)

    index.add(arr)
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)
    np.save(ids_path, np.array(job_ids, dtype=np.int64))

    # Update faiss_idx in DB
    for i, jid in enumerate(job_ids):
        db.session.query(Job).filter_by(id=jid).update({"faiss_idx": i})
    db.session.commit()

    logger.info(f"FAISS index built: {index.ntotal} vectors → {index_path}")
    return index.ntotal


def load_faiss_index(index_path: str, ids_path: str):
    global _faiss_index, _faiss_job_ids
    if not os.path.exists(index_path):
        logger.warning(f"FAISS index not found at {index_path}")
        return
    _faiss_index = faiss.read_index(index_path)
    _faiss_job_ids = np.load(ids_path).tolist()
    logger.info(f"FAISS index loaded: {_faiss_index.ntotal} vectors")


def faiss_search(query_vec: np.ndarray, k: int = 500) -> list[tuple[int, float]]:
    if _faiss_index is None or _faiss_index.ntotal == 0:
        return []
    q = query_vec.reshape(1, -1).astype(np.float32)
    k = min(k, _faiss_index.ntotal)
    scores, indices = _faiss_index.search(q, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(_faiss_job_ids):
            results.append((_faiss_job_ids[idx], float(score)))
    return results


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def build_bm25_index(cache_path: str) -> int:
    global _bm25, _bm25_job_ids

    all_jobs = db.session.query(
        Job.id, Job.title, Job.company, Job.skills, Job.description
    ).all()
    if not all_jobs:
        return 0

    corpus = []
    ids = []
    for j in all_jobs:
        skill_str = ""
        try:
            skill_str = " ".join(json.loads(j.skills or "[]"))
        except Exception:
            pass
        text = f"{j.title} {j.company} {skill_str} {(j.description or '')[:300]}"
        corpus.append(text.lower().split())
        ids.append(j.id)

    _bm25 = BM25Okapi(corpus)
    _bm25_job_ids = ids

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"bm25": _bm25, "ids": ids}, f)

    logger.info(f"BM25 index built: {len(ids)} documents")
    return len(ids)


def load_bm25_index(cache_path: str):
    global _bm25, _bm25_job_ids
    if not os.path.exists(cache_path):
        logger.warning(f"BM25 cache not found: {cache_path}")
        return
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    _bm25 = data["bm25"]
    _bm25_job_ids = data["ids"]
    logger.info(f"BM25 index loaded: {len(_bm25_job_ids)} documents")


def bm25_search(query: str, k: int = 500) -> list[tuple[int, float]]:
    if _bm25 is None:
        return []
    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)
    top_k = min(k, len(scores))
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(int(_bm25_job_ids[i]), float(scores[i])) for i in top_indices if scores[i] > 0]
    return results


# ---------------------------------------------------------------------------
# Append new jobs to live FAISS index (after streaming)
# ---------------------------------------------------------------------------

def append_to_faiss(new_job_ids: list[int], index_path: str, ids_path: str):
    global _faiss_index, _faiss_job_ids
    if _faiss_index is None:
        return

    jobs = db.session.query(Job).filter(Job.id.in_(new_job_ids)).all()
    texts = [_job_text(j) for j in jobs]
    if not texts:
        return

    vecs = encode_texts(texts)
    _faiss_index.add(vecs)

    for i, j in enumerate(jobs):
        j.faiss_idx = len(_faiss_job_ids) + i
    db.session.commit()

    _faiss_job_ids.extend([j.id for j in jobs])
    faiss.write_index(_faiss_index, index_path)
    np.save(ids_path, np.array(_faiss_job_ids, dtype=np.int64))

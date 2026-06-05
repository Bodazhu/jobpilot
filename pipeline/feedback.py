"""
Adaptive learning via Rocchio Algorithm (relevance feedback).

After each accept/reject/skip round, the user's query vector is updated:

    q_new = α·q_orig + β·mean(accepted_vecs) − γ·mean(rejected_vecs)

This is the classic Rocchio formula from information retrieval.
With each round, the query vector drifts toward accepted jobs and away from
rejected ones, producing measurably better NDCG@10 over multiple rounds.
"""

import json
import logging

import numpy as np

from models.database import Feedback, Job, UserSession, db
from pipeline.embed import encode_texts, faiss_search

logger = logging.getLogger(__name__)

# Rocchio hyperparameters
ALPHA = 1.0   # weight on original query
BETA = 0.8    # weight on relevant centroid (accepted jobs)
GAMMA = 0.1   # weight on non-relevant centroid (rejected jobs)


def record_feedback(session_id: str, job_id: int, signal: int):
    """
    Persist a single feedback signal.
    signal: 1=accept, -1=reject, 0=skip
    """
    existing = db.session.query(Feedback).filter_by(
        session_id=session_id, job_id=job_id
    ).first()
    if existing:
        existing.signal = signal
    else:
        db.session.add(Feedback(session_id=session_id, job_id=job_id, signal=signal))
    db.session.commit()


def apply_rocchio(session_id: str, base_vector: np.ndarray | None = None) -> np.ndarray | None:
    """
    Retrieve all feedback for this session, compute Rocchio-updated query vector.
    Returns the updated (normalised) numpy vector or None if no feedback yet.
    """
    session = db.session.query(UserSession).filter_by(session_id=session_id).first()
    if session is None:
        return None

    # Load base (original) query vector
    if base_vector is None:
        if session.query_vector:
            base_vector = np.array(json.loads(session.query_vector), dtype=np.float32)
        else:
            return None

    feedbacks = db.session.query(Feedback).filter_by(session_id=session_id).all()
    if not feedbacks:
        return base_vector

    accepted_ids = [f.job_id for f in feedbacks if f.signal == 1]
    rejected_ids = [f.job_id for f in feedbacks if f.signal == -1]

    # Encode accepted / rejected job texts
    accepted_vecs = _get_job_vectors(accepted_ids)
    rejected_vecs = _get_job_vectors(rejected_ids)

    # Rocchio update
    q = ALPHA * base_vector

    if accepted_vecs:
        relevant_centroid = np.mean(accepted_vecs, axis=0)
        q = q + BETA * relevant_centroid

    if rejected_vecs:
        irrelevant_centroid = np.mean(rejected_vecs, axis=0)
        q = q - GAMMA * irrelevant_centroid

    # Normalise to unit length (required for cosine similarity with FAISS IndexFlatIP)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm

    # Persist updated vector
    session.adapted_vector = json.dumps(q.tolist())
    session.round_num = (session.round_num or 0) + 1
    db.session.commit()

    return q.astype(np.float32)


def _get_job_vectors(job_ids: list[int]) -> list[np.ndarray]:
    """Encode the text of given jobs and return their vectors."""
    if not job_ids:
        return []

    jobs = db.session.query(Job).filter(Job.id.in_(job_ids)).all()
    texts = []
    for j in jobs:
        skill_str = ""
        try:
            skill_str = " ".join(json.loads(j.skills or "[]"))
        except Exception:
            pass
        texts.append(f"{j.title} {j.company} {skill_str} {(j.description or '')[:300]}")

    if not texts:
        return []

    vecs = encode_texts(texts)
    return [vecs[i] for i in range(len(vecs))]


def get_adapted_vector(session_id: str) -> np.ndarray | None:
    """Return the currently adapted vector for this session, or None."""
    session = db.session.query(UserSession).filter_by(session_id=session_id).first()
    if session and session.adapted_vector:
        return np.array(json.loads(session.adapted_vector), dtype=np.float32)
    if session and session.query_vector:
        return np.array(json.loads(session.query_vector), dtype=np.float32)
    return None


def compute_improvement_metrics(session_id: str, profile: dict) -> dict:
    """
    Simulate improvement measurement.
    Compares NDCG@10 before and after Rocchio update.
    Returns a dict with round_num, before_ndcg, after_ndcg.
    """
    from pipeline.rank import rank_jobs, compute_ndcg

    session = db.session.query(UserSession).filter_by(session_id=session_id).first()
    if not session or not session.query_vector:
        return {"round": 0, "before_ndcg": 0, "after_ndcg": 0, "improvement": 0}

    orig_vec = np.array(json.loads(session.query_vector), dtype=np.float32)
    adapted_vec = get_adapted_vector(session_id)

    # "Ground truth" = jobs the user accepted
    accepted_ids = set(
        f.job_id for f in db.session.query(Feedback)
        .filter_by(session_id=session_id, signal=1).all()
    )

    if not accepted_ids:
        return {"round": session.round_num or 0, "before_ndcg": 0.0, "after_ndcg": 0.0, "improvement": 0.0}

    results_before = rank_jobs(orig_vec, profile, candidate_pool=200, top_k=10)
    results_after = rank_jobs(adapted_vec, profile, candidate_pool=200, top_k=10)

    ndcg_before = compute_ndcg(results_before, accepted_ids, k=10)
    ndcg_after = compute_ndcg(results_after, accepted_ids, k=10)
    improvement = ndcg_after - ndcg_before

    return {
        "round": session.round_num or 0,
        "before_ndcg": round(ndcg_before, 4),
        "after_ndcg": round(ndcg_after, 4),
        "improvement": round(improvement, 4),
    }

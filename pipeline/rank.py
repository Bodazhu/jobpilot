"""
Multi-stage ranking pipeline.

Stage 1 — Hard filters: dealbreakers, salary, location
Stage 2 — Candidate generation: BM25 + FAISS hybrid retrieval (top-K pool)
Stage 3 — Re-ranking: combine BM25 score + dense similarity + skill overlap
Stage 4 — Feedback boost: Rocchio-adjusted query vector improves results over rounds

Metric reported: NDCG@10 (computed in benchmark mode)
"""

import json
import logging
import re

import numpy as np
from sqlalchemy import or_

from models.database import Job, db
from pipeline.embed import bm25_search, encode_texts, faiss_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def build_query_text(profile: dict) -> str:
    """Flatten user profile into a single query string."""
    parts = []
    if profile.get("target_roles"):
        parts.append(profile["target_roles"])
    if profile.get("skills"):
        parts.append(profile["skills"])
    if profile.get("resume_snippet"):
        parts.append(profile["resume_snippet"][:300])
    return " ".join(parts)


def encode_query(profile: dict) -> np.ndarray:
    query_text = build_query_text(profile)
    vec = encode_texts([query_text])[0]
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 1 — Hard filters
# ---------------------------------------------------------------------------

_LEVEL_PATTERNS = {
    "senior": re.compile(r"\b(senior|sr\.?|lead|principal|staff|director|vp|head of)\b", re.I),
    "junior": re.compile(r"\b(junior|jr\.?|entry.?level|associate|intern)\b", re.I),
}


def _exceeds_exp_requirement(description: str, max_years: int | None) -> bool:
    """Return True if job requires more years experience than max_years."""
    if max_years is None:
        return False
    pattern = re.compile(r"(\d+)\+?\s*(?:years?|yrs?)[\s\w]{0,20}experience", re.I)
    for m in pattern.finditer(description):
        required = int(m.group(1))
        if required > max_years:
            return True
    return False


def apply_hard_filters(
    job_ids: list[int],
    profile: dict,
) -> list[int]:
    """
    Remove jobs that violate dealbreakers or hard constraints.
    Returns filtered list of job_ids.
    """
    if not job_ids:
        return []

    dealbreakers_raw = profile.get("dealbreakers", "")
    dealbreaker_terms = [t.strip().lower() for t in re.split(r"[,;\n]", dealbreakers_raw) if t.strip()]

    # Expand common category dealbreakers
    _DEFENSE_COMPANIES = {"lockheed", "raytheon", "saic", "booz allen", "leidos",
                          "northrop", "bae systems", "l3harris", "general dynamics"}
    if any(t in ("defense", "military", "defence") for t in dealbreaker_terms):
        dealbreaker_terms.extend(["clearance", "classified", "secret"])
        dealbreaker_terms.extend(_DEFENSE_COMPANIES)

    salary_min = profile.get("salary_min")  # USD annual
    location_pref = (profile.get("location") or "").lower().strip()
    block_senior = profile.get("block_senior", False)
    block_junior = profile.get("block_junior", False)
    max_exp_years = profile.get("max_experience_years")
    no_contract = profile.get("no_contract", False)
    no_small_companies = profile.get("no_small_companies", False)
    us_only = profile.get("us_only", False)

    jobs = db.session.query(
        Job.id, Job.title, Job.company, Job.location, Job.country,
        Job.salary_min, Job.salary_max, Job.description
    ).filter(Job.id.in_(job_ids)).all()

    job_map = {j.id: j for j in jobs}
    kept = []

    for jid in job_ids:
        j = job_map.get(jid)
        if j is None:
            continue

        title_lower = (j.title or "").lower()
        company_lower = (j.company or "").lower()
        location_lower = (j.location or "").lower()
        desc_lower = (j.description or "").lower()

        # Dealbreaker: block if any term appears in title/company/description
        blocked = False
        for term in dealbreaker_terms:
            if term and (term in title_lower or term in company_lower or term in desc_lower):
                blocked = True
                break
        if blocked:
            continue

        # Salary filter (skip if max salary is below target)
        if salary_min and j.salary_max and j.salary_max < salary_min * 0.8:
            continue

        # Senior/Junior title filters
        if block_senior and _LEVEL_PATTERNS["senior"].search(j.title or ""):
            continue
        if block_junior and _LEVEL_PATTERNS["junior"].search(j.title or ""):
            continue

        # Experience year cap
        if max_exp_years and _exceeds_exp_requirement(j.description or "", max_exp_years):
            continue

        # Contract filter (Persona 2, 4)
        if no_contract and re.search(r"\b(contract|freelance|1099|temp|temporary)\b", desc_lower):
            continue

        # US only filter
        if us_only and j.country and j.country.upper() not in ("US", "USA", ""):
            continue

        # Location preference (allow remote OR matching location)
        if location_pref and location_pref not in ("any", "anywhere", "remote"):
            if location_pref not in location_lower and "remote" not in location_lower:
                # soft filter — don't hard-block, just deprioritize
                pass  # handled in scoring

        kept.append(jid)

    return kept


# ---------------------------------------------------------------------------
# Stage 2+3 — Hybrid retrieval + re-ranking
# ---------------------------------------------------------------------------

def _normalise(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalise a score dict."""
    if not scores:
        return scores
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _skill_overlap(job_skills_json: str, profile_skills: str) -> float:
    """Jaccard overlap between job skills and user skills."""
    try:
        job_skills = set(s.lower() for s in json.loads(job_skills_json))
    except Exception:
        job_skills = set(job_skills_json.lower().split())
    user_skills = set(re.findall(r"\w+", profile_skills.lower()))
    if not job_skills or not user_skills:
        return 0.0
    intersection = job_skills & user_skills
    union = job_skills | user_skills
    return len(intersection) / len(union)


def rank_jobs(
    query_vector: np.ndarray,
    profile: dict,
    candidate_pool: int = 500,
    top_k: int = 20,
    bm25_weight: float = 0.3,
    dense_weight: float = 0.7,
) -> list[dict]:
    """
    Full ranking pipeline.
    Returns list of result dicts (job + scores + explanation).
    """
    # ---- Candidate generation ----
    query_text = build_query_text(profile)

    bm25_results = bm25_search(query_text, k=candidate_pool)
    dense_results = faiss_search(query_vector, k=candidate_pool)

    # Merge candidates
    candidate_ids = list(set([jid for jid, _ in bm25_results] + [jid for jid, _ in dense_results]))

    # ---- Hard filters ----
    candidate_ids = apply_hard_filters(candidate_ids, profile)
    if not candidate_ids:
        logger.warning("All candidates filtered out")
        return []

    # ---- Score maps ----
    bm25_scores = {jid: score for jid, score in bm25_results if jid in set(candidate_ids)}
    dense_scores = {jid: score for jid, score in dense_results if jid in set(candidate_ids)}

    bm25_norm = _normalise(bm25_scores)
    dense_norm = _normalise(dense_scores)

    # ---- Fetch job records ----
    jobs = db.session.query(Job).filter(Job.id.in_(candidate_ids)).all()
    job_map = {j.id: j for j in jobs}

    user_skills = profile.get("skills", "")
    location_pref = (profile.get("location") or "").lower().strip()

    results = []
    for jid in candidate_ids:
        j = job_map.get(jid)
        if j is None:
            continue

        b = bm25_norm.get(jid, 0.0)
        d = dense_norm.get(jid, 0.0)
        skill_score = _skill_overlap(j.skills or "[]", user_skills)

        # Location bonus
        loc_bonus = 0.0
        if location_pref and location_pref not in ("any", "anywhere"):
            if location_pref in (j.location or "").lower() or "remote" in (j.location or "").lower():
                loc_bonus = 0.05

        final_score = bm25_weight * b + dense_weight * d + 0.1 * skill_score + loc_bonus

        # Build explanation
        matched_skills = _get_matched_skills(j.skills or "[]", user_skills)
        reason = _build_explanation(j, profile, matched_skills, b, d, skill_score)

        results.append({
            "job": j,
            "score": round(final_score, 4),
            "bm25_score": round(b, 4),
            "dense_score": round(d, 4),
            "skill_score": round(skill_score, 4),
            "matched_skills": matched_skills,
            "explanation": reason,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def _get_matched_skills(skills_json: str, user_skills: str) -> list[str]:
    try:
        job_skills = [s.lower() for s in json.loads(skills_json)]
    except Exception:
        job_skills = skills_json.lower().split()
    user_set = set(re.findall(r"\w+", user_skills.lower()))
    return [s for s in job_skills if s in user_set][:8]


def _build_explanation(job: Job, profile: dict, matched_skills: list, bm25: float, dense: float, skill: float) -> str:
    parts = []

    if matched_skills:
        parts.append(f"Matches your skills: {', '.join(matched_skills[:5])}")

    target = (profile.get("target_roles") or "").lower()
    if target and any(t.strip() in (job.title or "").lower() for t in target.split(",")):
        parts.append("Title aligns with your target role")

    loc_pref = (profile.get("location") or "").lower()
    if loc_pref and loc_pref not in ("any", "anywhere"):
        if loc_pref in (job.location or "").lower():
            parts.append(f"Located in your preferred area ({job.location})")
        elif "remote" in (job.location or "").lower():
            parts.append("Remote position available")

    if job.salary_min and profile.get("salary_min"):
        if job.salary_min >= float(profile.get("salary_min", 0)) * 0.9:
            parts.append(f"Salary within range (≥${job.salary_min:,.0f})")

    if dense > 0.5:
        parts.append("Strong profile-to-job semantic match")
    elif dense > 0.3:
        parts.append("Good semantic similarity to your profile")

    if not parts:
        parts.append("Relevant to your target role and skills")

    return " · ".join(parts)


# ---------------------------------------------------------------------------
# NDCG benchmark
# ---------------------------------------------------------------------------

def compute_ndcg(results: list[dict], relevant_ids: set[int], k: int = 10) -> float:
    """Compute NDCG@k. relevant_ids = ground truth set of relevant job IDs."""
    def dcg(rels):
        return sum(r / np.log2(i + 2) for i, r in enumerate(rels[:k]))

    gains = [1 if r["job"].id in relevant_ids else 0 for r in results[:k]]
    ideal = sorted(gains, reverse=True)

    dcg_val = dcg(gains)
    idcg_val = dcg(ideal)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

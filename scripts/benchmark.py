"""
Benchmark BAX-423 Techniques: BM25 vs Dense (SBERT) vs Hybrid

Creates a synthetic test set by labelling jobs as relevant based on
skill overlap with a test profile, then computes Recall@10 and NDCG@10
for each retrieval method.

Usage:
    python scripts/benchmark.py
"""

import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEST_PROFILES = [
    {
        "name": "ML Engineer (Aisha)",
        "target_roles": "ML Engineer, Applied Scientist",
        "skills": "Python, PyTorch, scikit-learn, machine learning, deep learning",
        "location": "Remote",
        "salary_min": 140000,
        "dealbreakers": "defense",
        "block_senior": True,
    },
    {
        "name": "Data Analyst (Marcus)",
        "target_roles": "Data Analyst, BI Analyst",
        "skills": "Python, SQL, Tableau, R, analytics",
        "location": "Any",
        "salary_min": 80000,
        "dealbreakers": "",
    },
    {
        "name": "MLOps Engineer (Priya)",
        "target_roles": "MLOps Engineer, ML Platform Engineer",
        "skills": "Kubernetes, Kafka, Spark, Python, AWS, TensorFlow",
        "location": "NYC",
        "salary_min": 200000,
        "dealbreakers": "",
    },
]


def compute_recall_at_k(results, relevant_ids, k=10):
    retrieved = {r["job"].id for r in results[:k]}
    return len(retrieved & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0


def compute_ndcg_at_k(results, relevant_ids, k=10):
    def dcg(rels):
        return sum(r / np.log2(i + 2) for i, r in enumerate(rels[:k]))
    gains = [1 if r["job"].id in relevant_ids else 0 for r in results[:k]]
    ideal = sorted(gains, reverse=True)
    d = dcg(gains)
    i = dcg(ideal)
    return d / i if i > 0 else 0.0


def main():
    from app import app
    from models.database import Job
    from pipeline.embed import (build_bm25_index, build_faiss_index,
                                 encode_texts, faiss_search, load_bm25_index,
                                 load_faiss_index, bm25_search)
    from pipeline.rank import apply_hard_filters, encode_query, rank_jobs
    from config import Config

    with app.app_context():
        job_count = Job.query.count()
        if job_count == 0:
            logger.error("No jobs in database. Run load_kaggle_data.py first.")
            sys.exit(1)

        logger.info(f"Benchmarking on {job_count} jobs")

        # Load indexes
        load_faiss_index(Config.FAISS_INDEX_PATH, Config.JOB_IDS_PATH)
        load_bm25_index(Config.BM25_CACHE_PATH)

        print("\n" + "="*70)
        print("JOBPILOT — BAX-423 Technique Benchmark")
        print("BM25 (Sparse) vs Sentence-BERT+FAISS (Dense) vs Hybrid")
        print("="*70)

        all_rows = []

        for profile in TEST_PROFILES:
            print(f"\n--- Profile: {profile['name']} ---")

            query_vec = encode_query(profile)
            query_text = f"{profile['target_roles']} {profile['skills']}"

            # Ground truth: relevance via skill OR keyword overlap in title+description.
            # Uses a broad definition so real-world noisy datasets still yield labels.
            user_skills = set(s.lower().strip() for s in profile["skills"].split(","))
            # Also treat individual words in target roles as signal words
            role_words = set()
            for t in profile["target_roles"].split(","):
                role_words.update(w for w in t.lower().split() if len(w) > 3)

            all_jobs = Job.query.limit(5000).all()
            relevant_ids = set()
            for j in all_jobs:
                title_lower = (j.title or "").lower()
                desc_lower  = (j.description or "").lower()[:300]
                combined    = title_lower + " " + desc_lower

                # Skill overlap in skills field
                try:
                    job_skills = set(s.lower() for s in json.loads(j.skills or "[]"))
                except Exception:
                    job_skills = set()
                skill_match = len(job_skills & user_skills) >= 1

                # Keyword overlap in title/description
                keyword_match = any(w in combined for w in role_words) or \
                                any(s in combined for s in user_skills)

                if skill_match or keyword_match:
                    relevant_ids.add(j.id)

            if not relevant_ids:
                logger.warning(f"No relevant jobs found for {profile['name']} — skipping")
                continue

            logger.info(f"Ground truth: {len(relevant_ids)} relevant jobs")

            # ---- BM25 only ----
            bm25_raw = bm25_search(query_text, k=100)
            bm25_filtered = apply_hard_filters([jid for jid, _ in bm25_raw], profile)
            bm25_jobs = Job.query.filter(Job.id.in_(bm25_filtered[:10])).all()
            bm25_results = [{"job": j, "score": 1.0} for j in bm25_jobs]
            bm25_recall = compute_recall_at_k(bm25_results, relevant_ids, k=10)
            bm25_ndcg = compute_ndcg_at_k(bm25_results, relevant_ids, k=10)

            # ---- Dense only ----
            dense_raw = faiss_search(query_vec, k=100)
            dense_filtered = apply_hard_filters([jid for jid, _ in dense_raw], profile)
            dense_jobs = Job.query.filter(Job.id.in_(dense_filtered[:10])).all()
            dense_results = [{"job": j, "score": 1.0} for j in dense_jobs]
            dense_recall = compute_recall_at_k(dense_results, relevant_ids, k=10)
            dense_ndcg = compute_ndcg_at_k(dense_results, relevant_ids, k=10)

            # ---- Hybrid ----
            hybrid_results = rank_jobs(query_vec, profile, candidate_pool=200, top_k=10)
            hybrid_recall = compute_recall_at_k(hybrid_results, relevant_ids, k=10)
            hybrid_ndcg = compute_ndcg_at_k(hybrid_results, relevant_ids, k=10)

            print(f"  {'Method':<20} {'Recall@10':>12} {'NDCG@10':>12}")
            print(f"  {'-'*44}")
            print(f"  {'BM25 (sparse)':<20} {bm25_recall:>12.4f} {bm25_ndcg:>12.4f}")
            print(f"  {'SBERT+FAISS (dense)':<20} {dense_recall:>12.4f} {dense_ndcg:>12.4f}")
            print(f"  {'Hybrid (our system)':<20} {hybrid_recall:>12.4f} {hybrid_ndcg:>12.4f}")

            all_rows.append({
                "profile": profile["name"],
                "bm25_recall": bm25_recall, "bm25_ndcg": bm25_ndcg,
                "dense_recall": dense_recall, "dense_ndcg": dense_ndcg,
                "hybrid_recall": hybrid_recall, "hybrid_ndcg": hybrid_ndcg,
            })

        if all_rows:
            print("\n" + "="*70)
            print("SUMMARY (averages across profiles)")
            print("="*70)
            avg = lambda key: sum(r[key] for r in all_rows) / len(all_rows)
            print(f"  {'Method':<20} {'Recall@10':>12} {'NDCG@10':>12}")
            print(f"  {'-'*44}")
            print(f"  {'BM25':<20} {avg('bm25_recall'):>12.4f} {avg('bm25_ndcg'):>12.4f}")
            print(f"  {'SBERT+FAISS':<20} {avg('dense_recall'):>12.4f} {avg('dense_ndcg'):>12.4f}")
            print(f"  {'Hybrid':<20} {avg('hybrid_recall'):>12.4f} {avg('hybrid_ndcg'):>12.4f}")
            print()
            print("Include this table in your technical brief.")


if __name__ == "__main__":
    main()

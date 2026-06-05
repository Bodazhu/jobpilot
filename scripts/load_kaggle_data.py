"""
Load the Kaggle job postings dataset into the JobPilot database,
then build the FAISS + BM25 search indexes.

Usage:
    python scripts/load_kaggle_data.py --csv data/job_postings.csv --limit 50000
    python scripts/load_kaggle_data.py --csv data/job_postings.csv --rebuild-index

The Kaggle dataset:
    https://www.kaggle.com/datasets/techmap/international-job-postings-september-2021
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Load Kaggle data into JobPilot")
    parser.add_argument("--csv", required=True, help="Path to job postings CSV file")
    parser.add_argument("--limit", type=int, default=50_000, help="Max rows to load (default 50000)")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild FAISS+BM25 after loading")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        logger.error(f"CSV not found: {args.csv}")
        sys.exit(1)

    from app import app
    from config import Config
    from models.database import db
    from pipeline.embed import build_bm25_index, build_faiss_index
    from pipeline.ingest import load_kaggle_csv

    with app.app_context():
        db.create_all()

        logger.info(f"Loading CSV: {args.csv} (limit={args.limit})")
        result = load_kaggle_csv(args.csv, limit=args.limit)
        logger.info(f"Load complete: {result}")

        if args.rebuild_index or True:  # always rebuild after load
            from models.database import Job
            job_count = Job.query.count()
            logger.info(f"Building search indexes for {job_count} jobs…")

            n_bm25 = build_bm25_index(Config.BM25_CACHE_PATH)
            logger.info(f"BM25 index: {n_bm25} documents")

            n_faiss = build_faiss_index(Config.FAISS_INDEX_PATH, Config.JOB_IDS_PATH)
            logger.info(f"FAISS index: {n_faiss} vectors")

            logger.info("All indexes built. Application is ready.")


if __name__ == "__main__":
    main()

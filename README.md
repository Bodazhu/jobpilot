# JobPilot — Smart Job Matcher
**BAX-423 Big Data · Spring 2026 · UC Davis**

## Quick Start (Local)

```bash
# 1. Create virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (optional)
cp .env.example .env
# Edit .env to add ANTHROPIC_API_KEY and ADZUNA credentials

# 4. Load the Kaggle dataset and build search indexes
# Download from: https://www.kaggle.com/datasets/techmap/international-job-postings-september-2021
python scripts/load_kaggle_data.py --csv data/job_postings.csv --limit 50000

# 5. Run the application
python app.py
# → Open http://localhost:8080
```

## Benchmark (BAX-423 Techniques)

```bash
python scripts/benchmark.py
# Prints Recall@10 and NDCG@10 for BM25 vs Dense vs Hybrid
```

## Deploy to Google Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/jobpilot

# Deploy
gcloud run deploy jobpilot \
  --image gcr.io/PROJECT_ID/jobpilot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars ANTHROPIC_API_KEY=xxx,ADZUNA_APP_ID=xxx,ADZUNA_APP_KEY=xxx
```

**Note:** Mount a persistent volume or use Cloud SQL for the SQLite database in production.
The FAISS index and BM25 cache must also be available at startup — either bake them into the
image (after loading data) or mount from Cloud Storage.

## Architecture

```
User uploads resume (PDF) → pdfplumber extracts text
↓
Profile encoding → Sentence-BERT all-MiniLM-L6-v2 → 384-dim query vector
↓
Stage 1: Hard filters (dealbreakers, salary, location, title level)
Stage 2: BM25 candidate retrieval (top-500) + FAISS ANN retrieval (top-500)
Stage 3: Hybrid re-ranking: 0.3×BM25 + 0.7×dense + 0.1×skill_overlap
↓
User sees ranked results with match scores + explanations
↓
User accepts/rejects/skips → Rocchio algorithm updates query vector
↓
Re-ranked results with measurably better NDCG@10
↓
Generate tailored resume → Claude API (or template fallback)
↓
Download CSV of top matches
```

## BAX-423 Techniques

| Technique | Module | Lecture |
|---|---|---|
| BM25 (sparse retrieval) | `pipeline/embed.py` | Text Mining / IR |
| Sentence-BERT + FAISS (dense retrieval) | `pipeline/embed.py` | NLP / Embeddings |
| Rocchio relevance feedback (adaptive learning) | `pipeline/feedback.py` | Information Retrieval |
| Hybrid ranking pipeline | `pipeline/rank.py` | Ranking Systems |

## Core Capabilities

| # | Capability | Status |
|---|---|---|
| 1 | Job Data Ingestion & Streaming | ✅ Kaggle CSV + Adzuna API polling |
| 2 | Profile Intake & Skill Matching | ✅ PDF resume + form |
| 3 | Embedding-Based Retrieval | ✅ SBERT + FAISS |
| 4 | Multi-Stage Ranking Pipeline | ✅ Hard filters → BM25 → dense → re-rank |
| 5 | Adaptive Learning | ✅ Rocchio algorithm, NDCG@10 tracked |
| 6 | Download Top Jobs | ✅ CSV download |

## Test Personas

| Persona | Key Constraints | Validated |
|---|---|---|
| Aisha (ML Pivoter) | No senior titles, no defense | ✅ |
| Marcus (New Grad) | No 3+ yr req, no contract | ✅ |
| Priya (Senior MLOps) | No junior, NYC/remote, $200k+ | ✅ |
| Kenji (Visa-constrained) | US only, no contract, research-focused | ✅ |

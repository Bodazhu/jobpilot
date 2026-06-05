# JobPilot — Smart Job Matcher
**BAX-423 Big Data · Spring 2026 · UC Davis**

**Live Demo:** https://jobpilot-fhpv.onrender.com
**GitHub:** https://github.com/Bodazhu/jobpilot

---

## Setup and Run (Single Command)

### Prerequisites
- Python 3.9–3.13
- Git

### Step 1 — Clone and install

```bash
git clone https://github.com/Bodazhu/jobpilot.git
cd jobpilot
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --prefer-binary -r requirements.txt
```

> **Note on PyTorch:** Python 3.13 does not yet have PyTorch wheels.
> The app uses `fastembed` (ONNX-based) instead of `sentence-transformers`.
> No PyTorch is required. All other packages install cleanly.

### Step 2 — Load the Kaggle dataset and build search indexes

Place the Kaggle sample CSV at `data/job_postings.csv`, then:

```bash
python scripts/load_kaggle_data.py --csv data/job_postings.csv --limit 50000
```

This loads jobs, deduplicates, builds the FAISS index, and builds the BM25 index.
**One-time setup — takes ~15–30 min depending on CPU (embedding 50k documents).**

A pre-sampled 10k CSV from the required Kaggle dataset is included at
`data/jobpilot_kaggle_10k.csv`. To use it instead:

```bash
python scripts/load_kaggle_data.py --csv data/jobpilot_kaggle_10k.csv --limit 10000
```

### Step 3 — Run the application

```bash
python app.py
```

Open **http://localhost:8080** in your browser.

### Single-command shortcut (after indexes are built)

```bash
source venv/bin/activate && python app.py
```

---

## Environment Variables (Optional)

Copy `.env.example` to `.env` and fill in:

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Flask session signing (auto-generated default works) |
| `ANTHROPIC_API_KEY` | Enables Claude-powered resume generation (template fallback without it) |
| `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` | Enables live job streaming from Adzuna API |

---

## Run Benchmark (BAX-423 Technique Comparison)

```bash
python scripts/benchmark.py
```

Prints Recall@10 and NDCG@10 for BM25 vs Dense vs Hybrid across 3 test profiles.

---

## Project Structure

```
jobpilot/
├── app.py                    # Flask application (all routes)
├── config.py                 # Configuration
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container for deployment
├── render.yaml               # Render deployment config
├── pipeline/
│   ├── embed.py              # BM25 + FAISS/fastembed embeddings (BAX-423 Technique 1+2)
│   ├── rank.py               # Multi-stage ranking pipeline + NDCG metric
│   ├── feedback.py           # Rocchio adaptive learning
│   ├── ingest.py             # Kaggle + Adzuna data ingestion + deduplication
│   └── resume_gen.py         # Resume tailoring (Claude API or template)
├── models/database.py        # SQLAlchemy models (Job, UserSession, Feedback)
├── templates/                # Jinja2 HTML templates
├── scripts/
│   ├── load_kaggle_data.py   # Load dataset + build indexes
│   └── benchmark.py         # BAX-423 technique benchmark
└── data/
    └── jobpilot_kaggle_10k.csv  # 10k real Kaggle job postings (sample)
```

---

## Architecture

```
PDF resume upload → pdfplumber extraction → 384-dim profile vector (BAAI/bge-small-en-v1.5)
↓
Stage 1: Hard filters (dealbreakers, salary, seniority, location, experience cap)
Stage 2: BM25 candidate pool (top-500) + FAISS ANN retrieval (top-500)
Stage 3: Hybrid re-ranking (0.3×BM25 + 0.7×dense + 0.1×skill overlap)
↓
Rocchio relevance feedback updates query vector from accept/reject signals
↓
Results page with match explanations + CSV download + Generate Resume
```

---

## Six Required Capabilities

| # | Capability | Implementation |
|---|---|---|
| 1 | Job Data Ingestion & Streaming | Kaggle CSV + Adzuna API polling (APScheduler), hash-based dedup |
| 2 | Profile Intake & Skill Matching | PDF/text upload via pdfplumber, form for prefs/dealbreakers |
| 3 | Embedding-Based Retrieval | BAAI/bge-small-en-v1.5 (fastembed/ONNX) + FAISS IndexFlatIP |
| 4 | Multi-Stage Ranking Pipeline | Hard filters → BM25 → dense re-rank → hybrid score; NDCG@10 |
| 5 | Adaptive Learning | Rocchio algorithm (q_new = α·q_orig + β·accepted − γ·rejected) |
| 6 | Download Top Jobs | CSV with title/company/location/salary/URL/description |

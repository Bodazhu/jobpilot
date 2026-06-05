# Key AI Prompts — JobPilot

BAX-423 Final Project · Spring 2026
Tool: Claude Code (claude-sonnet-4-6)

Each entry describes the original prompt, its purpose, and what was modified before use.

---

## 1. System Architecture Design

**Prompt:**
"You are helping me build a BAX-423 Final Project called JobPilot. Design the simplest possible
architecture that satisfies all six required capabilities: job data ingestion with streaming and
deduplication, profile intake with PDF parsing, embedding-based retrieval using ANN search,
a multi-stage ranking pipeline with hard filters and a reported evaluation metric, adaptive
learning from user feedback, and a CSV download. The system must run as a single deployable
unit — no Kafka, no Spark clusters, no microservices. Use Flask, SQLite, and FAISS."

**Purpose:** Establish the overall system design before writing any code.

**Modifications:** Switched from sentence-transformers to fastembed (ONNX-based) after
discovering PyTorch has no wheels for Python 3.13. Added APScheduler for Adzuna streaming
instead of a separate worker process.

---

## 2. Multi-Stage Ranking Pipeline

**Prompt:**
"Implement a multi-stage ranking pipeline in Python for a job matching system.
Stage 1: hard filters — apply dealbreakers (keyword match on title/company/description),
salary floor, seniority level (block senior/junior via regex), experience year cap (regex on
description), contract/temp filter, US-only filter.
Stage 2: BM25 candidate pool (top-500) and FAISS dense retrieval (top-500), merge candidates.
Stage 3: hybrid re-ranking using 0.3×BM25_normalised + 0.7×dense_normalised + 0.1×skill_overlap.
For each result, generate a human-readable explanation of why it ranked where it did."

**Purpose:** Build the core ranking system that drives all four persona pass criteria.

**Modifications:** Added location bonus scoring, Jaccard skill overlap, and a defense-company
expansion rule that translates 'defense' dealbreaker into a blocklist of known company names
(SAIC, Raytheon, Lockheed, etc.) since the Kaggle dataset lacks category metadata.

---

## 3. Rocchio Adaptive Learning

**Prompt:**
"Implement the Rocchio relevance feedback algorithm for a FAISS-based job search system.
After each round of user accept/reject/skip signals, update the query vector using:
q_new = α·q_orig + β·mean(accepted_vecs) − γ·mean(rejected_vecs)  [α=1.0, β=0.8, γ=0.1]
Persist the updated vector to SQLite (serialised as JSON). Track NDCG@10 before and after
each round and display the improvement in the UI."

**Purpose:** Satisfy the adaptive learning requirement with a demonstrable, measurable improvement.

**Modifications:** Added per-session vector persistence in SQLite so the query vector survives
page reloads. Added `compute_improvement_metrics()` to surface NDCG before/after deltas in the
results page banner.

---

## 4. Techmap Kaggle Schema Parser

**Prompt:**
"Write a Python ingestion function for the Techmap 'international-job-postings-september-2021'
Kaggle dataset. The CSV has nested JSON fields stored as strings. Parse:
- position → {'name': 'Job Title', 'workType': '...'} → extract name
- orgCompany → {'nameOrg': 'Company Name', ...} → extract nameOrg
- orgAddress → {'addressLine': 'City, ST', 'countryCode': 'US', ...} → extract addressLine + countryCode
- orgTags → {'CATEGORIES': ['Sales', None], ...} → extract CATEGORIES as JSON skill list
- salary → {'text': 'Dollar 65,000/Year'} → parse to annual float (handle hourly/weekly/monthly)
Auto-detect the schema and fall back to generic column mapping for other CSV formats."

**Purpose:** Load real Kaggle data correctly — the nested JSON fields are not handled by
standard CSV column mappers.

**Modifications:** Added a `is_techmap` schema detection flag based on presence of `position`,
`orgCompany`, `orgAddress`, `orgTags` columns. Added a defense-company expansion in the
deduplication hash and ingestion pipeline.

---

## 5. Flask Session Architecture

**Prompt:**
"Design a Flask session management system for a stateless job matching app where each user
has: (1) a profile with uploaded resume text, skills, preferences, and dealbreakers;
(2) a 384-dim query vector stored as JSON; (3) a Rocchio-adapted vector updated per feedback
round; (4) a feedback history (accept/reject/skip per job). Use SQLite via SQLAlchemy.
Store the session ID in a Flask cookie; store all data server-side in a UserSession table."

**Purpose:** Enable multi-step workflow (upload → results → feedback → generate resume)
without losing state between page loads.

**Modifications:** Removed nested `with app.app_context()` from `get_or_create_session()` after
discovering it called `db.session.remove()` on exit, invalidating the outer request's DB session
and causing the Generate Resume route to always redirect. Fixed the `generate_resume` guard to
synthesise a resume from profile data when no PDF file was uploaded (persona quick-fill case).

---

## 6. Explainability Panel

**Prompt:**
"Build an explainability panel for a job detail page. For any recommended job, show:
(1) Semantic Match % — cosine similarity between query vector and job vector, normalised to
the top result's score. Use FAISS index.reconstruct() to get the job's exact embedding vector
without a top-k search limit.
(2) Keyword Match % — BM25 score normalised to the top BM25 result.
(3) Skill Overlap % — fraction of user skills found in the job description text.
(4) Overall Score % — 0.7×semantic + 0.3×keyword + 0.1×skill.
(5) Bullet-point explanations: matched skills, title alignment, location match, salary fit.
Extract skill matches from description text, not from structured orgTags (which are too broad
in the Kaggle dataset)."

**Purpose:** Satisfy the 'Explain feature' rubric requirement with quantitative scores
and human-readable reasoning.

**Modifications:** Used `index.reconstruct(faiss_idx)` to compute the exact dot product
for any job (avoids the top-k search limit that was returning score=0 for lower-ranked jobs).
Added token-level skill extraction as fallback when full skill phrases don't appear verbatim
in descriptions.

# Key AI Prompts — JobPilot

BAX-423 Final Project · Spring 2026

Each prompt below was used with Claude (claude-sonnet-4-6) via Claude Code.
The output was reviewed, modified, and integrated into the codebase.

---

## 1. Initial architecture planning
**Prompt:** "Design the simplest architecture for a job matching system that covers all 6 required capabilities (ingestion, profile intake, embedding retrieval, multi-stage ranking, adaptive learning, download). Must run as a single Flask app with SQLite and FAISS. No Kafka, no Spark."

**Purpose:** Establish the overall system design. Chose Flask+SQLite+FAISS over more complex alternatives to minimise deployment risk.

**Modifications:** Added APScheduler-based streaming (instead of a separate process) and Rocchio algorithm for adaptive learning (instead of a neural approach).

---

## 2. Rocchio adaptive learning implementation
**Prompt:** "Implement the Rocchio relevance feedback algorithm in Python for a FAISS-based job search system. The user provides accept/reject/skip signals. Update the query vector: q_new = α·q_orig + β·mean(accepted_vecs) − γ·mean(rejected_vecs). Track NDCG@10 before and after each round."

**Purpose:** Implement adaptive learning that's grounded in a BAX-423 IR technique and measurably improves results.

**Modifications:** Added per-session vector persistence in SQLite (serialised as JSON), and a `compute_improvement_metrics()` function that shows NDCG change in the UI.

---

## 3. Multi-stage ranking pipeline
**Prompt:** "Write a Python ranking pipeline: hard filters (dealbreakers, salary, seniority) → BM25 candidate pool → FAISS dense re-rank → hybrid score (BM25_weight=0.3, dense_weight=0.7) + skill overlap bonus. Include explainability: for each result, generate a human-readable explanation."

**Purpose:** Implement the core ranking logic required by capability 4 of the spec.

**Modifications:** Added location bonus scoring, Jaccard skill overlap metric, and per-persona dealbreaker patterns (regex for experience years, contract roles, company size).

---

## 4. Jinja2 UI design
**Prompt:** "Create Jinja2 templates styled with Tailwind CSS and Alpine.js for a job matching app. Design should be modern and professional, similar to Jobright.ai. Key pages: landing/upload, results with feedback buttons, job detail with explain feature, analytics with Chart.js."

**Purpose:** Build the frontend without a Node.js build step (no React/Vue).

**Modifications:** Added the demo persona quick-fill buttons on the landing page, live NDCG metrics display on the results page, and score breakdown bars per job card.

---

## 5. Resume tailoring prompt (used at runtime in the app)
**Prompt used at runtime:** "You are an expert resume writer. Tailor the following resume for the target job posting. [TARGET ROLE] [KEY SKILLS TO HIGHLIGHT] [JOB DESCRIPTION] [ORIGINAL RESUME]. Rewrite the summary, reorder bullets for relevance, emphasise key skills. Return only the tailored resume text, max 500 words."

**Purpose:** Generate a tailored resume for any selected job via the Claude API.

**Modifications:** Added a template-based fallback for when no API key is configured, so the feature always works in demo conditions.

---

## 6. Benchmark script
**Prompt:** "Write a Python benchmark comparing BM25, SBERT+FAISS, and hybrid retrieval. Use synthetic ground truth: jobs are 'relevant' if title matches target role AND has ≥1 skill overlap. Compute Recall@10 and NDCG@10 for each method across 3 test personas."

**Purpose:** Produce the quantitative technique comparison required for the technical brief.

**Modifications:** Added per-persona result tables and a summary row with averages.

"""
JobPilot — Smart Job Matcher
BAX-423 Final Project · Spring 2026
"""

import io
import json
import logging
import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import pdfplumber
from flask import (Flask, flash, jsonify, redirect, render_template,
                   request, send_file, session, url_for)
from sqlalchemy import func

from config import Config
from models.database import Feedback, Job, StreamLog, UserSession, db
from pipeline.embed import (build_bm25_index, build_faiss_index,
                              load_bm25_index, load_faiss_index)
from pipeline.feedback import (apply_rocchio, compute_improvement_metrics,
                                get_adapted_vector, record_feedback)
from pipeline.ingest import fetch_adzuna_jobs
from pipeline.rank import encode_query, rank_jobs
from pipeline.resume_gen import tailor_resume

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

# Jinja2 helpers
@app.template_filter("format_number")
def format_number(n):
    try:
        return "{:,}".format(int(n))
    except Exception:
        return str(n)


def _ensure_data_dir():
    os.makedirs(app.config["DATA_DIR"], exist_ok=True)


# ---------------------------------------------------------------------------
# Index loading (called at startup)
# ---------------------------------------------------------------------------

def load_indexes():
    load_faiss_index(app.config["FAISS_INDEX_PATH"], app.config["JOB_IDS_PATH"])
    load_bm25_index(app.config["BM25_CACHE_PATH"])
    logger.info("Search indexes loaded")


# ---------------------------------------------------------------------------
# Streaming scheduler
# ---------------------------------------------------------------------------

def start_streaming_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()

    def stream_job():
        with app.app_context():
            logger.info("Streaming: fetching from Adzuna…")
            fetch_adzuna_jobs(
                app.config["ADZUNA_APP_ID"],
                app.config["ADZUNA_APP_KEY"],
            )

    scheduler.add_job(stream_job, "interval",
                      seconds=app.config["STREAM_INTERVAL"],
                      id="adzuna_stream")
    scheduler.start()
    logger.info("Streaming scheduler started")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def get_or_create_session() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    sid = session["sid"]
    with app.app_context():
        if not UserSession.query.filter_by(session_id=sid).first():
            db.session.add(UserSession(session_id=sid))
            db.session.commit()
    return sid


def get_user_session(sid: str) -> UserSession | None:
    return UserSession.query.filter_by(session_id=sid).first()


# ---------------------------------------------------------------------------
# Routes — Landing / Profile Intake
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    job_count = Job.query.count()
    stream_logs = StreamLog.query.order_by(StreamLog.ran_at.desc()).limit(5).all()
    return render_template("index.html", job_count=job_count, stream_logs=stream_logs)


@app.route("/upload", methods=["POST"])
def upload():
    """Accept resume upload + profile form, store in session, redirect to results."""
    sid = get_or_create_session()
    user_sess = get_user_session(sid)

    # Parse resume PDF
    resume_text = ""
    resume_file = request.files.get("resume")
    if resume_file and resume_file.filename.endswith(".pdf"):
        try:
            with pdfplumber.open(resume_file) as pdf:
                for page in pdf.pages:
                    resume_text += (page.extract_text() or "") + "\n"
        except Exception as e:
            logger.error(f"PDF parse error: {e}")
            flash("Could not read PDF. Please try again.", "error")
            return redirect(url_for("index"))
    elif resume_file and resume_file.filename:
        # Plain text resume
        resume_text = resume_file.read().decode("utf-8", errors="ignore")

    # Build profile from form
    profile = {
        "name": request.form.get("name", ""),
        "email": request.form.get("email", ""),
        "target_roles": request.form.get("target_roles", ""),
        "skills": request.form.get("skills", ""),
        "location": request.form.get("location", ""),
        "salary_min": _safe_float(request.form.get("salary_min")),
        "dealbreakers": request.form.get("dealbreakers", ""),
        "block_senior": "block_senior" in request.form,
        "block_junior": "block_junior" in request.form,
        "no_contract": "no_contract" in request.form,
        "us_only": "us_only" in request.form,
        "max_experience_years": _safe_int(request.form.get("max_experience_years")),
        "resume_snippet": resume_text[:400],
    }

    # Encode query vector
    query_vec = encode_query(profile)

    user_sess.resume_text = resume_text[:5000]
    user_sess.profile_json = json.dumps(profile)
    user_sess.query_vector = json.dumps(query_vec.tolist())
    user_sess.adapted_vector = None  # reset on new upload
    user_sess.round_num = 0
    user_sess.updated_at = datetime.utcnow()
    db.session.commit()

    return redirect(url_for("results"))


# ---------------------------------------------------------------------------
# Routes — Results
# ---------------------------------------------------------------------------

@app.route("/results")
def results():
    sid = get_or_create_session()
    user_sess = get_user_session(sid)

    if not user_sess or not user_sess.profile_json:
        flash("Please upload your resume first.", "info")
        return redirect(url_for("index"))

    profile = json.loads(user_sess.profile_json)

    # Use adapted vector if available, else original
    query_vec = get_adapted_vector(sid)
    if query_vec is None and user_sess.query_vector:
        query_vec = np.array(json.loads(user_sess.query_vector), dtype=np.float32)

    if query_vec is None:
        flash("No profile found. Please start over.", "error")
        return redirect(url_for("index"))

    ranked = rank_jobs(
        query_vec, profile,
        candidate_pool=app.config["CANDIDATE_POOL"],
        top_k=app.config["TOP_K"],
        bm25_weight=app.config["BM25_WEIGHT"],
        dense_weight=app.config["DENSE_WEIGHT"],
    )

    # Mark feedback state for display
    feedback_map = {}
    feedbacks = Feedback.query.filter_by(session_id=sid).all()
    for f in feedbacks:
        feedback_map[f.job_id] = f.signal

    metrics = compute_improvement_metrics(sid, profile)

    return render_template(
        "results.html",
        results=ranked,
        profile=profile,
        feedback_map=feedback_map,
        metrics=metrics,
        round_num=user_sess.round_num or 0,
    )


# ---------------------------------------------------------------------------
# Routes — Job Detail
# ---------------------------------------------------------------------------

@app.route("/job/<int:job_id>")
def job_detail(job_id):
    job = Job.query.get_or_404(job_id)
    sid = get_or_create_session()
    user_sess = get_user_session(sid)

    profile = {}
    if user_sess and user_sess.profile_json:
        profile = json.loads(user_sess.profile_json)

    # Find matched skills
    user_skills = profile.get("skills", "")
    import re
    try:
        job_skills = json.loads(job.skills or "[]")
    except Exception:
        job_skills = []
    user_skill_set = set(re.findall(r"\w+", user_skills.lower()))
    matched = [s for s in job_skills if s.lower() in user_skill_set]
    unmatched = [s for s in job_skills if s.lower() not in user_skill_set]

    return render_template(
        "job_detail.html",
        job=job,
        profile=profile,
        matched_skills=matched[:10],
        unmatched_skills=unmatched[:10],
    )


# ---------------------------------------------------------------------------
# Routes — Feedback
# ---------------------------------------------------------------------------

@app.route("/feedback", methods=["POST"])
def feedback():
    sid = get_or_create_session()
    data = request.get_json(force=True)
    job_id = int(data.get("job_id"))
    signal = int(data.get("signal"))  # 1, -1, 0

    if signal not in (-1, 0, 1):
        return jsonify({"error": "invalid signal"}), 400

    record_feedback(sid, job_id, signal)

    # Run Rocchio update
    user_sess = get_user_session(sid)
    if user_sess and user_sess.query_vector:
        base_vec = np.array(json.loads(user_sess.query_vector), dtype=np.float32)
        apply_rocchio(sid, base_vec)

    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Routes — Resume Generation
# ---------------------------------------------------------------------------

@app.route("/generate_resume/<int:job_id>")
def generate_resume(job_id):
    job = Job.query.get_or_404(job_id)
    sid = get_or_create_session()
    user_sess = get_user_session(sid)

    if not user_sess or not user_sess.resume_text:
        flash("Please upload your resume first.", "info")
        return redirect(url_for("index"))

    profile = json.loads(user_sess.profile_json or "{}")
    import re
    try:
        job_skills = json.loads(job.skills or "[]")
    except Exception:
        job_skills = []
    user_skill_set = set(re.findall(r"\w+", profile.get("skills", "").lower()))
    matched = [s for s in job_skills if s.lower() in user_skill_set]

    tailored = tailor_resume(
        resume_text=user_sess.resume_text,
        job=job,
        profile=profile,
        matched_skills=matched,
        api_key=app.config.get("ANTHROPIC_API_KEY", ""),
    )

    return render_template(
        "resume_preview.html",
        job=job,
        resume_text=tailored,
        profile=profile,
    )


@app.route("/download_resume/<int:job_id>")
def download_resume(job_id):
    """Download tailored resume as a .txt file."""
    job = Job.query.get_or_404(job_id)
    sid = get_or_create_session()
    user_sess = get_user_session(sid)

    if not user_sess or not user_sess.resume_text:
        return jsonify({"error": "no resume"}), 400

    profile = json.loads(user_sess.profile_json or "{}")
    import re
    try:
        job_skills = json.loads(job.skills or "[]")
    except Exception:
        job_skills = []
    user_skill_set = set(re.findall(r"\w+", profile.get("skills", "").lower()))
    matched = [s for s in job_skills if s.lower() in user_skill_set]

    tailored = tailor_resume(
        resume_text=user_sess.resume_text,
        job=job,
        profile=profile,
        matched_skills=matched,
        api_key=app.config.get("ANTHROPIC_API_KEY", ""),
    )

    buf = io.BytesIO(tailored.encode("utf-8"))
    filename = f"resume_{job.title[:30].replace(' ', '_')}.txt"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="text/plain")


# ---------------------------------------------------------------------------
# Routes — Download top matches (CSV)
# ---------------------------------------------------------------------------

@app.route("/download_jobs")
def download_jobs():
    sid = get_or_create_session()
    user_sess = get_user_session(sid)

    if not user_sess or not user_sess.profile_json:
        return jsonify({"error": "no profile"}), 400

    profile = json.loads(user_sess.profile_json)
    query_vec = get_adapted_vector(sid)
    if query_vec is None and user_sess.query_vector:
        query_vec = np.array(json.loads(user_sess.query_vector), dtype=np.float32)

    ranked = rank_jobs(query_vec, profile, candidate_pool=300, top_k=50)

    rows = []
    for r in ranked:
        j = r["job"]
        rows.append({
            "Title": j.title,
            "Company": j.company,
            "Location": j.location,
            "Salary Min": j.salary_min,
            "Salary Max": j.salary_max,
            "Currency": j.salary_currency,
            "Match Score": r["score"],
            "URL": j.url,
            "Description": (j.description or "")[:300],
            "Skills": j.skills,
        })

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="jobpilot_matches.csv",
                     mimetype="text/csv")


# ---------------------------------------------------------------------------
# Routes — Analytics
# ---------------------------------------------------------------------------

@app.route("/analytics")
def analytics():
    total_jobs = Job.query.count()

    # Top skills
    from collections import Counter
    skill_counter = Counter()
    for (skills_json,) in db.session.query(Job.skills).limit(10000).all():
        try:
            for s in json.loads(skills_json or "[]"):
                if s and len(s) > 1:
                    skill_counter[s.lower()] += 1
        except Exception:
            pass
    top_skills = skill_counter.most_common(20)

    # Salary distribution
    salary_data = db.session.query(Job.salary_min).filter(
        Job.salary_min.isnot(None),
        Job.salary_min > 1000,
        Job.salary_min < 500000,
    ).limit(5000).all()
    salaries = [s[0] for s in salary_data]

    salary_buckets = {"<50k": 0, "50-80k": 0, "80-120k": 0, "120-160k": 0, "160-200k": 0, "200k+": 0}
    for s in salaries:
        if s < 50000:
            salary_buckets["<50k"] += 1
        elif s < 80000:
            salary_buckets["50-80k"] += 1
        elif s < 120000:
            salary_buckets["80-120k"] += 1
        elif s < 160000:
            salary_buckets["120-160k"] += 1
        elif s < 200000:
            salary_buckets["160-200k"] += 1
        else:
            salary_buckets["200k+"] += 1

    # Top locations
    loc_counts = db.session.query(Job.location, func.count(Job.id)).group_by(Job.location)\
        .order_by(func.count(Job.id).desc()).limit(15).all()

    # Stream stats
    stream_logs = StreamLog.query.order_by(StreamLog.ran_at.desc()).limit(10).all()

    return render_template(
        "analytics.html",
        total_jobs=total_jobs,
        top_skills=top_skills,
        salary_buckets=salary_buckets,
        loc_counts=loc_counts,
        stream_logs=stream_logs,
    )


# ---------------------------------------------------------------------------
# Routes — Admin / Index rebuild
# ---------------------------------------------------------------------------

@app.route("/admin/rebuild_index", methods=["POST"])
def rebuild_index():
    """Rebuild FAISS + BM25 indexes. Call this after loading new data."""
    job_count = Job.query.count()
    if job_count == 0:
        return jsonify({"error": "No jobs in database. Load data first."}), 400

    n_faiss = build_faiss_index(app.config["FAISS_INDEX_PATH"], app.config["JOB_IDS_PATH"])
    n_bm25 = build_bm25_index(app.config["BM25_CACHE_PATH"])

    load_faiss_index(app.config["FAISS_INDEX_PATH"], app.config["JOB_IDS_PATH"])
    load_bm25_index(app.config["BM25_CACHE_PATH"])

    return jsonify({"faiss": n_faiss, "bm25": n_bm25, "ok": True})


@app.route("/admin/stream_now", methods=["POST"])
def stream_now():
    """Manually trigger a streaming fetch from Adzuna."""
    result = fetch_adzuna_jobs(
        app.config["ADZUNA_APP_ID"],
        app.config["ADZUNA_APP_KEY"],
    )
    return jsonify(result)


@app.route("/admin/stats")
def admin_stats():
    return jsonify({
        "total_jobs": Job.query.count(),
        "kaggle_jobs": Job.query.filter_by(source="kaggle").count(),
        "adzuna_jobs": Job.query.filter_by(source="adzuna").count(),
        "sessions": UserSession.query.count(),
        "feedbacks": Feedback.query.count(),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val):
    try:
        return float(str(val).replace(",", "").replace("$", "").replace("k", "000").strip())
    except Exception:
        return None


def _safe_int(val):
    try:
        return int(val)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# App startup
# ---------------------------------------------------------------------------

with app.app_context():
    _ensure_data_dir()
    db.create_all()
    load_indexes()
    if app.config["ADZUNA_APP_ID"]:
        start_streaming_scheduler()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Job(db.Model):
    __tablename__ = "jobs"
    id = db.Column(db.Integer, primary_key=True)
    dedup_hash = db.Column(db.String(64), unique=True, nullable=False, index=True)
    title = db.Column(db.Text, nullable=False)
    company = db.Column(db.Text, default="")
    location = db.Column(db.Text, default="")
    country = db.Column(db.String(10), default="")
    salary_min = db.Column(db.Float, nullable=True)
    salary_max = db.Column(db.Float, nullable=True)
    salary_currency = db.Column(db.String(10), default="USD")
    salary_type = db.Column(db.String(20), default="annual")
    description = db.Column(db.Text, default="")
    skills = db.Column(db.Text, default="")  # JSON list stored as text
    url = db.Column(db.Text, default="")
    source = db.Column(db.String(50), default="kaggle")
    faiss_idx = db.Column(db.Integer, nullable=True)  # position in FAISS index
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "salary_currency": self.salary_currency,
            "description": self.description[:500] if self.description else "",
            "skills": self.skills,
            "url": self.url,
            "source": self.source,
        }


class UserSession(db.Model):
    __tablename__ = "user_sessions"
    session_id = db.Column(db.String(64), primary_key=True)
    resume_text = db.Column(db.Text, default="")
    profile_json = db.Column(db.Text, default="{}")  # JSON: roles, skills, location, salary, dealbreakers
    query_vector = db.Column(db.Text, nullable=True)  # JSON-serialized numpy array (base query)
    adapted_vector = db.Column(db.Text, nullable=True)  # JSON after Rocchio updates
    round_num = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Feedback(db.Model):
    __tablename__ = "feedback"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), db.ForeignKey("user_sessions.session_id"), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey("jobs.id"), nullable=False)
    signal = db.Column(db.Integer, nullable=False)  # 1=accept, -1=reject, 0=skip
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class StreamLog(db.Model):
    __tablename__ = "stream_log"
    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String(50))
    ingested = db.Column(db.Integer, default=0)
    duplicates = db.Column(db.Integer, default=0)
    errors = db.Column(db.Integer, default=0)
    ran_at = db.Column(db.DateTime, default=datetime.utcnow)

"""
Data ingestion pipeline.
Handles:
  - Loading the Kaggle CSV dataset
  - Streaming new postings from Adzuna API
  - Deduplication via SHA-256 hash of (title + company + location)
"""

import ast
import hashlib
import json
import logging
import re
import time
from datetime import datetime

import pandas as pd
import requests

from models.database import Job, StreamLog, db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nested-field parsers for the Techmap / Kaggle schema
# ---------------------------------------------------------------------------

def _parse_dict_field(raw) -> dict:
    """Safely parse a stringified dict (as stored in CSV) into a real dict."""
    if not raw or (isinstance(raw, float)):
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return ast.literal_eval(str(raw))
    except Exception:
        return {}


def _techmap_title(raw) -> str:
    """position → {'name': 'Account Executive Sr', 'workType': '...'}"""
    d = _parse_dict_field(raw)
    return d.get("name", "").strip() or str(raw)[:100]


def _techmap_company(raw) -> str:
    """orgCompany → {'nameOrg': 'Brinks', ...}"""
    d = _parse_dict_field(raw)
    return d.get("nameOrg", "").strip()


def _techmap_location(raw) -> tuple[str, str]:
    """orgAddress → {'addressLine': 'New York, NY', 'countryCode': 'US', ...}
    Returns (location_string, country_code)."""
    d = _parse_dict_field(raw)
    location = d.get("addressLine", "") or f"{d.get('city','')} {d.get('state','')}".strip()
    country = d.get("countryCode", "") or d.get("country", "")
    return location.strip(), country.strip()[:10]


def _techmap_skills(raw) -> str:
    """orgTags → {'CATEGORIES': ['Sales', None], 'WORK_TYPES': [...]}"""
    d = _parse_dict_field(raw)
    cats = d.get("CATEGORIES", []) or []
    skills = [str(s).strip() for s in cats if s and str(s).strip().lower() != "none"]
    return json.dumps(skills[:20])


def _techmap_salary(raw) -> tuple[float | None, float | None]:
    """salary → {'text': 'Dollar 65,000/Year'} or NaN
    Returns (salary_min, salary_max) as annualised USD floats."""
    if not raw or (isinstance(raw, float)):
        return None, None
    d = _parse_dict_field(raw)
    text = d.get("text", "") if isinstance(d, dict) else str(raw)
    if not text:
        return None, None

    nums = re.findall(r"[\d,]+", text)
    if not nums:
        return None, None

    amount = float(nums[0].replace(",", ""))
    text_lower = text.lower()
    if "hour" in text_lower:
        amount = round(amount * 2080)
    elif "week" in text_lower:
        amount = round(amount * 52)
    elif "month" in text_lower:
        amount = round(amount * 12)
    elif "day" in text_lower:
        amount = round(amount * 260)

    # If two numbers given (range), use both
    if len(nums) >= 2:
        hi = float(nums[1].replace(",", ""))
        if "hour" in text_lower:
            hi = round(hi * 2080)
        return amount, hi
    return amount, amount


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def make_dedup_hash(title: str, company: str, location: str) -> str:
    raw = f"{title.lower().strip()}|{company.lower().strip()}|{location.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Kaggle CSV loader
# ---------------------------------------------------------------------------

# Column name aliases — the Kaggle dataset uses varied naming
_COL_MAP = {
    "title": ["title", "jobtitle", "job_title", "position"],
    "company": ["company", "companyname", "company_name", "employer"],
    "location": ["location", "city", "joblocation", "job_location"],
    "country": ["country", "countrycode", "country_code"],
    "description": ["description", "jobdescription", "job_description", "body"],
    "skills": ["skills", "required_skills", "tags"],
    "salary_min": ["salary_from", "minsalary", "min_salary", "salary_min", "salary_low"],
    "salary_max": ["salary_to", "maxsalary", "max_salary", "salary_max", "salary_high"],
    "salary_currency": ["salary_currency", "currency"],
    "salary_type": ["salary_type", "pay_period", "salarytype"],
    "url": ["url", "link", "apply_url", "job_url", "applicationurl"],
}


def _resolve_col(df_cols: list, aliases: list) -> str | None:
    lower = {c.lower().replace(" ", "_"): c for c in df_cols}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def _extract(row, col: str | None, default=""):
    if col is None:
        return default
    val = row.get(col, default)
    if pd.isna(val):
        return default
    return str(val).strip()


def _safe_float(val):
    try:
        v = float(str(val).replace(",", "").replace("$", "").strip())
        return v if v > 0 else None
    except Exception:
        return None


def load_kaggle_csv(filepath: str, limit: int = 50_000, batch_size: int = 1000) -> dict:
    """
    Load job postings from a Kaggle CSV into the database.
    Auto-detects the Techmap schema (international-job-postings-september-2021)
    and falls back to generic column mapping for other datasets.
    Returns a summary dict with counts.
    """
    logger.info(f"Loading Kaggle CSV: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip", nrows=limit)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1", on_bad_lines="skip", nrows=limit)

    cols = set(df.columns)
    is_techmap = {"position", "orgCompany", "orgAddress", "orgTags"}.issubset(cols)
    logger.info(f"Schema detected: {'Techmap/Kaggle' if is_techmap else 'Generic'}")

    ingested = 0
    duplicates = 0
    errors = 0
    rows_to_insert = []
    existing_hashes = set(h[0] for h in db.session.query(Job.dedup_hash).all())

    if is_techmap:
        for _, row in df.iterrows():
            try:
                title    = _techmap_title(row.get("position"))
                company  = _techmap_company(row.get("orgCompany"))
                location, country = _techmap_location(row.get("orgAddress"))
                description = str(row.get("text") or "")
                skills   = _techmap_skills(row.get("orgTags"))
                sal_min, sal_max = _techmap_salary(row.get("salary") or row.get("salaryRaw"))
                url      = str(row.get("url") or "")
                source_cc = str(row.get("sourceCC") or "")

                if not title or title == "nan":
                    continue

                h = make_dedup_hash(title, company, location)
                if h in existing_hashes:
                    duplicates += 1
                    continue

                existing_hashes.add(h)
                rows_to_insert.append(Job(
                    dedup_hash=h,
                    title=title,
                    company=company,
                    location=location,
                    country=(country or source_cc)[:10],
                    description=description[:3000],
                    skills=skills,
                    salary_min=sal_min,
                    salary_max=sal_max,
                    salary_currency="USD",
                    salary_type="annual",
                    url=url,
                    source="kaggle",
                ))

                if len(rows_to_insert) >= batch_size:
                    db.session.bulk_save_objects(rows_to_insert)
                    db.session.commit()
                    ingested += len(rows_to_insert)
                    logger.info(f"Inserted batch — total so far: {ingested}")
                    rows_to_insert = []

            except Exception as e:
                errors += 1
                logger.debug(f"Row error: {e}")

    else:
        # Generic schema fallback (for synthetic data or other CSV formats)
        col = {k: _resolve_col(list(cols), v) for k, v in _COL_MAP.items()}

        for _, row in df.iterrows():
            try:
                title       = _extract(row, col["title"]) or "Untitled"
                company     = _extract(row, col["company"])
                location    = _extract(row, col["location"])
                country     = (_extract(row, col["country"])[:10]
                               if col["country"] else "")
                description = _extract(row, col["description"])
                skills_raw  = _extract(row, col["skills"])
                url         = _extract(row, col["url"])
                skills      = _parse_skills(skills_raw)
                salary_min  = _safe_float(_extract(row, col["salary_min"], 0))
                salary_max  = _safe_float(_extract(row, col["salary_max"], 0))
                currency    = _extract(row, col["salary_currency"], "USD")[:10]
                sal_type    = _extract(row, col["salary_type"], "annual")[:20]

                if sal_type and "hour" in sal_type.lower():
                    salary_min = salary_min * 2080 if salary_min else None
                    salary_max = salary_max * 2080 if salary_max else None
                    sal_type = "annual"

                h = make_dedup_hash(title, company, location)
                if h in existing_hashes:
                    duplicates += 1
                    continue

                existing_hashes.add(h)
                rows_to_insert.append(Job(
                    dedup_hash=h,
                    title=title,
                    company=company,
                    location=location,
                    country=country[:10],
                    description=description,
                    skills=skills,
                    salary_min=salary_min,
                    salary_max=salary_max,
                    salary_currency=currency,
                    salary_type=sal_type,
                    url=url,
                    source="kaggle",
                ))

                if len(rows_to_insert) >= batch_size:
                    db.session.bulk_save_objects(rows_to_insert)
                    db.session.commit()
                    ingested += len(rows_to_insert)
                    logger.info(f"Inserted batch — total so far: {ingested}")
                    rows_to_insert = []

            except Exception as e:
                errors += 1
                logger.debug(f"Row error: {e}")

    if rows_to_insert:
        db.session.bulk_save_objects(rows_to_insert)
        db.session.commit()
        ingested += len(rows_to_insert)

    logger.info(f"Kaggle load complete — ingested={ingested}, dupes={duplicates}, errors={errors}")
    return {"ingested": ingested, "duplicates": duplicates, "errors": errors}


def _parse_skills(raw: str) -> str:
    """Normalise skills string to JSON list."""
    if not raw or raw == "nan":
        return "[]"
    # Try JSON parse first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return json.dumps([str(s).strip() for s in parsed if s])
    except Exception:
        pass
    # Comma or semicolon separated
    skills = [s.strip() for s in re.split(r"[,;|]", raw) if s.strip()]
    return json.dumps(skills[:30])


# ---------------------------------------------------------------------------
# Adzuna streaming
# ---------------------------------------------------------------------------

def fetch_adzuna_jobs(app_id: str, app_key: str, country: str = "us",
                      results_per_page: int = 50, max_pages: int = 5) -> dict:
    """
    Fetch fresh job postings from Adzuna API.
    Returns summary of ingested/duplicates.
    """
    if not app_id or not app_key:
        logger.warning("Adzuna credentials not set — skipping stream fetch")
        return {"ingested": 0, "duplicates": 0, "errors": 0}

    ingested = 0
    duplicates = 0
    errors = 0

    existing_hashes = set(h[0] for h in db.session.query(Job.dedup_hash).all())

    for page in range(1, max_pages + 1):
        url = (
            f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
            f"?app_id={app_id}&app_key={app_key}"
            f"&results_per_page={results_per_page}&content-type=application/json"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Adzuna fetch error page {page}: {e}")
            errors += 1
            break

        for posting in data.get("results", []):
            try:
                title = posting.get("title", "Untitled")
                company = posting.get("company", {}).get("display_name", "")
                location = posting.get("location", {}).get("display_name", "")
                description = posting.get("description", "")
                salary_min = posting.get("salary_min")
                salary_max = posting.get("salary_max")
                apply_url = posting.get("redirect_url", "")

                h = make_dedup_hash(title, company, location)
                if h in existing_hashes:
                    duplicates += 1
                    continue

                existing_hashes.add(h)
                job = Job(
                    dedup_hash=h,
                    title=title,
                    company=company,
                    location=location,
                    country="us",
                    description=description,
                    skills="[]",
                    salary_min=float(salary_min) if salary_min else None,
                    salary_max=float(salary_max) if salary_max else None,
                    salary_currency="USD",
                    salary_type="annual",
                    url=apply_url,
                    source="adzuna",
                )
                db.session.add(job)
                ingested += 1
            except Exception as e:
                errors += 1
                logger.debug(f"Adzuna posting error: {e}")

        db.session.commit()
        time.sleep(0.5)  # be polite

    log = StreamLog(source="adzuna", ingested=ingested, duplicates=duplicates, errors=errors)
    db.session.add(log)
    db.session.commit()

    logger.info(f"Adzuna stream complete — ingested={ingested}, dupes={duplicates}")
    return {"ingested": ingested, "duplicates": duplicates, "errors": errors}

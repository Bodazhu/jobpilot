"""
Generate a small sample CSV of synthetic job postings for testing
without needing the full Kaggle dataset.

Usage:
    python scripts/generate_sample_data.py --output data/sample_jobs.csv --count 500
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TITLES = [
    "Machine Learning Engineer", "Senior ML Engineer", "Data Scientist",
    "Applied Scientist", "ML Platform Engineer", "MLOps Engineer",
    "Data Analyst", "BI Analyst", "Junior Data Scientist", "Analytics Engineer",
    "Research Scientist", "AI Engineer", "Computer Vision Engineer",
    "NLP Engineer", "Data Engineer", "Senior Data Scientist",
    "ML Infrastructure Engineer", "Deep Learning Engineer",
    "Staff ML Engineer", "Principal Data Scientist",
    "Software Engineer - ML", "AI Research Scientist",
    "Business Intelligence Developer", "Quantitative Analyst",
    "ML Research Engineer",
]

COMPANIES = [
    "Google", "Meta", "Amazon", "Microsoft", "Apple", "Netflix", "Uber",
    "Airbnb", "Stripe", "OpenAI", "Anthropic", "DeepMind", "NVIDIA", "Intel",
    "Salesforce", "Adobe", "Databricks", "Snowflake", "Palantir", "Waymo",
    "Accenture", "Deloitte", "JPMorgan Chase", "Goldman Sachs", "Citi",
    "Kaiser Permanente", "Mayo Clinic", "UnitedHealth", "CVS Health",
    "Scale AI", "Weights & Biases", "Hugging Face", "Cohere", "Together AI",
    "Lyft", "DoorDash", "Instacart", "Robinhood", "Plaid", "Brex",
    "Quora", "Reddit", "Pinterest", "Twitter", "LinkedIn", "Slack",
    "DataRobot", "C3.ai", "SambaNova", "Cerebras Systems",
    "Lockheed Martin", "Raytheon", "SAIC", "Booz Allen Hamilton",
]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
    "Boston, MA", "Chicago, IL", "Los Angeles, CA", "Remote",
    "San Jose, CA", "Denver, CO", "Atlanta, GA", "Washington, DC",
    "Menlo Park, CA", "Redmond, WA", "Cambridge, MA",
    "London, UK", "Toronto, Canada", "Berlin, Germany",
    "Remote (US)", "Hybrid - NYC", "Hybrid - SF Bay Area",
]

SKILLS_POOL = [
    "Python", "SQL", "PyTorch", "TensorFlow", "scikit-learn", "pandas",
    "numpy", "Spark", "Kafka", "Kubernetes", "Docker", "AWS", "GCP", "Azure",
    "R", "Tableau", "Power BI", "Java", "Scala", "C++", "CUDA",
    "NLP", "computer vision", "deep learning", "machine learning",
    "data engineering", "MLOps", "feature engineering", "A/B testing",
    "statistical modeling", "time series", "LLMs", "transformers",
    "Airflow", "dbt", "Snowflake", "Databricks", "Hadoop",
    "PySpark", "XGBoost", "LightGBM", "FAISS", "vector databases",
    "reinforcement learning", "recommender systems", "graph neural networks",
]

DESCRIPTIONS_TEMPLATES = [
    """We are looking for a {title} to join our team at {company}.

Responsibilities:
- Design and implement ML models for production systems
- Collaborate with data engineers and product teams
- Monitor and improve model performance in production
- Write clean, well-documented code

Requirements:
- {exp_years}+ years of experience in {core_skill}
- Strong proficiency in Python and SQL
- Experience with {skill1} and {skill2}
- Excellent communication skills

Nice to have:
- Experience with {skill3}
- {extra}

We offer competitive compensation, full benefits, and a collaborative environment.
""",
    """Join {company} as a {title}!

About the Role:
You will build and deploy ML systems that serve millions of users.

What You'll Do:
- Develop scalable ML pipelines
- Run experiments and analyse results
- Work with large datasets using {skill1}
- Deploy models using {skill2}

Requirements:
- {exp_years}+ years of ML/data experience
- Proficiency in Python and {core_skill}
- Familiarity with {skill3}
- Bachelor's or Master's in CS, Statistics, or related field

{extra}
""",
]

def make_job(i):
    title = random.choice(TITLES)
    company = random.choice(COMPANIES)
    location = random.choice(LOCATIONS)

    is_senior = any(w in title.lower() for w in ("senior", "staff", "principal", "lead"))
    is_junior = any(w in title.lower() for w in ("junior", "entry", "associate"))
    exp_years = random.choice([0, 1, 2, 3] if is_junior else [3, 4, 5, 6, 7] if is_senior else [1, 2, 3, 4])

    skills = random.sample(SKILLS_POOL, random.randint(4, 10))
    core_skill = skills[0]
    skill1, skill2, skill3 = skills[1], skills[2], skills[3] if len(skills) > 3 else skills[0]

    is_defense = "defense" in company.lower() or company in ("Lockheed Martin", "Raytheon", "SAIC", "Booz Allen Hamilton")
    is_contract = random.random() < 0.08  # 8% chance contract role

    extra = ""
    if is_contract:
        extra = "This is a contract position (6-month initial term)."
    elif random.random() < 0.3:
        extra = "We are an equal opportunity employer and welcome candidates from all backgrounds."
    elif is_defense:
        extra = "Active security clearance required. US citizenship mandatory."

    template = random.choice(DESCRIPTIONS_TEMPLATES)
    description = template.format(
        title=title, company=company, exp_years=exp_years,
        core_skill=core_skill, skill1=skill1, skill2=skill2, skill3=skill3,
        extra=extra,
    )

    base_salary = {
        "Data Analyst": 75000, "BI Analyst": 80000,
        "Junior Data Scientist": 90000, "Analytics Engineer": 95000,
        "Data Scientist": 130000, "Machine Learning Engineer": 145000,
        "ML Engineer": 145000, "Applied Scientist": 150000,
        "Senior ML Engineer": 180000, "MLOps Engineer": 160000,
        "ML Platform Engineer": 165000, "Research Scientist": 170000,
        "AI Engineer": 155000, "Staff ML Engineer": 200000,
        "Principal Data Scientist": 210000,
    }.get(title, 130000)

    noise = random.randint(-20000, 30000)
    salary_min = max(60000, base_salary + noise)
    salary_max = salary_min + random.randint(20000, 60000)

    url = f"https://jobs.example.com/posting/{i:06d}"

    return {
        "title": title,
        "company": company,
        "location": location,
        "country": "US" if not any(c in location for c in ("UK", "Canada", "Germany")) else location.split(", ")[-1][:5],
        "description": description,
        "skills": ", ".join(skills),
        "salary_from": salary_min,
        "salary_to": salary_max,
        "salary_currency": "USD",
        "salary_type": "contract" if is_contract else "annual",
        "url": url,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sample_jobs.csv")
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    jobs = [make_job(i) for i in range(args.count)]

    fieldnames = ["title", "company", "location", "country", "description",
                  "skills", "salary_from", "salary_to", "salary_currency",
                  "salary_type", "url"]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(jobs)

    print(f"Generated {args.count} sample jobs → {args.output}")
    print(f"Load with: python scripts/load_kaggle_data.py --csv {args.output}")


if __name__ == "__main__":
    main()

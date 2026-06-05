"""
Resume tailoring engine.

Given a user's resume text and a target job description, produces a
tailored resume that highlights the most relevant skills and experience.

Uses the Anthropic Claude API if a key is configured;
falls back to a simple template-based approach.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claude-powered generation
# ---------------------------------------------------------------------------

def generate_resume_claude(
    resume_text: str,
    job_title: str,
    job_description: str,
    skills_to_highlight: list[str],
    api_key: str,
) -> str:
    """Call Claude API to rewrite/tailor the resume."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        skills_str = ", ".join(skills_to_highlight[:10]) if skills_to_highlight else "relevant skills"

        prompt = f"""You are an expert resume writer. Tailor the following resume for the target job posting.

TARGET ROLE: {job_title}

KEY SKILLS TO HIGHLIGHT: {skills_str}

JOB DESCRIPTION (first 600 chars):
{job_description[:600]}

ORIGINAL RESUME:
{resume_text[:2000]}

Instructions:
1. Rewrite the summary/objective to match the target role.
2. Reorder bullet points so the most relevant experience appears first.
3. Add/emphasise the key skills listed above.
4. Keep the same sections (Summary, Experience, Education, Skills).
5. Return ONLY the tailored resume text, formatted cleanly.
6. Keep it to 1 page worth of content (≤500 words).
"""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    except Exception as e:
        logger.error(f"Claude resume generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Template-based fallback
# ---------------------------------------------------------------------------

def generate_resume_template(
    resume_text: str,
    job_title: str,
    job_description: str,
    skills_to_highlight: list[str],
    profile: dict,
) -> str:
    """
    Simple template resume tailored to the job.
    Used when Claude API is unavailable.
    """
    name = profile.get("name", "Candidate")
    email = profile.get("email", "")
    phone = profile.get("phone", "")
    location = profile.get("location", "")

    skills_all = profile.get("skills", "")
    skill_list = [s.strip() for s in re.split(r"[,;]", skills_all) if s.strip()]

    # Merge with job-matched skills
    highlighted = list(dict.fromkeys(skills_to_highlight + skill_list))[:12]
    skills_section = ", ".join(highlighted) if highlighted else skills_all

    # Extract experience bullets from original resume (if any)
    exp_section = _extract_experience(resume_text)

    summary = (
        f"Results-driven professional with experience in {', '.join(skills_to_highlight[:3]) if skills_to_highlight else 'relevant areas'}. "
        f"Seeking a {job_title} role where I can apply my background in {skills_section[:80]}."
    )

    lines = [
        f"{'='*60}",
        f"{name.upper()}",
        f"{email}  |  {phone}  |  {location}",
        f"{'='*60}",
        "",
        "PROFESSIONAL SUMMARY",
        "-"*40,
        summary,
        "",
        "CORE SKILLS",
        "-"*40,
        skills_section,
        "",
    ]

    if exp_section:
        lines += ["EXPERIENCE", "-"*40, exp_section, ""]

    lines += [
        "EDUCATION",
        "-"*40,
        _extract_education(resume_text) or "(See original resume for education details)",
        "",
        f"{'='*60}",
        f"[Resume tailored for: {job_title}]",
    ]

    return "\n".join(lines)


def _extract_experience(text: str) -> str:
    """Pull experience section from resume text."""
    match = re.search(
        r"(experience|work history|employment)[:\s]*(.*?)(?=education|skills|projects|certifications|$)",
        text, re.I | re.DOTALL
    )
    if match:
        return match.group(2).strip()[:800]
    return ""


def _extract_education(text: str) -> str:
    match = re.search(
        r"(education|academic)[:\s]*(.*?)(?=experience|skills|projects|$)",
        text, re.I | re.DOTALL
    )
    if match:
        return match.group(2).strip()[:300]
    return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def tailor_resume(
    resume_text: str,
    job,
    profile: dict,
    matched_skills: list[str],
    api_key: str = "",
) -> str:
    """
    Tailor resume for a specific job.
    Uses Claude if api_key provided, otherwise template fallback.
    """
    job_title = job.title if hasattr(job, "title") else str(job.get("title", ""))
    job_desc = job.description if hasattr(job, "description") else str(job.get("description", ""))

    if api_key:
        result = generate_resume_claude(resume_text, job_title, job_desc, matched_skills, api_key)
        if result:
            return result

    return generate_resume_template(
        resume_text, job_title, job_desc, matched_skills, profile
    )

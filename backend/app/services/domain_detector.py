"""
Domain Detector — identifies the primary professional domain from a resume.
Uses experience-weighted classification with LLM + fallback.
"""

import logging
from app.utils.llm_client import call_llm_json

logger = logging.getLogger("careerpilot.domain_detector")

_FALLBACK = {"domain": "General Professional"}


def detect_resume_domain(resume_text: str) -> dict:
    """
    Detect the primary professional domain from resume text.
    Returns {"domain": "<domain name>"}.
    Falls back to {"domain": "General Professional"} on failure.
    """
    if not resume_text or not resume_text.strip():
        logger.warning("detect_resume_domain called with empty text.")
        return _FALLBACK

    prompt = f"""
You are an expert ATS domain classification system.

Identify the SINGLE primary professional domain of this candidate
based on their overall career profile.

WEIGHTING RULES — apply strictly in this order:

1. Work Experience: 75% weight
   - Job titles, responsibilities, achievements, industry context
   - A candidate with 5+ years in a field = that field is the domain
     regardless of their educational background

2. Skills and Certifications: 20% weight
   - Domain-specific tools, platforms, and professional certifications

3. Education: 5% weight
   - Only tiebreaker when experience is minimal or absent
   - NEVER override a strong work history

CLASSIFICATION RULES:

- Return the MOST SPECIFIC domain possible, not a generic category.
- If the candidate has clearly transitioned careers, use their MOST RECENT field.
- If they work in a specialized sub-field, name the sub-field.

DOMAIN EXAMPLES (for calibration — not an exhaustive list):

Work Profile → Domain
───────────────────────────────────────────────
Software Engineer → Software Engineering
Backend Developer → Software Engineering
Frontend Developer → Software Engineering
ML Engineer → Artificial Intelligence / ML
Data Analyst → Data Analytics
Data Scientist → Data Science
DevOps Engineer → DevOps & Cloud Infrastructure
Cybersecurity Analyst → Cybersecurity
Regional Sales Manager (Insurance) → Insurance Sales
Area Sales Manager (Health Insurance) → Insurance Sales
Financial Advisor → Financial Services
Investment Analyst → Investment & Finance
HR Manager → Human Resources
Talent Acquisition Specialist → Talent Acquisition
Supply Chain Manager → Supply Chain & Logistics
Warehouse Operations Manager → Logistics & Warehouse Operations
Digital Marketing Manager → Digital Marketing
Brand Manager → Brand Management
Fashion Designer → Fashion Design
Merchandiser → Retail Merchandising
Mechanical Design Engineer → Mechanical Engineering
Civil Site Engineer → Civil Engineering
Doctor / Physician → Healthcare
Registered Nurse → Nursing & Healthcare
Lawyer / Advocate → Legal
Teacher / Lecturer → Education
Graphic Designer → Graphic Design
UI/UX Designer → UI/UX Design
Product Manager → Product Management
Business Analyst → Business Analysis
Operations Manager → Operations Management

Return ONLY valid JSON with no additional text.

JSON FORMAT:
{{
    "domain": "<specific domain name>"
}}

Resume:
{resume_text}
"""

    result = call_llm_json(
        prompt=prompt,
        fallback=_FALLBACK,
        temperature=0.1,
        max_tokens=100,
    )

    domain = result.get("domain", "").strip()
    if not domain:
        logger.warning("Domain detection returned empty domain. Using fallback.")
        return _FALLBACK

    logger.info("Detected domain: %s", domain)
    return {"domain": domain}

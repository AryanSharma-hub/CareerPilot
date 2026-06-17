"""
Resume Structurer — extracts structured metadata from raw resume text.
Used for ATS scoring and context building.
"""

import logging
from app.utils.llm_client import call_llm_json

logger = logging.getLogger("careerpilot.resume_structurer")

_FALLBACK = {
    "years_of_experience": "0",
    "technical_skills": [],
    "soft_skills": [],
    "certifications": [],
    "education": [],
    "projects": [],
    "tools": [],
}


def extract_resume_structure(resume_text: str) -> dict:
    """
    Extract structured metadata from resume text.

    Note: technical_skills and tools in this output are used for
    ATS scoring context. The definitive skill data comes from
    skill_extractor.py which runs a more rigorous two-pass pipeline.

    Returns structured dict with years_of_experience, technical_skills,
    soft_skills, certifications, education, projects, tools.
    """
    if not resume_text or not resume_text.strip():
        logger.warning("extract_resume_structure called with empty text.")
        return _FALLBACK.copy()

    prompt = f"""
You are an expert resume information extraction system.

Extract structured metadata from this resume for ATS scoring purposes.

FIELD DEFINITIONS:

years_of_experience:
Total professional experience in years as a number string.
Examples: "0", "2", "5", "10", "15+"
If unclear, estimate from job history dates.

technical_skills:
Domain-specific technical skills, methodologies, and professional
capabilities. Be comprehensive and domain-aware.
Examples by domain:
• Software: Machine Learning, REST API Design, Microservices, Agile
• Insurance/Sales: Agency Management, Territory Sales, Distribution
• Finance: Financial Modeling, Risk Management, Budgeting
• Marketing: SEO, Campaign Management, Brand Strategy
• Logistics: Supply Chain, Inventory Control, Route Planning

tools:
Software, platforms, applications, frameworks, databases, cloud
services, CRM systems, analytics tools, design suites.
Examples: Excel, SAP, Salesforce, AWS, TensorFlow, Figma, AutoCAD

soft_skills:
Interpersonal and behavioral traits only.
Examples: Communication, Leadership, Teamwork, Problem Solving

certifications:
Professional certifications, licenses, and completed courses.
List as strings.

education:
Degrees, institutions, and years if available.
List as strings.

projects:
Project titles or brief descriptions. Include personal, academic,
and professional projects.
List as strings.

RULES:
- Extract comprehensively — do NOT leave arrays empty if data exists.
- Be domain-aware in skill extraction.
- Estimate years_of_experience from job history if not stated.

Return ONLY valid JSON. No markdown, no explanation.

JSON FORMAT:
{{
    "years_of_experience": "",
    "technical_skills": [],
    "soft_skills": [],
    "certifications": [],
    "education": [],
    "projects": [],
    "tools": []
}}

Resume:
{resume_text}
"""

    result = call_llm_json(
        prompt=prompt,
        fallback=_FALLBACK.copy(),
        temperature=0.1,
        max_tokens=1500,
    )

    # Ensure all keys present
    for key, default in _FALLBACK.items():
        if key not in result:
            result[key] = default
        elif not isinstance(result[key], type(default)):
            result[key] = default

    logger.info(
        "Resume structured | years=%s | skills=%d | tools=%d | projects=%d",
        result.get("years_of_experience", "?"),
        len(result.get("technical_skills", [])),
        len(result.get("tools", [])),
        len(result.get("projects", [])),
    )

    return result

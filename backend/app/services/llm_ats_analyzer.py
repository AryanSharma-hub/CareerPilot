"""
LLM ATS Analyzer — evaluates resume quality from a recruiter's perspective.
Uses full resume text, no truncation, no artificial brevity constraints.
"""

import json
import logging
from app.utils.llm_client import call_llm_json

logger = logging.getLogger("careerpilot.llm_ats_analyzer")

_FALLBACK = {
    "ats_score": 60,
    "strengths": [],
    "weaknesses": [],
    "missing_skills": [],
    "improved_bullets": [],
    "suggestions": [],
}


def analyze_resume_with_llm(
    resume_text: str,
    domain: str,
    structured_data: dict,
) -> dict:
    if not resume_text or not resume_text.strip():
        logger.warning("analyze_resume_with_llm called with empty text.")
        return _FALLBACK.copy()

    structured_summary = json.dumps(structured_data, indent=2, ensure_ascii=False)

    prompt = f"""
You are an elite ATS evaluation system and senior recruiter specializing in {domain} hiring.

Evaluate this resume thoroughly and return a detailed, accurate assessment.

ATS SCORE CALIBRATION (0-100):
• 85-100: Exceptional — strong domain fit, quantified impact, comprehensive skills
• 70-84: Good — solid experience, some metrics, minor gaps
• 55-69: Moderate — relevant but thin on metrics or specific skills
• 40-54: Weak — limited depth or major gaps
• 0-39: Very weak

Structured Resume Data:
{structured_summary}

Full Resume:
{resume_text}

EVALUATION INSTRUCTIONS:
- strengths: 3-5 specific strengths referencing actual resume content
- weaknesses: 3-5 honest gaps or improvement areas specific to this resume
- missing_skills: up to 5 skills commonly expected in {domain} but absent here
- improved_bullets: 3-5 rewritten bullets with stronger action verbs (preserve all facts, never invent metrics)
- suggestions: 4-6 concrete actionable improvements

STRICT RULES:
- Reference actual content from the resume — no generic statements
- Never invent achievements, metrics, or certifications not in the resume
- Never say "lacks metrics" if metrics clearly exist in the resume
- Be domain-specific: evaluate through the lens of {domain} hiring standards

Return ONLY valid JSON, no markdown, no explanation:
{{
    "ats_score": 0,
    "strengths": [],
    "weaknesses": [],
    "missing_skills": [],
    "improved_bullets": [],
    "suggestions": []
}}
"""

    result = call_llm_json(
        prompt=prompt,
        fallback=_FALLBACK.copy(),
        temperature=0.2,
        max_tokens=3000,
    )

    # Validate score
    try:
        score = int(result.get("ats_score", 60))
        score = max(0, min(100, score))
    except (TypeError, ValueError):
        score = 60
    result["ats_score"] = score

    # Ensure all fields are lists
    for field in ("strengths", "weaknesses", "missing_skills", "improved_bullets", "suggestions"):
        if not isinstance(result.get(field), list):
            result[field] = []

    # Fallback strengths if LLM failed entirely
    if not result["strengths"]:
        skills = structured_data.get("technical_skills", [])
        years  = structured_data.get("years_of_experience", "")
        if years:
            result["strengths"].append(f"{years} years of professional {domain} experience")
        if skills:
            result["strengths"].append(f"Strong {domain} technical skill set detected")
        result["strengths"].append("Resume successfully parsed and analyzed")

    logger.info("LLM ATS analysis | score=%d | strengths=%d | weaknesses=%d",
                result["ats_score"], len(result["strengths"]), len(result["weaknesses"]))
    return result

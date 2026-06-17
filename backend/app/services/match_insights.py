"""
Match Insights — generates a recruiter-style summary of job match results.
Produces: strengths, concerns, and overall recommendation.
"""

import json
import logging
from app.utils.llm_client import call_llm_json

logger = logging.getLogger("careerpilot.match_insights")

_FALLBACK = {
    "summary": "Analysis complete.",
    "strengths": [],
    "concerns": [],
    "recommendation": "Moderate Candidate",
}


def generate_match_insights(
    resume_data: dict,
    jd_data: dict,
    match_result: dict,
) -> dict:
    """
    Generate a recruiter-style match summary.

    Returns:
        {
            "summary": str,
            "strengths": list[str],
            "concerns": list[str],
            "recommendation": str,  # "Strong Candidate" | "Good Candidate" | "Moderate Candidate" | "Weak Candidate"
        }
    """
    match_pct = match_result.get("match_percentage", 0)

    prompt = f"""
You are a senior recruiter reviewing a candidate's fit for a role.

Write a concise, honest recruiter-style assessment.

MATCH DATA:
{json.dumps(match_result, indent=2, ensure_ascii=False)}

CANDIDATE SKILLS:
{json.dumps(resume_data, indent=2, ensure_ascii=False)}

JOB REQUIREMENTS:
{json.dumps(jd_data, indent=2, ensure_ascii=False)}

Generate:

1. summary (1-2 sentences): Overall impression of the candidate's fit.

2. strengths (3-5 items): Specific areas where the candidate aligns well.
   Use actual skill names from the data. Be specific.

3. concerns (2-4 items): Genuine gaps or risks. Be honest.
   Only mention concerns that are evidenced by missing skills.
   Do NOT fabricate concerns.

4. recommendation: One of exactly these values:
   - "Strong Candidate" (match >= 75%)
   - "Good Candidate" (match 60-74%)
   - "Moderate Candidate" (match 45-59%)
   - "Weak Candidate" (match < 45%)

   Current match: {match_pct}%

Return ONLY valid JSON:
{{
    "summary": "",
    "strengths": [],
    "concerns": [],
    "recommendation": ""
}}
"""

    result = call_llm_json(
        prompt=prompt,
        fallback=_FALLBACK.copy(),
        temperature=0.2,
        max_tokens=600,
    )

    # Validate recommendation value
    valid_recs = {"Strong Candidate", "Good Candidate", "Moderate Candidate", "Weak Candidate"}
    if result.get("recommendation") not in valid_recs:
        if match_pct >= 75:
            result["recommendation"] = "Strong Candidate"
        elif match_pct >= 60:
            result["recommendation"] = "Good Candidate"
        elif match_pct >= 45:
            result["recommendation"] = "Moderate Candidate"
        else:
            result["recommendation"] = "Weak Candidate"

    for field in ("strengths", "concerns"):
        if not isinstance(result.get(field), list):
            result[field] = []

    logger.info("Match insights | rec=%s | match=%d%%", result["recommendation"], match_pct)
    return result

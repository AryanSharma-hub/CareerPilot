"""
Impact Analyzer — evaluates the measurable business and operational
impact demonstrated in a resume using LLM analysis.
"""

import logging
from app.utils.llm_client import call_llm_json

logger = logging.getLogger("careerpilot.impact_analyzer")

_FALLBACK = {
    "impact_score": 50,
    "impact_strengths": [],
    "impact_weaknesses": ["Unable to complete impact analysis."],
}


def analyze_resume_impact(resume_text: str, domain: str) -> dict:
    """
    Evaluate the measurable impact and quantified achievements in a resume.

    Returns:
        {
            "impact_score": int (0-100),
            "impact_strengths": list[str],
            "impact_weaknesses": list[str],
        }
    """
    if not resume_text or not resume_text.strip():
        logger.warning("analyze_resume_impact called with empty text.")
        return _FALLBACK.copy()

    prompt = f"""
You are an elite ATS impact evaluator specializing in {domain} hiring.

Evaluate the measurable business and operational impact of this resume.

SCORING GUIDE:

90–100: Exceptional — strong quantified achievements throughout.
Multiple metrics: revenue, savings, percentages, headcounts,
productivity gains, operational improvements.

75–89: Strong — several meaningful metrics and clear business value.
Impact is evident even if not always quantified.

60–74: Moderate — some measurable outcomes present. Impact exists
but could be better quantified.

40–59: Weak — mostly responsibilities listed, few outcomes.
Metrics rare or absent.

0–39: Very weak — purely task-based descriptions, no impact evidence.

WHAT COUNTS AS STRONG IMPACT (domain-adaptive):

Software/Tech: performance improvements %, systems built, users served,
uptime %, cost savings, deployment frequency

Sales/Insurance: revenue generated, targets achieved (%), team size managed,
portfolio growth, market expansion, new accounts, territory coverage

Finance: cost savings, ROI, budget managed, forecast accuracy,
audit findings resolved

Operations/Logistics: efficiency improvements %, cost reductions,
inventory accuracy, order fulfillment %, on-time delivery %

Marketing: campaign ROI, lead generation, conversion rates,
social reach, brand growth metrics

HR: hiring targets met, attrition reduction, training completion rates,
engagement scores

Fashion/Design: collections launched, product lines developed,
sales contribution, market reception

IMPORTANT RULES:
- If strong quantified metrics ARE present, score MUST reflect this.
  Do NOT give a low score when clear metrics exist.
- Be domain-aware: a fashion designer's "launched 3 seasonal collections"
  IS strong impact evidence even without percentages.
- NEVER hallucinate metrics or achievements.
- ONLY evaluate what is explicitly stated in the resume.

Return ONLY valid JSON. No markdown, no explanation.

JSON FORMAT:
{{
    "impact_score": 0,
    "impact_strengths": [],
    "impact_weaknesses": []
}}

Domain: {domain}

Resume:
{resume_text}
"""

    result = call_llm_json(
        prompt=prompt,
        fallback=_FALLBACK.copy(),
        temperature=0.15,
        max_tokens=800,
    )

    # ── Validate score ──
    try:
        score = int(result.get("impact_score", 50))
        score = max(0, min(100, score))
    except (TypeError, ValueError):
        score = 50

    strengths = result.get("impact_strengths", [])
    weaknesses = result.get("impact_weaknesses", [])

    if not isinstance(strengths, list):
        strengths = []
    if not isinstance(weaknesses, list):
        weaknesses = []

    # ── Calibration guard ──
    # If LLM says weak but found 3+ strengths, likely miscalibrated
    if score < 45 and len(strengths) >= 3:
        logger.info(
            "Impact calibration fired: score=%d with %d strengths → adjusting to 65",
            score, len(strengths)
        )
        score = 65

    # Prevent unrealistic perfect scores
    score = min(score, 95)

    logger.info("Impact analysis complete | score=%d", score)

    return {
        "impact_score": score,
        "impact_strengths": strengths,
        "impact_weaknesses": weaknesses,
    }

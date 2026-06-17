"""
Project Analyzer — evaluates project-equivalent work strength in a resume.
Domain-aware: understands that projects manifest differently across industries.

Bug fix: original code capped `score` locally but read back from `data`
before updating it, causing the cap to have no effect on the calibration check.
"""

import logging
from app.utils.llm_client import call_llm_json

logger = logging.getLogger("careerpilot.project_analyzer")

_FALLBACK = {
    "project_score": 50,
    "project_strengths": [],
    "project_weaknesses": ["Unable to complete project analysis."],
}


def analyze_project_strength(resume_text: str, domain: str) -> dict:
    """
    Evaluate the quality and depth of project-equivalent work in a resume.

    Returns:
        {
            "project_score": int (0-100),
            "project_strengths": list[str],
            "project_weaknesses": list[str],
        }
    """
    if not resume_text or not resume_text.strip():
        logger.warning("analyze_project_strength called with empty text.")
        return _FALLBACK.copy()

    prompt = f"""
You are an expert professional portfolio evaluator specializing in {domain}.

Evaluate the project-equivalent work demonstrated in this resume.

WHAT COUNTS AS PROJECT-EQUIVALENT WORK (domain-adaptive):

Software / Tech:
Applications built, APIs developed, ML models deployed, systems
designed, open-source contributions, hackathons, research implementations.

Fashion / Design:
Seasonal collections developed, product lines launched, brand
collaborations, design portfolios, creative campaigns.

Marketing:
Campaign launches, brand development projects, content strategies,
product launches, market entry initiatives.

Finance:
Financial models built, investment analyses conducted, audits led,
reporting systems implemented, fund portfolios managed.

Operations / Logistics:
Process optimization initiatives, warehouse improvement projects,
route optimization systems, ERP implementations, cost reduction programs.

Research / Academic:
Publications, research studies, thesis projects, experimental designs,
data collection and analysis projects.

HR:
Talent programs designed, onboarding systems built, policy frameworks
created, engagement initiatives launched, compensation structures developed.

SCORING GUIDE:

90–100: Exceptional — complex, high-ownership, measurable outcomes,
innovation, leadership, large scale or business impact.

75–89: Strong — clear project ownership with measurable business or
operational impact. Multiple significant deliverables.

60–74: Moderate — meaningful project involvement with some complexity
and defined outcomes.

40–59: Basic — project exposure with limited ownership or impact detail.

0–39: Minimal — little or no meaningful project-equivalent work evident.

EVALUATION DIMENSIONS:
• Ownership: Was the candidate the lead or a contributor?
• Complexity: How technically or operationally challenging was the work?
• Scale: What was the scope? (Team size, budget, users, geography)
• Outcomes: Were results measured and meaningful?
• Innovation: Did the work solve novel problems?

STRICT RULES:
- NEVER invent projects not present in the resume.
- If strong project-equivalent work is clearly present, score should reflect it.
- Be domain-aware — fashion collections are as valid as software projects.
- Evaluate honestly: a strong portfolio deserves 75+.

Return ONLY valid JSON. No markdown, no explanation.

JSON FORMAT:
{{
    "project_score": 0,
    "project_strengths": [],
    "project_weaknesses": []
}}

Domain: {domain}

Resume:
{resume_text}
"""

    result = call_llm_json(
        prompt=prompt,
        fallback=_FALLBACK.copy(),
        temperature=0.1,
        max_tokens=800,
    )

    # ── Validate score ──
    try:
        score = int(result.get("project_score", 50))
        score = max(0, min(100, score))
    except (TypeError, ValueError):
        score = 50

    strengths = result.get("project_strengths", [])
    weaknesses = result.get("project_weaknesses", [])

    if not isinstance(strengths, list):
        strengths = []
    if not isinstance(weaknesses, list):
        weaknesses = []

    # ── Calibration guard ──
    # Bug fix: operate on local `score` variable throughout,
    # NOT on data["project_score"] which hasn't been updated yet.
    if score < 45 and len(strengths) >= 3:
        logger.info(
            "Project calibration fired: score=%d with %d strengths → adjusting to 70",
            score, len(strengths)
        )
        score = 70

    # Cap at 95 (applied once, after calibration)
    score = min(score, 95)

    logger.info("Project analysis complete | score=%d", score)

    return {
        "project_score": score,
        "project_strengths": strengths,
        "project_weaknesses": weaknesses,
    }

"""
ATS Score Engine — rule-based component scorer.
Computes sub-scores for experience, skills, formatting, then combines
with LLM-derived impact and project scores into a final ATS score.
"""

import re
import logging

logger = logging.getLogger("careerpilot.ats_score_engine")


# ──────────────────────────────────────────────
# COMPONENT SCORES
# ──────────────────────────────────────────────

def calculate_component_scores(
    structured_data: dict,
    resume_text: str,
) -> dict:
    """
    Compute rule-based component scores from structured resume data.

    Components:
    - experience_score: based on years of experience
    - skills_score: based on breadth of technical skills
    - impact_score: placeholder (overwritten by impact_analyzer result)
    - project_score: placeholder (overwritten by project_analyzer result)
    - formatting_score: based on resume structure signals

    Returns dict of component scores (all 0–100).
    """

    # ── Experience Score ──
    experience_score = 50
    years_raw = str(structured_data.get("years_of_experience", "0"))
    try:
        # Extract first number found (handles "5 years", "5+", "5-7 years")
        digits = re.findall(r'\d+', years_raw)
        years_num = int(digits[0]) if digits else 0

        if years_num >= 10:
            experience_score = 95
        elif years_num >= 7:
            experience_score = 88
        elif years_num >= 5:
            experience_score = 82
        elif years_num >= 3:
            experience_score = 73
        elif years_num >= 1:
            experience_score = 62
        else:
            experience_score = 45

    except Exception:
        logger.warning("Could not parse years_of_experience: %s", years_raw)
        experience_score = 50

    # ── Skills Score ──
    # Quality-adjusted: rewards breadth but plateaus at reasonable values
    technical_skills = structured_data.get("technical_skills", [])
    tools = structured_data.get("tools", [])
    total_skill_signals = len(technical_skills) + len(tools)

    if total_skill_signals >= 20:
        skills_score = 92
    elif total_skill_signals >= 15:
        skills_score = 85
    elif total_skill_signals >= 10:
        skills_score = 76
    elif total_skill_signals >= 6:
        skills_score = 66
    elif total_skill_signals >= 3:
        skills_score = 55
    else:
        skills_score = 40

    # ── Formatting Score ──
    text_lower = resume_text.lower()
    formatting_score = 65  # Base

    word_count = len(resume_text.split())
    if word_count >= 500:
        formatting_score += 15
    elif word_count >= 300:
        formatting_score += 8
    elif word_count < 200:
        formatting_score -= 15

    # Section presence checks
    if any(kw in text_lower for kw in ("experience", "work history", "employment")):
        formatting_score += 5
    if any(kw in text_lower for kw in ("education", "degree", "qualification")):
        formatting_score += 3
    if any(kw in text_lower for kw in ("skills", "competencies")):
        formatting_score += 3
    if any(kw in text_lower for kw in ("certification", "certified", "licence")):
        formatting_score += 4

    # Metrics presence (including Indian formats)
    metrics_patterns = [
        r'\d+%',           # percentages
        r'\d+x\b',         # multipliers
        r'\$\d+',          # USD amounts
        r'₹\d+',           # INR amounts
        r'\d+\s*(?:lakh|crore|lakhs|crores)',  # Indian large numbers
        r'\d{1,3}(?:,\d{3})+',  # comma-formatted numbers
        r'\d+\+',          # X+ counts
    ]
    has_metrics = any(
        re.search(p, resume_text, re.IGNORECASE)
        for p in metrics_patterns
    )
    if has_metrics:
        formatting_score += 5

    formatting_score = max(30, min(95, formatting_score))

    # ── Placeholder scores (overwritten by LLM analyzers) ──
    # These are intentionally neutral until replaced.
    impact_score_placeholder = 60
    project_score_placeholder = 55

    scores = {
        "experience_score": experience_score,
        "skills_score": skills_score,
        "impact_score": impact_score_placeholder,
        "project_score": project_score_placeholder,
        "formatting_score": formatting_score,
    }

    logger.info("Component scores: %s", scores)
    return scores


# ──────────────────────────────────────────────
# FINAL WEIGHTED ATS SCORE
# ──────────────────────────────────────────────

def calculate_final_ats_score(scores: dict) -> int:
    """
    Combine component scores into a final weighted ATS score.

    Weights:
    - Experience: 25% (career depth)
    - Skills:     22% (technical breadth)
    - Impact:     25% (measurable outcomes)
    - Projects:   15% (initiative and ownership)
    - Formatting: 13% (ATS readability)

    Total: 100%
    """
    final = (
        scores.get("experience_score", 50) * 0.25
        + scores.get("skills_score", 50) * 0.22
        + scores.get("impact_score", 60) * 0.25
        + scores.get("project_score", 55) * 0.15
        + scores.get("formatting_score", 65) * 0.13
    )
    return max(0, min(100, round(final)))

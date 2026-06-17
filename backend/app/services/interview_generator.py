"""
Interview Generator v3 — 5-category personalized interview questions.

Categories: Behavioral, Technical, Leadership, Skill Gap, Role-Specific.
(Matches frontend's expected schema.)

Fixes over previous version:
- Schema corrected from 3 categories to the 5 the frontend expects
  (previous mismatch caused Leadership/Role-Specific cards to never render)
- max_tokens raised to 3000 and prompt shortened to avoid JSON truncation
  (previously could silently return empty fallback with 200 OK)
- Automatic retry with a simpler prompt if first attempt returns empty
- Guaranteed non-empty output via local fallback generation, so the UI
  never shows a blank "Interview Prep" tab even if the LLM fails twice
"""

import json
import logging
from app.utils.llm_client import call_llm_json

logger = logging.getLogger("careerpilot.interview_generator")

_FALLBACK = {
    "behavioral_questions": [],
    "technical_questions": [],
    "leadership_questions": [],
    "skill_gap_questions": [],
    "role_specific_questions": [],
}


def _total_questions(data: dict) -> int:
    return sum(
        len(data.get(k, [])) for k in _FALLBACK
        if isinstance(data.get(k), list)
    )


def _local_fallback_questions(
    resume_analysis: dict,
    job_match_analysis: dict,
    target_role: str,
) -> dict:
    """
    Locally generated fallback questions when the LLM produces nothing
    usable after retries. Ensures the UI always has content to show.
    """
    domain = (
        resume_analysis.get("domain")
        or resume_analysis.get("detected_domain", {}).get("domain")
        or "your field"
    )
    skills = resume_analysis.get("technical_skills", [])[:3]
    tools  = resume_analysis.get("tools", [])[:2]
    missing = job_match_analysis.get("missing_skills", [])[:2]

    skill_str = ", ".join(skills) if skills else "your core skills"
    tool_str  = ", ".join(tools) if tools else "the tools listed on your resume"

    behavioral = [
        "Tell me about a time you had to manage competing priorities under a tight deadline.",
        "Describe a situation where you had to adapt to a significant change at work.",
        "Walk me through a time you collaborated with a difficult stakeholder or team member.",
        f"Describe a project where you took ownership from start to finish in {domain}.",
    ]
    technical = [
        f"Walk me through how you applied {skill_str} in a recent project.",
        f"How have you used {tool_str} in your day-to-day work?",
        f"What does success look like in a {target_role} role, and how do you measure it?",
        f"Describe the most technically challenging problem you've solved in {domain}.",
    ]
    leadership = [
        "Tell me about a time you had to influence a decision without formal authority.",
        "How do you approach mentoring or developing junior team members?",
        "Describe how you've handled a disagreement within your team.",
    ]
    skill_gap = (
        [f"This role requires {m}. How would you approach building that capability "
         f"given your background?" for m in missing]
        or [f"What is one area of {domain} you're actively working to strengthen, and how?"]
    )
    role_specific = [
        f"What attracted you to this {target_role} role specifically?",
        f"What do you think are the biggest challenges facing someone in a {target_role} position today?",
        f"How would you prioritize your first 90 days in this {target_role} role?",
    ]

    return {
        "behavioral_questions": behavioral,
        "technical_questions": technical,
        "leadership_questions": leadership,
        "skill_gap_questions": skill_gap,
        "role_specific_questions": role_specific,
    }


def generate_interview_questions(
    resume_analysis: dict,
    job_match_analysis: dict,
    target_role: str,
) -> dict:
    """
    Generate personalized interview questions across 5 categories.

    Returns:
        {
            "behavioral_questions":     list[str],
            "technical_questions":      list[str],
            "leadership_questions":     list[str],
            "skill_gap_questions":      list[str],
            "role_specific_questions":  list[str],
        }
    """
    if not target_role or not target_role.strip():
        logger.warning("generate_interview_questions called with no target role.")
        return _FALLBACK.copy()

    resume_summary = json.dumps(resume_analysis or {}, indent=2, ensure_ascii=False)
    match_summary  = json.dumps(job_match_analysis or {}, indent=2, ensure_ascii=False)

    # ── Attempt 1: full prompt ──
    prompt = f"""
You are an elite technical interviewer and hiring expert.

Generate a personalized interview question set for: {target_role}

CANDIDATE DATA:
{resume_summary}

JOB MATCH DATA:
{match_summary}

Generate exactly 5 categories. Keep EACH question under 20 words to
keep the response concise.

1. BEHAVIORAL (3-4 questions): STAR-method questions based on the
   candidate's actual experience. Cover leadership/ownership, failure,
   collaboration, pressure.

2. TECHNICAL (3-4 questions): Test depth in skills the candidate lists.
   Ask how they used specific tools. Vary difficulty.

3. LEADERSHIP (3-4 questions): Based on experience level — team
   management, mentoring, decision making, conflict resolution.

4. SKILL_GAP (3-4 questions): Focus on skills listed as MISSING in job
   match data. Frame as growth/learning questions.

5. ROLE_SPECIFIC (3-4 questions): Questions a domain expert would ask
   for {target_role}. Must require real experience to answer well.

Return ONLY valid JSON, no markdown, no explanation:
{{
    "behavioral_questions": [],
    "technical_questions": [],
    "leadership_questions": [],
    "skill_gap_questions": [],
    "role_specific_questions": []
}}
"""

    result = call_llm_json(
        prompt=prompt,
        fallback=_FALLBACK.copy(),
        temperature=0.4,
        max_tokens=3000,
    )

    # ── Attempt 2: simplified retry if first attempt produced nothing ──
    if _total_questions(result) == 0:
        logger.warning("Interview generation attempt 1 returned empty — retrying with simpler prompt.")

        simple_prompt = f"""
You are an interviewer. Generate interview questions for: {target_role}

Candidate skills: {json.dumps(resume_analysis.get('technical_skills', [])[:8])}
Candidate tools: {json.dumps(resume_analysis.get('tools', [])[:5])}
Missing skills: {json.dumps(job_match_analysis.get('missing_skills', [])[:5])}

Generate 3 short questions (under 15 words each) per category.

Return ONLY this JSON:
{{
    "behavioral_questions": ["...", "...", "..."],
    "technical_questions": ["...", "...", "..."],
    "leadership_questions": ["...", "...", "..."],
    "skill_gap_questions": ["...", "...", "..."],
    "role_specific_questions": ["...", "...", "..."]
}}
"""
        result = call_llm_json(
            prompt=simple_prompt,
            fallback=_FALLBACK.copy(),
            temperature=0.3,
            max_tokens=1200,
        )

    # ── Validate structure ──
    validated = {}
    for field in _FALLBACK:
        value = result.get(field, [])
        if not isinstance(value, list):
            value = []
        validated[field] = [
            q.strip() for q in value
            if isinstance(q, str) and q.strip()
        ]

    # ── Attempt 3: guaranteed local fallback if still empty ──
    if _total_questions(validated) == 0:
        logger.warning("Interview generation failed twice — using local fallback questions.")
        validated = _local_fallback_questions(resume_analysis, job_match_analysis, target_role)

    logger.info(
        "Interview questions | role=%s | behavioral=%d | technical=%d | "
        "leadership=%d | skill_gap=%d | role_specific=%d",
        target_role,
        len(validated["behavioral_questions"]),
        len(validated["technical_questions"]),
        len(validated["leadership_questions"]),
        len(validated["skill_gap_questions"]),
        len(validated["role_specific_questions"]),
    )

    return validated

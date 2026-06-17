"""
Career Context Builder — builds chatbot context from resume + ATS + job match data.
Fully null-safe — no KeyError possible regardless of input shape.
"""

import logging

logger = logging.getLogger("careerpilot.career_context_builder")


def _fmt_list(items, fallback="Not available"):
    if not items or not isinstance(items, list):
        return fallback
    clean = [str(i).strip() for i in items if i]
    return ", ".join(clean) if clean else fallback


def _fmt_score(value, fallback="N/A"):
    if value is None:
        return fallback
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _extract_domain(resume_analysis, ats_analysis):
    domain = resume_analysis.get("domain", "")
    if domain:
        return domain
    detected = ats_analysis.get("detected_domain", {})
    if isinstance(detected, dict):
        domain = detected.get("domain", "")
    if domain:
        return domain
    return "Not detected"


def build_career_context(resume_analysis, ats_analysis, job_match_analysis):
    resume_analysis    = resume_analysis    or {}
    ats_analysis       = ats_analysis       or {}
    job_match_analysis = job_match_analysis or {}

    domain      = _extract_domain(resume_analysis, ats_analysis)
    tech_skills = _fmt_list(resume_analysis.get("technical_skills"))
    tools       = _fmt_list(resume_analysis.get("tools"))
    soft_skills = _fmt_list(resume_analysis.get("soft_skills"))

    ats_score   = _fmt_score(ats_analysis.get("ats_score"))
    strengths   = _fmt_list(ats_analysis.get("strengths"))
    weaknesses  = _fmt_list(ats_analysis.get("weaknesses"))
    suggestions = _fmt_list(ats_analysis.get("suggestions"))

    bd               = ats_analysis.get("score_breakdown") or {}
    exp_score        = _fmt_score(bd.get("experience_score"))
    skills_score     = _fmt_score(bd.get("skills_score"))
    impact_score     = _fmt_score(bd.get("impact_score"))
    project_score    = _fmt_score(bd.get("project_score"))
    formatting_score = _fmt_score(bd.get("formatting_score"))

    match_pct        = _fmt_score(job_match_analysis.get("match_percentage"))
    domain_alignment = job_match_analysis.get("domain_alignment", "Not analyzed")
    matched_skills   = _fmt_list(job_match_analysis.get("matched_skills"))
    missing_skills   = _fmt_list(job_match_analysis.get("missing_skills"))
    matched_tools    = _fmt_list(job_match_analysis.get("matched_tools"))
    missing_tools    = _fmt_list(job_match_analysis.get("missing_tools"))
    recommendations  = _fmt_list(job_match_analysis.get("recommendations"))

    return f"""
CANDIDATE DOMAIN: {domain}

TECHNICAL SKILLS: {tech_skills}
TOOLS: {tools}
SOFT SKILLS: {soft_skills}

ATS SCORE: {ats_score}/100
Score Breakdown:
• Experience: {exp_score}/100  • Skills: {skills_score}/100
• Impact: {impact_score}/100   • Projects: {project_score}/100
• Formatting: {formatting_score}/100

Strengths: {strengths}
Weaknesses: {weaknesses}
Suggestions: {suggestions}

JOB MATCH: {match_pct}%  |  Domain Alignment: {domain_alignment}
Matched Skills: {matched_skills}
Missing Skills: {missing_skills}
Matched Tools: {matched_tools}
Missing Tools: {missing_tools}
Recommendations: {recommendations}
""".strip()

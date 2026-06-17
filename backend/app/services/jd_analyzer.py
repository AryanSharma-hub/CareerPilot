"""
JD Analyzer — extracts structured hiring requirements from job descriptions.
Fully domain-agnostic — works for any industry or role.
"""

import json
import logging
from app.utils.llm_client import call_llm_json

logger = logging.getLogger("careerpilot.jd_analyzer")

_FALLBACK = {
    "domain": "",
    "experience_level": "",
    "technical_skills": [],
    "tools": [],
    "soft_skills": [],
    "responsibilities": [],
}


def _normalize_list(items: list, title_case: bool = True) -> list[str]:
    seen, result = set(), []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.replace("-", " ").strip()
        if title_case:
            cleaned = cleaned.title()
        key = cleaned.lower()
        if key and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def analyze_job_description(job_description: str) -> dict:
    if not job_description or not job_description.strip():
        return _FALLBACK.copy()

    # ── Pass 1: Full extraction ──
    primary_prompt = f"""
You are an expert AI job requirement analyzer.

Extract ALL structured hiring requirements from this job description.
Be thorough — extract every skill, competency, tool, and requirement mentioned
or clearly implied by the responsibilities listed.

FIELD DEFINITIONS:

domain:
The primary professional field this role belongs to.
Identify it from job title, responsibilities, and required skills.
Be specific: "Senior Software Engineering" not just "Technology", "Insurance Sales" not just "Sales", "Digital Marketing" not just "Marketing".
Other examples: Software Engineering, Data Science, Insurance Sales,
Digital Marketing, Supply Chain Management, Investment Banking,
Human Resources, Civil Engineering, Healthcare, Legal, Education.

experience_level:
Required seniority. Examples: Entry Level, 2-3 Years Experience,
Senior (5+ years), Manager Level, Director Level, Executive Level.

technical_skills:
ALL domain expertise, methodologies, professional competencies, and
operational knowledge required for this role.

CLASSIFICATION RULE — technical_skills includes:
- Any domain-specific knowledge or expertise (design skills, engineering skills,
  medical knowledge, legal knowledge, financial modeling, sales techniques, etc.)
- Business competencies (revenue management, territory management, vendor management,
  product development, project management, operations, etc.)
- Methodologies and processes (Agile, Six Sigma, IFRS, lean manufacturing, etc.)
- Industry-specific techniques and practices
- Anything that requires professional training or domain experience

technical_skills does NOT include:
- Named software products or platforms (those go in tools)
- Generic personality traits (those go in soft_skills)

tools:
ONLY named software, platforms, applications, frameworks, databases,
cloud services, or hardware systems explicitly mentioned in the JD.
A tool has a brand or product name.
Examples: Adobe Illustrator, Excel, SAP, Salesforce, AutoCAD, TensorFlow,
AWS, Docker, Jira, HubSpot, QuickBooks, Tableau, Figma, MATLAB.
Do NOT infer tools — only include what is explicitly stated.

soft_skills:
ONLY interpersonal and behavioral traits.
Examples: Leadership, Communication, Teamwork, Problem Solving,
Adaptability, Critical Thinking, Attention to Detail, Creativity,
Time Management, Negotiation, Collaboration.

responsibilities:
The primary job duties listed in the JD as concise action phrases.

EXTRACTION RULES:
1. Extract ALL competencies from both the Requirements AND Responsibilities sections
2. Responsibilities often imply technical skills — extract them
3. Do NOT hallucinate requirements not present in the JD
4. Do NOT limit extraction — be comprehensive
5. Classify every item into exactly one category

Return ONLY valid JSON, no markdown:
{{
    "domain": "",
    "experience_level": "",
    "technical_skills": [],
    "tools": [],
    "soft_skills": [],
    "responsibilities": []
}}

Job Description:
{job_description}
"""

    primary_data = call_llm_json(
        prompt=primary_prompt,
        fallback=_FALLBACK.copy(),
        temperature=0.1,
        max_tokens=2000,
    )

    for key in _FALLBACK:
        if key not in primary_data:
            primary_data[key] = _FALLBACK[key]

    # ── Pass 2: Classification cleanup ──
    norm_prompt = f"""
You are an ATS skill classifier. Review and fix any misclassifications below.

RULES:

technical_skills — domain expertise and professional competencies.
If something requires professional training or domain knowledge → technical_skills.
Business competencies like Sales Management, Revenue Generation, Product Development,
Vendor Management, Territory Management, Financial Modeling, Patient Care,
Case Management, Curriculum Development, Supply Chain, etc. → technical_skills.

tools — ONLY named software/platforms/products with a brand name.
If it has a recognizable product name → tools.
Do NOT put domain knowledge in tools.

soft_skills — ONLY personality/behavioral traits like Leadership, Communication,
Teamwork, Problem Solving, Adaptability, Critical Thinking.
Do NOT put business competencies or domain skills in soft_skills.

ACTIONS:
1. Move any named software/tools found in technical_skills → tools
2. Move any business competencies found in soft_skills → technical_skills
3. Remove duplicates across categories
4. Do NOT add new items not in the input

Return ONLY valid JSON:
{{
    "technical_skills": [],
    "tools": [],
    "soft_skills": []
}}

Input to classify:
{json.dumps({
    "technical_skills": primary_data.get("technical_skills", []),
    "tools": primary_data.get("tools", []),
    "soft_skills": primary_data.get("soft_skills", []),
})}
"""

    normalized = call_llm_json(
        prompt=norm_prompt,
        fallback={
            "technical_skills": primary_data.get("technical_skills", []),
            "tools": primary_data.get("tools", []),
            "soft_skills": primary_data.get("soft_skills", []),
        },
        temperature=0,
        max_tokens=1500,
    )

    final = {
        "domain":           primary_data.get("domain", "").strip(),
        "experience_level": primary_data.get("experience_level", "").strip(),
        "responsibilities": primary_data.get("responsibilities", []),
        "technical_skills": normalized.get("technical_skills", primary_data.get("technical_skills", [])),
        "tools":            normalized.get("tools", primary_data.get("tools", [])),
        "soft_skills":      normalized.get("soft_skills", primary_data.get("soft_skills", [])),
    }

    # Deduplicate: remove tools from technical_skills
    tool_set_lower = {t.lower() for t in final["tools"]}
    final["technical_skills"] = _normalize_list([
        s for s in final["technical_skills"]
        if s.lower() not in tool_set_lower
    ])
    final["tools"]       = _normalize_list(final["tools"], title_case=False)
    final["soft_skills"] = _normalize_list(final["soft_skills"])

    # Remove soft skills overlapping with technical
    tech_set_lower = {s.lower() for s in final["technical_skills"]}
    final["soft_skills"] = [
        s for s in final["soft_skills"]
        if s.lower() not in tech_set_lower
        and s.lower() not in tool_set_lower
    ]

    logger.info("JD analyzed | domain=%s | technical=%d | tools=%d | soft=%d",
                final["domain"], len(final["technical_skills"]),
                len(final["tools"]), len(final["soft_skills"]))

    return final

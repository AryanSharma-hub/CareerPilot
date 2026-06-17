"""
Competency Inference — infers interpersonal and behavioral soft skills
from work experience descriptions using LLM analysis.
"""

import logging
from app.utils.llm_client import call_llm_json

logger = logging.getLogger("careerpilot.competency_inference")

_FALLBACK = {"inferred_soft_skills": []}

# Canonical valid soft skills for post-processing validation
VALID_SOFT_SKILLS = {
    "communication", "leadership", "teamwork", "collaboration",
    "problem solving", "critical thinking", "creativity", "adaptability",
    "negotiation", "time management", "conflict resolution",
    "decision making", "customer focus", "attention to detail",
    "emotional intelligence", "mentoring", "coaching", "empathy",
    "active listening", "presentation skills", "stakeholder management",
    "cross-functional collaboration", "people management",
    "change management", "resilience", "initiative", "self-motivation",
}

# Items that must never appear as soft skills
INVALID_AS_SOFT_SKILLS = {
    "cash handling", "customer service", "record keeping",
    "inventory management", "sales", "fashion design", "market research",
    "product development", "relationship management", "territory management",
    "business development", "revenue generation", "warehousing",
    "supply chain", "operations", "agency management", "distribution",
    "insurance sales", "branch operations", "channel sales",
    "financial planning", "portfolio management", "recruitment",
    "sourcing", "billing", "invoicing", "procurement",
}


def infer_competencies(experience_text: str, domain: str) -> dict:
    """
    Infer soft skills from work experience content.

    Returns: {"inferred_soft_skills": ["Communication", "Leadership", ...]}
    """
    if not experience_text or not experience_text.strip():
        logger.info("Empty experience text — skipping competency inference.")
        return _FALLBACK

    prompt = f"""
You are an expert behavioral competency analyst.

Your task is to infer interpersonal and behavioral soft skills from
work experience descriptions. These are skills demonstrated through
HOW someone worked, not WHAT they worked on.

WHAT TO INFER — valid soft skills only:

Communication — presenting ideas, writing reports, coordinating teams,
client-facing roles, cross-functional communication.

Leadership — managing people, directing projects, mentoring juniors,
taking ownership of outcomes.

Teamwork / Collaboration — working across departments, partnering with
vendors, co-leading initiatives.

Problem Solving — resolving operational issues, troubleshooting,
finding workarounds, handling escalations.

Critical Thinking — evaluating options, making data-driven decisions,
analyzing performance gaps.

Time Management — managing multiple priorities, meeting deadlines,
handling concurrent projects.

Conflict Resolution — mediating disputes, handling complaints,
de-escalating situations.

Customer Focus — serving clients, improving satisfaction, handling
feedback.

Adaptability — working in changing environments, pivoting strategies,
learning new domains quickly.

Decision Making — making calls independently, owning outcomes.

Negotiation — vendor negotiations, contract discussions, salary talks.

Attention To Detail — auditing, quality control, compliance work.

WHAT NOT TO INFER:

Do NOT return operational functions or technical duties as soft skills.
These are NEVER valid soft skills:

Cash Handling, Customer Service, Record Keeping,
Inventory Management, Sales, Fashion Design, Market Research,
Product Development, Relationship Management, Territory Management,
Business Development, Revenue Generation, Warehousing,
Supply Chain, Operations, Insurance Sales, Distribution,
Agency Management, Branch Operations, Financial Planning,
Portfolio Management, Procurement, Billing, Invoicing

INFERENCE RULES:

- Only infer when clearly supported by evidence in the text.
- Do NOT infer from job titles alone.
- Do NOT infer from listed skills sections.
- Return ONLY the inferred skills as a list.
- A maximum of 8 skills.

EXAMPLES:

"Managed a team of 12 sales executives across 3 districts"
→ Leadership, People Management, Communication

"Collaborated with product, marketing, and logistics teams"
→ Cross-Functional Collaboration, Communication, Teamwork

"Resolved escalated customer complaints within 24 hours"
→ Conflict Resolution, Customer Focus, Problem Solving

"Handled 5 concurrent projects with strict delivery timelines"
→ Time Management, Adaptability

"Trained and onboarded 20 new recruits quarterly"
→ Mentoring, Communication, Leadership

Return ONLY valid JSON. No markdown, no explanation.

JSON FORMAT:
{{
    "inferred_soft_skills": []
}}

Domain: {domain}

Experience Content:
{experience_text}
"""

    result = call_llm_json(
        prompt=prompt,
        fallback=_FALLBACK.copy(),
        temperature=0.1,
        max_tokens=300,
    )

    raw_skills = result.get("inferred_soft_skills", [])
    if not isinstance(raw_skills, list):
        return _FALLBACK

    # ──────────────────────────────────────────
    # POST-PROCESS: validate and deduplicate
    # ──────────────────────────────────────────

    cleaned = []
    seen = set()

    for skill in raw_skills:
        if not isinstance(skill, str):
            continue
        normalized = skill.replace("-", " ").strip().title()
        key = normalized.lower()

        # Reject known invalid entries
        if key in INVALID_AS_SOFT_SKILLS:
            logger.debug("Rejected invalid soft skill: %s", skill)
            continue

        # Deduplicate
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(normalized)

    logger.info("Inferred %d soft skills from experience.", len(cleaned))
    return {"inferred_soft_skills": cleaned}

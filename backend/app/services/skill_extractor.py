"""
Skill Extractor v3 — universal domain-agnostic skill extraction.

Key design:
- LLM does all domain reasoning — no hardcoded skill lists
- Strict classification: Technical Skills = professional capabilities ONLY
  Achievements, KPIs, metrics, and one-time outcomes are excluded
- Domain exclusion prevents wrong tool assignments in non-tech resumes
- Validation is flexible to avoid rejecting legitimate inferred skills
"""

import re
import logging
from app.utils.llm_client import call_llm_json

logger = logging.getLogger("careerpilot.skill_extractor")

DOMAIN_EXCLUDED_TOOLS = {
    "fashion":         {"React", "Next.js", "Vue.js", "Angular", "Node.js",
                        "FastAPI", "Django", "Flask", "TensorFlow", "PyTorch",
                        "Docker", "Kubernetes", "Spring Boot", "Scikit-learn",
                        "Pandas", "NumPy"},
    "graphic design":  {"React", "Node.js", "FastAPI", "Django", "Flask",
                        "TensorFlow", "PyTorch", "Docker", "Kubernetes"},
    "interior design": {"React", "Node.js", "TensorFlow", "PyTorch",
                        "Docker", "Kubernetes"},
    "marketing":       {"TensorFlow", "PyTorch", "Docker", "Kubernetes", "Spring Boot"},
    "human resources": {"React", "TensorFlow", "PyTorch", "Docker", "Kubernetes"},
    "insurance":       {"React", "TensorFlow", "PyTorch", "Docker",
                        "Kubernetes", "FastAPI"},
    "sales":           {"React", "TensorFlow", "PyTorch", "Docker", "Kubernetes"},
    "healthcare":      {"React", "TensorFlow", "PyTorch", "Docker", "Kubernetes"},
    "education":       {"TensorFlow", "PyTorch", "Docker", "Kubernetes"},
    "legal":           {"React", "TensorFlow", "PyTorch", "Docker", "Kubernetes"},
    "finance":         {"React", "TensorFlow", "PyTorch", "Docker",
                        "Kubernetes", "FastAPI"},
    "accounting":      {"React", "TensorFlow", "PyTorch", "Docker", "Kubernetes"},
}

KNOWN_TOOLS: list[str] = [
    "Microsoft Office", "Excel", "Word", "PowerPoint", "Outlook",
    "Google Sheets", "Google Docs", "Google Analytics",
    "Power BI", "Tableau", "Looker", "QlikView",
    "Adobe Illustrator", "Photoshop", "InDesign", "Premiere Pro",
    "After Effects", "Figma", "Sketch", "Canva", "CorelDRAW",
    "AutoCAD", "SolidWorks", "Revit", "CATIA", "3ds Max", "Blender", "CAD",
    "SAP", "SAP ERP", "Oracle", "Salesforce", "HubSpot",
    "Zoho CRM", "Microsoft Dynamics", "ServiceNow", "Workday",
    "NetSuite", "Tally", "QuickBooks",
    "AWS", "Azure", "GCP", "Google Cloud",
    "Docker", "Kubernetes", "Git", "GitHub", "GitLab", "Jenkins", "Linux",
    "React", "Next.js", "Vue.js", "Angular", "Node.js",
    "FastAPI", "Django", "Flask", "Spring Boot",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy",
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
    "Jira", "Asana", "Trello", "Monday.com", "Notion", "Slack",
    "MailChimp", "Hootsuite", "SEMrush", "Ahrefs", "Google Ads",
    "MATLAB", "Stata", "SPSS",
]

_FALLBACK = {"technical_skills": [], "soft_skills": [], "tools": []}


def _get_excluded_tools(domain: str) -> set:
    domain_lower = domain.lower()
    for category, excluded in DOMAIN_EXCLUDED_TOOLS.items():
        if category in domain_lower:
            return excluded
    return set()


def _detect_tools_by_keyword(resume_text: str, domain: str) -> list[str]:
    """
    Detect tools by keyword scan.
    Tools with >= 4 characters: substring match (safe for multi-char names).
    Tools with < 4 characters: word-boundary match to avoid false positives
    (e.g. a 1-char tool name matching every word in the text).
    """
    import re as _re
    text_lower = resume_text.lower()
    excluded   = _get_excluded_tools(domain)
    found = []
    for tool in KNOWN_TOOLS:
        if tool in excluded:
            continue
        tl = tool.lower()
        if len(tl) < 4:
            if _re.search(r'\b' + _re.escape(tl) + r'\b', text_lower):
                found.append(tool)
        else:
            if tl in text_lower:
                found.append(tool)
    return found


def _validate_against_resume(skills: list[str], resume_text: str) -> list[str]:
    """
    Validate extracted skills have textual evidence in the resume.
    Single-word: strict word boundary.
    2-3 word: full phrase OR any one key word present.
    4+ word: 40%+ key word coverage.
    """
    text_lower = resume_text.lower()
    validated  = []
    for skill in skills:
        if not isinstance(skill, str) or not skill.strip():
            continue
        skill_lower = skill.lower().strip()
        words       = skill_lower.split()
        if len(words) == 1:
            if re.search(r'\b' + re.escape(words[0]) + r'\b', text_lower):
                validated.append(skill)
        elif len(words) <= 3:
            if skill_lower in text_lower:
                validated.append(skill)
            else:
                key_words = [w for w in words if len(w) > 3]
                if any(re.search(r'\b' + re.escape(w) + r'\b', text_lower)
                       for w in key_words):
                    validated.append(skill)
        else:
            key_words = [w for w in words if len(w) > 3]
            if not key_words:
                continue
            match_ratio = sum(
                1 for w in key_words
                if re.search(r'\b' + re.escape(w) + r'\b', text_lower)
            ) / len(key_words)
            if match_ratio >= 0.4:
                validated.append(skill)
    return validated


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


def extract_skills_llm(resume_text: str, domain: str) -> dict:
    """
    Extract technical skills, tools, and soft skills from resume text.
    Domain-adaptive via LLM — no hardcoded skill lists.
    Achievements and metrics are explicitly excluded from skill lists.
    """
    if not resume_text or not resume_text.strip():
        return _FALLBACK.copy()

    excluded = _get_excluded_tools(domain)
    excluded_note = (
        f"\nDo NOT include these tools for {domain} unless explicitly written: "
        + ", ".join(sorted(excluded))
    ) if excluded else ""

    prompt = f"""
You are an expert ATS resume skill extractor.

The candidate works in: {domain}

Extract professional skills from this resume — both explicitly listed
AND clearly demonstrated through work responsibilities and job duties.

━━━ WHAT TO EXTRACT ━━━

TECHNICAL_SKILLS:
Professional competencies, domain expertise, methodologies, and capabilities.
A technical skill is something a recruiter would list as a job requirement.

Extract from responsibilities too:
"managed vendor relationships" → Vendor Management
"conducted market research" → Market Research
"led design team" → Team Leadership
"developed financial models" → Financial Modeling
"supervised product lifecycle" → Product Lifecycle Management

━━━ WHAT NOT TO EXTRACT ━━━

DO NOT extract these as technical skills — they are achievements or outcomes:
- Quantified results: "Revenue Growth", "Cost Savings", "Sales Boosting"
- Percentages or metrics: "46% Sales Increase", "38% Revenue Growth"
- Scale or efficiency outcomes: "Scale Efficiencies", "Cost Reduction"
- One-time accomplishments: "Award Winner", "Top Seller"
- Dissertation, thesis, or one-off project titles (named studies/projects)
- One-off named outputs that are not recurring skills (e.g. a single
  product name, a single campaign name, a single uniform/collection name)

RULE: If it describes an OUTCOME or RESULT → do NOT include it.
RULE: If it describes a CAPABILITY or PROCESS → include it.

EXAMPLES (illustrative across different domains):
✓ INCLUDE (capabilities/processes):
  Software: API Development, System Design, Code Review, Agile
  Sales: Territory Management, Vendor Management, Account Management
  Finance: Financial Modeling, Risk Assessment, Budgeting
  Marketing: Campaign Management, SEO, Brand Strategy
  Fashion/Design: Collection Development, Technical Design, Sketching,
    Fabric Selection, Trend Forecasting, Merchandising
  HR: Talent Acquisition, Performance Management, Onboarding
  Healthcare: Patient Assessment, Clinical Documentation, Care Planning
  Operations: Process Improvement, Inventory Management, Supply Chain

✗ EXCLUDE (outcomes/results/one-off items, regardless of domain):
  Revenue Growth, Cost Savings, Scale Efficiencies, Sales Boosting,
  46% Increase, $67,000 Savings, SKU Reduction, Award Winner,
  Top Performer, Employee of the Month, [Specific Project/Thesis Names]
  e.g. "Faux Leather Evolution Study", "Q3 Turnaround Project"

TOOLS:
Only named software/platforms with a brand name explicitly in the resume.{excluded_note}

SOFT_SKILLS:
Only interpersonal/behavioral traits with resume evidence.
Examples: Communication, Leadership, Teamwork, Problem Solving,
Adaptability, Creativity, Attention To Detail, Negotiation.
DO NOT put business competencies here.

━━━ RULES ━━━
1. Extract capabilities and competencies — not outcomes or achievements
2. Extract from responsibilities, not just the skills section
3. Be comprehensive for professional skills
4. Each item in exactly one category

Return ONLY valid JSON:
{{
    "technical_skills": [],
    "soft_skills": [],
    "tools": []
}}

Resume:
{resume_text}
"""

    extracted = call_llm_json(
        prompt=prompt,
        fallback=_FALLBACK.copy(),
        temperature=0.1,
        max_tokens=1500,
    )

    for key in ("technical_skills", "soft_skills", "tools"):
        if not isinstance(extracted.get(key), list):
            extracted[key] = []

    # Remove domain-excluded tools
    excluded_set = _get_excluded_tools(domain)
    extracted["tools"] = [t for t in extracted["tools"] if t not in excluded_set]

    # Keyword safety net for tools
    keyword_tools = _detect_tools_by_keyword(resume_text, domain)
    known_lower   = {t.lower(): t for t in KNOWN_TOOLS}
    all_tools     = list({
        known_lower.get(t.lower(), t)
        for t in extracted["tools"] + keyword_tools
    })
    extracted["tools"] = all_tools

    # Validate against resume text
    extracted["technical_skills"] = _validate_against_resume(
        extracted["technical_skills"], resume_text
    )
    extracted["soft_skills"] = _validate_against_resume(
        extracted["soft_skills"], resume_text
    )
    extracted["tools"] = _validate_against_resume(
        extracted["tools"], resume_text
    )

    # Normalize
    extracted["technical_skills"] = _normalize_list(extracted["technical_skills"])
    extracted["soft_skills"]       = _normalize_list(extracted["soft_skills"])
    extracted["tools"]             = _normalize_list(extracted["tools"], title_case=False)

    # Remove tools from technical_skills
    tool_set = {t.lower() for t in extracted["tools"]}
    extracted["technical_skills"] = [
        s for s in extracted["technical_skills"]
        if s.lower() not in tool_set
    ]

    # Remove soft skills overlapping with technical
    tech_set = {s.lower() for s in extracted["technical_skills"]}
    extracted["soft_skills"] = [
        s for s in extracted["soft_skills"]
        if s.lower() not in tech_set and s.lower() not in tool_set
    ]

    logger.info(
        "Skills | domain=%s | technical=%d | tools=%d | soft=%d",
        domain, len(extracted["technical_skills"]),
        len(extracted["tools"]), len(extracted["soft_skills"])
    )
    return extracted

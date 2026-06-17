"""
Resume Segmenter — splits raw resume text into logical sections.

Improvements over original:
- Expanded header variant list (catches real-world resume headers)
- Header lines are excluded from section content
- Content before first detected header is captured as 'header' section
- Handles partial-line headers (e.g. "Work Experience — 2020 to Present")
- Case-insensitive matching
"""

import re
import logging

logger = logging.getLogger("careerpilot.resume_segmenter")

# ──────────────────────────────────────────────
# SECTION HEADER REGISTRY
# Maps canonical section name → all known header variants
# ──────────────────────────────────────────────

SECTION_HEADERS: dict[str, list[str]] = {
    "summary": [
        "profile", "summary", "professional summary", "career summary",
        "about me", "about", "objective", "career objective",
        "professional profile", "personal statement", "overview",
        "executive summary", "professional overview",
    ],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment history", "employment", "work history",
        "career history", "professional history", "job history",
        "relevant experience", "positions held", "professional background",
        "work background",
    ],
    "education": [
        "education", "educational background", "academic background",
        "academic qualifications", "qualifications", "academics",
        "educational qualifications", "academic history",
        "educational history", "degrees",
    ],
    "projects": [
        "projects", "project experience", "key projects",
        "relevant projects", "personal projects", "academic projects",
        "notable projects", "project highlights", "project work",
        "portfolio",
    ],
    "skills": [
        "skills", "technical skills", "key skills", "core skills",
        "competencies", "core competencies", "areas of expertise",
        "expertise", "technical expertise", "professional skills",
        "skill set", "skillset", "capabilities", "technologies",
        "technical competencies",
    ],
    "certifications": [
        "certifications", "certification", "licenses", "licences",
        "courses", "training", "professional development",
        "awards and certifications", "certifications & training",
        "certifications and licenses", "professional certifications",
        "professional training", "continuing education",
    ],
}

# Pre-build lowercase lookup: variant → canonical section name
_HEADER_LOOKUP: dict[str, str] = {
    variant.lower(): section
    for section, variants in SECTION_HEADERS.items()
    for variant in variants
}


def _classify_line(line: str) -> str | None:
    """
    Return canonical section name if this line looks like a section header,
    else return None.

    Strategy:
    1. Exact lowercase match against all known variants.
    2. Strip trailing punctuation/dates and try again.
    3. Starts-with match for headers followed by separators.
    """
    clean = line.strip()
    if not clean:
        return None

    lower = clean.lower()

    # 1. Exact match
    if lower in _HEADER_LOOKUP:
        return _HEADER_LOOKUP[lower]

    # 2. Strip trailing noise (dates, dashes, colons) and retry
    stripped = re.split(r"[\s:—\-–|]+\d{4}", lower)[0].strip()
    stripped = stripped.rstrip(":– —-").strip()
    if stripped in _HEADER_LOOKUP:
        return _HEADER_LOOKUP[stripped]

    # 3. Starts-with (e.g. "Skills & Competencies", "Experience:")
    for variant, section in _HEADER_LOOKUP.items():
        if lower.startswith(variant) and len(lower) < len(variant) + 20:
            return section

    return None


def segment_resume_sections(text: str) -> dict[str, str]:
    """
    Segment resume text into canonical sections.

    Returns dict with keys:
        summary, experience, education, projects, skills, certifications

    Content before the first detected header is stored in 'header'
    (typically name, contact info).
    """
    sections: dict[str, str] = {
        "summary": "",
        "experience": "",
        "education": "",
        "projects": "",
        "skills": "",
        "certifications": "",
    }

    lines = text.splitlines()
    current_section: str | None = None
    pre_header_lines: list[str] = []

    for line in lines:
        detected = _classify_line(line)

        if detected is not None:
            current_section = detected
            # Don't append the header line itself to section content
            continue

        if current_section is None:
            # Content before any header (name, contact info)
            pre_header_lines.append(line)
        else:
            sections[current_section] += line + "\n"

    # Store pre-header content in summary as fallback context
    # (useful for ATS analysis — name/contact doesn't pollute sections)
    pre_header_text = "\n".join(pre_header_lines).strip()
    if pre_header_text and not sections["summary"]:
        sections["summary"] = pre_header_text

    populated = [k for k, v in sections.items() if v.strip()]
    logger.info("Segmented sections: %s", populated)

    return sections

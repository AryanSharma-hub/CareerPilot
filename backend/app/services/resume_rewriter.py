"""
Resume Rewriter v3 — ATS-optimized bullet rewriting.

Fixes over v2:
- max_tokens raised to 3000 (prevents truncation causing blank output)
- Resume text capped at 4000 chars to stay within token budget
- Prompt shortened and made domain-agnostic (no hardcoded role categories)
- Automatic retry with simpler prompt when first attempt returns empty
- Local fallback generates basic bullets from resume text so UI never blanks
"""

import logging
from app.utils.llm_client import call_llm_json

logger = logging.getLogger("careerpilot.resume_rewriter")

_FALLBACK = {"optimized_bullets": []}

MAX_RESUME_CHARS = 4000   # cap to stay within token budget reliably


def _build_prompt(resume_text: str, target_role: str) -> str:
    return f"""
You are a senior resume writer and ATS optimization expert.

Rewrite the experience bullets from this resume for the target role: {target_role}

RULES:
1. Start every bullet with a strong action verb relevant to {target_role}
2. Keep all facts, metrics, numbers, and dates exactly as they appear
3. Never invent percentages, revenue, headcounts, or certifications
4. Transform task descriptions into achievement statements
5. Include ATS-friendly keywords natural for {target_role}
6. Keep each bullet to 1-2 lines
7. Generate 10-15 bullets covering the strongest experience

Return ONLY valid JSON, no markdown:
{{
    "optimized_bullets": ["bullet 1", "bullet 2", ...]
}}

RESUME:
{resume_text}
"""


def _build_simple_prompt(resume_text: str, target_role: str) -> str:
    """Shorter retry prompt used when first attempt fails or returns empty."""
    return f"""
You are a resume writer. Rewrite 8 experience bullets from this resume for: {target_role}

Rules: strong action verbs, preserve all facts and numbers, no invented metrics.

Return ONLY valid JSON:
{{"optimized_bullets": ["bullet 1", "bullet 2", "bullet 3", "bullet 4",
                        "bullet 5", "bullet 6", "bullet 7", "bullet 8"]}}

RESUME (excerpt):
{resume_text[:2000]}
"""


def _local_fallback_bullets(resume_text: str, target_role: str) -> list[str]:
    """
    Last-resort fallback: extract raw bullet lines from resume text and
    lightly clean them, so the UI always has something to show.
    These are NOT rewritten — just cleaned originals.
    """
    bullets = []
    for line in resume_text.split("\n"):
        line = line.strip()
        # Pick lines that look like bullet points
        if line.startswith(("•", "-", "*", "·")):
            cleaned = line.lstrip("•-*· ").strip()
            if len(cleaned) > 20:
                bullets.append(cleaned)
        elif len(line) > 40 and line[0].isupper() and not line.endswith(":"):
            bullets.append(line)
    # Return up to 10 cleaned lines
    return bullets[:10] if bullets else [
        f"Experienced professional seeking {target_role} opportunities.",
        "Please try rewriting again — the AI encountered a temporary error.",
    ]


def rewrite_resume_bullets(resume_text: str, target_role: str) -> dict:
    if not resume_text or not resume_text.strip():
        return {"target_role": target_role, "rewritten_resume": _FALLBACK.copy()}
    if not target_role or not target_role.strip():
        return {"target_role": "", "rewritten_resume": _FALLBACK.copy()}

    # Cap resume length to stay within token budget
    truncated = resume_text[:MAX_RESUME_CHARS]

    # ── Attempt 1: full prompt ──
    result = call_llm_json(
        prompt=_build_prompt(truncated, target_role),
        fallback=_FALLBACK.copy(),
        temperature=0.3,
        max_tokens=3000,
    )

    bullets = result.get("optimized_bullets", [])
    if not isinstance(bullets, list):
        bullets = []
    bullets = [b.strip() for b in bullets if isinstance(b, str) and b.strip()]

    # ── Attempt 2: retry with simpler prompt if empty ──
    if not bullets:
        logger.warning("Rewrite attempt 1 returned empty — retrying with simpler prompt.")
        result2 = call_llm_json(
            prompt=_build_simple_prompt(truncated, target_role),
            fallback=_FALLBACK.copy(),
            temperature=0.2,
            max_tokens=1500,
        )
        bullets = result2.get("optimized_bullets", [])
        if not isinstance(bullets, list):
            bullets = []
        bullets = [b.strip() for b in bullets if isinstance(b, str) and b.strip()]

    # ── Attempt 3: local fallback so UI never shows blank ──
    if not bullets:
        logger.warning("Rewrite attempt 2 returned empty — using local fallback.")
        bullets = _local_fallback_bullets(resume_text, target_role)

    logger.info("Rewrite | role=%s | bullets=%d", target_role, len(bullets))

    return {
        "target_role": target_role.strip(),
        "rewritten_resume": {"optimized_bullets": bullets},
    }

"""
Match Engine v5 — Generic Capability Matching (no hardcoded synonym dictionaries).

Pipeline for each JD requirement:
  1. Exact match (after minimal normalization) → Direct
  2. Semantic similarity vs extracted resume skill tags → Direct/Related
  3. Semantic similarity vs resume full-text chunks (experience, achievements,
     projects, responsibilities) → Direct/Related
  4. LLM-based capability inference for anything still unmatched:
     reasons about responsibility-to-skill and achievement-to-competency
     relationships generically (e.g. "managed a team of 10" → Leadership,
     "increased sales by 40%" → Commercial Impact, "worked with vendors
     throughout manufacturing" → Supplier Coordination)

A skill is marked "missing" only after ALL four steps find no evidence.

No domain-specific or hardcoded synonym mappings — generalization comes
from embeddings (steps 2-3) and LLM reasoning (step 4), both of which
work identically across any profession.
"""

import json
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.llm_client import call_llm_json

logger = logging.getLogger("careerpilot.match_engine")

_embedding_model = None

# ──────────────────────────────────────────────
# THRESHOLDS
# ──────────────────────────────────────────────

DIRECT_THRESHOLD            = 0.80   # Strong semantic match → full credit
RELATED_THRESHOLD_TAGS      = 0.60   # Moderate match vs skill tags → partial credit
RELATED_THRESHOLD_FULLTEXT  = 0.48   # Moderate match vs resume text chunks (longer, noisier)

MAX_INFERENCE_ITEMS = 15   # Cap items sent to LLM inference pass per category
MAX_RESUME_CHARS_FOR_INFERENCE = 6000


# ──────────────────────────────────────────────
# EMBEDDING MODEL (singleton)
# ──────────────────────────────────────────────

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading SentenceTransformer model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformer loaded.")
    return _embedding_model


def _normalize(text: str) -> str:
    """Minimal generic normalization — no synonym mapping."""
    return " ".join(text.lower().strip().split())


def _normalize_list(items: list[str]) -> list[str]:
    seen, result = set(), []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        norm = _normalize(item)
        if norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


# ──────────────────────────────────────────────
# RESUME TEXT CHUNKING
# Splits full resume text into overlapping windows so that
# multi-sentence context (e.g. a full bullet point) is preserved
# for semantic comparison against JD requirements.
# ──────────────────────────────────────────────

def _build_text_chunks(resume_full_text: str, max_chunks: int = 150) -> list[str]:
    if not resume_full_text:
        return []

    raw = [
        c.strip() for c in resume_full_text.replace("\n", ". ").split(". ")
        if len(c.strip()) > 15
    ]

    chunks, seen = [], set()
    for i in range(len(raw)):
        window = " ".join(raw[max(0, i - 1): i + 2])
        if window and window not in seen:
            seen.add(window)
            chunks.append(window)

    return chunks[:max_chunks]


# ──────────────────────────────────────────────
# CORE TIERED MATCH
# ──────────────────────────────────────────────

def tiered_semantic_match(
    resume_items: list[str],
    jd_items: list[str],
    resume_full_text: str = "",
    enable_llm_inference: bool = True,
) -> dict:
    """
    Match JD requirements against resume content using embeddings +
    optional LLM capability inference.

    Returns:
        {
            "direct":  [(jd_item, evidence, score), ...],
            "related": [(jd_item, evidence, score_or_none, reason_or_none), ...],
            "missing": [jd_item, ...]
        }
    """
    if not jd_items:
        return {"direct": [], "related": [], "missing": []}

    resume_norm = _normalize_list(resume_items)
    jd_norm     = _normalize_list(jd_items)

    resume_set = set(resume_norm)
    model      = get_embedding_model()

    resume_tag_embeddings = (
        model.encode(resume_norm, show_progress_bar=False)
        if resume_norm else None
    )

    text_chunks = _build_text_chunks(resume_full_text)
    text_embeddings = (
        model.encode(text_chunks, show_progress_bar=False)
        if text_chunks else None
    )

    direct, related, still_unmatched = [], [], []

    for jd_item in jd_norm:
        # ── Step 1: exact match ──
        if jd_item in resume_set:
            direct.append((jd_item, jd_item, 1.0))
            continue

        best_score, best_evidence, best_source = 0.0, "", ""

        # ── Step 2: semantic similarity vs resume skill tags ──
        if resume_tag_embeddings is not None and len(resume_norm) > 0:
            jd_emb   = model.encode([jd_item], show_progress_bar=False)
            sims     = cosine_similarity(jd_emb, resume_tag_embeddings)[0]
            idx      = int(sims.argmax())
            score    = float(sims[idx])
            if score > best_score:
                best_score, best_evidence, best_source = score, resume_norm[idx], "tag"

        # ── Step 3: semantic similarity vs resume text chunks ──
        if text_embeddings is not None and len(text_chunks) > 0:
            jd_emb_t  = model.encode([jd_item], show_progress_bar=False)
            sims_t    = cosine_similarity(jd_emb_t, text_embeddings)[0]
            idx_t     = int(sims_t.argmax())
            score_t   = float(sims_t[idx_t])
            # Text-chunk scores run lower due to noise — compare against
            # a fultext-specific direct threshold separately below.
            if score_t > best_score and best_source != "tag":
                best_score, best_evidence, best_source = score_t, text_chunks[idx_t], "text"
            elif score_t >= DIRECT_THRESHOLD and best_source == "tag" and best_score < DIRECT_THRESHOLD:
                # Text chunk alone clears direct bar even if tag didn't
                best_score, best_evidence, best_source = score_t, text_chunks[idx_t], "text"

        # ── Classify based on source-appropriate thresholds ──
        if best_source == "tag":
            if best_score >= DIRECT_THRESHOLD:
                direct.append((jd_item, best_evidence, best_score))
                continue
            elif best_score >= RELATED_THRESHOLD_TAGS:
                related.append((jd_item, best_evidence, best_score, None))
                continue

        elif best_source == "text":
            if best_score >= DIRECT_THRESHOLD:
                direct.append((jd_item, best_evidence, best_score))
                continue
            elif best_score >= RELATED_THRESHOLD_FULLTEXT:
                related.append((jd_item, best_evidence, best_score, None))
                continue

        # ── Steps 2-3 found nothing strong enough → candidate for LLM inference ──
        still_unmatched.append(jd_item)

    # ── Step 4: LLM-based capability inference for remaining items ──
    if enable_llm_inference and still_unmatched and resume_full_text:
        inferred_related, inferred_missing = _infer_capabilities_via_llm(
            still_unmatched, resume_full_text
        )
        for jd_item, reason in inferred_related.items():
            related.append((jd_item, "", None, reason))
        missing = inferred_missing
    else:
        missing = still_unmatched

    return {"direct": direct, "related": related, "missing": missing}


# ──────────────────────────────────────────────
# LLM CAPABILITY INFERENCE
# Generic responsibility-to-skill and achievement-to-competency reasoning.
# Works identically across any domain — no hardcoded mappings.
# ──────────────────────────────────────────────

_INFERENCE_FALLBACK = {"results": []}


def _infer_capabilities_via_llm(
    candidate_requirements: list[str],
    resume_full_text: str,
) -> tuple[dict[str, str], list[str]]:
    """
    For each candidate requirement, ask the LLM whether the resume's
    experience, responsibilities, achievements, or projects provide
    indirect-but-meaningful evidence — even if worded completely
    differently from the requirement.

    Returns:
        (related_dict, missing_list)
        related_dict: {requirement: evidence_reason}
        missing_list: requirements with no meaningful evidence
    """
    if not candidate_requirements:
        return {}, []

    items = candidate_requirements[:MAX_INFERENCE_ITEMS]
    overflow = candidate_requirements[MAX_INFERENCE_ITEMS:]

    resume_excerpt = resume_full_text[:MAX_RESUME_CHARS_FOR_INFERENCE]

    prompt = f"""
You are an expert recruiter performing capability matching.

For each JOB REQUIREMENT below, determine whether the candidate's resume
provides MEANINGFUL EVIDENCE for it — even if the resume uses completely
different wording.

Look for these relationship types:
- Responsibility → Skill: a described duty implies the competency
  e.g. "Led product development" implies "Product Development Management"
  e.g. "Worked with vendors throughout manufacturing" implies "Supplier Coordination"
- Achievement → Competency: a quantified result implies the underlying skill
  e.g. "Increased sales by 40%" implies "Commercial Impact" / "Revenue Generation"
  e.g. "Managed a team of 10 engineers" implies "Leadership" / "Team Management"
- Process → Skill: a described workflow implies the methodology
  e.g. "Built and validated predictive models" implies "Machine Learning"

CLASSIFICATION:
- "related": the resume provides indirect but meaningful evidence via one
  of the relationship types above. Provide a ONE-SENTENCE reason citing
  the SPECIFIC resume content that supports it.
- "missing": no meaningful evidence anywhere in the resume — not in
  skills, experience, achievements, projects, or certifications.

Be reasonably generous — if a responsibility or achievement plausibly
demonstrates the requirement, mark it "related". Only mark "missing"
when there is truly nothing connecting the resume to the requirement.

JOB REQUIREMENTS TO CHECK:
{json.dumps(items, ensure_ascii=False)}

CANDIDATE RESUME (experience, achievements, projects, responsibilities):
{resume_excerpt}

Return ONLY valid JSON, no markdown:
{{
    "results": [
        {{"requirement": "...", "status": "related", "evidence": "one sentence reason"}},
        {{"requirement": "...", "status": "missing", "evidence": null}}
    ]
}}
"""

    result = call_llm_json(
        prompt=prompt,
        fallback=_INFERENCE_FALLBACK.copy(),
        temperature=0.1,
        max_tokens=2000,
    )

    related: dict[str, str] = {}
    missing: list[str] = []
    classified = set()

    for entry in result.get("results", []):
        if not isinstance(entry, dict):
            continue
        req    = entry.get("requirement", "")
        status = entry.get("status", "missing")
        evidence = entry.get("evidence")

        # Match back to original casing/item
        match = next((i for i in items if _normalize(i) == _normalize(req)), None)
        if not match:
            continue

        classified.add(match)
        if status == "related" and evidence:
            related[match] = evidence
        else:
            missing.append(match)

    # Anything the LLM didn't classify (parse issues) → missing (safe default)
    for item in items:
        if item not in classified:
            missing.append(item)

    # Overflow items (beyond MAX_INFERENCE_ITEMS) → missing without inference
    missing.extend(overflow)

    logger.info(
        "LLM inference | checked=%d | related=%d | missing=%d",
        len(items), len(related), len(missing)
    )

    return related, missing


# ──────────────────────────────────────────────
# MAIN MATCH ENGINE
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# SCORE CALIBRATION HELPERS
#
# These functions recalibrate the FINAL SCORE only.
# They do NOT alter tiered_semantic_match's matched/related/missing
# lists, evidence, or reasons — UI and explanations are unchanged.
# ──────────────────────────────────────────────

CLUSTER_SIMILARITY_THRESHOLD = 0.78

# ──────────────────────────────────────────────
# MATCH CREDIT WEIGHTS
# Applied per item inside _category_score.
# NOT domain-specific — universally express match quality tiers.
# ──────────────────────────────────────────────
DIRECT_WEIGHT            = 1.00   # stated skill, exact or synonym match
LLM_INFERRED_WEIGHT      = 0.92   # LLM confirmed evidence from resume text
EMBEDDING_RELATED_WEIGHT = 0.55   # semantic similarity without explicit reasoning

# ──────────────────────────────────────────────
# FINAL SCORE COMPOSITION
#
# The formula has two stages:
#
# Stage 1 — Capability score (tools excluded):
#   capability_score = skills_pct * SKILLS_SHARE + soft_pct * SOFT_SHARE
#                      blended with ecosystem_sim via ECO_WEIGHT
#   base_score = capability_score * (1 - ECO_WEIGHT) + eco_sim*100 * ECO_WEIGHT
#
# Stage 2 — Tool penalty (applied after, not additive):
#   missing_tool_ratio = clustered_missing_tools / total_jd_tools
#   tool_penalty = missing_tool_ratio * MAX_TOOL_PENALTY
#   final_score = base_score * (1 - tool_penalty)
#
# Why penalty-based instead of additive:
#   If tools are an additive component (tools_score * 0.10), a candidate
#   who matches ALL skills/soft but is missing 3 tools scores 10 points
#   lower than a candidate whose JD has no tools at all — even though
#   tools are learnable specifics that recruiters treat as minor gaps.
#   Making tools a penalty on the already-computed base_score ensures:
#     - Missing 0 tools: no penalty (0%)
#     - Missing all tools: at most MAX_TOOL_PENALTY reduction (5%)
#     - Tools can never affect the score by more than 10 points total
#
# Domain alignment (ecosystem_sim) contribution is continuous — no
# buckets, no domain-specific thresholds. A sim of 0.72 contributes
# sim=0.72 * ECO_WEIGHT=0.38 contributes ~27 pts toward the final score
# which industry or role is being evaluated.
# ──────────────────────────────────────────────

SKILLS_SHARE    = 0.75   # within capability_score: skills vs soft
SOFT_SHARE      = 0.25
ECO_WEIGHT      = 0.38   # ecosystem/domain alignment (continuous, no buckets)
MAX_TOOL_PENALTY = 0.05   # max 5% reduction from missing tools

MAX_MATCH_SCORE = 97


def _match_band(score: int) -> str:
    """Map a final score to a recruiter-style fit band."""
    if score >= 90: return "Exceptional Fit"
    if score >= 80: return "Strong Fit"
    if score >= 70: return "Good Fit"
    if score >= 60: return "Moderate Fit"
    return "Weak Fit"


def _cluster_count(items: list[str], threshold: float = CLUSTER_SIMILARITY_THRESHOLD) -> int:
    """
    Count distinct semantic clusters among a list of strings.

    Used to avoid penalizing near-duplicate missing requirements as
    separate gaps — e.g. three differently-worded variants of the same
    underlying tool or competency describe one gap, not three.

    Does not change which items are reported as missing (UI lists are
    untouched) — only changes how many "units" they count as for the
    score denominator.
    """
    if not items:
        return 0
    if len(items) == 1:
        return 1

    model = get_embedding_model()
    embs  = model.encode(items, show_progress_bar=False)
    sims  = cosine_similarity(embs)

    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sims[i][j] >= threshold:
                union(i, j)

    return len({find(i) for i in range(n)})


def _category_score(result: dict) -> tuple[float, int]:
    """
    Compute weighted score and effective total units for one category
    (technical_skills, tools, or soft_skills).

    Weighting:
      Direct match               = 1.00
      Related (LLM-inferred)     = 0.85  — rewards inferred capabilities
                                            (e.g. "managed a team of 10"
                                            → Leadership)
      Related (embedding-based)  = 0.65  — partial credit for semantic
                                            similarity without explicit
                                            reasoning
      Missing                    = 0.00

    Missing items are clustered (see _cluster_count) so near-duplicate
    gaps count as one unit in the denominator instead of several.

    Returns: (weighted_sum, effective_total_units)
    """
    direct_count = len(result["direct"])

    # related entries are (jd_item, evidence, score, reason)
    # reason is not None  → LLM-inferred capability match
    # reason is None      → embedding-based semantic match
    llm_related_count = sum(1 for r in result["related"] if r[3] is not None)
    emb_related_count = len(result["related"]) - llm_related_count

    weighted_sum = (
        direct_count        * DIRECT_WEIGHT
        + llm_related_count * LLM_INFERRED_WEIGHT
        + emb_related_count * EMBEDDING_RELATED_WEIGHT
    )

    effective_missing = _cluster_count(result["missing"])
    effective_total   = direct_count + len(result["related"]) + effective_missing

    return weighted_sum, effective_total


def calculate_job_match(
    resume_data: dict,
    jd_data: dict,
    resume_full_text: str = "",
) -> dict:
    """
    Compute resume-to-job match using generic capability matching.

    Accepts optional resume_full_text for full-text semantic matching
    and LLM-based capability inference (steps 3-4 of the pipeline).
    """
    resume_skills = resume_data.get("technical_skills", [])
    jd_skills     = jd_data.get("technical_skills", [])
    resume_tools  = resume_data.get("tools", [])
    jd_tools      = jd_data.get("tools", [])
    resume_soft   = resume_data.get("soft_skills", [])
    jd_soft       = jd_data.get("soft_skills", [])

    # LLM inference is most valuable for technical_skills and soft_skills,
    # where responsibility/achievement → competency reasoning applies.
    # Tools are concrete named products — inference adds little value there.
    skills_result = tiered_semantic_match(
        resume_skills, jd_skills, resume_full_text, enable_llm_inference=True
    )
    tools_result = tiered_semantic_match(
        resume_tools, jd_tools, resume_full_text, enable_llm_inference=False
    )
    soft_result = tiered_semantic_match(
        resume_soft, jd_soft, resume_full_text, enable_llm_inference=True
    )

    # ── Per-category weighted scores ──
    # _category_score uses DIRECT/LLM_INFERRED/EMBEDDING_RELATED weights
    # and clusters near-duplicate missing items to avoid double-penalizing.
    skills_weighted, skills_total = _category_score(skills_result)
    tools_weighted,  tools_total  = _category_score(tools_result)
    soft_weighted,   soft_total   = _category_score(soft_result)

    skills_pct = (skills_weighted / skills_total * 100) if skills_total > 0 else 100.0
    tools_pct  = (tools_weighted  / tools_total  * 100) if tools_total  > 0 else 100.0
    soft_pct   = (soft_weighted   / soft_total   * 100) if soft_total   > 0 else 100.0

    # ── Stage 1: capability score (skills + soft, tools excluded) ──
    # Tools are deliberately excluded from this stage so that missing
    # named products never suppress the core capability assessment.
    if skills_total > 0 and soft_total > 0:
        capability_score = skills_pct * SKILLS_SHARE + soft_pct * SOFT_SHARE
    elif skills_total > 0:
        capability_score = skills_pct
    elif soft_total > 0:
        capability_score = soft_pct
    else:
        capability_score = 50.0

    # ── Ecosystem similarity (domain alignment + experience proxy) ──
    # Computed from full extracted skill sets — a continuous 0-1 signal
    # that captures overall domain relevance independently of which
    # individual skill names were matched. No domain-specific thresholds.
    resume_all = resume_skills + resume_tools + resume_soft
    jd_all     = jd_skills    + jd_tools     + jd_soft

    if resume_all and jd_all:
        model = get_embedding_model()
        embs  = model.encode(
            [" ".join(resume_all), " ".join(jd_all)],
            show_progress_bar=False
        )
        ecosystem_sim = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
    else:
        ecosystem_sim = 0.5

    # ── Stage 1 final: blend capability with domain alignment ──
    base_score = (
        capability_score * (1.0 - ECO_WEIGHT)
        + ecosystem_sim * 100.0 * ECO_WEIGHT
    )

    # ── Stage 2: tool penalty (at most MAX_TOOL_PENALTY = 5% reduction) ──
    # Only applied when the JD specifies tools. The penalty is proportional
    # to the fraction of clustered tool gaps — missing all tools reduces
    # the score by at most 10%, missing none applies no penalty.
    # This ensures tools are treated as learnable specifics, not
    # core competency blockers.
    if tools_total > 0:
        missing_tools_count = _cluster_count(tools_result["missing"])
        missing_ratio = missing_tools_count / tools_total
        tool_penalty  = missing_ratio * MAX_TOOL_PENALTY
    else:
        tool_penalty = 0.0

    raw_score = base_score * (1.0 - tool_penalty)

    match_percentage = int(max(0, min(MAX_MATCH_SCORE, round(raw_score))))
    match_band = _match_band(match_percentage)

    domain_alignment = (
        "Strong"   if ecosystem_sim >= 0.65 else
        "Moderate" if ecosystem_sim >= 0.40 else
        "Weak"
    )

    # ── Flatten for API response ──
    matched_skills = [d[0] for d in skills_result["direct"]]
    related_skills = []
    for entry in skills_result["related"]:
        jd_item, evidence, score, reason = entry
        related_skills.append({
            "skill": jd_item,
            "related_to": evidence if evidence else "inferred from experience",
            "score": round(score, 2) if score is not None else None,
            "reason": reason,
        })
    missing_skills = skills_result["missing"]

    matched_tools = [d[0] for d in tools_result["direct"]]
    related_tools = []
    for entry in tools_result["related"]:
        jd_item, evidence, score, reason = entry
        related_tools.append({
            "skill": jd_item,
            "related_to": evidence if evidence else "inferred from experience",
            "score": round(score, 2) if score is not None else None,
            "reason": reason,
        })
    missing_tools = tools_result["missing"]

    matched_soft = [d[0] for d in soft_result["direct"]]
    related_soft = []
    for entry in soft_result["related"]:
        jd_item, evidence, score, reason = entry
        related_soft.append({
            "skill": jd_item,
            "related_to": evidence if evidence else "inferred from experience",
            "score": round(score, 2) if score is not None else None,
            "reason": reason,
        })
    missing_soft = soft_result["missing"]

    # ── Recommendations ──
    recommendations = []
    for s in missing_skills[:4]:
        recommendations.append(f"Develop or highlight: {s.title()}")
    for r in related_skills[:2]:
        recommendations.append(f"Emphasize '{r['skill'].title()}' more explicitly in your resume")
    for t in missing_tools[:3]:
        recommendations.append(f"Gain experience with: {t.title()}")

    logger.info(
        "Match | %d%% | direct=%d | related=%d | missing=%d | ecosystem=%.2f",
        match_percentage, len(matched_skills), len(related_skills),
        len(missing_skills), ecosystem_sim
    )

    return {
        "match_percentage":        match_percentage,
        "match_band":              match_band,
        "domain_alignment":        domain_alignment,
        "domain_similarity_score": round(ecosystem_sim, 2),
        "matched_skills":          matched_skills,
        "related_skills":          related_skills,
        "missing_skills":          missing_skills,
        "matched_tools":           matched_tools,
        "related_tools":           related_tools,
        "missing_tools":           missing_tools,
        "matched_competencies":    matched_soft,
        "related_competencies":    related_soft,
        "missing_competencies":    missing_soft,
        "recommendations":         recommendations,
    }

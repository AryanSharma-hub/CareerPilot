"""
Resume Routes v3 — production-hardened with edge case handling,
match insights, tiered skill matching, and graceful fallbacks.
"""

import logging
import fitz
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import os

from app.services.resume_parser import extract_email, extract_phone, extract_linkedin
from app.services.llm_ats_analyzer import analyze_resume_with_llm
from app.services.domain_detector import detect_resume_domain
from app.services.resume_structurer import extract_resume_structure
from app.services.ats_score_engine import calculate_component_scores, calculate_final_ats_score
from app.services.impact_analyzer import analyze_resume_impact
from app.services.skill_extractor import extract_skills_llm
from app.services.resume_segmenter import segment_resume_sections
from app.services.competency_inference import infer_competencies
from app.services.project_analyzer import analyze_project_strength
from app.services.jd_analyzer import analyze_job_description
from app.services.match_engine import calculate_job_match
from app.services.match_insights import generate_match_insights
from app.services.career_context_builder import build_career_context
from app.services.career_chatbot import generate_career_response
from app.services.chat_memory import save_user_memory, get_user_memory
from app.services.interview_generator import generate_interview_questions
from app.services.resume_rewriter import rewrite_resume_bullets

logger = logging.getLogger("careerpilot.routes.resume")

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Request Models ──

class JobDescriptionRequest(BaseModel):
    resume_text: str
    job_description: str

class CareerChatRequest(BaseModel):
    user_id: str
    resume_analysis: dict
    ats_analysis: dict
    job_match_analysis: dict
    user_question: str

class InterviewRequest(BaseModel):
    resume_analysis: dict
    job_match_analysis: dict
    target_role: str

class ResumeRewriteRequest(BaseModel):
    resume_text: str
    target_role: str


# ── Helpers ──

def _err(msg: str, code: int = 500):
    raise HTTPException(status_code=code, detail=msg)


def _safe_text(sections: dict, keys: list, full_text: str, min_len: int = 80) -> str:
    combined = "\n".join(sections.get(k, "") for k in keys).strip()
    return combined if len(combined) >= min_len else full_text


def _extract_memory_keys(question: str) -> dict:
    q = question.lower()
    role_map = {
        "machine learning": "Machine Learning Engineer",
        "ml engineer": "Machine Learning Engineer",
        "data scientist": "Data Scientist",
        "data analyst": "Data Analyst",
        "software engineer": "Software Engineer",
        "backend developer": "Backend Developer",
        "frontend developer": "Frontend Developer",
        "full stack": "Full Stack Developer",
        "product manager": "Product Manager",
        "devops": "DevOps Engineer",
        "fashion designer": "Fashion Designer",
        "ui ux": "UI/UX Designer",
        "graphic designer": "Graphic Designer",
        "financial analyst": "Financial Analyst",
        "marketing manager": "Marketing Manager",
        "sales manager": "Sales Manager",
        "hr manager": "HR Manager",
    }
    for kw, role in role_map.items():
        if kw in q:
            return {"target_role": role}
    return {}


# ── Upload Resume ──

@router.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        _err("Only PDF files are supported.", 400)

    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Extract text
        try:
            doc = fitz.open(file_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
        except Exception as e:
            logger.error("PDF extraction failed: %s", e)
            _err("Failed to extract text from PDF.", 422)

        # Edge case: image-based or empty PDF
        if not text or len(text.strip()) < 50:
            _err(
                "The PDF appears to be scanned/image-based or empty. "
                "Please upload a text-based PDF resume.",
                422
            )

        sections = segment_resume_sections(text)

        email    = extract_email(text)
        phone    = extract_phone(text)
        linkedin = extract_linkedin(text)

        domain_data = detect_resume_domain(text)
        domain      = domain_data.get("domain", "General Professional")

        skill_text = _safe_text(
            sections,
            ["skills", "experience", "projects", "education"],
            text
        )
        skill_data = extract_skills_llm(skill_text, domain)

        competency_data = infer_competencies(
            sections.get("experience", "") or text[:3000],
            domain
        )

        impact_text   = _safe_text(sections, ["experience", "skills", "projects"], text, 50)
        impact_analysis = analyze_resume_impact(impact_text, domain)

        project_text    = _safe_text(sections, ["projects", "experience", "skills"], text, 30)
        project_analysis = analyze_project_strength(project_text, domain)

        structured_data = extract_resume_structure(text)
        structured_data["technical_skills"] = skill_data["technical_skills"]
        structured_data["tools"]            = skill_data["tools"]

        tech_set = {s.lower() for s in structured_data["technical_skills"]}
        combined_soft = list({
            s for s in (
                skill_data["soft_skills"] +
                competency_data.get("inferred_soft_skills", [])
            )
            if s.lower() not in tech_set
        })
        structured_data["soft_skills"] = combined_soft

        ats_text     = _safe_text(sections, ["experience", "skills", "projects", "education"], text)
        llm_analysis = analyze_resume_with_llm(ats_text, domain, structured_data)

        component_scores = calculate_component_scores(structured_data, text)
        component_scores["impact_score"]  = impact_analysis["impact_score"]
        component_scores["project_score"] = project_analysis["project_score"]

        weighted_score = calculate_final_ats_score(component_scores)
        final_score    = max(0, min(100, round(weighted_score * 0.85 + llm_analysis["ats_score"] * 0.15)))
        llm_analysis["ats_score"]       = final_score
        llm_analysis["score_breakdown"] = component_scores

        logger.info("Resume | file=%s | domain=%s | score=%d", file.filename, domain, final_score)

        return {
            "filename":              file.filename,
            "detected_domain":       domain_data,
            "structured_resume_data": structured_data,
            "email":                 email,
            "phone":                 phone,
            "linkedin":              linkedin,
            "skills": {
                "technical_skills": structured_data["technical_skills"],
                "soft_skills":      structured_data["soft_skills"],
                "tools":            structured_data["tools"],
            },
            "segmented_sections":    sections,
            "impact_analysis":       impact_analysis,
            "project_analysis":      project_analysis,
            "llm_analysis":          llm_analysis,
            "extracted_text":        text,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("upload_resume failed: %s", e, exc_info=True)
        _err(f"Resume analysis failed: {str(e)}")


# ── Match Job ──

@router.post("/match-job")
async def match_job(data: JobDescriptionRequest):
    if not data.resume_text or not data.resume_text.strip():
        _err("resume_text is required.", 400)
    if not data.job_description or not data.job_description.strip():
        _err("job_description is required.", 400)

    # Edge case: very short JD
    if len(data.job_description.strip()) < 30:
        _err("Job description is too short. Please paste the full job description.", 400)

    try:
        domain_data = detect_resume_domain(data.resume_text)
        domain      = domain_data.get("domain", "General Professional")
        sections    = segment_resume_sections(data.resume_text)

        skill_text  = _safe_text(sections, ["skills", "experience", "projects", "education"], data.resume_text)
        skill_data  = extract_skills_llm(skill_text, domain)
        comp_data   = infer_competencies(sections.get("experience", "") or data.resume_text[:3000], domain)

        tech_set = {s.lower() for s in skill_data["technical_skills"]}
        resume_data = {
            "domain": domain,
            "technical_skills": skill_data["technical_skills"],
            "tools":            skill_data["tools"],
            "soft_skills": list({
                s for s in skill_data["soft_skills"] + comp_data.get("inferred_soft_skills", [])
                if s.lower() not in tech_set
            }),
        }

        jd_data      = analyze_job_description(data.job_description)
        match_result = calculate_job_match(resume_data, jd_data, resume_full_text=data.resume_text)
        insights     = generate_match_insights(resume_data, jd_data, match_result)

        logger.info("Match | domain=%s | score=%d%%", domain, match_result.get("match_percentage", 0))

        return {
            "resume_analysis":          resume_data,
            "job_description_analysis": jd_data,
            "job_match_analysis":       match_result,
            "match_insights":           insights,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("match_job failed: %s", e, exc_info=True)
        _err(f"Job match failed: {str(e)}")


# ── Career Chat ──

@router.post("/career-chat")
async def career_chat(data: CareerChatRequest):
    if not data.user_question or not data.user_question.strip():
        _err("user_question is required.", 400)

    try:
        memory   = get_user_memory(data.user_id)
        new_keys = _extract_memory_keys(data.user_question)
        for k, v in new_keys.items():
            save_user_memory(data.user_id, k, v)
            memory[k] = v

        career_context = build_career_context(
            data.resume_analysis,
            data.ats_analysis,
            data.job_match_analysis,
        )
        if memory:
            memory_lines    = "\n".join(f"• {k}: {v}" for k, v in memory.items())
            career_context += f"\n\nUser Preferences:\n{memory_lines}"

        response = generate_career_response(data.user_question, career_context)

        return {"memory": memory, "career_response": response}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("career_chat failed: %s", e, exc_info=True)
        _err(f"Career chat failed: {str(e)}")


# ── Generate Interview ──

@router.post("/generate-interview")
async def generate_interview(data: InterviewRequest):
    if not data.target_role or not data.target_role.strip():
        _err("target_role is required.", 400)

    try:
        questions = generate_interview_questions(
            data.resume_analysis,
            data.job_match_analysis,
            data.target_role,
        )
        return {"target_role": data.target_role, "interview_questions": questions}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("generate_interview failed: %s", e, exc_info=True)
        _err(f"Interview generation failed: {str(e)}")


# ── Rewrite Resume ──

@router.post("/rewrite-resume")
async def rewrite_resume(data: ResumeRewriteRequest):
    if not data.resume_text or not data.resume_text.strip():
        _err("resume_text is required.", 400)
    if not data.target_role or not data.target_role.strip():
        _err("target_role is required.", 400)

    try:
        result = rewrite_resume_bullets(data.resume_text, data.target_role)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("rewrite_resume failed: %s", e, exc_info=True)
        _err(f"Resume rewrite failed: {str(e)}")

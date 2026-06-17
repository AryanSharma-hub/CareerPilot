"""
CareerPilot — FastAPI Application Entry Point (Production Ready)
CORS updated to accept requests from any Vercel frontend URL.
"""

import os
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes.resume import router as resume_router
from app.rag.rag_engine import load_knowledge_base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("careerpilot.main")

app = FastAPI(
    title="CareerPilot AI",
    description="AI-powered Resume Analysis, ATS Scoring, and Job Matching Platform",
    version="2.0.0",
)

# ── CORS ──
# FRONTEND_URL is set as an environment variable on Render.
# Falls back to localhost for local development.
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method, request.url.path, exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again."},
    )

@app.on_event("startup")
async def startup_event():
    logger.info("CareerPilot starting up...")
    rag_ok = load_knowledge_base()
    if not rag_ok:
        logger.warning("RAG knowledge base not loaded.")
    logger.info("CareerPilot ready.")

app.include_router(resume_router)

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "CareerPilot AI Backend",
        "version": "2.0.0",
    }

"""
RAG Engine — knowledge base retrieval for career guidance.

Improvements over original:
- Singleton embedding model (shared with match_engine via get_embedding_model)
- Duplicate insertion guard (checks collection count before loading)
- Graceful handling of missing career_data.txt
- generate_rag_response is now actually used and exported cleanly
- Minimum chunk length filter (skips blank/tiny paragraphs)
- Structured logging throughout
"""

import os
import logging

import chromadb
from app.utils.llm_client import call_llm
from app.services.match_engine import get_embedding_model
from typing import Any
logger = logging.getLogger("careerpilot.rag_engine")

# ──────────────────────────────────────────────
# CHROMADB — in-process persistent client
# ──────────────────────────────────────────────

_chroma_client: Any = None
_collection: Any = None

MIN_CHUNK_LENGTH = 40   # Skip chunks shorter than this
RAG_RESULTS_COUNT = 3   # Number of docs to retrieve per query


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.Client()
        _collection = _chroma_client.get_or_create_collection(
            name="career_knowledge"
        )
    return _collection


# ──────────────────────────────────────────────
# LOAD KNOWLEDGE BASE
# ──────────────────────────────────────────────

def load_knowledge_base() -> bool:
    """
    Load career_data.txt into ChromaDB vector store.

    Safe to call on startup:
    - Skips loading if collection already has documents (hot-reload safe)
    - Gracefully handles missing data file
    - Filters out empty/tiny chunks

    Returns True if loaded successfully, False otherwise.
    """
    collection = _get_collection()

    # ── Duplicate insertion guard ──
    existing_count = collection.count()
    if existing_count > 0:
        logger.info(
            "Knowledge base already loaded (%d chunks). Skipping reload.",
            existing_count,
        )
        return True

    # ── Locate data file ──
    base_dir = os.path.dirname(__file__)
    file_path = os.path.abspath(
        os.path.join(base_dir, "..", "data", "career_data.txt")
    )

    if not os.path.exists(file_path):
        logger.warning(
            "career_data.txt not found at %s. "
            "RAG knowledge base will be empty. "
            "Career chatbot will operate without retrieval context.",
            file_path,
        )
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.error("Failed to read career_data.txt: %s", e)
        return False

    # ── Chunk and embed ──
    raw_chunks = text.split("\n\n")
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) >= MIN_CHUNK_LENGTH]

    if not chunks:
        logger.warning("career_data.txt produced no usable chunks.")
        return False

    model = get_embedding_model()

    try:
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()
        collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
        )
        logger.info(
            "Knowledge base loaded: %d chunks from %s", len(chunks), file_path
        )
        return True

    except Exception as e:
        logger.error("Failed to load knowledge base into ChromaDB: %s", e)
        return False


# ──────────────────────────────────────────────
# RETRIEVAL
# ──────────────────────────────────────────────

def retrieve_relevant_docs(query: str) -> list[str]:
    """
    Retrieve the most relevant knowledge base chunks for a query.
    Returns empty list if collection is empty or query fails.
    """
    if not query or not query.strip():
        return []

    collection = _get_collection()

    if collection.count() == 0:
        logger.debug("Knowledge base is empty — skipping retrieval.")
        return []

    try:
        model = get_embedding_model()
        query_embedding = model.encode([query], show_progress_bar=False).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(RAG_RESULTS_COUNT, collection.count()),
        )

        docs = results.get("documents", [[]])[0]
        logger.debug("Retrieved %d docs for query: %.60s", len(docs), query)
        return docs

    except Exception as e:
        logger.error("RAG retrieval failed: %s", e)
        return []


# ──────────────────────────────────────────────
# GENERATE RAG RESPONSE
# ──────────────────────────────────────────────

def generate_rag_response(query: str) -> str:
    """
    Generate a response using retrieved knowledge base context.
    Falls back to direct LLM response if no docs retrieved.
    """
    if not query or not query.strip():
        return "Please provide a question."

    docs = retrieve_relevant_docs(query)

    if docs:
        context = "\n\n---\n\n".join(docs)
        prompt = f"""
You are a career knowledge expert.

Use the context below to answer the question accurately and concisely.
If the context does not contain a direct answer, use your general knowledge
but clearly state that.

Context:
{context}

Question:
{query}
"""
    else:
        # No RAG context available — answer directly
        prompt = f"""
You are a career knowledge expert.
Answer the following career question accurately and concisely.

Question:
{query}
"""

    try:
        return call_llm(prompt=prompt, temperature=0.3, max_tokens=600)
    except Exception as e:
        logger.error("generate_rag_response failed: %s", e)
        return "Unable to generate a response at this time. Please try again."

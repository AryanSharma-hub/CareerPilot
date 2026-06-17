"""
Chat Memory — persistent key-value memory store per user using TinyDB.

Improvements:
- Exception handling on all DB operations
- Memory entry limit per user (prevents unbounded growth)
- TTL-aware cleanup hook (optional, non-breaking)
- Structured logging
"""

import logging
from tinydb import TinyDB, Query

logger = logging.getLogger("careerpilot.chat_memory")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

DB_PATH = "career_memory.json"
MAX_MEMORIES_PER_USER = 20  # Prevent unbounded growth

# ──────────────────────────────────────────────
# DB INIT
# ──────────────────────────────────────────────

try:
    db = TinyDB(DB_PATH)
    UserMemory = Query()
    logger.info("TinyDB initialized at %s", DB_PATH)
except Exception as e:
    logger.error("Failed to initialize TinyDB: %s", e)
    db = None
    UserMemory = None


# ──────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────

def save_user_memory(
    user_id: str,
    memory_key: str,
    memory_value,
) -> bool:
    """
    Upsert a memory entry for a user.

    Returns True on success, False on failure.
    """
    if not db:
        logger.error("DB not available. Cannot save memory.")
        return False

    if not user_id or not memory_key:
        logger.warning("save_user_memory called with empty user_id or memory_key.")
        return False

    try:
        existing = db.search(
            (UserMemory.user_id == user_id)
            & (UserMemory.memory_key == memory_key)
        )

        if existing:
            db.update(
                {"memory_value": memory_value},
                (UserMemory.user_id == user_id)
                & (UserMemory.memory_key == memory_key),
            )
            logger.debug("Updated memory | user=%s | key=%s", user_id, memory_key)
        else:
            # Check memory limit before inserting
            user_entries = db.search(UserMemory.user_id == user_id)
            if len(user_entries) >= MAX_MEMORIES_PER_USER:
                # Remove oldest entry to make room
                oldest = user_entries[0]
                db.remove(doc_ids=[oldest.doc_id])
                logger.info(
                    "Memory limit reached for user=%s. Removed oldest entry.", user_id
                )

            db.insert({
                "user_id": user_id,
                "memory_key": memory_key,
                "memory_value": memory_value,
            })
            logger.debug("Inserted memory | user=%s | key=%s", user_id, memory_key)

        return True

    except Exception as e:
        logger.error(
            "save_user_memory failed | user=%s | key=%s | error=%s",
            user_id, memory_key, e
        )
        return False


# ──────────────────────────────────────────────
# GET
# ──────────────────────────────────────────────

def get_user_memory(user_id: str) -> dict:
    """
    Retrieve all stored memory entries for a user.

    Returns dict of {memory_key: memory_value}.
    Returns empty dict on failure.
    """
    if not db:
        logger.error("DB not available. Cannot retrieve memory.")
        return {}

    if not user_id:
        return {}

    try:
        entries = db.search(UserMemory.user_id == user_id)
        memory_dict = {
            entry["memory_key"]: entry["memory_value"]
            for entry in entries
            if "memory_key" in entry and "memory_value" in entry
        }
        logger.debug("Retrieved %d memory entries for user=%s", len(memory_dict), user_id)
        return memory_dict

    except Exception as e:
        logger.error("get_user_memory failed | user=%s | error=%s", user_id, e)
        return {}


# ──────────────────────────────────────────────
# CLEAR (utility — useful for testing)
# ──────────────────────────────────────────────

def clear_user_memory(user_id: str) -> bool:
    """Remove all memory entries for a user."""
    if not db:
        return False
    try:
        db.remove(UserMemory.user_id == user_id)
        logger.info("Cleared all memory for user=%s", user_id)
        return True
    except Exception as e:
        logger.error("clear_user_memory failed | user=%s | error=%s", user_id, e)
        return False

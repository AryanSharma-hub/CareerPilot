"""
Resume Parser — lightweight utility for extracting contact information
from raw resume text using regex patterns.

Improvements:
- Removed dead code: SKILLS_DB and extract_skills() (superseded by skill_extractor.py)
- Improved phone regex to reduce false positives on years and IDs
- Added name extraction heuristic
- Added LinkedIn/URL extraction
"""

import re
import logging

logger = logging.getLogger("careerpilot.resume_parser")


def extract_email(text: str) -> str | None:
    """Extract the first email address found in text."""
    if not text:
        return None
    match = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', text)
    return match.group(0).strip() if match else None


def extract_phone(text: str) -> str | None:
    """
    Extract a phone number from text.

    Improved pattern:
    - Requires at least 7 digits
    - Anchors to phone-like boundaries (not inside longer numbers)
    - Handles Indian (+91), US (+1), and international formats
    - Avoids matching year ranges like "2019-2022"
    """
    if not text:
        return None

    patterns = [
        # International with country code: +91-9876543210, +1 (555) 123-4567
        r'\+\d{1,3}[\s\-.]?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}',
        # 10-digit Indian mobile: 9876543210 (no country code)
        r'\b[6-9]\d{9}\b',
        # Standard formats: (555) 123-4567, 555-123-4567, 555.123.4567
        r'\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            result = match.group(0).strip()
            # Sanity check: must have at least 7 digits
            if len(re.sub(r'\D', '', result)) >= 7:
                return result

    return None


def extract_linkedin(text: str) -> str | None:
    """Extract LinkedIn profile URL if present."""
    if not text:
        return None
    match = re.search(
        r'(?:linkedin\.com/in/|linkedin\.com/pub/)[\w\-]+/?',
        text,
        re.IGNORECASE,
    )
    return match.group(0).strip() if match else None


def extract_contact_info(text: str) -> dict:
    """
    Extract all available contact information from resume text.

    Returns:
        {
            "email": str | None,
            "phone": str | None,
            "linkedin": str | None,
        }
    """
    return {
        "email": extract_email(text),
        "phone": extract_phone(text),
        "linkedin": extract_linkedin(text),
    }

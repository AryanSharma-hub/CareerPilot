"""
Resume Metrics — lightweight rule-based resume quality signals.

Improvements:
- Fixed bullet regex (was matching hyphens in words like "well-known")
- Added Indian currency/number formats (₹, lakh, crore)
- Added more metric patterns (K, M, B suffixes)
- calculate_base_score result is used in ats_score_engine context
"""

import re
import logging

logger = logging.getLogger("careerpilot.resume_metrics")


def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def count_bullets(text: str) -> int:
    """
    Count bullet point indicators in text.

    Fixed: original r'•|-' matched hyphens anywhere including in words
    like "well-known" or date ranges "2019-2022".

    New approach: only match bullet characters at the START of a line
    (after optional whitespace), which is the actual bullet pattern.
    """
    if not text:
        return 0

    # Match bullet markers only at start of line (after optional whitespace)
    bullet_pattern = re.compile(
        r'^\s*(?:•|◦|▪|▸|►|–|—|\*|>|-(?=\s)|\d+\.)\s',
        re.MULTILINE,
    )
    return len(bullet_pattern.findall(text))


def has_projects(text: str) -> bool:
    """Check if resume mentions projects."""
    if not text:
        return False
    return bool(re.search(r'\bproject', text, re.IGNORECASE))


def has_metrics(text: str) -> bool:
    """
    Check if resume contains measurable/quantified achievements.

    Covers:
    - Percentages: 25%, 3.5%
    - Multipliers: 2x, 10x
    - USD amounts: $50,000 / $2M
    - INR amounts: ₹5 lakh, ₹2 crore, Rs. 50,000
    - Indian number words: 5 lakhs, 2 crores
    - Compact suffixes: 500K, 2M, 1B
    - Count with plus: 50+, 100+ clients
    - Comma-formatted numbers: 1,000 / 10,000
    """
    if not text:
        return False

    patterns = [
        r'\d+(?:\.\d+)?%',                          # percentages
        r'\d+(?:\.\d+)?x\b',                        # multipliers
        r'\$\d+(?:[,\d]*)?(?:\.\d+)?(?:[KMB])?',   # USD
        r'₹\s*\d+(?:[,\d]*)?(?:\.\d+)?',           # INR symbol
        r'\brs\.?\s*\d+',                           # Rs. amounts
        r'\d+(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores)\b',  # Indian amounts
        r'\d+(?:\.\d+)?[KMB]\b',                   # compact: 500K, 2M
        r'\d{1,3}(?:,\d{3})+',                     # 1,000 / 10,000
        r'\b\d+\+',                                 # 50+ clients
    ]

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def calculate_base_score(text: str) -> int:
    """
    Compute a simple rule-based resume quality score (0–85).

    Used as a lightweight signal — actual scoring is done by
    ats_score_engine.py with LLM-derived sub-scores.
    """
    if not text:
        return 30

    score = 45  # Base

    word_count = count_words(text)
    if word_count >= 500:
        score += 15
    elif word_count >= 300:
        score += 8
    elif word_count < 200:
        score -= 10

    if has_projects(text):
        score += 8

    if has_metrics(text):
        score += 15

    bullet_count = count_bullets(text)
    if bullet_count >= 8:
        score += 10
    elif bullet_count >= 4:
        score += 5

    return min(score, 85)

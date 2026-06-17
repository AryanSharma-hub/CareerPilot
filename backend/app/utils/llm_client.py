"""
Centralized LLM client — retry logic, fallback, JSON repair, rate limit handling.
Primary model: google/gemini-flash-1.5
Fallback model: google/gemini-flash-1.5-8b
"""

import os
import re
import json
import time
import logging

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("careerpilot.llm")

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set in .env")
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


PRIMARY_MODEL  = "google/gemini-2.5-flash"
FALLBACK_MODEL = "google/gemini-flash-1.5"


def repair_json(raw: str) -> str:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


def safe_parse_json(raw: str, fallback=None):
    if fallback is None:
        fallback = {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(repair_json(raw))
        except json.JSONDecodeError as e:
            logger.error("JSON parse failed: %s | Raw: %.200s", e, raw)
            return fallback


def call_llm(prompt: str, temperature=0.1, max_tokens=3000,
             model=PRIMARY_MODEL, retries=3, retry_delay=2.0) -> str:
    client = get_client()
    last_error = None
    models_to_try = [model] + ([FALLBACK_MODEL] if model != FALLBACK_MODEL else [])

    for current_model in models_to_try:
        for attempt in range(1, retries + 1):
            try:
                logger.debug("LLM call | model=%s | attempt=%d", current_model, attempt)
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60,
                )
                content = response.choices[0].message.content
                if content:
                    return content
                logger.warning("Empty LLM response on attempt %d", attempt)
            except RateLimitError as e:
                wait = retry_delay * (2 ** attempt)
                logger.warning("Rate limit. Waiting %.1fs", wait)
                time.sleep(wait)
                last_error = e
            except (APITimeoutError, APIConnectionError) as e:
                logger.warning("API error attempt %d: %s", attempt, e)
                time.sleep(retry_delay)
                last_error = e
            except Exception as e:
                logger.error("Unexpected LLM error: %s", e)
                last_error = e
                break

    raise RuntimeError(f"LLM call failed. Last error: {last_error}")


def call_llm_json(prompt: str, fallback=None, temperature=0.1,
                  max_tokens=3000, model=PRIMARY_MODEL):
    if fallback is None:
        fallback = {}
    try:
        raw = call_llm(prompt=prompt, temperature=temperature,
                       max_tokens=max_tokens, model=model)
        return safe_parse_json(raw, fallback=fallback)
    except RuntimeError as e:
        logger.error("call_llm_json failed: %s", e)
        return fallback

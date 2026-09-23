import json
import logging
import requests

from config import settings, validate_ask_ai_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an intent parser for a commercial real estate property intelligence system.
Given a user question, output ONLY a raw JSON object matching this schema, with no markdown fences and no extra text:

{
  "intent": "search" | "nearest" | "compare" | "unknown",
  "property_names": ["string", ...],
  "nearby_count": integer or null
}

Rules:
- "search": the user wants to find or get info about one property.
- "nearest": the user wants nearby/nearest properties to one named property. Set nearby_count if a number is mentioned, otherwise null.
- "compare": the user wants to compare two or more named properties, or one property against its nearest neighbors.
- "unknown": the question does not fit any of the above categories.
- property_names must be extracted exactly as the user typed them. Never invent or guess a property name that was not mentioned.
- Respond with raw JSON only, nothing else."""


def _call_model(user_query: str) -> str:
    validate_ask_ai_settings()

    headers = {
        "Content-Type": "application/json",
        "api-key": settings.ASK_AI_API_KEY,
        "Authorization": f"Bearer {settings.ASK_AI_API_KEY}",
    }
    payload = {
        "model": settings.ASK_AI_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        "temperature": 0,
        "max_tokens": 300,
    }

    response = requests.post(
        settings.ASK_AI_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=settings.ASK_AI_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _empty_result() -> dict:
    return {"intent": "unknown", "property_names": [], "nearby_count": None}


def resolve_intent(user_query: str) -> dict:
    try:
        raw = _call_model(user_query)
    except Exception as e:
        logger.error("Ask AI model call failed: %s", e)
        return _empty_result()

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
    except Exception as e:
        logger.error("Ask AI returned unparseable JSON: %s | raw=%s", e, raw)
        return _empty_result()

    intent = parsed.get("intent", "unknown")
    if intent not in ("search", "nearest", "compare", "unknown"):
        intent = "unknown"

    names = parsed.get("property_names") or []
    if not isinstance(names, list):
        names = [str(names)]
    names = [str(n).strip() for n in names if str(n).strip()]

    nearby_count = parsed.get("nearby_count")
    try:
        nearby_count = int(nearby_count) if nearby_count is not None else None
    except (TypeError, ValueError):
        nearby_count = None

    return {"intent": intent, "property_names": names, "nearby_count": nearby_count}

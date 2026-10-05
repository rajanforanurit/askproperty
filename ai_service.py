import json
import logging
import re
import requests
import financial_service
from config import settings, validate_ask_ai_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an intent parser for a commercial real estate property intelligence system.
Convert the user's question into ONE raw JSON object and output nothing else: no markdown fences, no explanations.

Schema:
{
  "intent": "search" | "nearest" | "compare" | "list" | "unknown",
  "property_names": ["string", ...],
  "nearby_count": integer or null,
  "max_distance_miles": number or null,
  "constraints": [{"field": "string", "op": "string", "value": any}],
  "fields": ["string", ...],
  "unsupported": ["string", ...],
  "data_view": "actual" | "budget" | "both" | null
}

Intent rules:
- "search": find or get information about one named property.
- "nearest": find properties near one named property, including requests with a distance limit such as "within 10 miles" or "under 10 miles".
- "compare": compare named properties with each other, or compare one property with other properties (nearby, similar, or meeting conditions). If the user says "this property" or "the selected property" without a name, leave property_names empty.
- "list": the user wants properties matching conditions but names no reference property.
- "unknown": anything else.

Field rules:
- property_names: names exactly as the user typed them. Never invent or guess a name.
- nearby_count: the number of properties requested, otherwise null.
- max_distance_miles: a distance limit in miles when the user gives one, otherwise null. Convert kilometres to miles.
- constraints: conditions on property attributes. Use ONLY the field names listed below. Allowed op values: "eq", "ne", "in", "not_in", "gte", "lte", "gt", "lt", "between", "same_as_subject".
  Use "same_as_subject" (value null) for words like "similar" or "same market/type/size as the property". Use "between" with a two-item array. Use values exactly as listed when a list of values is given.
- fields: attribute names the user explicitly wants shown or compared (for example "compare their market and year built"), using ONLY the attribute names listed below. Use [] when the user names no specific attributes.
- unsupported: short descriptions of any condition the user asked for that cannot be expressed with the listed fields, for example a price or bedroom requirement. Never force such a condition into constraints.
- data_view: "actual" for actual results, "budget" for budget or forecast figures, "both" for actual versus budget, variance or over/under budget, otherwise null.

Available constraint fields:
{catalog}

Examples of the format (the names and values here are placeholders):
Question: Find properties within 10 miles of <Name A>
{"intent":"nearest","property_names":["<Name A>"],"nearby_count":null,"max_distance_miles":10,"constraints":[],"fields":[],"unsupported":[],"data_view":null}
Question: Compare <Name A> with 5 similar properties in the same market
{"intent":"compare","property_names":["<Name A>"],"nearby_count":5,"max_distance_miles":null,"constraints":[{"field":"Market","op":"same_as_subject","value":null}],"fields":["YearBuilt"],"unsupported":[],"data_view":null}
Question: Compare <Name A> and <Name B> budget versus actual
{"intent":"compare","property_names":["<Name A>","<Name B>"],"nearby_count":null,"max_distance_miles":null,"constraints":[],"fields":[],"unsupported":[],"data_view":"both"}

Respond with the JSON object only."""


def _build_prompt(catalog: str) -> str:
    text = catalog.strip() if catalog else "(none available)"
    return SYSTEM_PROMPT.replace("{catalog}", text)


def _call_model(user_query: str, catalog: str = "") -> str:
    validate_ask_ai_settings()

    headers = {
        "Content-Type": "application/json",
        "api-key": settings.ASK_AI_API_KEY,
        "Authorization": f"Bearer {settings.ASK_AI_API_KEY}",
    }
    payload = {
        "model": settings.ASK_AI_MODEL_NAME,
        "messages": [
            {"role": "system", "content": _build_prompt(catalog)},
            {"role": "user", "content": user_query},
        ],
        "temperature": 0,
        "max_tokens": settings.ASK_AI_MAX_TOKENS,
    }

    response = requests.post(
        settings.ASK_AI_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=settings.ASK_AI_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    return message.get("content") or ""


def _strip_reasoning(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"<think>", text, flags=re.IGNORECASE) and not re.search(r"</think>", text, flags=re.IGNORECASE):
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.replace("</think>", "")


def _extract_json(raw: str):
    text = _strip_reasoning(raw)
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except ValueError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


def _empty_result() -> dict:
    return {
        "intent": "unknown",
        "property_names": [],
        "nearby_count": None,
        "max_distance_miles": None,
        "constraints": [],
        "fields": [],
        "unsupported": [],
        "data_view": None,
    }


def _positive_int(value):
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number <= 0:
        return None
    return min(number, 500.0)


def resolve_intent(user_query: str, catalog: str = "") -> dict:
    try:
        raw = _call_model(user_query, catalog)
    except Exception as e:
        logger.error("Ask AI model call failed: %s", e)
        return _empty_result()

    parsed = _extract_json(raw)
    if parsed is None:
        logger.error("Ask AI returned no parseable JSON | raw=%s", raw[:500])
        return _empty_result()

    intent = parsed.get("intent", "unknown")
    if intent not in ("search", "nearest", "compare", "list", "unknown"):
        intent = "unknown"

    names = parsed.get("property_names") or []
    if not isinstance(names, list):
        names = [str(names)]
    names = [str(n).strip() for n in names if str(n).strip()]

    try:
        data_view = financial_service.normalize_view(parsed.get("data_view"))
    except ValueError:
        data_view = None

    constraints = parsed.get("constraints")
    fields = parsed.get("fields")
    unsupported = parsed.get("unsupported")

    return {
        "intent": intent,
        "property_names": names,
        "nearby_count": _positive_int(parsed.get("nearby_count")),
        "max_distance_miles": _positive_float(parsed.get("max_distance_miles")),
        "constraints": constraints if isinstance(constraints, list) else [],
        "fields": fields if isinstance(fields, list) else [],
        "unsupported": unsupported if isinstance(unsupported, list) else [],
        "data_view": data_view,
    }

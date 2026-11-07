import copy
import hashlib
import json
import logging
import re
import time
import requests
from requests.adapters import HTTPAdapter
import financial_service
import prompt_builder
from cache import KeyedTTLCache
from config import settings, validate_ask_ai_settings

logger = logging.getLogger(__name__)

# One pooled session: re-uses the TCP/TLS connection to the model endpoint instead of paying a fresh
# handshake (typically 100-400 ms) on every question.
_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=16))
_session.mount("http://", HTTPAdapter(pool_connections=4, pool_maxsize=16))

_intent_cache = KeyedTTLCache(
    ttl_seconds=max(1, settings.INTENT_CACHE_TTL_SECONDS),
    max_entries=settings.INTENT_CACHE_MAX_ENTRIES,
    name="intent",
)


def _extra_params() -> dict:
    raw = settings.ASK_AI_EXTRA_PARAMS
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        logger.warning("ASK_AI_EXTRA_PARAMS is not valid JSON; ignoring it.")
        return {}
    return value if isinstance(value, dict) else {}

class IntentServiceError(Exception):
    """The language-understanding step failed. `code` is safe to show to the user."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _build_prompt(user_query: str, catalog: str, context: str = "") -> str:
    return prompt_builder.build_dynamic_prompt(user_query, catalog, context)


_TOOL_NAME = "interpret_request"


def _output_params() -> dict:
    mode = settings.ASK_AI_OUTPUT_MODE
    if mode == "json_object":
        return {"response_format": {"type": "json_object"}}
    if mode == "tools":
        return {
            "tools": [{"type": "function", "function": {
                "name": _TOOL_NAME,
                "description": "Describe what the system should do for the user's question.",
                "parameters": prompt_builder.INTERPRETATION_SCHEMA,
            }}],
            "tool_choice": {"type": "function", "function": {"name": _TOOL_NAME}},
        }
    return {}


def _call_model(user_query: str, catalog: str = "", context: str = "", repair_of: str = "") -> str:
    validate_ask_ai_settings()

    headers = {
        "Content-Type": "application/json",
        "api-key": settings.ASK_AI_API_KEY,
        "Authorization": f"Bearer {settings.ASK_AI_API_KEY}",
    }
    messages = [
        {"role": "system", "content": _build_prompt(user_query, catalog, context)},
        {"role": "user", "content": user_query},
    ]
    if repair_of:
        messages += [
            {"role": "assistant", "content": repair_of[:1500]},
            {"role": "user", "content": "That was not a valid JSON object. Reply with the JSON object only."},
        ]
    payload = {
        "model": settings.ASK_AI_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": settings.ASK_AI_MAX_TOKENS,
    }
    payload.update(_output_params())
    payload.update(_extra_params())

    response = _session.post(
        settings.ASK_AI_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=settings.ASK_AI_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if calls:
        arguments = (calls[0].get("function") or {}).get("arguments")
        if isinstance(arguments, str) and arguments.strip():
            return arguments
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
        "metrics": [],
        "metric_constraints": [],
        "sort": None,
        "period": None,
        "visualization": None,
        "chart_metric": None,
        "location": None,
        "context_ref": "none",
        "new_topic": None,
        "include_nearby": False,
        "operation": None,
        "edit_names": [],
        "with_names": [],
    }


def _name_list(value) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(n).strip() for n in value if n is not None and str(n).strip()][:12]


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


def _is_transient(error: Exception) -> bool:
    if isinstance(error, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
        return error.response.status_code in (408, 409, 425, 429, 500, 502, 503, 504)
    return False


def _call_model_with_retry(user_query: str, catalog: str, context: str = "", attempts: int = 2, repair_of: str = "") -> str:
    for attempt in range(1, attempts + 1):
        try:
            return _call_model(user_query, catalog, context, repair_of)
        except Exception as e:
            if attempt >= attempts or not _is_transient(e):
                raise
            logger.warning("Ask AI model call failed (attempt %d/%d), retrying: %s", attempt, attempts, e)
            time.sleep(1.0)
    return ""


def resolve_intent(user_query: str, catalog: str = "", context: str = "") -> dict:
    """Parsed intent for a question. Identical (question, context, catalog) triples are served from memory."""
    if settings.INTENT_CACHE_TTL_SECONDS <= 0:
        return _resolve_intent_uncached(user_query, catalog, context)
    digest = hashlib.sha1((catalog or "").encode("utf-8")).hexdigest()
    key = (" ".join((user_query or "").lower().split()), (context or "").strip(), digest)
    # Callers mutate the returned dict, so every caller gets its own copy.
    return copy.deepcopy(_intent_cache.get(key, lambda: _resolve_intent_uncached(user_query, catalog, context)))


def _resolve_intent_uncached(user_query: str, catalog: str = "", context: str = "") -> dict:
    try:
        raw = _call_model_with_retry(user_query, catalog, context)
    except requests.exceptions.Timeout as e:
        logger.error("Ask AI model timed out: %s", type(e).__name__)
        raise IntentServiceError("ai_timeout")
    except RuntimeError as e:
        logger.error("Ask AI is not configured: %s", e)
        raise IntentServiceError("service_not_configured")
    except Exception as e:
        logger.error("Ask AI model call failed: %s: %s", type(e).__name__, e)
        raise IntentServiceError("ai_unavailable")

    parsed = _extract_json(raw)
    if parsed is None:
        # One format-repair attempt: models occasionally answer in prose or with broken JSON.
        try:
            raw = _call_model_with_retry(user_query, catalog, context, attempts=1, repair_of=raw)
            parsed = _extract_json(raw)
        except requests.exceptions.Timeout:
            raise IntentServiceError("ai_timeout")
        except Exception as e:
            logger.error("Ask AI repair call failed: %s: %s", type(e).__name__, e)
    if parsed is None:
        logger.error("Ask AI returned no parseable JSON | raw=%s", raw[:500])
        raise IntentServiceError("ai_bad_response")

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
    metrics = parsed.get("metrics")
    metric_constraints = parsed.get("metric_constraints")
    sort = parsed.get("sort")
    period = parsed.get("period")
    visualization = parsed.get("visualization")
    chart_metric = parsed.get("chart_metric")

    location = parsed.get("location")
    if isinstance(location, str):
        location = {"text": location, "type": None}
    if isinstance(location, dict) and str(location.get("text") or "").strip():
        loc_type = str(location.get("type") or "").strip().lower() or None
        location = {"text": str(location["text"]).strip()[:200], "type": loc_type}
    else:
        location = None
    context_ref = str(parsed.get("context_ref") or "none").strip().lower()
    if context_ref not in ("none", "results", "reference"):
        context_ref = "none"
    new_topic = parsed.get("new_topic")
    operation = str(parsed.get("operation") or "").strip().lower().replace("-", "_") or None
    if operation not in prompt_builder.OPERATIONS:
        operation = None
    edit_names = _name_list(parsed.get("edit_names"))
    with_names = _name_list(parsed.get("with_names"))
    if operation in ("remove", "add", "keep_only", "replace") and not edit_names and not with_names and names:
        # The model put the targets in property_names; for an edit operation that is what it meant.
        edit_names = list(names)
        names = []
    if operation in ("remove", "add", "keep_only", "replace"):
        names = []

    return {
        "location": location,
        "context_ref": context_ref,
        "new_topic": new_topic if isinstance(new_topic, bool) else None,
        "include_nearby": bool(parsed.get("include_nearby")),
        "operation": operation,
        "edit_names": edit_names,
        "with_names": with_names,
        "intent": intent,
        "property_names": names,
        "nearby_count": _positive_int(parsed.get("nearby_count")),
        "max_distance_miles": _positive_float(parsed.get("max_distance_miles")),
        "constraints": constraints if isinstance(constraints, list) else [],
        "fields": fields if isinstance(fields, list) else [],
        "unsupported": unsupported if isinstance(unsupported, list) else [],
        "data_view": data_view,
        "metrics": metrics if isinstance(metrics, list) else [],
        "metric_constraints": metric_constraints if isinstance(metric_constraints, list) else [],
        "sort": sort if isinstance(sort, dict) else None,
        "period": period if isinstance(period, dict) else None,
        "visualization": str(visualization).strip().lower() if isinstance(visualization, str) and visualization.strip() else None,
        "chart_metric": str(chart_metric).strip() if isinstance(chart_metric, str) and chart_metric.strip() else None,
    }

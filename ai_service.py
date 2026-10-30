import copy
import hashlib
import json
import logging
import re
import time
import requests
from requests.adapters import HTTPAdapter
import financial_service
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
  "data_view": "actual" | "budget" | "both" | null,
  "metrics": ["string", ...],
  "metric_constraints": [{"metric": "string", "op": "string", "value": any, "basis": "string"}],
  "sort": {"metric": "string", "direction": "asc" | "desc"} | null,
  "period": {"from": integer or null, "to": integer or null} | null,
  "visualization": "string" | null,
  "chart_metric": "string" | null,
  "location": {"text": "string", "type": "city" | "state" | "zip" | "address" | "county" | "neighborhood" | "region"} | null,
  "context_ref": "none" | "results" | "reference",
  "new_topic": true | false | null,
  "include_nearby": true | false,
  "response_focus": "locate" | "compare" | "financial" | "analysis" | null
}

Intent rules:
- "search": find or get information about one named property.
- "nearest": find properties near one named property, including requests with a distance limit such as "within 10 miles" or "under 10 miles".
- "compare": compare named properties with each other, or compare one property with other properties (nearby, similar, or meeting conditions). If the user says "this property" or "the selected property" without a name, leave property_names empty.
- "list": the user wants properties matching conditions (attributes or metrics) but names no reference property. Ranking requests such as "largest properties by unit count" are "list" with a sort.
- "unknown": anything else.

Location and conversation rules:
- location: a geographic place the user names instead of a property: a city, state, ZIP code, street address, county or neighborhood (for example "properties in Austin, TX", "near 123 Main St Dallas", "within 15 miles of 75201"). Copy the text exactly as typed. null when the user names no place. Never put a place into property_names.
- context_ref: "results" when the question refers to the properties already shown ("these properties", "compare them", "what about their financials"); "reference" when it refers to the original/anchor property or properties ("show nearby properties", "the original property"); otherwise "none". Use it ONLY when the conversation context below contains such properties.
- new_topic: true when the question is independent of the conversation context, false when it builds on it, null when there is no context.
- include_nearby: true when the user wants nearby properties included alongside the named or referenced ones (for example "compare these with nearby properties within 10 miles").
- response_focus: "locate" for finding/listing/showing properties (including nearby searches), "compare" for comparing properties, "financial" for questions mainly about financial figures, "analysis" for broad analysis across a portfolio, several states or markets. null if unclear.
- When the user names several states, cities or markets (\"across Texas and Florida\"), use a constraint such as State in [\"TX\",\"FL\"] instead of location, and set response_focus to \"analysis\".
- When the user only changes the distance ("what about 15 miles?"), set context_ref to "reference".
- Resolve references using the conversation context. Do NOT copy property names from the context into property_names; use context_ref instead.

Field rules:
- property_names: names exactly as the user typed them. Never invent or guess a name.
- nearby_count: the number of properties requested, otherwise null.
- max_distance_miles: a distance limit in miles when the user gives one, otherwise null. Convert kilometres to miles.
- constraints: conditions on property attributes. Use ONLY the field names listed below. Allowed op values: "eq", "ne", "in", "not_in", "gte", "lte", "gt", "lt", "between", "same_as_subject".
  Use "same_as_subject" (value null) for words like "similar" or "same market/type/size as the property". Use "between" with a two-item array. Use values exactly as listed when a list of values is given.
- fields: attribute names the user explicitly wants shown or compared (for example "compare their market and year built"), using ONLY the attribute names listed below. Use [] when the user names no specific attributes.
- unsupported: short descriptions of any condition the user asked for that cannot be expressed with the listed fields, for example a price or bedroom requirement. Never force such a condition into constraints.
- data_view: "actual" for actual results, "budget" for budget or forecast figures, "both" for actual versus budget, variance or over/under budget, otherwise null.

Metric rules (metrics come from unit data and are listed below the constraint fields):
- metrics: metric keys the user wants shown, for example "show unit counts" or "unit count and occupancy". Use [] otherwise.
- metric_constraints: conditions on a metric. op is one of "gte", "lte", "gt", "lt", "eq", "between". basis says what value means:
  "absolute" for a plain number (more than 200 units -> gt 200),
  "percent_of_subject" for a share of the reference property's value (at least 80% of its unit count -> gte 80),
  "percent_above_subject" for larger than the reference by a percentage (20% more units than <Name A> -> gte 20),
  "percent_below_subject" for smaller than the reference by a percentage,
  "percentile" for position among the candidate properties (top 10% by units -> gte 90).
  Relative bases need a reference property. If the user gives a percentage without a clear basis, choose the most natural one and keep it; never invent numbers.
- sort: ordering by a metric, for example "largest" -> {"metric":"units","direction":"desc"}, "smallest" -> asc. null when no ordering is asked.
- period: a YYYYMM month range only when the user names months or a year (for example "in 2026-09 to 2026-10" -> from 202609, to 202610; a single month uses the same value for both). null otherwise.
- visualization: "bar" or "pie" (or another listed output format) ONLY when the user explicitly asks for a chart, graph or plot. Otherwise null. Never choose a chart on your own.
- chart_metric: the metric key to chart. Default to "units" when a chart is requested without a metric.
- Never calculate or guess results. Only describe what to retrieve; the system executes the query against the data.

Available constraint fields:
{catalog}

Conversation context (may be empty):
{context}

Examples of the format (the names and values here are placeholders):
Question: Find properties within 10 miles of <Name A>
{"intent":"nearest","property_names":["<Name A>"],"nearby_count":null,"max_distance_miles":10,"constraints":[],"fields":[],"unsupported":[],"data_view":null}
Question: Compare <Name A> with 5 similar properties in the same market
{"intent":"compare","property_names":["<Name A>"],"nearby_count":5,"max_distance_miles":null,"constraints":[{"field":"Market","op":"same_as_subject","value":null}],"fields":["YearBuilt"],"unsupported":[],"data_view":null}
Question: Compare <Name A> and <Name B> budget versus actual
{"intent":"compare","property_names":["<Name A>","<Name B>"],"nearby_count":null,"max_distance_miles":null,"constraints":[],"fields":[],"unsupported":[],"data_view":"both"}

Question: Compare properties within 20 miles of <Name A> whose unit count is at least 20% above it as a bar chart
{"intent":"compare","property_names":["<Name A>"],"nearby_count":null,"max_distance_miles":20,"constraints":[],"fields":[],"unsupported":[],"data_view":null,"metrics":["units"],"metric_constraints":[{"metric":"units","op":"gte","value":20,"basis":"percent_above_subject"}],"sort":null,"period":null,"visualization":"bar","chart_metric":"units"}
Question: Show the 10 largest properties by unit count in <a state value> as a pie chart
{"intent":"list","property_names":[],"nearby_count":10,"max_distance_miles":null,"constraints":[{"field":"State","op":"eq","value":"<a state value>"}],"fields":[],"unsupported":[],"data_view":null,"metrics":["units"],"metric_constraints":[],"sort":{"metric":"units","direction":"desc"},"period":null,"visualization":"pie","chart_metric":"units"}

Question: Compare these properties with nearby properties within 10 miles   (context lists properties already shown)
{"intent":"compare","property_names":[],"nearby_count":null,"max_distance_miles":10,"constraints":[],"fields":[],"unsupported":[],"data_view":null,"location":null,"context_ref":"results","new_topic":false,"include_nearby":true,"response_focus":"compare"}
Question: Find properties in Austin, TX
{"intent":"list","property_names":[],"nearby_count":null,"max_distance_miles":null,"constraints":[],"fields":[],"unsupported":[],"data_view":null,"location":{"text":"Austin, TX","type":"city"},"context_ref":"none","new_topic":null,"include_nearby":false,"response_focus":"locate"}

Respond with the JSON object only."""


class IntentServiceError(Exception):
    """The language-understanding step failed. `code` is safe to show to the user."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _build_prompt(catalog: str, context: str = "") -> str:
    text = catalog.strip() if catalog else "(none available)"
    ctx = context.strip() if context else "(none)"
    return SYSTEM_PROMPT.replace("{catalog}", text).replace("{context}", ctx)


def _call_model(user_query: str, catalog: str = "", context: str = "") -> str:
    validate_ask_ai_settings()

    headers = {
        "Content-Type": "application/json",
        "api-key": settings.ASK_AI_API_KEY,
        "Authorization": f"Bearer {settings.ASK_AI_API_KEY}",
    }
    payload = {
        "model": settings.ASK_AI_MODEL_NAME,
        "messages": [
            {"role": "system", "content": _build_prompt(catalog, context)},
            {"role": "user", "content": user_query},
        ],
        "temperature": 0,
        "max_tokens": settings.ASK_AI_MAX_TOKENS,
    }
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
        "response_focus": None,
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


def _is_transient(error: Exception) -> bool:
    if isinstance(error, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
        return error.response.status_code in (408, 409, 425, 429, 500, 502, 503, 504)
    return False


def _call_model_with_retry(user_query: str, catalog: str, context: str = "", attempts: int = 2) -> str:
    for attempt in range(1, attempts + 1):
        try:
            return _call_model(user_query, catalog, context)
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
    focus = str(parsed.get("response_focus") or "").strip().lower() or None
    if focus not in ("locate", "compare", "financial", "analysis"):
        focus = None

    return {
        "location": location,
        "context_ref": context_ref,
        "new_topic": new_topic if isinstance(new_topic, bool) else None,
        "include_nearby": bool(parsed.get("include_nearby")),
        "response_focus": focus,
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

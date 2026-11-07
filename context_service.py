"""Conversation state carried by the client between requests (the backend stays stateless).

The state describes the LAST SUCCESSFUL result and how it was built, so every follow-up can operate on it:

  reference_keys / reference_point   the anchor of the search (a property, or a geocoded address/place)
  pool_keys                          everything the search retrieved (plus properties the user added)
  base_keys                          the selection the search started with (used by "start over")
  result_keys                        the ACTIVE selection, reference properties included
  excluded_keys                      properties the user removed
  constraints / metric_constraints / sort / count     refinements applied on top of the pool
  location / radius_miles / period / metrics / fields / chart             the rest of the search context

A failed or empty answer never changes it: `preserved()` hands back the previous state untouched.
"""
import re
from typing import Optional

STATE_VERSION = 2
MAX_KEYS = 300
MAX_QUERIES = 4

EDIT_OPERATIONS = ("remove", "add", "replace", "keep_only")
STATE_OPERATIONS = EDIT_OPERATIONS + ("refine", "analyze", "reset")
# Operations that make no sense without existing results; the others fall back to a new search.
NEEDS_STATE = ("remove", "replace", "keep_only", "reset")

PRONOUN_RE = re.compile(
    r"\b(these|those|them|their|they|it|its|same|that property|this property|the selected|selected propert\w+|"
    r"the above|previous|earlier|original)\b",
    re.IGNORECASE,
)
NEARBY_RE = re.compile(r"\b(nearby|near|around|within|surrounding|closest|nearest)\b", re.IGNORECASE)


def _keys(value) -> list:
    if not isinstance(value, list):
        return []
    seen, out = set(), []
    for item in value:
        key = str(item).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
        if len(out) >= MAX_KEYS:
            break
    return out


def _dicts(value, limit: int = 8) -> list:
    return [v for v in value if isinstance(v, dict)][:limit] if isinstance(value, list) else []


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _point(raw) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    lat, lng = _number(raw.get("lat")), _number(raw.get("lng"))
    if lat is None or lng is None or not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return None
    return {"lat": lat, "lng": lng, "label": str(raw.get("label") or "")[:200], "source": str(raw.get("source") or "")[:20]}


def clean(raw) -> Optional[dict]:
    """Validate client-supplied context. Anything unexpected is dropped, never trusted."""
    if not isinstance(raw, dict):
        return None
    ctx = {
        "v": STATE_VERSION,
        "original_query": str(raw.get("original_query") or "")[:500],
        "last_query": str(raw.get("last_query") or "")[:500],
        "topic_queries": [str(q)[:300] for q in (raw.get("topic_queries") or [])[-MAX_QUERIES:]]
        if isinstance(raw.get("topic_queries"), list) else [],
        "reference_keys": _keys(raw.get("reference_keys")),
        "reference_point": _point(raw.get("reference_point")),
        "result_keys": _keys(raw.get("result_keys")),
        "pool_keys": _keys(raw.get("pool_keys")),
        "base_keys": _keys(raw.get("base_keys")),
        "excluded_keys": _keys(raw.get("excluded_keys")),
        "radius_miles": None,
        "location": None,
        "constraints": _dicts(raw.get("constraints")),
        "metric_constraints": _dicts(raw.get("metric_constraints")),
        "sort": raw.get("sort") if isinstance(raw.get("sort"), dict) else None,
        "count": None,
        "intent": str(raw.get("intent") or "")[:20],
        "period": raw.get("period") if isinstance(raw.get("period"), dict) else None,
        "metrics": [str(m)[:40] for m in raw.get("metrics", [])][:8] if isinstance(raw.get("metrics"), list) else [],
        "fields": [str(m)[:40] for m in raw.get("fields", [])][:16] if isinstance(raw.get("fields"), list) else [],
        "chart": raw.get("chart") if isinstance(raw.get("chart"), dict) else None,
        "last_operation": str(raw.get("last_operation") or "")[:20],
    }
    radius = _number(raw.get("radius_miles"))
    ctx["radius_miles"] = radius if radius and 0 < radius <= 500 else None
    count = _number(raw.get("count"))
    ctx["count"] = int(count) if count and count > 0 else None
    loc = raw.get("location")
    if isinstance(loc, dict) and loc.get("requested"):
        ctx["location"] = {"requested": str(loc["requested"])[:200], "type": str(loc.get("type") or "")[:20]}
    refs = set(ctx["reference_keys"])
    if not ctx["pool_keys"]:
        ctx["pool_keys"] = [k for k in ctx["result_keys"] if k not in refs]
    if not ctx["base_keys"]:
        ctx["base_keys"] = [k for k in ctx["result_keys"] if k not in refs]
    if not ctx["reference_keys"] and not ctx["result_keys"] and not ctx["location"] and not ctx["reference_point"]:
        return None
    return ctx


def active_keys(ctx: Optional[dict]) -> list:
    """The current selection without the reference properties."""
    if not ctx:
        return []
    refs = set(ctx["reference_keys"])
    return [k for k in ctx["result_keys"] if k not in refs]


def has_selection(ctx: Optional[dict]) -> bool:
    return bool(ctx and (ctx["result_keys"] or ctx["reference_keys"] or ctx["pool_keys"]))


def describe_filters(constraints: list, metric_constraints: list) -> list:
    ops = {"eq": "=", "ne": "!=", "in": "in", "not_in": "not in", "gte": ">=", "lte": "<=", "gt": ">", "lt": "<", "between": "between"}
    out = []
    for c in constraints or []:
        value = c.get("value")
        shown = ", ".join(map(str, value)) if isinstance(value, list) else value
        out.append(f"{c.get('field')} {ops.get(c.get('op'), c.get('op'))} {shown}" if c.get("op") != "same_as_subject"
                   else f"{c.get('field')} same as the reference")
    for c in metric_constraints or []:
        value = c.get("value")
        shown = " to ".join(f"{v:g}" for v in value) if isinstance(value, list) else f"{value:g}"
        basis = "" if c.get("basis") == "absolute" else f" ({str(c.get('basis')).replace('_', ' ')})"
        out.append(f"{c.get('metric')} {ops.get(c.get('op'), c.get('op'))} {shown}{basis}")
    return out


def prompt_summary(ctx: Optional[dict], name_of, hints: Optional[list] = None) -> str:
    """The compact state block for the intent parser: only what a follow-up needs to be understood."""
    if not ctx:
        return ""

    def names(keys, limit=12):
        shown = [n for n in (name_of(k) for k in keys[:limit]) if n]
        extra = len(keys) - len(shown)
        return ", ".join(shown) + (f" (+{extra} more)" if extra > 0 else "")

    lines = []
    if ctx["original_query"]:
        lines.append(f"Original question: {ctx['original_query']}")
    if ctx["last_query"] and ctx["last_query"] != ctx["original_query"]:
        lines.append(f"Previous question: {ctx['last_query']}")
    if ctx["reference_keys"]:
        lines.append(f"Reference property ({len(ctx['reference_keys'])}, always kept): {names(ctx['reference_keys'])}")
    elif ctx["reference_point"]:
        lines.append(f"Reference point (always kept): {ctx['reference_point'].get('label') or 'a map location'}")
    if ctx["location"]:
        lines.append(f"Search location: {ctx['location']['requested']}")
    if ctx["radius_miles"]:
        lines.append(f"Search radius: {ctx['radius_miles']:g} miles")
    active = active_keys(ctx)
    if active:
        lines.append(f"Current selection ({len(active)} of {len(ctx['pool_keys'])} retrieved): {names(active)}")
    if ctx["excluded_keys"]:
        lines.append(f"Removed by the user: {names(ctx['excluded_keys'], 8)}")
    filters = describe_filters(ctx["constraints"], ctx["metric_constraints"])
    if filters:
        lines.append("Active filters: " + "; ".join(filters))
    for hint in hints or []:
        lines.append(hint)
    return "\n".join(lines)


def decide(parsed: dict, ctx: Optional[dict], query: str) -> dict:
    """Decide whether this question builds on the conversation or starts a new topic."""
    operation = parsed.get("operation")
    if operation in NEEDS_STATE and (not ctx or not has_selection(ctx)):
        # Nothing to edit: the executor explains this instead of silently running a different search.
        return {"use": "state", "new_topic": False, "refine": False, "operation": operation}
    if not ctx:
        return {"use": "none", "new_topic": True, "refine": False, "operation": "new"}
    if operation in STATE_OPERATIONS:
        return {"use": "state", "new_topic": False, "refine": False, "operation": operation}

    use = parsed.get("context_ref") or "none"
    if use not in ("none", "results", "reference"):
        use = "none"
    has_anchor = bool(parsed.get("property_names") or parsed.get("location"))

    if use == "none" and not has_anchor and PRONOUN_RE.search(query or ""):
        wants_nearby = parsed.get("intent") == "nearest" or NEARBY_RE.search(query or "")
        use = "reference" if wants_nearby and (ctx["reference_keys"] or ctx["reference_point"]) else "results"

    if use == "results" and (ctx["reference_keys"] or ctx["reference_point"]) and ctx["radius_miles"] and parsed.get("max_distance_miles"):
        # "what about 15 miles?" re-measures from the original reference, not from the previous matches.
        use = "reference"

    if use == "reference" and not (ctx["reference_keys"] or ctx["reference_point"]):
        use = "results"
    if use == "results" and not ctx["result_keys"]:
        use = "reference" if ctx["reference_keys"] else "none"

    if use != "none":
        return {"use": use, "new_topic": False, "refine": False, "operation": "new"}

    explicit = parsed.get("new_topic")
    new_topic = True if has_anchor else (True if explicit is None else bool(explicit))
    refine = (not new_topic) and parsed.get("intent") == "list"
    return {"use": "none", "new_topic": new_topic, "refine": refine, "operation": "new"}


def keys_for(ctx: Optional[dict], use: str) -> list:
    if not ctx or use in ("none", "state"):
        return []
    if use == "reference":
        return ctx["reference_keys"]
    return ctx["result_keys"] or ctx["reference_keys"]


def merge_constraints(old: list, new: list) -> list:
    fields = {c.get("field") for c in new if isinstance(c, dict)}
    kept = [c for c in old if isinstance(c, dict) and c.get("field") not in fields]
    return kept + list(new)


def merge_metric_constraints(old: list, new: list) -> list:
    metrics = {c.get("metric") for c in new if isinstance(c, dict)}
    kept = [c for c in old if isinstance(c, dict) and c.get("metric") not in metrics]
    return kept + list(new)


def preserved(prev: Optional[dict]) -> Optional[dict]:
    """A failed or empty answer leaves the last successful state exactly as it was."""
    return prev


def _origin_point(location: Optional[dict]) -> Optional[dict]:
    origin = (location or {}).get("origin")
    return _point({**origin, "label": origin.get("label") or location.get("requested")}) if isinstance(origin, dict) else None


def next_context(prev: Optional[dict], decision: dict, query: str, plan: dict,
                 reference_keys: list, result_keys: list, location: Optional[dict],
                 pool_keys: Optional[list] = None, patch: Optional[dict] = None) -> dict:
    continuing = bool(prev) and not decision["new_topic"]
    # A re-search around the same reference ("what about 15 miles?") keeps the reference and the filters but
    # retrieves a fresh pool, so earlier removals no longer apply to it.
    same_pool = continuing and decision.get("use") != "reference"
    topic = (prev["topic_queries"] if continuing and prev else [])[-(MAX_QUERIES - 1):] + [query[:300]]
    refs = _keys(reference_keys) or (prev["reference_keys"] if continuing and prev else [])
    point = _origin_point(location) or (prev["reference_point"] if continuing and prev else None)
    results = _keys(result_keys)
    ref_set = set(refs)
    chosen = [k for k in results if k not in ref_set]
    if pool_keys is not None:
        pool = _keys(pool_keys)
    elif same_pool and prev:
        pool = _keys(prev["pool_keys"] + chosen)
    else:
        pool = chosen
    ctx = {
        "v": STATE_VERSION,
        "original_query": prev["original_query"] if continuing and prev and prev["original_query"] else query[:500],
        "last_query": query[:500],
        "topic_queries": topic,
        "reference_keys": refs,
        "reference_point": point,
        "result_keys": _keys(refs + chosen),
        "pool_keys": _keys([k for k in pool if k not in ref_set]),
        "base_keys": (prev["base_keys"] if same_pool and prev and prev["base_keys"] else chosen),
        "excluded_keys": prev["excluded_keys"] if same_pool and prev else [],
        "radius_miles": plan.get("radius") or (prev["radius_miles"] if continuing and prev else None),
        "location": {"requested": location["requested"], "type": location["type"]} if location else
        (prev["location"] if continuing and prev else None),
        "constraints": plan.get("constraints") or (prev["constraints"] if continuing and prev else []),
        "metric_constraints": plan.get("metric_constraints") or (prev["metric_constraints"] if continuing and prev else []),
        "sort": plan.get("sort") or (prev["sort"] if continuing and prev else None),
        "count": plan.get("count") or (prev["count"] if continuing and prev else None),
        "intent": plan.get("intent") or "",
        "period": {"from": plan.get("ym_from"), "to": plan.get("ym_to")} if plan.get("ym_from") else (prev["period"] if continuing and prev else None),
        "metrics": plan.get("display_metrics") or (prev["metrics"] if continuing and prev else []),
        "fields": plan.get("fields") or (prev["fields"] if continuing and prev else []),
        "chart": {"type": plan["chart"], "metric": plan.get("chart_metric")} if plan.get("chart") else (prev["chart"] if continuing and prev else None),
        "last_operation": plan.get("operation") or "new",
    }
    if patch:
        ctx.update(patch)
        ctx["result_keys"] = _keys(ctx["reference_keys"] + [k for k in ctx["result_keys"] if k not in set(ctx["reference_keys"])])
    return ctx


def after_manual_selection(prev: Optional[dict], selected_keys: list) -> Optional[dict]:
    """The user ticked/unticked properties: keep the state coherent so the next question builds on what they see."""
    if not prev:
        return None
    refs = set(prev["reference_keys"])
    keys = [k for k in _keys(selected_keys) if k not in refs]
    before = set(active_keys(prev))
    removed_now = before - set(keys)
    excluded = [k for k in prev["excluded_keys"] if k not in set(keys)] + [k for k in removed_now if k not in prev["excluded_keys"]]
    out = dict(prev)
    out.update({
        "result_keys": _keys(prev["reference_keys"] + keys),
        "pool_keys": _keys(prev["pool_keys"] + [k for k in keys if k not in set(prev["pool_keys"])]),
        "excluded_keys": _keys(excluded),
    })
    return out

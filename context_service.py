"""Conversation context carried by the client between requests (stateless backend)."""
import re
from typing import Optional

MAX_KEYS = 300
MAX_QUERIES = 4

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


def clean(raw) -> Optional[dict]:
    """Validate client-supplied context. Anything unexpected is dropped, never trusted."""
    if not isinstance(raw, dict):
        return None
    ctx = {
        "original_query": str(raw.get("original_query") or "")[:500],
        "last_query": str(raw.get("last_query") or "")[:500],
        "topic_queries": [str(q)[:300] for q in (raw.get("topic_queries") or [])[-MAX_QUERIES:]]
        if isinstance(raw.get("topic_queries"), list) else [],
        "reference_keys": _keys(raw.get("reference_keys")),
        "result_keys": _keys(raw.get("result_keys")),
        "radius_miles": None,
        "location": None,
        "constraints": raw.get("constraints") if isinstance(raw.get("constraints"), list) else [],
        "intent": str(raw.get("intent") or "")[:20],
        "period": raw.get("period") if isinstance(raw.get("period"), dict) else None,
    }
    try:
        radius = float(raw.get("radius_miles"))
        ctx["radius_miles"] = radius if 0 < radius <= 500 else None
    except (TypeError, ValueError):
        pass
    loc = raw.get("location")
    if isinstance(loc, dict) and loc.get("requested"):
        ctx["location"] = {"requested": str(loc["requested"])[:200], "type": str(loc.get("type") or "")[:20]}
    if not ctx["reference_keys"] and not ctx["result_keys"] and not ctx["location"]:
        return None
    return ctx


def prompt_summary(ctx: Optional[dict], name_of) -> str:
    """Short text the intent parser can use to resolve 'these properties', 'compare them', etc."""
    if not ctx:
        return ""
    def names(keys):
        shown = [name_of(k) for k in keys[:12]]
        shown = [n for n in shown if n]
        extra = len(keys) - len(shown)
        return ", ".join(shown) + (f" (+{extra} more)" if extra > 0 else "")
    lines = []
    if ctx["original_query"]:
        lines.append(f"Original question: {ctx['original_query']}")
    if ctx["last_query"] and ctx["last_query"] != ctx["original_query"]:
        lines.append(f"Previous question: {ctx['last_query']}")
    if ctx["reference_keys"]:
        lines.append(f"Reference properties ({len(ctx['reference_keys'])}): {names(ctx['reference_keys'])}")
    if ctx["result_keys"]:
        lines.append(f"Properties currently shown ({len(ctx['result_keys'])}): {names(ctx['result_keys'])}")
    if ctx["location"]:
        lines.append(f"Location in context: {ctx['location']['requested']}")
    if ctx["radius_miles"]:
        lines.append(f"Distance in context: {ctx['radius_miles']:g} miles")
    return "\n".join(lines)


def decide(parsed: dict, ctx: Optional[dict], query: str) -> dict:
    """Decide whether this question builds on the conversation or starts a new topic."""
    if not ctx:
        return {"use": "none", "new_topic": True, "refine": False}

    use = parsed.get("context_ref") or "none"
    if use not in ("none", "results", "reference"):
        use = "none"
    has_anchor = bool(parsed.get("property_names") or parsed.get("location"))

    if use == "none" and not has_anchor and PRONOUN_RE.search(query or ""):
        wants_nearby = parsed.get("intent") == "nearest" or NEARBY_RE.search(query or "")
        use = "reference" if wants_nearby and ctx["reference_keys"] else "results"

    if use == "results" and ctx["reference_keys"] and ctx["radius_miles"] and parsed.get("max_distance_miles"):
        # "what about 15 miles?" re-measures from the original reference, not from the previous matches.
        use = "reference"

    if use == "reference" and not ctx["reference_keys"]:
        use = "results"
    if use == "results" and not ctx["result_keys"]:
        use = "reference" if ctx["reference_keys"] else "none"

    if use != "none":
        return {"use": use, "new_topic": False, "refine": False}

    explicit = parsed.get("new_topic")
    new_topic = True if has_anchor else (True if explicit is None else bool(explicit))
    refine = (not new_topic) and parsed.get("intent") == "list"
    return {"use": "none", "new_topic": new_topic, "refine": refine}


def keys_for(ctx: Optional[dict], use: str) -> list:
    if not ctx or use == "none":
        return []
    if use == "reference":
        return ctx["reference_keys"] or ctx["result_keys"]
    return ctx["result_keys"] or ctx["reference_keys"]


def merge_constraints(old: list, new: list) -> list:
    fields = {c.get("field") for c in new if isinstance(c, dict)}
    kept = [c for c in old if isinstance(c, dict) and c.get("field") not in fields]
    return kept + list(new)


def next_context(prev: Optional[dict], decision: dict, query: str, plan: dict,
                 reference_keys: list, result_keys: list, location: Optional[dict]) -> dict:
    continuing = bool(prev) and not decision["new_topic"]
    topic = (prev["topic_queries"] if continuing and prev else [])[-(MAX_QUERIES - 1):] + [query[:300]]
    return {
        "original_query": prev["original_query"] if continuing and prev and prev["original_query"] else query[:500],
        "last_query": query[:500],
        "topic_queries": topic,
        "reference_keys": _keys(reference_keys),
        "result_keys": _keys(result_keys),
        "radius_miles": plan.get("radius") or (prev["radius_miles"] if continuing and prev else None),
        "location": {"requested": location["requested"], "type": location["type"]} if location else
        (prev["location"] if continuing and prev else None),
        "constraints": plan.get("constraints") or [],
        "intent": plan.get("intent") or "",
        "period": {"from": plan.get("ym_from"), "to": plan.get("ym_to")} if plan.get("ym_from") else None,
    }

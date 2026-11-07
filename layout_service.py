"""Decide which response sections serve the user's question."""
import re

FINANCIAL_RE = re.compile(
    r"\b(financ\w*|revenue|income|noi|expense\w*|budget\w*|variance|p&l|profit\w*|cash ?flow|rent roll|"
    r"over budget|under budget|actuals?)\b", re.IGNORECASE)
COMPARE_RE = re.compile(r"\b(compare\w*|versus|vs\.?|benchmark\w*|analy[sz]\w*|against|stack up|performance)\b", re.IGNORECASE)
BROAD_RE = re.compile(r"\b(across|each state|by state|per state|state[- ]wise|different states|portfolio)\b", re.IGNORECASE)
LOCATE_RE = re.compile(r"\b(find|list|show|nearby|near|where|locate|within|located|properties (in|near|around))\b", re.IGNORECASE)

FOCUSES = ("locate", "compare", "financial", "analysis")


def infer_focus(query: str, intent: str, parsed_focus: str = None, has_view: bool = False) -> str:
    if parsed_focus in FOCUSES:
        return parsed_focus
    text = query or ""
    financial = bool(FINANCIAL_RE.search(text)) or has_view
    compare = bool(COMPARE_RE.search(text)) or intent == "compare"
    if BROAD_RE.search(text) or (financial and compare):
        return "analysis"
    if financial:
        return "financial"
    if compare:
        return "compare"
    return "locate"


SECTIONS = ("map", "financial", "properties")


def sections_for(focus: str, intent: str, has_geography: bool, single_property: bool) -> list:
    """Whenever there is something to show, ALL sections are required. The model never decides this;
    `focus` only says which one is emphasised, it never removes one."""
    if intent == "unknown":
        return []
    return list(SECTIONS)


def classify(query: str, intent: str, parsed_focus: str = None, has_view: bool = False,
             has_geography: bool = False, single_property: bool = False) -> dict:
    focus = infer_focus(query, intent, parsed_focus, has_view)
    return {"focus": focus, "sections": sections_for(focus, intent, has_geography, single_property)}


def enforce(layout: dict, response: dict) -> dict:
    """Application-level completeness contract, applied to every response that has properties:
    map + property list + expense comparison are always listed, and `complete` reports what is ready."""
    rows = (response.get("table") or {}).get("rows") or []
    if not rows:
        return {"focus": layout.get("focus", ""), "sections": [], "complete": {}}
    if response.get("financials"):
        financial = "ready"
    elif response.get("financials_error"):
        financial = "error"
    elif response.get("financials_pending"):
        financial = "pending"
    else:
        financial = "missing"
    location = response.get("location") or {}
    mapped = int((response.get("task") or {}).get("plotted") or 0) > 0 or bool(location.get("origin"))
    return {
        "focus": layout.get("focus", ""),
        "sections": list(SECTIONS),
        "complete": {"map": mapped, "properties": True, "financial": financial},
    }

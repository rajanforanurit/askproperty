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


def sections_for(focus: str, intent: str, has_geography: bool, single_property: bool) -> list:
    """Sections use the UI's names: map, financial, properties."""
    if intent == "unknown":
        return []
    if focus == "locate":
        return ["map", "properties"]
    if focus == "financial":
        return ["financial", "properties"] + (["map"] if has_geography else [])
    if focus in ("compare", "analysis"):
        return ["map", "financial", "properties"]
    return ["map", "properties"]


def classify(query: str, intent: str, parsed_focus: str = None, has_view: bool = False,
             has_geography: bool = False, single_property: bool = False) -> dict:
    focus = infer_focus(query, intent, parsed_focus, has_view)
    return {"focus": focus, "sections": sections_for(focus, intent, has_geography, single_property)}

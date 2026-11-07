"""Builds the intent-parser prompt for ONE question from only the context that question needs.

The model's job is narrow: understand the wording and describe what to do (JSON). Everything deterministic
(resolving names to property ids, applying filters, editing the selection, assembling the response) is done
by the backend. Instead of one ever-growing static prompt, the prompt is assembled from small modules:

  core           always: role, output contract, field schema
  search         always: intent / location / field / metric rules
  follow_up      only when the conversation already has results: operation rules
  state block    only when the conversation already has results: reference, selection, filters
  examples       a few worked examples picked by similarity to the question
"""
import re

OPERATIONS = ("new", "refine", "remove", "add", "replace", "keep_only", "analyze", "reset")

# Single source of truth for the structured output. Used for "tools" / "json_object" modes and checked against
# the schema text below by the test-suite so the two cannot drift.
INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["search", "nearest", "compare", "list", "unknown"]},
        "operation": {"type": ["string", "null"], "enum": list(OPERATIONS) + [None]},
        "property_names": {"type": "array", "items": {"type": "string"}},
        "edit_names": {"type": "array", "items": {"type": "string"}},
        "with_names": {"type": "array", "items": {"type": "string"}},
        "nearby_count": {"type": ["integer", "null"]},
        "max_distance_miles": {"type": ["number", "null"]},
        "constraints": {"type": "array", "items": {"type": "object"}},
        "fields": {"type": "array", "items": {"type": "string"}},
        "unsupported": {"type": "array", "items": {"type": "string"}},
        "data_view": {"type": ["string", "null"], "enum": ["actual", "budget", "both", None]},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "metric_constraints": {"type": "array", "items": {"type": "object"}},
        "sort": {"type": ["object", "null"]},
        "period": {"type": ["object", "null"]},
        "visualization": {"type": ["string", "null"]},
        "chart_metric": {"type": ["string", "null"]},
        "location": {"type": ["object", "null"]},
        "context_ref": {"type": "string", "enum": ["none", "results", "reference"]},
        "new_topic": {"type": ["boolean", "null"]},
        "include_nearby": {"type": "boolean"},
    },
    "required": ["intent"],
}

CORE = """You are an intent parser for a commercial real estate property intelligence system.
Convert the user's question into ONE raw JSON object and output nothing else: no markdown fences, no explanations.
You only DESCRIBE what to do. The system resolves property ids, applies filters, edits the selection, retrieves
data and builds the response. Never calculate or guess results, never invent names, never choose which parts of
the answer to show.

Schema (omit fields that do not apply):
{
  "intent": "search" | "nearest" | "compare" | "list" | "unknown",
  "property_names": ["string"],
  "nearby_count": integer or null,
  "max_distance_miles": number or null,
  "constraints": [{"field": "string", "op": "string", "value": any}],
  "fields": ["string"],
  "unsupported": ["string"],
  "data_view": "actual" | "budget" | "both" | null,
  "metrics": ["string"],
  "metric_constraints": [{"metric": "string", "op": "string", "value": any, "basis": "string"}],
  "sort": {"metric": "string", "direction": "asc" | "desc"} | null,
  "period": {"from": integer or null, "to": integer or null} | null,
  "visualization": "string" | null,
  "chart_metric": "string" | null,
  "location": {"text": "string", "type": "city" | "state" | "zip" | "address" | "county" | "neighborhood" | "region"} | null__FOLLOW_SCHEMA__
}"""

FOLLOW_SCHEMA = """,
  "operation": "new" | "refine" | "remove" | "add" | "replace" | "keep_only" | "analyze" | "reset",
  "edit_names": ["string"],
  "with_names": ["string"],
  "context_ref": "none" | "results" | "reference",
  "new_topic": true | false,
  "include_nearby": true | false"""

SEARCH = """
Intent rules:
- "search": information about one named property. "nearest": properties near one named property, including a distance limit.
- "compare": compare named properties, or one property with others (nearby, similar, meeting conditions). "this property" without a name -> property_names empty.
- "list": properties matching conditions with no reference property; rankings ("largest by units") are "list" with a sort.
- "unknown": anything else.

Location: a geographic place named instead of a property (city, state, ZIP, street address, county, neighborhood), copied exactly as typed.
null when no place is named. Never put a place into property_names. Several states/markets -> a State "in" constraint, not location.

Field rules:
- property_names: names exactly as typed. nearby_count: number requested. max_distance_miles: convert km to miles.
- constraints: ONLY the fields listed below. ops: eq, ne, in, not_in, gte, lte, gt, lt, between (two-item array), same_as_subject (value null; for "similar"/"same market"). Use listed values exactly.
- fields: attributes the user wants shown or compared, from the list below; [] otherwise.
- unsupported: conditions that cannot be expressed with the listed fields (price, bedrooms). Never force them into constraints.
- data_view: "actual", "budget", or "both" (actual vs budget, variance); null otherwise.

Metric rules (from unit data):
- metrics: keys to show. metric_constraints op: gte, lte, gt, lt, eq, between. basis: "absolute" (more than 200 units -> gt 200), "percent_of_subject", "percent_above_subject", "percent_below_subject" (relative to the reference property), "percentile" (top 10% -> gte 90).
- sort: {"metric":"units","direction":"desc"} for "largest", asc for "smallest"; null otherwise.
- period: YYYYMM range only when months/years are named. visualization: "bar"/"pie" ONLY when a chart is explicitly requested; chart_metric defaults to "units".
"""

FOLLOW_UP = """
This conversation already has results (see "Conversation state"). Decide how the question relates to them and set "operation":
- "remove": take named properties out of the current selection. edit_names = names as typed.
- "add": put named properties into the selection. edit_names = names as typed.
- "replace": swap properties: edit_names = the ones to remove, with_names = the ones to add.
- "keep_only": the selection should be exactly the named properties ("just A and B", "only compare these two"). edit_names = names.
- "refine": narrow or reorder the CURRENT results by conditions, count or sort ("more than 200 units", "only Class A", "exclude Reston", "top 5 by units"). Put the conditions in constraints / metric_constraints / sort / nearby_count. Do not set location.
- "analyze": work with the current selection unchanged: compare it, show its expenses/financials/budget, chart it. Set data_view for financial questions; intent "compare" when asked to compare.
- "reset": return to the original results ("start over", "show everything again").
- "new": the question is independent (new place, new property, new topic), or only changes the distance ("what about 15 miles?" -> context_ref "reference", max_distance_miles).
Rules:
- Names go in edit_names / with_names for remove, add, replace, keep_only; copy them as typed (use the exact spelling from "Names in the question" when listed). Never copy names from the state into property_names.
- context_ref: "results" when the question refers to the properties shown, "reference" when it refers to the original/anchor property or search point, otherwise "none". new_topic: false when the question builds on the state.
- include_nearby: true when nearby properties should be added to named or referenced ones.
- When unsure whether it edits the selection or starts a new search, prefer the operation that keeps the existing selection.
"""

# (when, question, json). Placeholders in <> stand for values the model must copy from the real question.
EXAMPLES = [
    ("fresh", "Find properties within 10 miles of <Name A>",
     '{"intent":"nearest","property_names":["<Name A>"],"max_distance_miles":10}'),
    ("fresh", "Find all properties within 30 miles of <street address, city, state ZIP>",
     '{"intent":"list","max_distance_miles":30,"location":{"text":"<street address, city, state ZIP>","type":"address"}}'),
    ("fresh", "Find properties in Austin, TX",
     '{"intent":"list","location":{"text":"Austin, TX","type":"city"}}'),
    ("fresh", "Compare <Name A> with 5 similar properties in the same market",
     '{"intent":"compare","property_names":["<Name A>"],"nearby_count":5,"constraints":[{"field":"Market","op":"same_as_subject","value":null}]}'),
    ("fresh", "Compare <Name A> and <Name B> budget versus actual",
     '{"intent":"compare","property_names":["<Name A>","<Name B>"],"data_view":"both"}'),
    ("fresh", "Compare properties within 20 miles of <Name A> whose unit count is at least 20% above it as a bar chart",
     '{"intent":"compare","property_names":["<Name A>"],"max_distance_miles":20,"metrics":["units"],"metric_constraints":[{"metric":"units","op":"gte","value":20,"basis":"percent_above_subject"}],"visualization":"bar","chart_metric":"units"}'),
    ("fresh", "Show the 10 largest properties by unit count in <a state value> as a pie chart",
     '{"intent":"list","nearby_count":10,"constraints":[{"field":"State","op":"eq","value":"<a state value>"}],"sort":{"metric":"units","direction":"desc"},"visualization":"pie","chart_metric":"units"}'),
    ("state", "Remove <Name A> from the comparison",
     '{"intent":"compare","operation":"remove","edit_names":["<Name A>"],"context_ref":"results","new_topic":false}'),
    ("state", "Also add <Name B>",
     '{"intent":"compare","operation":"add","edit_names":["<Name B>"],"context_ref":"results","new_topic":false}'),
    ("state", "Replace <Name A> with <Name B>",
     '{"intent":"compare","operation":"replace","edit_names":["<Name A>"],"with_names":["<Name B>"],"context_ref":"results","new_topic":false}'),
    ("state", "Just keep <Name A> and <Name B>",
     '{"intent":"compare","operation":"keep_only","edit_names":["<Name A>","<Name B>"],"context_ref":"results","new_topic":false}'),
    ("state", "Show properties with more than 200 units",
     '{"intent":"list","operation":"refine","metrics":["units"],"metric_constraints":[{"metric":"units","op":"gt","value":200,"basis":"absolute"}],"context_ref":"results","new_topic":false}'),
    ("state", "Exclude the ones in <City>",
     '{"intent":"list","operation":"refine","constraints":[{"field":"City","op":"ne","value":"<City>"}],"context_ref":"results","new_topic":false}'),
    ("state", "Compare their operating expenses",
     '{"intent":"compare","operation":"analyze","data_view":"actual","context_ref":"results","new_topic":false}'),
    ("state", "Show it as a bar chart by units",
     '{"intent":"list","operation":"analyze","visualization":"bar","chart_metric":"units","context_ref":"results","new_topic":false}'),
    ("state", "What about 15 miles?",
     '{"intent":"nearest","operation":"new","max_distance_miles":15,"context_ref":"reference","new_topic":false}'),
    ("state", "Start over with the original results",
     '{"intent":"list","operation":"reset","context_ref":"results","new_topic":false}'),
    ("state", "Compare these with nearby properties within 10 miles",
     '{"intent":"compare","operation":"new","max_distance_miles":10,"context_ref":"results","new_topic":false,"include_nearby":true}'),
]

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


def pick_examples(query: str, has_state: bool, limit: int = 5) -> list:
    """The worked examples most similar to this question (word overlap); ties keep the bank order."""
    q = _words(query)

    def rank(group: str, count: int) -> list:
        pool = [(i, e) for i, e in enumerate(EXAMPLES) if e[0] == group]
        pool.sort(key=lambda item: (-len(q & _words(item[1][1])), item[0]))
        return [e for _, e in pool[:count]]

    if has_state:
        chosen = rank("state", limit - 1) + rank("fresh", 1)
    else:
        chosen = rank("fresh", limit - 1)
    return chosen


def build_dynamic_prompt(query: str, catalog: str = "", state_block: str = "") -> str:
    has_state = bool(state_block and state_block.strip())
    parts = [CORE.replace("__FOLLOW_SCHEMA__", FOLLOW_SCHEMA if has_state else ""), SEARCH.rstrip()]
    if has_state:
        parts.append(FOLLOW_UP.rstrip())
    parts.append("Available constraint fields:\n" + (catalog.strip() if catalog and catalog.strip() else "(none available)"))
    if has_state:
        parts.append("Conversation state:\n" + state_block.strip())
    shots = pick_examples(query, has_state)
    parts.append("Examples (names and values are placeholders):\n" + "\n".join(f"Question: {q}\n{j}" for _, q, j in shots))
    parts.append("Respond with the JSON object only.")
    return "\n\n".join(parts)

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from config import settings

logger = logging.getLogger(__name__)

GENERIC_CHART_WORDS = re.compile(r"\b(chart|graph|plot|visuali[sz]e|visuali[sz]ation|diagram)\b", re.IGNORECASE)


@dataclass
class Visualization:
    name: str
    label: str
    keywords: list
    builder: Callable
    description: str = ""
    patterns: list = field(default_factory=list)

    def matches(self, query: str) -> bool:
        return any(p.search(query) for p in self.patterns)


_REGISTRY: dict = {}


def register(name: str, label: str, keywords: list, builder: Callable, description: str = "") -> None:
    patterns = [re.compile(r"\b" + re.escape(k).replace(r"\ ", r"[\s-]+") + r"s?\b", re.IGNORECASE) for k in keywords]
    _REGISTRY[name] = Visualization(name, label, keywords, builder, description, patterns)


def available() -> list:
    return list(_REGISTRY.keys())


def catalog_prompt() -> str:
    lines = ["Output formats (set \"visualization\" ONLY when the user explicitly asks for a chart or graph; otherwise null):"]
    for item in _REGISTRY.values():
        lines.append(f"- {item.name}: {item.description or item.label}")
    return "\n".join(lines)


def resolve(query: str, model_choice: Optional[str] = None, override: Optional[str] = None) -> Optional[str]:
    if override:
        key = str(override).strip().lower()
        if key in ("table", "none", ""):
            return None
        if key in _REGISTRY:
            return key
    text = query or ""
    for item in _REGISTRY.values():
        if item.matches(text):
            return item.name
    if model_choice and GENERIC_CHART_WORDS.search(text):
        key = str(model_choice).strip().lower()
        if key in _REGISTRY:
            return key
    return None


def _numeric(value) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def extract_series(table: dict, metric_key: str, label_key: str = "PropertyName") -> list:
    points = []
    for row in table.get("rows", []):
        values = row.get("values", {})
        number = _numeric(values.get(metric_key))
        label = values.get(label_key)
        if number is None or label is None:
            continue
        points.append({"key": row.get("key"), "label": str(label), "value": number})
    return points


def _column(table: dict, key: str) -> Optional[dict]:
    for column in table.get("columns", []):
        if column.get("key") == key:
            return column
    return None


def _round(value: float, decimals: int) -> float:
    return round(value, decimals)


def build_bar(table: dict, metric_key: str, title: Optional[str] = None) -> Optional[dict]:
    column = _column(table, metric_key)
    if column is None:
        return None
    points = extract_series(table, metric_key)
    if not points:
        return None
    decimals = column.get("decimals", 2)
    points = sorted(points, key=lambda p: p["value"], reverse=True)[: settings.CHART_MAX_BARS]
    return {
        "type": "bar",
        "title": title or f"{column['label']} by property",
        "metric": metric_key,
        "metric_label": column["label"],
        "value_type": column.get("type"),
        "labels": [p["label"] for p in points],
        "keys": [p["key"] for p in points],
        "datasets": [{"label": column["label"], "data": [_round(p["value"], decimals) for p in points]}],
    }


def build_pie(table: dict, metric_key: str, title: Optional[str] = None) -> Optional[dict]:
    column = _column(table, metric_key)
    if column is None:
        return None
    if column.get("type") == "percent":
        return None
    points = [p for p in extract_series(table, metric_key) if p["value"] > 0]
    if not points:
        return None
    decimals = column.get("decimals", 2)
    points = sorted(points, key=lambda p: p["value"], reverse=True)
    limit = max(2, settings.CHART_MAX_SLICES)
    if len(points) > limit:
        head = points[: limit - 1]
        tail_total = sum(p["value"] for p in points[limit - 1:])
        head.append({"key": None, "label": "Other", "value": tail_total})
        points = head
    total = sum(p["value"] for p in points)
    return {
        "type": "pie",
        "title": title or f"{column['label']} share by property",
        "metric": metric_key,
        "metric_label": column["label"],
        "value_type": column.get("type"),
        "labels": [p["label"] for p in points],
        "keys": [p["key"] for p in points],
        "datasets": [{"label": column["label"], "data": [_round(p["value"], decimals) for p in points]}],
        "percentages": [_round(p["value"] / total * 100.0, 2) for p in points],
    }


register("bar", "Bar chart", ["bar chart", "bar graph", "bar plot", "bars", "column chart", "histogram"], build_bar,
         "a bar chart comparing one metric across the returned properties")
register("pie", "Pie chart", ["pie chart", "pie graph", "pie", "donut chart", "doughnut chart"], build_pie,
         "a pie chart showing each returned property's share of one metric")


def build(chart: Optional[str], table: dict, metric_key: str, title: Optional[str] = None) -> tuple:
    if not chart:
        return None, None
    item = _REGISTRY.get(chart)
    if item is None:
        return None, f"The '{chart}' chart type is not supported. Showing a table instead."
    spec = item.builder(table, metric_key, title)
    if spec is None:
        return None, f"A {item.label.lower()} could not be built because the results have no usable values for the chosen metric. Showing a table instead."
    return spec, None

import logging
import re
from typing import Optional

import pandas as pd

import unit_service

logger = logging.getLogger(__name__)

METRICS = {
    "units": {
        "label": "Units",
        "column": "Units",
        "kind": "number",
        "decimals": 1,
        "description": "average unit count for the period (unit-days divided by calendar days)",
        "aliases": ["units", "unit", "unitcount", "unitcounts", "numberofunits", "totalunits", "apartments", "size"],
    },
    "occupied_units": {
        "label": "Occupied Units",
        "column": "OccupiedUnits",
        "kind": "number",
        "decimals": 1,
        "description": "average occupied units for the period",
        "aliases": ["occupiedunits", "occupiedunitcount", "occupied", "leasedunits"],
    },
    "occupancy_pct": {
        "label": "Occupancy %",
        "column": "OccupancyPct",
        "kind": "percent",
        "decimals": 1,
        "description": "occupied unit-days as a percentage of unit-days",
        "aliases": ["occupancy", "occupancypct", "occupancypercent", "occupancyrate", "percentoccupied"],
    },
    "avg_unit_sqft": {
        "label": "Avg Unit Sq Ft",
        "column": "AvgUnitSqft",
        "kind": "number",
        "decimals": 0,
        "description": "unit-day weighted average unit area",
        "aliases": ["avgunitsqft", "averageunitsize", "avgunitsize", "unitsize", "unitsquare", "avgunitarea"],
    },
    "total_sqft": {
        "label": "Total Sq Ft",
        "column": "TotalSqft",
        "kind": "number",
        "decimals": 0,
        "description": "average total unit area for the period",
        "aliases": ["totalsqft", "totalsquarefeet", "totalarea", "squarefootage", "sqft"],
    },
    "unit_days": {
        "label": "Unit Days",
        "column": "UnitDays",
        "kind": "number",
        "decimals": 0,
        "description": "total unit-days in the period",
        "aliases": ["unitdays"],
    },
}

BASES = {
    "absolute": "absolute",
    "value": "absolute",
    "percent_of_subject": "percent_of_subject",
    "pct_of_subject": "percent_of_subject",
    "percent_above_subject": "percent_above_subject",
    "pct_above_subject": "percent_above_subject",
    "percent_below_subject": "percent_below_subject",
    "pct_below_subject": "percent_below_subject",
    "percentile": "percentile",
}

OP_ALIASES = {
    "=": "eq", "==": "eq", "eq": "eq", "equals": "eq",
    ">=": "gte", "gte": "gte", "min": "gte", "at_least": "gte",
    "<=": "lte", "lte": "lte", "max": "lte", "at_most": "lte",
    ">": "gt", "gt": "gt", "above": "gt", "over": "gt",
    "<": "lt", "lt": "lt", "below": "lt", "under": "lt",
    "between": "between", "range": "between",
}

RELATIVE_BASES = {"percent_of_subject", "percent_above_subject", "percent_below_subject"}

MAX_METRIC_CONSTRAINTS = 5


def _canon(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


_ALIAS_LOOKUP = {}
for _key, _meta in METRICS.items():
    _ALIAS_LOOKUP[_canon(_key)] = _key
    for _alias in _meta["aliases"]:
        _ALIAS_LOOKUP[_canon(_alias)] = _key


def resolve_metric(name) -> Optional[str]:
    if name is None:
        return None
    return _ALIAS_LOOKUP.get(_canon(name))


def metric_definition(key: str) -> dict:
    return METRICS[key]


def metric_columns(keys) -> list:
    return [METRICS[k]["column"] for k in keys if k in METRICS]


def catalog_prompt() -> str:
    lines = ["Available metrics (computed from unit data; use the metric key exactly):"]
    for key, meta in METRICS.items():
        lines.append(f"- {key}: {meta['description']}")
    return "\n".join(lines)


def _coerce_number(value) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        cleaned = value.replace(",", "").replace("%", "").replace("$", "").strip()
        try:
            number = float(cleaned)
        except ValueError:
            return None
    else:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def sanitize_metrics(raw) -> tuple:
    if not isinstance(raw, list):
        return [], []
    chosen = []
    unsupported = []
    for item in raw:
        key = resolve_metric(item)
        if key is None:
            label = str(item).strip()
            if label and len(unsupported) < 6:
                unsupported.append(f"Metric '{label}' is not available")
            continue
        if key not in chosen:
            chosen.append(key)
    return chosen, unsupported


def sanitize_metric_constraints(raw) -> tuple:
    clean = []
    unsupported = []
    if not isinstance(raw, list):
        return clean, unsupported
    for item in raw[:MAX_METRIC_CONSTRAINTS]:
        if not isinstance(item, dict):
            continue
        key = resolve_metric(item.get("metric"))
        if key is None:
            label = str(item.get("metric", "")).strip()
            if label and len(unsupported) < 6:
                unsupported.append(f"Metric '{label}' is not available")
            continue
        op = OP_ALIASES.get(str(item.get("op", "gte")).strip().lower())
        basis = BASES.get(str(item.get("basis") or "absolute").strip().lower().replace("-", "_").replace(" ", "_"))
        if op is None or basis is None:
            unsupported.append(f"{METRICS[key]['label']} ({item.get('op')}, {item.get('basis')})")
            continue
        value = item.get("value")
        if op == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            low, high = _coerce_number(value[0]), _coerce_number(value[1])
            if low is None or high is None:
                continue
            value = [min(low, high), max(low, high)]
        else:
            value = _coerce_number(value)
            if value is None:
                continue
        if basis == "percentile":
            bounds = value if isinstance(value, list) else [value]
            if any(b < 0 or b > 100 for b in bounds):
                unsupported.append(f"{METRICS[key]['label']} percentile must be between 0 and 100")
                continue
        if basis in ("percent_above_subject", "percent_below_subject", "percent_of_subject"):
            bounds = value if isinstance(value, list) else [value]
            if any(b < 0 for b in bounds):
                continue
        clean.append({"metric": key, "op": op, "basis": basis, "value": value})
    return clean, unsupported


def sanitize_sort(raw) -> tuple:
    if not isinstance(raw, dict):
        return None, []
    key = resolve_metric(raw.get("metric"))
    if key is None:
        label = str(raw.get("metric", "")).strip()
        return None, ([f"Sort metric '{label}' is not available"] if label else [])
    direction = str(raw.get("direction") or "desc").strip().lower()
    direction = "asc" if direction in ("asc", "ascending", "lowest", "smallest") else "desc"
    return {"metric": key, "direction": direction}, []


def display_metrics(requested, constraints, sort, chart_metric=None) -> list:
    keys = []
    for key in list(requested or []):
        if key not in keys:
            keys.append(key)
    for constraint in constraints or []:
        if constraint["metric"] not in keys:
            keys.append(constraint["metric"])
    if sort and sort["metric"] not in keys:
        keys.append(sort["metric"])
    if chart_metric and chart_metric not in keys:
        keys.append(chart_metric)
    return keys


def ensure_metrics(df: pd.DataFrame, keys, ym_from=None, ym_to=None) -> tuple:
    columns = [c for c in metric_columns(keys) if c not in df.columns]
    frame, meta = unit_service.get_period_metrics(ym_from, ym_to)
    if not columns:
        return df, frame, meta
    available = [c for c in columns if c in frame.columns]
    if not available or frame.empty:
        for column in columns:
            df = df.assign(**{column: float("nan")})
        return df, frame, meta
    lookup = frame[["PropertyBizKey"] + available].drop_duplicates("PropertyBizKey")
    merged = df.merge(lookup, on="PropertyBizKey", how="left")
    merged.index = df.index
    return merged, frame, meta


def attach_metrics(records: list, keys, ym_from=None, ym_to=None) -> list:
    if not keys or not records:
        return records
    wanted = [c for c in metric_columns(keys)]
    frame, _ = unit_service.get_period_metrics(ym_from, ym_to)
    lookup = {}
    if not frame.empty:
        lookup = {str(r["PropertyBizKey"]): r for r in frame.to_dict(orient="records")}
    result = []
    for record in records:
        enriched = dict(record)
        source = lookup.get(str(record.get("PropertyBizKey")))
        for column in wanted:
            if column in enriched and enriched[column] is not None:
                continue
            value = source.get(column) if source else None
            enriched[column] = None if value is None or pd.isna(value) else float(value)
        result.append(enriched)
    return result


def _subject_value(frame: pd.DataFrame, subject: Optional[dict], column: str) -> Optional[float]:
    if subject is None or frame.empty or column not in frame.columns:
        return None
    match = frame[frame["PropertyBizKey"] == str(subject.get("PropertyBizKey")).strip()]
    if match.empty:
        return None
    value = match.iloc[0][column]
    return None if pd.isna(value) else float(value)


def _threshold(basis: str, value: float, subject_value: Optional[float], series: pd.Series) -> Optional[float]:
    if basis == "absolute":
        return value
    if basis == "percentile":
        clean = series.dropna()
        if clean.empty:
            return None
        return float(clean.quantile(value / 100.0))
    if subject_value is None:
        return None
    if basis == "percent_of_subject":
        return subject_value * value / 100.0
    if basis == "percent_above_subject":
        return subject_value * (1.0 + value / 100.0)
    if basis == "percent_below_subject":
        return subject_value * (1.0 - value / 100.0)
    return None


def _describe(constraint: dict) -> str:
    meta = METRICS[constraint["metric"]]
    value = constraint["value"]
    shown = value if not isinstance(value, list) else f"{value[0]:g} to {value[1]:g}"
    if not isinstance(value, list):
        shown = f"{value:g}"
    return f"{meta['label']} {constraint['op']} {shown} ({constraint['basis']})"


def apply_metric_constraints(df: pd.DataFrame, constraints: list, ym_from=None, ym_to=None, subject: Optional[dict] = None) -> tuple:
    applied = []
    unsupported = []
    if not constraints:
        return df, applied, unsupported

    keys = [c["metric"] for c in constraints]
    df, frame, _ = ensure_metrics(df, keys, ym_from, ym_to)

    for constraint in constraints:
        column = METRICS[constraint["metric"]]["column"]
        if column not in df.columns or df[column].dropna().empty:
            unsupported.append(f"{METRICS[constraint['metric']]['label']}: no unit data available for the selected period")
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        basis = constraint["basis"]
        op = constraint["op"]
        value = constraint["value"]
        subject_value = _subject_value(frame, subject, column) if basis in RELATIVE_BASES else None

        if basis in RELATIVE_BASES and subject_value is None:
            unsupported.append(
                f"{METRICS[constraint['metric']]['label']}: a reference property with unit data is required for '{basis}'"
            )
            continue

        if op == "between":
            low = _threshold(basis, value[0], subject_value, series)
            high = _threshold(basis, value[1], subject_value, series)
            if low is None or high is None:
                unsupported.append(_describe(constraint))
                continue
            low, high = min(low, high), max(low, high)
            mask = (series >= low) & (series <= high)
        else:
            limit = _threshold(basis, value, subject_value, series)
            if limit is None:
                unsupported.append(_describe(constraint))
                continue
            if op == "gte":
                mask = series >= limit
            elif op == "gt":
                mask = series > limit
            elif op == "lte":
                mask = series <= limit
            elif op == "lt":
                mask = series < limit
            else:
                mask = (series - limit).abs() < 1e-9
        df = df[mask.fillna(False)]
        applied.append(constraint)

    return df, applied, unsupported


def apply_sort(df: pd.DataFrame, sort: Optional[dict], ym_from=None, ym_to=None, fallback: Optional[str] = None) -> pd.DataFrame:
    if sort:
        df, _, _ = ensure_metrics(df, [sort["metric"]], ym_from, ym_to)
        column = METRICS[sort["metric"]]["column"]
        return df.sort_values(column, ascending=sort["direction"] == "asc", na_position="last", kind="mergesort")
    if fallback and fallback in df.columns:
        return df.sort_values(fallback, kind="mergesort")
    return df


def table_columns(keys) -> list:
    return [
        {"key": METRICS[k]["column"], "label": METRICS[k]["label"], "type": METRICS[k]["kind"], "decimals": METRICS[k]["decimals"]}
        for k in keys if k in METRICS
    ]

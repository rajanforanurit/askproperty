import difflib
import logging
import re
import uuid
from typing import Optional

import pandas as pd

import property_service
from config import settings

logger = logging.getLogger(__name__)

FILTER_FIELDS = {
    "State": "category",
    "City": "category",
    "Market": "category",
    "MSA": "category",
    "AssetType": "category",
    "PropertySizeCategory": "category",
    "PropertyAgeBucket": "category",
    "PropertyStatus": "category",
    "ManagementCompany": "category",
    "FundCurrent": "category",
    "YearBuilt": "number",
}

OP_ALIASES = {
    "=": "eq", "==": "eq", "eq": "eq", "equals": "eq", "is": "eq",
    "!=": "ne", "ne": "ne", "neq": "ne", "not": "ne",
    "in": "in", "not_in": "not_in", "nin": "not_in",
    ">=": "gte", "gte": "gte", "min": "gte",
    "<=": "lte", "lte": "lte", "max": "lte",
    ">": "gt", "gt": "gt",
    "<": "lt", "lt": "lt",
    "between": "between", "range": "between",
    "same_as_subject": "same_as_subject", "same": "same_as_subject", "similar": "same_as_subject",
}

VALID_OPS = {
    "category": {"eq", "ne", "in", "not_in", "same_as_subject"},
    "number": {"eq", "gte", "lte", "gt", "lt", "between"},
}

MAX_CONSTRAINTS = 8
MAX_LIST_VALUES = 12
MAX_FLAGGED = 6
PROMPT_VALUE_LIMIT = 40

TABLE_COLUMNS = [
    ("PropertyName", "Property", "text"),
    ("PropertyBizKey", "Property ID", "text"),
    ("City", "City", "text"),
    ("State", "State", "text"),
    ("Market", "Market", "text"),
    ("AssetType", "Asset Type", "text"),
    ("PropertySizeCategory", "Size Category", "text"),
    ("PropertyAgeBucket", "Age Bucket", "text"),
    ("YearBuilt", "Year Built", "year"),
    ("PropertyStatus", "Status", "text"),
    ("ManagementCompany", "Management Company", "text"),
    ("distance_miles", "Distance (mi)", "distance"),
]


def _canon(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def new_task_id() -> str:
    return uuid.uuid4().hex


def clean_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def clean_record(record: dict) -> dict:
    return {key: clean_value(value) for key, value in record.items()}


def field_catalog() -> list:
    df = property_service.get_all_properties()
    catalog = []
    for field, kind in FILTER_FIELDS.items():
        if field not in df.columns:
            continue
        series = df[field].dropna()
        if kind == "category":
            values = sorted({str(v).strip() for v in series if str(v).strip()})
            catalog.append({
                "field": field,
                "type": kind,
                "distinct": len(values),
                "values": values if len(values) <= PROMPT_VALUE_LIMIT else None,
            })
        else:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                continue
            catalog.append({
                "field": field,
                "type": kind,
                "min": clean_value(numeric.min()),
                "max": clean_value(numeric.max()),
            })
    return catalog


def catalog_prompt() -> str:
    try:
        catalog = field_catalog()
    except Exception as e:
        logger.warning("Could not build field catalog: %s", e)
        return ""
    lines = []
    for item in catalog:
        if item["type"] == "category":
            if item["values"] is not None:
                lines.append(f"- {item['field']} (category): {', '.join(item['values'])}")
            else:
                lines.append(f"- {item['field']} (category, {item['distinct']} distinct values, use the user's wording)")
        else:
            lines.append(f"- {item['field']} (number): {item['min']} to {item['max']}")
    return "\n".join(lines)


def _coerce_number(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def sanitize_constraints(raw, flagged=None):
    lookup = {_canon(field): field for field in FILTER_FIELDS}
    clean = []
    unsupported = []

    for item in flagged or []:
        text = str(item).strip()
        if text and text not in unsupported:
            unsupported.append(text)
    unsupported = unsupported[:MAX_FLAGGED]

    if not isinstance(raw, list):
        return clean, unsupported

    for item in raw[:MAX_CONSTRAINTS]:
        if not isinstance(item, dict):
            continue
        field_name = str(item.get("field", "")).strip()
        field = lookup.get(_canon(field_name))
        if field is None:
            if field_name and field_name not in unsupported and len(unsupported) < MAX_FLAGGED:
                unsupported.append(field_name)
            continue

        kind = FILTER_FIELDS[field]
        op = OP_ALIASES.get(str(item.get("op", "eq")).strip().lower())
        if op is None or op not in VALID_OPS[kind]:
            label = f"{field} ({item.get('op')})"
            if label not in unsupported and len(unsupported) < MAX_FLAGGED:
                unsupported.append(label)
            continue

        value = item.get("value")
        if op == "same_as_subject":
            clean.append({"field": field, "op": op, "value": None})
            continue

        if kind == "category":
            if op in ("in", "not_in"):
                values = value if isinstance(value, list) else [value]
                values = [str(v).strip() for v in values if v is not None and str(v).strip()][:MAX_LIST_VALUES]
                if not values:
                    continue
                clean.append({"field": field, "op": op, "value": values})
            else:
                if value is None or not str(value).strip():
                    continue
                clean.append({"field": field, "op": op, "value": str(value).strip()})
            continue

        if op == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            low, high = _coerce_number(value[0]), _coerce_number(value[1])
            if low is None or high is None:
                continue
            clean.append({"field": field, "op": op, "value": [min(low, high), max(low, high)]})
            continue

        number = _coerce_number(value)
        if number is None:
            continue
        clean.append({"field": field, "op": op, "value": number})

    return clean, unsupported


def _match_value(requested: str, universe: list) -> list:
    folded = requested.strip().casefold()
    exact = [u for u in universe if u.casefold() == folded]
    if exact:
        return exact
    contains = [u for u in universe if folded and (folded in u.casefold() or u.casefold() in folded)]
    if contains and len(contains) <= 5:
        return contains
    close = difflib.get_close_matches(requested.strip(), universe, n=1, cutoff=0.8)
    return close


def apply_constraints(df: pd.DataFrame, constraints: list, subject: Optional[dict] = None):
    applied = []
    unsupported = []

    for constraint in constraints or []:
        field = constraint["field"]
        op = constraint["op"]
        value = constraint.get("value")

        if field not in df.columns:
            unsupported.append(field)
            continue

        if FILTER_FIELDS.get(field) == "category":
            column = df[field].astype(str).str.strip()
            folded = column.str.casefold()
            if op == "same_as_subject":
                subject_value = None if subject is None else clean_value(subject.get(field))
                if subject_value is None or not str(subject_value).strip():
                    unsupported.append(f"{field} (the selected property has no value)")
                    continue
                resolved = [str(subject_value).strip()]
                mask = folded.isin([r.casefold() for r in resolved])
            else:
                requested = [value] if op in ("eq", "ne") else list(value)
                universe = sorted({v for v in column if v and v.lower() not in ("nan", "none")})
                resolved = []
                for item in requested:
                    matches = _match_value(str(item), universe)
                    if matches:
                        resolved.extend(m for m in matches if m not in resolved)
                    else:
                        unsupported.append(f"{field}: no property data matches '{item}'")
                if not resolved:
                    if op in ("eq", "in"):
                        df = df.iloc[0:0]
                    continue
                mask = folded.isin([r.casefold() for r in resolved])
                if op in ("ne", "not_in"):
                    mask = ~mask
            df = df[mask]
            applied.append({"field": field, "op": op, "value": resolved if len(resolved) > 1 else resolved[0]})
            continue

        series = pd.to_numeric(df[field], errors="coerce")
        if op == "between":
            mask = (series >= value[0]) & (series <= value[1])
        elif op == "gte":
            mask = series >= value
        elif op == "lte":
            mask = series <= value
        elif op == "gt":
            mask = series > value
        elif op == "lt":
            mask = series < value
        else:
            mask = series == value
        df = df[mask]
        applied.append({"field": field, "op": op, "value": value})

    return df, applied, unsupported


def filter_properties(constraints: list, active_only: Optional[bool] = None, limit: Optional[int] = None):
    if active_only is None:
        active_only = not any(c["field"] == "PropertyStatus" for c in constraints)
    cap = settings.RADIUS_RESULT_CAP
    limit = min(limit or cap, cap)
    df = property_service.get_all_properties(active_only=active_only)
    df, applied, unsupported = apply_constraints(df, constraints, None)
    total = len(df)
    rows = [clean_record(r) for r in df.sort_values("PropertyName").head(limit).to_dict(orient="records")]
    return {"results": rows, "applied_constraints": applied, "unsupported_constraints": unsupported, "total_matches": total, "limit": limit}


ALWAYS_FIELDS = ["PropertyBizKey", "PropertyName"]
MAP_FIELDS = ["Latitude", "Longitude"]
MAX_FIELDS = 12


def sanitize_fields(raw):
    """Validate the attribute names the model says the user wants shown/compared.

    Returns (fields, unsupported). `fields` is [] when the user named no valid
    attributes, so callers fall back to the default comparison fields.
    """
    if not isinstance(raw, list) or not raw:
        return [], []

    allowed = {
        _canon(key): key
        for key, _label, _kind in TABLE_COLUMNS
        if key not in ALWAYS_FIELDS and key != "distance_miles"
    }
    allowed.update({_canon(key): key for key in FILTER_FIELDS})

    chosen = []
    unsupported = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            continue
        key = allowed.get(_canon(item))
        if key is None:
            close = difflib.get_close_matches(_canon(item), list(allowed), n=1, cutoff=0.85)
            key = allowed[close[0]] if close else None
        if key is None:
            if len(unsupported) < MAX_FLAGGED:
                unsupported.append(f"Attribute '{item.strip()}' is not available in the property data")
            continue
        if key not in chosen:
            chosen.append(key)

    if not chosen:
        return [], unsupported

    chosen = chosen[:MAX_FIELDS]
    fields = list(ALWAYS_FIELDS) + chosen + [f for f in MAP_FIELDS if f not in chosen]
    return fields, unsupported


def _valid_coords(record: dict) -> bool:
    lat, lng = record.get("Latitude"), record.get("Longitude")
    return isinstance(lat, (int, float)) and isinstance(lng, (int, float)) and lat == lat and lng == lng


def build_markers(subject: Optional[dict], others: list, role: str) -> list:
    markers = []
    entries = ([("subject", subject)] if subject else []) + [(role, o) for o in others]
    for rank, (marker_role, record) in enumerate(entries):
        clean = clean_record(record)
        if not _valid_coords(clean):
            continue
        markers.append({
            "key": str(clean.get("PropertyBizKey")),
            "name": clean.get("PropertyName"),
            "lat": clean["Latitude"],
            "lng": clean["Longitude"],
            "role": marker_role,
            "rank": rank,
            "distance_miles": clean.get("distance_miles"),
        })
    return markers


def build_table(subject: Optional[dict], others: list, fields: Optional[list] = None) -> dict:
    records = ([subject] if subject else []) + list(others)
    has_distance = any(clean_record(r).get("distance_miles") is not None for r in records)
    wanted = None
    if fields:
        wanted = set(fields) | {"PropertyName"}
    columns = [
        {"key": key, "label": label, "type": kind}
        for key, label, kind in TABLE_COLUMNS
        if (key != "distance_miles" or has_distance) and (wanted is None or key in wanted or key == "distance_miles")
    ]
    rows = []
    for index, record in enumerate(records):
        clean = clean_record(record)
        rows.append({
            "key": str(clean.get("PropertyBizKey")),
            "is_subject": bool(subject) and index == 0,
            "values": {c["key"]: clean.get(c["key"]) for c in columns},
        })
    return {"columns": columns, "rows": rows}


def build_task(
    task_id: str,
    kind: str,
    subject: Optional[dict],
    others: list,
    role: str,
    view: Optional[str],
    radius: Optional[float],
    requested_count: Optional[int],
    applied: list,
    unsupported: list,
    notes: list,
    fields: Optional[list] = None,
) -> dict:
    markers = build_markers(subject, others, role)
    total = len(others) + (1 if subject else 0)
    return {
        "task": {
            "id": task_id,
            "kind": kind,
            "subject_key": str(subject.get("PropertyBizKey")) if subject else None,
            "view": view,
            "radius_miles": radius,
            "requested_count": requested_count,
            "returned_count": len(others),
            "constraints_applied": applied,
            "constraints_unsupported": unsupported,
            "plotted": len(markers),
            "unplotted": total - len(markers),
            "notes": notes,
        },
        "markers": markers,
        "table": build_table(subject, others, fields),
    }

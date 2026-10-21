import logging
import numpy as np
import pandas as pd

from property_service import get_all_properties, get_by_bizkey
from config import settings
import metric_service
import query_service

logger = logging.getLogger(__name__)

EARTH_RADIUS_MILES = 3958.8


def haversine_distance_miles(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_MILES * c


def find_nearest_properties(
    target_biz_key: str,
    count: int = None,
    active_only: bool = True,
    exclude_missing_coords: bool = True,
    return_all: bool = False,
) -> dict:
    count = count or settings.DEFAULT_NEARBY_COUNT
    count = min(count, settings.MAX_NEARBY_COUNT)

    target = get_by_bizkey(target_biz_key)
    if target is None:
        raise ValueError(f"Property with PropertyBizKey '{target_biz_key}' not found.")

    if pd.isna(target.get("Latitude")) or pd.isna(target.get("Longitude")):
        raise ValueError(
            f"Property '{target.get('PropertyName')}' ({target_biz_key}) has no coordinates on file."
        )

    df = get_all_properties(active_only=active_only)

    if exclude_missing_coords:
        df = df.dropna(subset=["Latitude", "Longitude"])

    df = df[df["PropertyBizKey"] != target["PropertyBizKey"]]

    if df.empty:
        return {"target": target, "nearby": []}

    distances = haversine_distance_miles(
        target["Latitude"], target["Longitude"],
        df["Latitude"].values, df["Longitude"].values,
    )
    df = df.copy()
    df["distance_miles"] = np.round(distances, 2)

    ordered = df.sort_values("distance_miles")
    nearest = ordered if return_all else ordered.head(count)

    return {
        "target": target,
        "nearby": nearest.to_dict(orient="records"),
    }


def find_relevant_properties(
    target_biz_key: str,
    count: int = None,
    max_distance_miles: float = None,
    constraints: list = None,
    active_only: bool = None,
    metric_constraints: list = None,
    sort: dict = None,
    display_metrics: list = None,
    ym_from: int = None,
    ym_to: int = None,
    return_all: bool = False,
) -> dict:
    target = get_by_bizkey(target_biz_key)
    if target is None:
        raise ValueError(f"Property with PropertyBizKey '{target_biz_key}' not found.")

    if pd.isna(target.get("Latitude")) or pd.isna(target.get("Longitude")):
        raise ValueError(
            f"Property '{target.get('PropertyName')}' ({target_biz_key}) has no coordinates on file."
        )

    constraints = constraints or []
    if active_only is None:
        active_only = not any(c["field"] == "PropertyStatus" for c in constraints)

    cap = settings.RADIUS_RESULT_CAP if max_distance_miles else settings.MAX_NEARBY_COUNT
    requested = count or (cap if max_distance_miles else settings.DEFAULT_NEARBY_COUNT)
    limit = max(1, min(int(requested), cap))

    df = get_all_properties(active_only=active_only).dropna(subset=["Latitude", "Longitude"])
    df = df[df["PropertyBizKey"] != target["PropertyBizKey"]]
    df, applied, unsupported = query_service.apply_constraints(df, constraints, target)

    base = {
        "target": query_service.clean_record(target),
        "nearby": [],
        "applied_constraints": applied,
        "unsupported_constraints": unsupported,
        "metric_constraints_applied": [],
        "total_matches": 0,
        "limit": limit,
    }
    if df.empty:
        return base

    distances = haversine_distance_miles(
        target["Latitude"], target["Longitude"],
        df["Latitude"].values, df["Longitude"].values,
    )
    df = df.copy()
    df["distance_miles"] = np.round(distances, 2)

    if max_distance_miles:
        df = df[df["distance_miles"] <= float(max_distance_miles)]

    df, metric_applied, metric_unsupported = metric_service.apply_metric_constraints(
        df, metric_constraints or [], ym_from, ym_to, target,
    )
    base["metric_constraints_applied"] = metric_applied
    base["unsupported_constraints"] = unsupported + metric_unsupported

    df, _, _ = metric_service.ensure_metrics(df, display_metrics or [], ym_from, ym_to)

    base["total_matches"] = len(df)
    if sort:
        ordered = metric_service.apply_sort(df, sort, ym_from, ym_to)
    else:
        ordered = df.sort_values("distance_miles")
    nearest = ordered if return_all else ordered.head(limit)
    base["nearby"] = [query_service.clean_record(r) for r in nearest.to_dict(orient="records")]
    return base


def find_near_references(
    reference_keys: list,
    max_distance_miles: float,
    constraints: list = None,
    metric_constraints: list = None,
    sort: dict = None,
    display_metrics: list = None,
    ym_from: int = None,
    ym_to: int = None,
) -> dict:
    """Every property within the radius of ANY reference property (no count limit)."""
    refs, skipped = [], []
    for key in reference_keys:
        record = get_by_bizkey(key)
        if record is None:
            continue
        if pd.isna(record.get("Latitude")) or pd.isna(record.get("Longitude")):
            skipped.append(str(record.get("PropertyName")))
        else:
            refs.append(record)
    if not refs:
        raise ValueError("None of the reference properties has coordinates on file.")

    constraints = constraints or []
    active_only = not any(c["field"] == "PropertyStatus" for c in constraints)
    df = get_all_properties(active_only=active_only)
    skipped_candidates = int(df[["Latitude", "Longitude"]].isna().any(axis=1).sum())
    df = df.dropna(subset=["Latitude", "Longitude"])
    ref_keys = {str(r["PropertyBizKey"]) for r in refs}
    df = df[~df["PropertyBizKey"].isin(ref_keys)]
    df, applied, unsupported = query_service.apply_constraints(df, constraints, refs[0])

    base = {
        "references": [query_service.clean_record(r) for r in refs],
        "nearby": [],
        "applied_constraints": applied,
        "unsupported_constraints": unsupported,
        "metric_constraints_applied": [],
        "skipped_references": skipped,
        "skipped_candidates": skipped_candidates,
    }
    if df.empty:
        return base

    best = np.full(len(df), np.inf)
    nearest_ref = np.array([""] * len(df), dtype=object)
    for ref in refs:
        d = haversine_distance_miles(ref["Latitude"], ref["Longitude"], df["Latitude"].values, df["Longitude"].values)
        closer = d < best
        best = np.where(closer, d, best)
        nearest_ref = np.where(closer, str(ref["PropertyName"]), nearest_ref)
    df = df.copy()
    df["distance_miles"] = np.round(best, 2)
    df["nearest_reference"] = nearest_ref
    df = df[df["distance_miles"] <= float(max_distance_miles)]

    df, metric_applied, metric_unsupported = metric_service.apply_metric_constraints(
        df, metric_constraints or [], ym_from, ym_to, refs[0],
    )
    base["metric_constraints_applied"] = metric_applied
    base["unsupported_constraints"] = unsupported + metric_unsupported
    df, _, _ = metric_service.ensure_metrics(df, display_metrics or [], ym_from, ym_to)
    ordered = metric_service.apply_sort(df, sort, ym_from, ym_to) if sort else df.sort_values("distance_miles")
    base["nearby"] = [query_service.clean_record(r) for r in ordered.to_dict(orient="records")]
    return base

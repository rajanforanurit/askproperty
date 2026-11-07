"""Backend operations on the conversation's property selection.

The language model only says WHAT the user wants ("remove", "Woodburn I"). Everything below is deterministic:
resolving a typed name to a property id inside the right scope, editing the selection, applying filters to
the retrieved pool and computing distances from the original reference. Pure functions over the state dict
from context_service plus the cached property table.
"""
import difflib
import re
from typing import Optional

import numpy as np
import pandas as pd

import geo_service
import metric_service
import property_service
import query_service

ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}
FILLER_LEAD = {"the"}
MARGIN = 0.1


# ------------------------------------------------------------------ name resolution

def tokens(text) -> list:
    words = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower().replace("&", " and ")).split()
    while words and words[0] in FILLER_LEAD:
        words = words[1:]
    return [ROMAN.get(w, w) for w in words]


def _contiguous(inner: list, outer: list) -> bool:
    n = len(inner)
    return n > 0 and any(outer[i:i + n] == inner for i in range(len(outer) - n + 1))


def score(mention: list, name: list) -> float:
    if not mention or not name:
        return 0.0
    if mention == name:
        return 1.0
    if _contiguous(mention, name):
        return 0.7 + 0.25 * len(mention) / len(name)
    if _contiguous(name, mention):
        return 0.65 + 0.25 * len(name) / len(mention)
    # Typo tolerance must never blur numbers: "Woodburn 1" and "Woodburn 2" are different properties.
    if [t for t in mention if any(c.isdigit() for c in t)] != [t for t in name if any(c.isdigit() for c in t)]:
        return 0.0
    ratio = difflib.SequenceMatcher(None, " ".join(mention), " ".join(name)).ratio()
    return ratio * 0.8 if ratio >= 0.84 else 0.0


def resolve_name(text: str, candidates: dict) -> dict:
    """Match one typed name against {key: name}. Exact beats partial; close calls are reported, never guessed."""
    mention = tokens(text)
    scored = sorted(((score(mention, tokens(name)), key, name) for key, name in candidates.items()), reverse=True)
    scored = [s for s in scored if s[0] > 0]
    if not scored:
        return {"text": text, "status": "missing", "key": None, "name": None, "candidates": []}
    top = scored[0][0]
    close = [s for s in scored if top - s[0] < (0.0005 if top >= 1.0 else MARGIN)]
    if len(close) == 1:
        return {"text": text, "status": "resolved", "key": close[0][1], "name": close[0][2], "candidates": []}
    return {"text": text, "status": "ambiguous", "key": None, "name": None,
            "candidates": [{"key": k, "name": n} for _, k, n in close[:6]]}


def resolve_preferring(text: str, preferred: dict, wide: dict) -> dict:
    """Resolve against everything, using the current selection only to break a tie between equally good matches."""
    hit = resolve_name(text, wide)
    if hit["status"] == "ambiguous":
        inside = [c for c in hit["candidates"] if c["key"] in preferred]
        if len(inside) == 1:
            return {"text": text, "status": "resolved", "key": inside[0]["key"], "name": inside[0]["name"], "candidates": []}
    return hit


def resolve_names(names: list, scope: dict, universe: Optional[dict] = None, protected: Optional[set] = None) -> list:
    """Resolve each name inside `scope`. A name found only in `universe` is 'outside'; one that is a
    protected (reference) property is 'reference'."""
    out = []
    for text in names:
        hit = resolve_name(text, scope)
        if hit["status"] == "missing":
            if protected:
                p = resolve_name(text, {k: universe[k] for k in protected if universe and k in universe} if universe else {})
                if p["status"] == "resolved":
                    out.append({**p, "status": "reference"})
                    continue
            if universe is not None:
                wide = resolve_name(text, universe)
                if wide["status"] == "resolved":
                    out.append({**wide, "status": "outside"})
                    continue
                if wide["status"] == "ambiguous":
                    out.append(wide)
                    continue
        out.append(hit)
    return out


def mention_hints(query: str, ctx: Optional[dict], universe: dict) -> list:
    """Lines for the prompt: properties the question seems to name, and where each one currently is."""
    if not ctx or not query:
        return []
    q = tokens(query)
    if not q:
        return []
    active, excluded = set(ctx["result_keys"]), set(ctx["excluded_keys"])
    found = []
    for key, name in universe.items():
        nt = tokens(name)
        if nt and (_contiguous(nt, q) or (len(nt) >= 2 and score(nt, q[:len(nt)]) >= 0.9)):
            where = "reference" if key in ctx["reference_keys"] else "in selection" if key in active else "removed" if key in excluded else "not in selection"
            found.append((-len(nt), name, where))
    found.sort()
    if not found:
        return []
    return ["Names in the question that match known properties: " + "; ".join(f'"{n}" ({w})' for _, n, w in found[:8])]


# ------------------------------------------------------------------ the selection itself

def origin_of(ctx: dict) -> Optional[tuple]:
    """Where distances are measured from: the reference property if there is one, else the reference point."""
    for record in property_service.get_many(ctx["reference_keys"]):
        lat, lng = record.get("Latitude"), record.get("Longitude")
        if lat is not None and lng is not None and lat == lat and lng == lng:
            return float(lat), float(lng)
    point = ctx.get("reference_point")
    return (point["lat"], point["lng"]) if point else None


def apply_edit(ctx: dict, operation: str, remove_keys: list, add_keys: list) -> dict:
    """The state patch for a membership edit. Removed properties are remembered so a later filter keeps them out."""
    refs = set(ctx["reference_keys"])
    active = [k for k in ctx["result_keys"] if k not in refs]
    excluded = list(ctx["excluded_keys"])
    pool = list(ctx["pool_keys"])
    if operation == "keep_only":
        keep = list(dict.fromkeys(add_keys))
        excluded += [k for k in active if k not in keep and k not in excluded]
        active = keep
    else:
        for k in remove_keys:
            if k in active:
                active.remove(k)
            if k not in excluded:
                excluded.append(k)
    if operation != "keep_only":
        for k in add_keys:
            if k not in active:
                active.append(k)
    for k in add_keys:
        if k in excluded:
            excluded.remove(k)
        if k not in pool:
            pool.append(k)
    return {"result_keys": list(ctx["reference_keys"]) + active, "excluded_keys": excluded, "pool_keys": pool}


def evaluate(ctx: dict, constraints: list, metric_constraints: list, sort: Optional[dict], count: Optional[int],
             ym_from=None, ym_to=None) -> dict:
    """Apply filters to the retrieved pool (minus removed properties). Returns the matches in display order."""
    refs = ctx["reference_keys"]
    candidates = [k for k in ctx["pool_keys"] if k not in set(ctx["excluded_keys"]) and k not in set(refs)]
    records = property_service.get_many(candidates)
    ref_records = property_service.get_many(refs)
    subject = ref_records[0] if ref_records else None
    df = pd.DataFrame(records) if records else property_service.get_all_properties().iloc[0:0]
    df, applied, unsupported = query_service.apply_constraints(df, constraints, subject)
    df, metric_applied, metric_unsupported = metric_service.apply_metric_constraints(df, metric_constraints, ym_from, ym_to, subject)
    origin = origin_of(ctx)
    if origin and not df.empty:
        lat = pd.to_numeric(df["Latitude"], errors="coerce")
        lng = pd.to_numeric(df["Longitude"], errors="coerce")
        miles = geo_service.haversine_distance_miles(origin[0], origin[1], lat.fillna(0).values, lng.fillna(0).values)
        df = df.assign(distance_miles=np.where((lat.notna() & lng.notna()).values, np.round(miles, 2), np.nan))
    if sort and not df.empty:
        df = metric_service.apply_sort(df, sort, ym_from, ym_to, fallback="PropertyName")
    elif origin and "distance_miles" in df.columns:
        df = df.sort_values(["distance_miles", "PropertyName"], na_position="last")
    elif not df.empty:
        df = df.sort_values("PropertyName")
    if count:
        df = df.head(int(count))
    return {
        "records": [query_service.clean_record(r) for r in df.to_dict(orient="records")],
        "applied": applied, "metric_applied": metric_applied,
        "unsupported": unsupported + metric_unsupported,
        "origin": origin,
    }


def records_for(keys: list, origin: Optional[tuple], sort: Optional[dict] = None, ym_from=None, ym_to=None) -> list:
    """Records for explicit keys with distances from the reference, ordered by the sort or by distance."""
    out = []
    for record in property_service.get_many(keys):
        item = query_service.clean_record(record)
        lat, lng = item.get("Latitude"), item.get("Longitude")
        if origin and lat is not None and lng is not None and lat == lat and lng == lng:
            item["distance_miles"] = round(float(geo_service.haversine_distance_miles(
                origin[0], origin[1], np.array([lat]), np.array([lng]))[0]), 2)
        out.append(item)
    if sort and out:
        frame = metric_service.apply_sort(pd.DataFrame(out), sort, ym_from, ym_to, fallback="PropertyName")
        return [query_service.clean_record(r) for r in frame.to_dict(orient="records")]
    out.sort(key=lambda r: (r.get("distance_miles") is None, r.get("distance_miles") or 0, str(r.get("PropertyName"))))
    return out

import logging
import secrets
import json
from datetime import datetime, timezone
from typing import Any, Optional
import numpy as np
import requests
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import pandas as pd
from pydantic import BaseModel
import property_service
import geo_service
import comparison_service
import financial_service
import query_service
import ai_service
import unit_service
import metric_service
import visualization_service
import errors
import location_service
import context_service
import layout_service
from db import check_connection
from config import settings, validate_secret_key_settings, validate_savechat_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ask Property Data Intelligence API", version="1.26.0")


@app.exception_handler(StarletteHTTPException)
async def _http_error(_request, exc):
    detail = errors.normalise_detail(exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail}, headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def _validation_error(_request, _exc):
    return JSONResponse(status_code=422, content={"detail": errors.normalise_detail(400, None)})


@app.exception_handler(Exception)
async def _unexpected_error(_request, exc):
    rid = errors.new_request_id()
    logger.error("Unhandled error [%s]", rid, exc_info=exc)
    status, code = errors.classify_exception(exc)
    return JSONResponse(status_code=status, content={"detail": {"code": code, "message": errors.DEFAULT_MESSAGES[code], "request_id": rid}})


class CompareRequest(BaseModel):
    biz_keys: list[str]
    fields: Optional[list[str]] = None
    view: Optional[str] = None
    year_month_from: Optional[int] = None
    year_month_to: Optional[int] = None
    top_accounts: Optional[int] = None


class AskAIRequest(BaseModel):
    query: str
    view: Optional[str] = None
    context_property_key: Optional[str] = None
    year_month_from: Optional[int] = None
    year_month_to: Optional[int] = None
    top_accounts: Optional[int] = None
    chart: Optional[str] = None
    context: Optional[dict] = None


class SelectionRequest(BaseModel):
    biz_keys: list[str]
    kind: str = "compare"
    subject_key: Optional[str] = None
    reference_keys: Optional[list[str]] = None
    origin: Optional[dict] = None
    view: Optional[str] = None
    fields: Optional[list[str]] = None
    metrics: Optional[list[str]] = None
    year_month_from: Optional[int] = None
    year_month_to: Optional[int] = None
    top_accounts: Optional[int] = None
    chart: Optional[str] = None
    chart_metric: Optional[str] = None


class SaveChatRequest(BaseModel):
    query: str
    response: Any
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[dict] = None


def verify_secret_key(x_secret_key: Optional[str] = Header(default=None)):
    try:
        validate_secret_key_settings()
    except RuntimeError as e:
        logger.error("%s", e)
        raise errors.fail(503, "service_not_configured")

    if not x_secret_key or not secrets.compare_digest(
        x_secret_key.encode("utf-8"), settings.SECRET_KEY.encode("utf-8")
    ):
        raise errors.fail(401, "auth_failed")


def _resolve_view(value: Optional[str]) -> Optional[str]:
    try:
        return financial_service.normalize_view(value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _selection_size() -> int:
    return max(1, settings.DEFAULT_SELECTED_COUNT)


def _endpoint_selection(target: dict, pool: list, selected: list) -> dict:
    plan = {"display_metrics": [], "ym_from": None, "ym_to": None}
    full = _complete_pool(pool, anchors=[target], exclude=[str(target["PropertyBizKey"])])
    return _selection_payload(plan, full, selected, str(target["PropertyBizKey"]))


def _financial_payload(
    biz_keys: list[str],
    view: Optional[str],
    ym_from: Optional[int],
    ym_to: Optional[int],
    top_accounts: Optional[int] = None,
) -> dict:
    if not view:
        return {}
    try:
        return comparison_service.financial_payload(biz_keys, view, ym_from, ym_to, top_accounts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.on_event("startup")
def warm_cache():
    try:
        df = property_service.get_all_properties(force_refresh=True)
        logger.info("Startup cache warm: %d properties loaded.", len(df))
    except Exception as e:
        logger.error("Cache warm-up failed at startup: %s", e)

    try:
        financial_service.warm()
        logger.info("Startup cache warm: GL account lookup loaded.")
    except Exception as e:
        logger.error("GL account warm-up failed at startup: %s", e)


@app.get("/health")
def health():
    db_ok = check_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "service":"Ask Property Data Intelligence API",
        "database_connected": db_ok,
        "cache_loaded": property_service._cache.is_loaded,
        "cache_age_seconds": round(property_service._cache.age_seconds, 1),
        "savechat_configured": bool(settings.SAVECHAT_URI),
    }


@app.post("/admin/refresh-cache", dependencies=[Depends(verify_secret_key)])
def refresh_cache():
    df = property_service.refresh_now()
    financial = financial_service.refresh_now()
    return {"refreshed": True, "row_count": len(df), "financial": financial}


@app.get("/admin/financial-diagnostics", dependencies=[Depends(verify_secret_key)])
def financial_diagnostics():
    return financial_service.diagnostics()


@app.get("/admin/unit-diagnostics", dependencies=[Depends(verify_secret_key)])
def unit_diagnostics():
    return unit_service.diagnostics()


@app.get("/properties/search", dependencies=[Depends(verify_secret_key)])
def search_properties(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
    active_only: bool = Query(False),
):
    results = property_service.search_by_name(q, limit=limit, active_only=active_only)
    return {"query": q, "count": len(results), "results": results}


class _Query:
    def __init__(self, query: str):
        self.query = query


@app.get("/properties/in-area", dependencies=[Depends(verify_secret_key)])
def properties_in_area(
    name: str = Query(..., min_length=1, max_length=200),
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    radius_miles: Optional[float] = Query(None, gt=0, le=500),
    north: Optional[float] = None,
    south: Optional[float] = None,
    east: Optional[float] = None,
    west: Optional[float] = None,
):
    """Properties inside an area the client already geocoded, with a nearby fallback."""
    use_bounds = None not in (north, south, east, west) and south <= north and west <= east
    active = property_service.get_all_properties(active_only=True)
    everything = property_service.get_all_properties()
    info = {
        "requested": name, "normalized": name, "type": "region", "status": "exact", "message": None,
        "origin": {"lat": lat, "lng": lng, "source": "client", "label": name},
        "radius_miles": radius_miles, "scope": None, "matched_by": "bounds" if use_bounds else "radius",
        "geocoder": "ok", "unmapped_count": 0, "included_inactive": False,
    }
    warnings = []

    def inside(frame):
        valid = location_service._valid_coords(frame)
        if valid.empty:
            return valid.assign(distance_miles=pd.Series(dtype=float))
        miles = geo_service.haversine_distance_miles(lat, lng, valid["Latitude"].values, valid["Longitude"].values)
        valid = valid.copy()
        valid["distance_miles"] = np.round(miles, 2)
        if use_bounds:
            valid = valid[valid["Latitude"].between(south, north) & valid["Longitude"].between(west, east)]
        else:
            valid = valid[valid["distance_miles"] <= float(radius_miles or 3)]
        return valid.sort_values("distance_miles")

    found = inside(active)
    if found.empty:
        found = inside(everything)
        if not found.empty:
            info["included_inactive"] = True
            warnings.append({"code": "inactive_only", "message": "Only inactive or disposed properties are in this area, so they are shown."})
    unmapped = int(len(everything) - len(location_service._valid_coords(everything)))
    if unmapped:
        warnings.append({"code": "missing_coordinates",
                         "message": f"{unmapped} properties have no usable coordinates and could not be checked against this area."})

    total = len(found)
    if found.empty:
        for step in settings.LOCATION_FALLBACK_RADII_MILES:
            use_bounds_before, use_bounds = use_bounds, None
            radius_before, radius_miles = radius_miles, step
            found = inside(active)
            use_bounds, radius_miles = use_bounds_before, radius_before
            if not found.empty:
                info.update(status="nearby", radius_miles=step, matched_by="radius", message=(
                    f"No properties were found for your requested location. Here are nearby properties you can explore instead (within {step:g} miles)."))
                break
        if found.empty:
            widest = settings.LOCATION_FALLBACK_RADII_MILES[-1] if settings.LOCATION_FALLBACK_RADII_MILES else 0
            info.update(status="empty", message=f"No properties were found in {name} or within {widest:g} miles of it.")

    plan = {
        "intent": "list", "names": [], "view": None, "task_id": query_service.new_task_id(), "constraints": [],
        "metric_constraints": [], "sort": None, "fields": [], "chart": None, "chart_metric": None,
        "display_metrics": [], "ym_from": None, "ym_to": None, "radius": None, "count": None,
        "ctx": None, "decision": {"use": "none", "new_topic": True, "refine": False},
        "layout": {"focus": "locate", "sections": ["map", "properties"]},
    }
    pool = [query_service.clean_record(r) for r in found.to_dict(orient="records")]
    if not pool:
        return _empty_response(plan, _Query(f"Properties in {name}"), "no_results", info["message"], "list", info, warnings)
    selected = _initial_selection(plan, pool, False)
    keys = [str(r["PropertyBizKey"]) for r in selected]
    notes = [info["message"]] if info["message"] else []
    task = _task(plan, "list", None, selected, "result", None, None, [], [], notes)
    response = {
        "intent": "search", "results": selected, "unresolved": [], **task, **_selection_payload(plan, _complete_pool(pool, origin=(lat, lng)), selected),
        "area": {"name": name, "lat": lat, "lng": lng, "radius_miles": radius_miles if not use_bounds else None,
                 "bounds": {"north": north, "south": south, "east": east, "west": west} if use_bounds else None,
                 "total_matches": total if info["status"] == "exact" else len(pool)},
    }
    return _finalize(response, _Query(f"Properties in {name}"), plan, [], keys, info, warnings)


@app.get("/properties/{biz_key}", dependencies=[Depends(verify_secret_key)])
def get_property(biz_key: str):
    record = property_service.get_by_bizkey(biz_key)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Property '{biz_key}' not found.")
    return record

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/properties/{biz_key}/unit-count", dependencies=[Depends(verify_secret_key)])
def property_unit_count(
    biz_key: str,
    year_month_from: Optional[int] = Query(None),
    year_month_to: Optional[int] = Query(None),
):
    record = property_service.get_by_bizkey(biz_key)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Property '{biz_key}' not found.")
    try:
        frame, meta = unit_service.get_period_metrics(year_month_from, year_month_to, keys=[biz_key])
        monthly = unit_service.get_monthly_unit_counts(biz_key, year_month_from, year_month_to)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    metrics = query_service.clean_record(frame.iloc[0].to_dict()) if not frame.empty else None
    return {
        "PropertyBizKey": str(record["PropertyBizKey"]),
        "PropertyName": record["PropertyName"],
        "period": meta,
        "metrics": metrics,
        "monthly": monthly,
    }


@app.get("/properties/{biz_key}/financials", dependencies=[Depends(verify_secret_key)])
def property_financials(
    biz_key: str,
    view: str = Query("actual"),
    year_month_from: Optional[int] = Query(None),
    year_month_to: Optional[int] = Query(None),
    top_accounts: Optional[int] = Query(None, ge=1, le=500),
):
    resolved_view = _resolve_view(view) or "actual"
    if property_service.get_by_bizkey(biz_key) is None:
        raise HTTPException(status_code=404, detail=f"Property '{biz_key}' not found.")
    try:
        return financial_service.get_property_financials(
            biz_key, resolved_view, year_month_from, year_month_to, top_accounts
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Financials lookup failed for %s: %s", biz_key, e)
        raise HTTPException(status_code=502, detail="Financial data is temporarily unavailable.")


@app.get("/properties/{biz_key}/nearest", dependencies=[Depends(verify_secret_key)])
def nearest_properties(
    biz_key: str,
    count: int = Query(None, ge=1, le=settings.MAX_NEARBY_COUNT),
    active_only: bool = Query(True),
    view: Optional[str] = Query(None),
    year_month_from: Optional[int] = Query(None),
    year_month_to: Optional[int] = Query(None),
):
    resolved_view = _resolve_view(view)
    try:
        result = geo_service.find_nearest_properties(
            target_biz_key=biz_key,
            count=count,
            active_only=active_only,
            return_all=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    pool = result["nearby"]
    selected = pool[:_selection_size()]
    result = {**result, "nearby": selected, **_endpoint_selection(result["target"], pool, selected)}
    if resolved_view:
        all_keys = [result["target"]["PropertyBizKey"]] + [p["PropertyBizKey"] for p in selected]
        result = {**result, **_financial_payload(all_keys, resolved_view, year_month_from, year_month_to)}
    return result


@app.post("/properties/compare", dependencies=[Depends(verify_secret_key)])
def compare_properties(req: CompareRequest):
    if len(req.biz_keys) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two PropertyBizKeys to compare.")
    resolved_view = _resolve_view(req.view)
    try:
        return comparison_service.build_comparison(
            req.biz_keys,
            fields=req.fields,
            view=resolved_view,
            ym_from=req.year_month_from,
            ym_to=req.year_month_to,
            top_accounts=req.top_accounts,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/properties/{biz_key}/compare-with-nearest", dependencies=[Depends(verify_secret_key)])
def compare_with_nearest(
    biz_key: str,
    count: int = Query(None, ge=1, le=settings.MAX_NEARBY_COUNT),
    active_only: bool = Query(True),
    view: Optional[str] = Query(None),
    year_month_from: Optional[int] = Query(None),
    year_month_to: Optional[int] = Query(None),
    top_accounts: Optional[int] = Query(None, ge=1, le=500),
):
    resolved_view = _resolve_view(view)
    try:
        nearest = geo_service.find_nearest_properties(
            target_biz_key=biz_key, count=count, active_only=active_only, return_all=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    pool = nearest["nearby"]
    selected = pool[:_selection_size()]
    all_keys = [nearest["target"]["PropertyBizKey"]] + [p["PropertyBizKey"] for p in selected]
    comparison = comparison_service.build_comparison(all_keys)

    response = {
        "target": nearest["target"],
        "nearby": selected,
        "comparison": comparison,
        **_endpoint_selection(nearest["target"], pool, selected),
    }
    response.update(_financial_payload(all_keys, resolved_view, year_month_from, year_month_to, top_accounts))
    return response


MAX_FINANCIAL_PROPERTIES = 25


def _dedupe(items: list) -> list:
    seen = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return seen


def _nearby_error(error: Exception) -> HTTPException:
    text = str(error)
    if "coordinates" in text:
        return errors.fail(422, "no_coordinates", text.replace("Property with PropertyBizKey", "Property"))
    return errors.fail(404, "not_found", text)


def _find_nearby(target_key: str, count: Optional[int], radius: Optional[float], constraints: list, plan: dict) -> dict:
    try:
        if radius is None and not constraints and not plan["metric_constraints"] and not plan["sort"]:
            result = geo_service.find_nearest_properties(target_key, count=count, return_all=True)
            return {
                "target": query_service.clean_record(result["target"]),
                "nearby": [query_service.clean_record(r) for r in result["nearby"]],
                "applied": [],
                "metric_applied": [],
                "unsupported": [],
            }
        result = geo_service.find_relevant_properties(
            target_key,
            count=count,
            max_distance_miles=radius,
            constraints=constraints,
            metric_constraints=plan["metric_constraints"],
            sort=plan["sort"],
            display_metrics=plan["display_metrics"],
            ym_from=plan["ym_from"],
            ym_to=plan["ym_to"],
            return_all=True,
        )
    except ValueError as e:
        raise _nearby_error(e)
    return {
        "target": result["target"],
        "nearby": result["nearby"],
        "applied": result["applied_constraints"],
        "metric_applied": result["metric_constraints_applied"],
        "unsupported": result["unsupported_constraints"],
    }


def _task_notes(count: Optional[int], returned: int, radius: Optional[float], applied: list, unsupported: list) -> list:
    notes = []
    if unsupported:
        notes.append(
            "These conditions could not be applied because the property data has no matching field or value: "
            + "; ".join(unsupported) + "."
        )
    if returned == 0:
        notes.append("No properties matched the request.")
    elif count and returned < count:
        where = f" within {radius:g} miles" if radius else ""
        extra = " and your conditions" if applied else ""
        noun = "property" if returned == 1 else "properties"
        notes.append(f"Only {returned} {noun} matched{where}{extra}.")
    return notes


def _financial_bundle(keys: list, view: Optional[str], req: "AskAIRequest") -> tuple:
    if not view:
        return {}, []
    limited = keys[:MAX_FINANCIAL_PROPERTIES]
    notes = []
    if len(keys) > len(limited):
        notes.append(f"Financial figures are shown for the first {len(limited)} of {len(keys)} properties.")
    payload = _financial_payload(limited, view, req.year_month_from, req.year_month_to, req.top_accounts)
    return payload, notes


SLIM_FIELDS = [key for key, _label, _kind in query_service.TABLE_COLUMNS] + ["Latitude", "Longitude"]


def _slim_record(record: dict, metric_keys: list) -> dict:
    clean = query_service.clean_record(record)
    wanted = list(SLIM_FIELDS) + metric_service.metric_columns(metric_keys)
    return {key: clean.get(key) for key in wanted if key in clean}


def _with_distances(records: list, origin: Optional[dict]) -> list:
    if not origin:
        return records
    lat, lng = origin.get("Latitude"), origin.get("Longitude")
    if lat is None or lng is None or lat != lat or lng != lng:
        return records
    out = []
    for record in records:
        item = dict(record)
        rlat, rlng = item.get("Latitude"), item.get("Longitude")
        if item.get("distance_miles") is None and rlat is not None and rlng is not None and rlat == rlat and rlng == rlng:
            miles = geo_service.haversine_distance_miles(lat, lng, np.array([rlat]), np.array([rlng]))
            item["distance_miles"] = round(float(miles[0]), 2)
        out.append(item)
    return out


def _named_pool(named: list) -> list:
    """Explicitly requested properties first, then every other property in the database."""
    seen = {str(p["PropertyBizKey"]) for p in named}
    everything = property_service.get_all_properties().sort_values("PropertyName").to_dict(orient="records")
    rest = [r for r in everything if str(r["PropertyBizKey"]) not in seen]
    return list(named) + rest


def _selection_payload(plan: dict, pool: list, selected: list, subject_key: Optional[str] = None,
                       pinned_keys: Optional[list] = None) -> dict:
    """Describe every available property and which ones are currently selected."""
    chosen = {str(r["PropertyBizKey"]) for r in selected}
    pinned = [str(k) for k in (pinned_keys or ([subject_key] if subject_key else []))]
    metric_keys = plan["display_metrics"]
    enriched = metric_service.attach_metrics([dict(r) for r in pool], metric_keys, plan["ym_from"], plan["ym_to"])
    properties = []
    for record in enriched:
        item = _slim_record(record, metric_keys)
        key = str(item.get("PropertyBizKey"))
        if key in pinned:
            continue
        item["PropertyBizKey"] = key
        item["selected"] = key in chosen
        properties.append(item)
    return {
        "properties": properties,
        "selection": {
            "default_count": max(1, len([r for r in selected if str(r["PropertyBizKey"]) not in pinned])),
            "selected_keys": [str(r["PropertyBizKey"]) for r in selected if str(r["PropertyBizKey"]) not in pinned],
            "subject_key": subject_key if len(pinned) <= 1 else None,
            "pinned_keys": pinned,
            "total_available": len(properties),
        },
    }


def _unknown_response(task_id: str, parsed: dict, message: str, code: str = "not_understood") -> dict:
    return {
        "intent": "unknown",
        "code": code,
        "message": message,
        "raw": parsed,
        "task": {"id": task_id, "kind": "unknown", "notes": [message]},
        "markers": [],
        "table": {"columns": [], "rows": []},
    }


def _plan_period(req: "AskAIRequest", parsed: dict) -> tuple:
    if req.year_month_from is not None or req.year_month_to is not None:
        try:
            period = unit_service.normalize_period(req.year_month_from, req.year_month_to)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return period
    model_period = parsed.get("period")
    if model_period:
        try:
            period = unit_service.normalize_period(model_period.get("from"), model_period.get("to"))
        except (ValueError, AttributeError):
            period = None
        if period:
            return period
    return None, None


def _build_plan(req: "AskAIRequest") -> dict:
    button_view = _resolve_view(req.view)
    ctx = context_service.clean(req.context)
    names_by_key = property_service.name_lookup() if ctx else {}
    summary = context_service.prompt_summary(ctx, lambda k: names_by_key.get(str(k)))
    parsed = ai_service.resolve_intent(req.query, query_service.catalog_prompt(), summary)
    intent = parsed["intent"]
    decision = context_service.decide(parsed, ctx, req.query)
    ctx_keys = context_service.keys_for(ctx, decision["use"])
    continuing = ctx is not None and not decision["new_topic"]

    raw_constraints = parsed.get("constraints")
    if decision["refine"] and ctx and ctx["constraints"]:
        raw_constraints = context_service.merge_constraints(ctx["constraints"], raw_constraints or [])
    location = parsed.get("location")
    if decision["refine"] and not location and ctx and ctx["location"]:
        location = {"text": ctx["location"]["requested"], "type": ctx["location"]["type"] or None}
    if continuing and not parsed.get("period") and ctx and ctx["period"]:
        parsed["period"] = ctx["period"]

    radius = parsed.get("max_distance_miles")
    wants_nearby = intent == "nearest" or parsed.get("include_nearby")
    if continuing and radius is None and wants_nearby and ctx and ctx["radius_miles"]:
        radius = ctx["radius_miles"]

    layout = layout_service.classify(
        req.query, intent, parsed.get("response_focus"),
        has_view=bool(button_view or parsed.get("data_view")),
        has_geography=bool(location or radius),
        single_property=len(parsed["property_names"]) == 1,
    )
    sections = list(layout["sections"])
    view = button_view or parsed.get("data_view")
    if view and "financial" not in sections and sections:
        sections.insert(1 if sections[0] == "map" else 0, "financial")
    if "financial" in sections and not view:
        view = "actual"
    if "financial" not in sections:
        view = None
    layout = {"focus": layout["focus"], "sections": sections}

    constraints, flagged = query_service.sanitize_constraints(raw_constraints, parsed.get("unsupported"))
    fields, unsupported_fields = query_service.sanitize_fields(parsed.get("fields"))
    metric_keys, flagged_metrics = metric_service.sanitize_metrics(parsed.get("metrics"))
    metric_constraints, flagged_metric_constraints = metric_service.sanitize_metric_constraints(parsed.get("metric_constraints"))
    sort, flagged_sort = metric_service.sanitize_sort(parsed.get("sort"))
    flagged = _dedupe(flagged + unsupported_fields + flagged_metrics + flagged_metric_constraints + flagged_sort)

    chart = visualization_service.resolve(req.query, parsed.get("visualization"), req.chart)
    chart_metric = None
    if chart:
        chart_metric = metric_service.resolve_metric(parsed.get("chart_metric"))
        if chart_metric is None:
            if sort:
                chart_metric = sort["metric"]
            elif metric_keys:
                chart_metric = metric_keys[0]
            elif metric_constraints:
                chart_metric = metric_constraints[0]["metric"]
            else:
                chart_metric = "units"

    display_metrics = metric_service.display_metrics(metric_keys, metric_constraints, sort, chart_metric)
    ym_from, ym_to = _plan_period(req, parsed)

    return {
        "parsed": parsed,
        "intent": intent,
        "names": parsed["property_names"],
        "view": view,
        "task_id": query_service.new_task_id(),
        "radius": radius,
        "count": parsed.get("nearby_count"),
        "constraints": constraints,
        "metric_constraints": metric_constraints,
        "sort": sort,
        "fields": fields,
        "flagged": flagged,
        "chart": chart,
        "chart_metric": chart_metric,
        "display_metrics": display_metrics,
        "ym_from": ym_from,
        "ym_to": ym_to,
        "context_key": (req.context_property_key or "").strip(),
        "location": location,
        "proximity": bool(location) and (intent == "nearest" or bool(location_service.PROXIMITY_RE.search(req.query or ""))),
        "include_nearby": bool(parsed.get("include_nearby")),
        "ctx": ctx,
        "decision": decision,
        "ctx_keys": ctx_keys,
        "layout": layout,
    }


def _period_notes(plan: dict) -> tuple:
    if not plan["display_metrics"]:
        return None, []
    try:
        period = unit_service.describe_period(plan["ym_from"], plan["ym_to"])
    except ValueError:
        period = None
    if period is None:
        return None, ["Unit data is not available, so unit-based metrics could not be calculated."]
    if period["source"] == "product_table":
        return period, ["Unit counts come from the property table because monthly unit data is unavailable."]
    note = (
        f"Unit metrics are averages over {period['year_month_from']} to {period['year_month_to']} "
        f"({period['calendar_days']} calendar days): total unit-days divided by calendar days."
    )
    return period, [note]


def _task(plan: dict, kind: str, subject: Optional[dict], others: list, role: str, radius: Optional[float],
          count: Optional[int], applied: list, unsupported: list, notes: list,
          metric_applied: Optional[list] = None) -> dict:
    period, period_notes = _period_notes(plan)
    keys = plan["display_metrics"]
    subject_record = metric_service.attach_metrics([subject], keys, plan["ym_from"], plan["ym_to"])[0] if subject else None
    other_records = metric_service.attach_metrics(list(others), keys, plan["ym_from"], plan["ym_to"])
    extra = {
        "period": period,
        "metrics": keys,
        "metric_constraints_applied": metric_applied or [],
        "sort": plan["sort"],
        "fields": plan["fields"],
        "interpretation": {
            "intent": plan["intent"],
            "property_names": plan["names"],
            "constraints": plan["constraints"],
            "metric_constraints": plan["metric_constraints"],
            "sort": plan["sort"],
            "visualization": plan["chart"],
            "chart_metric": plan["chart_metric"],
        },
    }
    return query_service.build_task(
        plan["task_id"], kind, subject_record, other_records, role, plan["view"], radius, count,
        applied, unsupported, notes + period_notes, plan["fields"], keys, extra,
    )


def _present(response: dict, plan: dict) -> dict:
    response["output_format"] = "table"
    response["visualization"] = None
    chart = plan["chart"]
    if not chart:
        return response
    table = response.get("table") or {}
    if not table.get("rows"):
        return response
    column = metric_service.metric_definition(plan["chart_metric"])["column"]
    spec, note = visualization_service.build(chart, table, column)
    if spec is not None:
        response["output_format"] = chart
        response["visualization"] = spec
    elif note:
        response.setdefault("task", {}).setdefault("notes", []).append(note)
    return response


_AI_STATUS = {"ai_timeout": 504, "ai_unavailable": 503, "ai_bad_response": 502, "service_not_configured": 503}


@app.post("/ask-ai", dependencies=[Depends(verify_secret_key)])
def ask_ai(req: AskAIRequest):
    try:
        return _ask_ai(req)
    except HTTPException:
        raise
    except ai_service.IntentServiceError as e:
        raise errors.fail(_AI_STATUS.get(e.code, 503), e.code)
    except Exception as e:
        rid = errors.new_request_id()
        logger.exception("ask-ai failed [%s]", rid)
        status, code = errors.classify_exception(e)
        raise errors.fail(status, code, request_id=rid)


def _complete_pool(pool: list, anchors: Optional[list] = None, origin: Optional[tuple] = None,
                   exclude: Optional[list] = None) -> list:
    """The selection list always shows every property; a distance or filter only decides what is ticked."""
    skip = {str(r["PropertyBizKey"]) for r in pool} | {str(k) for k in (exclude or [])}
    frame = property_service.get_all_properties(active_only=True)
    rest = frame[~frame["PropertyBizKey"].astype(str).isin(skip)].copy()
    if rest.empty:
        return list(pool)

    points = []
    for anchor in anchors or []:
        lat, lng = anchor.get("Latitude"), anchor.get("Longitude")
        if lat is not None and lng is not None and lat == lat and lng == lng:
            points.append((float(lat), float(lng)))
    if origin:
        points.append((float(origin[0]), float(origin[1])))

    if points:
        lat = pd.to_numeric(rest["Latitude"], errors="coerce")
        lng = pd.to_numeric(rest["Longitude"], errors="coerce")
        valid = (lat.notna() & lng.notna()).values
        best = np.full(len(rest), np.nan)
        for plat, plng in points:
            d = geo_service.haversine_distance_miles(plat, plng, lat.fillna(0).values, lng.fillna(0).values)
            d = np.where(valid, d, np.nan)
            best = np.fmin(best, d)
        rest["distance_miles"] = np.round(best, 2)
        rest = rest.sort_values(["distance_miles", "PropertyName"], na_position="last")
    else:
        rest = rest.sort_values("PropertyName")
    return list(pool) + [query_service.clean_record(r) for r in rest.to_dict(orient="records")]


def _initial_selection(plan: dict, pool: list, radius_driven: bool) -> list:
    """Which properties start ticked: everything the radius qualifies, an explicit count, else the default."""
    if not pool:
        return []
    if radius_driven:
        return list(pool)
    count = plan.get("count")
    size = min(int(count), len(pool)) if count else _selection_size()
    return pool[:max(1, size)]


def _mark_references(response: dict, reference_keys: list) -> dict:
    refs = {str(k) for k in reference_keys}
    for row in (response.get("table") or {}).get("rows", []):
        row["is_subject"] = row.get("key") in refs
    for marker in response.get("markers", []):
        if marker.get("key") in refs:
            marker["role"] = "subject"
    task = response.get("task")
    if isinstance(task, dict):
        task["reference_keys"] = [str(k) for k in reference_keys]
    return response


def _collect_warnings(response: dict, extra: Optional[list] = None) -> list:
    warnings = list(extra or [])
    task = response.get("task") or {}
    if task.get("unplotted"):
        warnings.append({
            "code": "missing_coordinates",
            "message": "Some properties cannot be mapped because their location coordinates are unavailable.",
        })
    if response.get("financials_error"):
        warnings.append({
            "code": "financials_unavailable",
            "message": "The property results are available, but the financial data could not be loaded.",
        })
    seen, out = set(), []
    for w in warnings:
        if w["code"] not in seen:
            seen.add(w["code"])
            out.append(w)
    return out


def _finalize(response: dict, req: "AskAIRequest", plan: dict, reference_keys: list, result_keys: list,
              location: Optional[dict] = None, warnings: Optional[list] = None) -> dict:
    response["layout"] = plan["layout"]
    response["warnings"] = _collect_warnings(response, warnings)
    if location:
        response["location"] = location
    response["context"] = context_service.next_context(
        plan["ctx"], plan["decision"], req.query, plan, reference_keys, result_keys, location,
    )
    return response


def _ask_ai(req: AskAIRequest):
    plan = _build_plan(req)
    response = _execute(req, plan)
    return _present(response, plan)


def _empty_response(plan: dict, req: "AskAIRequest", code: str, message: str, intent: str,
                    location: Optional[dict] = None, warnings: Optional[list] = None) -> dict:
    response = {
        "intent": intent if intent != "list" else "search",
        "code": code,
        "message": message,
        "unresolved": [],
        "task": {"id": plan["task_id"], "kind": intent, "notes": [message]},
        "markers": [],
        "table": {"columns": [], "rows": []},
    }
    return _finalize(response, req, plan, [], [], location, warnings)


def _execute_location(req: AskAIRequest, plan: dict):
    intent = plan["intent"]
    wanted = plan["location"]
    applied_holder = {"applied": [], "unsupported": [], "metric_applied": []}

    def constrain(df):
        df, applied, unsupported = query_service.apply_constraints(df, plan["constraints"], None)
        df, metric_applied, metric_unsupported = metric_service.apply_metric_constraints(
            df, plan["metric_constraints"], plan["ym_from"], plan["ym_to"], None,
        )
        applied_holder.update(applied=applied, unsupported=unsupported + metric_unsupported, metric_applied=metric_applied)
        return df

    outcome = location_service.search(wanted["text"], wanted.get("type"), plan["radius"], constrain, plan["proximity"])
    info = outcome["location"]
    warnings = outcome["warnings"]
    records = outcome["records"]

    if info["status"] == "unresolved":
        return _empty_response(plan, req, "location_unresolved", info["message"], intent, info, warnings)
    if not records:
        code = "no_results" if info["status"] == "empty" else "location_unresolved"
        message = info["message"] or f"No properties matched {wanted['text']}."
        if plan["constraints"] or plan["metric_constraints"]:
            message += " Try expanding the location or adjusting your criteria."
        return _empty_response(plan, req, code, message, intent, info, warnings)

    if plan["sort"]:
        frame = pd.DataFrame(records)
        frame, _, _ = metric_service.ensure_metrics(frame, plan["display_metrics"], plan["ym_from"], plan["ym_to"])
        frame = metric_service.apply_sort(frame, plan["sort"], plan["ym_from"], plan["ym_to"], fallback="PropertyName")
        records = [query_service.clean_record(r) for r in frame.to_dict(orient="records")]

    radius_driven = bool(plan["radius"]) and info["status"] == "exact"
    pool = records
    selected = _initial_selection(plan, pool, radius_driven)
    keys = [str(r["PropertyBizKey"]) for r in selected]
    financials, fin_notes = _financial_bundle(keys, plan["view"], req)
    unsupported = _dedupe(plan["flagged"] + applied_holder["unsupported"])
    applied_all = applied_holder["applied"] + applied_holder["metric_applied"]
    notes = _task_notes(plan["count"], len(pool), plan["radius"], applied_all, unsupported) + fin_notes
    if info["message"] and info["status"] in ("nearby", "exact"):
        notes.insert(0, info["message"])
    if len(pool) > len(selected):
        notes.append(f"{len(pool)} properties found; {len(selected)} are selected. Tick others to add them.")
    task = _task(plan, "list", None, selected, "result", plan["radius"], plan["count"],
                 applied_holder["applied"], unsupported, notes, applied_holder["metric_applied"])
    origin = info.get("origin")
    full = _complete_pool(pool, origin=(origin["lat"], origin["lng"]) if origin else None)
    selection = _selection_payload(plan, full, selected)
    response = {"intent": "search", "results": selected, "unresolved": [], **task, **financials, **selection}
    return _finalize(response, req, plan, [], keys, info, warnings)


def _reference_response(req: AskAIRequest, plan: dict, kind: str, refs: list, pool: list, radius: float,
                        unsupported: list, applied: list, metric_applied: list, unresolved: list,
                        extra_warnings: list, extra_notes: list):
    selected = list(pool)
    ref_keys = [str(r["PropertyBizKey"]) for r in refs]
    keys = ref_keys + [str(p["PropertyBizKey"]) for p in selected]
    comparison = None
    if kind == "compare" and len(keys) >= 2:
        comparison = comparison_service.build_comparison(
            keys, fields=plan["fields"], ym_from=plan["ym_from"], ym_to=plan["ym_to"],
        )
    financials, fin_notes = _financial_bundle(keys, plan["view"], req)
    notes = _task_notes(plan["count"], len(pool), radius, applied + metric_applied, unsupported) + fin_notes + extra_notes
    role = "comparison" if kind == "compare" else "nearby"
    task = _task(plan, kind, None, [query_service.clean_record(r) for r in refs] + selected, role, radius,
                 plan["count"], applied, unsupported, notes, metric_applied)
    response = {
        "intent": kind,
        "results": [query_service.clean_record(r) for r in refs] + selected,
        "references": [query_service.clean_record(r) for r in refs],
        "unresolved": unresolved,
        **task,
        **financials,
        **_selection_payload(plan, _complete_pool(pool, anchors=refs, exclude=ref_keys), selected, None, ref_keys),
    }
    if comparison is not None:
        response["comparison"] = comparison
    _mark_references(response, ref_keys)
    warnings = list(extra_warnings)
    if not pool:
        warnings.append({
            "code": "no_nearby",
            "message": f"No eligible properties were found within {radius:g} miles of the selected properties.",
        })
    return _finalize(response, req, plan, ref_keys, keys, None, warnings)


def _execute(req: AskAIRequest, plan: dict):
    parsed = plan["parsed"]
    intent = plan["intent"]
    names = plan["names"]
    view = plan["view"]
    task_id = plan["task_id"]
    radius = plan["radius"]
    count = plan["count"]
    constraints = plan["constraints"]
    metric_constraints = plan["metric_constraints"]
    flagged = plan["flagged"]
    fields = plan["fields"]
    context_key = plan["context_key"]
    ctx_keys = plan["ctx_keys"]

    generic_message = "I couldn't tell what you want to find. Try naming a property, a place (city, state, ZIP) or the conditions you care about."

    if intent == "unknown":
        return _finalize(_unknown_response(task_id, parsed, generic_message), req, plan, [], [])

    if plan["location"] and not names and not ctx_keys:
        return _execute_location(req, plan)

    if intent == "list" and not names:
        refine_only = bool(ctx_keys) and not constraints and not metric_constraints and not plan["sort"]
        if not refine_only:
            if not constraints and not metric_constraints and not plan["sort"]:
                return _finalize(_unknown_response(task_id, parsed, generic_message), req, plan, [], [])
            listing = query_service.filter_properties(
                constraints,
                limit=count,
                metric_constraints=metric_constraints,
                sort=plan["sort"],
                display_metrics=plan["display_metrics"],
                ym_from=plan["ym_from"],
                ym_to=plan["ym_to"],
                return_all=True,
                restrict_keys=ctx_keys or None,
            )
            pool = listing["results"]
            unsupported = _dedupe(flagged + listing["unsupported_constraints"])
            if not pool:
                message = "No properties matched your current filters. Try expanding the location or adjusting your criteria."
                return _empty_response(plan, req, "no_results", message, intent)
            rows = _initial_selection(plan, pool, False)
            keys = [str(r["PropertyBizKey"]) for r in rows]
            financials, fin_notes = _financial_bundle(keys, view, req)
            applied_all = listing["applied_constraints"] + listing["metric_constraints_applied"]
            notes = _task_notes(count, len(pool), None, applied_all, unsupported) + fin_notes
            if len(pool) > len(rows):
                notes.append(f"{len(pool)} properties matched; the first {len(rows)} are selected. Tick others to add them.")
            task = _task(
                plan, "list", None, rows, "result", None, count,
                listing["applied_constraints"], unsupported, notes, listing["metric_constraints_applied"],
            )
            selection = _selection_payload(plan, _complete_pool(pool), rows)
            response = {"intent": "search", "results": rows, "unresolved": [], **task, **financials, **selection}
            return _finalize(response, req, plan, [], keys)

    if intent != "list" and not names and not ctx_keys and not context_key:
        return _finalize(_unknown_response(task_id, parsed, generic_message), req, plan, [], [])

    resolved = []
    unresolved = []
    seen = set()
    for record in property_service.get_many(ctx_keys):
        resolved.append(record)
        seen.add(str(record["PropertyBizKey"]))
    for name in names:
        matches = property_service.search_by_name(name, limit=1)
        if matches and str(matches[0]["PropertyBizKey"]) not in seen:
            resolved.append(matches[0])
            seen.add(str(matches[0]["PropertyBizKey"]))
        elif not matches:
            unresolved.append(name)

    if not names and not ctx_keys and context_key:
        context = property_service.get_by_bizkey(context_key)
        if context is not None:
            resolved = [context]

    if not resolved:
        if names:
            label = ", ".join(names)
            message = f"No property matching \"{label}\" was found. Check the spelling, or search by city, state or ZIP code."
        else:
            message = "The properties from earlier in this conversation are no longer available. Ask your question again."
        response = _empty_response(plan, req, "no_match", message, intent)
        response["unresolved"] = unresolved
        return response

    effective = "nearest" if intent == "list" else intent
    wants_nearby = effective == "nearest" or radius is not None or plan["include_nearby"]
    driven = radius is not None

    if effective == "search" or (effective == "compare" and len(resolved) >= 2 and not wants_nearby):
        chosen = resolved if effective == "compare" else resolved[:_selection_size()]
        if effective == "compare":
            chosen = resolved[:_selection_size()] if not ctx_keys else resolved
        subject = chosen[0] if len(chosen) == 1 else None
        biz_keys = [str(p["PropertyBizKey"]) for p in chosen]
        financials, fin_notes = _financial_bundle(biz_keys, view, req)
        if effective == "compare":
            comparison = comparison_service.build_comparison(
                biz_keys, fields=fields, ym_from=plan["ym_from"], ym_to=plan["ym_to"],
            )
            task = _task(
                plan, "compare", query_service.clean_record(chosen[0]),
                [query_service.clean_record(p) for p in chosen[1:]], "comparison",
                None, None, [], flagged, fin_notes,
            )
            selection = _selection_payload(plan, _named_pool(resolved), chosen)
            response = {"intent": "compare", "comparison": comparison, "unresolved": unresolved, **task, **financials, **selection}
        else:
            others = [] if subject else chosen
            task = _task(plan, "search", subject, others, "result", None, None, [], flagged, fin_notes)
            selection = _selection_payload(plan, _named_pool(resolved), chosen)
            response = {"intent": "search", "results": chosen, "unresolved": unresolved, **task, **financials, **selection}
        return _finalize(response, req, plan, [], biz_keys)

    kind = "compare" if effective == "compare" else "nearest"

    if len(resolved) >= 2:
        used_radius = radius if radius is not None else settings.DEFAULT_REFERENCE_RADIUS_MILES
        extra_notes = []
        if radius is None:
            extra_notes.append(f"No distance was given, so properties within {used_radius:g} miles were included.")
        try:
            found = geo_service.find_near_references(
                [str(p["PropertyBizKey"]) for p in resolved], used_radius, constraints,
                plan["metric_constraints"], plan["sort"], plan["display_metrics"], plan["ym_from"], plan["ym_to"],
            )
        except ValueError as e:
            raise _nearby_error(e)
        warnings = []
        if found["skipped_references"]:
            warnings.append({
                "code": "missing_coordinates",
                "message": "Some reference properties have no coordinates and were not used as search centres: "
                           + ", ".join(found["skipped_references"]) + ".",
            })
        if found["skipped_candidates"]:
            warnings.append({
                "code": "missing_coordinates",
                "message": f"{found['skipped_candidates']} other properties have no usable coordinates and could not be checked for distance.",
            })
        unsupported = _dedupe(flagged + found["unsupported_constraints"])
        return _reference_response(
            req, plan, kind, found["references"], found["nearby"], used_radius, unsupported,
            found["applied_constraints"], found["metric_constraints_applied"], unresolved, warnings, extra_notes,
        )

    found = _find_nearby(resolved[0]["PropertyBizKey"], count, radius, constraints, plan)
    unsupported = _dedupe(flagged + found["unsupported"])
    pool = found["nearby"]
    selected = _initial_selection(plan, pool, driven)
    target_key = str(found["target"]["PropertyBizKey"])
    all_keys = [target_key] + [str(p["PropertyBizKey"]) for p in selected]
    notes = _task_notes(count, len(pool), radius, found["applied"] + found["metric_applied"], unsupported)
    comparison = None
    if kind == "compare":
        comparison = comparison_service.build_comparison(
            all_keys, fields=fields, ym_from=plan["ym_from"], ym_to=plan["ym_to"],
        )
    financials, fin_notes = _financial_bundle(all_keys, view, req)
    task = _task(
        plan, kind, found["target"], selected, "comparison" if kind == "compare" else "nearby", radius, count,
        found["applied"], unsupported, notes + fin_notes, found["metric_applied"],
    )
    selection = _selection_payload(
        plan, _complete_pool(pool, anchors=[found["target"]], exclude=[target_key]), selected, target_key,
    )
    response = {
        "intent": kind,
        "target": found["target"],
        "nearby": selected,
        "unresolved": unresolved,
        **task,
        **financials,
        **selection,
    }
    if comparison is not None:
        response["comparison"] = comparison
    warnings = []
    if not pool:
        where = f" within {radius:g} miles" if radius else ""
        warnings.append({
            "code": "no_nearby",
            "message": f"No eligible properties were found{where} of the selected property.",
        })
    return _finalize(response, req, plan, [target_key], all_keys, None, warnings)


_SELECTION_ROLES = {"nearest": "nearby", "compare": "comparison", "list": "result", "search": "result"}


@app.post("/properties/selection", dependencies=[Depends(verify_secret_key)])
def update_selection(req: SelectionRequest):
    """Rebuild map markers, table, comparison and financials for the current checkbox selection."""
    kind = (req.kind or "compare").strip().lower()
    if kind not in _SELECTION_ROLES:
        raise HTTPException(status_code=400, detail=f"Unsupported kind '{req.kind}'.")
    view = _resolve_view(req.view)
    try:
        period = unit_service.normalize_period(req.year_month_from, req.year_month_to)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    ym_from, ym_to = period if period else (None, None)

    metric_keys, _ = metric_service.sanitize_metrics(req.metrics)
    fields, _ = query_service.sanitize_fields(req.fields)
    chart = visualization_service.resolve("", None, req.chart)
    chart_metric = None
    if chart:
        chart_metric = metric_service.resolve_metric(req.chart_metric) or (metric_keys[0] if metric_keys else "units")
    display_metrics = metric_service.display_metrics(metric_keys, [], None, chart_metric)

    subject_key = (req.subject_key or "").strip()
    pinned_keys = _dedupe([str(k).strip() for k in (req.reference_keys or []) if str(k).strip()])
    if len(pinned_keys) >= 2:
        return _selection_with_references(
            req, kind, view, ym_from, ym_to, fields, chart, chart_metric, display_metrics, pinned_keys,
        )
    if len(pinned_keys) == 1 and not subject_key:
        subject_key = pinned_keys[0]
    subject = None
    if subject_key:
        subject = property_service.get_by_bizkey(subject_key)
        if subject is None:
            raise errors.fail(404, "not_found", f"Property '{subject_key}' not found.")

    records, not_found = [], []
    for key in _dedupe([str(k).strip() for k in req.biz_keys]):
        if subject is not None and key == str(subject["PropertyBizKey"]):
            continue
        record = property_service.get_by_bizkey(key)
        if record is None:
            not_found.append(key)
        else:
            records.append(record)

    origin_point = None
    if subject is None and isinstance(req.origin, dict):
        try:
            origin_point = {"Latitude": float(req.origin["lat"]), "Longitude": float(req.origin["lng"])}
        except (KeyError, TypeError, ValueError):
            origin_point = None

    if subject is not None:
        records = _with_distances(records, subject)
        records.sort(key=lambda r: (r.get("distance_miles") is None, r.get("distance_miles") or 0))
    elif origin_point is not None:
        records = _with_distances(records, origin_point)
        records.sort(key=lambda r: (r.get("distance_miles") is None, r.get("distance_miles") or 0))
    elif kind == "compare" and records:
        subject = records.pop(0)
    elif kind == "search" and len(records) == 1:
        subject = records.pop(0)

    plan = {
        "intent": kind, "names": [], "view": view, "task_id": query_service.new_task_id(),
        "constraints": [], "metric_constraints": [], "sort": None, "fields": fields,
        "chart": chart, "chart_metric": chart_metric, "display_metrics": display_metrics,
        "ym_from": ym_from, "ym_to": ym_to,
    }
    role = _SELECTION_ROLES[kind]
    notes = [f"Properties not found: {', '.join(not_found)}."] if not_found else []
    task = _task(plan, kind, subject, records, role, None, None, [], [], notes)

    everyone = ([subject] if subject is not None else []) + records
    keys = [str(p["PropertyBizKey"]) for p in everyone]
    shape = {}
    if subject_key and subject is not None:
        shape = {
            "target": query_service.clean_record(subject),
            "nearby": [query_service.clean_record(r) for r in records],
        }
    else:
        shape = {"results": [query_service.clean_record(r) for r in everyone]}
    response = {
        "intent": "search" if kind == "list" else kind,
        "unresolved": not_found,
        **shape,
        **task,
        "selection": {
            "selected_keys": [k for k in keys if k != (subject_key or None)],
            "subject_key": subject_key or None,
            "pinned_keys": [subject_key] if subject_key else [],
            "total_selected": len(everyone),
        },
    }
    if not keys:
        response["task"]["notes"].append("No properties are selected.")
        return _present(response, plan)

    if len(keys) >= 2:
        try:
            response["comparison"] = comparison_service.build_comparison(
                keys, fields=fields, ym_from=ym_from, ym_to=ym_to,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        response["comparison"] = None

    financials, fin_notes = _financial_bundle(keys, view, req)
    response.update(financials)
    response["task"]["notes"].extend(fin_notes)
    response["warnings"] = _collect_warnings(response)
    return _present(response, plan)


def _selection_with_references(req, kind, view, ym_from, ym_to, fields, chart, chart_metric,
                               display_metrics, pinned_keys):
    """Selection update when several reference properties are pinned (portfolio vs. nearby)."""
    refs = property_service.get_many(pinned_keys)
    if not refs:
        raise errors.fail(404, "not_found", "The reference properties could not be found.")
    ref_set = {str(r["PropertyBizKey"]) for r in refs}
    others, not_found = [], []
    for key in _dedupe([str(k).strip() for k in req.biz_keys]):
        if key in ref_set:
            continue
        record = property_service.get_by_bizkey(key)
        (others if record is not None else not_found).append(record if record is not None else key)

    origins = [r for r in refs if r.get("Latitude") == r.get("Latitude") and r.get("Longitude") == r.get("Longitude")
               and r.get("Latitude") is not None]
    enriched = []
    for record in others:
        item = dict(record)
        best = None
        for origin in origins:
            near = _with_distances([record], origin)[0].get("distance_miles")
            if near is not None and (best is None or near < best[0]):
                best = (near, str(origin["PropertyName"]))
        if best:
            item["distance_miles"], item["nearest_reference"] = best
        enriched.append(item)
    enriched.sort(key=lambda r: (r.get("distance_miles") is None, r.get("distance_miles") or 0))

    plan = {
        "intent": kind, "names": [], "view": view, "task_id": query_service.new_task_id(),
        "constraints": [], "metric_constraints": [], "sort": None, "fields": fields,
        "chart": chart, "chart_metric": chart_metric, "display_metrics": display_metrics,
        "ym_from": ym_from, "ym_to": ym_to,
    }
    role = _SELECTION_ROLES[kind]
    notes = [f"Properties not found: {', '.join(not_found)}."] if not_found else []
    everyone = list(refs) + enriched
    task = _task(plan, kind, None, everyone, role, None, None, [], [], notes)
    keys = [str(p["PropertyBizKey"]) for p in everyone]
    response = {
        "intent": "search" if kind == "list" else kind,
        "results": [query_service.clean_record(r) for r in everyone],
        "references": [query_service.clean_record(r) for r in refs],
        "unresolved": not_found,
        **task,
        "selection": {
            "selected_keys": [k for k in keys if k not in ref_set],
            "subject_key": None,
            "pinned_keys": [str(r["PropertyBizKey"]) for r in refs],
            "total_selected": len(keys),
        },
    }
    _mark_references(response, [str(r["PropertyBizKey"]) for r in refs])
    if len(keys) >= 2:
        response["comparison"] = comparison_service.build_comparison(keys, fields=fields, ym_from=ym_from, ym_to=ym_to)
    financials, fin_notes = _financial_bundle(keys, view, req)
    response.update(financials)
    response["task"]["notes"].extend(fin_notes)
    response["warnings"] = _collect_warnings(response)
    return _present(response, plan)


def _scrub_nan(value):
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {k: _scrub_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_nan(v) for v in value]
    return value


@app.post("/save-chat", dependencies=[Depends(verify_secret_key)])
def save_chat(req: SaveChatRequest):
    try:
        validate_savechat_settings()
    except RuntimeError as e:
        logger.error("%s", e)
        raise errors.fail(503, "save_not_configured")

    saved_at = datetime.now(timezone.utc).isoformat()
    try:
        payload = jsonable_encoder({
            "query": req.query,
            "response": req.response,
            "user_id": req.user_id,
            "session_id": req.session_id,
            "metadata": req.metadata,
            "saved_at": saved_at,
        })
        body = json.dumps(_scrub_nan(payload), default=str, allow_nan=False)
    except Exception:
        logger.exception("Saving chat failed: payload is not serialisable")
        raise errors.fail(400, "invalid_request", "The response could not be prepared for saving.")

    last_error = None
    for attempt in range(2):
        try:
            resp = requests.post(
                settings.SAVECHAT_URI,
                data=body.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                timeout=settings.SAVECHAT_TIMEOUT_SECONDS,
            )
        except requests.Timeout as e:
            last_error = f"Timeout: {e}"
            break
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
            continue
        if 200 <= resp.status_code < 300:
            return {"saved": True, "saved_at": saved_at}
        last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
        if resp.status_code < 500:
            break

    logger.error("Saving chat failed: %s", last_error)
    raise errors.fail(502, "save_failed")

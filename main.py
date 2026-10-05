import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

import property_service
import geo_service
import comparison_service
import financial_service
import query_service
import ai_service
from db import check_connection
from config import settings, validate_secret_key_settings, validate_savechat_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ask Property Data Intelligence API", version="1.23.0")


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
        raise HTTPException(status_code=503, detail="Service is not configured.")

    if not x_secret_key or not secrets.compare_digest(
        x_secret_key.encode("utf-8"), settings.SECRET_KEY.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing secret key.")


def _resolve_view(value: Optional[str]) -> Optional[str]:
    try:
        return financial_service.normalize_view(value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    }


@app.post("/admin/refresh-cache", dependencies=[Depends(verify_secret_key)])
def refresh_cache():
    df = property_service.refresh_now()
    financial = financial_service.refresh_now()
    return {"refreshed": True, "row_count": len(df), "financial": financial}


@app.get("/admin/financial-diagnostics", dependencies=[Depends(verify_secret_key)])
def financial_diagnostics():
    return financial_service.diagnostics()


@app.get("/properties/search", dependencies=[Depends(verify_secret_key)])
def search_properties(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
    active_only: bool = Query(False),
):
    results = property_service.search_by_name(q, limit=limit, active_only=active_only)
    return {"query": q, "count": len(results), "results": results}


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
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if resolved_view:
        all_keys = [result["target"]["PropertyBizKey"]] + [p["PropertyBizKey"] for p in result["nearby"]]
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
            target_biz_key=biz_key, count=count, active_only=active_only,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    all_keys = [nearest["target"]["PropertyBizKey"]] + [p["PropertyBizKey"] for p in nearest["nearby"]]
    comparison = comparison_service.build_comparison(all_keys)

    response = {
        "target": nearest["target"],
        "nearby": nearest["nearby"],
        "comparison": comparison,
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


def _find_nearby(target_key: str, count: Optional[int], radius: Optional[float], constraints: list) -> dict:
    try:
        if radius is None and not constraints:
            result = geo_service.find_nearest_properties(target_key, count=count)
            return {
                "target": query_service.clean_record(result["target"]),
                "nearby": [query_service.clean_record(r) for r in result["nearby"]],
                "applied": [],
                "unsupported": [],
            }
        result = geo_service.find_relevant_properties(
            target_key, count=count, max_distance_miles=radius, constraints=constraints,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "target": result["target"],
        "nearby": result["nearby"],
        "applied": result["applied_constraints"],
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


def _unknown_response(task_id: str, parsed: dict, message: str) -> dict:
    return {
        "intent": "unknown",
        "message": message,
        "raw": parsed,
        "task": {"id": task_id, "kind": "unknown", "notes": [message]},
        "markers": [],
        "table": {"columns": [], "rows": []},
    }


@app.post("/ask-ai", dependencies=[Depends(verify_secret_key)])
def ask_ai(req: AskAIRequest):
    button_view = _resolve_view(req.view)
    parsed = ai_service.resolve_intent(req.query, query_service.catalog_prompt())
    intent = parsed["intent"]
    names = parsed["property_names"]
    view = button_view or parsed.get("data_view")
    task_id = query_service.new_task_id()
    radius = parsed.get("max_distance_miles")
    count = parsed.get("nearby_count")
    constraints, flagged = query_service.sanitize_constraints(parsed.get("constraints"), parsed.get("unsupported"))
    fields, unsupported_fields = query_service.sanitize_fields(parsed.get("fields"))
    flagged = _dedupe(flagged + unsupported_fields)
    context_key = (req.context_property_key or "").strip()

    generic_message = "Could not understand the request. Try rephrasing or specify a property name."

    if intent == "unknown":
        return _unknown_response(task_id, parsed, generic_message)

    if intent == "list" and not names:
        if not constraints:
            return _unknown_response(task_id, parsed, generic_message)
        listing = query_service.filter_properties(constraints, limit=count)
        rows = listing["results"]
        unsupported = _dedupe(flagged + listing["unsupported_constraints"])
        keys = [str(r["PropertyBizKey"]) for r in rows]
        financials, fin_notes = _financial_bundle(keys, view, req)
        notes = _task_notes(count, len(rows), None, listing["applied_constraints"], unsupported) + fin_notes
        if listing["total_matches"] > len(rows):
            notes.append(f"Showing {len(rows)} of {listing['total_matches']} matching properties.")
        task = query_service.build_task(
            task_id, "list", None, rows, "result", view, None, count,
            listing["applied_constraints"], unsupported, notes, fields,
        )
        return {"intent": "search", "results": rows, "unresolved": [], **task, **financials}

    if intent != "list" and not names and not context_key:
        return _unknown_response(task_id, parsed, generic_message)

    resolved = []
    unresolved = []
    for name in names:
        matches = property_service.search_by_name(name, limit=1)
        if matches:
            resolved.append(matches[0])
        else:
            unresolved.append(name)

    if not names and context_key:
        context = property_service.get_by_bizkey(context_key)
        if context is not None:
            resolved = [context]

    if not resolved:
        label = ", ".join(names) if names else context_key
        return {
            "intent": intent if intent != "list" else "search",
            "message": f"No properties found matching: {label}",
            "unresolved": unresolved,
            "task": {"id": task_id, "kind": intent, "notes": [f"No properties found matching: {label}"]},
            "markers": [],
            "table": {"columns": [], "rows": []},
        }

    effective = "nearest" if intent == "list" else intent

    if effective == "search":
        subject = resolved[0] if len(resolved) == 1 else None
        others = [] if subject else resolved
        keys = [str(p["PropertyBizKey"]) for p in resolved]
        financials, fin_notes = _financial_bundle(keys, view, req)
        task = query_service.build_task(
            task_id, "search", subject, others, "result", view, None, None, [], flagged, fin_notes, fields,
        )
        return {"intent": "search", "results": resolved, "unresolved": unresolved, **task, **financials}

    if effective == "nearest":
        found = _find_nearby(resolved[0]["PropertyBizKey"], count, radius, constraints)
        unsupported = _dedupe(flagged + found["unsupported"])
        keys = [str(found["target"]["PropertyBizKey"])] + [str(p["PropertyBizKey"]) for p in found["nearby"]]
        financials, fin_notes = _financial_bundle(keys, view, req)
        notes = _task_notes(count, len(found["nearby"]), radius, found["applied"], unsupported) + fin_notes
        task = query_service.build_task(
            task_id, "nearest", found["target"], found["nearby"], "nearby", view, radius, count,
            found["applied"], unsupported, notes, fields,
        )
        return {
            "intent": "nearest",
            "target": found["target"],
            "nearby": found["nearby"],
            "unresolved": unresolved,
            **task,
            **financials,
        }

    if effective == "compare":
        if len(resolved) >= 2:
            biz_keys = [str(p["PropertyBizKey"]) for p in resolved]
            comparison = comparison_service.build_comparison(biz_keys, fields=fields)
            financials, fin_notes = _financial_bundle(biz_keys, view, req)
            task = query_service.build_task(
                task_id, "compare", query_service.clean_record(resolved[0]),
                [query_service.clean_record(p) for p in resolved[1:]], "comparison",
                view, None, None, [], flagged, fin_notes, fields,
            )
            return {"intent": "compare", "comparison": comparison, "unresolved": unresolved, **task, **financials}

        found = _find_nearby(resolved[0]["PropertyBizKey"], count, radius, constraints)
        unsupported = _dedupe(flagged + found["unsupported"])
        all_keys = [str(found["target"]["PropertyBizKey"])] + [str(p["PropertyBizKey"]) for p in found["nearby"]]
        comparison = comparison_service.build_comparison(all_keys, fields=fields)
        financials, fin_notes = _financial_bundle(all_keys, view, req)
        notes = _task_notes(count, len(found["nearby"]), radius, found["applied"], unsupported) + fin_notes
        task = query_service.build_task(
            task_id, "compare", found["target"], found["nearby"], "comparison", view, radius, count,
            found["applied"], unsupported, notes, fields,
        )
        return {
            "intent": "compare",
            "target": found["target"],
            "nearby": found["nearby"],
            "comparison": comparison,
            "unresolved": unresolved,
            **task,
            **financials,
        }

    return _unknown_response(task_id, parsed, generic_message)


@app.post("/save-chat", dependencies=[Depends(verify_secret_key)])
def save_chat(req: SaveChatRequest):
    try:
        validate_savechat_settings()
    except RuntimeError as e:
        logger.error("%s", e)
        raise HTTPException(status_code=503, detail="Chat saving is not configured.")

    payload = {
        "query": req.query,
        "response": req.response,
        "user_id": req.user_id,
        "session_id": req.session_id,
        "metadata": req.metadata,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        resp = requests.post(
            settings.SAVECHAT_URI,
            json=payload,
            timeout=settings.SAVECHAT_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Saving chat failed: %s", e)
        raise HTTPException(status_code=502, detail="Failed to save chat.")

    return {"saved": True, "saved_at": payload["saved_at"]}

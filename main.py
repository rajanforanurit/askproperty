import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import property_service
import geo_service
import comparison_service
import ai_service
from db import check_connection
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Property Intelligence API", version="1.1.0")


class CompareRequest(BaseModel):
    biz_keys: list[str]
    fields: Optional[list[str]] = None


class AskAIRequest(BaseModel):
    query: str


@app.on_event("startup")
def warm_cache():
    try:
        df = property_service.get_all_properties(force_refresh=True)
        logger.info("Startup cache warm: %d properties loaded.", len(df))
    except Exception as e:
        logger.error("Cache warm-up failed at startup: %s", e)


@app.get("/health")
def health():
    db_ok = check_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "database_connected": db_ok,
        "cache_loaded": property_service._cache.is_loaded,
        "cache_age_seconds": round(property_service._cache.age_seconds, 1),
    }


@app.post("/admin/refresh-cache")
def refresh_cache():
    df = property_service.refresh_now()
    return {"refreshed": True, "row_count": len(df)}


@app.get("/properties/search")
def search_properties(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
    active_only: bool = Query(False),
):
    results = property_service.search_by_name(q, limit=limit, active_only=active_only)
    return {"query": q, "count": len(results), "results": results}


@app.get("/properties/{biz_key}")
def get_property(biz_key: str):
    record = property_service.get_by_bizkey(biz_key)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Property '{biz_key}' not found.")
    return record


@app.get("/properties/{biz_key}/nearest")
def nearest_properties(
    biz_key: str,
    count: int = Query(None, ge=1, le=settings.MAX_NEARBY_COUNT),
    active_only: bool = Query(True),
):
    try:
        result = geo_service.find_nearest_properties(
            target_biz_key=biz_key,
            count=count,
            active_only=active_only,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result


@app.post("/properties/compare")
def compare_properties(req: CompareRequest):
    if len(req.biz_keys) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two PropertyBizKeys to compare.")
    result = comparison_service.build_comparison(req.biz_keys, fields=req.fields)
    return result


@app.get("/properties/{biz_key}/compare-with-nearest")
def compare_with_nearest(
    biz_key: str,
    count: int = Query(None, ge=1, le=settings.MAX_NEARBY_COUNT),
    active_only: bool = Query(True),
):
    try:
        nearest = geo_service.find_nearest_properties(
            target_biz_key=biz_key, count=count, active_only=active_only,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    all_keys = [nearest["target"]["PropertyBizKey"]] + [p["PropertyBizKey"] for p in nearest["nearby"]]
    comparison = comparison_service.build_comparison(all_keys)

    return {
        "target": nearest["target"],
        "nearby": nearest["nearby"],
        "comparison": comparison,
    }


@app.post("/ask-ai")
def ask_ai(req: AskAIRequest):
    parsed = ai_service.resolve_intent(req.query)
    intent = parsed["intent"]
    names = parsed["property_names"]

    if intent == "unknown" or not names:
        return {
            "intent": "unknown",
            "message": "Could not understand the request. Try rephrasing or specify a property name.",
            "raw": parsed,
        }

    resolved = []
    unresolved = []
    for name in names:
        matches = property_service.search_by_name(name, limit=1)
        if matches:
            resolved.append(matches[0])
        else:
            unresolved.append(name)

    if not resolved:
        return {
            "intent": intent,
            "message": f"No properties found matching: {', '.join(names)}",
            "unresolved": unresolved,
        }

    if intent == "search":
        return {"intent": intent, "results": resolved, "unresolved": unresolved}

    if intent == "nearest":
        target = resolved[0]
        try:
            result = geo_service.find_nearest_properties(
                target["PropertyBizKey"], count=parsed["nearby_count"]
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"intent": intent, **result, "unresolved": unresolved}

    if intent == "compare":
        if len(resolved) >= 2:
            biz_keys = [p["PropertyBizKey"] for p in resolved]
            comparison = comparison_service.build_comparison(biz_keys)
            return {"intent": intent, "comparison": comparison, "unresolved": unresolved}

        target = resolved[0]
        try:
            nearest = geo_service.find_nearest_properties(
                target["PropertyBizKey"], count=parsed["nearby_count"]
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        all_keys = [nearest["target"]["PropertyBizKey"]] + [p["PropertyBizKey"] for p in nearest["nearby"]]
        comparison = comparison_service.build_comparison(all_keys)
        return {
            "intent": intent,
            "target": nearest["target"],
            "nearby": nearest["nearby"],
            "comparison": comparison,
            "unresolved": unresolved,
        }

    return {"intent": "unknown"}

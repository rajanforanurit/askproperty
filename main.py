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


@app.post("/ask-ai", dependencies=[Depends(verify_secret_key)])
def ask_ai(req: AskAIRequest):
    button_view = _resolve_view(req.view)
    parsed = ai_service.resolve_intent(req.query)
    intent = parsed["intent"]
    names = parsed["property_names"]
    view = button_view or parsed.get("data_view")

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
        keys = [p["PropertyBizKey"] for p in resolved]
        return {
            "intent": intent,
            "results": resolved,
            "unresolved": unresolved,
            **_financial_payload(keys, view, req.year_month_from, req.year_month_to, req.top_accounts),
        }

    if intent == "nearest":
        target = resolved[0]
        try:
            result = geo_service.find_nearest_properties(
                target["PropertyBizKey"], count=parsed["nearby_count"]
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        keys = [result["target"]["PropertyBizKey"]] + [p["PropertyBizKey"] for p in result["nearby"]]
        return {
            "intent": intent,
            **result,
            "unresolved": unresolved,
            **_financial_payload(keys, view, req.year_month_from, req.year_month_to, req.top_accounts),
        }

    if intent == "compare":
        if len(resolved) >= 2:
            biz_keys = [p["PropertyBizKey"] for p in resolved]
            comparison = comparison_service.build_comparison(biz_keys)
            return {
                "intent": intent,
                "comparison": comparison,
                "unresolved": unresolved,
                **_financial_payload(biz_keys, view, req.year_month_from, req.year_month_to, req.top_accounts),
            }

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
            **_financial_payload(all_keys, view, req.year_month_from, req.year_month_to, req.top_accounts),
        }

    return {"intent": "unknown"}


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

"""Structured, user-safe API errors. Technical detail goes to the logs, never to the client."""
import logging
import uuid
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

DEFAULT_MESSAGES = {
    "auth_failed": "Your API key could not be verified. Check your configuration and try again.",
    "service_not_configured": "The property analysis service is not fully configured.",
    "invalid_request": "The request was not valid. Check your input and try again.",
    "not_found": "The requested property could not be found.",
    "ai_unavailable": "The language-understanding service is unavailable right now. Please try again shortly.",
    "ai_timeout": "The question could not be interpreted in time. Please try again or simplify the question.",
    "ai_bad_response": "The question could not be interpreted. Try rephrasing it.",
    "database_unavailable": "The property database is temporarily unavailable. Please try again shortly.",
    "timeout": "The analysis took too long to complete. Try narrowing the property selection or simplifying your query.",
    "save_not_configured": "Saving is not configured on the backend.",
    "save_failed": "The response could not be saved to the configured destination.",
    "upstream_unavailable": "A data source needed for this request is temporarily unavailable. Please try again shortly.",
    "internal_error": "The property analysis failed unexpectedly.",
}

STATUS_CODES = {
    400: "invalid_request", 401: "auth_failed", 403: "auth_failed", 404: "not_found",
    408: "timeout", 502: "upstream_unavailable", 503: "service_not_configured", 504: "timeout",
}


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]


def fail(status: int, code: str, message: Optional[str] = None, **extra) -> HTTPException:
    detail = {"code": code, "message": message or DEFAULT_MESSAGES.get(code, DEFAULT_MESSAGES["internal_error"])}
    detail.update(extra)
    return HTTPException(status_code=status, detail=detail)


def normalise_detail(status: int, detail) -> dict:
    if isinstance(detail, dict) and "code" in detail:
        return detail
    code = STATUS_CODES.get(status, "internal_error" if status >= 500 else "invalid_request")
    message = detail if isinstance(detail, str) and detail else DEFAULT_MESSAGES.get(code, "")
    if not isinstance(detail, str) or not detail:
        message = DEFAULT_MESSAGES.get(code, DEFAULT_MESSAGES["internal_error"])
    return {"code": code, "message": message}


def classify_exception(exc: Exception) -> tuple:
    """Return (status, code) for an unexpected exception without exposing its text."""
    import requests
    from sqlalchemy.exc import SQLAlchemyError

    if isinstance(exc, requests.Timeout):
        return 504, "timeout"
    if isinstance(exc, SQLAlchemyError):
        return 503, "database_unavailable"
    if isinstance(exc, RuntimeError) and "environment variable" in str(exc).lower():
        return 503, "service_not_configured"
    if isinstance(exc, TimeoutError):
        return 504, "timeout"
    return 500, "internal_error"

import time
import urllib.parse
import logging
from typing import Any, Callable, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import InterfaceError, OperationalError

from config import settings, validate_settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None


def _build_pymssql_url() -> str:
    pwd_escaped = urllib.parse.quote_plus(settings.DB_PASSWORD)
    return f"mssql+pymssql://{settings.DB_USERNAME}:{pwd_escaped}@{settings.DB_SERVER}/{settings.DB_NAME}"


def _build_pyodbc_url() -> str:
    params = urllib.parse.quote_plus(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={settings.DB_SERVER};"
        f"DATABASE={settings.DB_NAME};"
        f"UID={settings.DB_USERNAME};"
        f"PWD={settings.DB_PASSWORD};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    return f"mssql+pyodbc:///?odbc_connect={params}"


def get_engine() -> Engine:
    global _engine
    if _engine is not None:
        return _engine

    validate_settings()

    if settings.DB_DRIVER == "pyodbc":
        url = _build_pyodbc_url()
    else:
        url = _build_pymssql_url()

    engine_kwargs = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "pool_size": 5,
        "max_overflow": 10,
    }
    if settings.DB_DRIVER != "pyodbc":
        # Fail a stuck login/query instead of hanging until the client gives up.
        engine_kwargs["connect_args"] = {"login_timeout": 30, "timeout": 90}

    _engine = create_engine(url, **engine_kwargs)
    logger.info("Database engine created (driver=%s, server=%s, db=%s)",
                settings.DB_DRIVER, settings.DB_SERVER, settings.DB_NAME)
    return _engine


def check_connection() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("Database connection check failed: %s", e)
        return False


TRANSIENT_DB_ERRORS = (OperationalError, InterfaceError)


def run_with_retry(fn: Callable[[], Any], attempts: int = 3, delay: float = 2.0) -> Any:
    """Run a DB call, retrying transient connection errors.

    Azure SQL can drop or pause connections; the first call after that fails and
    the next one succeeds. Retrying here keeps users from seeing that failure.
    """
    last: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except TRANSIENT_DB_ERRORS as e:
            last = e
            logger.warning("Transient database error (attempt %d/%d): %s", attempt, attempts, e)
            if attempt < attempts:
                time.sleep(delay * attempt)
    assert last is not None
    raise last


def read_sql_retry(query, params: Optional[dict] = None) -> pd.DataFrame:
    return run_with_retry(lambda: pd.read_sql(query, get_engine(), params=params))


def scalar_retry(query, params: Optional[dict] = None):
    def call():
        with get_engine().connect() as conn:
            return conn.execute(query, params or {}).scalar()
    return run_with_retry(call)

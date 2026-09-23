import urllib.parse
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

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

    _engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
    )
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

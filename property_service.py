import logging
import difflib
from typing import Optional

import pandas as pd
from sqlalchemy import text

from db import get_engine
from cache import TTLCache
from config import settings
import unit_service

logger = logging.getLogger(__name__)

PRODUCT_COLUMNS = [
    "PropertyBizKey", "PropertyName", "AccountingManager", "AccountsPayableAssociate",
    "AccountsPayableManager", "AcquisitionDate", "AssetManager", "AssetType", "City",
    "Controller", "DispositionDate", "FundCurrent", "Latitude", "Longitude",
    "MaintenanceDirector", "MaintenanceManager", "ManagementCompany", "Market",
    "MarketingManager", "MSA", "ProfitCenterCategoryCurrent", "ProfitCenterCodeCurrent",
    "ProfitCenterNameCurrent", "PropertyAddress", "PropertyAgeBucket", "PropertyCampus",
    "PropertyManagementSystemID", "PropertyManagerCurrent", "PropertySizeCategory",
    "PropertyStatus", "StabilizationDate", "State", "VicePresidentOperations",
    "YearBuilt", "Zipcode",
]


def _load_properties_from_db() -> pd.DataFrame:
    cols = ", ".join(f"[{c}]" for c in PRODUCT_COLUMNS)
    query = f"SELECT {cols} FROM [{settings.PRODUCT_TABLE}]"
    logger.info("Loading properties from %s", settings.PRODUCT_TABLE)

    df = pd.read_sql(text(query), get_engine())

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["PropertyBizKey"] = df["PropertyBizKey"].astype(str).str.strip()
    df["PropertyName"] = df["PropertyName"].astype(str).str.strip()

    logger.info("Loaded %d properties (%d with valid coordinates)",
                len(df), df[["Latitude", "Longitude"]].dropna().shape[0])
    return df


_cache = TTLCache(ttl_seconds=settings.CACHE_TTL_SECONDS, loader=_load_properties_from_db, name="properties")


def get_units_map(ym_from=None, ym_to=None) -> dict:
    return unit_service.get_units_map(ym_from, ym_to)


def get_all_properties(active_only: bool = False, force_refresh: bool = False) -> pd.DataFrame:
    df = _cache.get(force_refresh=force_refresh)
    if active_only:
        df = df[df["PropertyStatus"].isin(settings.ACTIVE_STATUSES)]
    return df.copy()


def _frame() -> pd.DataFrame:
    """The cached frame itself, NOT a copy. For read-only use; never mutate the result."""
    return _cache.get()


def get_by_bizkey(biz_key: str) -> Optional[dict]:
    df = _frame()
    match = df[df["PropertyBizKey"] == str(biz_key).strip()]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def get_many(keys: list) -> list:
    """Records for the given keys in the given order (one frame copy, unknown keys skipped)."""
    wanted = [str(k).strip() for k in keys]
    df = _frame()
    rows = df[df["PropertyBizKey"].isin(wanted)].drop_duplicates("PropertyBizKey")
    by_key = {str(r["PropertyBizKey"]): r for r in rows.to_dict(orient="records")}
    return [by_key[k] for k in wanted if k in by_key]


def name_lookup() -> dict:
    df = _frame()
    return dict(zip(df["PropertyBizKey"].astype(str), df["PropertyName"].astype(str)))


def search_by_name(query: str, limit: int = 5, active_only: bool = False) -> list[dict]:
    df = get_all_properties(active_only=True) if active_only else _frame()
    query_norm = query.strip().lower()

    exact = df[df["PropertyName"].str.lower() == query_norm]
    if not exact.empty:
        return exact.head(limit).to_dict(orient="records")

    substring = df[df["PropertyName"].str.lower().str.contains(query_norm, na=False, regex=False)]
    if not substring.empty:
        return substring.head(limit).to_dict(orient="records")

    all_names = df["PropertyName"].tolist()
    close = difflib.get_close_matches(query, all_names, n=limit, cutoff=0.6)
    if close:
        matched = df[df["PropertyName"].isin(close)]
        return matched.head(limit).to_dict(orient="records")

    return []


def refresh_now():
    unit_service.refresh()
    return get_all_properties(force_refresh=True)

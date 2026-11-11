import logging
import difflib
import re
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


# Words that appear in many property names and say nothing about WHICH property is meant.
# "ganga apartment" must not match "Lotus Apartments" just because of the shared generic word.
_GENERIC_TOKENS = {
    "apartment", "apartments", "apt", "apts", "the", "at", "of", "and", "on", "in", "by",
    "residence", "residences", "residential", "villa", "villas", "homes", "home", "living",
    "community", "communities", "property", "properties", "place", "flats", "flat",
    "condo", "condos", "townhomes", "townhome", "lofts", "loft", "village", "estates", "estate",
}
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _tokens(value) -> list:
    return [t for t in _TOKEN_SPLIT.split(str(value or "").lower()) if t]


def _is_generic(token: str) -> bool:
    if token in _GENERIC_TOKENS:
        return True
    # "apartmnts", "apartmentz" and similar typos of a generic word are still generic.
    return len(token) >= 6 and any(
        len(g) >= 6 and difflib.SequenceMatcher(None, token, g).ratio() >= 0.85 for g in _GENERIC_TOKENS
    )


def _distinctive(tokens: list) -> list:
    core = [t for t in tokens if not _is_generic(t)]
    return core or tokens


def _token_close(a: str, b: str) -> bool:
    """Same word, allowing one small typo. Very short words must match exactly."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 2:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


def _fuzzy_name_matches(query: str, df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Typo-tolerant match that never lets a generic shared word stand in for the real name.

    Every distinctive word the user typed must match a distinctive word of the property name
    (exactly, or within one small typo). One wrong distinctive word rejects the property.
    """
    wanted = _distinctive(_tokens(query))
    if not wanted:
        return df.iloc[0:0]
    scored = []
    for idx, name in df["PropertyName"].items():
        have = _distinctive(_tokens(name))
        if not have:
            continue
        total = 0.0
        for w in wanted:
            best = max((difflib.SequenceMatcher(None, w, h).ratio() for h in have if _token_close(w, h)), default=0.0)
            if best == 0.0:
                total = 0.0
                break
            total += best
        if total:
            scored.append((total / len(wanted), idx))
    if not scored:
        return df.iloc[0:0]
    scored.sort(key=lambda x: -x[0])
    return df.loc[[i for _, i in scored[:limit]]]


def search_by_name(query: str, limit: int = 5, active_only: bool = False) -> list[dict]:
    df = get_all_properties(active_only=True) if active_only else _frame()
    query_norm = (query or "").strip().lower()
    if not query_norm:
        return []

    names = df["PropertyName"].str.lower()

    exact = df[names == query_norm]
    if not exact.empty:
        return exact.head(limit).to_dict(orient="records")

    substring = df[names.str.contains(query_norm, na=False, regex=False)]
    if not substring.empty:
        return substring.head(limit).to_dict(orient="records")

    # Same name written with different punctuation/spacing ("Lake-View" vs "Lake View").
    canon_q = re.sub(r"[^a-z0-9]", "", query_norm)
    if canon_q:
        canon = df[names.str.replace(r"[^a-z0-9]", "", regex=True) == canon_q]
        if not canon.empty:
            return canon.head(limit).to_dict(orient="records")

    fuzzy = _fuzzy_name_matches(query, df, limit)
    return fuzzy.to_dict(orient="records") if not fuzzy.empty else []


def refresh_now():
    unit_service.refresh()
    return get_all_properties(force_refresh=True)

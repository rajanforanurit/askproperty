import calendar
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from cache import TTLCache
from config import settings
from db import get_engine

logger = logging.getLogger(__name__)

MONTHLY_COLUMNS = [
    "PropertyBizKey", "YearMonthKey", "UnitRows", "UnitDays",
    "OccupiedUnitDays", "SquareFeetDays", "MissingDaysRows",
]

PERIOD_COLUMNS = [
    "PropertyBizKey", "UnitDays", "CalendarDays", "Units", "OccupiedUnits",
    "OccupancyPct", "AvgUnitSqft", "TotalSqft", "UnitRows", "MonthsWithData", "MissingDaysRows",
]

_state = {"error": None, "source": None}


def _quote_part(part: str) -> str:
    return "[" + part.strip().strip("[]").replace("]", "]]") + "]"


def _quote_table(name: str) -> str:
    parts = [p for p in name.split(".") if p.strip()]
    return ".".join(_quote_part(p) for p in parts)


def validate_month(value, label: str = "year_month") -> int:
    try:
        month_key = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number in YYYYMM format.")
    year, month = divmod(month_key, 100)
    if not (1900 <= year <= 2200 and 1 <= month <= 12):
        raise ValueError(f"{label} must be a valid month in YYYYMM format.")
    return month_key


def days_in_month(year_month: int) -> int:
    year, month = divmod(int(year_month), 100)
    return calendar.monthrange(year, month)[1]


def shift_month(year_month: int, offset: int) -> int:
    year, month = divmod(int(year_month), 100)
    index = year * 12 + (month - 1) + offset
    new_year, new_month = divmod(index, 12)
    return new_year * 100 + new_month + 1


def month_range(ym_from: int, ym_to: int) -> list:
    ym_from = validate_month(ym_from, "year_month_from")
    ym_to = validate_month(ym_to, "year_month_to")
    if ym_from > ym_to:
        ym_from, ym_to = ym_to, ym_from
    months = []
    current = ym_from
    while current <= ym_to:
        months.append(current)
        current = shift_month(current, 1)
    return months


def calendar_days(ym_from: int, ym_to: int) -> int:
    return sum(days_in_month(m) for m in month_range(ym_from, ym_to))


def normalize_period(ym_from=None, ym_to=None) -> Optional[tuple]:
    if ym_from is None and ym_to is None:
        return None
    if ym_from is None:
        ym_from = ym_to
    if ym_to is None:
        ym_to = ym_from
    ym_from = validate_month(ym_from, "year_month_from")
    ym_to = validate_month(ym_to, "year_month_to")
    if ym_from > ym_to:
        ym_from, ym_to = ym_to, ym_from
    return ym_from, ym_to


def _empty_monthly() -> pd.DataFrame:
    return pd.DataFrame({
        "PropertyBizKey": pd.Series(dtype="object"),
        "YearMonthKey": pd.Series(dtype="int64"),
        "UnitRows": pd.Series(dtype="float64"),
        "UnitDays": pd.Series(dtype="float64"),
        "OccupiedUnitDays": pd.Series(dtype="float64"),
        "SquareFeetDays": pd.Series(dtype="float64"),
        "MissingDaysRows": pd.Series(dtype="float64"),
    })


def _empty_period() -> pd.DataFrame:
    frame = pd.DataFrame({column: pd.Series(dtype="float64") for column in PERIOD_COLUMNS})
    frame["PropertyBizKey"] = frame["PropertyBizKey"].astype("object")
    return frame


def _build_monthly_query() -> str:
    pk = _quote_part(settings.FACTUNIT_PROPERTY_COLUMN)
    uk = _quote_part(settings.FACTUNIT_UNIT_COLUMN)
    ym = _quote_part(settings.FACTUNIT_MONTH_COLUMN)
    days = _quote_part(settings.FACTUNIT_DAYS_COLUMN)
    occupied = (settings.FACTUNIT_OCCUPIED_COLUMN or "").strip()
    area = (settings.FACTUNIT_AREA_COLUMN or "").strip()
    agg = "SUM" if settings.UNIT_DUPLICATE_MODE == "sum" else "MAX"
    month_expr = f"TRY_CAST({ym} AS int)"

    occupied_expr = f"{agg}(TRY_CAST({_quote_part(occupied)} AS float))" if occupied else "CAST(NULL AS float)"
    area_expr = f"MAX(TRY_CAST({_quote_part(area)} AS float))" if area else "CAST(NULL AS float)"

    inner = (
        f"SELECT {pk} AS pk, {uk} AS uk, {month_expr} AS ym, "
        f"{agg}(TRY_CAST({days} AS float)) AS d, "
        f"{occupied_expr} AS occ, {area_expr} AS sq "
        f"FROM {_quote_table(settings.FACTUNIT_TABLE)} "
        f"WHERE {pk} IS NOT NULL AND {uk} IS NOT NULL "
        f"AND {month_expr} BETWEEN 190001 AND 220012 "
        f"AND {month_expr} % 100 BETWEEN 1 AND 12 "
        f"GROUP BY {pk}, {uk}, {month_expr}"
    )

    dim = (
        "SELECT CASE g.ym % 100 "
        "WHEN 2 THEN CASE WHEN (g.ym / 100) % 4 = 0 AND ((g.ym / 100) % 100 <> 0 OR (g.ym / 100) % 400 = 0) "
        "THEN 29 ELSE 28 END "
        "WHEN 4 THEN 30 WHEN 6 THEN 30 WHEN 9 THEN 30 WHEN 11 THEN 30 ELSE 31 END AS dim"
    )
    capped = (
        "SELECT CASE WHEN g.d IS NULL OR g.d < 0 THEN 0 "
        "WHEN g.d > m.dim THEN m.dim ELSE g.d END AS cd"
    )

    return (
        "SELECT g.pk AS PropertyBizKey, g.ym AS YearMonthKey, "
        "COUNT(*) AS UnitRows, "
        "SUM(c.cd) AS UnitDays, "
        "SUM(ISNULL(g.occ, 0) * c.cd) AS OccupiedUnitDays, "
        "SUM(ISNULL(g.sq, 0) * c.cd) AS SquareFeetDays, "
        "SUM(CASE WHEN g.d IS NULL THEN 1 ELSE 0 END) AS MissingDaysRows "
        f"FROM ({inner}) g "
        f"CROSS APPLY ({dim}) m "
        f"CROSS APPLY ({capped}) c "
        "GROUP BY g.pk, g.ym"
    )


def _load_monthly() -> pd.DataFrame:
    try:
        df = pd.read_sql(text(_build_monthly_query()), get_engine())
    except Exception as e:
        logger.warning("Unit fact table unavailable (%s): %s", settings.FACTUNIT_TABLE, e)
        _state["error"] = f"{type(e).__name__}: {e}"
        return _empty_monthly()

    df["PropertyBizKey"] = df["PropertyBizKey"].astype(str).str.strip()
    df["YearMonthKey"] = pd.to_numeric(df["YearMonthKey"], errors="coerce")
    df = df.dropna(subset=["YearMonthKey"])
    df["YearMonthKey"] = df["YearMonthKey"].astype("int64")
    for column in ["UnitRows", "UnitDays", "OccupiedUnitDays", "SquareFeetDays", "MissingDaysRows"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype("float64")
    df = (
        df.groupby(["PropertyBizKey", "YearMonthKey"], as_index=False)[
            ["UnitRows", "UnitDays", "OccupiedUnitDays", "SquareFeetDays", "MissingDaysRows"]
        ].sum()
    )
    _state["error"] = None
    logger.info("Loaded %d property-month unit records from %s", len(df), settings.FACTUNIT_TABLE)
    return df[MONTHLY_COLUMNS]


_monthly_cache = TTLCache(ttl_seconds=settings.CACHE_TTL_SECONDS, loader=_load_monthly, name="unit-months")


def _load_product_units() -> dict:
    column = (settings.UNITS_COLUMN or "").strip().strip("[]")
    if not column:
        return {}
    query = f"SELECT [PropertyBizKey], {_quote_part(column)} AS Units FROM {_quote_table(settings.PRODUCT_TABLE)}"
    try:
        df = pd.read_sql(text(query), get_engine())
    except Exception as e:
        logger.warning("Product-table unit counts unavailable (column '%s'): %s", column, e)
        return {}
    df["PropertyBizKey"] = df["PropertyBizKey"].astype(str).str.strip()
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce")
    df = df.dropna(subset=["Units"])
    df = df[df["Units"] > 0]
    return {row.PropertyBizKey: float(row.Units) for row in df.itertuples(index=False)}


_product_units_cache = TTLCache(ttl_seconds=settings.CACHE_TTL_SECONDS, loader=_load_product_units, name="product-units")


def get_monthly(force_refresh: bool = False) -> pd.DataFrame:
    return _monthly_cache.get(force_refresh=force_refresh)


def refresh() -> dict:
    monthly = get_monthly(force_refresh=True)
    _product_units_cache.get(force_refresh=True)
    return {"unit_month_rows": int(len(monthly)), "error": _state["error"]}


def latest_month() -> Optional[int]:
    monthly = get_monthly()
    if monthly.empty:
        return None
    return int(monthly["YearMonthKey"].max())


def resolve_period(ym_from=None, ym_to=None) -> Optional[tuple]:
    explicit = normalize_period(ym_from, ym_to)
    if explicit is not None:
        return explicit
    latest = latest_month()
    if latest is None:
        return None
    span = max(1, int(settings.UNIT_DEFAULT_PERIOD_MONTHS or 1))
    return shift_month(latest, -(span - 1)), latest


def aggregate_unit_rows(raw: pd.DataFrame, mode: str = "max") -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_monthly()
    df = raw.copy()
    df["PropertyBizKey"] = df["PropertyBizKey"].astype(str).str.strip()
    df["UnitBizKey"] = df["UnitBizKey"].astype(str).str.strip()
    df["YearMonthKey"] = pd.to_numeric(df["YearMonthKey"], errors="coerce")
    df = df.dropna(subset=["YearMonthKey"])
    df = df[(df["YearMonthKey"] >= 190001) & (df["YearMonthKey"] <= 220012)]
    df = df[(df["YearMonthKey"] % 100 >= 1) & (df["YearMonthKey"] % 100 <= 12)]
    if df.empty:
        return _empty_monthly()
    df["YearMonthKey"] = df["YearMonthKey"].astype("int64")
    for column in ["DaysCount", "OccupiedUnitCount", "UnitSquare"]:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    func = "sum" if mode == "sum" else "max"
    unit_month = df.groupby(["PropertyBizKey", "UnitBizKey", "YearMonthKey"], as_index=False).agg(
        d=("DaysCount", func),
        occ=("OccupiedUnitCount", func),
        sq=("UnitSquare", "max"),
    )
    dim = unit_month["YearMonthKey"].map(days_in_month).astype(float)
    missing = unit_month["d"].isna()
    capped = unit_month["d"].fillna(0.0).clip(lower=0.0)
    capped = np.minimum(capped, dim)
    unit_month["cd"] = capped
    unit_month["missing"] = missing.astype(float)
    unit_month["occ_days"] = unit_month["occ"].fillna(0.0) * capped
    unit_month["sq_days"] = unit_month["sq"].fillna(0.0) * capped
    unit_month["rows"] = 1.0

    monthly = unit_month.groupby(["PropertyBizKey", "YearMonthKey"], as_index=False).agg(
        UnitRows=("rows", "sum"),
        UnitDays=("cd", "sum"),
        OccupiedUnitDays=("occ_days", "sum"),
        SquareFeetDays=("sq_days", "sum"),
        MissingDaysRows=("missing", "sum"),
    )
    return monthly[MONTHLY_COLUMNS]


def compute_period_metrics(monthly: pd.DataFrame, ym_from: int, ym_to: int, keys=None) -> pd.DataFrame:
    ym_from = validate_month(ym_from, "year_month_from")
    ym_to = validate_month(ym_to, "year_month_to")
    if ym_from > ym_to:
        ym_from, ym_to = ym_to, ym_from
    total_days = calendar_days(ym_from, ym_to)
    if total_days <= 0 or monthly is None or monthly.empty:
        return _empty_period()

    window = monthly[(monthly["YearMonthKey"] >= ym_from) & (monthly["YearMonthKey"] <= ym_to)]
    if keys is not None:
        window = window[window["PropertyBizKey"].isin([str(k).strip() for k in keys])]
    if window.empty:
        return _empty_period()

    grouped = window.groupby("PropertyBizKey", as_index=False).agg(
        UnitDays=("UnitDays", "sum"),
        OccupiedUnitDays=("OccupiedUnitDays", "sum"),
        SquareFeetDays=("SquareFeetDays", "sum"),
        UnitRows=("UnitRows", "sum"),
        MonthsWithData=("YearMonthKey", "nunique"),
        MissingDaysRows=("MissingDaysRows", "sum"),
    )
    grouped["CalendarDays"] = float(total_days)
    grouped["Units"] = grouped["UnitDays"] / total_days
    grouped["OccupiedUnits"] = grouped["OccupiedUnitDays"] / total_days
    has_days = grouped["UnitDays"] > 0
    safe_days = grouped["UnitDays"].where(has_days, np.nan)
    grouped["OccupancyPct"] = np.minimum(grouped["OccupiedUnitDays"] / safe_days * 100.0, 100.0)
    grouped["AvgUnitSqft"] = grouped["SquareFeetDays"] / safe_days
    grouped["TotalSqft"] = grouped["SquareFeetDays"] / total_days
    return grouped[PERIOD_COLUMNS].reset_index(drop=True)


def _fallback_period_frame() -> pd.DataFrame:
    units = _product_units_cache.get()
    frame = _empty_period()
    if not units:
        return frame
    frame = pd.DataFrame({"PropertyBizKey": list(units.keys()), "Units": list(units.values())})
    for column in PERIOD_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[PERIOD_COLUMNS]


def describe_period(ym_from=None, ym_to=None) -> Optional[dict]:
    period = resolve_period(ym_from, ym_to)
    monthly = get_monthly()
    if period is None or monthly.empty:
        if _product_units_cache.get():
            return {"year_month_from": None, "year_month_to": None, "calendar_days": None, "source": "product_table"}
        return None
    return {
        "year_month_from": period[0],
        "year_month_to": period[1],
        "calendar_days": calendar_days(*period),
        "source": settings.FACTUNIT_TABLE,
    }


def get_period_metrics(ym_from=None, ym_to=None, keys=None) -> tuple:
    monthly = get_monthly()
    period = resolve_period(ym_from, ym_to)
    if period is None or monthly.empty:
        frame = _fallback_period_frame()
        if keys is not None:
            frame = frame[frame["PropertyBizKey"].isin([str(k).strip() for k in keys])]
        meta = {"year_month_from": None, "year_month_to": None, "calendar_days": None, "source": "product_table"}
        return frame.reset_index(drop=True), meta
    frame = compute_period_metrics(monthly, period[0], period[1], keys)
    meta = {
        "year_month_from": period[0],
        "year_month_to": period[1],
        "calendar_days": calendar_days(*period),
        "source": settings.FACTUNIT_TABLE,
    }
    return frame, meta


def get_units_map(ym_from=None, ym_to=None) -> dict:
    frame, _ = get_period_metrics(ym_from, ym_to)
    if frame.empty:
        return {}
    positive = frame[frame["Units"] > 0]
    return {str(row.PropertyBizKey): float(row.Units) for row in positive.itertuples(index=False)}


def get_unit_count(biz_key: str, ym_from=None, ym_to=None) -> Optional[float]:
    frame, _ = get_period_metrics(ym_from, ym_to, keys=[biz_key])
    if frame.empty:
        return None
    value = frame.iloc[0]["Units"]
    return None if pd.isna(value) else float(value)


def get_monthly_unit_counts(biz_key: str, ym_from=None, ym_to=None) -> list:
    monthly = get_monthly()
    period = resolve_period(ym_from, ym_to)
    if monthly.empty or period is None:
        return []
    window = monthly[
        (monthly["PropertyBizKey"] == str(biz_key).strip())
        & (monthly["YearMonthKey"] >= period[0])
        & (monthly["YearMonthKey"] <= period[1])
    ].set_index("YearMonthKey")
    rows = []
    for month in month_range(*period):
        dim = days_in_month(month)
        unit_days = float(window.loc[month, "UnitDays"]) if month in window.index else 0.0
        rows.append({
            "YearMonthKey": month,
            "DaysInMonth": dim,
            "UnitDays": unit_days,
            "UnitCount": unit_days / dim,
        })
    return rows


def diagnostics() -> dict:
    monthly = get_monthly()
    return {
        "table": settings.FACTUNIT_TABLE,
        "columns": {
            "property": settings.FACTUNIT_PROPERTY_COLUMN,
            "unit": settings.FACTUNIT_UNIT_COLUMN,
            "month": settings.FACTUNIT_MONTH_COLUMN,
            "days": settings.FACTUNIT_DAYS_COLUMN,
            "occupied": settings.FACTUNIT_OCCUPIED_COLUMN,
            "area": settings.FACTUNIT_AREA_COLUMN,
        },
        "duplicate_mode": settings.UNIT_DUPLICATE_MODE,
        "loaded_rows": int(len(monthly)),
        "properties": int(monthly["PropertyBizKey"].nunique()) if not monthly.empty else 0,
        "first_month": int(monthly["YearMonthKey"].min()) if not monthly.empty else None,
        "latest_month": int(monthly["YearMonthKey"].max()) if not monthly.empty else None,
        "rows_missing_days": int(monthly["MissingDaysRows"].sum()) if not monthly.empty else 0,
        "error": _state["error"],
        "fallback_product_units": len(_product_units_cache.get()) if monthly.empty else None,
    }

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd
from sqlalchemy import bindparam, text

import property_service
from cache import KeyedTTLCache, TTLCache
from config import settings
from db import get_engine, read_sql_retry, scalar_retry

logger = logging.getLogger(__name__)

VIEW_ALIASES = {
    "actual": "actual",
    "actuals": "actual",
    "budget": "budget",
    "budgets": "budget",
    "forecast": "budget",
    "forecasting": "budget",
    "both": "both",
    "variance": "both",
    "compare": "both",
    "actual_vs_budget": "both",
    "budget_vs_actual": "both",
}

PROPERTY_INFO_FIELDS = ["PropertyBizKey", "PropertyName", "City", "State", "Market", "AssetType", "YearBuilt"]

ACCOUNT_KEY_CANDIDATES = ["GLAccountKey", "AccountKey"]
ACCOUNT_NAME_CANDIDATES = [
    "GLAccountName", "AccountName", "GLAccountDescription", "AccountDescription", "Description", "Name",
]
ACCOUNT_CATEGORY_CANDIDATES = [
    "GL Account Level 2", "GLAccountCategory", "AccountCategory", "GLAccountType", "AccountType",
    "GLAccountGroup", "AccountGroup", "Category", "Type",
]
ACCOUNT_CODE_CANDIDATES = ["GLAccountCode", "AccountCode", "GLAccountNumber", "AccountNumber"]

# ---------------------------------------------------------------------------------------------
# The ONLY financial lines this service returns (operating expense report).
# Detail lines are matched by name against the GL account hierarchy (every "GL Account Level N"
# column, deepest first, then the account name). Subtotals and the grand total are always
# calculated here from the detail lines, never read from the database, so they cannot double count.
# ---------------------------------------------------------------------------------------------
EXPENSE_GROUPS = [
    {
        "key": "total_property_level_expenses",
        "label": "TOTAL PROPERTY LEVEL EXPENSES",
        "items": [
            ("administrative", "ADMINISTRATIVE"),
            ("advertising_promo", "ADVERTISING PROMO"),
            ("professional_fees", "PROFESSIONAL FEES"),
            ("repair_and_maint_expenses", "REPAIR AND MAINT EXPENSES"),
            ("payroll_expense", "PAYROLL EXPENSE"),
            ("utility_expense", "UTILITY EXPENSE"),
        ],
    },
    {
        "key": "total_noncontrollable_expenses",
        "label": "TOTAL NONCONTROLLABLE EXPENSES",
        "items": [
            ("insurance_expense", "INSURANCE EXPENSE"),
            ("re_tax", "RE TAX"),
        ],
    },
]
GRAND_TOTAL_KEY = "total_operating_expenses"
GRAND_TOTAL_LABEL = "TOTAL OPERATING EXPENSES"

UNASSIGNED_GROUP = "_unassigned_group"


def _norm_label(value) -> str:
    """'Repair & Maint. Expenses' -> 'REPAIR MAINT EXPENSES' (case/punctuation/'and' insensitive)."""
    if value is None:
        return ""
    cleaned = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
    return " ".join(w for w in cleaned.split() if w != "AND")


ITEM_LOOKUP = {
    _norm_label(label): (key, label, group["key"])
    for group in EXPENSE_GROUPS
    for key, label in group["items"]
}
GROUP_LOOKUP = {_norm_label(g["label"]): g["key"] for g in EXPENSE_GROUPS}
ALL_LINE_KEYS = [key for g in EXPENSE_GROUPS for key, _ in g["items"]]
LEVEL_COLUMN_RE = re.compile(r"^\s*GL Account Level\s*(\d+)\s*$", re.IGNORECASE)


def normalize_view(value) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not cleaned:
        return None
    if cleaned not in VIEW_ALIASES:
        raise ValueError("view must be one of: actual, budget, both.")
    return VIEW_ALIASES[cleaned]


def _quote_part(part: str) -> str:
    return "[" + part.replace("]", "]]") + "]"


def _quote_table(name: str) -> str:
    parts = [p.strip().strip("[]") for p in name.split(".") if p.strip()]
    return ".".join(_quote_part(p) for p in parts)


def _quote_column(name: str) -> str:
    return _quote_part(name.strip().strip("[]"))


def _source(kind: str):
    if kind == "actual":
        return settings.FACTGL_TABLE, settings.ACTUAL_VALUE_COLUMN, settings.ACTUAL_BOOK_KEYS
    return settings.BUDGET_TABLE, settings.BUDGET_VALUE_COLUMN, settings.BUDGET_BOOK_KEYS


def _clean(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _text(value) -> Optional[str]:
    value = _clean(value)
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _num(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return round(number, 2)


def _records(df: pd.DataFrame) -> list:
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


def _find_column(df, override, candidates, contains, exclude, taken):
    lookup = {str(c).lower(): c for c in df.columns}
    if override and override.strip().lower() in lookup:
        return lookup[override.strip().lower()]
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None and found not in taken:
            return found
    for column in df.columns:
        low = str(column).lower()
        if column in taken:
            continue
        if any(token in low for token in contains) and not any(token in low for token in exclude):
            return column
    return None


def _match_line(level_values: list, name):
    """Return (line_key, path) for one GL account.

    level_values is ordered shallow -> deep. The deepest level that names one of the detail lines wins,
    so a parent bucket can never be counted together with its own children. If the account sits under a
    TOTAL group name but matches no detail line, it is flagged so it can be reported instead of silently
    inflating a total.
    """
    for value in reversed(level_values):
        hit = ITEM_LOOKUP.get(_norm_label(value))
        if hit:
            return hit[0]
    hit = ITEM_LOOKUP.get(_norm_label(name))
    if hit:
        return hit[0]
    for value in level_values:
        if _norm_label(value) in GROUP_LOOKUP:
            return UNASSIGNED_GROUP
    return None


def _load_gl_accounts() -> dict:
    state = {"loaded": False, "lookup": {}, "columns": {}, "all_columns": [], "row_count": 0, "error": None}
    try:
        df = pd.read_sql(text(f"SELECT * FROM {_quote_table(settings.GL_ACCOUNT_TABLE)}"), get_engine())
    except Exception as e:
        logger.warning("GL account lookup unavailable: %s", e)
        state["error"] = f"{type(e).__name__}: {e}"
        return state

    state["all_columns"] = [str(c) for c in df.columns]
    state["row_count"] = len(df)

    key_col = _find_column(df, settings.GL_ACCOUNT_KEY_COLUMN, ACCOUNT_KEY_CANDIDATES, [], [], set())
    if key_col is None:
        state["error"] = "Could not find the GL account key column. Set GL_ACCOUNT_KEY_COLUMN."
        return state

    taken = {key_col}
    name_col = _find_column(df, settings.GL_ACCOUNT_NAME_COLUMN, ACCOUNT_NAME_CANDIDATES,
                            ["name", "description"], ["key"], taken)
    if name_col is not None:
        taken.add(name_col)
    code_col = _find_column(df, settings.GL_ACCOUNT_CODE_COLUMN, ACCOUNT_CODE_CANDIDATES,
                            ["code", "number"], ["key", "book"], taken)

    level_cols = sorted(
        ((int(m.group(1)), c) for c in df.columns if (m := LEVEL_COLUMN_RE.match(str(c)))),
        key=lambda pair: pair[0],
    )
    level_cols = [c for _, c in level_cols]
    if not level_cols:
        state["error"] = "No 'GL Account Level N' columns found in the GL account table."
        return state

    work = df.copy()
    work["_account_key"] = pd.to_numeric(work[key_col], errors="coerce")
    work = work.dropna(subset=["_account_key"]).drop_duplicates(subset=["_account_key"])

    only_under = _norm_label(settings.OPEX_ROOT_FILTER) if settings.OPEX_ROOT_FILTER else ""

    lookup = {}
    line_counts = {}
    line_paths = {}
    for record in work.to_dict(orient="records"):
        account_key = int(record["_account_key"])
        levels = [_text(record.get(c)) for c in level_cols]
        name = _text(record.get(name_col)) if name_col is not None else None
        line = _match_line(levels, name)
        if line and only_under:
            if not any(only_under in _norm_label(v) for v in levels if v):
                line = None
        if line is None:
            continue
        lookup[account_key] = {
            "GLAccountCode": _text(record.get(code_col)) if code_col is not None else None,
            "GLAccountName": name or f"Account {account_key}",
            "_line": line,
        }
        line_counts[line] = line_counts.get(line, 0) + 1
        paths = line_paths.setdefault(line, set())
        if len(paths) < 6:
            paths.add(" > ".join(v for v in levels if v))

    state["loaded"] = True
    state["lookup"] = lookup
    state["columns"] = {
        "key": str(key_col),
        "name": str(name_col) if name_col is not None else None,
        "code": str(code_col) if code_col is not None else None,
        "levels": [str(c) for c in level_cols],
    }
    state["line_counts"] = line_counts
    state["line_paths"] = {k: sorted(v) for k, v in line_paths.items()}
    return state


_gl_accounts_cache = TTLCache(
    ttl_seconds=settings.CACHE_TTL_SECONDS, loader=_load_gl_accounts, name="gl_accounts"
)

_data_cache = KeyedTTLCache(
    ttl_seconds=settings.FINANCIAL_CACHE_TTL_SECONDS,
    max_entries=settings.FINANCIAL_CACHE_MAX_ENTRIES,
    name="financials",
)


def _gl_state() -> dict:
    state = _gl_accounts_cache.get()
    if not state["loaded"]:
        _gl_accounts_cache.invalidate()
    return state


def warm():
    _gl_accounts_cache.get(force_refresh=True)
    for kind in ("actual", "budget"):
        try:
            _max_month(kind)
        except Exception as e:
            logger.warning("Latest-month warm-up failed for %s: %s", kind, e)


def refresh_now() -> dict:
    _data_cache.clear()
    state = _gl_accounts_cache.get(force_refresh=True)
    return {"gl_accounts_loaded": state["loaded"], "gl_account_count": len(state["lookup"])}


def _clean_keys(biz_keys) -> list:
    seen = set()
    result = []
    for key in biz_keys or []:
        cleaned = str(key).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _property_lookup(keys: list) -> dict:
    df = property_service.get_all_properties()
    subset = df[df["PropertyBizKey"].isin(keys)]
    units = property_service.get_units_map()
    result = {}
    for record in subset.to_dict(orient="records"):
        info = {f: _clean(record.get(f)) for f in PROPERTY_INFO_FIELDS}
        info["Units"] = units.get(record["PropertyBizKey"])
        result[record["PropertyBizKey"]] = info
    return result


def _validate_month(value, label: str) -> int:
    try:
        month_key = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number in YYYYMM format.")
    year, month = divmod(month_key, 100)
    if not (1900 <= year <= 2200 and 1 <= month <= 12):
        raise ValueError(f"{label} must be a valid month in YYYYMM format.")
    return month_key


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "PropertyBizKey": pd.Series(dtype="object"),
        "GLAccountKey": pd.Series(dtype="int64"),
        "YearMonthKey": pd.Series(dtype="int64"),
        "GLBookKey": pd.Series(dtype="int64"),
        "Amount": pd.Series(dtype="float64"),
    })


def _fetch_amounts(kind: str, keys: list, ym_from: int, ym_to: int) -> pd.DataFrame:
    table, value_col, books = _source(kind)
    pk = _quote_column(settings.FIN_PROPERTY_KEY_COLUMN)

    sql = (
        f"SELECT {pk} AS PropertyBizKey, GLAccountKey, YearMonthKey, GLBookKey, "
        f"SUM({_quote_column(value_col)}) AS Amount "
        f"FROM {_quote_table(table)} "
        f"WHERE {pk} IN :keys AND YearMonthKey BETWEEN :ym_from AND :ym_to"
    )
    params = {"keys": list(keys), "ym_from": ym_from, "ym_to": ym_to}
    expanding = [bindparam("keys", expanding=True)]

    if books:
        sql += " AND GLBookKey IN :books"
        params["books"] = list(books)
        expanding.append(bindparam("books", expanding=True))

    sql += f" GROUP BY {pk}, GLAccountKey, YearMonthKey, GLBookKey"

    logger.info("Fetching %s amounts from %s for %d properties (%s-%s)",
                kind, table, len(keys), ym_from, ym_to)
    df = read_sql_retry(text(sql).bindparams(*expanding), params)

    if df.empty:
        return _empty_frame()

    df["PropertyBizKey"] = df["PropertyBizKey"].astype(str).str.strip()
    df["GLAccountKey"] = pd.to_numeric(df["GLAccountKey"], errors="coerce")
    df["YearMonthKey"] = pd.to_numeric(df["YearMonthKey"], errors="coerce")
    df["GLBookKey"] = pd.to_numeric(df["GLBookKey"], errors="coerce").fillna(-1)
    df = df.dropna(subset=["GLAccountKey", "YearMonthKey"])
    df["GLAccountKey"] = df["GLAccountKey"].astype("int64")
    df["YearMonthKey"] = df["YearMonthKey"].astype("int64")
    df["GLBookKey"] = df["GLBookKey"].astype("int64")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0).astype(float)
    return df.reset_index(drop=True)


def _cached_amounts(kind: str, keys: list, ym_from: int, ym_to: int) -> pd.DataFrame:
    cache_key = ("amounts", kind, tuple(sorted(keys)), ym_from, ym_to)
    return _data_cache.get(cache_key, lambda: _fetch_amounts(kind, keys, ym_from, ym_to))


class _NoMonth(Exception):
    pass


def _max_month(kind: str, keys: Optional[list] = None) -> Optional[int]:
    """Latest YearMonthKey in the actual/budget table (cached; warmed at startup)."""
    table, _, books = _source(kind)

    sql = f"SELECT MAX(YearMonthKey) FROM {_quote_table(table)}"
    params = {}
    statement = None
    if books:
        sql += " WHERE GLBookKey IN :books"
        params["books"] = list(books)
        statement = text(sql).bindparams(bindparam("books", expanding=True))
    else:
        statement = text(sql)

    def loader():
        value = scalar_retry(statement, params)
        if value is None:
            raise _NoMonth()
        return int(value)

    try:
        return _data_cache.get(("max_month", kind), loader)
    except _NoMonth:
        return None


def _latest_month(keys: list, view: Optional[str] = None) -> Optional[int]:
    order = ["budget", "actual"] if view == "budget" else ["actual", "budget"]
    for kind in order:
        latest = _max_month(kind, keys)
        if latest is not None:
            return latest
    return None


def _resolve_period(keys: list, ym_from, ym_to, view: Optional[str] = None):
    ym_from = _validate_month(ym_from, "year_month_from") if ym_from is not None else None
    ym_to = _validate_month(ym_to, "year_month_to") if ym_to is not None else None

    if ym_to is None:
        ym_to = _latest_month(keys, view)
        if ym_to is None:
            return None
    if ym_from is None:
        ym_from = (ym_to // 100) * 100 + 1
    if ym_from > ym_to:
        raise ValueError("year_month_from must not be later than year_month_to.")
    return ym_from, ym_to


def _apply_account_scope(frames: dict):
    """Keep only accounts that belong to one of the operating expense lines and tag them with that line.

    Expenses are stored as negative dollars in the GL; OPEX_SIGN_MULTIPLIER (default -1) turns them into
    the positive figures shown in the operating expense report.
    """
    excluded = {}
    notes = []
    state = _gl_state()
    if not state["loaded"]:
        notes.append("GL account table is unavailable, so operating expense lines could not be identified.")
        return {k: _empty_frame_with_line() for k in frames}, excluded, notes

    line_of = {key: info["_line"] for key, info in state["lookup"].items()}
    sign = settings.OPEX_SIGN_MULTIPLIER
    scoped = {}
    unassigned_amount = 0.0
    for kind, frame in frames.items():
        if frame.empty:
            scoped[kind] = _empty_frame_with_line()
            excluded[kind] = {"accounts": 0, "amount": 0.0}
            continue
        tagged = frame.copy()
        tagged["Line"] = tagged["GLAccountKey"].map(line_of)
        outside = tagged[tagged["Line"].isna()]
        excluded[kind] = {
            "accounts": int(outside["GLAccountKey"].nunique()),
            "amount": round(float(outside["Amount"].sum()), 2),
        }
        stray = tagged[tagged["Line"] == UNASSIGNED_GROUP]
        if not stray.empty:
            unassigned_amount += float(stray["Amount"].sum()) * sign
            notes.append(
                f"{kind.capitalize()}: {stray['GLAccountKey'].nunique()} account(s) sit under a TOTAL expense group "
                f"but match none of its detail lines ({round(float(stray['Amount'].sum()) * sign, 2)}); excluded."
            )
        kept = tagged[tagged["Line"].isin(ALL_LINE_KEYS)].copy()
        kept["Amount"] = kept["Amount"] * sign
        scoped[kind] = kept.reset_index(drop=True)

    counts = state.get("line_counts", {})
    missing = [label for g in EXPENSE_GROUPS for key, label in g["items"] if not counts.get(key)]
    if missing:
        notes.append("No GL accounts matched these lines (check the GL hierarchy names): " + ", ".join(missing))
    return scoped, excluded, notes


def _empty_frame_with_line() -> pd.DataFrame:
    frame = _empty_frame()
    frame["Line"] = pd.Series(dtype="object")
    return frame


def _load(view: str, keys: list, ym_from, ym_to):
    kinds = ["actual", "budget"] if view == "both" else [view]
    period = _resolve_period(keys, ym_from, ym_to, view)
    if period is None:
        return None, {kind: _empty_frame() for kind in kinds}, {}, []
    if len(kinds) > 1:
        with ThreadPoolExecutor(max_workers=len(kinds)) as pool:
            futures = {kind: pool.submit(_cached_amounts, kind, keys, period[0], period[1]) for kind in kinds}
            frames = {kind: futures[kind].result() for kind in kinds}
    else:
        frames = {kinds[0]: _cached_amounts(kinds[0], keys, period[0], period[1])}
    frames, excluded, notes = _apply_account_scope(frames)
    return {"year_month_from": period[0], "year_month_to": period[1]}, frames, excluded, notes


def _books_and_warnings(frames: dict):
    books = {}
    warnings = []
    for kind, frame in frames.items():
        included = sorted(int(b) for b in frame["GLBookKey"].unique()) if not frame.empty else []
        books[kind] = included
        configured = settings.ACTUAL_BOOK_KEYS if kind == "actual" else settings.BUDGET_BOOK_KEYS
        if not configured and len(included) > 1:
            warnings.append(
                f"{kind.capitalize()} data spans GL books {included}. "
                f"Set {kind.upper()}_BOOK_KEYS to avoid mixing books."
            )
    return books, warnings


def _merge_amounts(frames: dict, group_cols: list) -> pd.DataFrame:
    merged = None
    for kind, frame in frames.items():
        grouped = frame.groupby(group_cols, as_index=False)["Amount"].sum().rename(columns={"Amount": kind})
        merged = grouped if merged is None else merged.merge(grouped, on=group_cols, how="outer")
    for kind in frames:
        merged[kind] = merged[kind].fillna(0.0)
    return merged


def _amounts(actual, budget, view: str) -> dict:
    if view == "actual":
        value = _num(actual)
        return {"actual": value, "value": value}
    if view == "budget":
        value = _num(budget)
        return {"budget": value, "value": value}
    actual_value = _num(actual)
    budget_value = _num(budget)
    variance = round(actual_value - budget_value, 2)
    variance_pct = round(variance / abs(budget_value) * 100, 2) if budget_value else None
    return {
        "actual": actual_value,
        "budget": budget_value,
        "variance": variance,
        "variance_pct": variance_pct,
    }


def _row_amounts(record: dict, view: str) -> dict:
    return _amounts(record.get("actual", 0.0), record.get("budget", 0.0), view)


def _per_unit(amount: float, units) -> Optional[float]:
    try:
        count = float(units)
    except (TypeError, ValueError):
        return None
    if not count:
        return None
    return round(amount / count, 2)


def _with_units(base: dict, view: str, units) -> dict:
    """Add the report's '$ Per Unit' figure next to each amount."""
    result = dict(base)
    result["units"] = units
    if view == "both":
        result["actual_per_unit"] = _per_unit(base["actual"], units)
        result["budget_per_unit"] = _per_unit(base["budget"], units)
    else:
        result["per_unit"] = _per_unit(base["value"], units)
    return result


def _line_pairs(frames: dict, by_property: bool) -> dict:
    """{owner: {line_key: (actual, budget)}} where owner is a PropertyBizKey, or '' when not by_property."""
    group_cols = ["PropertyBizKey", "Line"] if by_property else ["Line"]
    table = _merge_amounts(frames, group_cols)
    pairs = {}
    for record in table.to_dict(orient="records"):
        owner = record["PropertyBizKey"] if by_property else ""
        pairs.setdefault(owner, {})[record["Line"]] = (record.get("actual", 0.0), record.get("budget", 0.0))
    return pairs


def _add(a: tuple, b: tuple) -> tuple:
    return (a[0] + b[0], a[1] + b[1])


def _report_values(line_pairs: dict) -> list:
    """Ordered report rows [(key, label, level, parent, (actual, budget))] with calculated totals."""
    rows = []
    grand = (0.0, 0.0)
    for group in EXPENSE_GROUPS:
        children = []
        subtotal = (0.0, 0.0)
        for key, label in group["items"]:
            value = line_pairs.get(key, (0.0, 0.0))
            subtotal = _add(subtotal, value)
            children.append((key, label, "item", group["key"], value))
        rows.append((group["key"], group["label"], "total", None, subtotal))
        rows.extend(children)
        grand = _add(grand, subtotal)
    rows.append((GRAND_TOTAL_KEY, GRAND_TOTAL_LABEL, "grand_total", None, grand))
    return rows


def _line_items_single(frames: dict, view: str, units) -> list:
    values = _report_values(_line_pairs(frames, False).get("", {}))
    return [
        {"key": key, "label": label, "level": level, "parent": parent,
         **_with_units(_amounts(pair[0], pair[1], view), view, units)}
        for key, label, level, parent, pair in values
    ]


def _line_items_matrix(frames: dict, view: str, keys: list, units_by_key: dict) -> list:
    pairs = _line_pairs(frames, True)
    per_owner = {owner: _report_values(pairs.get(owner, {})) for owner in keys}
    rows = []
    for index, (key, label, level, parent, _) in enumerate(per_owner[keys[0]]):
        rows.append({
            "key": key,
            "label": label,
            "level": level,
            "parent": parent,
            "values": {
                owner: _with_units(
                    _amounts(per_owner[owner][index][4][0], per_owner[owner][index][4][1], view),
                    view, units_by_key.get(owner),
                )
                for owner in keys
            },
        })
    return rows


def _frame_total(frames: dict, kind: str) -> float:
    frame = frames.get(kind)
    if frame is None or frame.empty:
        return 0.0
    return float(frame["Amount"].sum())


def get_property_financials(biz_key, view=None, ym_from=None, ym_to=None, top_accounts=None) -> dict:
    """Operating expense report for one property. top_accounts is accepted for API compatibility and ignored."""
    view = normalize_view(view) or "actual"
    key = str(biz_key).strip()
    props = _property_lookup([key])
    if key not in props:
        raise ValueError(f"Property '{key}' not found.")
    units = props[key].get("Units")

    period, frames, excluded, notes = _load(view, [key], ym_from, ym_to)
    books, warnings = _books_and_warnings(frames)
    warnings = warnings + notes
    has_data = any(not frame.empty for frame in frames.values())

    totals = _with_units(
        _amounts(_frame_total(frames, "actual"), _frame_total(frames, "budget"), view), view, units
    )

    months = []
    if has_data:
        month_df = _merge_amounts(frames, ["YearMonthKey"]).sort_values("YearMonthKey")
        for record in month_df.to_dict(orient="records"):
            months.append({"YearMonthKey": int(record["YearMonthKey"]), **_row_amounts(record, view)})

    return {
        "view": view,
        "property": props[key],
        "period": period,
        "books": books,
        "has_data": has_data,
        "excluded_other_accounts": excluded,
        "totals": totals,
        "line_items": _line_items_single(frames, view, units),
        "months": months,
        "warnings": warnings,
    }


def compare_financials(biz_keys, view=None, ym_from=None, ym_to=None, top_accounts=None) -> dict:
    """Operating expense report side by side for several properties."""
    view = normalize_view(view) or "actual"
    keys = _clean_keys(biz_keys)
    if not keys:
        raise ValueError("At least one PropertyBizKey is required.")

    props = _property_lookup(keys)
    found = [k for k in keys if k in props]
    not_found = [k for k in keys if k not in props]

    if not found:
        return {
            "view": view,
            "period": None,
            "books": {},
            "properties": [],
            "line_items": [],
            "not_found": not_found,
            "no_data": [],
            "warnings": [],
        }

    period, frames, excluded, notes = _load(view, found, ym_from, ym_to)
    books, warnings = _books_and_warnings(frames)
    warnings = warnings + notes
    has_data = any(not frame.empty for frame in frames.values())
    units_by_key = {k: props[k].get("Units") for k in found}

    totals_map = {}
    if has_data:
        totals_df = _merge_amounts(frames, ["PropertyBizKey"])
        totals_map = {rec["PropertyBizKey"]: rec for rec in totals_df.to_dict(orient="records")}

    property_rows = []
    no_data = []
    for key in found:
        record = totals_map.get(key)
        if record is None:
            no_data.append(key)
        property_rows.append({
            **props[key],
            "has_data": record is not None,
            **_with_units(_row_amounts(record or {}, view), view, units_by_key[key]),
        })

    return {
        "view": view,
        "period": period,
        "books": books,
        "excluded_other_accounts": excluded,
        "properties": property_rows,
        "line_items": _line_items_matrix(frames, view, found, units_by_key),
        "not_found": not_found,
        "no_data": no_data,
        "warnings": warnings,
    }


def _query_rows(sql: str, params: Optional[dict] = None) -> list:
    with get_engine().connect() as conn:
        result = conn.execute(text(sql), params or {})
        return [dict(row) for row in result.mappings().all()]


def _error_text(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:300]}"


def diagnostics() -> dict:
    pk = _quote_column(settings.FIN_PROPERTY_KEY_COLUMN)
    output = {
        "settings": {
            "factgl_table": settings.FACTGL_TABLE,
            "budget_table": settings.BUDGET_TABLE,
            "gl_account_table": settings.GL_ACCOUNT_TABLE,
            "gl_book_table": settings.GL_BOOK_TABLE,
            "property_key_column": settings.FIN_PROPERTY_KEY_COLUMN,
            "actual_value_column": settings.ACTUAL_VALUE_COLUMN,
            "budget_value_column": settings.BUDGET_VALUE_COLUMN,
            "actual_book_keys": settings.ACTUAL_BOOK_KEYS,
            "budget_book_keys": settings.BUDGET_BOOK_KEYS,
            "opex_sign_multiplier": settings.OPEX_SIGN_MULTIPLIER,
            "opex_root_filter": settings.OPEX_ROOT_FILTER,
        },
        "tables": {},
        "gl_accounts": {},
        "gl_books": {},
    }

    try:
        product_keys = set(property_service.get_all_properties()["PropertyBizKey"])
    except Exception as e:
        product_keys = set()
        output["product_error"] = _error_text(e)

    for kind in ("actual", "budget"):
        table, value_col, _ = _source(kind)
        quoted = _quote_table(table)
        section = {"table": table, "value_column": value_col}
        try:
            stats = _query_rows(
                f"SELECT COUNT(*) AS row_count, MIN(YearMonthKey) AS min_year_month, "
                f"MAX(YearMonthKey) AS max_year_month, COUNT(DISTINCT {pk}) AS property_count FROM {quoted}"
            )[0]
            section.update(stats)
            section["books"] = _query_rows(
                f"SELECT GLBookKey, COUNT(*) AS row_count FROM {quoted} GROUP BY GLBookKey ORDER BY GLBookKey"
            )
            table_keys = {
                str(r["k"]).strip()
                for r in _query_rows(f"SELECT DISTINCT {pk} AS k FROM {quoted}")
                if r["k"] is not None
            }
            section["properties_matching_product_table"] = len(table_keys & product_keys)
            section["unmatched_key_sample"] = sorted(table_keys - product_keys)[:10]
        except Exception as e:
            section["error"] = _error_text(e)
        output["tables"][kind] = section

    state = _gl_accounts_cache.get(force_refresh=True)
    output["gl_accounts"] = {
        "loaded": state["loaded"],
        "row_count": state["row_count"],
        "all_columns": state["all_columns"],
        "detected_columns": state["columns"],
        "accounts_per_line": {
            label: state.get("line_counts", {}).get(key, 0)
            for g in EXPENSE_GROUPS for key, label in g["items"]
        },
        "unassigned_group_accounts": state.get("line_counts", {}).get(UNASSIGNED_GROUP, 0),
        "hierarchy_paths_per_line": {
            label: state.get("line_paths", {}).get(key, [])
            for g in EXPENSE_GROUPS for key, label in g["items"]
        },
        "error": state["error"],
    }

    try:
        book_df = pd.read_sql(text(f"SELECT * FROM {_quote_table(settings.GL_BOOK_TABLE)}"), get_engine())
        output["gl_books"] = {"row_count": len(book_df), "rows": _records(book_df.head(50))}
    except Exception as e:
        output["gl_books"] = {"error": _error_text(e)}

    return output

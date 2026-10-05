import logging
from typing import Optional

import pandas as pd
from sqlalchemy import bindparam, text

import property_service
from cache import KeyedTTLCache, TTLCache
from config import settings
from db import get_engine

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

SECTION_LABELS = {
    "revenue": "Revenue",
    "operating_expenses": "Operating Expenses",
    "capital_expenses": "Capital Expenses",
    "replacements": "Replacements",
    "other": "Other",
}

SUMMARY_LINES = [
    ("revenue", "Revenue"),
    ("operating_expenses", "Operating Expenses"),
    ("noi", "Net Operating Income"),
    ("capital_expenses", "Capital Expenses"),
    ("replacements", "Replacements"),
    ("net_cash_flow", "Net Cash Flow After Capital"),
]


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


def _classify_section(level2, level3) -> str:
    l2 = (level2 or "").lower()
    l3 = (level3 or "").lower()
    if "capital expense" in l2:
        return "capital_expenses"
    if "replacement" in l2:
        return "replacements"
    if "net operating income" in l2:
        if "revenue" in l3:
            return "revenue"
        if "operating expense" in l3:
            return "operating_expenses"
    return "other"


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
    category_col = _find_column(df, settings.GL_ACCOUNT_CATEGORY_COLUMN, ACCOUNT_CATEGORY_CANDIDATES,
                                ["category", "type", "group", "class"], ["key"], taken)
    if category_col is not None:
        taken.add(category_col)
    code_col = _find_column(df, settings.GL_ACCOUNT_CODE_COLUMN, ACCOUNT_CODE_CANDIDATES,
                            ["code", "number"], ["key", "book"], taken)
    level2_col = _find_column(df, settings.GL_ACCOUNT_LEVEL2_COLUMN, ["GL Account Level 2"], [], [], set())
    level3_col = _find_column(df, settings.GL_ACCOUNT_LEVEL3_COLUMN, ["GL Account Level 3"], [], [], set())

    work = df.copy()
    work["_account_key"] = pd.to_numeric(work[key_col], errors="coerce")
    work = work.dropna(subset=["_account_key"]).drop_duplicates(subset=["_account_key"])

    lookup = {}
    section_counts = {}
    for record in work.to_dict(orient="records"):
        account_key = int(record["_account_key"])
        name = _text(record.get(name_col)) if name_col is not None else None
        level2 = _text(record.get(level2_col)) if level2_col is not None else None
        level3 = _text(record.get(level3_col)) if level3_col is not None else None
        section = _classify_section(level2, level3)
        section_counts[section] = section_counts.get(section, 0) + 1
        lookup[account_key] = {
            "GLAccountCode": _text(record.get(code_col)) if code_col is not None else None,
            "GLAccountName": name or f"Account {account_key}",
            "GLAccountCategory": _text(record.get(category_col)) if category_col is not None else None,
            "GLAccountSection": SECTION_LABELS[section],
            "_section": section,
        }

    state["loaded"] = True
    state["lookup"] = lookup
    state["columns"] = {
        "key": str(key_col),
        "name": str(name_col) if name_col is not None else None,
        "category": str(category_col) if category_col is not None else None,
        "code": str(code_col) if code_col is not None else None,
        "level2": str(level2_col) if level2_col is not None else None,
        "level3": str(level3_col) if level3_col is not None else None,
    }
    state["section_counts"] = section_counts
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


def _account_info(account_key) -> dict:
    account_key = int(account_key)
    info = _gl_state()["lookup"].get(account_key)
    if info is None:
        return {
            "GLAccountCode": None,
            "GLAccountName": f"Account {account_key}",
            "GLAccountCategory": None,
            "GLAccountSection": SECTION_LABELS["other"],
        }
    return {k: v for k, v in info.items() if not k.startswith("_")}


def _section_key(account_key) -> str:
    info = _gl_state()["lookup"].get(int(account_key))
    if info is None:
        return "other"
    return info["_section"]


def warm():
    _gl_accounts_cache.get(force_refresh=True)


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
    df = pd.read_sql(text(sql).bindparams(*expanding), get_engine(), params=params)

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


def _max_month(kind: str, keys: list) -> Optional[int]:
    table, _, books = _source(kind)
    pk = _quote_column(settings.FIN_PROPERTY_KEY_COLUMN)

    sql = f"SELECT MAX(YearMonthKey) FROM {_quote_table(table)} WHERE {pk} IN :keys"
    params = {"keys": list(keys)}
    expanding = [bindparam("keys", expanding=True)]

    if books:
        sql += " AND GLBookKey IN :books"
        params["books"] = list(books)
        expanding.append(bindparam("books", expanding=True))

    def loader():
        with get_engine().connect() as conn:
            value = conn.execute(text(sql).bindparams(*expanding), params).scalar()
        return int(value) if value is not None else None

    return _data_cache.get(("max_month", kind, tuple(sorted(keys))), loader)


def _latest_month(keys: list) -> Optional[int]:
    latest = _max_month("actual", keys)
    if latest is None:
        latest = _max_month("budget", keys)
    return latest


def _resolve_period(keys: list, ym_from, ym_to):
    ym_from = _validate_month(ym_from, "year_month_from") if ym_from is not None else None
    ym_to = _validate_month(ym_to, "year_month_to") if ym_to is not None else None

    if ym_to is None:
        ym_to = _latest_month(keys)
        if ym_to is None:
            return None
    if ym_from is None:
        ym_from = (ym_to // 100) * 100 + 1
    if ym_from > ym_to:
        raise ValueError("year_month_from must not be later than year_month_to.")
    return ym_from, ym_to


def _apply_account_scope(frames: dict):
    excluded = {}
    notes = []
    if settings.INCLUDE_UNMAPPED_ACCOUNTS:
        return frames, excluded, notes

    state = _gl_state()
    if not state["loaded"]:
        notes.append("GL account table is unavailable, so accounts outside the cash flow hierarchy could not be excluded.")
        return frames, excluded, notes

    known = list(state["lookup"].keys())
    scoped = {}
    for kind, frame in frames.items():
        if frame.empty:
            scoped[kind] = frame
            excluded[kind] = {"accounts": 0, "amount": 0.0}
            continue
        mask = frame["GLAccountKey"].isin(known)
        dropped = frame.loc[~mask]
        excluded[kind] = {
            "accounts": int(dropped["GLAccountKey"].nunique()),
            "amount": round(float(dropped["Amount"].sum()), 2),
        }
        scoped[kind] = frame.loc[mask].reset_index(drop=True)
    return scoped, excluded, notes


def _load(view: str, keys: list, ym_from, ym_to):
    kinds = ["actual", "budget"] if view == "both" else [view]
    period = _resolve_period(keys, ym_from, ym_to)
    if period is None:
        return None, {kind: _empty_frame() for kind in kinds}, {}, []
    frames = {kind: _cached_amounts(kind, keys, period[0], period[1]) for kind in kinds}
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


def _add_sort_column(df: pd.DataFrame, view: str) -> pd.DataFrame:
    if view == "actual":
        df["_sort"] = df["actual"].abs()
    elif view == "budget":
        df["_sort"] = df["budget"].abs()
    else:
        df["_sort"] = df[["actual", "budget"]].abs().max(axis=1)
    return df


def _ordered_lines(include_other: bool) -> list:
    lines = list(SUMMARY_LINES)
    if include_other:
        lines.insert(len(lines) - 1, ("other", "Other / Unmapped"))
    return lines


def _section_pairs(frames: dict, by_property: bool) -> dict:
    group_cols = ["PropertyBizKey", "Section"] if by_property else ["Section"]
    tagged = {}
    for kind, frame in frames.items():
        copy = frame.copy()
        copy["Section"] = copy["GLAccountKey"].map(_section_key).astype(object)
        tagged[kind] = copy
    table = _merge_amounts(tagged, group_cols)
    pairs = {}
    for record in table.to_dict(orient="records"):
        owner = record["PropertyBizKey"] if by_property else ""
        pairs.setdefault(owner, {})[record["Section"]] = (record.get("actual", 0.0), record.get("budget", 0.0))
    return pairs


def _line_values(section_pairs: dict) -> dict:
    def pick(key):
        return section_pairs.get(key, (0.0, 0.0))

    revenue = pick("revenue")
    opex = pick("operating_expenses")
    return {
        "revenue": revenue,
        "operating_expenses": opex,
        "noi": (revenue[0] + opex[0], revenue[1] + opex[1]),
        "capital_expenses": pick("capital_expenses"),
        "replacements": pick("replacements"),
        "other": pick("other"),
        "net_cash_flow": (
            sum(p[0] for p in section_pairs.values()),
            sum(p[1] for p in section_pairs.values()),
        ),
    }


def _summary_single(frames: dict, view: str) -> list:
    values = _line_values(_section_pairs(frames, False).get("", {}))
    include_other = any(values["other"])
    return [
        {"key": key, "label": label, **_amounts(values[key][0], values[key][1], view)}
        for key, label in _ordered_lines(include_other)
    ]


def _summary_matrix(frames: dict, view: str, keys: list) -> list:
    pairs = _section_pairs(frames, True)
    per_owner = {owner: _line_values(pairs.get(owner, {})) for owner in keys}
    include_other = any(any(v["other"]) for v in per_owner.values())
    rows = []
    for key, label in _ordered_lines(include_other):
        rows.append({
            "key": key,
            "label": label,
            "values": {
                owner: _amounts(per_owner[owner][key][0], per_owner[owner][key][1], view)
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
    view = normalize_view(view) or "actual"
    key = str(biz_key).strip()
    props = _property_lookup([key])
    if key not in props:
        raise ValueError(f"Property '{key}' not found.")

    period, frames, excluded, notes = _load(view, [key], ym_from, ym_to)
    books, warnings = _books_and_warnings(frames)
    warnings = warnings + notes
    has_data = any(not frame.empty for frame in frames.values())

    totals = _amounts(_frame_total(frames, "actual"), _frame_total(frames, "budget"), view)

    accounts = []
    months = []
    if has_data:
        account_df = _add_sort_column(_merge_amounts(frames, ["GLAccountKey"]), view)
        account_df = account_df.sort_values("_sort", ascending=False)
        if top_accounts:
            account_df = account_df.head(int(top_accounts))
        for record in account_df.to_dict(orient="records"):
            accounts.append({
                "GLAccountKey": int(record["GLAccountKey"]),
                **_account_info(record["GLAccountKey"]),
                **_row_amounts(record, view),
            })

        month_df = _merge_amounts(frames, ["YearMonthKey"]).sort_values("YearMonthKey")
        for record in month_df.to_dict(orient="records"):
            months.append({
                "YearMonthKey": int(record["YearMonthKey"]),
                **_row_amounts(record, view),
            })

    return {
        "view": view,
        "property": props[key],
        "period": period,
        "books": books,
        "has_data": has_data,
        "excluded_unmapped": excluded,
        "totals": totals,
        "summary": _summary_single(frames, view),
        "accounts": accounts,
        "months": months,
        "warnings": warnings,
    }


def compare_financials(biz_keys, view=None, ym_from=None, ym_to=None, top_accounts=None) -> dict:
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
            "summary": [],
            "accounts": [],
            "not_found": not_found,
            "no_data": [],
            "warnings": [],
        }

    period, frames, excluded, notes = _load(view, found, ym_from, ym_to)
    books, warnings = _books_and_warnings(frames)
    warnings = warnings + notes
    has_data = any(not frame.empty for frame in frames.values())

    totals_map = {}
    matrix_records = []
    if has_data:
        totals_df = _merge_amounts(frames, ["PropertyBizKey"])
        totals_map = {rec["PropertyBizKey"]: rec for rec in totals_df.to_dict(orient="records")}
        matrix_df = _add_sort_column(_merge_amounts(frames, ["GLAccountKey", "PropertyBizKey"]), view)
        matrix_records = matrix_df.to_dict(orient="records")

    property_rows = []
    no_data = []
    for key in found:
        record = totals_map.get(key)
        if record is None:
            no_data.append(key)
        property_rows.append({
            **props[key],
            "has_data": record is not None,
            **_row_amounts(record or {}, view),
        })

    by_account = {}
    account_weight = {}
    for record in matrix_records:
        account_key = int(record["GLAccountKey"])
        by_account.setdefault(account_key, {})[record["PropertyBizKey"]] = record
        account_weight[account_key] = account_weight.get(account_key, 0.0) + float(record["_sort"])

    ordered_accounts = sorted(account_weight, key=lambda k: account_weight[k], reverse=True)
    if top_accounts:
        ordered_accounts = ordered_accounts[: int(top_accounts)]

    account_rows = []
    for account_key in ordered_accounts:
        values = {}
        for key in found:
            values[key] = _row_amounts(by_account[account_key].get(key, {}), view)
        account_rows.append({
            "GLAccountKey": account_key,
            **_account_info(account_key),
            "values": values,
        })

    return {
        "view": view,
        "period": period,
        "books": books,
        "excluded_unmapped": excluded,
        "properties": property_rows,
        "summary": _summary_matrix(frames, view, found),
        "accounts": account_rows,
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
            "include_unmapped_accounts": settings.INCLUDE_UNMAPPED_ACCOUNTS,
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
        "section_counts": state.get("section_counts", {}),
        "sample": [{"GLAccountKey": k, **v} for k, v in list(state["lookup"].items())[:5]],
        "error": state["error"],
    }

    try:
        book_df = pd.read_sql(text(f"SELECT * FROM {_quote_table(settings.GL_BOOK_TABLE)}"), get_engine())
        output["gl_books"] = {"row_count": len(book_df), "rows": _records(book_df.head(50))}
    except Exception as e:
        output["gl_books"] = {"error": _error_text(e)}

    return output

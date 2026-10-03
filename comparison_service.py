import logging
from typing import Optional

import financial_service
from property_service import get_by_bizkey

logger = logging.getLogger(__name__)

DEFAULT_COMPARISON_FIELDS = [
    "PropertyBizKey",
    "PropertyName",
    "City",
    "State",
    "Market",
    "AssetType",
    "PropertySizeCategory",
    "PropertyAgeBucket",
    "YearBuilt",
    "PropertyStatus",
    "ManagementCompany",
    "Latitude",
    "Longitude",
]


def financial_payload(
    biz_keys: list[str],
    view: Optional[str],
    ym_from: Optional[int] = None,
    ym_to: Optional[int] = None,
    top_accounts: Optional[int] = None,
) -> dict:
    view = financial_service.normalize_view(view)
    if not view:
        return {}

    keys = [str(k).strip() for k in biz_keys if str(k).strip()]
    if not keys:
        return {}

    try:
        if len(keys) == 1:
            data = financial_service.get_property_financials(keys[0], view, ym_from, ym_to, top_accounts)
        else:
            data = financial_service.compare_financials(keys, view, ym_from, ym_to, top_accounts)
    except ValueError:
        raise
    except Exception as e:
        logger.error("Financial lookup failed: %s", e)
        return {
            "view": view,
            "financials": None,
            "financials_error": "Financial data is temporarily unavailable.",
        }

    return {"view": view, "financials": data}


def build_comparison(
    biz_keys: list[str],
    fields: Optional[list[str]] = None,
    view: Optional[str] = None,
    ym_from: Optional[int] = None,
    ym_to: Optional[int] = None,
    top_accounts: Optional[int] = None,
) -> dict:
    fields = fields or DEFAULT_COMPARISON_FIELDS
    rows = []
    not_found = []
    found_keys = []

    for key in biz_keys:
        record = get_by_bizkey(key)
        if record is None:
            not_found.append(key)
            continue
        found_keys.append(str(record["PropertyBizKey"]))
        rows.append({field: record.get(field) for field in fields})

    result = {
        "fields": fields,
        "properties": rows,
        "not_found": not_found,
    }

    if view and found_keys:
        result.update(financial_payload(found_keys, view, ym_from, ym_to, top_accounts))

    return result

import logging
from typing import Optional

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


def build_comparison(biz_keys: list[str], fields: Optional[list[str]] = None) -> dict:
    fields = fields or DEFAULT_COMPARISON_FIELDS
    rows = []
    not_found = []

    for key in biz_keys:
        record = get_by_bizkey(key)
        if record is None:
            not_found.append(key)
            continue
        rows.append({field: record.get(field) for field in fields})

    return {
        "fields": fields,
        "properties": rows,
        "not_found": not_found,
    }

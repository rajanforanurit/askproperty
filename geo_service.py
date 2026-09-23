import logging
import numpy as np
import pandas as pd

from property_service import get_all_properties, get_by_bizkey
from config import settings

logger = logging.getLogger(__name__)

EARTH_RADIUS_MILES = 3958.8


def haversine_distance_miles(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_MILES * c


def find_nearest_properties(
    target_biz_key: str,
    count: int = None,
    active_only: bool = True,
    exclude_missing_coords: bool = True,
) -> dict:
    count = count or settings.DEFAULT_NEARBY_COUNT
    count = min(count, settings.MAX_NEARBY_COUNT)

    target = get_by_bizkey(target_biz_key)
    if target is None:
        raise ValueError(f"Property with PropertyBizKey '{target_biz_key}' not found.")

    if pd.isna(target.get("Latitude")) or pd.isna(target.get("Longitude")):
        raise ValueError(
            f"Property '{target.get('PropertyName')}' ({target_biz_key}) has no coordinates on file."
        )

    df = get_all_properties(active_only=active_only)

    if exclude_missing_coords:
        df = df.dropna(subset=["Latitude", "Longitude"])

    df = df[df["PropertyBizKey"] != target["PropertyBizKey"]]

    if df.empty:
        return {"target": target, "nearby": []}

    distances = haversine_distance_miles(
        target["Latitude"], target["Longitude"],
        df["Latitude"].values, df["Longitude"].values,
    )
    df = df.copy()
    df["distance_miles"] = np.round(distances, 2)

    nearest = df.sort_values("distance_miles").head(count)

    return {
        "target": target,
        "nearby": nearest.to_dict(orient="records"),
    }

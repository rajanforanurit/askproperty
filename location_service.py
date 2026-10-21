"""Location understanding for property search.

Pipeline: classify the requested location -> match it against the property table ->
resolve coordinates (geocoder, then database evidence) -> fall back to nearby properties.
A geocoder failure is never reported as "no properties".
"""
import logging
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

import geo_service
import property_service
from config import settings

logger = logging.getLogger(__name__)

STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "district of columbia": "DC", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
    "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}
ABBRS = set(STATES.values())
ABBR_TO_NAME = {v: k for k, v in STATES.items()}

STREET_WORDS = {
    "street": "st", "st": "st", "avenue": "ave", "ave": "ave", "av": "ave", "road": "rd", "rd": "rd",
    "drive": "dr", "dr": "dr", "boulevard": "blvd", "blvd": "blvd", "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "parkway": "pkwy", "pkwy": "pkwy", "highway": "hwy", "hwy": "hwy",
    "place": "pl", "pl": "pl", "circle": "cir", "cir": "cir", "way": "way", "trail": "trl", "trl": "trl",
    "north": "n", "south": "s", "east": "e", "west": "w", "n": "n", "s": "s", "e": "e", "w": "w",
    "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt",
}

ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
PROXIMITY_RE = re.compile(r"\b(near|nearby|around|close to|closest to|nearest to|next to|surrounding|in the vicinity of)\b", re.IGNORECASE)
STREET_RE = re.compile(r"^\s*\d+[a-z]?\s+\S+")


def _norm_text(value) -> str:
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm_street(value) -> str:
    words = _norm_text(value).split()
    return " ".join(STREET_WORDS.get(w, w) for w in words)


def _norm_city(value) -> str:
    text = _norm_text(value)
    text = re.sub(r"^(saint|st)\s", "st ", text)
    text = re.sub(r"^(mount|mt)\s", "mt ", text)
    text = re.sub(r"^(fort|ft)\s", "ft ", text)
    return text


def _state_of(token: str) -> Optional[str]:
    raw = (token or "").strip().strip(".")
    if raw.upper() in ABBRS:
        return raw.upper()
    return STATES.get(raw.lower())


def classify(text: str, hint: Optional[str] = None) -> dict:
    """Understand what kind of place was typed. Never changes its meaning."""
    raw = (text or "").strip()
    info = {"raw": raw, "type": "place", "normalized": raw, "street": None, "city": None, "state": None, "zip": None}
    if not raw:
        return info

    # The ZIP is the LAST 5-digit group; a leading 5-digit group followed by words is a street number.
    zip_match = None
    for m in ZIP_RE.finditer(raw):
        followed_by_words = bool(re.match(r"\s+[A-Za-z]", raw[m.end():]))
        if m.start() == 0 and followed_by_words:
            continue
        zip_match = m
    zip_code = zip_match.group(1) if zip_match else None
    remainder = (raw[:zip_match.start()] + " " + raw[zip_match.end():]).strip(" ,") if zip_match else raw
    remainder = re.sub(r"\s+", " ", remainder).strip(" ,")

    parts = [p.strip() for p in remainder.split(",") if p.strip()]
    state = None
    city = None
    street = None

    if len(parts) >= 2:
        tail = parts[-1].split()
        # "Austin, TX" / "Austin, Texas" / "123 Main St, Austin, TX"
        state = _state_of(parts[-1]) or (_state_of(tail[0]) if tail else None)
        if state:
            parts = parts[:-1]
        if parts:
            city = parts[-1]
            parts = parts[:-1]
        if parts:
            street = ", ".join(parts)
        if street is None and city and STREET_RE.match(city):
            street, city = city, None
    elif len(parts) == 1:
        single = parts[0]
        words = single.split()
        if STREET_RE.match(single):
            street = single
        elif _state_of(single) and not zip_code:
            state = _state_of(single)
        elif len(words) >= 2 and _state_of(words[-1]) and not single.lower().endswith(("york", "jersey", "mexico", "hampshire")):
            # "Austin TX"
            state = _state_of(words[-1])
            city = " ".join(words[:-1])
        elif len(words) >= 3 and _state_of(" ".join(words[-2:])):
            state = _state_of(" ".join(words[-2:]))
            city = " ".join(words[:-2])
        elif single.lower().endswith(" county"):
            info["type"] = "county"
        else:
            city = single

    if street:
        info["type"] = "address"
    elif zip_code and not city and not state:
        info["type"] = "zip"
    elif city and single_is_county(city):
        info["type"] = "county"
    elif city:
        info["type"] = "city"
    elif state:
        info["type"] = "state"
    elif zip_code:
        info["type"] = "zip"
    elif hint in ("neighborhood", "region", "county", "place"):
        info["type"] = hint

    if hint == "neighborhood" and info["type"] in ("city", "place"):
        info["type"] = "neighborhood"

    info.update({"street": street, "city": city, "state": state, "zip": zip_code})
    pieces = [street, city, state, zip_code]
    info["normalized"] = ", ".join(str(p) for p in pieces if p) or raw
    return info


def single_is_county(value: str) -> bool:
    return str(value or "").strip().lower().endswith(" county")


def _zip5(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d{3,5})", expand=False).fillna("").str.zfill(5)


def match_properties(df: pd.DataFrame, loc: dict) -> tuple:
    """Return (matching rows, how they matched)."""
    if df.empty:
        return df, None
    city_norm = df["City"].map(_norm_city) if "City" in df else pd.Series("", index=df.index)
    state_up = df["State"].astype(str).str.strip().str.upper() if "State" in df else pd.Series("", index=df.index)
    zip_col = _zip5(df["Zipcode"]) if "Zipcode" in df else pd.Series("", index=df.index)

    def by_state(frame_mask):
        if loc.get("state"):
            return frame_mask & (state_up == loc["state"])
        return frame_mask

    kind = loc["type"]
    if kind == "address":
        street = _norm_street(loc["street"])
        addr = df["PropertyAddress"].map(_norm_street) if "PropertyAddress" in df else pd.Series("", index=df.index)
        mask = addr.str.contains(re.escape(street), na=False) if street else pd.Series(False, index=df.index)
        if loc.get("zip"):
            narrowed = mask & (zip_col == loc["zip"])
            mask = narrowed if narrowed.any() else mask
        if loc.get("city"):
            narrowed = mask & (city_norm == _norm_city(loc["city"]))
            mask = narrowed if narrowed.any() else mask
        return df[by_state(mask)] if mask.any() else df[mask], "address"

    if kind == "zip":
        return df[zip_col == loc["zip"]], "zip"

    if kind == "state":
        return df[state_up == loc["state"]], "state"

    if kind in ("city", "neighborhood", "county", "place"):
        name = loc.get("city") or loc.get("raw")
        norm = _norm_city(re.sub(r"\bcounty\b", "", name, flags=re.I)) if kind == "county" else _norm_city(name)
        if loc.get("zip"):
            exact_zip = df[zip_col == loc["zip"]]
            if not exact_zip.empty:
                return exact_zip, "zip"
        exact = by_state(city_norm == norm)
        if exact.any():
            return df[exact], "city"
        partial = by_state(city_norm.str.contains(re.escape(norm), na=False)) if norm else exact
        if partial.any():
            return df[partial], "city"
        haystack = pd.Series("", index=df.index)
        for column in ("Market", "MSA", "PropertyCampus", "PropertyAddress"):
            if column in df:
                haystack = haystack + " " + df[column].map(_norm_text)
        wide = by_state(haystack.str.contains(re.escape(_norm_text(name)), na=False)) if norm else exact
        if wide.any():
            return df[wide], "market"
        return df[exact], None
    return df.iloc[0:0], None


# ---------------------------------------------------------------- geocoding

class GeocodeError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


_geocode_cache: dict = {}
_GEOCODE_TTL = 24 * 3600


def geocoder_configured() -> bool:
    """Geocoding is on unless GEOCODE_PROVIDER=none. Without configuration the free public
    services (US Census for street addresses, OpenStreetMap Nominatim for the rest) are used."""
    return settings.GEOCODE_PROVIDER != "none"


def _configured_providers() -> list:
    provider = settings.GEOCODE_PROVIDER
    if provider == "none":
        return []
    if provider == "google":
        return ["google"]
    if provider == "nominatim" and settings.GEOCODE_URL:
        return ["nominatim"]
    return ["census", "nominatim_public"]


def _geocode_google(query: str) -> Optional[dict]:
    url = settings.GEOCODE_URL or "https://maps.googleapis.com/maps/api/geocode/json"
    resp = requests.get(url, params={"address": query, "key": settings.GEOCODE_API_KEY, "region": "us"},
                        timeout=settings.GEOCODE_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status")
    if status == "ZERO_RESULTS":
        return None
    if status == "OK" and data.get("results"):
        top = data["results"][0]
        loc = top["geometry"]["location"]
        return {"lat": float(loc["lat"]), "lng": float(loc["lng"]), "label": top.get("formatted_address") or query,
                "precision": top["geometry"].get("location_type", "").lower() or None}
    raise GeocodeError(f"provider_status_{status}")


def _geocode_nominatim(query: str, url: str) -> Optional[dict]:
    params = {"q": query, "format": "json", "limit": 1, "countrycodes": "us"}
    if settings.GEOCODE_API_KEY and settings.GEOCODE_PROVIDER == "nominatim":
        params["key"] = settings.GEOCODE_API_KEY
    resp = requests.get(url, params=params, timeout=settings.GEOCODE_TIMEOUT_SECONDS,
                        headers={"User-Agent": "AskPropertyData/1.0"})
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    top = data[0]
    return {"lat": float(top["lat"]), "lng": float(top["lon"]), "label": top.get("display_name") or query,
            "precision": top.get("type")}


def _geocode_census(query: str) -> Optional[dict]:
    if not STREET_RE.match(query):
        return None
    resp = requests.get(
        "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
        params={"address": query, "benchmark": "Public_AR_Current", "format": "json"},
        timeout=settings.GEOCODE_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    matches = (resp.json().get("result") or {}).get("addressMatches") or []
    if not matches:
        return None
    top = matches[0]
    return {"lat": float(top["coordinates"]["y"]), "lng": float(top["coordinates"]["x"]),
            "label": top.get("matchedAddress") or query, "precision": "address"}


def geocode(query: str) -> Optional[dict]:
    """Return {lat, lng, label, precision} or None for a genuine no-match. Raises GeocodeError on failure."""
    key = _norm_text(query)
    if not key:
        return None
    cached = _geocode_cache.get(key)
    if cached and time.time() - cached[0] < _GEOCODE_TTL:
        return cached[1]
    providers = _configured_providers()
    if not providers:
        raise GeocodeError("not_configured")

    result = None
    answered = False
    last_reason = "unavailable"
    for provider in providers:
        try:
            if provider == "google":
                found = _geocode_google(query)
            elif provider == "nominatim":
                found = _geocode_nominatim(query, settings.GEOCODE_URL)
            elif provider == "census":
                found = _geocode_census(query)
            else:
                found = _geocode_nominatim(query, "https://nominatim.openstreetmap.org/search")
        except GeocodeError as e:
            last_reason = e.reason
            continue
        except requests.Timeout:
            last_reason = "timeout"
            continue
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.warning("Geocoding failed (%s): %s", provider, type(e).__name__)
            last_reason = "unavailable"
            continue
        answered = True
        if found:
            result = found
            break
    if result is None and not answered:
        raise GeocodeError(last_reason)
    _geocode_cache[key] = (time.time(), result)
    return result


def resolve_origin(loc: dict) -> tuple:
    """Find coordinates for a requested place, relaxing from street address to ZIP to city.

    Returns (found | None, level | None, failure_reason | None).
    """
    attempts = []
    if loc.get("street"):
        attempts.append(("address", loc["raw"]))
        attempts.append(("address", loc["normalized"]))
    elif loc["type"] in ("zip", "city", "neighborhood", "county", "place", "state"):
        attempts.append((loc["type"] if loc["type"] in ("zip", "state") else "city", loc["raw"]))
    state = loc.get("state") or ""
    if loc.get("zip"):
        attempts.append(("zip", f"{loc['zip']}, {state}".strip(", ") + ", USA"))
    if loc.get("city"):
        attempts.append(("city", f"{loc['city']}, {state}".strip(", ") + ", USA"))
    seen, failure, answered = set(), None, False
    for level, query in attempts:
        if not query or query.lower() in seen:
            continue
        seen.add(query.lower())
        try:
            found = geocode(query)
        except GeocodeError as e:
            failure = e.reason
            continue
        answered = True
        if found:
            return found, level, None
    return None, None, (None if answered else failure)


def _valid_coords(df: pd.DataFrame) -> pd.DataFrame:
    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    lng = pd.to_numeric(df["Longitude"], errors="coerce")
    ok = lat.between(-90, 90) & lng.between(-180, 180) & ~((lat == 0) & (lng == 0))
    return df[ok.fillna(False)]


def fill_missing_coordinates(records: list) -> tuple:
    """Try to geocode addresses for records without coordinates. Returns (records, unmapped_count)."""
    budget = settings.LOCATION_MAX_GEOCODE_MISSING
    unmapped = 0
    out = []
    for record in records:
        lat, lng = record.get("Latitude"), record.get("Longitude")
        has = lat is not None and lng is not None and lat == lat and lng == lng
        if has:
            out.append(record)
            continue
        item = dict(record)
        address = ", ".join(str(item.get(k)) for k in ("PropertyAddress", "City", "State", "Zipcode") if item.get(k))
        if address and budget > 0 and geocoder_configured():
            budget -= 1
            try:
                found = geocode(address)
            except GeocodeError:
                found = None
            if found:
                item["Latitude"], item["Longitude"] = found["lat"], found["lng"]
                item["coordinates_source"] = "geocoded"
                out.append(item)
                continue
        unmapped += 1
        out.append(item)
    return out, unmapped


# -------------------------------------------------------------- main search

def _centroid(df: pd.DataFrame) -> Optional[tuple]:
    valid = _valid_coords(df)
    if valid.empty:
        return None
    return float(valid["Latitude"].median()), float(valid["Longitude"].median())


def _within(df: pd.DataFrame, origin: tuple, radius: Optional[float]) -> pd.DataFrame:
    valid = _valid_coords(df)
    if valid.empty:
        return valid.assign(distance_miles=pd.Series(dtype=float))
    miles = geo_service.haversine_distance_miles(origin[0], origin[1], valid["Latitude"].values, valid["Longitude"].values)
    valid = valid.copy()
    valid["distance_miles"] = np.round(miles, 2)
    if radius:
        valid = valid[valid["distance_miles"] <= float(radius)]
    return valid.sort_values("distance_miles")


def _records(df: pd.DataFrame) -> list:
    import query_service
    return [query_service.clean_record(r) for r in df.to_dict(orient="records")]


LEVEL_LABEL = {"address": "the address", "zip": "the ZIP code area", "city": "the city", "state": "the state"}


def _db_fallback(loc: dict, active: pd.DataFrame) -> tuple:
    """Same ZIP, then same city, then same ZIP prefix. Returns (df, scope) with no distances."""
    if loc.get("zip"):
        by_zip, _ = match_properties(active, dict(loc, type="zip"))
        if not by_zip.empty:
            return by_zip.sort_values("PropertyName"), "zip"
    if loc.get("city"):
        by_city, _ = match_properties(active, dict(loc, type="city", street=None))
        if not by_city.empty:
            return by_city.sort_values("PropertyName"), "city"
    if loc.get("zip"):
        zips = _zip5(active["Zipcode"]) if "Zipcode" in active else pd.Series("", index=active.index)
        close = active[zips.str[:3] == loc["zip"][:3]]
        if loc.get("state") and "State" in close:
            close = close[close["State"].astype(str).str.upper() == loc["state"]]
        if not close.empty:
            gap = (pd.to_numeric(_zip5(close["Zipcode"]), errors="coerce") - int(loc["zip"])).abs()
            return close.assign(_gap=gap).sort_values(["_gap", "PropertyName"]).drop(columns="_gap"), "zip_area"
    return active.iloc[0:0], None


SCOPE_LABEL = {"zip": "ZIP code", "city": "city", "zip_area": "ZIP code area"}


def search(location_text: str, hint: Optional[str] = None, radius: Optional[float] = None,
           constraint_fn=None, proximity: bool = False) -> dict:
    """Run the location decision process.

    proximity=True means "near/around X": rank the closest properties, X need not be a property.
    constraint_fn(df) -> df lets the caller apply attribute filters to candidate rows.
    """
    loc = classify(location_text, hint)
    info = {
        "requested": loc["raw"], "normalized": loc["normalized"], "type": loc["type"],
        "status": "exact", "message": None, "origin": None, "radius_miles": radius,
        "scope": None, "matched_by": None, "geocoder": "skipped", "unmapped_count": 0,
        "included_inactive": False, "proximity": bool(proximity),
    }
    result = {"location": info, "records": [], "warnings": []}

    def finish(status, message, records):
        info["status"], info["message"] = status, message
        result["records"] = records
        return result

    active = property_service.get_all_properties(active_only=True)
    everything = property_service.get_all_properties()
    if constraint_fn:
        active, everything = constraint_fn(active), constraint_fn(everything)

    matched, how = match_properties(active, loc)
    if matched.empty:
        inactive_match, inactive_how = match_properties(everything, loc)
        if not inactive_match.empty:
            matched, how = inactive_match, inactive_how
            info["included_inactive"] = True
            result["warnings"].append({
                "code": "inactive_only",
                "message": "Only inactive or disposed properties match this location, so they are shown.",
            })
    info["matched_by"] = how
    pool = everything if info["included_inactive"] else active

    # ---- resolve an origin when distance matters (radius / "near" / no exact match)
    origin = None
    level = None
    geo_failed = None
    if radius or proximity or matched.empty:
        found, level, geo_failed = resolve_origin(loc)
        if found:
            origin = (found["lat"], found["lng"])
            info["geocoder"] = "ok"
            info["origin"] = {"lat": found["lat"], "lng": found["lng"], "source": "geocoder",
                              "label": found["label"], "level": level}
        else:
            info["geocoder"] = "failed" if geo_failed else "no_match"
        # A property at the requested address is itself reliable evidence of where it is.
        if not matched.empty and (origin is None or (how == "address" and level != "address")):
            centre = _centroid(matched)
            if centre:
                origin = centre
                level = "address" if how == "address" else ("zip" if how == "zip" else "city")
                info["origin"] = {"lat": centre[0], "lng": centre[1], "source": "database",
                                  "label": "Matching properties", "level": level}

    missing = len(pool) - len(_valid_coords(pool))
    coords_warning = {
        "code": "missing_coordinates",
        "message": f"{missing} properties have no usable coordinates and could not be included in the distance ranking.",
    }

    # ---- distance-based results: radius or "near X"
    if (radius or proximity) and origin is not None:
        steps = [radius] if radius else (settings.LOCATION_PROXIMITY_RADII_MILES or [25])
        near = pool.iloc[0:0]
        used = steps[-1]
        for step in steps:
            near = _within(pool, origin, step)
            used = step
            if not near.empty:
                break
        if missing:
            result["warnings"].append(coords_warning)
        info["radius_miles"] = used
        approximate = level in ("zip", "city", "state") and loc["type"] == "address"
        if approximate:
            where = f"the centre of {LEVEL_LABEL[level]} ({(info['origin'] or {}).get('label', '')})".replace(" ()", "")
            note = f" We couldn't pinpoint the exact address, so distances are measured from {where}."
        else:
            note = ""
        if near.empty:
            return finish("empty", f"No eligible properties were found within {used:g} miles of {loc['raw']}.{note}", [])
        if radius:
            message = f"{len(near)} properties within {used:g} miles of {loc['raw']}.{note}"
        else:
            message = f"Showing the {len(near)} closest properties to {loc['raw']} (within {used:g} miles), nearest first.{note}"
        return finish("nearby" if approximate else "exact", message, _records(near))

    # ---- no usable origin for a distance request, or no exact match: use what the data can prove
    if (radius or proximity or matched.empty) and origin is None and (matched.empty or radius):
        if radius:
            return finish("unresolved", _unresolved_message(loc, geo_failed, True), [])
        broader, scope = _db_fallback(loc, active)
        if not broader.empty:
            info["scope"] = scope
            return finish(
                "nearby",
                f"We couldn't pinpoint {loc['raw']} on the map, so these are properties in the same {SCOPE_LABEL[scope]} "
                f"that may be relevant to your search. Distances are not shown because the exact location could not be resolved.",
                _records(broader),
            )
        return finish("unresolved", _unresolved_message(loc, geo_failed, proximity), [])

    if proximity and origin is None and not matched.empty:
        centre = _centroid(matched)
        if centre is None:
            return finish("exact", None, _records(matched.sort_values("PropertyName")))

    # ---- exact matches
    if not matched.empty:
        records = _records(matched.sort_values("PropertyName"))
        if origin and info["origin"] and info["origin"]["source"] == "geocoder":
            near = _within(matched, origin, None)
            lookup = {str(r["PropertyBizKey"]): r["distance_miles"] for r in near.to_dict(orient="records")}
            for r in records:
                if str(r["PropertyBizKey"]) in lookup:
                    r["distance_miles"] = lookup[str(r["PropertyBizKey"])]
        records, unmapped = fill_missing_coordinates(records)
        info["unmapped_count"] = unmapped
        if unmapped:
            result["warnings"].append({
                "code": "missing_coordinates",
                "message": "Some properties cannot be mapped because their location coordinates are unavailable.",
            })
        return finish("exact", None, records)

    return finish("unresolved", _unresolved_message(loc, geo_failed, proximity), [])


def _unresolved_message(loc: dict, reason: Optional[str], proximity: bool = False) -> str:
    base = f"We couldn't pinpoint \"{loc['raw']}\" and none of your properties are in the same ZIP code, city or ZIP area, so there is no reliable basis for ranking properties near it."
    if reason == "not_configured":
        return base + " Address lookup is turned off (GEOCODE_PROVIDER=none). Try a city, state or ZIP code that appears in your properties."
    if reason in ("timeout", "unavailable") or (reason or "").startswith("provider_status"):
        return base + " The address lookup service is not responding, so this does not mean there are no properties there. Please try again shortly."
    return base + " Check the spelling, or try a nearby city, state or ZIP code."

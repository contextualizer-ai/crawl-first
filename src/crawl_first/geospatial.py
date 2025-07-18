"""
Geospatial utilities for crawl-first.

Handles coordinate processing, geocoding, reverse geocoding, and distance calculations.
"""

import time
from math import asin, cos, radians, sin, sqrt
from time import sleep
from typing import Any, Dict, Optional

import requests
from geopy.geocoders import Nominatim

from .cache import cache_key, get_cache, save_cache

# Nominatim API rate limiting configuration
# 1.1 seconds ensures we stay under the 1 request/second limit
# This is PREVENTIVE rate limiting - enforced by Nominatim's usage policy
NOMINATIM_RATE_LIMIT_SECONDS = 1.1

# Global variable to track last Nominatim request time
_last_nominatim_request_time: Optional[float] = None


def nominatim_rate_limit() -> None:
    """Apply rate limiting for Nominatim API (1 req/sec).

    This is PREVENTIVE rate limiting - we sleep to avoid violating
    Nominatim's usage policy, not because we got an error.

    Only sleeps if less than NOMINATIM_RATE_LIMIT_SECONDS have passed
    since the last Nominatim request. Updates the last request timestamp.
    """
    global _last_nominatim_request_time

    current_time = time.time()

    if _last_nominatim_request_time is not None:
        time_since_last_request = current_time - _last_nominatim_request_time
        if time_since_last_request < NOMINATIM_RATE_LIMIT_SECONDS:
            sleep_time = NOMINATIM_RATE_LIMIT_SECONDS - time_since_last_request
            sleep(sleep_time)
            # Recalculate current time after sleep for accuracy
            current_time = time.time()

    # Update the timestamp with consistent timing
    _last_nominatim_request_time = current_time


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points in meters."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c * 6371000  # Earth radius in meters


def get_elevation(lat: float, lon: float) -> Optional[float]:
    """Get elevation for coordinates using multiple free APIs with fallbacks."""
    key = cache_key({"lat": round(lat, 6), "lon": round(lon, 6)})
    cached = get_cache("elevation", key)
    if cached:
        return cached.get("elevation")

    elevation = None

    # Try Open-Elevation API (free, no rate limits)
    try:
        url = "https://api.open-elevation.com/api/v1/lookup"
        params = {"locations": f"{lat},{lon}"}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            elevation = float(data["results"][0]["elevation"])
    except Exception:
        pass

    # Fallback to USGS Elevation Point Query Service (US only, but free)
    if elevation is None:
        try:
            url = "https://nationalmap.gov/epqs/pqs.php"
            params = {"x": str(lon), "y": str(lat), "units": "Meters", "output": "json"}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "USGS_Elevation_Point_Query_Service" in data:
                result = data["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]
                if result.get("Elevation") not in [None, -1000000]:
                    elevation = float(result["Elevation"])
        except Exception:
            pass

    save_cache("elevation", key, {"elevation": elevation})
    return elevation


def geocode_location_name(location_name: str) -> Dict[str, Any]:
    """Geocode location name to get coordinates using Nominatim."""
    key = cache_key({"location_name": location_name.lower().strip()})
    cached = get_cache("geocode", key)
    if cached:
        return cached

    geolocator = Nominatim(user_agent="biosample-analyzer/1.0")

    result: Dict[str, Any] = {
        "location_name": location_name,
        "coordinates": None,
        "display_name": None,
        "place_type": None,
        "address": {},
        "administrative": {},
        "error": None,
    }

    try:
        nominatim_rate_limit()
        location = geolocator.geocode(location_name, language="en")

        if location:
            result["coordinates"] = {
                "latitude": location.latitude,
                "longitude": location.longitude,
            }
            result["display_name"] = location.address
            result["place_type"] = location.raw.get("type", "unknown")

            address = location.raw.get("address", {})
            address_fields = [
                "house_number",
                "road",
                "neighbourhood",
                "suburb",
                "village",
                "town",
                "city",
                "municipality",
                "county",
                "state",
                "country",
                "postcode",
                "country_code",
            ]

            for field in address_fields:
                if field in address:
                    result["address"][field] = address[field]
        else:
            result["error"] = "Location not found"

    except Exception as e:
        result["error"] = str(e)

    save_cache("geocode", key, result)
    return result


def reverse_geocode(lat: float, lon: float) -> Dict[str, Any]:
    """Reverse geocode coordinates to get place information using Nominatim."""
    key = cache_key({"lat": round(lat, 5), "lon": round(lon, 5)})
    cached = get_cache("reverse_geocode", key)
    if cached:
        return cached

    geolocator = Nominatim(user_agent="biosample-analyzer/1.0")

    result = {
        "coordinates": {"latitude": lat, "longitude": lon},
        "display_name": None,
        "place_type": None,
        "address": {},
        "administrative": {},
        "error": None,
    }

    try:
        nominatim_rate_limit()
        location = geolocator.reverse(f"{lat}, {lon}", language="en", zoom=18)

        if location:
            result["display_name"] = location.address
            result["place_type"] = location.raw.get("type", "unknown")

            address = location.raw.get("address", {})
            address_fields = [
                "house_number",
                "road",
                "neighbourhood",
                "suburb",
                "village",
                "town",
                "city",
                "municipality",
                "county",
                "state",
                "country",
                "postcode",
                "country_code",
            ]

            # Safely access the address dict
            address_dict = result["address"]
            assert isinstance(address_dict, dict)  # Help mypy understand this is a dict

            for field in address_fields:
                if field in address:
                    address_dict[field] = address[field]

            # Safely access the administrative dict
            admin_dict = result["administrative"]
            assert isinstance(admin_dict, dict)  # Help mypy understand this is a dict

            admin_fields = ["state_district", "region", "province", "state_code"]
            for field in admin_fields:
                if field in address:
                    admin_dict[field] = address[field]

            if "amenity" in location.raw:
                result["amenity"] = location.raw["amenity"]
            if "natural" in location.raw:
                result["natural"] = location.raw["natural"]
            if "landuse" in location.raw:
                result["landuse"] = location.raw["landuse"]
        else:
            # Explicitly assign string to the error field
            error_msg: Any = "No location found"
            result["error"] = error_msg

    except Exception as e:
        # Explicitly assign string to the error field
        error_str: Any = str(e)
        result["error"] = error_str

    save_cache("reverse_geocode", key, result)
    return result

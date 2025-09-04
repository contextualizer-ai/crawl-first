"""
Geospatial enrichment module using working APIs.

Provides elevation, weather, location context, and nearby features
using free, reliable APIs that replace ORNL dependencies.
"""

import hashlib
import json
import requests
import time
from typing import Dict, Any, Optional
from pathlib import Path


class GeospatialCache:
    """File-based cache for geospatial API responses."""
    
    def __init__(self, cache_dir: str = "cache/geospatial"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _cache_key(self, api_name: str, params: Dict[str, Any]) -> str:
        """Generate cache key from API name and parameters."""
        param_str = json.dumps(params, sort_keys=True)
        key = hashlib.md5(f"{api_name}:{param_str}".encode()).hexdigest()
        return key
        
    def get(self, api_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result."""
        key = self._cache_key(api_name, params)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached_data = json.load(f)
                    
                # Check if cache is less than 30 days old
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < 30 * 24 * 3600:  # 30 days
                    return cached_data
                else:
                    cache_file.unlink()  # Remove expired cache
            except (json.JSONDecodeError, OSError):
                pass
        return None
        
    def set(self, api_name: str, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Cache result."""
        key = self._cache_key(api_name, params)
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except OSError:
            pass  # Fail silently if can't write cache


# Global cache instance
_cache = GeospatialCache()


def get_elevation_open_elevation(lat: float, lon: float) -> Dict[str, Any]:
    """Get elevation using Open Elevation API."""
    cache_params = {"lat": round(lat, 4), "lon": round(lon, 4)}
    
    # Check cache first
    cached = _cache.get("open_elevation", cache_params)
    if cached is not None:
        return cached
    
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {"locations": f"{lat},{lon}"}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results"):
            result_data = data["results"][0]
            result = {
                "elevation_meters": result_data.get("elevation"),
                "latitude": result_data.get("latitude"),
                "longitude": result_data.get("longitude"),
                "data_source": "Open Elevation API",
                "success": True
            }
        else:
            result = {"elevation_meters": None, "success": False, "data_source": "Open Elevation API"}
            
        # Cache the result
        _cache.set("open_elevation", cache_params, result)
        return result
        
    except Exception as e:
        result = {
            "elevation_meters": None,
            "error": str(e),
            "data_source": "Open Elevation API",
            "success": False
        }
        _cache.set("open_elevation", cache_params, result)
        return result


def get_elevation_usgs(lat: float, lon: float) -> Dict[str, Any]:
    """Get elevation using USGS Elevation Point Query Service (US only)."""
    cache_params = {"lat": round(lat, 4), "lon": round(lon, 4)}
    
    # Check cache first
    cached = _cache.get("usgs_elevation", cache_params)
    if cached is not None:
        return cached
    
    url = "https://nationalmap.gov/epqs/pqs.php"
    params = {
        "x": lon,
        "y": lat, 
        "units": "Meters",
        "output": "json"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "USGS_Elevation_Point_Query_Service" in data:
            query_result = data["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]
            elevation = query_result.get("Elevation")
            
            # USGS returns -1000000 for no data
            if elevation and elevation != -1000000:
                result = {
                    "elevation_meters": float(elevation),
                    "latitude": lat,
                    "longitude": lon,
                    "data_source": "USGS Elevation Point Query Service",
                    "units": "meters",
                    "success": True
                }
            else:
                result = {"elevation_meters": None, "success": False, "data_source": "USGS Elevation Point Query Service"}
        else:
            result = {"elevation_meters": None, "success": False, "data_source": "USGS Elevation Point Query Service"}
            
        # Cache the result
        _cache.set("usgs_elevation", cache_params, result)
        return result
        
    except Exception as e:
        result = {
            "elevation_meters": None,
            "error": str(e),
            "data_source": "USGS Elevation Point Query Service", 
            "success": False
        }
        _cache.set("usgs_elevation", cache_params, result)
        return result


def get_weather_data_open_meteo(lat: float, lon: float, date: str) -> Dict[str, Any]:
    """Get historical weather data using Open-Meteo API."""
    cache_params = {"lat": round(lat, 3), "lon": round(lon, 3), "date": date}
    
    # Check cache first
    cached = _cache.get("open_meteo", cache_params)
    if cached is not None:
        return cached
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean",
        "timezone": "UTC"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "daily" in data and data["daily"]:
            daily = data["daily"]
            
            # Extract first (and only) day's data
            weather_data = {}
            for key, values in daily.items():
                if key != "time" and values and len(values) > 0 and values[0] is not None:
                    weather_data[key] = values[0]
            
            result = {
                "date": date,
                "latitude": lat,
                "longitude": lon,
                "weather_data": weather_data,
                "data_source": "Open-Meteo Historical Weather API",
                "success": True
            }
        else:
            result = {"weather_data": {}, "success": False, "data_source": "Open-Meteo Historical Weather API"}
            
        # Cache the result
        _cache.set("open_meteo", cache_params, result)
        return result
            
    except Exception as e:
        result = {
            "weather_data": {},
            "error": str(e),
            "data_source": "Open-Meteo Historical Weather API",
            "success": False
        }
        _cache.set("open_meteo", cache_params, result)
        return result


def get_reverse_geocoding_nominatim(lat: float, lon: float) -> Dict[str, Any]:
    """Get location information using Nominatim reverse geocoding."""
    cache_params = {"lat": round(lat, 3), "lon": round(lon, 3)}
    
    # Check cache first
    cached = _cache.get("nominatim", cache_params)
    if cached is not None:
        return cached
    
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1,
        "zoom": 18
    }
    
    headers = {
        "User-Agent": "BiosampleEnrichment/1.0 (research purposes)"
    }
    
    try:
        # Be respectful with rate limiting
        time.sleep(1.1)  # Nominatim requires max 1 request/second
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data and "address" in data:
            address = data["address"]
            
            result = {
                "display_name": data.get("display_name"),
                "place_type": data.get("type"),
                "osm_type": data.get("osm_type"),
                "country": address.get("country"),
                "country_code": address.get("country_code"),
                "state": address.get("state"),
                "county": address.get("county"),
                "city": address.get("city") or address.get("town") or address.get("village"),
                "postcode": address.get("postcode"),
                "latitude": float(data.get("lat", lat)),
                "longitude": float(data.get("lon", lon)),
                "data_source": "OpenStreetMap Nominatim",
                "success": True
            }
        else:
            result = {"success": False, "data_source": "OpenStreetMap Nominatim"}
            
        # Cache the result
        _cache.set("nominatim", cache_params, result)
        return result
            
    except Exception as e:
        result = {
            "error": str(e),
            "data_source": "OpenStreetMap Nominatim",
            "success": False
        }
        _cache.set("nominatim", cache_params, result)
        return result


def get_nearby_features_overpass(lat: float, lon: float, radius_km: float = 1.0) -> Dict[str, Any]:
    """Get nearby geographic features using Overpass API."""
    cache_params = {"lat": round(lat, 3), "lon": round(lon, 3), "radius_km": radius_km}
    
    # Check cache first
    cached = _cache.get("overpass", cache_params)
    if cached is not None:
        return cached
    
    url = "https://overpass-api.de/api/interpreter"
    
    # Overpass query for nearby natural features
    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:25];
    (
      way["natural"](around:{radius_m},{lat},{lon});
      way["landuse"](around:{radius_m},{lat},{lon});
      way["water"](around:{radius_m},{lat},{lon});
      relation["natural"](around:{radius_m},{lat},{lon});
    );
    out geom;
    """
    
    try:
        response = requests.post(url, data=query, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        features = []
        if "elements" in data:
            for element in data["elements"][:10]:  # Limit to first 10 features
                tags = element.get("tags", {})
                if tags:
                    feature = {
                        "type": element.get("type"),
                        "id": element.get("id"),
                        "tags": tags,
                        "natural": tags.get("natural"),
                        "landuse": tags.get("landuse"), 
                        "water": tags.get("water"),
                        "name": tags.get("name")
                    }
                    features.append(feature)
        
        result = {
            "latitude": lat,
            "longitude": lon,
            "radius_km": radius_km,
            "features_found": len(features),
            "features": features,
            "data_source": "OpenStreetMap Overpass API",
            "success": True
        }
        
        # Cache the result
        _cache.set("overpass", cache_params, result)
        return result
        
    except Exception as e:
        result = {
            "error": str(e),
            "features": [],
            "data_source": "OpenStreetMap Overpass API",
            "success": False
        }
        _cache.set("overpass", cache_params, result)
        return result


def enrich_location(lat: float, lon: float, collection_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive location enrichment using multiple working APIs.
    
    Args:
        lat: Latitude
        lon: Longitude  
        collection_date: Date in YYYY-MM-DD format (optional)
        
    Returns:
        Dict with enriched location data
    """
    enriched_data = {
        "original_coordinates": {"latitude": lat, "longitude": lon},
        "collection_date": collection_date,
        "enrichment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "elevation": {},
        "weather": {},
        "location_context": {},
        "nearby_features": {},
        "enrichment_summary": {}
    }
    
    # 1. Get elevation data
    elevation = get_elevation_open_elevation(lat, lon)
    enriched_data["elevation"] = elevation
    
    # Fallback to USGS for US locations if Open Elevation fails
    if not elevation.get("success") and (-170 <= lon <= -60 and 15 <= lat <= 75):
        elevation_usgs = get_elevation_usgs(lat, lon)
        if elevation_usgs.get("success"):
            enriched_data["elevation"] = elevation_usgs
    
    # 2. Get weather data if date provided
    if collection_date:
        weather = get_weather_data_open_meteo(lat, lon, collection_date)
        enriched_data["weather"] = weather
    
    # 3. Get location context
    location = get_reverse_geocoding_nominatim(lat, lon)
    enriched_data["location_context"] = location
    
    # 4. Get nearby features
    features = get_nearby_features_overpass(lat, lon, radius_km=1.0)
    enriched_data["nearby_features"] = features
    
    # 5. Create enrichment summary
    successful_enrichments = []
    if enriched_data["elevation"].get("success"):
        successful_enrichments.append("elevation")
    if enriched_data["weather"].get("success"):
        successful_enrichments.append("weather")
    if enriched_data["location_context"].get("success"):
        successful_enrichments.append("location_context")
    if enriched_data["nearby_features"].get("success"):
        successful_enrichments.append("nearby_features")
    
    enriched_data["enrichment_summary"] = {
        "successful_enrichments": successful_enrichments,
        "total_enrichments": len(successful_enrichments),
        "apis_used": [
            "Open Elevation API",
            "USGS Elevation Point Query Service", 
            "Open-Meteo Historical Weather API",
            "OpenStreetMap Nominatim",
            "OpenStreetMap Overpass API"
        ]
    }
    
    return enriched_data
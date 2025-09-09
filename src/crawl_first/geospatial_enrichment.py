"""
Geospatial enrichment module using working APIs.

Provides elevation, weather, location context, and nearby features
using free, reliable APIs that replace ORNL dependencies.
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def validate_api_response(
    response_data: Dict[str, Any],
    api_name: str,
    required_fields: Optional[list] = None,
    success_indicators: Optional[list] = None
) -> Dict[str, Any]:
    """
    Validate API response data and add standardized metadata.
    
    Args:
        response_data: Raw response data from API
        api_name: Name of the API for logging
        required_fields: List of required fields in response
        success_indicators: List of fields that indicate success
    
    Returns:
        Validated response with success flag and error details
    """
    if not isinstance(response_data, dict):
        return {
            "success": False,
            "error": f"{api_name}: Expected dict response, got {type(response_data).__name__}",
            "data_source": api_name,
            "validation_timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    # Check for existing error conditions
    if response_data.get("success") is False:
        return {
            **response_data,
            "validation_timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    # Validate required fields
    if required_fields:
        missing_fields = [field for field in required_fields if field not in response_data]
        if missing_fields:
            return {
                "success": False,
                "error": f"{api_name}: Missing required fields: {missing_fields}",
                "data_source": api_name,
                "available_fields": list(response_data.keys()),
                "validation_timestamp": datetime.utcnow().isoformat() + "Z"
            }
    
    # Check success indicators
    if success_indicators:
        has_success_indicator = any(
            field in response_data and response_data[field] is not None 
            for field in success_indicators
        )
        if not has_success_indicator:
            return {
                **response_data,
                "success": False,
                "error": f"{api_name}: No valid data found (missing success indicators: {success_indicators})",
                "data_source": api_name,
                "validation_timestamp": datetime.utcnow().isoformat() + "Z"
            }
    
    # Add success flag and validation metadata if not present
    validated_response = response_data.copy()
    if "success" not in validated_response:
        validated_response["success"] = True
    if "data_source" not in validated_response:
        validated_response["data_source"] = api_name
    validated_response["validation_timestamp"] = datetime.utcnow().isoformat() + "Z"
    
    return validated_response


class APIRateLimiter:
    """
    Simple rate limiter for API calls with per-service tracking.
    
    Tracks request counts and timing to help identify potential rate limit violations.
    """
    
    def __init__(self, rate_limit_file: str = None):
        if rate_limit_file is None:
            rate_limit_file = str(Path(__file__).parent.parent.parent / "cache" / "rate_limits.json")
        
        self.rate_limit_file = Path(rate_limit_file)
        self.rate_limit_file.parent.mkdir(parents=True, exist_ok=True)
        
        # API rate limit configurations (requests per minute)
        self.rate_limits = {
            "open_elevation": 300,      # 300/min estimated
            "usgs_elevation": 120,      # Conservative estimate
            "open_meteo": 10000,        # 10000/day free tier
            "nominatim": 60,            # 1/second = 60/min
            "overpass": 60,             # Conservative for WMS/Overpass
            "isric_soilgrids": 300,     # Conservative estimate
            "esa_worldcover": 120,      # Conservative for WMS
            "nlcd": 120,                # Conservative for WMS
            "default": 60               # Default conservative limit
        }
        
        self.request_history = self._load_request_history()
    
    def _load_request_history(self) -> Dict[str, list]:
        """Load request history from file."""
        if self.rate_limit_file.exists():
            try:
                with open(self.rate_limit_file) as f:
                    data = json.load(f)
                    # Filter out old requests (older than 24 hours)
                    cutoff = datetime.now() - timedelta(hours=24)
                    for service in data:
                        data[service] = [
                            timestamp for timestamp in data[service]
                            if datetime.fromisoformat(timestamp) > cutoff
                        ]
                    return data
            except (json.JSONDecodeError, OSError, ValueError):
                pass
        return {}
    
    def _save_request_history(self):
        """Save request history to file."""
        try:
            with open(self.rate_limit_file, 'w') as f:
                json.dump(self.request_history, f, indent=2)
        except OSError:
            pass  # Fail silently if can't save
    
    def check_rate_limit(self, service: str) -> Dict[str, Any]:
        """
        Check if we're approaching rate limits for a service.
        
        Args:
            service: API service name
            
        Returns:
            Dict with rate limit status and recommendations
        """
        if service not in self.request_history:
            self.request_history[service] = []
        
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        one_hour_ago = now - timedelta(hours=1)
        
        # Count recent requests
        recent_requests = [
            datetime.fromisoformat(timestamp)
            for timestamp in self.request_history[service]
            if datetime.fromisoformat(timestamp) > one_minute_ago
        ]
        
        hourly_requests = [
            datetime.fromisoformat(timestamp)
            for timestamp in self.request_history[service]
            if datetime.fromisoformat(timestamp) > one_hour_ago
        ]
        
        rate_limit = self.rate_limits.get(service, self.rate_limits["default"])
        requests_per_minute = len(recent_requests)
        requests_per_hour = len(hourly_requests)
        
        # Calculate usage percentage
        usage_percentage = (requests_per_minute / rate_limit) * 100
        
        status = {
            "service": service,
            "requests_last_minute": requests_per_minute,
            "requests_last_hour": requests_per_hour,
            "rate_limit_per_minute": rate_limit,
            "usage_percentage": round(usage_percentage, 1),
            "status": "ok",
            "recommended_delay": 0,
            "timestamp": now.isoformat()
        }
        
        # Determine status and recommendations
        if usage_percentage > 90:
            status["status"] = "critical"
            status["recommended_delay"] = 60  # Wait 1 minute
        elif usage_percentage > 70:
            status["status"] = "warning"
            status["recommended_delay"] = 5   # Wait 5 seconds
        elif usage_percentage > 50:
            status["status"] = "caution"
            status["recommended_delay"] = 1   # Wait 1 second
        
        return status
    
    def record_request(self, service: str) -> None:
        """Record a new API request for rate limit tracking."""
        if service not in self.request_history:
            self.request_history[service] = []
        
        self.request_history[service].append(datetime.now().isoformat())
        
        # Clean up old requests (older than 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.request_history[service] = [
            timestamp for timestamp in self.request_history[service]
            if datetime.fromisoformat(timestamp) > cutoff
        ]
        
        # Save periodically (every 10 requests to avoid too much I/O)
        if len(self.request_history[service]) % 10 == 0:
            self._save_request_history()
    
    def get_rate_limit_summary(self) -> Dict[str, Any]:
        """Get summary of rate limit status for all tracked services."""
        summary = {
            "services": {},
            "total_requests_last_hour": 0,
            "most_active_service": None,
            "timestamp": datetime.now().isoformat()
        }
        
        for service in self.request_history:
            if self.request_history[service]:  # Only include services with requests
                status = self.check_rate_limit(service)
                summary["services"][service] = status
                summary["total_requests_last_hour"] += status["requests_last_hour"]
        
        if summary["services"]:
            most_active = max(
                summary["services"].items(),
                key=lambda x: x[1]["requests_last_hour"]
            )
            summary["most_active_service"] = {
                "name": most_active[0],
                "requests": most_active[1]["requests_last_hour"]
            }
        
        return summary


# Global rate limiter instance
_rate_limiter = APIRateLimiter()


from .coordinate_utils import (
    get_distance_quality_rating,
    haversine_distance,
    validate_coordinates,
)

# Import crosswalk system for linked data normalization
try:
    from .crosswalk_loader import normalize_osm_natural, normalize_usda_texture
except ImportError:
    print(
        "Warning: crosswalk_loader not available - linked data normalization disabled"
    )
    normalize_usda_texture = None
    normalize_osm_natural = None


class GeospatialCache:
    """Enhanced file-based cache with dataset-aware TTLs and geohash precision."""

    # Dataset-specific TTLs (in seconds)
    DATASET_TTLS = {
        "elevation": 365 * 24 * 3600,  # 1 year - static DEM data
        "weather": 7 * 24 * 3600,  # 1 week - historical weather
        "soil_classification": 365 * 24 * 3600,  # 1 year - static taxonomy
        "soil_properties": 90 * 24 * 3600,  # 3 months - SoilGrids updates
        "ecoregion": 365 * 24 * 3600,  # 1 year - static TEOW data
        "location_context": 30 * 24 * 3600,  # 1 month - OSM/admin data
        "nearby_features": 30 * 24 * 3600,  # 1 month - OSM features
        "default": 30 * 24 * 3600,  # 1 month default
    }

    # Geohash precision levels (digits) for coordinate rounding
    GEOHASH_PRECISION = {
        "elevation": 5,  # ~2.4km precision for DEM
        "soil_properties": 5,  # ~2.4km precision for SoilGrids
        "soil_classification": 6,  # ~610m precision for taxonomy
        "weather": 4,  # ~20km precision for weather
        "ecoregion": 4,  # ~20km precision for ecoregions
        "location_context": 6,  # ~610m precision for admin
        "nearby_features": 6,  # ~610m precision for features
        "default": 5,  # ~2.4km default
    }

    # Dataset versions for cache invalidation
    DATASET_VERSIONS = {
        "soil_properties": "soilgrids_v2.0_crosswalk_v1",  # Updated for crosswalk integration
        "soil_classification": "sda_2024",
        "ecoregion": "teow_2017",
        "elevation": "srtm_v4",
        "weather": "meteostat",
        "location_context": "osm_nominatim",
        "nearby_features": "osm_overpass",
    }

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            # Default to cache in project root
            cache_dir = str(
                Path(__file__).parent.parent.parent / "cache" / "geospatial"
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _round_coordinates(self, lat: float, lon: float, precision: int) -> tuple:
        """Round coordinates to geohash precision level."""
        # Convert precision to decimal places (approximate)
        # Geohash precision 5 ≈ 4 decimal places
        decimal_places = max(1, precision - 1)
        factor = 10**decimal_places
        return (round(lat * factor) / factor, round(lon * factor) / factor)

    def _cache_key(self, api_name: str, params: Dict[str, Any]) -> str:
        """Generate versioned cache key with coordinate rounding."""
        # Apply coordinate rounding if lat/lon present
        rounded_params = params.copy()
        precision = self.GEOHASH_PRECISION.get(
            api_name, self.GEOHASH_PRECISION["default"]
        )

        if "lat" in params and "lon" in params:
            lat_rounded, lon_rounded = self._round_coordinates(
                params["lat"], params["lon"], precision
            )
            rounded_params["lat"] = lat_rounded
            rounded_params["lon"] = lon_rounded

        # Add dataset version to cache key
        version = self.DATASET_VERSIONS.get(api_name, "v1")
        rounded_params["_dataset_version"] = version

        param_str = json.dumps(rounded_params, sort_keys=True)
        key = hashlib.md5(f"{api_name}:{param_str}".encode()).hexdigest()
        return key

    def get(self, api_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result with dataset-aware TTL."""
        key = self._cache_key(api_name, params)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached_data = json.load(f)

                # Use dataset-specific TTL
                ttl = self.DATASET_TTLS.get(api_name, self.DATASET_TTLS["default"])
                cache_age = time.time() - cache_file.stat().st_mtime

                if cache_age < ttl:
                    return cached_data
                else:
                    cache_file.unlink()  # Remove expired cache
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def set(
        self, api_name: str, params: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Cache result with metadata."""
        key = self._cache_key(api_name, params)
        cache_file = self.cache_dir / f"{key}.json"

        # Add cache metadata
        current_time = time.time()
        cache_data = {
            "cached_at": current_time,
            "cached_at_iso": datetime.utcfromtimestamp(current_time).isoformat() + "Z",
            "dataset_version": self.DATASET_VERSIONS.get(api_name, "v1"),
            "ttl_seconds": self.DATASET_TTLS.get(
                api_name, self.DATASET_TTLS["default"]
            ),
            "data": result,
        }

        try:
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except OSError:
            pass  # Fail silently if can't write cache

    def _has_enhanced_metadata(self, api_name: str, data: Dict[str, Any]) -> bool:
        """Check if cached data has enhanced metadata (coordinate tracking, etc.)."""
        if api_name == "weather":
            # Check if weather data has coordinate tracking metadata
            return (
                "coordinate_distance_km" in data
                or "coordinate_metadata" in data
                or "actual_data_coordinates" in data
            )
        return True  # Other APIs don't need special enhanced metadata checks

    def get_cached_data(
        self, api_name: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached data, extracting from metadata wrapper."""
        cached = self.get(api_name, params)
        if cached and isinstance(cached, dict):
            # Handle both old format (direct data) and new format (with metadata)
            if "data" in cached:
                return cached["data"]
            else:
                return cached  # Old format compatibility
        return None


# Global cache instance
_cache = GeospatialCache()


def get_elevation_open_elevation(lat: float, lon: float) -> Dict[str, Any]:
    """Get elevation using Open Elevation API."""
    cache_params = {"lat": round(lat, 4), "lon": round(lon, 4)}

    # Check cache first (handle both old and new cache formats)
    cached = _cache.get_cached_data("elevation", cache_params)
    if cached is not None:
        return cached

    # Check rate limits before making request
    rate_status = _rate_limiter.check_rate_limit("open_elevation")
    if rate_status["recommended_delay"] > 0:
        print(f"⏳ Rate limit warning for Open Elevation API: {rate_status['usage_percentage']}% usage, waiting {rate_status['recommended_delay']}s")
        time.sleep(rate_status["recommended_delay"])

    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {"locations": f"{lat},{lon}"}

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        # Record the API request for rate limiting
        _rate_limiter.record_request("open_elevation")
        
        data = response.json()

        if data.get("results"):
            result_data = data["results"][0]
            result = {
                "elevation_meters": result_data.get("elevation"),
                "latitude": result_data.get("latitude"),
                "longitude": result_data.get("longitude"),
                "data_source": "Open Elevation API",
                "success": True,
            }
        else:
            result = {
                "elevation_meters": None,
                "success": False,
                "data_source": "Open Elevation API",
            }

        # Add rate limiting metadata
        result["rate_limit_status"] = _rate_limiter.check_rate_limit("open_elevation")
        
        # Validate and cache the result
        validated_result = validate_api_response(
            result, 
            "Open Elevation API",
            success_indicators=["elevation_meters"]
        )
        _cache.set("elevation", cache_params, validated_result)
        return validated_result

    except Exception as e:
        result = {
            "elevation_meters": None,
            "error": str(e),
            "data_source": "Open Elevation API",
            "success": False,
        }
        validated_result = validate_api_response(result, "Open Elevation API")
        _cache.set("elevation", cache_params, validated_result)
        return validated_result


def get_elevation_usgs(lat: float, lon: float, max_retries: int = 3) -> Dict[str, Any]:
    """Get elevation using USGS Elevation Point Query Service (US only)."""
    cache_params = {"lat": round(lat, 4), "lon": round(lon, 4)}

    # Check cache first - only return if it's a successful result
    cached = _cache.get("usgs_elevation", cache_params)
    if cached is not None and cached.get("success", False):
        return cached

    url = "https://epqs.nationalmap.gov/v1/json"
    params = {"x": lon, "y": lat, "units": "Meters"}
    
    last_error = None
    
    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** (attempt - 1)
                print(f"⏳ USGS retry attempt {attempt + 1}/{max_retries}, waiting {wait_time}s")
                time.sleep(wait_time)
                
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Check if response content type is JSON
            content_type = response.headers.get('content-type', '').lower()
            if 'application/json' not in content_type and 'text/json' not in content_type:
                last_error = f"Unexpected content type: {content_type}"
                if attempt < max_retries - 1:
                    continue  # Retry
                break  # Final attempt failed
            
            # Validate JSON content before parsing
            try:
                data = response.json()
            except json.JSONDecodeError as je:
                last_error = f"JSON decode error: {str(je)}"
                if attempt < max_retries - 1:
                    continue  # Retry
                break  # Final attempt failed
            
            # Process successful JSON response
            if not isinstance(data, dict):
                last_error = f"Expected JSON object, got {type(data).__name__}"
                if attempt < max_retries - 1:
                    continue  # Retry
                break  # Final attempt failed

            # v1 JSON API returns elevation directly in "value" field
            elevation_str = data.get("value")
            if elevation_str and elevation_str != "-1000000":
                try:
                    elevation_float = float(elevation_str)
                    result = {
                        "elevation_meters": elevation_float,
                        "latitude": lat,
                        "longitude": lon,
                        "data_source": "USGS Elevation Point Query Service v1",
                        "units": "meters",
                        "success": True,
                        "raw_elevation": elevation_str,
                        "attempts": attempt + 1,
                    }
                    _cache.set("usgs_elevation", cache_params, result)
                    return result
                except (ValueError, TypeError) as ve:
                    last_error = f"Could not convert elevation to float: {elevation_str}, error: {str(ve)}"
                    if attempt < max_retries - 1:
                        continue  # Retry
                    break  # Final attempt failed
            else:
                # No elevation data available - this is a valid result, not a failure
                result = {
                    "elevation_meters": None,
                    "success": False,
                    "data_source": "USGS Elevation Point Query Service v1",
                    "note": "No elevation data available (outside US bounds or water body)",
                    "raw_elevation": elevation_str,
                    "attempts": attempt + 1,
                }
                _cache.set("usgs_elevation", cache_params, result)
                return result
                
        except requests.RequestException as e:
            last_error = f"HTTP request error: {str(e)}"
            if attempt < max_retries - 1:
                continue  # Retry
            break  # Final attempt failed
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            if attempt < max_retries - 1:
                continue  # Retry
            break  # Final attempt failed
    
    # All retries failed
    result = {
        "elevation_meters": None,
        "error": f"All {max_retries} attempts failed. Last error: {last_error}",
        "data_source": "USGS Elevation Point Query Service",
        "success": False,
        "attempts": max_retries
    }
    _cache.set("usgs_elevation", cache_params, result)
    return result


def add_weather_station_distance_metadata(
    sample_lat: float,
    sample_lon: float,
    station_lat: float,
    station_lon: float,
    station_name: str = "Unknown Station",
) -> Dict[str, Any]:
    """
    Add distance metadata when weather station coordinates are known.

    Args:
        sample_lat, sample_lon: Sample coordinates
        station_lat, station_lon: Weather station coordinates
        station_name: Name/ID of the weather station

    Returns:
        Dict with distance metadata and quality assessment
    """
    distance_km = haversine_distance(sample_lat, sample_lon, station_lat, station_lon)
    quality_rating = get_distance_quality_rating(distance_km)

    return {
        "station_info": {
            "name": station_name,
            "coordinates": {"lat": station_lat, "lon": station_lon},
            "formatted_coordinates": validate_coordinates(station_lat, station_lon)[
                "metadata"
            ]["formatted"],
        },
        "sample_coordinates": {"lat": sample_lat, "lon": sample_lon},
        "distance_analysis": {
            "distance_km": round(distance_km, 2),
            "quality_rating": quality_rating,
            "representativeness": (
                "high"
                if distance_km < 25
                else "moderate" if distance_km < 100 else "low"
            ),
        },
        "data_quality_notes": [
            f"Weather data from station {distance_km:.1f}km away",
            f"Data quality: {quality_rating['rating']} ({quality_rating['description']})",
        ],
    }


def get_weather_data_open_meteo(lat: float, lon: float, date: str) -> Dict[str, Any]:
    """Get historical weather data using Open-Meteo API with multi-pass approach."""
    cache_params = {"lat": round(lat, 3), "lon": round(lon, 3), "date": date}

    # Check cache first, but only use if it has enhanced metadata
    cached = _cache.get_cached_data("weather", cache_params)
    if cached is not None and _cache._has_enhanced_metadata("weather", cached):
        return cached

    url = "https://archive-api.open-meteo.com/v1/archive"

    # Multi-pass parameter sets to avoid URL length limits and get richer data
    # PASS 1: Core temperature and precipitation (most reliable)
    pass1_params = [
        "temperature_2m_max",
        "temperature_2m_min", 
        "temperature_2m_mean",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
    ]

    # PASS 2: Wind, pressure, and humidity 
    pass2_params = [
        "wind_speed_10m_max",
        "wind_gusts_10m_max", 
        "wind_direction_10m_dominant",
        "relative_humidity_2m_mean",
        "dewpoint_2m_mean",
        "surface_pressure_mean",
        "cloudcover_mean",
    ]

    # PASS 3: Solar radiation and plant biology
    pass3_params = [
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration", 
        "sunshine_duration",
        "daylight_duration",
    ]

    # PASS 4: Advanced atmospheric and soil parameters
    pass4_params = [
        "vapour_pressure_deficit_mean",
        "soil_temperature_0cm_mean",
        "soil_moisture_0_1cm_mean",
    ]

    base_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "timezone": "UTC",
        "models": "best_match",
        "cell_selection": "nearest",
        "format": "json",
        "temperature_unit": "celsius",
        "windspeed_unit": "kmh", 
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
    }

    combined_weather_data = {}
    all_passes_metadata = []
    total_successful_params = 0
    successful_passes = 0
    returned_lat = lat
    returned_lon = lon
    returned_elevation = None
    coord_distance_km = 0

    # Execute multiple passes for comprehensive data
    parameter_sets = [
        ("Core Temperature & Precipitation", pass1_params),
        ("Wind & Atmospheric Conditions", pass2_params), 
        ("Solar & Plant Biology", pass3_params),
        ("Advanced Atmospheric & Soil", pass4_params),
    ]

    for pass_name, param_set in parameter_sets:
        try:
            params = base_params.copy()
            params["daily"] = ",".join(param_set)

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Record the API request for rate limiting
            _rate_limiter.record_request("open_meteo")
            
            data = response.json()

            pass_metadata = {
                "pass_name": pass_name,
                "parameters_requested": param_set,
                "success": False,
                "parameters_received": 0,
                "error": None
            }

            if "daily" in data and data["daily"]:
                daily = data["daily"]
                
                # Store metadata from first successful pass
                if successful_passes == 0:
                    returned_lat = data.get("latitude", lat)
                    returned_lon = data.get("longitude", lon) 
                    returned_elevation = data.get("elevation")
                    coord_distance_km = haversine_distance(lat, lon, returned_lat, returned_lon)

                # Extract daily data for this pass
                pass_weather_data = {}
                for key, values in daily.items():
                    if (
                        key != "time"
                        and values
                        and len(values) > 0
                        and values[0] is not None
                    ):
                        pass_weather_data[key] = values[0]
                        combined_weather_data[key] = values[0]

                pass_metadata["success"] = True
                pass_metadata["parameters_received"] = len(pass_weather_data)
                total_successful_params += len(pass_weather_data)
                successful_passes += 1

            all_passes_metadata.append(pass_metadata)

        except Exception as e:
            pass_metadata = {
                "pass_name": pass_name,
                "parameters_requested": param_set,
                "success": False,
                "parameters_received": 0,
                "error": str(e)
            }
            all_passes_metadata.append(pass_metadata)
            
            # Fallback strategy: For critical passes that fail, try essential parameters only
            if pass_name == "Core Temperature & Precipitation" and successful_passes == 0:
                try:
                    # Try minimal essential parameters if the core pass failed
                    fallback_params = base_params.copy()
                    fallback_params["daily"] = "temperature_2m_mean,precipitation_sum"
                    
                    response = requests.get(url, params=fallback_params, timeout=30)
                    response.raise_for_status()
                    _rate_limiter.record_request("open_meteo")
                    
                    data = response.json()
                    if "daily" in data and data["daily"]:
                        daily = data["daily"]
                        
                        # Store metadata from fallback
                        if successful_passes == 0:
                            returned_lat = data.get("latitude", lat)
                            returned_lon = data.get("longitude", lon) 
                            returned_elevation = data.get("elevation")
                            coord_distance_km = haversine_distance(lat, lon, returned_lat, returned_lon)
                        
                        fallback_weather_data = {}
                        for key, values in daily.items():
                            if key != "time" and values and len(values) > 0 and values[0] is not None:
                                fallback_weather_data[key] = values[0]
                                combined_weather_data[key] = values[0]
                        
                        fallback_metadata = {
                            "pass_name": f"{pass_name} (Fallback)",
                            "parameters_requested": ["temperature_2m_mean", "precipitation_sum"],
                            "success": True,
                            "parameters_received": len(fallback_weather_data),
                            "error": None,
                            "fallback_reason": str(e)
                        }
                        all_passes_metadata.append(fallback_metadata)
                        total_successful_params += len(fallback_weather_data)
                        successful_passes += 1
                except Exception:
                    # Even fallback failed - continue with other passes
                    pass

    # Build comprehensive result
    if successful_passes > 0:
        # Calculate coordinate metadata
        distance_quality = get_distance_quality_rating(coord_distance_km)
        coord_validation = validate_coordinates(lat, lon)
        returned_coord_validation = validate_coordinates(returned_lat, returned_lon)

        result = {
            "date": date,
            "requested_coordinates": {"latitude": lat, "longitude": lon},
            "actual_data_coordinates": {
                "latitude": returned_lat,
                "longitude": returned_lon,
            },
            "coordinate_distance_km": round(coord_distance_km, 2),
            "elevation_meters": returned_elevation,
            "weather_data": combined_weather_data,
            "multi_pass_summary": {
                "total_passes": len(parameter_sets),
                "successful_passes": successful_passes,
                "total_parameters": total_successful_params,
                "pass_details": all_passes_metadata
            },
            "data_source": "Open-Meteo Historical Weather API (Multi-Pass)",
            "coordinate_metadata": {
                "requested_validation": coord_validation,
                "actual_validation": returned_coord_validation,
                "distance_analysis": {
                    "coordinate_adjustment_km": round(coord_distance_km, 2),
                    "quality_rating": distance_quality,
                    "representativeness": (
                        "high" if coord_distance_km < 25
                        else "moderate" if coord_distance_km < 100 else "low"
                    ),
                },
                "formatted_requested": coord_validation["metadata"]["formatted"],
                "formatted_actual": returned_coord_validation["metadata"]["formatted"],
                "data_source_type": "multi_pass_grid_data",
                "spatial_resolution": "Open-Meteo grid point",
                "coordinate_note": f"Open-Meteo adjusted coordinates by {coord_distance_km:.2f}km",
            },
            "data_quality": {
                "source_type": "multi_pass_comprehensive",
                "confidence": distance_quality["rating"],
                "distance_quality_score": distance_quality["score"],
                "multi_pass_success_rate": f"{successful_passes}/{len(parameter_sets)}",
                "notes": [
                    f"Multi-pass approach: {successful_passes}/{len(parameter_sets)} passes successful",
                    f"Total parameters retrieved: {total_successful_params}",
                    f"Coordinate adjustment: {coord_distance_km:.2f}km",
                ],
            },
            "success": True,
        }
    else:
        result = {
            "weather_data": {},
            "multi_pass_summary": {
                "total_passes": len(parameter_sets),
                "successful_passes": 0,
                "pass_details": all_passes_metadata
            },
            "error": "All passes failed",
            "data_source": "Open-Meteo Historical Weather API (Multi-Pass)",
            "success": False,
        }

    # Cache the result
    _cache.set("weather", cache_params, result)
    return result


def get_reverse_geocoding_nominatim(lat: float, lon: float) -> Dict[str, Any]:
    """Get location information using Nominatim reverse geocoding."""
    cache_params = {"lat": round(lat, 3), "lon": round(lon, 3)}

    # Check cache first
    cached = _cache.get("nominatim", cache_params)
    if cached is not None:
        return cached

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json", "addressdetails": 1, "zoom": 18}

    headers = {"User-Agent": "BiosampleEnrichment/1.0 (research purposes)"}

    try:
        # Be respectful with rate limiting
        time.sleep(1.1)  # Nominatim requires max 1 request/second

        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Record the API request for rate limiting
        _rate_limiter.record_request("nominatim")
        
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
                "city": address.get("city")
                or address.get("town")
                or address.get("village"),
                "postcode": address.get("postcode"),
                "latitude": float(data.get("lat", lat)),
                "longitude": float(data.get("lon", lon)),
                "data_source": "OpenStreetMap Nominatim",
                "success": True,
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
            "success": False,
        }
        _cache.set("nominatim", cache_params, result)
        return result


# Import comprehensive OSM tag structure from src directory
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from crawl_first.osm import OSM_ENVIRONMENTAL_TAGS

    COMPREHENSIVE_OSM_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_OSM_AVAILABLE = False
    # Fallback minimal tags if comprehensive structure unavailable
    OSM_ENVIRONMENTAL_TAGS = {
        "natural": {"water", "wood", "grassland", "wetland", "hot_spring"},
        "landuse": {"forest", "farmland", "residential", "commercial", "industrial"},
        "water": {"lake", "pond", "river", "stream"},
        "waterway": {"river", "stream", "canal"},
    }


def build_comprehensive_overpass_query(lat: float, lon: float, radius_m: int) -> str:
    """Build comprehensive Overpass query using all environmental tag categories."""
    query_parts = []

    # Add queries for each tag category
    for category, values in OSM_ENVIRONMENTAL_TAGS.items():
        if values:  # Only add if category has values
            for value in values:
                # Add both ways and relations for comprehensive coverage
                query_parts.append(
                    f'way["{category}"="{value}"](around:{radius_m},{lat},{lon});'
                )
                query_parts.append(
                    f'relation["{category}"="{value}"](around:{radius_m},{lat},{lon});'
                )
                # Add nodes for point features (especially for natural features)
                if category in ["natural", "man_made", "amenity"]:
                    query_parts.append(
                        f'node["{category}"="{value}"](around:{radius_m},{lat},{lon});'
                    )

    query = f"""
    [out:json][timeout:60];
    (
      {' '.join(query_parts)}
    );
    out body center qt;
    """

    return query


def get_nearby_features_overpass(
    lat: float, lon: float, radius_km: float = 1.0
) -> Dict[str, Any]:
    """Get nearby geographic features using Overpass API."""
    cache_params = {"lat": round(lat, 3), "lon": round(lon, 3), "radius_km": radius_km}

    # Check cache first
    cached = _cache.get("overpass", cache_params)
    if cached is not None:
        return cached

    url = "https://overpass-api.de/api/interpreter"

    # Build comprehensive query using all environmental tag categories
    radius_m = int(radius_km * 1000)
    query = build_comprehensive_overpass_query(lat, lon, radius_m)

    try:
        response = requests.post(
            url, data=query, timeout=120
        )  # Increased timeout for comprehensive query
        response.raise_for_status()
        
        # Record the API request for rate limiting
        _rate_limiter.record_request("overpass")
        
        data = response.json()

        features = []
        feature_categories = {}

        if "elements" in data:
            # Process up to 50 features for more comprehensive coverage
            for element in data["elements"][:50]:
                tags = element.get("tags", {})
                if tags:
                    # Extract primary feature category and type
                    primary_category = None
                    primary_type = None

                    # Check all OSM environmental tag categories to determine primary classification
                    for category in OSM_ENVIRONMENTAL_TAGS.keys():
                        if (
                            category in tags
                            and tags[category] in OSM_ENVIRONMENTAL_TAGS[category]
                        ):
                            primary_category = category
                            primary_type = tags[category]
                            break

                    feature = {
                        "type": element.get("type"),
                        "id": element.get("id"),
                        "tags": tags,
                        "primary_category": primary_category,
                        "primary_type": primary_type,
                        "name": tags.get("name"),
                        # Keep specific common categories for backward compatibility
                        "natural": tags.get("natural"),
                        "landuse": tags.get("landuse"),
                        "water": tags.get("water"),
                        "waterway": tags.get("waterway"),
                        "wetland": tags.get("wetland"),
                        "leisure": tags.get("leisure"),
                        "man_made": tags.get("man_made"),
                        "building": tags.get("building"),
                        "amenity": tags.get("amenity"),
                    }
                    features.append(feature)

                    # Track feature categories for summary
                    if primary_category:
                        if primary_category not in feature_categories:
                            feature_categories[primary_category] = {}
                        if primary_type not in feature_categories[primary_category]:
                            feature_categories[primary_category][primary_type] = 0
                        feature_categories[primary_category][primary_type] += 1

        result = {
            "latitude": lat,
            "longitude": lon,
            "radius_km": radius_km,
            "features_found": len(features),
            "features": features,
            "feature_categories": feature_categories,
            "comprehensive_query": COMPREHENSIVE_OSM_AVAILABLE,
            "total_tag_categories": len(OSM_ENVIRONMENTAL_TAGS),
            "query_metadata": {
                "osm_tags_queried": list(OSM_ENVIRONMENTAL_TAGS.keys()),
                "max_features_returned": min(50, len(data.get("elements", []))),
                "timeout_seconds": 120,
                "query_type": (
                    "comprehensive_environmental"
                    if COMPREHENSIVE_OSM_AVAILABLE
                    else "basic_fallback"
                ),
            },
            "data_source": "OpenStreetMap Overpass API",
            "success": True,
        }

        # Cache the result
        _cache.set("overpass", cache_params, result)
        return result

    except Exception as e:
        result = {
            "error": str(e),
            "features": [],
            "data_source": "OpenStreetMap Overpass API",
            "success": False,
        }
        _cache.set("overpass", cache_params, result)
        return result


def get_soil_classification_nrcs_sda(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get US soil classification from USDA NRCS Soil Data Access (SDA).

    Source: https://SDMDataAccess.sc.egov.usda.gov/
    Approach: Schema-safe POST queries to get mapunit key, then soil taxonomy
    Coverage: US only (CONUS + territories)

    Args:
        lat: Latitude (must be within US bounds)
        lon: Longitude (must be within US bounds)

    Returns:
        Dict with USDA soil classification data using schema-safe SDA API
    """
    # Check US bounds (approximate)
    if not (-170 <= lon <= -60 and 15 <= lat <= 75):
        return {
            "soil_classification": None,
            "error": "Coordinates outside US bounds",
            "data_source": "USDA NRCS SDA",
            "success": False,
        }

    cache_params = {"lat": round(lat, 4), "lon": round(lon, 4)}

    # Check cache first
    cached = _cache.get("nrcs_sda", cache_params)
    if cached is not None:
        return cached

    SDA_URL = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"
    HDRS = {"User-Agent": "geospatial-enrichment/1.0", "Accept": "application/json"}

    try:
        # Step 0: Get available columns for component table (schema-safe)
        q_schema = "SELECT * FROM SDA_GetTabularSchema('component')"
        r_schema = requests.post(
            SDA_URL,
            data={
                "SERVICE": "query",
                "REQUEST": "query",
                "FORMAT": "JSON",
                "QUERY": q_schema,
            },
            headers=HDRS,
            timeout=30,
        )

        available_columns = set()
        if r_schema.status_code == 200:
            schema_data = r_schema.json()
            if schema_data.get("Table"):
                # Extract column names from schema (typically in format: [table, column, type, ...])
                available_columns = {
                    row[1].lower() for row in schema_data["Table"] if len(row) > 1
                }

        # Taxonomy columns we want to use (fallback to safe defaults if schema query fails)
        desired_tax_cols = ["taxorder", "taxsuborder", "taxsubgrp"]
        safe_tax_cols = (
            [col for col in desired_tax_cols if col in available_columns]
            if available_columns
            else ["taxorder", "taxsuborder", "taxsubgrp"]
        )

        # Step 1: Get mapunit key(s) at point (WGS84)
        wkt = f"POINT({lon} {lat})"
        q1 = f"SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt}')"
        r1 = requests.post(
            SDA_URL,
            data={
                "SERVICE": "query",
                "REQUEST": "query",
                "FORMAT": "JSON",
                "QUERY": q1,
            },
            headers=HDRS,
            timeout=30,
        )
        r1.raise_for_status()
        
        # Record the API request for rate limiting
        _rate_limiter.record_request("nrcs_sda")
        
        data = r1.json()

        # SDA returns format: {'Table': [['mukey1'], ['mukey2'], ...]}
        if not data or "Table" not in data or not data["Table"]:
            result = {
                "soil_classification": None,
                "error": "No SSURGO mapunit at point",
                "data_source": "USDA NRCS SDA",
                "success": False,
            }
            _cache.set("nrcs_sda", cache_params, result)
            return result

        # Extract mukeys from table format
        mukeys = [str(row[0]) for row in data["Table"]]

        # Step 2: Get the most representative component using available columns
        base_cols = ["mukey", "cokey", "compname", "comppct_r"]
        tax_cols_sql = ", ".join([f"c.{col}" for col in safe_tax_cols])

        import textwrap

        q2 = textwrap.dedent(
            f"""
          SELECT TOP 1
            c.mukey, c.cokey, c.compname, c.comppct_r,
            {tax_cols_sql}
          FROM component c
          WHERE c.mukey IN ({",".join(mukeys)})
          ORDER BY c.comppct_r DESC
        """
        ).strip()

        r2 = requests.post(
            SDA_URL,
            data={
                "SERVICE": "query",
                "REQUEST": "query",
                "FORMAT": "JSON",
                "QUERY": q2,
            },
            headers=HDRS,
            timeout=30,
        )
        r2.raise_for_status()
        comp_data = r2.json()

        # Extract first row from Table format
        if comp_data and "Table" in comp_data and comp_data["Table"]:
            row = comp_data["Table"][0]

            # Build classification dict with available data
            classification = {
                "mukeys": mukeys,
                "component_key": row[1],
                "component_name": row[2],
                "component_percent": row[3],
                "schema_columns_used": safe_tax_cols,
            }

            # Add taxonomy data based on available columns
            for i, col in enumerate(safe_tax_cols):
                if i + 4 < len(row):  # +4 to skip base columns
                    classification[f"taxonomic_{col.replace('tax', '')}"] = row[i + 4]

            result = {
                "soil_classification": classification,
                "data_source": "USDA NRCS SDA",
                "taxonomy_system": "USDA Soil Taxonomy",
                "documentation": "https://sdmdataaccess.sc.egov.usda.gov/",
                "schema_safe": True,
                "available_columns": (
                    len(available_columns)
                    if available_columns
                    else "schema_query_failed"
                ),
                "success": True,
            }
        else:
            result = {
                "soil_classification": {"mukeys": mukeys},
                "error": "Mapunit found but no component data available",
                "data_source": "USDA NRCS SDA",
                "success": False,
            }

        # Cache the result
        _cache.set("nrcs_sda", cache_params, result)
        return result

    except requests.RequestException as e:
        result = {
            "soil_classification": None,
            "error": f"Network error: {str(e)}",
            "data_source": "USDA NRCS SDA",
            "success": False,
        }
        _cache.set("nrcs_sda", cache_params, result)
        return result
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        result = {
            "soil_classification": None,
            "error": f"Parse error: {str(e)}",
            "data_source": "USDA NRCS SDA",
            "success": False,
        }
        _cache.set("nrcs_sda", cache_params, result)
        return result


def get_soil_classification_isric_soilgrids(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get global soil classification from ISRIC SoilGrids v2.0.

    Source: https://rest.isric.org/soilgrids/v2.0/
    Approach: REST API query for point-based soil classification
    Coverage: Global (250m resolution)

    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)

    Returns:
        Dict with ISRIC soil classification data including WRB classes
    """
    cache_params = {"lat": round(lat, 4), "lon": round(lon, 4)}

    # Check cache first
    cached = _cache.get("isric_soilgrids", cache_params)
    if cached is not None:
        return cached

    url = "https://rest.isric.org/soilgrids/v2.0/classification/query"
    params = {
        "lon": lon,
        "lat": lat,
        "number_classes": 3,  # Top 3 most probable soil classes
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        # Record the API request for rate limiting
        _rate_limiter.record_request("isric_soilgrids")
        
        data = response.json()

        if data.get("wrb_class_name"):
            # Extract soil classification data
            result = {
                "soil_classification": {
                    "primary_class": data.get("wrb_class_name"),
                    "primary_class_code": data.get("wrb_class_value"),
                    "primary_probability": (
                        data.get("wrb_class_probability", [{}])[0][1]
                        if data.get("wrb_class_probability")
                        else None
                    ),
                    "alternative_classes": data.get("wrb_class_probability", []),
                    "coordinates_queried": data.get("coordinates", [lon, lat]),
                },
                "data_source": "ISRIC SoilGrids v2.0",
                "classification_system": "WRB (World Reference Base)",
                "resolution": "250m",
                "documentation": "https://rest.isric.org/soilgrids/v2.0/",
                "success": True,
            }
        else:
            result = {
                "soil_classification": None,
                "error": "No soil classification data available",
                "data_source": "ISRIC SoilGrids v2.0",
                "success": False,
            }

        # Cache the result
        _cache.set("isric_soilgrids", cache_params, result)
        return result

    except requests.RequestException as e:
        result = {
            "soil_classification": None,
            "error": f"Network error: {str(e)}",
            "data_source": "ISRIC SoilGrids v2.0",
            "success": False,
        }
        _cache.set("isric_soilgrids", cache_params, result)
        return result
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        result = {
            "soil_classification": None,
            "error": f"Parse error: {str(e)}",
            "data_source": "ISRIC SoilGrids v2.0",
            "success": False,
        }
        _cache.set("isric_soilgrids", cache_params, result)
        return result


def soilgrids_wcs_point(
    lon: float,
    lat: float,
    coverage_id: str = "phh2o_0-5cm_Q0.5",
    map_path: str = "/map/phh2o.map",
    timeout: int = 30,
) -> Optional[bytes]:
    """
    Get single pixel from ISRIC SoilGrids WCS with dual-path fallback.

    Args:
        lon: Longitude
        lat: Latitude
        coverage_id: WCS coverage ID
        map_path: ISRIC map path
        timeout: Request timeout

    Returns:
        GeoTIFF bytes or None if failed
    """
    base_url = "https://maps.isric.org/mapserv"

    # Try WCS 2.0.1 first (modern, subset params)
    try:
        url = f"{base_url}?map={map_path}&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID={coverage_id}&FORMAT=image/tiff&subset=Long({lon},{lon})&subset=Lat({lat},{lat})"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    except Exception:
        # Fallback: WCS 1.0.0 with small bounding box (more reliable than single point)
        try:
            # Create small bounding box around the point (~500m)
            buffer = 0.0025  # ~0.25km radius
            min_lon, min_lat = lon - buffer, lat - buffer
            max_lon, max_lat = lon + buffer, lat + buffer
            bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"

            url = f"{base_url}?map={map_path}&SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage&COVERAGE={coverage_id}&FORMAT=GEOTIFF_INT16&CRS=EPSG:4326&BBOX={bbox}&WIDTH=3&HEIGHT=3"
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception:
            return None


def read_single_pixel_from_tiff_bytes(tif_bytes: bytes) -> Optional[float]:
    """
    Extract single pixel value from GeoTIFF bytes.
    Handles both single pixel and small arrays by extracting center pixel.

    Args:
        tif_bytes: GeoTIFF data as bytes

    Returns:
        Pixel value or None if no data/error
    """
    try:
        # Try to use rasterio if available
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(tif_bytes) as mem:
            with mem.open() as ds:
                arr = ds.read(1)  # Read band 1
                nodata = ds.nodata

                if arr.size > 0:
                    # For multi-pixel arrays, get center pixel
                    if arr.shape[0] > 1 or arr.shape[1] > 1:
                        center_y, center_x = arr.shape[0] // 2, arr.shape[1] // 2
                        val = float(arr[center_y, center_x])
                    else:
                        # Single pixel
                        val = float(arr.flat[0])

                    # Check for no-data values
                    if nodata is not None and val == nodata:
                        return None
                    # Additional check for common no-data values in SoilGrids
                    if val == -32768 or val == 32767:
                        return None

                    return val
        return None
    except ImportError:
        # Fallback: return metadata indicating GeoTIFF was retrieved
        return len(tif_bytes) if tif_bytes else None
    except Exception:
        return None


def usda_texture_class_from_percentages(sand_pct: float, clay_pct: float) -> str:
    """
    Classify USDA soil texture from sand and clay percentages.

    Uses the USDA soil texture triangle classification system.
    Silt percentage is calculated as: silt = 100 - sand - clay

    Args:
        sand_pct: Sand percentage (0-100)
        clay_pct: Clay percentage (0-100)

    Returns:
        USDA texture class name
    """
    # Calculate silt percentage
    silt_pct = 100 - sand_pct - clay_pct

    # Ensure percentages are valid
    if silt_pct < 0 or sand_pct < 0 or clay_pct < 0:
        return "Invalid"

    # USDA texture triangle classification rules
    if clay_pct >= 40:
        return "Clay"
    elif clay_pct >= 27:
        if sand_pct >= 45:
            return "Sandy clay"
        elif sand_pct >= 20:
            return "Clay loam"
        else:
            return "Silty clay"
    elif clay_pct >= 20:
        if sand_pct >= 45:
            return "Sandy clay loam"
        elif silt_pct >= 28:
            return "Silty clay loam"
        else:
            return "Clay loam"
    elif silt_pct >= 80:
        return "Silt"
    elif silt_pct >= 50:
        if sand_pct >= 20:
            return "Silt loam"
        else:
            return "Silt"
    elif sand_pct >= 85:
        return "Sand"
    elif sand_pct >= 70:
        return "Loamy sand"
    elif sand_pct >= 50:
        if silt_pct >= 28:
            return "Silt loam"
        else:
            return "Sandy loam"
    else:
        return "Loam"


def get_soil_properties_isric_wcs(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get soil properties from ISRIC SoilGrids using enhanced WCS.

    Source: https://maps.isric.org/mapserv
    Approach: WCS 2.0.1 → 1.0.0 fallback with actual pixel extraction
    Coverage: Global (250m resolution)

    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)

    Returns:
        Dict with actual soil property values from SoilGrids WCS
    """
    cache_params = {"lat": round(lat, 4), "lon": round(lon, 4)}

    # Check cache first
    cached = _cache.get_cached_data("soil_properties", cache_params)
    if cached is not None:
        return cached

    # Land/water gating: Check elevation to avoid expensive soil API calls on water
    # This is a performance optimization to avoid unnecessary WCS requests
    try:
        elevation_data = get_elevation_open_elevation(lat, lon)
        if elevation_data.get("success") and elevation_data.get("elevation_meters") is not None:
            elevation = elevation_data["elevation_meters"]
            # If significantly below sea level (< -10m), likely water body - skip soil queries
            if elevation < -10:
                result = {
                    "soil_properties": {},
                    "properties_retrieved": 0,
                    "total_attempted": 0,
                    "data_source": "ISRIC SoilGrids WCS (skipped - water body)",
                    "gating_reason": f"Elevation {elevation}m below sea level indicates water body",
                    "elevation_source": elevation_data.get("data_source", "elevation API"),
                    "success": False,
                    "skip_reason": "land_water_gating"
                }
                _cache.set("soil_properties", cache_params, result)
                return result
    except Exception:
        # If elevation check fails, proceed with soil queries (fail open)
        pass

    properties = {}

    # Define soil properties to extract with correct ISRIC scaling factors
    # Reference: https://data.isric.org/geonetwork/srv/eng/catalog.search#/metadata/
    soil_maps = [
        {
            "name": "ph_0_5cm",
            "map": "/map/phh2o.map",
            "coverage": "phh2o_0-5cm_Q0.5",
            "scale": 0.1,
            "unit": "pH",
        },  # pH×10 → pH
        {
            "name": "soc_0_5cm",
            "map": "/map/soc.map",
            "coverage": "soc_0-5cm_mean",
            "scale": 0.1,
            "unit": "g/kg",
        },  # dg/kg → g/kg
        {
            "name": "sand_0_5cm",
            "map": "/map/sand.map",
            "coverage": "sand_0-5cm_mean",
            "scale": 0.1,
            "unit": "%",
        },  # g/kg → %
        {
            "name": "silt_0_5cm",
            "map": "/map/silt.map",
            "coverage": "silt_0-5cm_mean",
            "scale": 0.1,
            "unit": "%",
        },  # g/kg → %
        {
            "name": "clay_0_5cm",
            "map": "/map/clay.map",
            "coverage": "clay_0-5cm_mean",
            "scale": 0.1,
            "unit": "%",
        },  # g/kg → %
        {
            "name": "bdod_0_5cm",
            "map": "/map/bdod.map",
            "coverage": "bdod_0-5cm_mean",
            "scale": 0.01,
            "unit": "g/cm³",
        },  # cg/cm³ → g/cm³
        {
            "name": "nitrogen_0_5cm",
            "map": "/map/nitrogen.map",
            "coverage": "nitrogen_0-5cm_mean",
            "scale": 0.01,
            "unit": "g/kg",
        },  # cg/kg → g/kg
        {
            "name": "ocd_0_5cm",
            "map": "/map/ocd.map",
            "coverage": "ocd_0-5cm_mean",
            "scale": 0.1,
            "unit": "kg/dm³",
        },  # hg/dm³ → kg/dm³
        {
            "name": "ocs_0_30cm",
            "map": "/map/ocs.map",
            "coverage": "ocs_0-30cm_mean",
            "scale": 0.1,
            "unit": "kg/m²",
        },  # t/ha×10 → kg/m²
    ]

    success_count = 0
    sand_value = None
    silt_value = None
    clay_value = None

    for prop in soil_maps:
        try:
            tif_bytes = soilgrids_wcs_point(lon, lat, prop["coverage"], prop["map"])
            if tif_bytes:
                pixel_value = read_single_pixel_from_tiff_bytes(tif_bytes)
                if pixel_value is not None:
                    # Apply scaling based on ISRIC unit conversions
                    actual_value = pixel_value * prop["scale"]
                    properties[prop["name"]] = {
                        "value": actual_value,
                        "unit": prop["unit"],
                        "raw_pixel": pixel_value,
                        "coverage_id": prop["coverage"],
                        "mapfile": prop["map"],
                        "wcs_version": "2.0.1 → 1.0.0 fallback",
                    }
                    success_count += 1

                    # Store texture values for USDA classification
                    if prop["name"] == "sand_0_5cm":
                        sand_value = actual_value
                    elif prop["name"] == "silt_0_5cm":
                        silt_value = actual_value
                    elif prop["name"] == "clay_0_5cm":
                        clay_value = actual_value
                else:
                    properties[prop["name"]] = {
                        "value": None,
                        "unit": prop["unit"],
                        "error": "No data pixel",
                    }
            else:
                properties[prop["name"]] = {
                    "value": None,
                    "unit": prop["unit"],
                    "error": "WCS request failed",
                }
        except Exception as e:
            properties[prop["name"]] = {
                "value": None,
                "unit": prop["unit"],
                "error": str(e),
            }

    # Add USDA texture classification using measured sand, silt, clay values
    if sand_value is not None and clay_value is not None:
        try:
            # Use measured silt if available, otherwise calculate
            calculated_silt = (
                100 - sand_value - clay_value if silt_value is None else silt_value
            )
            texture_class = usda_texture_class_from_percentages(sand_value, clay_value)

            # Get crosswalk-based normalization
            crosswalk_result = None
            if normalize_usda_texture:
                crosswalk_result = normalize_usda_texture(
                    texture_class, sand_value, clay_value, calculated_silt
                )

            texture_classification = {
                "usda_texture_class": texture_class,
                "sand_percent": sand_value,
                "clay_percent": clay_value,
                "silt_percent": calculated_silt,
                "silt_source": "measured" if silt_value is not None else "calculated",
                "classification_system": "USDA Soil Texture Triangle",
                "data_source": "ISRIC SoilGrids WCS",
            }

            # Add crosswalk normalization if available
            if crosswalk_result:
                texture_classification["raw_classification"] = crosswalk_result[
                    "raw_classification"
                ]
                texture_classification["linked_data"] = crosswalk_result["linked_data"]

            properties["texture_classification"] = texture_classification
        except Exception as e:
            properties["texture_classification"] = {
                "error": f"Texture classification failed: {str(e)}",
                "classification_system": "USDA Soil Texture Triangle",
            }

    # Determine success status based on data availability
    if success_count > 0:
        success_status = True
        data_quality = "complete"
    elif success_count == 0 and any("No data pixel" in prop.get("error", "") for prop in properties.values() if isinstance(prop, dict)):
        # Partial success: API worked but hit water/nodata pixels
        success_status = True
        data_quality = "partial"
    else:
        # Complete failure: API errors or request failures
        success_status = False
        data_quality = "failed"

    result = {
        "soil_properties": properties,
        "properties_retrieved": success_count,
        "total_attempted": len(soil_maps),
        "data_source": "ISRIC SoilGrids WCS",
        "wcs_versions": "2.0.1 → 1.0.0 fallback",
        "resolution": "250m",
        "documentation": "https://maps.isric.org/",
        "success": success_status,
        "data_quality": data_quality,
        "note": f"Retrieved {success_count}/{len(soil_maps)} soil properties" + (
            " - no data pixels indicate water/rock surface" if data_quality == "partial" else ""
        )
    }

    # Cache the result
    _cache.set("soil_properties", cache_params, result)
    return result


# Global TEOW variables for lazy loading
_teow_gdf = None
_teow_cols = None


def get_local_ecoregion(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get ecoregion and biome information using local TEOW 2017 data.

    Source: RESOLVE/TEOW 2017 (local shapefile)
    Coverage: Global terrestrial ecoregions
    Response time: ~0.001s (instant with spatial index)
    Cache effectiveness: Not needed - local lookup is already fast

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dict with ecoregion and biome information
    """
    global _teow_gdf, _teow_cols

    # Lazy load TEOW data on first use
    if _teow_gdf is None or _teow_cols is None:
        try:
            from .download_ecoregions import setup_local_ecoregions

            result = setup_local_ecoregions()
            if result:
                _teow_gdf, _teow_cols = result
            else:
                return {
                    "success": False,
                    "ecoregion_name": None,
                    "error": "Failed to load local TEOW 2017 data",
                    "data_source": "TEOW 2017 (RESOLVE)",
                    "license": "CC-BY 4.0",
                }
        except Exception as e:
            return {
                "success": False,
                "ecoregion_name": None,
                "error": f"Failed to import TEOW setup: {e}",
                "data_source": "TEOW 2017 (RESOLVE)",
                "license": "CC-BY 4.0",
            }

    # Perform local lookup
    try:
        from .download_ecoregions import teow_lookup_point

        result = teow_lookup_point(lon, lat, _teow_gdf, _teow_cols)

        if result:
            return result
        else:
            return {
                "success": False,
                "ecoregion_name": None,
                "error": "No ecoregion found at coordinates",
                "data_source": "TEOW 2017 (RESOLVE)",
                "license": "CC-BY 4.0",
            }

    except Exception as e:
        return {
            "success": False,
            "ecoregion_name": None,
            "error": f"Local lookup failed: {e}",
            "data_source": "TEOW 2017 (RESOLVE)",
            "license": "CC-BY 4.0",
        }


def get_wwf_ecoregion(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get WWF/RESOLVE Terrestrial Ecoregion using ArcGIS FeatureServer.

    Source: https://hub.arcgis.com/datasets/resolve::ecoregions-2017
    Coverage: Global terrestrial ecoregions

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dict with WWF ecoregion and biome information
    """
    cache_params = {"lat": round(lat, 3), "lon": round(lon, 3)}

    # Check cache first
    cached = _cache.get("wwf_ecoregions", cache_params)
    if cached is not None:
        return cached

    # WWF/RESOLVE Ecoregions 2017 FeatureServer
    url = "https://services3.arcgis.com/t6lYS2Pmd8iVx1fy/arcgis/rest/services/Ecoregions2017/FeatureServer/0/query"

    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "ECO_NAME,BIOME_NAME,REALM",
        "returnGeometry": "false",
        "f": "json",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("features") and len(data["features"]) > 0:
            attrs = data["features"][0]["attributes"]
            result = {
                "ecoregion_name": attrs.get("ECO_NAME"),
                "biome_name": attrs.get("BIOME_NAME"),
                "realm": attrs.get("REALM"),
                "data_source": "WWF/RESOLVE Terrestrial Ecoregions 2017",
                "success": True,
            }
        else:
            result = {
                "ecoregion_name": None,
                "error": "No ecoregion found at coordinates",
                "data_source": "WWF/RESOLVE Terrestrial Ecoregions 2017",
                "success": False,
            }

        # Cache the result
        _cache.set("wwf_ecoregions", cache_params, result)
        return result

    except Exception as e:
        result = {
            "ecoregion_name": None,
            "error": str(e),
            "data_source": "WWF/RESOLVE Terrestrial Ecoregions 2017",
            "success": False,
        }
        _cache.set("wwf_ecoregions", cache_params, result)
        return result


def classify_ecosystem_v0(
    lat: float,
    lon: float,
    elevation_data: Dict[str, Any],
    ecoregion_data: Dict[str, Any],
    nearby_features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Rules-based ecosystem classifier v0 for GOLD ecosystem path routing.

    Classifies samples into: terrestrial, inland_freshwater, nearshore, oceanic, host_associated
    Based on GPT5's specification for reproducible ecosystem routing.

    Args:
        lat: Latitude
        lon: Longitude
        elevation_data: Elevation information
        ecoregion_data: Ecoregion classification
        nearby_features: OSM/geographic features

    Returns:
        Dict with ecosystem classification and decision features
    """
    features = {
        "latitude": lat,
        "longitude": lon,
        "decision_timestamp": datetime.utcnow().isoformat() + "Z",
        "classifier_version": "v0_rules_only",
    }

    # Feature 1: Distance to coast (approximate using longitude bands)
    # Rough heuristic: closer to 0° or ±180° longitude = potentially closer to ocean
    lon_from_dateline = min(abs(lon), abs(180 - abs(lon)))
    features["longitude_from_dateline"] = lon_from_dateline

    # Feature 2: Bathymetry/elevation sign
    elevation_m = elevation_data.get("elevation_meters")
    features["elevation_meters"] = elevation_m
    features["is_below_sea_level"] = elevation_m is not None and elevation_m < 0
    features["is_near_sea_level"] = elevation_m is not None and abs(elevation_m) < 10

    # Feature 3: Ecoregion realm
    realm = ecoregion_data.get("realm")
    features["ecoregion_realm"] = realm
    features["is_marine_realm"] = (
        realm and "marine" in realm.lower() if realm else False
    )

    # Feature 4: Water proximity from nearby features (improved detection)
    water_features = []
    freshwater_indicators = []

    if nearby_features.get("success") and nearby_features.get("features"):
        for feature in nearby_features["features"]:
            # Check all relevant OSM tags for water features
            natural_tag = feature.get("natural", "").lower()
            water_tag = feature.get("water", "").lower()
            waterway_tag = feature.get("waterway", "").lower()
            leisure_tag = feature.get("leisure", "").lower()
            name = feature.get("name", "").lower()

            # Water feature detection
            water_indicators = [
                "water",
                "river",
                "lake",
                "stream",
                "pond",
                "reservoir",
                "spring",
            ]
            
            # Check natural tag
            if any(indicator in natural_tag for indicator in water_indicators):
                water_features.append(f"natural={natural_tag}")

            # Check waterway tag (major OSM category for water features)
            waterway_indicators = ["river", "stream", "canal", "drain", "ditch"]
            if waterway_tag and waterway_tag in waterway_indicators:
                water_features.append(f"waterway={waterway_tag}")

            # Check leisure tag for water-related features
            if leisure_tag in ["swimming_pool", "marina"]:
                water_features.append(f"leisure={leisure_tag}")
            
            # Check water tag explicitly (important for accurate classification)
            if water_tag in ["lake", "pond", "reservoir", "river", "stream"]:
                water_features.append(f"water={water_tag}")
            
            # Check for marine/saltwater indicators
            marine_indicators = ["bay", "ocean", "sea", "marine", "coast", "beach"]
            if (any(indicator in name for indicator in marine_indicators) or 
                any(indicator in natural_tag for indicator in marine_indicators)):
                water_features.append(f"marine_indicator=detected")

            # Freshwater detection - lakes, rivers, streams, springs are typically freshwater
            freshwater_indicators_list = [
                "lake",
                "river",
                "stream",
                "pond",
                "spring",
                "hot_spring",
            ]
            
            # Check natural tag for freshwater
            if any(
                indicator in natural_tag for indicator in freshwater_indicators_list
            ):
                freshwater_indicators.append(f"natural={natural_tag}")

            # Check waterway tag for freshwater (most waterways are freshwater)
            if waterway_tag in ["river", "stream", "canal", "drain"]:
                freshwater_indicators.append(f"waterway={waterway_tag}")

            # Also check water tag
            if water_tag in ["lake", "pond", "reservoir", "river"]:
                freshwater_indicators.append(f"water={water_tag}")

    features["nearby_water_features"] = water_features
    features["freshwater_features"] = freshwater_indicators
    features["has_nearby_water"] = len(water_features) > 0
    features["freshwater_nearby"] = len(freshwater_indicators) > 0
    
    # Enhanced marine detection
    marine_features = [f for f in water_features if "marine_indicator" in f or "marina" in f]
    features["marine_indicators"] = marine_features
    features["has_marine_indicators"] = len(marine_features) > 0

    # Feature 5: Country EEZ (simplified using lat/lon bounds)
    # This is a placeholder - in production would use actual EEZ shapefiles
    features["in_potential_eez"] = (
        lat > -60 and lat < 80
    ) and (  # Exclude Antarctica and extreme Arctic
        abs(lat) < 75
    )  # Exclude most polar regions

    # Rules-based classification
    ecosystem_path = "unknown"
    confidence = 0.0
    reasoning = []

    # Rule 1: Host-associated (placeholder - would need sample metadata)
    # For now, we can't determine this from geographic features alone

    # Rule 2: Oceanic
    if features["is_below_sea_level"] or (
        features["is_near_sea_level"] and features["is_marine_realm"]
    ):
        ecosystem_path = "oceanic"
        confidence = 0.8
        reasoning.append("Below sea level or near sea level with marine realm")

    # Rule 3: Nearshore (near water but above sea level)
    elif (
        features["is_near_sea_level"]
        and features["has_nearby_water"]
        and (features["has_marine_indicators"] or not features["freshwater_nearby"])
    ):
        ecosystem_path = "nearshore"
        confidence = 0.8 if features["has_marine_indicators"] else 0.7
        reasoning.append("Near sea level with nearby water features" + 
                        (" with marine indicators" if features["has_marine_indicators"] else " (non-freshwater)"))

    # Rule 4: Inland freshwater
    elif features["has_nearby_water"] and features["freshwater_nearby"]:
        ecosystem_path = "inland_freshwater"
        confidence = 0.7
        reasoning.append("Freshwater features detected nearby")

    # Rule 5: Terrestrial (default for land areas)
    elif elevation_m is not None and elevation_m > 10:
        ecosystem_path = "terrestrial"
        confidence = 0.6
        reasoning.append("Elevated land area without freshwater features")

    # Rule 6: Default terrestrial
    else:
        ecosystem_path = "terrestrial"
        confidence = 0.3
        reasoning.append(
            "Default classification - insufficient features for higher confidence"
        )

    return {
        "ecosystem_path": ecosystem_path,
        "confidence": confidence,
        "reasoning": reasoning,
        "decision_features": features,
        "classifier_version": "v0_rules_only",
        "note": "Rules-only classifier - LLM adjudicator will be added in v1",
        "success": True,
    }


def enrich_location(
    lat: float, lon: float, collection_date: Optional[str] = None
) -> Dict[str, Any]:
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
        "enrichment_timestamp": datetime.utcnow().isoformat() + "Z",
        "elevation": {},
        "weather": {},
        "location_context": {},
        "nearby_features": {},
        "soil_classification": {},
        "soil_properties": {},
        "ecoregion": {},
        "enrichment_summary": {},
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

    # 5. Get soil classification (prefer NRCS SDA for US locations)
    if -170 <= lon <= -60 and 15 <= lat <= 75:  # US bounds
        soil_classification = get_soil_classification_nrcs_sda(lat, lon)
        # If NRCS fails, fallback to global ISRIC SoilGrids
        if not soil_classification.get("success"):
            soil_classification_global = get_soil_classification_isric_soilgrids(
                lat, lon
            )
            if soil_classification_global.get("success"):
                soil_classification = soil_classification_global
    else:
        # Use global ISRIC SoilGrids for non-US locations
        soil_classification = get_soil_classification_isric_soilgrids(lat, lon)

    enriched_data["soil_classification"] = soil_classification

    # 6. Get soil properties from WCS
    soil_properties = get_soil_properties_isric_wcs(lat, lon)
    enriched_data["soil_properties"] = soil_properties

    # 7. Get ecoregion and biome information (local TEOW 2017)
    ecoregion = get_local_ecoregion(lat, lon)
    enriched_data["ecoregion"] = ecoregion

    # 8. Classify ecosystem path (v0 rules-only)
    ecosystem_classification = classify_ecosystem_v0(
        lat,
        lon,
        enriched_data["elevation"],
        enriched_data["ecoregion"],
        enriched_data["nearby_features"],
    )
    enriched_data["ecosystem_classification"] = ecosystem_classification

    # 9. Create enrichment summary
    successful_enrichments = []
    if enriched_data["elevation"].get("success"):
        successful_enrichments.append("elevation")
    if enriched_data["weather"].get("success"):
        successful_enrichments.append("weather")
    if enriched_data["location_context"].get("success"):
        successful_enrichments.append("location_context")
    if enriched_data["nearby_features"].get("success"):
        successful_enrichments.append("nearby_features")
    if enriched_data["soil_classification"].get("success"):
        successful_enrichments.append("soil_classification")
    if enriched_data["soil_properties"].get("success"):
        successful_enrichments.append("soil_properties")
    if enriched_data["ecoregion"].get("success"):
        successful_enrichments.append("ecoregion")
    if enriched_data["ecosystem_classification"].get("success"):
        successful_enrichments.append("ecosystem_classification")

    enriched_data["enrichment_summary"] = {
        "successful_enrichments": successful_enrichments,
        "total_enrichments": len(successful_enrichments),
        "apis_used": [
            "Open Elevation API",
            "USGS Elevation Point Query Service",
            "Open-Meteo Historical Weather API",
            "OpenStreetMap Nominatim",
            "OpenStreetMap Overpass API",
            "USDA NRCS Soil Data Access",
            "ISRIC SoilGrids v2.0",
            "ISRIC SoilGrids WCS",
            "WWF/RESOLVE Terrestrial Ecoregions 2017",
        ],
        "rate_limit_summary": _rate_limiter.get_rate_limit_summary(),
    }

    return enriched_data

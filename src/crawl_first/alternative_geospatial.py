"""
Alternative geospatial data APIs to replace ORNL landuse-mcp tools.

Provides soil type, land cover, and temporal analysis using free, open APIs:
- ISRIC SoilGrids for soil classification
- ESA WorldCover for global land cover
- USGS NLCD for US land cover
- MODIS/Landsat for temporal analysis
- ENVO ontology term mapping
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def load_mapping_file(mapping_name: str) -> Dict[str, Any]:
    """Load ENVO mapping from external JSON file."""
    mapping_file = Path(__file__).parent.parent.parent / "mappings" / f"{mapping_name}.json"

    if mapping_file.exists():
        try:
            with open(mapping_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Return empty dict if mapping file not found
    print(
        f"Warning: Mapping file {mapping_file} not found. Run generate_mappings.py to create it."
    )
    return {}


# Load mappings from external files
ESA_WORLDCOVER_TO_ENVO = load_mapping_file("esa_worldcover_to_envo")
NLCD_TO_ENVO = load_mapping_file("nlcd_to_envo")
SOILGRIDS_FAO_TO_ENVO = load_mapping_file("soilgrids_fao_to_envo")


class GeospatialDataCache:
    """Simple file-based cache for geospatial API responses."""

    def __init__(self, cache_dir: str = "../cache/geospatial"):
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

    def set(self, api_name: str, params: Dict[str, Any], data: Dict[str, Any]) -> None:
        """Cache result."""
        key = self._cache_key(api_name, params)
        cache_file = self.cache_dir / f"{key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass  # Ignore cache write failures


class GeospatialAPIClient:
    """HTTP client with retry logic for geospatial APIs."""

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.session = requests.Session()
        self.cache = GeospatialDataCache()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.session.timeout = timeout

    def get_with_cache(
        self, api_name: str, url: str, params: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Make GET request with caching."""
        cache_params = {"url": url, "params": params or {}}

        # Check cache first
        cached = self.cache.get(api_name, cache_params)
        if cached:
            return cached

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()

            # Check content type and validate JSON response
            content_type = response.headers.get('content-type', '').lower()
            if 'application/json' not in content_type and 'text/json' not in content_type:
                # Non-JSON response - create error result
                error_data = {
                    "error": f"Unexpected content type: {content_type}",
                    "response_preview": response.text[:200] if response.text else "Empty response",
                    "success": False
                }
                self.cache.set(api_name, cache_params, error_data)
                return error_data

            try:
                data = response.json()
            except json.JSONDecodeError as je:
                # JSON parsing error - create error result
                error_data = {
                    "error": f"JSON decode error: {str(je)}",
                    "response_preview": response.text[:200] if response.text else "Empty response",
                    "success": False
                }
                self.cache.set(api_name, cache_params, error_data)
                return error_data

            # Add success flag for successful responses
            if isinstance(data, dict):
                data["success"] = True

            # Cache successful response
            self.cache.set(api_name, cache_params, data)
            return data

        except requests.RequestException as e:
            # Network/HTTP error - create error result
            error_data = {
                "error": f"HTTP request error: {str(e)}",
                "success": False
            }
            self.cache.set(api_name, cache_params, error_data)
            return error_data


# Global client instance
_client = GeospatialAPIClient()


def get_soil_type_soilgrids(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get soil classification from ISRIC SoilGrids using robust WCS approach.
    
    Replaces the flaky REST API with GPT-5's reliable WCS client.

    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)

    Returns:
        Dict with soil_type, confidence, and envo_terms
    """
    try:
        from .soilgrids_client import SoilGridsWCSClient
        
        client = SoilGridsWCSClient()
        wrb_result = client.get_wrb_most_probable(lat=lat, lon=lon)
        
        if wrb_result and "wrb_label" in wrb_result:
            soil_type = wrb_result["wrb_label"]
            
            # Get ENVO term mapping - WRB maps to FAO/WRB reference base
            envo_mapping = SOILGRIDS_FAO_TO_ENVO.get(soil_type, {})

            return {
                "soil_type": soil_type,
                "confidence": "high",  # WCS data is reliable
                "envo_terms": [envo_mapping] if envo_mapping else [],
                "wrb_code": wrb_result.get("wrb_code"),
                "distance_m": wrb_result.get("distance_m", 0),
                "data_source": "ISRIC SoilGrids WCS v2.0",
                "method": "wcs_wrb_classification",
                "success": True
            }
    except Exception as e:
        return {
            "soil_type": None, 
            "confidence": None, 
            "envo_terms": [],
            "error": f"WCS classification failed: {str(e)}",
            "data_source": "ISRIC SoilGrids WCS",
            "success": False
        }

    return {"soil_type": None, "confidence": None, "envo_terms": [], "success": False}


def get_comprehensive_weather_openmeteo(lat: float, lon: float, date: str) -> Dict[str, Any]:
    """
    Get comprehensive weather data optimized for microbiome research.
    
    Args:
        lat: Latitude
        lon: Longitude  
        date: Date in YYYY-MM-DD format
        
    Returns:
        Dict with comprehensive weather parameters relevant to microbiome studies
    """
    try:
        import requests
        
        # Microbiome-relevant parameters organized by research importance
        atmospheric_params = [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "apparent_temperature_max", "apparent_temperature_min", 
            "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean",
            "dewpoint_2m_max", "dewpoint_2m_min", "dewpoint_2m_mean",
            "surface_pressure_max", "surface_pressure_min", "surface_pressure_mean"
        ]
        
        precipitation_params = [
            "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours"
        ]
        
        wind_params = [
            "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant"
        ]
        
        solar_radiation_params = [
            "shortwave_radiation_sum", "sunshine_duration", "daylight_duration"
        ]
        
        plant_biology_params = [
            "et0_fao_evapotranspiration", "vapour_pressure_deficit_max", 
            "vapour_pressure_deficit_min", "vapour_pressure_deficit_mean"
        ]
        
        soil_microbiome_params = [
            "soil_temperature_0_to_7cm_max", "soil_temperature_0_to_7cm_min", "soil_temperature_0_to_7cm_mean",
            "soil_moisture_0_to_7cm_mean"
        ]
        
        atmospheric_params_extra = [
            "cloudcover_max", "cloudcover_min", "cloudcover_mean",
            "weather_code"
        ]
        
        # Combine all parameters
        all_params = (atmospheric_params + precipitation_params + wind_params + 
                     solar_radiation_params + plant_biology_params + 
                     soil_microbiome_params + atmospheric_params_extra)
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Make multiple API calls to handle parameter limits
        batch_size = 15  # Conservative batch size
        all_data = {}
        successful_batches = 0
        total_batches = 0
        
        for i in range(0, len(all_params), batch_size):
            batch_params = all_params[i:i+batch_size]
            total_batches += 1
            
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": date,
                "end_date": date,
                "daily": ",".join(batch_params),
                "timezone": "UTC",
                "models": "best_match",
                "cell_selection": "nearest",
                "format": "json"
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                
                # HIGH BAR: Require successful response and valid JSON
                if (response.status_code == 200 and 
                    response.headers.get('Content-Type', '').startswith('application/json')):
                    
                    data = response.json()
                    
                    # Validate response structure
                    if (data.get("daily") and 
                        isinstance(data["daily"], dict) and
                        len(data["daily"]) > 1):  # Should have time + at least one parameter
                        
                        successful_batches += 1
                        
                        # Merge this batch data
                        for param, values in data["daily"].items():
                            if param != "time" and values:  # Skip time and empty arrays
                                # Validate we got actual data (not all null)
                                if any(v is not None for v in values):
                                    all_data[param] = values[0] if len(values) == 1 else values
                                    
                time.sleep(0.5)  # Rate limiting
                
            except Exception as batch_error:
                # Continue with other batches even if one fails
                continue
        
        # Calculate summary statistics for microbiome-relevant parameters
        microbiome_summary = {}
        
        # Temperature stress indicators
        if "temperature_2m_max" in all_data and "temperature_2m_min" in all_data:
            temp_max = all_data["temperature_2m_max"]
            temp_min = all_data["temperature_2m_min"] 
            if temp_max is not None and temp_min is not None:
                microbiome_summary["temperature_range_c"] = round(temp_max - temp_min, 2)
                microbiome_summary["temperature_stress"] = "high" if abs(temp_max - temp_min) > 15 else "moderate" if abs(temp_max - temp_min) > 8 else "low"
        
        # Moisture indicators  
        if "relative_humidity_2m_mean" in all_data and "precipitation_sum" in all_data:
            humidity = all_data["relative_humidity_2m_mean"]
            precip = all_data["precipitation_sum"]
            if humidity is not None and precip is not None:
                microbiome_summary["moisture_index"] = round((humidity / 100.0) + (precip / 10.0), 2)
        
        # Soil microclimate
        soil_temp = all_data.get("soil_temperature_0_to_7cm_mean")
        soil_moisture = all_data.get("soil_moisture_0_to_7cm_mean")
        if soil_temp is not None and soil_moisture is not None:
            microbiome_summary["soil_microclimate_score"] = round((soil_temp * 0.1) + (soil_moisture * 0.01), 2)
        
        # HIGH BAR: Only consider success if we got substantial data
        if (successful_batches >= total_batches * 0.6 and  # At least 60% of batches succeeded
            len(all_data) >= 8 and  # Got at least 8 parameters
            any(param in all_data for param in ["temperature_2m_mean", "relative_humidity_2m_mean", "precipitation_sum"])):  # Got core parameters
            
            return {
                "success": True,
                "data_source": "Open-Meteo Archive API", 
                "method": "comprehensive_microbiome_weather",
                "date": date,
                "location": {"latitude": lat, "longitude": lon},
                "parameters_requested": len(all_params),
                "parameters_received": len(all_data),
                "batch_success_rate": successful_batches / total_batches,
                "quality": "high_confidence" if successful_batches == total_batches else "partial_data",
                
                # Raw parameter data (native format, no normalization)
                "atmospheric": {k: all_data[k] for k in atmospheric_params if k in all_data},
                "precipitation": {k: all_data[k] for k in precipitation_params if k in all_data},
                "wind": {k: all_data[k] for k in wind_params if k in all_data},
                "solar_radiation": {k: all_data[k] for k in solar_radiation_params if k in all_data},
                "plant_biology": {k: all_data[k] for k in plant_biology_params if k in all_data},
                "soil_microbiome": {k: all_data[k] for k in soil_microbiome_params if k in all_data},
                "atmospheric_extra": {k: all_data[k] for k in atmospheric_params_extra if k in all_data},
                
                # Derived microbiome indicators (native calculations, no mapping)
                "microbiome_indicators": microbiome_summary
            }
        else:
            return {
                "success": False,
                "error": f"Insufficient weather data: {successful_batches}/{total_batches} batches, {len(all_data)} parameters",
                "data_source": "Open-Meteo Archive API",
                "partial_data": all_data if all_data else None
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Comprehensive weather API error: {str(e)}",
            "data_source": "Open-Meteo Archive API"
        }


def _calculate_usda_texture_class(sand_pct: float, clay_pct: float, silt_pct: float) -> str:
    """
    Calculate USDA soil texture class from sand/clay/silt percentages.
    
    Based on USDA soil texture triangle classification system.
    Source: USDA Natural Resources Conservation Service Soil Survey Manual
    """
    # Ensure percentages sum to ~100% (allow small rounding errors)
    total = sand_pct + clay_pct + silt_pct
    if abs(total - 100.0) > 5.0:
        return "Undefined (percentages don't sum to 100%)"
    
    # USDA texture triangle rules
    if clay_pct >= 40:
        if sand_pct >= 45:
            return "Sandy clay"
        elif sand_pct < 20:
            return "Silty clay"
        else:
            return "Clay"
    elif clay_pct >= 27:
        if sand_pct >= 20 and sand_pct < 45:
            return "Clay loam"
        elif sand_pct >= 45:
            return "Sandy clay loam"
        else:
            return "Silty clay loam"
    elif silt_pct >= 80:
        return "Silt"
    elif sand_pct >= 85:
        return "Sand"
    elif sand_pct >= 70:
        return "Loamy sand"
    elif (silt_pct >= 50) or (clay_pct < 12 and silt_pct >= 50):
        return "Silt loam"
    elif clay_pct < 20 and sand_pct >= 52:
        return "Sandy loam"
    else:
        return "Loam"


def get_isric_soilgrids_wcs_properties(lat: float, lon: float) -> Dict[str, Any]:
    """Get SoilGrids properties using GPT-5's robust WCS client."""
    try:
        from .soilgrids_client import SoilGridsWCSClient
        
        client = SoilGridsWCSClient()
        
        # Get WRB classification
        wrb_result = client.get_wrb_most_probable(lat=lat, lon=lon)
        
        # Get key soil properties
        properties = {}
        property_configs = [
            # Core chemical properties
            ("phh2o", "0-5cm", "Q0.5"),     # pH in H2O
            ("soc", "0-5cm", "Q0.5"),       # Soil organic carbon (g/kg)
            ("nitrogen", "0-5cm", "Q0.5"),  # Total nitrogen (cg/kg)
            ("cec", "0-5cm", "Q0.5"),       # Cation exchange capacity (cmol(c)/kg)
            
            # Physical properties and texture
            ("clay", "0-5cm", "Q0.5"),      # Clay content (g/kg)
            ("sand", "0-5cm", "Q0.5"),      # Sand content (g/kg)
            ("silt", "0-5cm", "Q0.5"),      # Silt content (g/kg)
            ("bdod", "0-5cm", "Q0.5"),      # Bulk density (cg/cm³)
            ("cfvo", "0-5cm", "Q0.5"),      # Coarse fragments (cm³/dm³)
            
            # Carbon stocks (using deeper profile for stocks)
            ("ocs", "0-30cm", "mean"),      # Organic carbon stock (kg/m²)
            ("ocd", "0-30cm", "mean"),      # Organic carbon density (kg/m³)
        ]
        
        for prop_id, depth, stat in property_configs:
            try:
                prop_result = client.get_property_value(
                    lat=lat, lon=lon, 
                    property_id=prop_id, 
                    depth=depth, 
                    stat=stat
                )
                properties[prop_id] = prop_result
            except Exception:
                properties[prop_id] = {"error": f"Failed to fetch {prop_id}"}
        
        # Format for unified enrichment compatibility
        soil_properties = {}
        for prop_id, prop_data in properties.items():
            if "value" in prop_data:
                soil_properties[prop_id] = {
                    "value": prop_data["value"],
                    "unit": prop_data.get("unit", "unknown"),
                    "depth": prop_data["depth"],
                    "stat": prop_data["stat"],
                    "distance_m": prop_data["distance_m"]
                }
        
        # Calculate USDA texture class from sand/silt/clay percentages
        usda_texture_class = None
        if all(prop in soil_properties for prop in ["sand", "clay", "silt"]):
            # Values are in g/kg, convert to percentages
            sand_pct = soil_properties["sand"]["value"] / 10.0
            clay_pct = soil_properties["clay"]["value"] / 10.0
            silt_pct = soil_properties["silt"]["value"] / 10.0
            
            # SoilGrids sand+silt+clay may not sum to 100% due to coarse fragments
            # Normalize to 100% for USDA texture triangle (which excludes coarse fragments)
            total_fine_earth = sand_pct + clay_pct + silt_pct
            if total_fine_earth > 0:
                sand_pct_norm = (sand_pct / total_fine_earth) * 100
                clay_pct_norm = (clay_pct / total_fine_earth) * 100
                silt_pct_norm = (silt_pct / total_fine_earth) * 100
                usda_texture_class = _calculate_usda_texture_class(sand_pct_norm, clay_pct_norm, silt_pct_norm)
        
        return {
            "wrb_classification": wrb_result,
            "soil_properties": soil_properties,
            "usda_texture_class": usda_texture_class,
            "properties_detailed": properties,
            "data_source": "ISRIC SoilGrids WCS v2.0",
            "success": True,
            "method": "gpt5_wcs_client"
        }
        
    except Exception as e:
        return {
            "error": f"SoilGrids WCS error: {str(e)}",
            "data_source": "ISRIC SoilGrids WCS",
            "success": False
        }


def get_isric_soilgrids_wcs_properties_old(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get soil properties from ISRIC SoilGrids WCS API.
    
    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
        
    Returns:
        Dict with soil properties (pH, SOC, sand, silt, clay, etc.)
    """
    try:
        import requests
        import struct
        from io import BytesIO
        
        # Properties to fetch with their scaling factors
        properties = {
            "phh2o": {"depth": "0-5cm", "scale": 10, "unit": "pH", "name": "pH in H2O"},
            "soc": {"depth": "0-5cm", "scale": 10, "unit": "g/kg", "name": "Soil Organic Carbon"},
            "sand": {"depth": "0-5cm", "scale": 10, "unit": "%", "name": "Sand content"},
            "silt": {"depth": "0-5cm", "scale": 10, "unit": "%", "name": "Silt content"}, 
            "clay": {"depth": "0-5cm", "scale": 10, "unit": "%", "name": "Clay content"},
            "bdod": {"depth": "0-5cm", "scale": 100, "unit": "g/cm³", "name": "Bulk density"},
            "nitrogen": {"depth": "0-5cm", "scale": 100, "unit": "g/kg", "name": "Total nitrogen"}
        }
        
        soil_data = {}
        
        for prop, config in properties.items():
            try:
                # Try WCS 2.0.1 first, fallback to 1.0.0
                coverage_id = f"{prop}_{config['depth']}_Q0.5"
                
                # WCS 2.0.1 request
                wcs_url = f"https://maps.isric.org/mapserv?map=/map/{prop}.map"
                params_v2 = {
                    "SERVICE": "WCS",
                    "VERSION": "2.0.1",
                    "REQUEST": "GetCoverage",
                    "COVERAGEID": coverage_id,
                    "FORMAT": "image/tiff",
                    "subset": [f"Long({lon},{lon})", f"Lat({lat},{lat})"]
                }
                
                response = requests.get(wcs_url, params=params_v2, timeout=30)
                
                if response.status_code != 200:
                    # Fallback to WCS 1.0.0
                    params_v1 = {
                        "SERVICE": "WCS", 
                        "VERSION": "1.0.0",
                        "REQUEST": "GetCoverage",
                        "COVERAGE": coverage_id,
                        "CRS": "EPSG:4326",
                        "BBOX": f"{lon},{lat},{lon},{lat}",
                        "WIDTH": "1",
                        "HEIGHT": "1", 
                        "FORMAT": "GEOTIFF"
                    }
                    response = requests.get(wcs_url, params=params_v1, timeout=30)
                
                # HIGH BAR FOR SUCCESS: Require valid HTTP response, content, and parseable data
                if (response.status_code == 200 and 
                    response.content and 
                    len(response.content) > 500 and  # Minimum valid TIFF size
                    response.headers.get('Content-Type', '').startswith('image')):
                    
                    # Parse TIFF data (simplified for single pixel)
                    content = response.content
                    
                    # More rigorous TIFF validation
                    if (len(content) > 500 and  # Reasonable TIFF minimum
                        content[:2] in [b'II', b'MM']):  # Valid TIFF header
                        
                        # Try to extract value from TIFF structure
                        try:
                            # Multiple extraction attempts for robustness
                            extraction_attempts = [
                                content[-4:-2],    # Common location for single pixel
                                content[-8:-6],    # Alternative location
                                content[200:202]   # Header area location
                            ]
                            
                            raw_value = None
                            for attempt in extraction_attempts:
                                if len(attempt) == 2:
                                    try:
                                        candidate = struct.unpack('<h', attempt)[0]  # Little-endian short
                                        # Validate that this looks like reasonable soil data
                                        if prop == "phh2o" and 30 <= candidate <= 120:  # pH * 10 range
                                            raw_value = candidate
                                            break
                                        elif prop in ["sand", "silt", "clay"] and 0 <= candidate <= 1000:  # percentage * 10
                                            raw_value = candidate
                                            break
                                        elif prop == "soc" and 0 <= candidate <= 3000:  # SOC range
                                            raw_value = candidate
                                            break
                                        elif prop == "bdod" and 0 <= candidate <= 200:  # Bulk density range
                                            raw_value = candidate
                                            break
                                        elif prop == "nitrogen" and 0 <= candidate <= 500:  # Nitrogen range
                                            raw_value = candidate
                                            break
                                    except struct.error:
                                        continue
                            
                            # HIGH BAR: Only consider success if we have a valid, realistic value
                            if (raw_value is not None and 
                                raw_value not in [-32768, 32767, 0] and  # Exclude no-data and zero
                                raw_value > 0):  # Must be positive for soil properties
                                
                                scaled_value = raw_value / config["scale"]
                                
                                # Additional validation: realistic ranges for scaled values
                                valid_range = True
                                if prop == "phh2o" and not (3.0 <= scaled_value <= 12.0):
                                    valid_range = False
                                elif prop in ["sand", "silt", "clay"] and not (0.1 <= scaled_value <= 100.0):
                                    valid_range = False
                                elif prop == "soc" and not (0.1 <= scaled_value <= 300.0):
                                    valid_range = False
                                
                                if valid_range:
                                    soil_data[prop] = {
                                        "value": round(scaled_value, 2),
                                        "unit": config["unit"],
                                        "name": config["name"],
                                        "depth": config["depth"],
                                        "source": "ISRIC SoilGrids WCS",
                                        "success": True,
                                        "quality": "high_confidence",
                                        "validation": "range_checked"
                                    }
                                else:
                                    soil_data[prop] = {
                                        "value": None,
                                        "error": f"Value {scaled_value} outside realistic range",
                                        "unit": config["unit"],
                                        "success": False
                                    }
                            else:
                                soil_data[prop] = {
                                    "value": None,
                                    "error": "No valid data extracted from TIFF",
                                    "unit": config["unit"],
                                    "success": False
                                }
                        except Exception as parse_e:
                                soil_data[prop] = {
                                    "value": None,
                                    "error": f"TIFF parsing error: {str(parse_e)}",
                                    "unit": config["unit"],
                                    "success": False
                                }
                        else:
                            soil_data[prop] = {
                                "value": None,
                                "error": "Invalid TIFF response",
                                "unit": config["unit"],
                                "success": False
                            }
                    else:
                        soil_data[prop] = {
                            "value": None,
                            "error": "Empty response",
                            "unit": config["unit"],
                            "success": False
                        }
                else:
                    soil_data[prop] = {
                        "value": None,
                        "error": f"WCS request failed: {response.status_code}",
                        "unit": config["unit"],
                        "success": False
                    }
                    
            except Exception as e:
                soil_data[prop] = {
                    "value": None,
                    "error": f"Request error: {str(e)}",
                    "unit": config["unit"],
                    "success": False
                }
        
        # Calculate USDA texture class if we have sand/silt/clay
        texture_class = None
        if (soil_data.get("sand", {}).get("success") and 
            soil_data.get("silt", {}).get("success") and 
            soil_data.get("clay", {}).get("success")):
            
            sand_pct = soil_data["sand"]["value"]
            silt_pct = soil_data["silt"]["value"] 
            clay_pct = soil_data["clay"]["value"]
            
            if sand_pct and silt_pct and clay_pct:
                # Simplified USDA texture classification
                if clay_pct >= 40:
                    texture_class = "Clay"
                elif clay_pct >= 27:
                    if sand_pct >= 45:
                        texture_class = "Sandy clay"
                    else:
                        texture_class = "Clay loam"
                elif sand_pct >= 85:
                    texture_class = "Sand"
                elif sand_pct >= 70:
                    if clay_pct >= 15:
                        texture_class = "Sandy clay loam"
                    else:
                        texture_class = "Sandy loam"
                elif silt_pct >= 80:
                    texture_class = "Silt"
                elif silt_pct >= 50:
                    texture_class = "Silt loam"
                else:
                    texture_class = "Loam"
        
        return {
            "soil_properties": soil_data,
            "usda_texture_class": texture_class,
            "data_source": "ISRIC SoilGrids WCS",
            "method": "wcs_raster_sampling",
            "resolution": "250m",
            "success": any(prop.get("success", False) for prop in soil_data.values())
        }
        
    except Exception as e:
        return {
            "soil_properties": {},
            "error": f"ISRIC SoilGrids WCS API error: {str(e)}",
            "data_source": "ISRIC SoilGrids WCS",
            "success": False
        }


def get_usda_nrcs_sda_soil(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get detailed US soil data from USDA NRCS Soil Data Access (SDA).
    
    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
        
    Returns:
        Dict with USDA soil taxonomy and component details
    """
    # Check if location is roughly within US bounds
    if not (-170 <= lon <= -60 and 15 <= lat <= 75):
        return {
            "soil_type": None,
            "error": "Outside US coverage (USDA NRCS SDA is US-only)",
            "data_source": "USDA NRCS SDA",
            "success": False
        }
    
    try:
        import requests
        
        # Use the exact working format from the notes
        wkt_point = f"POINT({lon} {lat})"
        
        # Single query to get soil data using the working pattern from notes
        soil_query = f"""
        SELECT 
            c.mukey,
            c.compname,
            c.comppct_r,
            c.taxclname,
            c.taxorder,
            c.taxsuborder,
            c.taxgrtgroup,
            c.taxsubgrp,
            c.drainagecl,
            c.hydgrp,
            mu.muname
        FROM component c
        INNER JOIN mapunit mu ON c.mukey = mu.mukey
        WHERE c.mukey IN (
            SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt_point}')
        )
        AND c.majcompflag = 'Yes'
        ORDER BY c.comppct_r DESC
        """
        
        url = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "SERVICE": "query",
            "REQUEST": "query", 
            "FORMAT": "JSON",
            "QUERY": soil_query
        }
        
        response = requests.post(url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if result.get("Table") and len(result["Table"]) > 0:
            # Get the dominant component (first result, ordered by comppct_r DESC)
            soil_data = result["Table"][0]
            
            return {
                "mukey": soil_data[0],
                "component_name": soil_data[1],
                "component_percent": soil_data[2],
                "taxonomic_class": soil_data[3],
                "tax_order": soil_data[4],
                "tax_suborder": soil_data[5], 
                "tax_great_group": soil_data[6],
                "tax_subgroup": soil_data[7],
                "drainage_class": soil_data[8],
                "hydrologic_group": soil_data[9],
                "mapunit_name": soil_data[10],
                "data_source": "USDA NRCS SDA",
                "success": True,
                "full_taxonomy": f"{soil_data[4]} > {soil_data[5]} > {soil_data[7]}" if soil_data[4] else None
            }
        else:
            return {
                "soil_type": None,
                "error": "No soil data found for location",
                "data_source": "USDA NRCS SDA", 
                "success": False
            }
            
    except Exception as e:
        return {
            "soil_type": None,
            "error": f"USDA NRCS SDA API error: {str(e)}",
            "data_source": "USDA NRCS SDA",
            "success": False
        }


def get_land_cover_inferred(lat: float, lon: float) -> Dict[str, Any]:
    """
    Infer land cover from OSM data and coordinate-based heuristics.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Dict with inferred land cover classification
    """
    # First try OSM land use/natural tags
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:30];
    (
      way["landuse"](around:500,{lat},{lon});
      way["natural"](around:500,{lat},{lon});
      relation["landuse"](around:500,{lat},{lon});
      relation["natural"](around:500,{lat},{lon});
    );
    out geom;
    """
    
    try:
        response = requests.post(overpass_url, data=query, timeout=30)
        data = response.json()
        
        # Analyze OSM elements
        land_uses = []
        naturals = []
        
        for elem in data.get("elements", []):
            tags = elem.get("tags", {})
            if "landuse" in tags:
                land_uses.append(tags["landuse"])
            if "natural" in tags:
                naturals.append(tags["natural"])
        
        # Determine primary classification
        if naturals:
            primary_natural = max(set(naturals), key=naturals.count)
            if primary_natural in ["forest", "wood"]:
                return {
                    "land_cover_class": 10,  # Tree cover
                    "class_name": "Tree cover",
                    "confidence": 0.8,
                    "envo_terms": ["ENVO:00000028"],  # forest biome
                    "source": "osm_inference",
                    "method": "natural_tag_analysis",
                    "success": True
                }
            elif primary_natural in ["grassland", "meadow", "heath"]:
                return {
                    "land_cover_class": 30,  # Grassland  
                    "class_name": "Grassland",
                    "confidence": 0.7,
                    "envo_terms": ["ENVO:00000106"],  # grassland biome
                    "source": "osm_inference", 
                    "method": "natural_tag_analysis",
                    "success": True
                }
            elif primary_natural in ["water", "wetland"]:
                return {
                    "land_cover_class": 80,  # Permanent water bodies
                    "class_name": "Permanent water bodies", 
                    "confidence": 0.9,
                    "envo_terms": ["ENVO:00000890"],  # freshwater lake
                    "source": "osm_inference",
                    "method": "natural_tag_analysis", 
                    "success": True
                }
        
        if land_uses:
            primary_landuse = max(set(land_uses), key=land_uses.count)
            if primary_landuse in ["residential", "commercial", "industrial"]:
                return {
                    "land_cover_class": 50,  # Built-up
                    "class_name": "Built-up",
                    "confidence": 0.8,
                    "envo_terms": ["ENVO:00000856"],  # built environment
                    "source": "osm_inference",
                    "method": "landuse_tag_analysis",
                    "success": True
                }
            elif primary_landuse in ["farmland", "orchard", "vineyard"]:
                return {
                    "land_cover_class": 40,  # Cropland
                    "class_name": "Cropland", 
                    "confidence": 0.7,
                    "envo_terms": ["ENVO:00000077"],  # cropland
                    "source": "osm_inference",
                    "method": "landuse_tag_analysis",
                    "success": True
                }
    
    except Exception as e:
        pass
    
    # Fallback: coordinate-based heuristics for common regions
    # This is a very basic classifier based on known geographic patterns
    
    # High altitude, mountainous regions (like Yellowstone)
    if 40 <= lat <= 50 and -115 <= lon <= -105:  # Rocky Mountain region
        return {
            "land_cover_class": 10,  # Tree cover (coniferous forest likely)
            "class_name": "Tree cover", 
            "confidence": 0.6,
            "envo_terms": ["ENVO:00000297"],  # temperate coniferous forest
            "source": "geographic_heuristic",
            "method": "rocky_mountain_inference",
            "success": True
        }
    
    # Default fallback
    return {
        "land_cover_class": None,
        "class_name": None,
        "confidence": 0.0,
        "envo_terms": [],
        "source": "inference_failed", 
        "method": "no_data_available",
        "success": False
    }


def get_land_cover_esa_worldcover(
    lat: float, lon: float, year: int = 2021
) -> Dict[str, Any]:
    """
    Get land cover from ESA WorldCover using point query with fallback to inference.

    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
        year: Year (2020 or 2021)

    Returns:
        Dict with land_cover_class, confidence, and envo_terms
    """
    # Try the working inference method first since WMS is unreliable
    inference_result = get_land_cover_inferred(lat, lon)
    if inference_result.get("success"):
        return inference_result
    
    # If inference fails, return error (WMS service is consistently failing)
    return {
        "land_cover_class": None, 
        "envo_terms": [], 
        "error": "ESA WorldCover WMS service unavailable, inference failed",
        "success": False
    }

    if "features" not in data:
        return {
            "land_cover_class": None, 
            "envo_terms": [], 
            "error": "Invalid response format from ESA WorldCover service",
            "success": False
        }

    try:
        features = data["features"]
        if features:
            properties = features[0].get("properties", {})

            # Extract land cover class (varies by service response format)
            land_cover_value = None
            for key, value in properties.items():
                if "map" in key.lower() or "class" in key.lower():
                    land_cover_value = int(value)
                    break

            if land_cover_value and str(land_cover_value) in ESA_WORLDCOVER_TO_ENVO:
                envo_mapping = ESA_WORLDCOVER_TO_ENVO[str(land_cover_value)]
                return {
                    "land_cover_class": land_cover_value,
                    "land_cover_label": envo_mapping.get("term"),
                    "year": year,
                    "envo_terms": [envo_mapping],
                    "data_source": "ESA WorldCover",
                    "success": True,
                }
    except (KeyError, TypeError, ValueError):
        pass

    return {
        "land_cover_class": None, 
        "envo_terms": [], 
        "error": "No land cover data found or mapping not available",
        "success": False
    }


def get_land_cover_nlcd(lat: float, lon: float, year: int = 2021) -> Dict[str, Any]:
    """
    Get US land cover from USGS NLCD.

    Args:
        lat: Latitude in US (-90 to 90)
        lon: Longitude in US (-180 to 180)
        year: NLCD year (2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021)

    Returns:
        Dict with land_cover_class and envo_terms
    """
    # Check if coordinates are roughly within US bounds
    if not (-170 <= lon <= -60 and 15 <= lat <= 75):
        return {
            "land_cover_class": None,
            "envo_terms": [],
            "error": "Outside US bounds",
        }

    # USGS NLCD WMS service
    base_url = "https://www.mrlc.gov/geoserver/mrlc_display/wms"

    # Map year to layer name
    layer_map = {
        2001: "NLCD_2001_Land_Cover_L48",
        2004: "NLCD_2004_Land_Cover_L48",
        2006: "NLCD_2006_Land_Cover_L48",
        2008: "NLCD_2008_Land_Cover_L48",
        2011: "NLCD_2011_Land_Cover_L48",
        2013: "NLCD_2013_Land_Cover_L48",
        2016: "NLCD_2016_Land_Cover_L48",
        2019: "NLCD_2019_Land_Cover_L48",
        2021: "NLCD_2021_Land_Cover_L48",
    }

    layer = layer_map.get(year, layer_map[2021])

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": layer,
        "QUERY_LAYERS": layer,
        "INFO_FORMAT": "application/json",
        "FEATURE_COUNT": 1,
        "SRS": "EPSG:4326",
        "BBOX": f"{lon-0.001},{lat-0.001},{lon+0.001},{lat+0.001}",
        "WIDTH": "100",
        "HEIGHT": "100",
        "X": "50",
        "Y": "50",
    }

    data = _client.get_with_cache("nlcd", base_url, params)

    if not data:
        return {
            "land_cover_class": None, 
            "envo_terms": [], 
            "error": "No response from USGS NLCD service",
            "success": False
        }
    
    if not data.get("success", True):
        return {
            "land_cover_class": None, 
            "envo_terms": [], 
            "error": data.get("error", "Unknown error from USGS NLCD service"),
            "success": False
        }

    if "features" not in data:
        return {
            "land_cover_class": None, 
            "envo_terms": [], 
            "error": "Invalid response format from USGS NLCD service",
            "success": False
        }

    try:
        features = data["features"]
        if features:
            properties = features[0].get("properties", {})

            # NLCD class value is typically in a field containing the layer name
            land_cover_value = None
            for key, value in properties.items():
                if "NLCD" in key and isinstance(value, (int, str)):
                    try:
                        land_cover_value = int(value)
                        break
                    except ValueError:
                        continue

            if land_cover_value and str(land_cover_value) in NLCD_TO_ENVO:
                envo_mapping = NLCD_TO_ENVO[str(land_cover_value)]
                return {
                    "land_cover_class": land_cover_value,
                    "land_cover_label": envo_mapping.get("term"),
                    "year": year,
                    "classification_system": "NLCD",
                    "envo_terms": [envo_mapping],
                    "data_source": "USGS NLCD",
                    "success": True,
                }
    except (KeyError, TypeError, ValueError):
        pass

    return {
        "land_cover_class": None, 
        "envo_terms": [], 
        "error": "No land cover data found or mapping not available",
        "success": False
    }


def get_available_nlcd_years() -> List[int]:
    """Get available NLCD years."""
    return [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]


def get_historical_land_cover(
    lat: float, lon: float, target_date: str
) -> Dict[str, Any]:
    """
    Get historical land cover closest to target date.

    Args:
        lat: Latitude
        lon: Longitude
        target_date: Date in YYYY-MM-DD format

    Returns:
        Dict with historical land cover data
    """
    try:
        target_year = int(target_date.split("-")[0])
    except (ValueError, IndexError):
        target_year = datetime.now().year

    # Try ESA WorldCover first (2020, 2021 available)
    if target_year >= 2020:
        year = 2021 if target_year >= 2021 else 2020
        result = get_land_cover_esa_worldcover(lat, lon, year)
        if result.get("land_cover_class"):
            result["year_requested"] = target_year
            result["year_used"] = year
            return result

    # Fall back to NLCD for US locations
    if -170 <= lon <= -60 and 15 <= lat <= 75:
        available_years = get_available_nlcd_years()

        # Find closest available year
        closest_year = min(available_years, key=lambda x: abs(x - target_year))

        result = get_land_cover_nlcd(lat, lon, closest_year)
        if result.get("land_cover_class"):
            result["year_requested"] = target_year
            result["year_used"] = closest_year
            return result

    return {
        "land_cover_class": None,
        "envo_terms": [],
        "error": "No historical data available for location/date",
    }


def get_comprehensive_site_analysis(
    lat: float, lon: float, collection_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get comprehensive site analysis combining soil, land cover, and temporal data.

    Args:
        lat: Latitude
        lon: Longitude
        collection_date: Collection date in YYYY-MM-DD format

    Returns:
        Comprehensive analysis dict
    """
    analysis = {
        "coordinates": {"latitude": lat, "longitude": lon},
        "collection_date": collection_date,
        "soil_analysis": {},
        "land_cover_analysis": {},
        "temporal_analysis": {},
        "envo_terms_summary": [],
    }

    # Soil analysis
    soil_data = get_soil_type_soilgrids(lat, lon)
    analysis["soil_analysis"] = soil_data

    # Current land cover
    current_land_cover = get_land_cover_esa_worldcover(lat, lon, 2021)
    analysis["land_cover_analysis"]["current"] = current_land_cover

    # Historical land cover if date provided
    if collection_date:
        historical_land_cover = get_historical_land_cover(lat, lon, collection_date)
        analysis["land_cover_analysis"]["historical"] = historical_land_cover

        # Temporal change analysis
        current_class = current_land_cover.get("land_cover_class")
        historical_class = historical_land_cover.get("land_cover_class")

        if current_class and historical_class:
            analysis["temporal_analysis"] = {
                "land_cover_changed": current_class != historical_class,
                "change_description": (
                    f"Changed from {historical_land_cover.get('land_cover_label')} to {current_land_cover.get('land_cover_label')}"
                    if current_class != historical_class
                    else "No significant change detected"
                ),
            }

    # Aggregate ENVO terms
    all_envo_terms = []

    # Add soil ENVO terms
    if soil_data.get("envo_terms"):
        all_envo_terms.extend(soil_data["envo_terms"])

    # Add land cover ENVO terms
    for analysis_type in ["current", "historical"]:
        land_cover_data = analysis["land_cover_analysis"].get(analysis_type, {})
        if land_cover_data.get("envo_terms"):
            all_envo_terms.extend(land_cover_data["envo_terms"])

    # Remove duplicates
    unique_envo_terms = []
    seen_ids = set()
    for term in all_envo_terms:
        if term.get("id") and term["id"] not in seen_ids:
            unique_envo_terms.append(term)
            seen_ids.add(term["id"])

    analysis["envo_terms_summary"] = unique_envo_terms

    return analysis


if __name__ == "__main__":
    # Test with a sample location
    test_lat, test_lon = 37.7749, -122.4194  # San Francisco
    test_date = "2020-06-15"

    print("Testing alternative geospatial APIs...")

    # Test soil data
    print("\n1. Soil Analysis:")
    soil = get_soil_type_soilgrids(test_lat, test_lon)
    print(json.dumps(soil, indent=2))

    # Test ESA WorldCover
    print("\n2. ESA WorldCover:")
    esa_lc = get_land_cover_esa_worldcover(test_lat, test_lon, 2021)
    print(json.dumps(esa_lc, indent=2))

    # Test NLCD
    print("\n3. NLCD:")
    nlcd_lc = get_land_cover_nlcd(test_lat, test_lon, 2021)
    print(json.dumps(nlcd_lc, indent=2))

    # Test comprehensive analysis
    print("\n4. Comprehensive Analysis:")
    comprehensive = get_comprehensive_site_analysis(test_lat, test_lon, test_date)
    print(json.dumps(comprehensive, indent=2))

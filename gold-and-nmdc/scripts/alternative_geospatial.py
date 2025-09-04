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
    mapping_file = Path(__file__).parent / "mappings" / f"{mapping_name}.json"
    
    if mapping_file.exists():
        try:
            with open(mapping_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    
    # Return empty dict if mapping file not found
    print(f"Warning: Mapping file {mapping_file} not found. Run generate_mappings.py to create it.")
    return {}


# Load mappings from external files
ESA_WORLDCOVER_TO_ENVO = load_mapping_file("esa_worldcover_to_envo")
NLCD_TO_ENVO = load_mapping_file("nlcd_to_envo") 
SOILGRIDS_FAO_TO_ENVO = load_mapping_file("soilgrids_fao_to_envo")


class GeospatialDataCache:
    """Simple file-based cache for geospatial API responses."""
    
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
        
    def set(self, api_name: str, params: Dict[str, Any], data: Dict[str, Any]) -> None:
        """Cache result."""
        key = self._cache_key(api_name, params)
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            with open(cache_file, 'w') as f:
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
        
    def get_with_cache(self, api_name: str, url: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Make GET request with caching."""
        cache_params = {"url": url, "params": params or {}}
        
        # Check cache first
        cached = self.cache.get(api_name, cache_params)
        if cached:
            return cached
            
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache successful response
            self.cache.set(api_name, cache_params, data)
            return data
            
        except requests.RequestException:
            return None


# Global client instance
_client = GeospatialAPIClient()


def get_soil_type_soilgrids(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get soil classification from ISRIC SoilGrids.
    
    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
        
    Returns:
        Dict with soil_type, confidence, and envo_terms
    """
    url = "https://rest.soilgrids.org/soilgrids/v2.0/classification/query"
    params = {
        "lon": lon,
        "lat": lat,
        "number_classes": 3  # Top 3 most likely soil types
    }
    
    data = _client.get_with_cache("soilgrids", url, params)
    if not data:
        return {"soil_type": None, "confidence": None, "envo_terms": []}
        
    try:
        properties = data.get("properties", {})
        most_likely = properties.get("most_probable_class")
        
        if most_likely:
            # Get ENVO term mapping
            envo_mapping = SOILGRIDS_FAO_TO_ENVO.get(most_likely, {})
            
            return {
                "soil_type": most_likely,
                "confidence": properties.get("confidence", {}).get(most_likely),
                "envo_terms": [envo_mapping] if envo_mapping else [],
                "alternatives": properties.get("classification", {}),
                "data_source": "ISRIC SoilGrids v2.0"
            }
    except (KeyError, TypeError):
        pass
        
    return {"soil_type": None, "confidence": None, "envo_terms": []}


def get_land_cover_esa_worldcover(lat: float, lon: float, year: int = 2021) -> Dict[str, Any]:
    """
    Get land cover from ESA WorldCover using point query.
    
    Args:
        lat: Latitude (-90 to 90) 
        lon: Longitude (-180 to 180)
        year: Year (2020 or 2021)
        
    Returns:
        Dict with land_cover_class, confidence, and envo_terms
    """
    # ESA WorldCover WMS service
    base_url = "https://services.terrascope.be/wms/v2"
    
    # Map service for different years
    if year == 2020:
        layer = "worldcover_2020_map"
    else:
        layer = "worldcover_2021_map"
        
    # WMS GetFeatureInfo request for point query
    params = {
        'SERVICE': 'WMS',
        'VERSION': '1.3.0', 
        'REQUEST': 'GetFeatureInfo',
        'LAYERS': layer,
        'QUERY_LAYERS': layer,
        'INFO_FORMAT': 'application/json',
        'FEATURE_COUNT': 1,
        'CRS': 'EPSG:4326',
        'BBOX': f'{lat-0.001},{lon-0.001},{lat+0.001},{lon+0.001}',
        'WIDTH': '100',
        'HEIGHT': '100',
        'I': '50',  # X coordinate
        'J': '50'   # Y coordinate  
    }
    
    url = f"{base_url}/worldcover"
    data = _client.get_with_cache("esa_worldcover", url, params)
    
    if not data or 'features' not in data:
        return {"land_cover_class": None, "envo_terms": []}
        
    try:
        features = data['features']
        if features:
            properties = features[0].get('properties', {})
            
            # Extract land cover class (varies by service response format)
            land_cover_value = None
            for key, value in properties.items():
                if 'map' in key.lower() or 'class' in key.lower():
                    land_cover_value = int(value)
                    break
                    
            if land_cover_value and str(land_cover_value) in ESA_WORLDCOVER_TO_ENVO:
                envo_mapping = ESA_WORLDCOVER_TO_ENVO[str(land_cover_value)]
                return {
                    "land_cover_class": land_cover_value,
                    "land_cover_label": envo_mapping.get("term"),
                    "year": year,
                    "envo_terms": [envo_mapping],
                    "data_source": "ESA WorldCover"
                }
    except (KeyError, TypeError, ValueError):
        pass
        
    return {"land_cover_class": None, "envo_terms": []}


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
        return {"land_cover_class": None, "envo_terms": [], "error": "Outside US bounds"}
        
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
        2021: "NLCD_2021_Land_Cover_L48"
    }
    
    layer = layer_map.get(year, layer_map[2021])
    
    params = {
        'SERVICE': 'WMS',
        'VERSION': '1.1.1',
        'REQUEST': 'GetFeatureInfo',
        'LAYERS': layer,
        'QUERY_LAYERS': layer,
        'INFO_FORMAT': 'application/json',
        'FEATURE_COUNT': 1,
        'SRS': 'EPSG:4326',
        'BBOX': f'{lon-0.001},{lat-0.001},{lon+0.001},{lat+0.001}',
        'WIDTH': '100',
        'HEIGHT': '100',
        'X': '50',
        'Y': '50'
    }
    
    data = _client.get_with_cache("nlcd", base_url, params)
    
    if not data or 'features' not in data:
        return {"land_cover_class": None, "envo_terms": []}
        
    try:
        features = data['features'] 
        if features:
            properties = features[0].get('properties', {})
            
            # NLCD class value is typically in a field containing the layer name
            land_cover_value = None
            for key, value in properties.items():
                if 'NLCD' in key and isinstance(value, (int, str)):
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
                    "data_source": "USGS NLCD"
                }
    except (KeyError, TypeError, ValueError):
        pass
        
    return {"land_cover_class": None, "envo_terms": []}


def get_available_nlcd_years() -> List[int]:
    """Get available NLCD years."""
    return [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]


def get_historical_land_cover(lat: float, lon: float, target_date: str) -> Dict[str, Any]:
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
        target_year = int(target_date.split('-')[0])
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
        "error": "No historical data available for location/date"
    }


def get_comprehensive_site_analysis(lat: float, lon: float, collection_date: Optional[str] = None) -> Dict[str, Any]:
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
        "envo_terms_summary": []
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
                "change_description": f"Changed from {historical_land_cover.get('land_cover_label')} to {current_land_cover.get('land_cover_label')}" if current_class != historical_class else "No significant change detected"
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
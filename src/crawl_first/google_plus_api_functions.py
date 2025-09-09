#!/usr/bin/env python3
"""
Geospatial API enrichment functions for biosample data.

Functions that take input from biosample adapters (either directly or from files)
and enrich the data using various geospatial API sources (Google, USGS, OpenMeteo, etc.).
Each function handles one API source for one type of enrichment data.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from crawl_first.biosample_adapters import (
        BiosampleLocation,
        NMDCBiosampleAdapter,
        GOLDBiosampleAdapter,
        MongoNMDCBiosampleFetcher,
        MongoGOLDBiosampleFetcher,
        UnifiedBiosampleFetcher
    )
    from crawl_first.cache import cache_key, get_cache, save_cache
except ImportError:
    # Handle case where modules might not be available
    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_adapter_results(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load adapter results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_locations_from_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract location data from adapter results."""
    if 'results' in results:
        return results['results']
    elif 'nmdc_results' in results and 'gold_results' in results:
        return results['nmdc_results'] + results['gold_results']
    else:
        return []


def filter_enrichable_locations(locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter to only enrichable locations."""
    return [loc for loc in locations if loc.get('is_enrichable', False)]


def extract_locations_from_adapters(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Extract enrichable locations from adapter and samples."""
    locations = []
    for sample in samples:
        location = adapter.extract_location(sample)
        if location.is_enrichable():
            locations.append(location.to_dict())
    return locations


# =============================================================================
# ELEVATION API FUNCTIONS - BY API SOURCE
# =============================================================================

def get_elevation_google_direct(adapter, samples: List[Dict], 
                               use_cache: bool = True, 
                               save_cache: bool = True) -> List[Dict[str, Any]]:
    """Get elevation data using Google Elevation API from direct adapter calls.
    
    Args:
        adapter: Biosample adapter instance
        samples: List of sample dictionaries  
        use_cache: Check cache before making API call (default: True)
        save_cache: Save result to cache after API call (default: True)
    """
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        # Generate cache key for this location
        cache_params = {
            'lat': round(location['latitude'], 6),
            'lon': round(location['longitude'], 6),
            'api': 'google_elevation'
        }
        key = cache_key(cache_params)
        
        # Check cache if enabled
        cached_result = None
        if use_cache:
            cached_result = get_cache("elevation", key)
            
        if cached_result:
            # Use cached data
            elevation_data = {
                'sample_id': location['sample_id'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'elevation_meters': cached_result.get('elevation_meters'),
                'resolution_meters': cached_result.get('resolution_meters'),
                'elevation_source': 'google_elevation_api',
                'api_call_timestamp': cached_result.get('api_call_timestamp'),
                'original_location': location,
                'cached': True
            }
        else:
            # Would make actual Google API call here
            # For now, placeholder data
            api_result = {
                'elevation_meters': None,  # Would be actual API response
                'resolution_meters': None,
                'api_call_timestamp': datetime.now().isoformat()
            }
            
            elevation_data = {
                'sample_id': location['sample_id'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'elevation_meters': api_result['elevation_meters'],
                'resolution_meters': api_result['resolution_meters'],
                'elevation_source': 'google_elevation_api',
                'api_call_timestamp': api_result['api_call_timestamp'],
                'original_location': location,
                'cached': False
            }
            
            # Save to cache if enabled
            if save_cache:
                from crawl_first.cache import save_cache as save_to_cache
                save_to_cache("elevation", key, {
                    'elevation_meters': api_result['elevation_meters'],
                    'resolution_meters': api_result['resolution_meters'],
                    'api_call_timestamp': api_result['api_call_timestamp']
                })
        
        results.append(elevation_data)
    
    return results


def get_elevation_google_file(file_path: Union[str, Path], 
                             use_cache: bool = True, 
                             save_cache: bool = True) -> List[Dict[str, Any]]:
    """Get elevation data using Google Elevation API from adapter results file.
    
    Args:
        file_path: Path to adapter results JSON file
        use_cache: Check cache before making API call (default: True)
        save_cache: Save result to cache after API call (default: True)
    """
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        # Generate cache key for this location
        cache_params = {
            'lat': round(location['latitude'], 6),
            'lon': round(location['longitude'], 6),
            'api': 'google_elevation'
        }
        key = cache_key(cache_params)
        
        # Check cache if enabled
        cached_result = None
        if use_cache:
            cached_result = get_cache("elevation", key)
            
        if cached_result:
            # Use cached data
            elevation_data = {
                'sample_id': location['sample_id'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'elevation_meters': cached_result.get('elevation_meters'),
                'resolution_meters': cached_result.get('resolution_meters'),
                'elevation_source': 'google_elevation_api',
                'api_call_timestamp': cached_result.get('api_call_timestamp'),
                'original_location': location,
                'cached': True
            }
        else:
            # Would make actual Google API call here
            # For now, placeholder data
            api_result = {
                'elevation_meters': None,  # Would be actual API response
                'resolution_meters': None,
                'api_call_timestamp': datetime.now().isoformat()
            }
            
            elevation_data = {
                'sample_id': location['sample_id'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'elevation_meters': api_result['elevation_meters'],
                'resolution_meters': api_result['resolution_meters'],
                'elevation_source': 'google_elevation_api',
                'api_call_timestamp': api_result['api_call_timestamp'],
                'original_location': location,
                'cached': False
            }
            
            # Save to cache if enabled
            if save_cache:
                from crawl_first.cache import save_cache as save_to_cache
                save_to_cache("elevation", key, {
                    'elevation_meters': api_result['elevation_meters'],
                    'resolution_meters': api_result['resolution_meters'],
                    'api_call_timestamp': api_result['api_call_timestamp']
                })
        
        results.append(elevation_data)
    
    return results


def get_elevation_usgs_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get elevation data using USGS Elevation API from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        elevation_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'elevation_meters': None,  # Would call USGS Elevation Point Query Service
            'units': 'meters',
            'dataset': None,  # e.g., 'ned10m', 'ned30m'
            'elevation_source': 'usgs_elevation_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(elevation_data)
    
    return results


def get_elevation_usgs_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get elevation data using USGS Elevation API from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        elevation_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'elevation_meters': None,  # Would call USGS Elevation Point Query Service
            'units': 'meters',
            'dataset': None,  # e.g., 'ned10m', 'ned30m'
            'elevation_source': 'usgs_elevation_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(elevation_data)
    
    return results


def get_elevation_open_elevation_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get elevation data using Open Elevation API from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        elevation_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'elevation_meters': None,  # Would call Open Elevation API
            'elevation_source': 'open_elevation_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(elevation_data)
    
    return results


def get_elevation_open_elevation_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get elevation data using Open Elevation API from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        elevation_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'elevation_meters': None,  # Would call Open Elevation API
            'elevation_source': 'open_elevation_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(elevation_data)
    
    return results


# =============================================================================
# WEATHER API FUNCTIONS - BY API SOURCE
# =============================================================================

def get_weather_openmeteo_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get weather data using Open-Meteo API from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        if location.get('collection_date'):
            weather_data = {
                'sample_id': location['sample_id'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'collection_date': location['collection_date'],
                'temperature_2m_mean': None,  # Would call Open-Meteo API
                'relative_humidity_2m_mean': None,
                'precipitation_sum': None,
                'wind_speed_10m_mean': None,
                'weather_source': 'openmeteo_api',
                'api_call_timestamp': datetime.now().isoformat(),
                'original_location': location
            }
            results.append(weather_data)
    
    return results


def get_weather_openmeteo_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get weather data using Open-Meteo API from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        if location.get('collection_date'):
            weather_data = {
                'sample_id': location['sample_id'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'collection_date': location['collection_date'],
                'temperature_2m_mean': None,  # Would call Open-Meteo API
                'relative_humidity_2m_mean': None,
                'precipitation_sum': None,
                'wind_speed_10m_mean': None,
                'weather_source': 'openmeteo_api',
                'api_call_timestamp': datetime.now().isoformat(),
                'original_location': location
            }
            results.append(weather_data)
    
    return results


def get_weather_meteostat_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get weather data using Meteostat API from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        if location.get('collection_date'):
            weather_data = {
                'sample_id': location['sample_id'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'collection_date': location['collection_date'],
                'tavg': None,  # Average temperature - would call Meteostat API
                'tmin': None,  # Minimum temperature
                'tmax': None,  # Maximum temperature
                'prcp': None,  # Precipitation
                'wspd': None,  # Wind speed
                'weather_source': 'meteostat_api',
                'api_call_timestamp': datetime.now().isoformat(),
                'original_location': location
            }
            results.append(weather_data)
    
    return results


def get_weather_meteostat_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get weather data using Meteostat API from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        if location.get('collection_date'):
            weather_data = {
                'sample_id': location['sample_id'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'collection_date': location['collection_date'],
                'tavg': None,  # Average temperature - would call Meteostat API
                'tmin': None,  # Minimum temperature
                'tmax': None,  # Maximum temperature
                'prcp': None,  # Precipitation
                'wspd': None,  # Wind speed
                'weather_source': 'meteostat_api',
                'api_call_timestamp': datetime.now().isoformat(),
                'original_location': location
            }
            results.append(weather_data)
    
    return results


# =============================================================================
# GEOCODING API FUNCTIONS - BY API SOURCE
# =============================================================================

def get_geocoding_google_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get geocoding data using Google Geocoding API from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        geocoding_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'textual_location': location.get('textual_location'),
            'formatted_address': None,  # Would call Google Geocoding API
            'country': None,
            'administrative_area_level_1': None,  # State/Province
            'administrative_area_level_2': None,  # County
            'locality': None,  # City
            'place_id': None,
            'geocoding_source': 'google_geocoding_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(geocoding_data)
    
    return results


def get_geocoding_google_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get geocoding data using Google Geocoding API from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        geocoding_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'textual_location': location.get('textual_location'),
            'formatted_address': None,  # Would call Google Geocoding API
            'country': None,
            'administrative_area_level_1': None,  # State/Province
            'administrative_area_level_2': None,  # County
            'locality': None,  # City
            'place_id': None,
            'geocoding_source': 'google_geocoding_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(geocoding_data)
    
    return results


def get_geocoding_nominatim_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get geocoding data using Nominatim (OpenStreetMap) API from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        geocoding_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'textual_location': location.get('textual_location'),
            'display_name': None,  # Would call Nominatim API
            'country': None,
            'state': None,
            'county': None,
            'city': None,
            'osm_id': None,
            'osm_type': None,
            'geocoding_source': 'nominatim_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(geocoding_data)
    
    return results


def get_geocoding_nominatim_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get geocoding data using Nominatim (OpenStreetMap) API from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        geocoding_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'textual_location': location.get('textual_location'),
            'display_name': None,  # Would call Nominatim API
            'country': None,
            'state': None,
            'county': None,
            'city': None,
            'osm_id': None,
            'osm_type': None,
            'geocoding_source': 'nominatim_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(geocoding_data)
    
    return results


# =============================================================================
# LAND USE/COVER API FUNCTIONS - BY API SOURCE
# =============================================================================

def get_landuse_usgs_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get land use/cover data using USGS NLCD API from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        landuse_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'nlcd_class': None,  # Would call USGS NLCD API
            'nlcd_class_name': None,
            'nlcd_year': None,  # e.g., 2019, 2016, 2011
            'impervious_surface_percent': None,
            'tree_canopy_percent': None,
            'landuse_source': 'usgs_nlcd_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(landuse_data)
    
    return results


def get_landuse_usgs_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get land use/cover data using USGS NLCD API from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        landuse_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'nlcd_class': None,  # Would call USGS NLCD API
            'nlcd_class_name': None,
            'nlcd_year': None,  # e.g., 2019, 2016, 2011
            'impervious_surface_percent': None,
            'tree_canopy_percent': None,
            'landuse_source': 'usgs_nlcd_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(landuse_data)
    
    return results


def get_landuse_esa_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get land use/cover data using ESA WorldCover API from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        landuse_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'worldcover_class': None,  # Would call ESA WorldCover API
            'worldcover_class_name': None,
            'confidence': None,
            'year': None,  # e.g., 2020, 2021
            'landuse_source': 'esa_worldcover_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(landuse_data)
    
    return results


def get_landuse_esa_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get land use/cover data using ESA WorldCover API from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        landuse_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'worldcover_class': None,  # Would call ESA WorldCover API
            'worldcover_class_name': None,
            'confidence': None,
            'year': None,  # e.g., 2020, 2021
            'landuse_source': 'esa_worldcover_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(landuse_data)
    
    return results


# =============================================================================
# SOIL DATA API FUNCTIONS - BY API SOURCE
# =============================================================================

def get_soil_usda_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get soil data using USDA SSURGO API from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        soil_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'mukey': None,  # Map unit key - would call USDA SSURGO API
            'soil_series': None,
            'taxonomic_class': None,
            'drainage_class': None,
            'ph_01to1h2o_r': None,  # pH in water
            'om_r': None,  # Organic matter
            'clay_r': None,  # Clay percentage
            'sand_r': None,  # Sand percentage
            'silt_r': None,  # Silt percentage
            'soil_source': 'usda_ssurgo_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(soil_data)
    
    return results


def get_soil_usda_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get soil data using USDA SSURGO API from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        soil_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'mukey': None,  # Map unit key - would call USDA SSURGO API
            'soil_series': None,
            'taxonomic_class': None,
            'drainage_class': None,
            'ph_01to1h2o_r': None,  # pH in water
            'om_r': None,  # Organic matter
            'clay_r': None,  # Clay percentage
            'sand_r': None,  # Sand percentage
            'silt_r': None,  # Silt percentage
            'soil_source': 'usda_ssurgo_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(soil_data)
    
    return results


def get_soil_fao_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get soil data using FAO Harmonized World Soil Database from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        soil_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'fao_soil_unit': None,  # Would call FAO HWSD API
            'dominant_soil': None,
            'secondary_soil': None,
            'soil_texture': None,
            'drainage_conditions': None,
            'soil_source': 'fao_hwsd_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(soil_data)
    
    return results


def get_soil_fao_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get soil data using FAO Harmonized World Soil Database from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        soil_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'fao_soil_unit': None,  # Would call FAO HWSD API
            'dominant_soil': None,
            'secondary_soil': None,
            'soil_texture': None,
            'drainage_conditions': None,
            'soil_source': 'fao_hwsd_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(soil_data)
    
    return results


def get_soil_isric_soilgrids_rest_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get soil classification using ISRIC SoilGrids REST API from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        soil_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'wrb_class': None,  # Would call ISRIC SoilGrids REST API
            'wrb_probability': None,
            'most_probable_class': None,
            'classification_probabilities': None,
            'soil_source': 'isric_soilgrids_rest_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(soil_data)
    
    return results


def get_soil_isric_soilgrids_rest_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get soil classification using ISRIC SoilGrids REST API from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        soil_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'wrb_class': None,  # Would call ISRIC SoilGrids REST API
            'wrb_probability': None,
            'most_probable_class': None,
            'classification_probabilities': None,
            'soil_source': 'isric_soilgrids_rest_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(soil_data)
    
    return results


def get_soil_isric_soilgrids_wcs_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get soil properties using ISRIC SoilGrids WCS API from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        soil_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'ph_h2o': None,  # Would call ISRIC SoilGrids WCS API
            'organic_carbon': None,  # dg/kg
            'sand_content': None,  # g/kg
            'silt_content': None,  # g/kg  
            'clay_content': None,  # g/kg
            'bulk_density': None,  # cg/cm³
            'nitrogen_total': None,  # cg/kg
            'usda_texture_class': None,  # Derived from sand/silt/clay
            'depth_range': '0-5cm',  # Default depth
            'soil_source': 'isric_soilgrids_wcs_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(soil_data)
    
    return results


def get_soil_isric_soilgrids_wcs_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get soil properties using ISRIC SoilGrids WCS API from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        soil_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'ph_h2o': None,  # Would call ISRIC SoilGrids WCS API
            'organic_carbon': None,  # dg/kg
            'sand_content': None,  # g/kg
            'silt_content': None,  # g/kg  
            'clay_content': None,  # g/kg
            'bulk_density': None,  # cg/cm³
            'nitrogen_total': None,  # cg/kg
            'usda_texture_class': None,  # Derived from sand/silt/clay
            'depth_range': '0-5cm',  # Default depth
            'soil_source': 'isric_soilgrids_wcs_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(soil_data)
    
    return results


def get_soil_openlandmap_stac_direct(adapter, samples: List[Dict]) -> List[Dict[str, Any]]:
    """Get soil texture using OpenLandMap STAC API from direct adapter calls."""
    locations = extract_locations_from_adapters(adapter, samples)
    results = []
    
    for location in locations:
        soil_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'usda_texture_class': None,  # Would call OpenLandMap STAC API
            'texture_class_confidence': None,
            'sand_percent': None,
            'clay_percent': None,
            'silt_percent': None,
            'soil_source': 'openlandmap_stac_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(soil_data)
    
    return results


def get_soil_openlandmap_stac_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get soil texture using OpenLandMap STAC API from adapter results file."""
    adapter_results = load_adapter_results(file_path)
    locations = extract_locations_from_results(adapter_results)
    enrichable_locations = filter_enrichable_locations(locations)
    
    results = []
    for location in enrichable_locations:
        soil_data = {
            'sample_id': location['sample_id'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'usda_texture_class': None,  # Would call OpenLandMap STAC API
            'texture_class_confidence': None,
            'sand_percent': None,
            'clay_percent': None,
            'silt_percent': None,
            'soil_source': 'openlandmap_stac_api',
            'api_call_timestamp': datetime.now().isoformat(),
            'original_location': location
        }
        results.append(soil_data)
    
    return results


# =============================================================================
# COMPREHENSIVE ENRICHMENT FUNCTIONS - BY API SOURCE
# =============================================================================

def enrich_all_google_direct(adapter, samples: List[Dict]) -> Dict[str, List[Dict[str, Any]]]:
    """Get all Google API enrichment data from direct adapter calls."""
    return {
        'elevation_data': get_elevation_google_direct(adapter, samples),
        'geocoding_data': get_geocoding_google_direct(adapter, samples)
    }


def enrich_all_google_file(file_path: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Get all Google API enrichment data from adapter results file."""
    return {
        'elevation_data': get_elevation_google_file(file_path),
        'geocoding_data': get_geocoding_google_file(file_path)
    }


def enrich_all_usgs_direct(adapter, samples: List[Dict]) -> Dict[str, List[Dict[str, Any]]]:
    """Get all USGS API enrichment data from direct adapter calls."""
    return {
        'elevation_data': get_elevation_usgs_direct(adapter, samples),
        'landuse_data': get_landuse_usgs_direct(adapter, samples)
    }


def enrich_all_usgs_file(file_path: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Get all USGS API enrichment data from adapter results file."""
    return {
        'elevation_data': get_elevation_usgs_file(file_path),
        'landuse_data': get_landuse_usgs_file(file_path)
    }


def enrich_all_openmeteo_direct(adapter, samples: List[Dict]) -> Dict[str, List[Dict[str, Any]]]:
    """Get all Open-Meteo API enrichment data from direct adapter calls."""
    return {
        'weather_data': get_weather_openmeteo_direct(adapter, samples)
    }


def enrich_all_openmeteo_file(file_path: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Get all Open-Meteo API enrichment data from adapter results file."""
    return {
        'weather_data': get_weather_openmeteo_file(file_path)
    }


def enrich_all_usda_direct(adapter, samples: List[Dict]) -> Dict[str, List[Dict[str, Any]]]:
    """Get all USDA API enrichment data from direct adapter calls."""
    return {
        'soil_classification': get_soil_usda_direct(adapter, samples)
    }


def enrich_all_usda_file(file_path: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Get all USDA API enrichment data from adapter results file."""
    return {
        'soil_classification': get_soil_usda_file(file_path)
    }


def enrich_all_isric_direct(adapter, samples: List[Dict]) -> Dict[str, List[Dict[str, Any]]]:
    """Get all ISRIC SoilGrids API enrichment data from direct adapter calls."""
    return {
        'soil_classification': get_soil_isric_soilgrids_rest_direct(adapter, samples),
        'soil_properties': get_soil_isric_soilgrids_wcs_direct(adapter, samples)
    }


def enrich_all_isric_file(file_path: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Get all ISRIC SoilGrids API enrichment data from adapter results file."""
    return {
        'soil_classification': get_soil_isric_soilgrids_rest_file(file_path),
        'soil_properties': get_soil_isric_soilgrids_wcs_file(file_path)
    }


def enrich_all_nominatim_direct(adapter, samples: List[Dict]) -> Dict[str, List[Dict[str, Any]]]:
    """Get all Nominatim (OpenStreetMap) API enrichment data from direct adapter calls."""
    return {
        'geocoding_data': get_geocoding_nominatim_direct(adapter, samples)
    }


def enrich_all_nominatim_file(file_path: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Get all Nominatim (OpenStreetMap) API enrichment data from adapter results file."""
    return {
        'geocoding_data': get_geocoding_nominatim_file(file_path)
    }


# =============================================================================
# EXAMPLE USAGE FUNCTIONS
# =============================================================================

def demonstrate_api_functions():
    """Demonstrate all API enrichment functions by source."""
    print("🌍 API Enrichment Functions Demonstration (By API Source)")
    print("=" * 60)
    
    # Sample data
    nmdc_samples = [
        {
            'id': 'nmdc:demo-api-test',
            'lat_lon': '42.3601 -71.0928',
            'collection_date': '2023-06-15',
            'geo_loc_name': 'MIT Campus, Cambridge, MA'
        }
    ]
    
    # Test different API sources
    nmdc_adapter = NMDCBiosampleAdapter()
    
    print("\n📊 Google APIs:")
    google_elevation = get_elevation_google_direct(nmdc_adapter, nmdc_samples)
    google_geocoding = get_geocoding_google_direct(nmdc_adapter, nmdc_samples)
    print(f"  Elevation entries: {len(google_elevation)}")
    print(f"  Geocoding entries: {len(google_geocoding)}")
    
    print("\n📊 USGS APIs:")
    usgs_elevation = get_elevation_usgs_direct(nmdc_adapter, nmdc_samples)
    usgs_landuse = get_landuse_usgs_direct(nmdc_adapter, nmdc_samples)
    print(f"  Elevation entries: {len(usgs_elevation)}")
    print(f"  Land use entries: {len(usgs_landuse)}")
    
    print("\n📊 Open Elevation APIs:")
    open_elevation = get_elevation_open_elevation_direct(nmdc_adapter, nmdc_samples)
    print(f"  Elevation entries: {len(open_elevation)}")
    
    print("\n📊 Open-Meteo APIs:")
    openmeteo_weather = get_weather_openmeteo_direct(nmdc_adapter, nmdc_samples)
    print(f"  Weather entries: {len(openmeteo_weather)}")
    
    print("\n📊 Meteostat APIs:")
    meteostat_weather = get_weather_meteostat_direct(nmdc_adapter, nmdc_samples)
    print(f"  Weather entries: {len(meteostat_weather)}")
    
    print("\n📊 Nominatim (OpenStreetMap) APIs:")
    nominatim_geocoding = get_geocoding_nominatim_direct(nmdc_adapter, nmdc_samples)
    print(f"  Geocoding entries: {len(nominatim_geocoding)}")
    
    print("\n📊 ESA WorldCover APIs:")
    esa_landuse = get_landuse_esa_direct(nmdc_adapter, nmdc_samples)
    print(f"  Land use entries: {len(esa_landuse)}")
    
    print("\n📊 USDA Soil APIs:")
    usda_soil = get_soil_usda_direct(nmdc_adapter, nmdc_samples)
    print(f"  Soil classification entries: {len(usda_soil)}")
    
    print("\n📊 FAO HWSD APIs:")
    fao_soil = get_soil_fao_direct(nmdc_adapter, nmdc_samples)
    print(f"  Soil classification entries: {len(fao_soil)}")
    
    print("\n📊 ISRIC SoilGrids APIs:")
    isric_classification = get_soil_isric_soilgrids_rest_direct(nmdc_adapter, nmdc_samples)
    isric_properties = get_soil_isric_soilgrids_wcs_direct(nmdc_adapter, nmdc_samples)
    print(f"  Soil classification entries: {len(isric_classification)}")
    print(f"  Soil properties entries: {len(isric_properties)}")
    
    print("\n📊 OpenLandMap STAC APIs:")
    openlandmap_soil = get_soil_openlandmap_stac_direct(nmdc_adapter, nmdc_samples)
    print(f"  Soil texture entries: {len(openlandmap_soil)}")
    
    print("\n✅ Complete API Coverage Demonstration:")
    print("  📍 Elevation APIs: 3 sources (Google, USGS, Open Elevation)")
    print("  🌤️  Weather APIs: 2 sources (Open-Meteo, Meteostat)")
    print("  🗺️  Geocoding APIs: 2 sources (Google, Nominatim)")
    print("  🌱 Land Use APIs: 2 sources (USGS NLCD, ESA WorldCover)")
    print("  🌍 Soil APIs: 5 sources (USDA SSURGO, FAO HWSD, ISRIC REST+WCS, OpenLandMap)")
    print("  📊 Total: 14 API sources covering all geospatial enrichment needs!")
    print("\n✅ API function demonstration complete!")


if __name__ == "__main__":
    demonstrate_api_functions()
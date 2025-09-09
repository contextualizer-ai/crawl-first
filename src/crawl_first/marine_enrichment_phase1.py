#!/usr/bin/env python3

import os
import json
import logging
import requests
import xarray as xr
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

@dataclass
class MarineResult:
    """Marine/oceanographic enrichment result for Phase 1"""
    success: bool
    provider: str
    data: Dict[str, Any]
    error: Optional[str] = None
    point_distance_km: Optional[float] = None
    data_source: Optional[str] = None

def get_marine_enrichment_phase1(
    lat: float, 
    lon: float, 
    collection_date: Optional[str] = None,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Phase 1 Marine Enrichment: SST, Chlorophyll-a, and Bathymetry
    
    Uses API/On-demand approach with ERDDAP and WCS point queries for efficient
    high-throughput marine biosample enrichment with 25-year historical coverage.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees  
        collection_date: Date in YYYY-MM-DD format (optional)
        timeout: Request timeout in seconds
        
    Returns:
        Dict containing Phase 1 marine data:
        - sea_surface_temperature: NOAA OISST v2.1 via ERDDAP
        - chlorophyll_a: ESA OC-CCI v6 via ERDDAP (NEFSC)
        - bathymetry: GEBCO global grid via WCS
    """
    results = {
        "phase": "1",
        "description": "Core marine parameters via API/On-demand",
        "providers_attempted": [],
        "successful_providers": [],
        "failed_providers": [],
        "data": {}
    }
    
    # Phase 1 Provider Configuration
    providers = [
        {
            "name": "noaa_oisst_sst",
            "function": _get_noaa_oisst_sst,
            "description": "NOAA OISST v2.1 Sea Surface Temperature",
            "schema_slots": ["temp", "sampleCollectionTemperature"],
            "coverage": "1981-present",
            "priority": "HIGHEST"
        },
        {
            "name": "esa_oc_cci_chlorophyll", 
            "function": _get_esa_oc_cci_chlorophyll,
            "description": "ESA Ocean Colour CCI Chlorophyll-a",
            "schema_slots": ["chlorophyll"],
            "coverage": "1997-present", 
            "priority": "HIGHEST"
        },
        {
            "name": "gebco_bathymetry",
            "function": _get_gebco_bathymetry,
            "description": "GEBCO Global Bathymetry",
            "schema_slots": ["tot_depth_water_col", "depthInMeters", "elev"],
            "coverage": "Static global",
            "priority": "HIGHEST"
        }
    ]
    
    # Execute Phase 1 providers
    for provider_config in providers:
        provider_name = provider_config["name"]
        results["providers_attempted"].append(provider_name)
        
        try:
            logger.info(f"Fetching marine data from {provider_name} for ({lat}, {lon}) on {collection_date}")
            
            result = provider_config["function"](lat, lon, collection_date, timeout)
            
            if result.success:
                results["successful_providers"].append(provider_name)
                results["data"][provider_name] = {
                    "provider": result.provider,
                    "success": True,
                    "data": result.data,
                    "data_source": result.data_source,
                    "point_distance_km": result.point_distance_km,
                    "schema_slots": provider_config["schema_slots"],
                    "description": provider_config["description"],
                    "coverage": provider_config["coverage"]
                }
                logger.info(f"Successfully fetched {provider_name} data")
            else:
                results["failed_providers"].append(provider_name)
                results["data"][provider_name] = {
                    "provider": result.provider,
                    "success": False,
                    "error": result.error,
                    "description": provider_config["description"]
                }
                logger.warning(f"Failed to fetch {provider_name} data: {result.error}")
                
        except Exception as e:
            results["failed_providers"].append(provider_name)
            results["data"][provider_name] = {
                "provider": provider_name,
                "success": False,
                "error": f"Exception during {provider_name} fetch: {str(e)}",
                "description": provider_config["description"]
            }
            logger.error(f"Exception in {provider_name}: {e}")
    
    # Calculate overall success metrics
    total_providers = len(providers)
    successful_count = len(results["successful_providers"])
    results["success_rate"] = successful_count / total_providers if total_providers > 0 else 0
    results["overall_success"] = successful_count > 0
    
    logger.info(f"Phase 1 marine enrichment completed: {successful_count}/{total_providers} providers successful")
    
    return results

def _get_noaa_oisst_sst(lat: float, lon: float, collection_date: Optional[str], timeout: int) -> MarineResult:
    """
    Fetch Sea Surface Temperature from NOAA OISST v2.1 via ERDDAP
    
    Coverage: 1981-09-01 to present, 0.25° resolution, daily
    Schema slots: temp, sampleCollectionTemperature
    """
    try:
        # ERDDAP endpoint for NOAA OISST v2.1 (current dataset with 1981-present coverage)
        base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
        dataset_id = "ncdcOisst21Agg"
        
        # Format date for ERDDAP (ISO format with Z suffix)
        if collection_date:
            try:
                date_obj = datetime.strptime(collection_date, "%Y-%m-%d")
                erddap_date = date_obj.strftime("%Y-%m-%dT12:00:00Z")  # OISST uses 12:00:00Z
            except ValueError:
                logger.warning(f"Invalid date format {collection_date}, using latest available")
                erddap_date = "last"
        else:
            erddap_date = "last"
        
        # Convert longitude to 0-360 format for OISST
        lon_360 = lon if lon >= 0 else lon + 360
        
        # Build ERDDAP griddap URL for point query
        # Format: dataset.nc?variable[time][zlev][lat][lon]
        params = {
            "sst": f"[({erddap_date}):1:({erddap_date})][0:1:0][({lat}):1:({lat})][({lon_360}):1:({lon_360})]"
        }
        
        url = f"{base_url}/{dataset_id}.nc"
        
        logger.debug(f"NOAA OISST ERDDAP request: {url} with params: {params}")
        
        # Use xarray to read NetCDF directly from ERDDAP
        query_url = f"{url}?sst{params['sst']}"
        
        response = requests.get(query_url, timeout=timeout)
        response.raise_for_status()
        
        # Save response to temporary file and read with xarray
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        try:
            # Read NetCDF data
            ds = xr.open_dataset(tmp_path)
            
            # Extract SST value (convert from K to C if needed)
            sst_value = float(ds['sst'].values.flatten()[0])
            
            # Check if temperature is in Kelvin (typical for OISST)
            if sst_value > 100:  # Likely Kelvin
                sst_celsius = sst_value - 273.15
            else:
                sst_celsius = sst_value
            
            # Extract metadata
            actual_time = ds['time'].values[0]
            actual_lat = float(ds['latitude'].values[0])
            actual_lon_360 = float(ds['longitude'].values[0])
            
            # Convert longitude back to -180 to 180 format
            actual_lon = actual_lon_360 if actual_lon_360 <= 180 else actual_lon_360 - 360
            
            # Calculate distance from requested point
            point_distance_km = _calculate_distance(lat, lon, actual_lat, actual_lon)
            
            ds.close()
            
            return MarineResult(
                success=True,
                provider="noaa_oisst_sst",
                data={
                    "sea_surface_temperature_c": round(sst_celsius, 2),
                    "sst_kelvin": round(sst_value, 2),
                    "measurement_date": str(actual_time)[:10],
                    "actual_coordinates": {"lat": actual_lat, "lon": actual_lon},
                    "spatial_resolution": "0.25°",
                    "temporal_resolution": "Daily"
                },
                data_source="NOAA OISST v2.1 via NCEI ERDDAP",
                point_distance_km=point_distance_km
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except requests.RequestException as e:
        return MarineResult(
            success=False,
            provider="noaa_oisst_sst", 
            data={},
            error=f"ERDDAP request failed: {str(e)}"
        )
    except Exception as e:
        return MarineResult(
            success=False,
            provider="noaa_oisst_sst",
            data={},
            error=f"SST processing error: {str(e)}"
        )

def _get_esa_oc_cci_chlorophyll(lat: float, lon: float, collection_date: Optional[str], timeout: int) -> MarineResult:
    """
    Fetch Chlorophyll-a from ESA Ocean Colour CCI v6 via ERDDAP (NOAA NEFSC)
    
    Coverage: 1997-09-04 to present, ~1 km resolution, daily
    Schema slots: chlorophyll
    """
    try:
        # ERDDAP endpoint for ESA OC-CCI at NOAA NEFSC (confirmed dataset ID)
        base_url = "https://comet.nefsc.noaa.gov/erddap/griddap"
        dataset_id = "occci_v6_daily_1km"
        
        # Format date for ERDDAP
        if collection_date:
            try:
                date_obj = datetime.strptime(collection_date, "%Y-%m-%d")
                erddap_date = date_obj.strftime("%Y-%m-%dT00:00:00Z")
                
                # Check if date is before OC-CCI coverage (1997-09-04)
                if date_obj < datetime(1997, 9, 4):
                    return MarineResult(
                        success=False,
                        provider="esa_oc_cci_chlorophyll",
                        data={},
                        error=f"Date {collection_date} is before OC-CCI coverage (starts 1997-09-04)"
                    )
            except ValueError:
                logger.warning(f"Invalid date format {collection_date}, using latest available")
                erddap_date = "last"
        else:
            erddap_date = "last"
        
        # Build ERDDAP griddap URL for chlor_a (confirmed variable name)
        params = {
            "chlor_a": f"[({erddap_date}):1:({erddap_date})][({lat}):1:({lat})][({lon}):1:({lon})]"
        }
        
        url = f"{base_url}/{dataset_id}.nc"
        query_url = f"{url}?chlor_a{params['chlor_a']}"
        
        logger.debug(f"ESA OC-CCI ERDDAP request: {query_url}")
        
        response = requests.get(query_url, timeout=timeout)
        response.raise_for_status()
        
        # Process NetCDF response with xarray
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        try:
            ds = xr.open_dataset(tmp_path)
            
            # Extract chlorophyll-a value (mg/m³)
            chl_value = float(ds['chlor_a'].values.flatten()[0])
            
            # Extract metadata
            actual_time = ds['time'].values[0]
            actual_lat = float(ds['latitude'].values[0])
            actual_lon = float(ds['longitude'].values[0])
            
            point_distance_km = _calculate_distance(lat, lon, actual_lat, actual_lon)
            
            ds.close()
            
            return MarineResult(
                success=True,
                provider="esa_oc_cci_chlorophyll",
                data={
                    "chlorophyll_a_mg_m3": round(chl_value, 4),
                    "measurement_date": str(actual_time)[:10],
                    "actual_coordinates": {"lat": actual_lat, "lon": actual_lon},
                    "spatial_resolution": "~1 km",
                    "temporal_resolution": "Daily"
                },
                data_source="ESA Ocean Colour CCI v6 via NOAA NEFSC ERDDAP",
                point_distance_km=point_distance_km
            )
            
        finally:
            os.unlink(tmp_path)
            
    except requests.RequestException as e:
        return MarineResult(
            success=False,
            provider="esa_oc_cci_chlorophyll",
            data={},
            error=f"ERDDAP request failed: {str(e)}"
        )
    except Exception as e:
        return MarineResult(
            success=False,
            provider="esa_oc_cci_chlorophyll",
            data={},
            error=f"Chlorophyll processing error: {str(e)}"
        )

def _get_gebco_bathymetry(lat: float, lon: float, collection_date: Optional[str], timeout: int) -> MarineResult:
    """
    Fetch Bathymetry/Depth from GEBCO Global Grid via WMS GetFeatureInfo
    
    Coverage: Global static, 15 arc-second resolution
    Schema slots: tot_depth_water_col, depthInMeters, elev
    """
    try:
        # GEBCO WMS endpoint (corrected URL)
        base_url = "https://wms.gebco.net/mapserv"
        
        # WMS GetFeatureInfo request for a single point
        # GEBCO 2024 Grid layer
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetFeatureInfo", 
            "LAYERS": "GEBCO_2024_Grid",
            "CRS": "EPSG:4326",
            "BBOX": f"{lat-0.01},{lon-0.01},{lat+0.01},{lon+0.01}",
            "WIDTH": "101",
            "HEIGHT": "101", 
            "I": "50",  # Center pixel X
            "J": "50",  # Center pixel Y
            "INFO_FORMAT": "text/plain",
            "FEATURE_COUNT": "1"
        }
        
        logger.debug(f"GEBCO WCS request: {base_url} with params: {params}")
        
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        
        # Process text response from GetFeatureInfo
        response_text = response.text.strip()
        
        # Parse elevation value from response
        # GEBCO GetFeatureInfo typically returns: "Value: -1234.5" or similar format
        try:
            if "Value:" in response_text:
                elevation_str = response_text.split("Value:")[-1].strip()
                center_elevation = float(elevation_str)
            elif response_text.replace("-", "").replace(".", "").isdigit():
                # Direct numeric value
                center_elevation = float(response_text)
            else:
                # Try to extract any numeric value
                import re
                numbers = re.findall(r'-?\d+\.?\d*', response_text)
                if numbers:
                    center_elevation = float(numbers[0])
                else:
                    return MarineResult(
                        success=False,
                        provider="gebco_bathymetry",
                        data={},
                        error=f"Could not parse elevation from response: {response_text}"
                    )
        except (ValueError, IndexError) as e:
            return MarineResult(
                success=False,
                provider="gebco_bathymetry", 
                data={},
                error=f"Error parsing elevation value: {str(e)}, response: {response_text}"
            )
        
        # Calculate distance (minimal since we're querying exact coordinates)
        point_distance_km = 0.0
        
        # Determine if marine (negative elevation = below sea level)
        is_marine = center_elevation < 0
        
        return MarineResult(
            success=True,
            provider="gebco_bathymetry",
            data={
                "elevation_m": round(center_elevation, 1),
                "water_depth_m": round(abs(center_elevation), 1) if is_marine else None,
                "is_marine": is_marine,
                "actual_coordinates": {"lat": lat, "lon": lon},
                "spatial_resolution": "15 arc-second",
                "note": "Negative elevation = below sea level (marine), positive = above sea level (land)"
            },
            data_source="GEBCO 2024 Grid via WMS GetFeatureInfo",
            point_distance_km=point_distance_km
        )
            
    except requests.RequestException as e:
        return MarineResult(
            success=False,
            provider="gebco_bathymetry",
            data={},
            error=f"WCS request failed: {str(e)}"
        )
    except Exception as e:
        return MarineResult(
            success=False,
            provider="gebco_bathymetry",
            data={},
            error=f"Bathymetry processing error: {str(e)}"
        )

def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula"""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in kilometers
    
    return round(c * r, 2)

def test_phase1_marine_enrichment():
    """Test Phase 1 marine enrichment with known marine coordinates"""
    
    # Test coordinates: Deep ocean in Pacific (truly marine)
    test_lat = -30.0  # South Pacific Ocean
    test_lon = -120.0  # Far from any coast
    test_date = "2010-06-15"  # Within OISST coverage (2002-2011)
    
    print(f"Testing Phase 1 Marine Enrichment at ({test_lat}, {test_lon}) on {test_date}")
    print("=" * 80)
    
    results = get_marine_enrichment_phase1(test_lat, test_lon, test_date)
    
    print(f"Overall Success: {results['overall_success']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Successful Providers: {results['successful_providers']}")
    print(f"Failed Providers: {results['failed_providers']}")
    print()
    
    for provider_name, provider_data in results['data'].items():
        print(f"Provider: {provider_name}")
        print(f"  Success: {provider_data['success']}")
        print(f"  Description: {provider_data['description']}")
        
        if provider_data['success']:
            print(f"  Data Source: {provider_data['data_source']}")
            print(f"  Schema Slots: {provider_data['schema_slots']}")
            print(f"  Distance: {provider_data['point_distance_km']} km")
            print(f"  Data: {json.dumps(provider_data['data'], indent=4)}")
        else:
            print(f"  Error: {provider_data.get('error', 'Unknown error')}")
        print()

if __name__ == "__main__":
    # Install required dependencies
    print("Phase 1 Marine Enrichment Implementation")
    print("Required dependencies: requests, xarray, pandas")
    
    # Run test
    test_phase1_marine_enrichment()
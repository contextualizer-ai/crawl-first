#!/usr/bin/env python3

import os
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class MarineResult:
    """Marine/oceanographic enrichment result"""
    success: bool
    provider: str
    data: Dict[str, Any]
    error: Optional[str] = None

def get_marine_enrichment_multi_provider(
    lat: float, 
    lon: float, 
    collection_date: Optional[str] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Get marine/oceanographic data from multiple providers for coastal biosample enrichment.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees  
        collection_date: Date in YYYY-MM-DD format (optional)
        timeout: Request timeout in seconds
        
    Returns:
        Dict containing multi-provider marine data
    """
    providers = ["open_meteo_marine", "noaa_erddap", "longhurst_provinces"]
    results = {}
    
    # Query all providers
    for provider in providers:
        try:
            if provider == "open_meteo_marine":
                result = _get_open_meteo_marine(lat, lon, collection_date, timeout)
            elif provider == "noaa_erddap":
                result = _get_noaa_erddap_marine(lat, lon, collection_date, timeout)
            elif provider == "longhurst_provinces":
                result = _get_longhurst_province(lat, lon, collection_date, timeout)
            else:
                result = MarineResult(False, provider, {}, f"Unknown provider: {provider}")
                
            results[provider] = {
                "provider": result.provider,
                "success": result.success,
                **result.data
            }
            if not result.success:
                results[provider]["error"] = result.error
                
        except Exception as e:
            logger.error(f"Error querying {provider}: {e}")
            results[provider] = {
                "provider": provider,
                "success": False,
                "error": str(e)
            }
    
    # Determine primary provider (first successful)
    successful_providers = [p for p in providers if results.get(p, {}).get("success", False)]
    primary_provider = successful_providers[0] if successful_providers else None
    
    # Build response following multi-provider pattern
    response = {
        "all_providers": results,
        "providers_queried": providers,
        "successful_providers": successful_providers,
        "primary_provider": primary_provider
    }
    
    # Add primary provider data to top level if available
    if primary_provider and results[primary_provider]["success"]:
        primary_data = results[primary_provider].copy()
        primary_data.pop("provider", None)
        primary_data.pop("success", None)
        primary_data.pop("error", None)
        response.update(primary_data)
        response["provider"] = primary_provider
        response["success"] = True
    else:
        response["success"] = False
        response["error"] = "All marine providers failed"
    
    return response

def _get_open_meteo_marine(
    lat: float, 
    lon: float, 
    collection_date: Optional[str] = None, 
    timeout: int = 30
) -> MarineResult:
    """
    Get marine data from Open-Meteo Marine Weather API.
    
    Provides sea surface temperature, wave height, ocean currents.
    """
    try:
        # Use collection date if provided, otherwise current date
        if collection_date:
            start_date = end_date = collection_date
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            start_date = end_date = today
            
        # Open-Meteo Marine API parameters
        base_url = "https://marine-api.open-meteo.com/v1/marine"
        
        # Marine variables of interest for biosamples
        hourly_vars = [
            "wave_height",
            "wave_direction", 
            "wave_period",
            "ocean_current_velocity",
            "ocean_current_direction",
            "sea_surface_temperature"
        ]
        
        # Note: Many daily variables are not available in Open-Meteo Marine API
        # We'll calculate daily stats from hourly data instead
        daily_vars = []  # Start with empty list to avoid API errors
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_vars),
            "timezone": "UTC",
            "format": "json"
        }
        
        # Only add daily params if we have daily variables
        if daily_vars:
            params["daily"] = ",".join(daily_vars)
        
        logger.info(f"Querying Open-Meteo Marine API for {lat}, {lon} on {start_date}")
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        
        # Process hourly data into DataFrame for consistency
        hourly_df = None
        if "hourly" in data and data["hourly"]:
            hourly_data = data["hourly"]
            if "time" in hourly_data:
                hourly_df = pd.DataFrame(hourly_data)
                hourly_df["time"] = pd.to_datetime(hourly_df["time"])
                hourly_df.set_index("time", inplace=True)
        
        # Calculate daily statistics from hourly data
        daily_stats = {}
        if hourly_df is not None and not hourly_df.empty:
            # Sea surface temperature statistics
            if "sea_surface_temperature" in hourly_df.columns:
                sst_data = hourly_df["sea_surface_temperature"].dropna()
                if not sst_data.empty:
                    daily_stats["sea_surface_temperature"] = {
                        "min_c": float(sst_data.min()),
                        "max_c": float(sst_data.max()),
                        "mean_c": float(sst_data.mean()),
                        "std_c": float(sst_data.std())
                    }
            
            # Wave height statistics
            if "wave_height" in hourly_df.columns:
                wave_data = hourly_df["wave_height"].dropna()
                if not wave_data.empty:
                    daily_stats["wave_height"] = {
                        "min_m": float(wave_data.min()),
                        "max_m": float(wave_data.max()),
                        "mean_m": float(wave_data.mean()),
                        "std_m": float(wave_data.std())
                    }
            
            # Ocean current velocity statistics
            if "ocean_current_velocity" in hourly_df.columns:
                current_data = hourly_df["ocean_current_velocity"].dropna()
                if not current_data.empty:
                    daily_stats["ocean_current_velocity"] = {
                        "min_ms": float(current_data.min()),
                        "max_ms": float(current_data.max()),
                        "mean_ms": float(current_data.mean()),
                        "std_ms": float(current_data.std())
                    }
        
        # Process daily data
        daily_data = data.get("daily", {})
        
        result_data = {
            "source_url": response.url,
            "date": start_date,
            "coordinates": {"lat": lat, "lon": lon},
            "spatial_resolution": "0.08° (~8km)",
            "temporal_resolution": "hourly",
            "data_source": "Open-Meteo Marine Weather API",
            "daily_statistics": daily_stats,
            "raw_daily": daily_data,
            "hourly_count": len(hourly_df) if hourly_df is not None else 0,
            "variables_available": len([v for v in hourly_vars if v in data.get("hourly", {})]),
            "variables_requested": len(hourly_vars)
        }
        
        # Add hourly DataFrame as string for JSON serialization
        if hourly_df is not None:
            result_data["hourly_df"] = str(hourly_df.head(24))  # First 24 hours for inspection
        
        return MarineResult(True, "open_meteo_marine", result_data)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for Open-Meteo Marine: {e}")
        return MarineResult(False, "open_meteo_marine", {}, str(e))
    except Exception as e:
        logger.error(f"Error processing Open-Meteo Marine data: {e}")
        return MarineResult(False, "open_meteo_marine", {}, str(e))

def _get_noaa_erddap_marine(
    lat: float, 
    lon: float, 
    collection_date: Optional[str] = None, 
    timeout: int = 30
) -> MarineResult:
    """
    Get marine data from NOAA ERDDAP servers.
    
    Focuses on chlorophyll-a and additional SST data.
    """
    try:
        # For this implementation, we'll query VIIRS chlorophyll data
        # ERDDAP griddap endpoint for VIIRS chlorophyll
        base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
        
        # Use recent VIIRS dataset (adjust based on available data)
        dataset_id = "erdVHNchla8day"  # VIIRS chlorophyll 8-day composite
        
        # Date handling - ERDDAP expects specific format
        if collection_date:
            try:
                # Convert to datetime and format for ERDDAP
                date_obj = datetime.strptime(collection_date, "%Y-%m-%d")
                erddap_date = date_obj.strftime("%Y-%m-%dT12:00:00Z")
            except ValueError:
                logger.warning(f"Invalid date format: {collection_date}, using current date")
                erddap_date = datetime.now().strftime("%Y-%m-%dT12:00:00Z")
        else:
            erddap_date = datetime.now().strftime("%Y-%m-%dT12:00:00Z")
        
        # Build ERDDAP query URL for point data
        # Format: [dataset].csv?var1[time_constraint][lat_constraint][lon_constraint]
        lat_constraint = f"[({lat}):1:({lat})]"
        lon_constraint = f"[({lon}):1:({lon})]"
        time_constraint = f"[({erddap_date}):1:({erddap_date})]"
        
        query_url = f"{base_url}/{dataset_id}.csv?chlor_a{time_constraint}{lat_constraint}{lon_constraint}"
        
        logger.info(f"Querying NOAA ERDDAP for chlorophyll at {lat}, {lon} on {collection_date}")
        
        response = requests.get(query_url, timeout=timeout)
        response.raise_for_status()
        
        # Parse CSV response
        lines = response.text.strip().split('\n')
        if len(lines) < 3:  # Header + units + data
            return MarineResult(False, "noaa_erddap", {}, "No data returned from ERDDAP")
        
        headers = lines[0].split(',')
        units = lines[1].split(',')
        
        # Process data rows
        data_rows = []
        for line in lines[2:]:
            if line.strip():
                data_rows.append(line.split(','))
        
        if not data_rows:
            return MarineResult(False, "noaa_erddap", {}, "No chlorophyll data available for this location/date")
        
        # Extract chlorophyll value(s)
        chlorophyll_values = []
        for row in data_rows:
            if len(row) > 3:  # time, lat, lon, chlor_a
                try:
                    chl_val = float(row[3])
                    if chl_val > 0:  # Valid chlorophyll values are positive
                        chlorophyll_values.append(chl_val)
                except (ValueError, IndexError):
                    continue
        
        if not chlorophyll_values:
            return MarineResult(False, "noaa_erddap", {}, "No valid chlorophyll data found")
        
        # Calculate statistics if multiple values
        chl_stats = {
            "mean_mg_m3": float(sum(chlorophyll_values) / len(chlorophyll_values)),
            "min_mg_m3": float(min(chlorophyll_values)),
            "max_mg_m3": float(max(chlorophyll_values)),
            "count": len(chlorophyll_values)
        }
        
        result_data = {
            "source_url": query_url,
            "dataset_id": dataset_id,
            "date": collection_date or datetime.now().strftime("%Y-%m-%d"),
            "coordinates": {"lat": lat, "lon": lon},
            "data_source": "NOAA ERDDAP - VIIRS Chlorophyll",
            "spatial_resolution": "4km",
            "temporal_resolution": "8-day composite",
            "chlorophyll_a": chl_stats,
            "units": "mg/m³",
            "methodology": "Satellite remote sensing - VIIRS",
            "data_rows_found": len(data_rows)
        }
        
        return MarineResult(True, "noaa_erddap", result_data)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for NOAA ERDDAP: {e}")
        return MarineResult(False, "noaa_erddap", {}, str(e))
    except Exception as e:
        logger.error(f"Error processing NOAA ERDDAP data: {e}")
        return MarineResult(False, "noaa_erddap", {}, str(e))

def _get_longhurst_province(
    lat: float, 
    lon: float, 
    collection_date: Optional[str] = None,
    timeout: int = 30
) -> MarineResult:
    """
    Get Longhurst biogeographical province from Marine Regions API.
    
    Longhurst provinces are ocean regions based on phytoplankton distribution
    and are critical for marine biogeographical context.
    """
    try:
        # Marine Regions API for Longhurst provinces
        base_url = "https://www.marineregions.org/rest/getGazetteerRecordsByLatLon.json"
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "typeID": "55"  # Longhurst provinces type ID
        }
        
        logger.info(f"Querying Marine Regions for Longhurst province at {lat}, {lon}")
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            # Try alternative approach - may be on land or no data
            return MarineResult(False, "longhurst_provinces", {}, "No Longhurst province data found for this location")
        
        # Extract Longhurst province information
        province_info = data[0] if isinstance(data, list) else data
        
        result_data = {
            "source_url": response.url,
            "coordinates": {"lat": lat, "lon": lon},
            "date": collection_date or datetime.now().strftime("%Y-%m-%d"),
            "data_source": "Marine Regions - Longhurst Provinces v4 (2010)",
            "province_id": province_info.get("MRGID"),
            "province_name": province_info.get("preferredGazetteerName"),
            "province_code": province_info.get("source"),
            "biome": _extract_longhurst_biome(province_info.get("preferredGazetteerName", "")),
            "ocean_basin": province_info.get("placeType"),
            "methodology": "Point-in-polygon lookup using Longhurst (1995, 1998, 2006) biogeographical classification",
            "description": "Biogeographical province based on phytoplankton distribution patterns",
            "citation": "Flanders Marine Institute (2009). Longhurst Provinces. Available online at https://www.marineregions.org/",
            "version": "v4 (March 2010)",
            "temporal_variability": "Boundaries are dynamic and move with seasonal/interannual changes"
        }
        
        return MarineResult(True, "longhurst_provinces", result_data)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for Longhurst provinces: {e}")
        return MarineResult(False, "longhurst_provinces", {}, str(e))
    except Exception as e:
        logger.error(f"Error processing Longhurst province data: {e}")
        return MarineResult(False, "longhurst_provinces", {}, str(e))

def _extract_longhurst_biome(province_name: str) -> str:
    """
    Extract the Longhurst biome from province name.
    
    The four principal biomes are: Polar, Westerlies, Trade winds, Coastal
    """
    if not province_name:
        return "unknown"
    
    name_lower = province_name.lower()
    
    if any(term in name_lower for term in ["polar", "arctic", "antarctic"]):
        return "Polar"
    elif any(term in name_lower for term in ["westerlies", "westerly"]):
        return "Westerlies"  
    elif any(term in name_lower for term in ["trade", "tropical", "equatorial"]):
        return "Trade winds"
    elif any(term in name_lower for term in ["coastal", "upwelling", "shelf"]):
        return "Coastal"
    else:
        return "unclassified"

def _determine_marine_relevance(lat: float, lon: float, distance_to_coast_km: float) -> Dict[str, Any]:
    """
    Determine if location is relevant for marine enrichment.
    
    Args:
        lat: Latitude
        lon: Longitude 
        distance_to_coast_km: Distance to nearest coastline in km
        
    Returns:
        Dict with relevance assessment
    """
    # Marine data is most relevant for coastal locations
    if distance_to_coast_km <= 10:
        relevance = "high"
        reason = "Coastal location - marine parameters highly relevant"
    elif distance_to_coast_km <= 50:
        relevance = "medium" 
        reason = "Near-coastal location - marine influence possible"
    elif distance_to_coast_km <= 100:
        relevance = "low"
        reason = "Inland location with potential marine climate influence"
    else:
        relevance = "minimal"
        reason = "Inland location - marine parameters minimally relevant"
    
    return {
        "relevance": relevance,
        "reason": reason,
        "distance_to_coast_km": distance_to_coast_km,
        "recommended_parameters": _get_recommended_marine_params(relevance)
    }

def _get_recommended_marine_params(relevance: str) -> List[str]:
    """Get recommended marine parameters based on location relevance."""
    if relevance == "high":
        return [
            "sea_surface_temperature",
            "chlorophyll_a", 
            "wave_height",
            "ocean_currents",
            "salinity",
            "turbidity"
        ]
    elif relevance == "medium":
        return [
            "sea_surface_temperature",
            "chlorophyll_a",
            "wave_height"  
        ]
    elif relevance == "low":
        return [
            "sea_surface_temperature"
        ]
    else:
        return []

if __name__ == "__main__":
    # Test with Yellowstone location (inland)
    test_lat = 44.428
    test_lon = -110.5885
    test_date = "2021-08-20"
    
    print("Testing marine enrichment...")
    result = get_marine_enrichment_multi_provider(test_lat, test_lon, test_date)
    print(json.dumps(result, indent=2))
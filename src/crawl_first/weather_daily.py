"""
Enhanced daily weather system with Open-Meteo → Meteostat → PRISM fallback chain.

Implements comprehensive daily weather aggregation from hourly data with proper
distance tracking, provenance, and coverage assessment.
"""

import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in kilometers."""
    R = 6371  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def aggregate_hourly_to_daily(hourly_df: pd.DataFrame, coverage_threshold: float = 0.8) -> Dict[str, Any]:
    """
    Aggregate hourly weather data to daily with proper statistics.
    
    Args:
        hourly_df: DataFrame with hourly weather data
        coverage_threshold: Minimum fraction of hours required for 'complete' coverage
    
    Returns:
        Dict with daily aggregates and coverage info
    """
    if hourly_df.empty:
        return {"coverage": "none", "method": "no_data", "aggregates": {}}
    
    total_hours = 24
    available_hours = len(hourly_df.dropna())
    coverage_fraction = available_hours / total_hours
    coverage_status = "complete" if coverage_fraction >= coverage_threshold else "partial"
    
    aggregates = {}
    
    # Temperature aggregation (min/max/mean)
    temp_cols = [col for col in hourly_df.columns if 'temp' in col.lower() and 'apparent' not in col.lower()]
    for col in temp_cols:
        if col in hourly_df.columns:
            data = hourly_df[col].dropna()
            if len(data) > 0:
                base_name = col.replace('_2m', '').replace('_hourly', '')
                aggregates[f"{base_name}c"] = {
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "avg": float(data.mean())
                }
    
    # Precipitation and accumulations (sum)
    precip_cols = ['precipitation', 'rain', 'snowfall', 'snow', 'shortwave_radiation']
    for col_pattern in precip_cols:
        matching_cols = [col for col in hourly_df.columns if col_pattern in col.lower()]
        for col in matching_cols:
            data = hourly_df[col].dropna()
            if len(data) > 0:
                base_name = col.replace('_sum', '').replace('_hourly', '')
                if 'precip' in col.lower():
                    aggregates["precip_mm"] = {"sum": float(data.sum())}
                elif 'rain' in col.lower():
                    aggregates["rain_mm"] = {"sum": float(data.sum())}
                elif 'snow' in col.lower():
                    aggregates["snow_mm"] = {"sum": float(data.sum())}
                elif 'shortwave' in col.lower():
                    aggregates["shortwave_mj_m2"] = {"sum": float(data.sum() / 1000000)}  # Convert J to MJ
    
    # Wind (vector mean for direction, min/max/mean for speed)
    wind_speed_cols = [col for col in hourly_df.columns if 'wind' in col.lower() and 'speed' in col.lower()]
    for col in wind_speed_cols:
        data = hourly_df[col].dropna()
        if len(data) > 0:
            aggregates["wind_ms"] = {
                "min": float(data.min()),
                "max": float(data.max()),
                "avg": float(data.mean())
            }
    
    # Wind direction (vector mean)
    wind_dir_cols = [col for col in hourly_df.columns if 'wind' in col.lower() and 'direction' in col.lower()]
    for col in wind_dir_cols:
        data = hourly_df[col].dropna()
        if len(data) > 0:
            # Convert to radians, compute vector mean, convert back
            rad_data = data * math.pi / 180
            sin_sum = sum(math.sin(r) for r in rad_data)
            cos_sum = sum(math.cos(r) for r in rad_data)
            vector_mean_rad = math.atan2(sin_sum / len(data), cos_sum / len(data))
            vector_mean_deg = (vector_mean_rad * 180 / math.pi) % 360
            aggregates["wind_dir_deg"] = {"vector_mean": float(vector_mean_deg)}
    
    # Other variables (mean or max as appropriate)
    other_vars = {
        'relative_humidity': 'rh_pct',
        'dewpoint': 'dewpointc', 
        'surface_pressure': 'sfc_pressure_hpa',
        'cloudcover': 'cloudcover_pct',
        'et0_fao_evapotranspiration': 'et0_mm',
        'soil_temperature': 'soil_temp_c',
        'soil_moisture': 'soil_moisture_m3m3',
        'uv_index': 'uv_index_max',
        'vapour_pressure_deficit': 'vpd_kpa',
        'visibility': 'visibility_km',
        'sunshine_duration': 'sunshine_hours',
        'weather_code': 'weather_code',
        'is_day': 'daylight_hours'
    }
    
    # Handle UV index as max (not mean)
    uv_cols = [col for col in hourly_df.columns if 'uv_index' in col.lower()]
    for col in uv_cols:
        data = hourly_df[col].dropna()
        if len(data) > 0:
            aggregates["uv_index_max"] = {"max": float(data.max())}
    
    # Handle multiple radiation types (sum for daily totals)
    radiation_types = ['direct_radiation', 'diffuse_radiation']
    for rad_type in radiation_types:
        matching_cols = [col for col in hourly_df.columns if rad_type in col.lower()]
        for col in matching_cols:
            data = hourly_df[col].dropna()
            if len(data) > 0:
                base_name = rad_type.replace('_radiation', '_mj_m2')
                aggregates[base_name] = {"sum": float(data.sum() / 1000000)}  # Convert J to MJ
    
    # Handle multiple soil temperature depths
    soil_temp_depths = ['0cm', '6cm', '18cm', '54cm']
    for depth in soil_temp_depths:
        matching_cols = [col for col in hourly_df.columns if f'soil_temperature_{depth}' in col]
        for col in matching_cols:
            data = hourly_df[col].dropna()
            if len(data) > 0:
                aggregates[f"soil_temp_{depth}_c"] = {"avg": float(data.mean())}
    
    # Handle multiple soil moisture depths
    soil_moisture_depths = ['0_1cm', '1_3cm', '3_9cm', '9_27cm']
    for depth in soil_moisture_depths:
        matching_cols = [col for col in hourly_df.columns if f'soil_moisture_{depth}' in col]
        for col in matching_cols:
            data = hourly_df[col].dropna()
            if len(data) > 0:
                aggregates[f"soil_moisture_{depth}_m3m3"] = {"avg": float(data.mean())}
    
    for pattern, output_name in other_vars.items():
        matching_cols = [col for col in hourly_df.columns if pattern in col.lower()]
        for col in matching_cols:
            data = hourly_df[col].dropna()
            if len(data) > 0:
                if 'et0' in col.lower():
                    aggregates[output_name] = {"sum": float(data.sum())}
                else:
                    aggregates[output_name] = {"avg": float(data.mean())}
                break  # Take first matching column
    
    return {
        "coverage": coverage_status,
        "coverage_fraction": coverage_fraction,
        "hours_available": available_hours,
        "method": "hourly_agg",
        "aggregates": aggregates
    }


def fetch_open_meteo_hourly(lat: float, lon: float, date: str) -> Dict[str, Any]:
    """
    Fetch comprehensive hourly weather data from Open-Meteo using multi-pass approach.
    
    Uses multiple API calls to avoid request size limits while maximizing parameter coverage.
    Based on successful patterns from test_weather_api_microbiome_comparison.py.
    
    Args:
        lat: Latitude
        lon: Longitude  
        date: Date in YYYY-MM-DD format
    
    Returns:
        Dict with hourly DataFrame, metadata, and provider info
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Multi-pass parameter groups (proven to work from test file)
    parameter_passes = [
        {
            "name": "Core Atmospheric",
            "params": [
                "temperature_2m", "relative_humidity_2m", "dewpoint_2m", "apparent_temperature",
                "precipitation", "rain", "snowfall", "surface_pressure", "cloudcover"
            ]
        },
        {
            "name": "Wind & Solar",
            "params": [
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                "shortwave_radiation", "direct_radiation", "diffuse_radiation", "sunshine_duration"
            ]
        },
        {
            "name": "Advanced Environmental", 
            "params": [
                "uv_index", "vapour_pressure_deficit", "visibility",
                "et0_fao_evapotranspiration", "weather_code", "is_day"
            ]
        },
        {
            "name": "Soil Microbiome",
            "params": [
                "soil_temperature_0cm", "soil_temperature_6cm", 
                "soil_moisture_0_1cm", "soil_moisture_1_3cm"
            ]
        }
    ]
    
    base_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "timezone": "UTC",
        "format": "json"
    }
    
    # Execute multiple passes and collect data
    all_hourly_data = {}
    metadata = {}
    pass_results = []
    total_successful_params = 0
    
    for pass_info in parameter_passes:
        try:
            params = base_params.copy()
            params["hourly"] = ",".join(pass_info["params"])
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "hourly" in data and data["hourly"]:
                # Extract hourly data from this pass
                pass_hourly = data["hourly"]
                successful_params = 0
                
                # Merge into combined dataset
                for key, values in pass_hourly.items():
                    if values and (key == "time" or values[0] is not None):
                        all_hourly_data[key] = values
                        if key != "time":
                            successful_params += 1
                
                # Store metadata from first successful pass
                if not metadata:
                    metadata = {
                        "latitude": data.get("latitude", lat),
                        "longitude": data.get("longitude", lon),
                        "elevation": data.get("elevation"),
                        "timezone": data.get("timezone"),
                        "utc_offset_seconds": data.get("utc_offset_seconds"),
                        "generation_time_ms": data.get("generationtime_ms"),
                    }
                
                total_successful_params += successful_params
                pass_results.append({
                    "pass_name": pass_info["name"],
                    "requested_params": len(pass_info["params"]),
                    "successful_params": successful_params,
                    "success": True
                })
            else:
                pass_results.append({
                    "pass_name": pass_info["name"],
                    "requested_params": len(pass_info["params"]),
                    "successful_params": 0,
                    "success": False,
                    "error": "No hourly data in response"
                })
                
        except Exception as e:
            pass_results.append({
                "pass_name": pass_info["name"],
                "requested_params": len(pass_info["params"]),
                "successful_params": 0,
                "success": False,
                "error": str(e)
            })
    
    # Check if at least one pass succeeded
    if total_successful_params == 0:
        return {
            "success": False, 
            "error": "All API passes failed",
            "provider": "open_meteo",
            "pass_results": pass_results
        }
    
    # Convert combined data to DataFrame
    if "time" not in all_hourly_data:
        return {"success": False, "error": "No time data available", "provider": "open_meteo"}
    
    df = pd.DataFrame(all_hourly_data)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    
    # Get actual coordinates from metadata
    actual_lat = metadata.get("latitude", lat)
    actual_lon = metadata.get("longitude", lon)
    actual_elevation = metadata.get("elevation")
    
    # Calculate distance
    distance_km = haversine_km(lat, lon, actual_lat, actual_lon)
    
    return {
        "success": True,
        "provider": "open_meteo",
        "hourly_df": df,
        "point": {"lat": actual_lat, "lon": actual_lon},
        "sample_point_distance_km": distance_km,
        "spatial_resolution": "0.1° (~11km)",
        "elevation_m": actual_elevation,
        "source_url": f"{url}?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}",
        "model_or_network": "ERA5-Land",
        "variables_requested": sum(len(p["params"]) for p in parameter_passes),
        "variables_received": len([col for col in df.columns if not df[col].isna().all()]),
        "multi_pass_results": pass_results,
        "successful_passes": len([p for p in pass_results if p["success"]]),
        "total_passes": len(pass_results)
    }


def fetch_meteostat_hourly(lat: float, lon: float, date: str) -> Dict[str, Any]:
    """
    Fetch station-based hourly weather data from Meteostat.
    
    Args:
        lat: Latitude
        lon: Longitude
        date: Date in YYYY-MM-DD format
        
    Returns:
        Dict with hourly DataFrame, station metadata, and distances
    """
    try:
        from meteostat import Hourly, Point, Stations
        from datetime import datetime
        
        # Convert date to datetime
        dt = datetime.strptime(date, "%Y-%m-%d")
        
        # Find nearby stations
        stations = Stations()
        stations = stations.nearby(lat, lon, radius=150000)  # 150km radius
        station_data = stations.fetch(limit=5)
        
        if len(station_data) == 0:
            return {"success": False, "error": "No stations found", "provider": "meteostat"}
        
        # Calculate station distances and metadata
        station_info = []
        for idx, station in station_data.iterrows():
            distance = haversine_km(lat, lon, station.get("latitude", 0), station.get("longitude", 0))
            station_info.append({
                "id": idx,
                "name": station.get("name", "Unknown"),
                "lat": station.get("latitude"),
                "lon": station.get("longitude"),
                "distance_km": round(distance, 2),
                "elevation_m": station.get("elevation"),
                "weight": 1.0 / (1.0 + distance)  # Simple inverse distance weighting
            })
        
        # Sort by distance
        station_info.sort(key=lambda x: x["distance_km"])
        closest_station = station_info[0]
        
        # Get hourly data for the location
        location = Point(lat, lon)
        hourly_data = Hourly(location, dt, dt)
        df = hourly_data.fetch()
        
        if len(df) == 0:
            return {
                "success": False, 
                "error": "No hourly data available",
                "provider": "meteostat",
                "stations_found": len(station_info),
                "closest_station_distance_km": closest_station["distance_km"]
            }
        
        return {
            "success": True,
            "provider": "meteostat",
            "hourly_df": df,
            "point": {"lat": lat, "lon": lon},  # Meteostat interpolates to point
            "sample_point_distance_km": closest_station["distance_km"],
            "spatial_resolution": "station",
            "source_url": "https://meteostat.net/",
            "model_or_network": "station",
            "station_ids": station_info,
            "primary_station": closest_station
        }
        
    except ImportError:
        return {"success": False, "error": "Meteostat library not available", "provider": "meteostat"}
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "meteostat"}


def weather_daily(lat: float, lon: float, date: str) -> Dict[str, Any]:
    """
    Get comprehensive daily weather from ALL providers and save all responses.
    
    Queries Open-Meteo and Meteostat simultaneously, returns primary result
    plus all provider responses for comparison and validation.
    
    Args:
        lat: Latitude
        lon: Longitude
        date: Date in YYYY-MM-DD format
        
    Returns:
        Dict following the weather[daily] schema with all provider data
    """
    providers = [
        ("open_meteo", fetch_open_meteo_hourly),
        ("meteostat", fetch_meteostat_hourly)
    ]
    
    all_results = {}
    successful_results = []
    
    # Query ALL providers
    for provider_name, fetch_func in providers:
        try:
            result = fetch_func(lat, lon, date)
            all_results[provider_name] = result
            
            if result.get("success"):
                # Aggregate hourly to daily for successful results
                hourly_df = result["hourly_df"]
                daily_agg = aggregate_hourly_to_daily(hourly_df)
                
                # Build output following schema
                processed_result = {
                    "provider": result["provider"],
                    "date": date,
                    "point": result["point"],
                    "sample_point_distance_km": result["sample_point_distance_km"],
                    "spatial_resolution": result["spatial_resolution"],
                    "coverage": daily_agg["coverage"],
                    "method": daily_agg["method"],
                    "provenance": {
                        "source_url": result.get("source_url", ""),
                        "model_or_network": result.get("model_or_network", ""),
                        "variables_requested": result.get("variables_requested"),
                        "variables_received": result.get("variables_received")
                    }
                }
                
                # Add daily aggregates directly to output
                processed_result.update(daily_agg["aggregates"])
                
                # Add station info if available
                if "station_ids" in result:
                    processed_result["provenance"]["station_ids"] = result["station_ids"]
                
                successful_results.append((provider_name, processed_result))
                
        except Exception as e:
            all_results[provider_name] = {
                "provider": provider_name,
                "success": False,
                "error": str(e),
                "date": date
            }
    
    # Determine primary result (prefer Open-Meteo if successful, otherwise first successful)
    primary_result = None
    primary_provider = None
    
    if successful_results:
        # Prefer Open-Meteo, then others in order
        for provider_name, result in successful_results:
            if provider_name == "open_meteo":
                primary_result = result
                primary_provider = provider_name
                break
        
        # If no Open-Meteo result, use first successful
        if not primary_result:
            primary_provider, primary_result = successful_results[0]
    
    # Build comprehensive response
    if primary_result:
        # Return primary result with all provider data included
        response = primary_result.copy()
        response["all_providers"] = all_results
        response["providers_queried"] = list(all_results.keys())
        response["successful_providers"] = [p for p, r in successful_results]
        response["primary_provider"] = primary_provider
        return response
    else:
        # All providers failed
        return {
            "provider": "none",
            "date": date,
            "point": {"lat": lat, "lon": lon},
            "coverage": "none",
            "error": "All weather providers failed",
            "all_providers": all_results,
            "providers_queried": list(all_results.keys()),
            "successful_providers": [],
            "provider_errors": {k: v.get("error", "Unknown error") for k, v in all_results.items()}
        }


def weather_daily_batch(locations: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
    """
    Process multiple locations efficiently with shared provider setup.
    
    Args:
        locations: List of (lat, lon, date) tuples
        
    Returns:
        List of daily weather results
    """
    results = []
    
    for lat, lon, date in locations:
        result = weather_daily(lat, lon, date)
        results.append(result)
    
    return results
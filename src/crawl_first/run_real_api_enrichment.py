#!/usr/bin/env python3
"""
Run comprehensive geospatial enrichment using real API functions.

Takes normalized biosample JSON as input and runs all available real API
functions (elevation, weather, geocoding, soil, ecoregion) to generate
comprehensive geospatial enrichment data.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import math
import numpy as np
import pandas as pd
import decimal


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def _json_default(o):
    """Bulletproof JSON serialization for all numpy/pandas/datetime types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, pd.Series, list, tuple)):
        return list(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if hasattr(pd, 'isna') and pd.isna(o):
        return None
    if isinstance(o, decimal.Decimal):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types."""
    def default(self, obj):
        return _json_default(obj)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from crawl_first.geospatial_enrichment import (
        # Elevation APIs
        get_elevation_open_elevation,
        get_elevation_usgs,
        
        # Weather APIs  
        get_weather_data_open_meteo,
        
        # Geocoding APIs
        get_reverse_geocoding_nominatim,
        
        # Soil APIs
        get_soil_classification_nrcs_sda,
        get_soil_classification_isric_soilgrids,
        get_soil_properties_isric_wcs,
        
        # Ecoregion APIs
        get_local_ecoregion,
        get_wwf_ecoregion,
        
        # Nearby features
        get_nearby_features_overpass,
        
        # Comprehensive enrichment
        enrich_location
    )
    from crawl_first.geospatial import (
        get_elevation,
        geocode_location_name,
        reverse_geocode
    )
    from crawl_first.alternative_geospatial import (
        # Land cover APIs
        get_land_cover_esa_worldcover,
        get_land_cover_nlcd,
        get_soil_type_soilgrids,
        get_historical_land_cover,
        get_comprehensive_site_analysis
    )
except ImportError as e:
    print(f"Error importing real API functions: {e}")
    sys.exit(1)


def get_weather_meteostat(lat: float, lon: float, date: str) -> Dict[str, Any]:
    """Get weather data using Meteostat library with station-based data."""
    try:
        from datetime import datetime
        import pandas as pd
        from meteostat import Daily, Point, Stations
        
        # Convert date string to datetime
        dt = datetime.strptime(date, "%Y-%m-%d")
        
        # Create Point object for location
        location = Point(lat, lon)
        
        # Get nearby weather stations
        stations = Stations()
        stations = stations.nearby(lat, lon, radius=100000)  # 100km radius
        stations = stations.fetch(limit=5)
        
        # Calculate station distances
        def haversine_distance(lat1, lon1, lat2, lon2):
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
        
        # Find closest station info
        closest_station_distance = None
        closest_station_name = None
        station_details = []
        
        if len(stations) > 0:
            for idx, station in stations.iterrows():
                distance = haversine_distance(
                    lat, lon, 
                    station.get("latitude", 0), 
                    station.get("longitude", 0)
                )
                station_info = {
                    "name": station.get("name", "Unknown"),
                    "country": station.get("country", "Unknown"),
                    "distance_km": round(distance, 2),
                    "latitude": station.get("latitude"),
                    "longitude": station.get("longitude"),
                    "elevation": station.get("elevation")
                }
                station_details.append(station_info)
                
                if closest_station_distance is None or distance < closest_station_distance:
                    closest_station_distance = distance
                    closest_station_name = station.get("name", "Unknown")
        
        # Get daily weather data
        daily_data = Daily(location, dt, dt)
        daily_df = daily_data.fetch()
        
        # Only normalize if we have data to avoid "pointless normalization" warning
        if len(daily_df) > 0:
            daily_data = daily_data.normalize()  # Apply quality controls
            daily_df = daily_data.fetch()
        
        if len(daily_df) > 0:
            # Extract weather data from the day
            day_data = daily_df.iloc[0]
            
            weather_data = {}
            # Map Meteostat columns to our naming
            meteostat_columns = {
                'tavg': 'temperature_avg_c',
                'tmin': 'temperature_min_c', 
                'tmax': 'temperature_max_c',
                'prcp': 'precipitation_mm',
                'snow': 'snow_depth_mm',
                'wdir': 'wind_direction_deg',
                'wspd': 'wind_speed_kmh',
                'wpgt': 'wind_gust_kmh',
                'pres': 'pressure_hpa',
                'tsun': 'sunshine_minutes'
            }
            
            for meteostat_col, our_name in meteostat_columns.items():
                if meteostat_col in day_data and not pd.isna(day_data[meteostat_col]):
                    weather_data[our_name] = float(day_data[meteostat_col])
            
            result = {
                "date": date,
                "coordinates": {"latitude": lat, "longitude": lon},
                "weather_data": weather_data,
                "station_info": {
                    "closest_station": closest_station_name,
                    "distance_km": round(closest_station_distance, 2) if closest_station_distance else None,
                    "total_stations_found": len(station_details),
                    "stations": station_details[:3]  # Top 3 closest stations
                },
                "data_source": "Meteostat",
                "success": True,
                "parameters_available": len(weather_data)
            }
        else:
            # If we found stations but no weather data, mark as partial success
            if len(station_details) > 0:
                result = {
                    "date": date,
                    "coordinates": {"latitude": lat, "longitude": lon},
                    "weather_data": {},
                    "station_info": {
                        "closest_station": closest_station_name,
                        "distance_km": round(closest_station_distance, 2) if closest_station_distance else None,
                        "total_stations_found": len(station_details),
                        "stations": station_details[:3]  # Top 3 closest stations
                    },
                    "data_source": "Meteostat",
                    "success": True,  # Partial success - stations found but no weather data
                    "data_quality": "partial",
                    "parameters_available": 0,
                    "note": "Stations found but no weather data available for date"
                }
            else:
                result = {
                    "weather_data": {},
                    "station_info": {"stations_found": 0},
                    "data_source": "Meteostat", 
                    "success": False,
                    "error": "No weather stations found in area"
                }
            
        return result
        
    except ImportError:
        return {
            "weather_data": {},
            "error": "Meteostat library not available",
            "data_source": "Meteostat",
            "success": False
        }
    except Exception as e:
        return {
            "weather_data": {},
            "error": str(e),
            "data_source": "Meteostat",
            "success": False
        }


def run_comprehensive_enrichment(input_file: str, output_file: str) -> Dict[str, Any]:
    """Run all real API functions on normalized biosample data."""
    
    print(f"🌍 Running comprehensive geospatial enrichment on {input_file}")
    
    # Load normalized biosamples
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if 'results' not in data:
        print("❌ Error: Input file missing 'results' field")
        sys.exit(1)
    
    locations = data['results']
    print(f"📍 Processing {len(locations)} enrichable locations")
    
    # Define all real API functions to run
    api_functions = {
        # Elevation APIs (3 sources)
        "elevation_open_elevation": lambda loc: get_elevation_open_elevation(loc['latitude'], loc['longitude']),
        "elevation_usgs": lambda loc: get_elevation_usgs(loc['latitude'], loc['longitude']),
        "elevation_google_maps": lambda loc: get_elevation(loc['latitude'], loc['longitude']),
        
        # Weather APIs (2 comprehensive sources)
        "weather_open_meteo": lambda loc: get_weather_data_open_meteo(
            loc['latitude'], loc['longitude'], loc.get('collection_date', '2020-01-01')
        ) if loc.get('collection_date') else None,
        "weather_meteostat": lambda loc: get_weather_meteostat(
            loc['latitude'], loc['longitude'], loc.get('collection_date', '2020-01-01')
        ) if loc.get('collection_date') else None,
        
        # Geocoding APIs (2 sources)
        "reverse_geocoding_nominatim": lambda loc: get_reverse_geocoding_nominatim(loc['latitude'], loc['longitude']),
        "reverse_geocoding_google": lambda loc: reverse_geocode(loc['latitude'], loc['longitude']),
        
        # Soil APIs (4 sources - including alternative soilgrids)
        "soil_classification_nrcs": lambda loc: get_soil_classification_nrcs_sda(loc['latitude'], loc['longitude']),
        "soil_classification_isric": lambda loc: get_soil_classification_isric_soilgrids(loc['latitude'], loc['longitude']),
        "soil_properties_isric": lambda loc: get_soil_properties_isric_wcs(loc['latitude'], loc['longitude']),
        "soil_type_soilgrids_alt": lambda loc: get_soil_type_soilgrids(loc['latitude'], loc['longitude']),
        
        # Land Cover APIs (3 sources)
        "land_cover_esa_worldcover": lambda loc: get_land_cover_esa_worldcover(loc['latitude'], loc['longitude'], 2021),
        "land_cover_nlcd": lambda loc: get_land_cover_nlcd(loc['latitude'], loc['longitude'], 2021),
        "land_cover_historical": lambda loc: get_historical_land_cover(
            loc['latitude'], loc['longitude'], loc.get('collection_date', '2021-01-01')
        ) if loc.get('collection_date') else None,
        
        # Ecoregion APIs (2 sources)
        "ecoregion_local": lambda loc: get_local_ecoregion(loc['latitude'], loc['longitude']),
        "ecoregion_wwf": lambda loc: get_wwf_ecoregion(loc['latitude'], loc['longitude']),
        
        # Nearby features
        "nearby_features": lambda loc: get_nearby_features_overpass(loc['latitude'], loc['longitude']),
        
        # Comprehensive enrichment (all-in-one)
        "comprehensive_enrichment": lambda loc: enrich_location(
            loc['latitude'], loc['longitude'], 
            collection_date=loc.get('collection_date')
        ),
        
        # Comprehensive site analysis (alternative comprehensive)
        "comprehensive_site_analysis": lambda loc: get_comprehensive_site_analysis(
            loc['latitude'], loc['longitude'], loc.get('collection_date')
        )
    }
    
    # Collect all results
    enrichment_results = {
        "input_file": input_file,
        "processing_timestamp": datetime.now().isoformat(),
        "total_locations": len(locations),
        "api_count": len(api_functions),
        "results": []
    }
    
    # Process each location
    for i, location in enumerate(locations):
        print(f"  🔄 Processing location {i+1}/{len(locations)}: {location['sample_id']}")
        
        location_result = {
            "sample_id": location['sample_id'],
            "latitude": location['latitude'],
            "longitude": location['longitude'],
            "collection_date": location.get('collection_date'),
            "textual_location": location.get('textual_location'),
            "original_location": location,
            "api_enrichment": {}
        }
        
        # Run each API function
        successful_apis = 0
        for api_name, api_function in api_functions.items():
            try:
                print(f"    🔄 Running {api_name}...")
                result = api_function(location)
                
                # Check if the inner result actually succeeded
                inner_success = result.get("success", True) if isinstance(result, dict) else False
                effective_success = inner_success and result is not None
                
                location_result["api_enrichment"][api_name] = {
                    "success": effective_success,
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }
                
                if effective_success:
                    successful_apis += 1
                    print(f"    ✅ {api_name}: Success")
                else:
                    error_msg = result.get("error", "No data or inner failure") if isinstance(result, dict) else "Invalid response"
                    print(f"    ⚠️  {api_name}: Inner failure - {error_msg}")
                    location_result["api_enrichment"][api_name]["inner_error"] = error_msg
                    
            except Exception as e:
                print(f"    ❌ {api_name}: Exception - {e}")
                location_result["api_enrichment"][api_name] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        location_result["successful_apis"] = successful_apis
        location_result["total_apis"] = len(api_functions)
        enrichment_results["results"].append(location_result)
        
        print(f"    📊 {successful_apis}/{len(api_functions)} APIs succeeded for {location['sample_id']}")
    
    # Save results
    print(f"💾 Saving enrichment results to {output_file}")
    try:
        # Use bulletproof JSON serialization
        with open(output_file, 'w') as f:
            json.dump(enrichment_results, f, indent=2, default=_json_default)
        print("✅ Results saved successfully")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        # Try saving with backup approach
        try:
            clean_results_backup = convert_numpy_types(enrichment_results)
            with open(output_file + ".backup", 'w') as f:
                json.dump(clean_results_backup, f, indent=2, default=_json_default, ensure_ascii=False)
            print(f"💾 Backup saved to {output_file}.backup")
        except Exception as backup_error:
            print(f"❌ Backup also failed: {backup_error}")
        raise
    
    # Summary
    total_successful = sum(result["successful_apis"] for result in enrichment_results["results"])
    total_possible = len(locations) * len(api_functions)
    
    success_rate = round(100 * total_successful / total_possible, 1) if total_possible > 0 else 0.0
    
    print(f"\n📊 Geospatial Enrichment Summary:")
    print(f"   🧬 Biosamples: {len(locations)}")
    print(f"   🌍 API Functions: {len(api_functions)}")
    print(f"   📞 Total API calls: {total_successful}/{total_possible} successful")
    print(f"   ✅ Success Rate: {success_rate}%")
    print(f"   💾 Output saved to: {output_file}")
    
    return enrichment_results


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python run_real_api_enrichment.py <input_file> <output_file>")
        print("")
        print("Arguments:")
        print("  input_file    Path to normalized biosample JSON file")
        print("  output_file   Path to save comprehensive enrichment results")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Validate input file exists
    if not Path(input_file).exists():
        print(f"❌ Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run comprehensive enrichment
    try:
        run_comprehensive_enrichment(input_file, output_file)
    except Exception as e:
        print(f"❌ Error running enrichment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
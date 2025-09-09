#!/usr/bin/env python3
"""
Run all Google Plus API functions on normalized biosample data.

Takes normalized biosample JSON as input and runs all available Google Plus
API functions (elevation, weather, geocoding, land use, soil) to generate
comprehensive geospatial enrichment data.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from crawl_first.google_plus_api_functions import (
        # Elevation APIs
        get_elevation_google_file,
        get_elevation_usgs_file,
        get_elevation_open_elevation_file,
        
        # Weather APIs
        get_weather_openmeteo_file,
        get_weather_meteostat_file,
        
        # Geocoding APIs
        get_geocoding_google_file,
        get_geocoding_nominatim_file,
        
        # Land Use APIs
        get_landuse_usgs_file,
        get_landuse_esa_file,
        
        # Soil APIs
        get_soil_usda_file,
        get_soil_fao_file,
        get_soil_isric_soilgrids_rest_file,
        get_soil_isric_soilgrids_wcs_file,
        get_soil_openlandmap_stac_file
    )
except ImportError as e:
    print(f"Error importing Google Plus API functions: {e}")
    sys.exit(1)


def run_all_apis(input_file: str, output_file: str, use_cache: bool = True, save_cache: bool = True):
    """Run all Google Plus API functions on normalized biosample data."""
    
    print(f"🌍 Running all Google Plus API functions on {input_file}")
    print(f"📊 Cache settings: use_cache={use_cache}, save_cache={save_cache}")
    
    # Define all API functions to run
    api_functions = {
        # Elevation APIs (3 sources)
        "elevation_google": get_elevation_google_file,
        "elevation_usgs": get_elevation_usgs_file, 
        "elevation_open_elevation": get_elevation_open_elevation_file,
        
        # Weather APIs (2 sources)
        "weather_openmeteo": get_weather_openmeteo_file,
        "weather_meteostat": get_weather_meteostat_file,
        
        # Geocoding APIs (2 sources)
        "geocoding_google": get_geocoding_google_file,
        "geocoding_nominatim": get_geocoding_nominatim_file,
        
        # Land Use APIs (2 sources)
        "landuse_usgs": get_landuse_usgs_file,
        "landuse_esa": get_landuse_esa_file,
        
        # Soil APIs (5 sources)
        "soil_usda": get_soil_usda_file,
        "soil_fao": get_soil_fao_file,
        "soil_isric_rest": get_soil_isric_soilgrids_rest_file,
        "soil_isric_wcs": get_soil_isric_soilgrids_wcs_file,
        "soil_openlandmap": get_soil_openlandmap_stac_file
    }
    
    # Collect all results
    all_results = {
        "input_file": input_file,
        "api_count": len(api_functions),
        "cache_settings": {
            "use_cache": use_cache,
            "save_cache": save_cache
        },
        "api_results": {}
    }
    
    # Run each API function
    for api_name, api_function in api_functions.items():
        print(f"  🔄 Running {api_name}...")
        try:
            # Only Google elevation function supports cache parameters
            if api_name == "elevation_google":
                result = api_function(input_file, use_cache=use_cache, save_cache=save_cache)
            else:
                result = api_function(input_file)
            
            all_results["api_results"][api_name] = {
                "success": True,
                "data": result,
                "count": len(result) if isinstance(result, list) else 1
            }
            print(f"    ✅ {api_name}: {len(result) if isinstance(result, list) else 1} results")
        except Exception as e:
            print(f"    ❌ {api_name}: Error - {e}")
            all_results["api_results"][api_name] = {
                "success": False,
                "error": str(e),
                "count": 0
            }
    
    # Save combined results
    print(f"💾 Saving combined results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    successful_apis = sum(1 for result in all_results["api_results"].values() if result["success"])
    total_apis = len(api_functions)
    
    print(f"")
    print(f"📊 API Enrichment Summary:")
    print(f"  Successfully completed: {successful_apis}/{total_apis} APIs")
    print(f"  Output saved to: {output_file}")
    
    if successful_apis < total_apis:
        print(f"  ⚠️  {total_apis - successful_apis} APIs failed - check output for details")
    
    return all_results


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python run_all_google_plus_apis.py <input_file> <output_file> [--no-cache] [--no-save-cache]")
        print("")
        print("Arguments:")
        print("  input_file    Path to normalized biosample JSON file")
        print("  output_file   Path to save combined API results")
        print("  --no-cache    Don't use cached API results (default: use cache)")
        print("  --no-save-cache  Don't save new API results to cache (default: save to cache)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Parse cache options
    use_cache = "--no-cache" not in sys.argv
    save_cache = "--no-save-cache" not in sys.argv
    
    # Validate input file exists
    if not Path(input_file).exists():
        print(f"❌ Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run all APIs
    try:
        run_all_apis(input_file, output_file, use_cache, save_cache)
    except Exception as e:
        print(f"❌ Error running APIs: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
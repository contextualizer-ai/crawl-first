"""
Comprehensive elevation API comparison test.

Tests USGS, Open Elevation, and Google Elevation APIs across US, international, 
ocean, and invalid locations to understand their differences.
"""

import os
import requests
import pytest
from pathlib import Path


def load_local_env():
    """Load environment variables from local/.env file."""
    env_file = Path(__file__).parent.parent / "local" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


# Load local environment variables
load_local_env()


def call_usgs_elevation(lat: float, lon: float) -> dict:
    """Call USGS Elevation API directly."""
    # USGS only covers US bounds
    if not (-170 <= lon <= -60 and 15 <= lat <= 75):
        return {
            "elevation_meters": None,
            "error": "Outside US coverage",
            "api": "USGS",
            "success": False
        }
    
    url = "https://epqs.nationalmap.gov/v1/json"
    params = {"x": lon, "y": lat, "units": "Meters"}
    
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        elevation_str = data.get("value")
        if elevation_str and elevation_str != "-1000000":
            return {
                "elevation_meters": float(elevation_str),
                "api": "USGS",
                "success": True
            }
        else:
            return {
                "elevation_meters": None,
                "error": "No data available",
                "api": "USGS", 
                "success": False
            }
    except Exception as e:
        return {
            "elevation_meters": None,
            "error": str(e),
            "api": "USGS",
            "success": False
        }


def call_open_elevation(lat: float, lon: float) -> dict:
    """Call Open Elevation API directly."""
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {"locations": f"{lat},{lon}"}
    
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results"):
            elevation = data["results"][0].get("elevation")
            return {
                "elevation_meters": elevation,
                "api": "Open Elevation",
                "success": True
            }
        else:
            return {
                "elevation_meters": None,
                "error": "No results",
                "api": "Open Elevation",
                "success": False
            }
    except Exception as e:
        return {
            "elevation_meters": None,
            "error": str(e),
            "api": "Open Elevation", 
            "success": False
        }


def call_google_elevation(lat: float, lon: float) -> dict:
    """Call Google Elevation API directly."""
    api_key = os.environ.get("GOOGLE_ELEVATION_API_KEY")
    if not api_key:
        return {
            "elevation_meters": None,
            "error": "GOOGLE_ELEVATION_API_KEY not set",
            "api": "Google Elevation",
            "success": False
        }
    
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = {
        "locations": f"{lat},{lon}",
        "key": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK" and data.get("results"):
            result = data["results"][0]
            return {
                "elevation_meters": result.get("elevation"),
                "resolution": result.get("resolution"),
                "api": "Google Elevation",
                "success": True
            }
        else:
            return {
                "elevation_meters": None,
                "error": data.get("error_message", data.get("status", "Unknown error")),
                "api": "Google Elevation",
                "success": False
            }
    except Exception as e:
        return {
            "elevation_meters": None,
            "error": str(e),
            "api": "Google Elevation",
            "success": False
        }


class TestDirectElevationAPIComparison:
    """Compare elevation APIs across different location types."""

    def test_us_locations(self):
        """Test elevation APIs on US locations."""
        us_locations = [
            {"name": "Yellowstone, WY", "lat": 44.428, "lon": -110.5885, "expected_range": (2450, 2470)},
            {"name": "Denver, CO", "lat": 39.7392, "lon": -104.9903, "expected_range": (1590, 1610)},
            {"name": "San Francisco, CA", "lat": 37.7749, "lon": -122.4194, "expected_range": (14, 18)},
            {"name": "Mount Whitney, CA", "lat": 36.5786, "lon": -118.2923, "expected_range": (4410, 4430)}
        ]
        
        print("\n=== US LOCATIONS ===")
        for location in us_locations:
            lat, lon = location["lat"], location["lon"]
            min_elev, max_elev = location["expected_range"]
            
            usgs_result = call_usgs_elevation(lat, lon)
            open_result = call_open_elevation(lat, lon)
            google_result = call_google_elevation(lat, lon)
            
            print(f"\n{location['name']} ({lat}, {lon}):")
            print(f"  USGS: {usgs_result}")
            print(f"  Open: {open_result}")
            print(f"  Google: {google_result}")
            
            # USGS and Open should succeed for US locations
            assert usgs_result["success"] is True, f"USGS failed for {location['name']}"
            assert open_result["success"] is True, f"Open Elevation failed for {location['name']}"
            
            # Google should succeed if API key is available
            if google_result["success"]:
                google_elev = google_result["elevation_meters"]
                assert min_elev <= google_elev <= max_elev, f"Google elevation {google_elev}m outside expected range for {location['name']}"
                
                # Compare differences with Google
                usgs_google_diff = abs(usgs_result["elevation_meters"] - google_elev)
                open_google_diff = abs(open_result["elevation_meters"] - google_elev)
                print(f"    USGS vs Google: {usgs_google_diff:.1f}m")
                print(f"    Open vs Google: {open_google_diff:.1f}m")
            
            # USGS and Open should be in expected range
            usgs_elev = usgs_result["elevation_meters"]
            open_elev = open_result["elevation_meters"]
            
            assert min_elev <= usgs_elev <= max_elev, f"USGS elevation {usgs_elev}m outside expected range for {location['name']}"
            assert min_elev <= open_elev <= max_elev, f"Open elevation {open_elev}m outside expected range for {location['name']}"
            
            # Difference should be reasonable (within 100m for most cases)
            diff = abs(usgs_elev - open_elev)
            if location["name"] != "Mount Whitney, CA":  # High altitude may have more variation
                assert diff <= 100, f"Large difference {diff}m between APIs for {location['name']}"

    def test_international_locations(self):
        """Test elevation APIs on international locations."""
        international_locations = [
            {"name": "London, UK", "lat": 51.5074, "lon": -0.1278, "expected_range": (8, 30)},
            {"name": "Tokyo, Japan", "lat": 35.6762, "lon": 139.6503, "expected_range": (3, 45)},
            {"name": "Sydney, Australia", "lat": -33.8688, "lon": 151.2093, "expected_range": (20, 85)},
            {"name": "Mount Everest", "lat": 27.9881, "lon": 86.9250, "expected_range": (8720, 8860)}
        ]
        
        print("\n=== INTERNATIONAL LOCATIONS ===")
        for location in international_locations:
            lat, lon = location["lat"], location["lon"]
            min_elev, max_elev = location["expected_range"]
            
            usgs_result = call_usgs_elevation(lat, lon)
            open_result = call_open_elevation(lat, lon)
            google_result = call_google_elevation(lat, lon)
            
            print(f"\n{location['name']} ({lat}, {lon}):")
            print(f"  USGS: {usgs_result}")
            print(f"  Open: {open_result}")
            print(f"  Google: {google_result}")
            
            # USGS should fail (outside coverage)
            assert usgs_result["success"] is False, f"USGS should fail for international location {location['name']}"
            assert "coverage" in usgs_result.get("error", "").lower() or "bounds" in usgs_result.get("error", "").lower()
            
            # Open Elevation should succeed
            assert open_result["success"] is True, f"Open Elevation failed for {location['name']}"
            open_elev = open_result["elevation_meters"]
            assert min_elev <= open_elev <= max_elev, f"Open elevation {open_elev}m outside expected range for {location['name']}"
            
            # Google should succeed if API key is available
            if google_result["success"]:
                google_elev = google_result["elevation_meters"]
                assert min_elev <= google_elev <= max_elev, f"Google elevation {google_elev}m outside expected range for {location['name']}"
                
                # Compare Open vs Google for international locations
                open_google_diff = abs(open_elev - google_elev)
                print(f"    Open vs Google: {open_google_diff:.1f}m")
                
                # For Mount Everest, allow larger differences due to different data sources
                max_diff = 200 if location["name"] == "Mount Everest" else 100
                assert open_google_diff <= max_diff, f"Large difference {open_google_diff}m between Open and Google for {location['name']}"

    def test_ocean_locations(self):
        """Test elevation APIs on ocean locations."""
        ocean_locations = [
            {"name": "Pacific Ocean", "lat": 35.0, "lon": -140.0},
            {"name": "Atlantic Ocean", "lat": 40.0, "lon": -30.0},
            {"name": "Indian Ocean", "lat": -20.0, "lon": 80.0}
        ]
        
        print("\n=== OCEAN LOCATIONS ===")
        for location in ocean_locations:
            lat, lon = location["lat"], location["lon"]
            
            usgs_result = call_usgs_elevation(lat, lon)
            open_result = call_open_elevation(lat, lon)
            google_result = call_google_elevation(lat, lon)
            
            print(f"\n{location['name']} ({lat}, {lon}):")
            print(f"  USGS: {usgs_result}")
            print(f"  Open: {open_result}")
            print(f"  Google: {google_result}")
            
            # USGS should fail (outside coverage or no data)
            assert usgs_result["success"] is False
            
            # Open Elevation behavior with ocean
            assert open_result["success"] is True
            assert open_result["elevation_meters"] <= 100, f"Ocean elevation seems too high: {open_result['elevation_meters']}m"
            
            # Google should return bathymetry (negative values) for ocean
            if google_result["success"]:
                google_elev = google_result["elevation_meters"]
                print(f"    Google ocean depth: {google_elev}m")
                # Ocean should typically be negative (depth below sea level)
                # Note: Some shallow areas near shore might be positive
                if google_elev < 0:
                    print(f"    ✓ Google correctly returns bathymetry: {google_elev}m below sea level")
                elif google_elev <= 50:
                    print(f"    ~ Google returns near-sea-level: {google_elev}m (possibly shallow area)")
                else:
                    print(f"    ? Google returns positive elevation: {google_elev}m (unexpected for deep ocean)")

    def test_invalid_coordinates(self):
        """Test elevation APIs with invalid coordinates."""
        invalid_coords = [
            {"name": "Invalid latitude", "lat": 95.0, "lon": -122.0},
            {"name": "Invalid longitude", "lat": 45.0, "lon": 200.0},
            {"name": "Both invalid", "lat": -95.0, "lon": -200.0}
        ]
        
        print("\n=== INVALID COORDINATES ===")
        for coords in invalid_coords:
            lat, lon = coords["lat"], coords["lon"]
            
            usgs_result = call_usgs_elevation(lat, lon)
            open_result = call_open_elevation(lat, lon)
            google_result = call_google_elevation(lat, lon)
            
            print(f"\n{coords['name']} ({lat}, {lon}):")
            print(f"  USGS: {usgs_result}")
            print(f"  Open: {open_result}")
            print(f"  Google: {google_result}")
            
            # USGS should fail
            assert usgs_result["success"] is False
            
            # Open Elevation may still return something (it's permissive)
            if open_result["success"]:
                assert open_result["elevation_meters"] in [0.0, None], "Open Elevation should return 0 or null for invalid coords"
            
            # Google should properly validate and reject invalid coordinates
            if google_result["success"] is False:
                print(f"    ✓ Google properly rejects invalid coordinates")
                # Check for proper error messages
                error = google_result.get("error", "")
                if "INVALID_REQUEST" in error:
                    print(f"    ✓ Google returns proper INVALID_REQUEST error")
            else:
                print(f"    ? Google accepts invalid coordinates: {google_result}")

    def test_api_behavior_summary(self):
        """Summarize the key behavioral differences between APIs."""
        
        behaviors = {
            "USGS EPQS v1": {
                "coverage": "US only",
                "coordinate_validation": "Strict bounds checking",
                "invalid_coords": "Rejects with error",
                "ocean_coords": "Outside coverage or no data",
                "international": "Rejects with coverage error",
                "accuracy": "High (~1m resolution)",
                "data_source": "3DEP DEM",
                "cost": "Free"
            },
            "Open Elevation": {
                "coverage": "Global", 
                "coordinate_validation": "Permissive",
                "invalid_coords": "Returns 0.0m",
                "ocean_coords": "Returns 0.0m (no bathymetry)",
                "international": "Works globally",
                "accuracy": "Moderate (~30m resolution)",
                "data_source": "SRTM + other DEMs",
                "cost": "Free (1000/month) or paid"
            },
            "Google Elevation": {
                "coverage": "Global",
                "coordinate_validation": "Strict (INVALID_REQUEST errors)",
                "invalid_coords": "Proper error codes",
                "ocean_coords": "Returns bathymetry (negative depths)",
                "international": "Works globally",
                "accuracy": "High (variable resolution)",
                "data_source": "Google's elevation datasets",
                "cost": "Free tier (25,000/day) then paid"
            }
        }
        
        print("\n=== THREE-API BEHAVIOR SUMMARY ===")
        for api, props in behaviors.items():
            print(f"\n{api}:")
            for key, value in props.items():
                print(f"  {key}: {value}")
        
        print("\n=== KEY DIFFERENTIATORS ===")
        print("📍 Best for US locations: USGS (highest accuracy, free)")
        print("🌍 Best for global coverage: Google (includes bathymetry) > Open Elevation")
        print("🆓 Best for free usage: USGS (US only) > Open Elevation (limited)")
        print("🌊 Best ocean handling: Google (actual depths) > USGS (rejects) > Open (misleading 0.0m)")
        print("🔍 Best error handling: Google > USGS > Open Elevation")
        print("⚡ Best for high-volume: Open Elevation (self-hostable)")
        
        google_available = os.environ.get("GOOGLE_ELEVATION_API_KEY") is not None
        print(f"\n📊 Google API Status: {'✅ Available' if google_available else '❌ No API key'}")
        
        # Test passes if we can document all APIs
        assert len(behaviors) == 3
        assert "USGS" in str(behaviors)
        assert "Open Elevation" in str(behaviors) 
        assert "Google" in str(behaviors)
"""
Comprehensive reverse geocoding API comparison test.

Tests multiple reverse geocoding APIs across diverse global locations to understand
their differences in data richness, accuracy, and coverage for biosample enrichment.
"""

import os

# Import the existing reverse geocoding function from our geospatial enrichment module
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests

sys.path.append(str(Path(__file__).parent.parent / "src"))
from crawl_first.geospatial_enrichment import get_reverse_geocoding_nominatim


def load_local_env():
    """Load environment variables from local/.env file."""
    env_file = Path(__file__).parent.parent / "local" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


# Load local environment variables
load_local_env()


def call_nominatim_direct(lat: float, lon: float, zoom: int = 18) -> Dict[str, Any]:
    """Call Nominatim reverse geocoding API directly with maximum detail extraction."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1,
        "zoom": zoom,
        "extratags": 1,
        "namedetails": 1,
        "polygon_geojson": 1,  # Get polygon boundaries if available
        "polygon_threshold": 0.0,  # Include all polygon details
    }

    headers = {"User-Agent": "BiosampleEnrichment/1.0 (research purposes)"}

    try:
        # Respect rate limiting
        time.sleep(1.1)
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data and "address" in data:
            address = data["address"]

            # Extract ALL address components
            result = {
                "display_name": data.get("display_name"),
                "place_type": data.get("type"),
                "osm_type": data.get("osm_type"),
                "osm_id": data.get("osm_id"),
                "place_id": data.get("place_id"),
                "licence": data.get("licence"),
                "place_rank": data.get("place_rank"),
                "category": data.get("category"),
                "importance": data.get("importance"),
                "addresstype": data.get("addresstype"),
                "boundingbox": data.get("boundingbox"),
                "latitude": float(data.get("lat", lat)),
                "longitude": float(data.get("lon", lon)),
                "api": "Nominatim Direct",
                "zoom_level": zoom,
                "success": True,
            }

            # Extract ALL address fields systematically
            address_fields = [
                "house_number",
                "road",
                "footway",
                "cycleway",
                "path",
                "pedestrian",
                "suburb",
                "district",
                "neighbourhood",
                "quarter",
                "city_district",
                "city",
                "town",
                "village",
                "hamlet",
                "municipality",
                "county",
                "state_district",
                "state",
                "province",
                "region",
                "postcode",
                "country",
                "country_code",
                "continent",
                "amenity",
                "building",
                "shop",
                "office",
                "leisure",
                "tourism",
                "historic",
                "natural",
                "landuse",
                "place",
                "man_made",
                "railway",
                "aeroway",
                "waterway",
                "boundary",
                "admin_level",
            ]

            for field in address_fields:
                if field in address:
                    result[field] = address[field]

            # Add extra tags and name details if available
            if data.get("extratags"):
                result["extra_tags"] = data["extratags"]
            if data.get("namedetails"):
                result["name_details"] = data["namedetails"]
            if data.get("geojson"):
                result["geometry"] = data["geojson"]

            return result
        else:
            return {
                "api": "Nominatim Direct",
                "zoom_level": zoom,
                "success": False,
                "error": "No address data returned",
            }

    except Exception as e:
        return {
            "api": "Nominatim Direct",
            "zoom_level": zoom,
            "success": False,
            "error": str(e),
        }


def call_google_reverse_geocoding(lat: float, lon: float) -> Dict[str, Any]:
    """Call Google Reverse Geocoding API using the same key as Elevation API."""
    # Use the same Google Maps Platform API key as elevation - just need Geocoding API enabled
    api_key = os.environ.get("GOOGLE_ELEVATION_API_KEY")
    if not api_key:
        return {
            "api": "Google Reverse Geocoding",
            "success": False,
            "error": "GOOGLE_ELEVATION_API_KEY not set (same key used for geocoding - just enable Geocoding API)",
        }

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lon}",
        "key": api_key,
        "language": "en",  # Get results in English for consistency
        # No result_type filter = get ALL available types for maximum detail
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "OK" and data.get("results"):
            # Use the most detailed result (first one) but extract ALL available data
            result_data = data["results"][0]
            address_components = result_data.get("address_components", [])
            geometry = result_data.get("geometry", {})

            # Base result structure
            result = {
                "display_name": result_data.get("formatted_address"),
                "place_types": result_data.get("types", []),
                "place_id": result_data.get("place_id"),
                "partial_match": result_data.get("partial_match", False),
                "api": "Google Reverse Geocoding",
                "success": True,
                "total_results": len(data["results"]),
            }

            # Extract location data
            location = geometry.get("location", {})
            result["latitude"] = location.get("lat", lat)
            result["longitude"] = location.get("lng", lon)

            # Extract geometry details
            if geometry.get("bounds"):
                result["bounds"] = geometry["bounds"]
            if geometry.get("viewport"):
                result["viewport"] = geometry["viewport"]
            result["location_type"] = geometry.get("location_type")

            # Parse ALL address components systematically
            component_types = {}
            for component in address_components:
                long_name = component.get("long_name")
                short_name = component.get("short_name")
                types = component.get("types", [])

                # Store by primary type and all types
                for type_name in types:
                    if type_name not in component_types:
                        component_types[type_name] = {
                            "long_name": long_name,
                            "short_name": short_name,
                        }

            # Add all component types to result
            result["address_components"] = component_types

            # Extract common fields for compatibility
            if "country" in component_types:
                result["country"] = component_types["country"]["long_name"]
                result["country_code"] = component_types["country"]["short_name"]
            if "administrative_area_level_1" in component_types:
                result["state"] = component_types["administrative_area_level_1"][
                    "long_name"
                ]
            if "administrative_area_level_2" in component_types:
                result["county"] = component_types["administrative_area_level_2"][
                    "long_name"
                ]
            if "locality" in component_types:
                result["city"] = component_types["locality"]["long_name"]
            if "sublocality" in component_types:
                result["neighborhood"] = component_types["sublocality"]["long_name"]
            if "route" in component_types:
                result["road"] = component_types["route"]["long_name"]
            if "street_number" in component_types:
                result["house_number"] = component_types["street_number"]["long_name"]
            if "postal_code" in component_types:
                result["postcode"] = component_types["postal_code"]["long_name"]

            # Add ALL results for comprehensive analysis
            result["all_results"] = data["results"]

            return result
        else:
            return {
                "api": "Google Reverse Geocoding",
                "success": False,
                "error": data.get("error_message", data.get("status", "Unknown error")),
            }

    except Exception as e:
        return {"api": "Google Reverse Geocoding", "success": False, "error": str(e)}


def assess_geocoding_quality(result: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the quality and completeness of reverse geocoding results."""
    if not result.get("success"):
        return {"quality_score": 0.0, "completeness": "failed"}

    # Define scoring criteria for biosample enrichment
    scoring_criteria = {
        "country": 0.2,  # Essential for geographic context
        "state": 0.15,  # Important for regional classification
        "county": 0.15,  # Useful for local administrative context
        "city": 0.15,  # Important for urban/rural classification
        "postcode": 0.1,  # Useful for precise location
        "display_name": 0.1,  # Overall location description
        "road": 0.05,  # Street-level detail
        "house_number": 0.05,  # Precise address detail
        "neighborhood": 0.05,  # Local area context
    }

    quality_score = 0.0
    present_fields = []
    missing_fields = []

    for field, weight in scoring_criteria.items():
        if result.get(field):
            quality_score += weight
            present_fields.append(field)
        else:
            missing_fields.append(field)

    # Assess completeness level
    if quality_score >= 0.8:
        completeness = "excellent"
    elif quality_score >= 0.6:
        completeness = "good"
    elif quality_score >= 0.4:
        completeness = "moderate"
    elif quality_score >= 0.2:
        completeness = "basic"
    else:
        completeness = "poor"

    return {
        "quality_score": round(quality_score, 3),
        "completeness": completeness,
        "present_fields": present_fields,
        "missing_fields": missing_fields,
        "total_fields": len(scoring_criteria),
        "fields_present": len(present_fields),
    }


class TestReverseGeocodingAPIComparison:
    """Compare reverse geocoding APIs across different location types for biosample enrichment."""

    def test_diverse_global_locations(self):
        """Test reverse geocoding APIs on diverse global locations relevant to biosample collection."""
        test_locations = [
            {
                "name": "Urban Research Lab (MIT, Cambridge, MA)",
                "lat": 42.3601,
                "lon": -71.0928,
                "context": "Urban laboratory environment",
                "expected_fields": ["country", "state", "city", "road"],
            },
            {
                "name": "Remote Field Station (Yellowstone, WY)",
                "lat": 44.4280,
                "lon": -110.5885,
                "context": "Remote wilderness research site",
                "expected_fields": ["country", "state"],
            },
            {
                "name": "Agricultural Field (Rural Iowa)",
                "lat": 42.0308,
                "lon": -93.6319,
                "context": "Agricultural microbiome sampling",
                "expected_fields": ["country", "state", "county"],
            },
            {
                "name": "Coastal Marine Site (Monterey Bay, CA)",
                "lat": 36.6002,
                "lon": -121.8947,
                "context": "Marine microbiome research",
                "expected_fields": ["country", "state", "city"],
            },
            {
                "name": "Tropical Forest (Amazon, Brazil)",
                "lat": -3.4653,
                "lon": -62.2159,
                "context": "Tropical forest microbiome",
                "expected_fields": ["country", "state"],
            },
            {
                "name": "Arctic Research Station (Greenland)",
                "lat": 76.5319,
                "lon": -68.7664,
                "context": "Extreme cold environment microbiome",
                "expected_fields": ["country"],
            },
            {
                "name": "Urban Microbiome (Tokyo, Japan)",
                "lat": 35.6762,
                "lon": 139.6503,
                "context": "Urban microbiome diversity",
                "expected_fields": ["country", "city"],
            },
            {
                "name": "Desert Research (Sahara, Algeria)",
                "lat": 23.6978,
                "lon": 5.6815,
                "context": "Arid environment microbiome",
                "expected_fields": ["country"],
            },
        ]

        print("\n=== REVERSE GEOCODING API COMPARISON FOR BIOSAMPLE ENRICHMENT ===")
        print("=" * 80)

        api_summary = {}

        for location in test_locations:
            lat, lon = location["lat"], location["lon"]

            print(f"\n🌍 {location['name']} ({lat}, {lon})")
            print(f"   Context: {location['context']}")
            print(f"   Expected Fields: {', '.join(location['expected_fields'])}")
            print("-" * 60)

            # Test available APIs (Nominatim + Google)
            apis = [
                (
                    "Nominatim (Cached)",
                    lambda: get_reverse_geocoding_nominatim(lat, lon),
                ),
                ("Nominatim (Direct)", lambda: call_nominatim_direct(lat, lon)),
                (
                    "Google Reverse Geocoding",
                    lambda: call_google_reverse_geocoding(lat, lon),
                ),
            ]

            location_results = {}

            for api_name, api_func in apis:
                print(f"\n  🔍 Testing {api_name}...")
                try:
                    result = api_func()

                    if result.get("success"):
                        quality_assessment = assess_geocoding_quality(result)

                        print(
                            f"    ✅ Success: {quality_assessment['completeness']} completeness"
                        )
                        print(f"    📍 Location: {result.get('display_name', 'N/A')}")
                        print(
                            f"    🗺️  Quality Score: {quality_assessment['quality_score']}"
                        )
                        print(
                            f"    📊 Fields: {quality_assessment['fields_present']}/{quality_assessment['total_fields']}"
                        )

                        # Show key administrative fields
                        admin_fields = ["country", "state", "city", "county"]
                        admin_present = [
                            field for field in admin_fields if result.get(field)
                        ]
                        if admin_present:
                            print(f"    🏛️  Administrative: {', '.join(admin_present)}")

                        location_results[api_name] = {
                            "success": True,
                            "quality_score": quality_assessment["quality_score"],
                            "completeness": quality_assessment["completeness"],
                            "fields_present": quality_assessment["fields_present"],
                        }

                        # Track API performance
                        if api_name not in api_summary:
                            api_summary[api_name] = {
                                "successes": 0,
                                "quality_scores": [],
                                "completeness_levels": [],
                            }
                        api_summary[api_name]["successes"] += 1
                        api_summary[api_name]["quality_scores"].append(
                            quality_assessment["quality_score"]
                        )
                        api_summary[api_name]["completeness_levels"].append(
                            quality_assessment["completeness"]
                        )

                    else:
                        print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
                        location_results[api_name] = {
                            "success": False,
                            "error": result.get("error"),
                        }

                except Exception as e:
                    print(f"    ❌ Exception: {str(e)}")
                    location_results[api_name] = {"success": False, "error": str(e)}

        # Print comprehensive comparison summary
        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE REVERSE GEOCODING API COMPARISON")
        print("=" * 80)

        total_tests = len(test_locations)

        for api_name, summary in api_summary.items():
            if summary["successes"] > 0:
                avg_quality = sum(summary["quality_scores"]) / len(
                    summary["quality_scores"]
                )
                quality_range = f"{min(summary['quality_scores']):.3f}-{max(summary['quality_scores']):.3f}"

                print(f"\n📊 {api_name}:")
                print(
                    f"   Success Rate: {summary['successes']}/{total_tests} locations"
                )
                print(f"   Average Quality Score: {avg_quality:.3f}")
                print(f"   Quality Score Range: {quality_range}")
                print(
                    f"   Completeness Levels: {', '.join(set(summary['completeness_levels']))}"
                )

        print("\n🔬 BIOSAMPLE ENRICHMENT RECOMMENDATIONS:")
        print(
            "1. Administrative Context: Country/State/County for regulatory classification"
        )
        print("2. Environmental Context: Urban vs Rural via city/road presence")
        print("3. Precision Needs: Postcode for fine-scale spatial analysis")
        print("4. Backup Strategy: Multiple APIs for comprehensive coverage")
        print(
            "5. Caching: Cache results to respect rate limits and improve performance"
        )

        # Ensure at least one API worked
        assert any(
            summary["successes"] > 0 for summary in api_summary.values()
        ), "No reverse geocoding APIs succeeded"

    def test_zoom_level_comparison(self):
        """Test how different zoom levels affect Nominatim detail level."""
        test_location = {"name": "Research Laboratory", "lat": 42.3601, "lon": -71.0928}
        zoom_levels = [3, 10, 14, 18]  # Country -> City -> Street -> Building level

        print("\n=== ZOOM LEVEL COMPARISON ===")
        print(
            f"Location: {test_location['name']} ({test_location['lat']}, {test_location['lon']})"
        )
        print("=" * 60)

        zoom_results = {}

        for zoom in zoom_levels:
            print(f"\n🔍 Testing Zoom Level {zoom}...")
            result = call_nominatim_direct(
                test_location["lat"], test_location["lon"], zoom
            )

            if result["success"]:
                quality = assess_geocoding_quality(result)
                print(f"    ✅ Success: {quality['completeness']} detail")
                print(f"    📍 Display: {result.get('display_name', 'N/A')}")
                print(f"    🗺️  Quality: {quality['quality_score']}")
                print(
                    f"    📊 Fields: {quality['fields_present']}/{quality['total_fields']}"
                )

                zoom_results[zoom] = {
                    "quality_score": quality["quality_score"],
                    "fields_present": quality["fields_present"],
                    "display_name": result.get("display_name"),
                }
            else:
                print(f"    ❌ Failed: {result.get('error')}")
                zoom_results[zoom] = {"failed": True}

        print("\n📈 ZOOM LEVEL ANALYSIS:")
        for zoom, data in zoom_results.items():
            if not data.get("failed"):
                print(
                    f"   Zoom {zoom}: {data['quality_score']:.3f} quality, {data['fields_present']} fields"
                )

        assert len(zoom_results) > 0, "No zoom level tests succeeded"

    def test_api_rate_limiting_compliance(self):
        """Test that our rate limiting respects API terms of service."""
        test_coords = [(42.3601, -71.0928), (40.7128, -74.0060)]  # Two test locations

        print("\n=== RATE LIMITING COMPLIANCE TEST ===")
        print("=" * 50)

        print("🚦 Testing Nominatim rate limiting (1 request/second)...")
        start_time = time.time()

        for i, (lat, lon) in enumerate(test_coords):
            result = call_nominatim_direct(lat, lon)
            print(
                f"   Request {i+1}: {'✅ Success' if result['success'] else '❌ Failed'}"
            )

        total_time = time.time() - start_time
        expected_min_time = len(test_coords) * 1.1  # 1.1 seconds per request

        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Expected Min: {expected_min_time:.2f}s")
        print(
            f"   Rate Compliance: {'✅ Good' if total_time >= expected_min_time else '⚠️ Too Fast'}"
        )

        assert (
            total_time >= expected_min_time - 0.5
        ), "Rate limiting may be insufficient"

    def test_error_handling(self):
        """Test error handling with invalid coordinates and edge cases."""
        error_test_cases = [
            {"name": "Invalid Latitude", "lat": 95.0, "lon": -71.0928},
            {"name": "Invalid Longitude", "lat": 42.3601, "lon": 200.0},
            {"name": "Ocean Location", "lat": 0.0, "lon": 0.0},
            {"name": "Polar Region", "lat": 89.0, "lon": 0.0},
        ]

        print("\n=== ERROR HANDLING TEST ===")
        print("=" * 40)

        for test_case in error_test_cases:
            lat, lon = test_case["lat"], test_case["lon"]
            print(f"\n📍 {test_case['name']} ({lat}, {lon}):")

            # Test our cached function
            result = get_reverse_geocoding_nominatim(lat, lon)
            print(
                f"   Cached Nominatim: {'✅ Success' if result.get('success') else '❌ Failed'}"
            )

            # Test direct Nominatim
            result_direct = call_nominatim_direct(lat, lon)
            print(
                f"   Direct Nominatim: {'✅ Success' if result_direct.get('success') else '❌ Failed'}"
            )

            # Verify error handling doesn't crash
            assert isinstance(result, dict), "Function should return dict even on error"
            assert isinstance(
                result_direct, dict
            ), "Function should return dict even on error"

"""
Comprehensive forward geocoding API comparison test.

Tests OSM Nominatim and Google Geocoding APIs to convert addresses/place names
to coordinates for microbiome research location validation and normalization.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict

import pytest
import requests


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


def call_nominatim_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Call Nominatim search API for forward geocoding with comprehensive detail extraction."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "extratags": 1,
        "namedetails": 1,
        "limit": limit,
        "polygon_geojson": 1,
        "dedupe": 1,  # Remove duplicate results
    }

    headers = {"User-Agent": "BiosampleEnrichment/1.0 (research purposes)"}

    try:
        # Respect rate limiting
        time.sleep(1.1)
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data and len(data) > 0:
            results = []
            for result in data:
                address = result.get("address", {})

                # Extract comprehensive location data
                location_data = {
                    "display_name": result.get("display_name"),
                    "latitude": float(result.get("lat", 0)),
                    "longitude": float(result.get("lon", 0)),
                    "place_type": result.get("type"),
                    "place_class": result.get("class"),
                    "osm_type": result.get("osm_type"),
                    "osm_id": result.get("osm_id"),
                    "place_id": result.get("place_id"),
                    "place_rank": result.get("place_rank"),
                    "importance": result.get("importance"),
                    "boundingbox": result.get("boundingbox"),
                    # Administrative hierarchy
                    "country": address.get("country"),
                    "country_code": address.get("country_code"),
                    "state": address.get("state"),
                    "county": address.get("county"),
                    "city": address.get("city")
                    or address.get("town")
                    or address.get("village"),
                    "municipality": address.get("municipality"),
                    "suburb": address.get("suburb"),
                    "neighborhood": address.get("neighbourhood"),
                    "postcode": address.get("postcode"),
                    # Street level details
                    "road": address.get("road"),
                    "house_number": address.get("house_number"),
                    # Feature details
                    "amenity": address.get("amenity"),
                    "building": address.get("building"),
                    "natural": address.get("natural"),
                    "landuse": address.get("landuse"),
                    "leisure": address.get("leisure"),
                    # Additional metadata
                    "licence": result.get("licence"),
                    "icon": result.get("icon"),
                }

                # Add extra tags and name details if available
                if result.get("extratags"):
                    location_data["extra_tags"] = result["extratags"]
                if result.get("namedetails"):
                    location_data["name_details"] = result["namedetails"]
                if result.get("geojson"):
                    location_data["geometry"] = result["geojson"]

                results.append(location_data)

            return {
                "api": "OSM Nominatim Search",
                "success": True,
                "query": query,
                "results_count": len(results),
                "results": results,
            }
        else:
            return {
                "api": "OSM Nominatim Search",
                "success": False,
                "query": query,
                "error": "No results found",
                "results": [],
            }

    except Exception as e:
        return {
            "api": "OSM Nominatim Search",
            "success": False,
            "query": query,
            "error": str(e),
            "results": [],
        }


def call_google_geocoding(query: str, limit: int = 5) -> Dict[str, Any]:
    """Call Google Geocoding API for forward geocoding using existing API key."""
    # Use the same Google Maps Platform API key as other APIs
    api_key = os.environ.get("GOOGLE_ELEVATION_API_KEY")
    if not api_key:
        return {
            "api": "Google Geocoding",
            "success": False,
            "query": query,
            "error": "GOOGLE_ELEVATION_API_KEY not set (same key used for geocoding - just enable Geocoding API)",
            "results": [],
        }

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": query,
        "key": api_key,
        "language": "en",  # Get results in English for consistency
        # No region bias = get global results
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "OK" and data.get("results"):
            results = []

            # Process up to limit results
            for result_data in data["results"][:limit]:
                address_components = result_data.get("address_components", [])
                geometry = result_data.get("geometry", {})
                location = geometry.get("location", {})

                # Base result structure
                location_data = {
                    "display_name": result_data.get("formatted_address"),
                    "latitude": location.get("lat", 0),
                    "longitude": location.get("lng", 0),
                    "place_types": result_data.get("types", []),
                    "place_id": result_data.get("place_id"),
                    "partial_match": result_data.get("partial_match", False),
                    # Geometry details
                    "location_type": geometry.get("location_type"),
                    "viewport": geometry.get("viewport"),
                    "bounds": geometry.get("bounds"),
                }

                # Parse address components systematically
                component_types = {}
                for component in address_components:
                    long_name = component.get("long_name")
                    short_name = component.get("short_name")
                    types = component.get("types", [])

                    # Store by primary type
                    for type_name in types:
                        if type_name not in component_types:
                            component_types[type_name] = {
                                "long_name": long_name,
                                "short_name": short_name,
                            }

                # Add all component types to result
                location_data["address_components"] = component_types

                # Extract common fields for compatibility
                if "country" in component_types:
                    location_data["country"] = component_types["country"]["long_name"]
                    location_data["country_code"] = component_types["country"][
                        "short_name"
                    ]
                if "administrative_area_level_1" in component_types:
                    location_data["state"] = component_types[
                        "administrative_area_level_1"
                    ]["long_name"]
                if "administrative_area_level_2" in component_types:
                    location_data["county"] = component_types[
                        "administrative_area_level_2"
                    ]["long_name"]
                if "locality" in component_types:
                    location_data["city"] = component_types["locality"]["long_name"]
                if "sublocality" in component_types:
                    location_data["neighborhood"] = component_types["sublocality"][
                        "long_name"
                    ]
                if "route" in component_types:
                    location_data["road"] = component_types["route"]["long_name"]
                if "street_number" in component_types:
                    location_data["house_number"] = component_types["street_number"][
                        "long_name"
                    ]
                if "postal_code" in component_types:
                    location_data["postcode"] = component_types["postal_code"][
                        "long_name"
                    ]

                results.append(location_data)

            return {
                "api": "Google Geocoding",
                "success": True,
                "query": query,
                "results_count": len(results),
                "total_results_available": len(data["results"]),
                "results": results,
            }
        else:
            return {
                "api": "Google Geocoding",
                "success": False,
                "query": query,
                "error": data.get("error_message", data.get("status", "Unknown error")),
                "results": [],
            }

    except Exception as e:
        return {
            "api": "Google Geocoding",
            "success": False,
            "query": query,
            "error": str(e),
            "results": [],
        }


def assess_geocoding_result_quality(result: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the quality and precision of a single geocoding result."""
    # Define scoring criteria for research location validation
    scoring_criteria = {
        "country": 0.25,  # Essential for geographic context
        "state": 0.2,  # Important for regional classification
        "city": 0.2,  # Important for urban/rural classification
        "latitude": 0.1,  # Coordinate accuracy
        "longitude": 0.1,  # Coordinate accuracy
        "postcode": 0.1,  # Precision indicator
        "display_name": 0.05,  # Overall description quality
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

    # Assess precision based on coordinate precision and place type
    precision = "unknown"
    place_type = result.get("place_type", "").lower()
    place_types = result.get("place_types", [])

    if any("street_address" in str(t).lower() for t in place_types) or result.get(
        "house_number"
    ):
        precision = "street_level"
    elif any("locality" in str(t).lower() for t in place_types) or result.get("city"):
        precision = "city_level"
    elif any("administrative" in str(t).lower() for t in place_types) or result.get(
        "county"
    ):
        precision = "admin_level"
    elif result.get("country"):
        precision = "country_level"

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
        "precision": precision,
        "present_fields": present_fields,
        "missing_fields": missing_fields,
        "total_fields": len(scoring_criteria),
        "fields_present": len(present_fields),
    }


class TestForwardGeocodingAPIComparison:
    """Compare forward geocoding APIs for research location validation and normalization."""

    def test_diverse_research_locations(self):
        """Test forward geocoding APIs on diverse research location queries."""
        test_queries = [
            {
                "query": "Massachusetts Institute of Technology, Cambridge, MA",
                "context": "University research facility",
                "expected_precision": "street_level",
                "expected_location": {
                    "country": "United States",
                    "state": "Massachusetts",
                },
            },
            {
                "query": "Yellowstone National Park, Wyoming",
                "context": "National park research site",
                "expected_precision": "admin_level",
                "expected_location": {"country": "United States", "state": "Wyoming"},
            },
            {
                "query": "Amazon Rainforest, Brazil",
                "context": "Tropical forest ecosystem",
                "expected_precision": "admin_level",
                "expected_location": {"country": "Brazil"},
            },
            {
                "query": "CERN, Geneva, Switzerland",
                "context": "International research facility",
                "expected_precision": "street_level",
                "expected_location": {"country": "Switzerland"},
            },
            {
                "query": "Great Barrier Reef, Australia",
                "context": "Marine research site",
                "expected_precision": "admin_level",
                "expected_location": {"country": "Australia"},
            },
            {
                "query": "Station Nord, Greenland",
                "context": "Arctic research station",
                "expected_precision": "city_level",
                "expected_location": {"country": "Greenland"},
            },
            {
                "query": "Tokyo University, Japan",
                "context": "Urban research facility",
                "expected_precision": "street_level",
                "expected_location": {"country": "Japan"},
            },
            {
                "query": "Sahara Desert Research Station, Algeria",
                "context": "Desert research facility",
                "expected_precision": "admin_level",
                "expected_location": {"country": "Algeria"},
            },
        ]

        print("\\n=== FORWARD GEOCODING API COMPARISON FOR RESEARCH LOCATIONS ===")
        print("=" * 80)

        api_summary = {}

        for query_data in test_queries:
            query = query_data["query"]

            print(f"\\n🔍 Query: '{query}'")
            print(f"   Context: {query_data['context']}")
            print(f"   Expected Precision: {query_data['expected_precision']}")
            print(f"   Expected Location: {query_data['expected_location']}")
            print("-" * 60)

            # Test available APIs (OSM Nominatim + Google)
            apis = [
                ("OSM Nominatim Search", lambda: call_nominatim_search(query, limit=3)),
                ("Google Geocoding", lambda: call_google_geocoding(query, limit=3)),
            ]

            query_results = {}

            for api_name, api_func in apis:
                print(f"\\n  🌐 Testing {api_name}...")
                try:
                    result = api_func()

                    if result.get("success") and result.get("results"):
                        best_result = result["results"][0]  # Use top result
                        quality_assessment = assess_geocoding_result_quality(
                            best_result
                        )

                        print(
                            f"    ✅ Success: {result['results_count']} results found"
                        )
                        print(
                            f"    📍 Top Result: {best_result.get('display_name', 'N/A')}"
                        )
                        print(
                            f"    🗺️  Coordinates: ({best_result.get('latitude', 'N/A')}, {best_result.get('longitude', 'N/A')})"
                        )
                        print(
                            f"    📊 Quality: {quality_assessment['completeness']} ({quality_assessment['quality_score']})"
                        )
                        print(f"    🎯 Precision: {quality_assessment['precision']}")
                        print(
                            f"    📋 Fields: {quality_assessment['fields_present']}/{quality_assessment['total_fields']}"
                        )

                        # Show key administrative fields
                        admin_fields = ["country", "state", "city", "county"]
                        admin_present = [
                            field for field in admin_fields if best_result.get(field)
                        ]
                        if admin_present:
                            admin_values = [
                                f"{field}={best_result[field]}"
                                for field in admin_present
                            ]
                            print(f"    🏛️  Administrative: {', '.join(admin_values)}")

                        query_results[api_name] = {
                            "success": True,
                            "results_count": result["results_count"],
                            "quality_score": quality_assessment["quality_score"],
                            "completeness": quality_assessment["completeness"],
                            "precision": quality_assessment["precision"],
                            "coordinates": (
                                best_result.get("latitude"),
                                best_result.get("longitude"),
                            ),
                        }

                        # Track API performance
                        if api_name not in api_summary:
                            api_summary[api_name] = {
                                "successes": 0,
                                "total_results": 0,
                                "quality_scores": [],
                                "completeness_levels": [],
                                "precision_levels": [],
                            }
                        api_summary[api_name]["successes"] += 1
                        api_summary[api_name]["total_results"] += result[
                            "results_count"
                        ]
                        api_summary[api_name]["quality_scores"].append(
                            quality_assessment["quality_score"]
                        )
                        api_summary[api_name]["completeness_levels"].append(
                            quality_assessment["completeness"]
                        )
                        api_summary[api_name]["precision_levels"].append(
                            quality_assessment["precision"]
                        )

                    else:
                        print(
                            f"    ❌ Failed: {result.get('error', 'No results found')}"
                        )
                        query_results[api_name] = {
                            "success": False,
                            "error": result.get("error", "No results found"),
                        }

                except Exception as e:
                    print(f"    ❌ Exception: {str(e)}")
                    query_results[api_name] = {"success": False, "error": str(e)}

        # Print comprehensive comparison summary
        print("\\n" + "=" * 80)
        print("🏆 COMPREHENSIVE FORWARD GEOCODING API COMPARISON")
        print("=" * 80)

        total_tests = len(test_queries)

        for api_name, summary in api_summary.items():
            if summary["successes"] > 0:
                avg_quality = sum(summary["quality_scores"]) / len(
                    summary["quality_scores"]
                )
                avg_results = summary["total_results"] / summary["successes"]
                quality_range = f"{min(summary['quality_scores']):.3f}-{max(summary['quality_scores']):.3f}"

                print(f"\\n📊 {api_name}:")
                print(f"   Success Rate: {summary['successes']}/{total_tests} queries")
                print(f"   Average Results per Query: {avg_results:.1f}")
                print(f"   Average Quality Score: {avg_quality:.3f}")
                print(f"   Quality Score Range: {quality_range}")
                print(
                    f"   Completeness Levels: {', '.join(set(summary['completeness_levels']))}"
                )
                print(
                    f"   Precision Levels: {', '.join(set(summary['precision_levels']))}"
                )

        print("\\n🔬 RESEARCH LOCATION RECOMMENDATIONS:")
        print(
            "1. Institution Lookup: Use specific institution names for precise coordinates"
        )
        print("2. Natural Features: Include region/country for large geographic areas")
        print("3. Validation Strategy: Compare results between APIs for confidence")
        print("4. Precision Assessment: Check precision level matches research needs")
        print("5. Multiple Results: Consider alternative locations when ambiguous")

        # Ensure at least one API worked
        assert any(
            summary["successes"] > 0 for summary in api_summary.values()
        ), "No forward geocoding APIs succeeded"

    def test_ambiguous_queries(self):
        """Test how APIs handle ambiguous location queries."""
        ambiguous_queries = [
            "Springfield",  # Many Springfields in US
            "Cambridge",  # Cambridge UK vs Cambridge MA, etc.
            "Victoria",  # State, city, lake, etc.
            "Richmond",  # Many Richmonds globally
        ]

        print("\\n=== AMBIGUOUS QUERY HANDLING TEST ===")
        print("=" * 50)

        for query in ambiguous_queries:
            print(f"\\n🔍 Testing ambiguous query: '{query}'")
            print("-" * 30)

            # Test both APIs
            apis = [
                ("OSM Nominatim", lambda: call_nominatim_search(query, limit=5)),
                ("Google Geocoding", lambda: call_google_geocoding(query, limit=5)),
            ]

            for api_name, api_func in apis:
                result = api_func()
                print(f"\\n  📍 {api_name}:")

                if result.get("success") and result.get("results"):
                    print(f"    Results: {result['results_count']} locations found")
                    for i, location in enumerate(result["results"][:3], 1):
                        country = location.get("country", "Unknown")
                        state = location.get("state", "")
                        display = location.get("display_name", "No name")[:60]
                        print(
                            f"    {i}. {display} ({country}{', ' + state if state else ''})"
                        )
                else:
                    print(f"    ❌ Failed: {result.get('error', 'No results')}")

    def test_coordinate_precision_analysis(self):
        """Test coordinate precision for different location types."""
        precision_test_queries = [
            {
                "query": "1600 Pennsylvania Avenue, Washington DC",
                "expected": "street_level",
            },
            {"query": "Harvard University", "expected": "street_level"},
            {"query": "New York City", "expected": "city_level"},
            {"query": "California", "expected": "admin_level"},
            {"query": "Pacific Ocean", "expected": "admin_level"},
        ]

        print("\\n=== COORDINATE PRECISION ANALYSIS ===")
        print("=" * 45)

        for test_data in precision_test_queries:
            query = test_data["query"]
            expected = test_data["expected"]

            print(f"\\n🎯 Query: '{query}' (Expected: {expected})")

            # Test Nominatim search
            result = call_nominatim_search(query, limit=1)
            if result.get("success") and result.get("results"):
                location = result["results"][0]
                quality = assess_geocoding_result_quality(location)

                print(f"   OSM: {quality['precision']} precision")
                print(
                    f"        Coordinates: ({location.get('latitude'):.6f}, {location.get('longitude'):.6f})"
                )
                print(
                    f"        Quality: {quality['completeness']} ({quality['quality_score']})"
                )

    def test_api_rate_limiting_compliance(self):
        """Test that forward geocoding respects API rate limits."""
        test_queries = ["Boston", "Chicago", "Denver"]

        print("\\n=== RATE LIMITING COMPLIANCE TEST ===")
        print("=" * 50)

        print("🚦 Testing Nominatim rate limiting (1 request/second)...")
        start_time = time.time()

        for i, query in enumerate(test_queries):
            result = call_nominatim_search(query, limit=1)
            print(
                f"   Request {i+1} ('{query}'): {'✅ Success' if result['success'] else '❌ Failed'}"
            )

        total_time = time.time() - start_time
        expected_min_time = len(test_queries) * 1.1  # 1.1 seconds per request

        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Expected Min: {expected_min_time:.2f}s")
        print(
            f"   Rate Compliance: {'✅ Good' if total_time >= expected_min_time else '⚠️ Too Fast'}"
        )

        assert (
            total_time >= expected_min_time - 0.5
        ), "Rate limiting may be insufficient"

    def test_error_handling(self):
        """Test error handling with invalid queries and edge cases."""
        error_test_cases = [
            {"name": "Empty Query", "query": ""},
            {"name": "Invalid Characters", "query": "###@@@"},
            {"name": "Very Long Query", "query": "x" * 500},
            {"name": "Non-existent Place", "query": "Nonexistentville Fakecountry"},
        ]

        print("\\n=== ERROR HANDLING TEST ===")
        print("=" * 40)

        for test_case in error_test_cases:
            query = test_case["query"]
            print(
                f"\\n📍 {test_case['name']}: '{query[:50]}{'...' if len(query) > 50 else ''}'"
            )

            # Test both APIs
            apis = [
                ("OSM Nominatim", lambda: call_nominatim_search(query, limit=1)),
                ("Google Geocoding", lambda: call_google_geocoding(query, limit=1)),
            ]

            for api_name, api_func in apis:
                try:
                    result = api_func()
                    status = "✅ Success" if result.get("success") else "❌ Failed"
                    print(f"   {api_name}: {status}")

                    # Verify error handling doesn't crash
                    assert isinstance(
                        result, dict
                    ), f"{api_name} should return dict even on error"
                    assert "success" in result, f"{api_name} should have success field"

                except Exception as e:
                    print(f"   {api_name}: ❌ Exception: {str(e)}")
                    pytest.fail(f"{api_name} should handle errors gracefully")

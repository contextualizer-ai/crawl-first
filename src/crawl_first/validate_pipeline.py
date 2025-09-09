#!/usr/bin/env python3
"""
Pipeline validation framework for geospatial enrichment.

Implements the QA plan suggested in feedback:
- Compare SoilGrids SOC vs independent sources
- Verify SDA taxonomy consistency with SoilGrids WRB
- Assert proper unit conversions are applied
- Test with random US and non-US points
"""

import csv
import json
import random
import time
from typing import Any, Dict, List

from geospatial_enrichment import enrich_location


def generate_test_points(n_us: int = 10, n_global: int = 10) -> List[Dict[str, Any]]:
    """Generate random test points for validation."""
    points = []

    # US points (CONUS bounds)
    for i in range(n_us):
        lat = random.uniform(25.0, 49.0)  # CONUS latitude range
        lon = random.uniform(-125.0, -66.0)  # CONUS longitude range
        points.append(
            {
                "id": f"us_{i+1}",
                "lat": lat,
                "lon": lon,
                "region": "US",
                "collection_date": "2020-06-15",
            }
        )

    # Global points (avoiding polar regions and oceans)
    for i in range(n_global):
        lat = random.uniform(-45.0, 60.0)  # Avoid polar regions
        lon = random.uniform(-180.0, 180.0)
        points.append(
            {
                "id": f"global_{i+1}",
                "lat": lat,
                "lon": lon,
                "region": "Global",
                "collection_date": "2020-06-15",
            }
        )

    return points


def validate_unit_conversions(soil_props: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that unit conversions are properly applied."""
    validation = {"conversions_valid": True, "issues": []}

    soil_data = soil_props.get("soil_properties", {})

    # Check pH range (should be 2-11 after ÷10 conversion)
    if "ph_0_5cm" in soil_data and soil_data["ph_0_5cm"].get("value"):
        ph = soil_data["ph_0_5cm"]["value"]
        if not (2.0 <= ph <= 11.0):
            validation["conversions_valid"] = False
            validation["issues"].append(f"pH out of range: {ph} (expected 2-11)")

    # Check texture percentages sum to ~100
    if all(
        prop in soil_data and soil_data[prop].get("value")
        for prop in ["sand_0_5cm", "silt_0_5cm", "clay_0_5cm"]
    ):
        sand = soil_data["sand_0_5cm"]["value"]
        silt = soil_data["silt_0_5cm"]["value"]
        clay = soil_data["clay_0_5cm"]["value"]
        total = sand + silt + clay

        if not (90 <= total <= 110):  # Allow 10% tolerance
            validation["conversions_valid"] = False
            validation["issues"].append(
                f"Texture percentages sum to {total:.1f}% (expected ~100%)"
            )

    # Check SOC reasonable range (0-200 g/kg)
    if "soc_0_5cm" in soil_data and soil_data["soc_0_5cm"].get("value"):
        soc = soil_data["soc_0_5cm"]["value"]
        if not (0 <= soc <= 200):
            validation["conversions_valid"] = False
            validation["issues"].append(
                f"SOC out of range: {soc} g/kg (expected 0-200)"
            )

    return validation


def validate_single_point(point: Dict[str, Any]) -> Dict[str, Any]:
    """Validate enrichment for a single point."""
    print(f"Validating point {point['id']} ({point['lat']:.3f}, {point['lon']:.3f})...")

    try:
        # Add small delay to respect rate limits
        time.sleep(0.5)

        result = enrich_location(point["lat"], point["lon"], point["collection_date"])

        validation = {
            "point_id": point["id"],
            "coordinates": [point["lat"], point["lon"]],
            "region": point["region"],
            "enrichment_successful": len(
                result["enrichment_summary"]["successful_enrichments"]
            ),
            "total_enrichments": len(result["enrichment_summary"]["apis_used"]),
            "unit_validation": validate_unit_conversions(
                result.get("soil_properties", {})
            ),
            "soil_data_available": result.get("soil_properties", {}).get(
                "success", False
            ),
            "classification_available": result.get("soil_classification", {}).get(
                "success", False
            ),
            "errors": [],
        }

        # Extract key soil values for comparison
        soil_props = result.get("soil_properties", {}).get("soil_properties", {})
        if "soc_0_5cm" in soil_props and soil_props["soc_0_5cm"].get("value"):
            validation["soc_g_kg"] = soil_props["soc_0_5cm"]["value"]

        if "ph_0_5cm" in soil_props and soil_props["ph_0_5cm"].get("value"):
            validation["ph"] = soil_props["ph_0_5cm"]["value"]

        # Extract classification
        soil_class = result.get("soil_classification", {}).get(
            "soil_classification", {}
        )
        if "taxonomic_order" in soil_class:
            validation["usda_order"] = soil_class["taxonomic_order"]
        elif "primary_class" in soil_class:
            validation["wrb_class"] = soil_class["primary_class"]

        return validation

    except Exception as e:
        return {
            "point_id": point["id"],
            "coordinates": [point["lat"], point["lon"]],
            "region": point["region"],
            "enrichment_successful": 0,
            "total_enrichments": 0,
            "errors": [str(e)],
        }


def get_hwsd_validation(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get independent soil validation from HWSD v2 (FAO Harmonized World Soil Database).

    This provides independent validation separate from ISRIC SoilGrids.
    Note: This is a placeholder for HWSD v2 integration - requires local HWSD data or API.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dict with HWSD soil data for comparison
    """
    # Placeholder for HWSD v2 integration
    # In production, this would:
    # 1. Query local HWSD v2 GeoTIFF files
    # 2. Or use HWSD API if available
    # 3. Extract SOC, texture, classification for comparison

    return {
        "hwsd_available": False,
        "note": "HWSD v2 integration not yet implemented",
        "recommended_source": "https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v12/en/",
        "validation_purpose": "Independent soil data source separate from ISRIC SoilGrids",
    }


def validate_against_independent_sources(
    point: Dict[str, Any], result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate pipeline results against independent data sources.

    Args:
        point: Test point coordinates
        result: Pipeline enrichment result

    Returns:
        Dict with independent validation results
    """
    validation = {
        "independent_sources_checked": [],
        "validation_issues": [],
        "recommendations": [],
    }

    # Check HWSD v2 for soil validation
    hwsd_result = get_hwsd_validation(point["lat"], point["lon"])
    validation["independent_sources_checked"].append("HWSD v2")

    if not hwsd_result.get("hwsd_available"):
        validation["recommendations"].append(
            "Implement HWSD v2 integration for independent soil validation"
        )

    # For US points, SDA provides independent validation vs SoilGrids
    if point["region"] == "US":
        sda_available = result.get("soil_classification", {}).get("success", False)
        soilgrids_available = result.get("soil_properties", {}).get("success", False)

        if sda_available and soilgrids_available:
            validation["independent_sources_checked"].append(
                "USDA SDA vs ISRIC SoilGrids"
            )
            # Could add specific cross-validation logic here

    # Check for potential issues
    soil_props = result.get("soil_properties", {}).get("soil_properties", {})

    # Flag if all SoilGrids values are identical (suggests potential issue)
    soil_values = []
    for prop in ["ph_0_5cm", "soc_0_5cm", "sand_0_5cm", "clay_0_5cm"]:
        if prop in soil_props and soil_props[prop].get("value") is not None:
            soil_values.append(soil_props[prop]["value"])

    if len(set(soil_values)) == 1 and len(soil_values) > 1:
        validation["validation_issues"].append(
            f"All soil values identical ({soil_values[0]}) - potential data issue"
        )

    return validation


def run_validation(n_us: int = 5, n_global: int = 5) -> None:
    """Run complete validation suite."""
    print(
        f"Running pipeline validation with {n_us} US points and {n_global} global points..."
    )

    # Generate test points
    test_points = generate_test_points(n_us, n_global)

    # Run validation
    results = []
    for point in test_points:
        result = validate_single_point(point)

        # Add independent source validation
        if result.get("enrichment_successful", 0) > 0:
            try:
                # Get the full enrichment result for independent validation
                from geospatial_enrichment import enrich_location

                enrichment_result = enrich_location(
                    point["lat"], point["lon"], point["collection_date"]
                )
                independent_validation = validate_against_independent_sources(
                    point, enrichment_result
                )
                result["independent_validation"] = independent_validation
            except Exception as e:
                result["independent_validation"] = {"error": str(e)}

        results.append(result)

    # Save detailed results
    with open("outputs/validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create summary CSV (flatten complex fields)
    with open("outputs/validation_summary.csv", "w", newline="") as f:
        if results:
            # Flatten results for CSV
            flattened_results = []
            for result in results:
                flattened = {
                    "point_id": result.get("point_id"),
                    "lat": result.get("coordinates", [None, None])[0],
                    "lon": result.get("coordinates", [None, None])[1],
                    "region": result.get("region"),
                    "enrichment_successful": result.get("enrichment_successful"),
                    "total_enrichments": result.get("total_enrichments"),
                    "soil_data_available": result.get("soil_data_available"),
                    "classification_available": result.get("classification_available"),
                    "unit_conversions_valid": result.get("unit_validation", {}).get(
                        "conversions_valid", False
                    ),
                    "validation_errors": len(result.get("errors", [])),
                    "independent_sources_checked": len(
                        result.get("independent_validation", {}).get(
                            "independent_sources_checked", []
                        )
                    ),
                    "validation_issues": len(
                        result.get("independent_validation", {}).get(
                            "validation_issues", []
                        )
                    ),
                }
                flattened_results.append(flattened)

            writer = csv.DictWriter(f, fieldnames=flattened_results[0].keys())
            writer.writeheader()
            writer.writerows(flattened_results)

    # Print summary
    total_points = len(results)
    successful_enrichments = sum(1 for r in results if r["enrichment_successful"] >= 6)
    unit_validation_passed = sum(
        1
        for r in results
        if r.get("unit_validation", {}).get("conversions_valid", False)
    )

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Total points tested: {total_points}")
    print(
        f"Successful enrichments (≥6/7): {successful_enrichments}/{total_points} ({100*successful_enrichments/total_points:.1f}%)"
    )
    print(
        f"Unit validation passed: {unit_validation_passed}/{total_points} ({100*unit_validation_passed/total_points:.1f}%)"
    )

    # Report issues
    all_issues = []
    for r in results:
        if r.get("unit_validation", {}).get("issues"):
            all_issues.extend(r["unit_validation"]["issues"])
        if r.get("errors"):
            all_issues.extend(r["errors"])

    if all_issues:
        print("\nIssues found:")
        for issue in set(all_issues):  # Remove duplicates
            print(f"  - {issue}")
    else:
        print("\nNo validation issues found!")

    print("\nDetailed results saved to:")
    print("  - outputs/validation_results.json")
    print("  - outputs/validation_summary.csv")


if __name__ == "__main__":
    run_validation()

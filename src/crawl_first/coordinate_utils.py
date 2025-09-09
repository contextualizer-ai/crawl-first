#!/usr/bin/env python3
"""
Coordinate utility functions for geospatial calculations.

Provides distance calculations and coordinate transformations for the
unified biosample enrichment pipeline.
"""

import math
from typing import Any, Dict, Tuple


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.

    Args:
        lat1, lon1: Latitude and longitude of first point in decimal degrees
        lat2, lon2: Latitude and longitude of second point in decimal degrees

    Returns:
        Distance in kilometers

    Example:
        >>> haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)  # NYC to LA
        3944.42
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in kilometers
    earth_radius_km = 6371.0

    return earth_radius_km * c


def vincenty_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance using Vincenty's formulae (more accurate than Haversine).

    This is a simplified version. For production use, consider using geopy.distance.vincenty
    which handles edge cases and uses the full iterative algorithm.

    Args:
        lat1, lon1: Latitude and longitude of first point in decimal degrees
        lat2, lon2: Latitude and longitude of second point in decimal degrees

    Returns:
        Distance in kilometers
    """
    # For short distances, Haversine is sufficient and faster
    # This is a placeholder - in production, implement full Vincenty or use geopy
    return haversine_distance(lat1, lon1, lat2, lon2)


def calculate_coordinate_offset(
    lat: float, lon: float, distance_km: float, bearing_degrees: float
) -> Tuple[float, float]:
    """
    Calculate new coordinates given a starting point, distance, and bearing.

    Args:
        lat, lon: Starting coordinates in decimal degrees
        distance_km: Distance to travel in kilometers
        bearing_degrees: Bearing in degrees (0 = North, 90 = East, etc.)

    Returns:
        Tuple of (new_latitude, new_longitude)
    """
    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_degrees)

    # Earth radius in km
    R = 6371.0

    # Calculate new latitude
    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(distance_km / R)
        + math.cos(lat_rad) * math.sin(distance_km / R) * math.cos(bearing_rad)
    )

    # Calculate new longitude
    new_lon_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat_rad),
        math.cos(distance_km / R) - math.sin(lat_rad) * math.sin(new_lat_rad),
    )

    # Convert back to degrees
    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)

    return new_lat, new_lon


def get_coordinate_bounds(lat: float, lon: float, radius_km: float) -> Dict[str, float]:
    """
    Get bounding box coordinates for a circle of given radius around a point.

    Args:
        lat, lon: Center point coordinates in decimal degrees
        radius_km: Radius in kilometers

    Returns:
        Dict with min_lat, max_lat, min_lon, max_lon
    """
    # Calculate approximate degree offsets
    # 1 degree latitude ≈ 111 km
    lat_offset = radius_km / 111.0

    # 1 degree longitude varies by latitude: ≈ 111 * cos(latitude) km
    lon_offset = radius_km / (111.0 * math.cos(math.radians(lat)))

    return {
        "min_lat": lat - lat_offset,
        "max_lat": lat + lat_offset,
        "min_lon": lon - lon_offset,
        "max_lon": lon + lon_offset,
        "center_lat": lat,
        "center_lon": lon,
        "radius_km": radius_km,
    }


def coordinate_precision_km(lat: float) -> Dict[str, float]:
    """
    Calculate the distance represented by 1 decimal degree at a given latitude.

    Args:
        lat: Latitude in decimal degrees

    Returns:
        Dict with km_per_degree_lat and km_per_degree_lon
    """
    # 1 degree latitude is always approximately 111 km
    km_per_degree_lat = 111.0

    # 1 degree longitude varies by latitude
    km_per_degree_lon = 111.0 * math.cos(math.radians(lat))

    return {
        "km_per_degree_lat": km_per_degree_lat,
        "km_per_degree_lon": km_per_degree_lon,
        "decimal_places_for_100m": 3,  # ~111m precision at equator
        "decimal_places_for_10m": 4,  # ~11m precision at equator
        "decimal_places_for_1m": 5,  # ~1m precision at equator
    }


def format_coordinates(lat: float, lon: float, precision: int = 6) -> str:
    """
    Format coordinates as a human-readable string.

    Args:
        lat, lon: Coordinates in decimal degrees
        precision: Number of decimal places

    Returns:
        Formatted coordinate string
    """
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"

    return f"{abs(lat):.{precision}f}°{lat_dir}, {abs(lon):.{precision}f}°{lon_dir}"


def validate_coordinates(lat: float, lon: float) -> Dict[str, Any]:
    """
    Validate coordinate values and provide metadata.

    Args:
        lat, lon: Coordinates to validate

    Returns:
        Dict with validation results and metadata
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "metadata": {
            "hemisphere_lat": "Northern" if lat >= 0 else "Southern",
            "hemisphere_lon": "Eastern" if lon >= 0 else "Western",
            "formatted": format_coordinates(lat, lon),
            "precision_info": coordinate_precision_km(lat),
        },
    }

    # Validate latitude range
    if not (-90 <= lat <= 90):
        validation["valid"] = False
        validation["errors"].append(f"Latitude {lat} outside valid range [-90, 90]")

    # Validate longitude range
    if not (-180 <= lon <= 180):
        validation["valid"] = False
        validation["errors"].append(f"Longitude {lon} outside valid range [-180, 180]")

    # Warnings for unusual locations
    if abs(lat) < 0.1 and abs(lon) < 0.1:
        validation["warnings"].append("Coordinates very close to 0,0 (Null Island)")

    if lat == 0:
        validation["warnings"].append("Coordinates on the Equator")

    if lon == 0:
        validation["warnings"].append("Coordinates on the Prime Meridian")

    if abs(lat) > 80:
        validation["warnings"].append("High latitude location (potential polar region)")

    return validation


def find_nearest_station_distance(
    sample_lat: float, sample_lon: float, station_coordinates: list
) -> Dict[str, Any]:
    """
    Find the nearest weather station and calculate distance.

    Args:
        sample_lat, sample_lon: Sample coordinates
        station_coordinates: List of dicts with 'lat', 'lon', 'name' keys

    Returns:
        Dict with nearest station info and distance
    """
    if not station_coordinates:
        return {
            "nearest_station": None,
            "distance_km": None,
            "error": "No station coordinates provided",
        }

    nearest_distance = float("inf")
    nearest_station = None

    for station in station_coordinates:
        try:
            distance = haversine_distance(
                sample_lat, sample_lon, station["lat"], station["lon"]
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_station = station

        except (KeyError, TypeError, ValueError):
            continue

    if nearest_station:
        return {
            "nearest_station": nearest_station,
            "distance_km": round(nearest_distance, 2),
            "distance_quality": get_distance_quality_rating(nearest_distance),
            "sample_coordinates": {"lat": sample_lat, "lon": sample_lon},
            "station_coordinates": {
                "lat": nearest_station["lat"],
                "lon": nearest_station["lon"],
            },
        }
    else:
        return {
            "nearest_station": None,
            "distance_km": None,
            "error": "Could not calculate distance to any station",
        }


def get_distance_quality_rating(distance_km: float) -> Dict[str, Any]:
    """
    Provide quality rating for data based on distance from source.

    Args:
        distance_km: Distance in kilometers

    Returns:
        Dict with quality rating and description
    """
    if distance_km < 1:
        return {
            "rating": "excellent",
            "score": 5,
            "description": "Very close proximity (<1km) - high confidence",
        }
    elif distance_km < 5:
        return {
            "rating": "very_good",
            "score": 4,
            "description": "Close proximity (<5km) - good confidence",
        }
    elif distance_km < 25:
        return {
            "rating": "good",
            "score": 3,
            "description": "Moderate distance (<25km) - reasonable confidence",
        }
    elif distance_km < 100:
        return {
            "rating": "fair",
            "score": 2,
            "description": "Significant distance (<100km) - use with caution",
        }
    else:
        return {
            "rating": "poor",
            "score": 1,
            "description": "Large distance (>100km) - low confidence",
        }


if __name__ == "__main__":
    # Test the functions
    print("Testing coordinate utility functions...")

    # Test distance calculation
    nyc_lat, nyc_lon = 40.7128, -74.0060
    la_lat, la_lon = 34.0522, -118.2437

    distance = haversine_distance(nyc_lat, nyc_lon, la_lat, la_lon)
    print(f"Distance NYC to LA: {distance:.2f} km")

    # Test coordinate validation
    validation = validate_coordinates(nyc_lat, nyc_lon)
    print(f"NYC coordinates valid: {validation['valid']}")
    print(f"Formatted: {validation['metadata']['formatted']}")

    # Test bounding box
    bounds = get_coordinate_bounds(nyc_lat, nyc_lon, 50)
    print(f"50km radius around NYC: {bounds}")

    # Test station distance calculation
    fake_stations = [
        {"lat": 40.7, "lon": -74.0, "name": "Station A"},
        {"lat": 40.8, "lon": -74.1, "name": "Station B"},
    ]

    nearest = find_nearest_station_distance(nyc_lat, nyc_lon, fake_stations)
    print(f"Nearest station: {nearest}")

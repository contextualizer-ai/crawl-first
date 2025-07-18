"""
Biosample analysis orchestration for crawl-first.

Coordinates comprehensive analysis of NMDC biosample records.
"""

from typing import Any, Dict, Optional

from landuse_mcp.main import get_landuse_dates

from .analysis import (
    cached_fetch_nmdc_entity_by_id,
    find_closest_landuse_date,
    get_land_cover_analysis,
    get_publication_analysis,
    get_soil_analysis,
    get_weather_analysis,
    parse_collection_date,
)
from .cache import cache_key, get_cache, save_cache
from .geospatial import calculate_distance, geocode_location_name, get_elevation
from .osm import query_osm_features, summarize_osm_features


def get_geospatial_analysis(
    lat: float, lon: float, radius: float = 1000
) -> Dict[str, Any]:
    """Get comprehensive geospatial analysis for a location."""
    from .geospatial import reverse_geocode

    analysis: Dict[str, Any] = {
        "coordinates": {"latitude": lat, "longitude": lon},
        "elevation": None,
        "place_info": {},
        "osm_features": {},
        "environmental_summary": {},
    }

    # Get elevation
    elevation = get_elevation(lat, lon)
    if elevation is not None:
        analysis["elevation"] = {
            "meters": elevation,
            "feet": round(elevation * 3.28084, 1),
        }

    # Get place information via reverse geocoding
    place_info = reverse_geocode(lat, lon)
    analysis["place_info"] = place_info

    # Get nearby OSM features
    osm_features = query_osm_features(lat, lon, radius)
    osm_summary = summarize_osm_features(osm_features, lat, lon)
    analysis["osm_features"] = osm_summary

    # Create environmental summary - counts come from feature_counts metadata
    feature_counts = osm_summary["metadata"]["feature_counts"]

    # Calculate counts from metadata (includes both named and unnamed features)
    water_count = 0
    natural_count = 0
    protected_count = 0

    for category, types in feature_counts.items():
        for feature_type, count in types.items():
            if category in ["water", "waterway"] or feature_type in [
                "water",
                "lake",
                "pond",
                "river",
                "stream",
            ]:
                water_count += count
            if category == "natural" or feature_type in [
                "forest",
                "wood",
                "grassland",
                "wetland",
            ]:
                natural_count += count
            if feature_type in [
                "nature_reserve",
                "national_park",
                "protected_area",
                "conservation",
            ]:
                protected_count += count

    # Create environmental summary with explicit typing
    env_summary: Dict[str, Any] = {
        "total_environmental_features": osm_summary["metadata"]["total_features"],
        "water_features_nearby": water_count,
        "natural_areas_nearby": natural_count,
        "protected_areas_nearby": protected_count,
        "dominant_land_types": [],
        "ecological_context": "",
    }
    analysis["environmental_summary"] = env_summary

    # Determine dominant land types
    if feature_counts:
        all_types = []
        for category, types in feature_counts.items():
            for type_name, count in types.items():
                all_types.append((f"{category}:{type_name}", count))

        dominant = sorted(all_types, key=lambda x: x[1], reverse=True)[:3]
        analysis["environmental_summary"]["dominant_land_types"] = [
            {"type": t[0], "count": t[1]} for t in dominant
        ]

    # Generate ecological context description
    eco_description = []
    if water_count > 0:
        eco_description.append(f"{water_count} water feature(s)")
    if natural_count > 0:
        eco_description.append(f"{natural_count} natural area(s)")
    if protected_count > 0:
        eco_description.append(f"{protected_count} protected area(s)")

    if eco_description:
        analysis["environmental_summary"][
            "ecological_context"
        ] = f"Location has {', '.join(eco_description)} within {radius}m radius"
    else:
        analysis["environmental_summary"][
            "ecological_context"
        ] = f"No major environmental features found within {radius}m radius"

    return analysis


def _extract_biosample_coordinates_and_date(
    full_biosample: Dict[str, Any],
) -> tuple[
    Optional[float], Optional[float], Optional[str], Optional[str], Optional[str]
]:
    """Extract coordinates, location name, and date from biosample data."""
    # Extract coordinates
    lat_lon = full_biosample.get("lat_lon", {})
    asserted_lat = lat_lon.get("latitude") if isinstance(lat_lon, dict) else None
    asserted_lon = lat_lon.get("longitude") if isinstance(lat_lon, dict) else None

    # Extract geo_loc_name for geocoding
    geo_loc_name = full_biosample.get("geo_loc_name", {})
    location_name = (
        geo_loc_name.get("has_raw_value") if isinstance(geo_loc_name, dict) else None
    )

    # Extract collection date
    collection_date = full_biosample.get("collection_date", {})
    date = (
        collection_date.get("has_raw_value")
        if isinstance(collection_date, dict)
        else None
    )
    parsed_date = parse_collection_date(date) if date else None

    return asserted_lat, asserted_lon, location_name, date, parsed_date


def _analyze_asserted_coordinates(
    asserted_lat: float,
    asserted_lon: float,
    date: Optional[str],
    parsed_date: Optional[str],
    search_radius: int,
) -> Dict[str, Any]:
    """Perform all analyses using asserted coordinates."""
    inferred = {}

    # Soil analysis
    inferred["soil_from_asserted_coords"] = get_soil_analysis(
        asserted_lat, asserted_lon
    )

    # Land cover analysis
    if date:
        inferred["land_cover_from_asserted_coords"] = get_land_cover_analysis(
            asserted_lat, asserted_lon, date
        )

        # Available landuse dates
        if parsed_date:
            landuse_dates_info = _get_landuse_dates_info(
                asserted_lat, asserted_lon, parsed_date
            )
            if landuse_dates_info:
                inferred["landuse_dates_from_asserted_coords"] = landuse_dates_info

        # Weather analysis
        weather = get_weather_analysis(asserted_lat, asserted_lon, date)
        if weather:
            inferred["weather_from_asserted_coords"] = weather

    # Geospatial analysis
    geospatial_asserted = get_geospatial_analysis(
        asserted_lat, asserted_lon, radius=search_radius
    )
    inferred["geospatial_from_asserted_coords"] = geospatial_asserted

    return inferred


def _get_landuse_dates_info(
    lat: float, lon: float, parsed_date: str
) -> Optional[Dict[str, Any]]:
    """Get landuse dates information with neighboring dates."""
    try:
        dates_key = cache_key({"lat": round(lat, 4), "lon": round(lon, 4)})
        cached_dates = get_cache("landuse_dates", dates_key)
        if not cached_dates:
            cached_dates = get_landuse_dates(lat=lat, lon=lon)
            save_cache("landuse_dates", dates_key, cached_dates)

        if cached_dates and isinstance(cached_dates, list):
            closest_date = find_closest_landuse_date(parsed_date, cached_dates)
            if closest_date:
                try:
                    idx = cached_dates.index(closest_date)
                    neighbor_dates = []
                    if idx > 0:
                        neighbor_dates.append(cached_dates[idx - 1])
                    neighbor_dates.append(closest_date)
                    if idx < len(cached_dates) - 1:
                        neighbor_dates.append(cached_dates[idx + 1])

                    return {
                        "target_collection_date": parsed_date,
                        "closest_available_date": closest_date,
                        "neighboring_dates": neighbor_dates,
                    }
                except ValueError:
                    pass
    except Exception:
        pass
    return None


def _process_geocoded_coordinates(location_name: str) -> Optional[Dict[str, Any]]:
    """Process geocoded coordinates from location name."""
    geocode_result = geocode_location_name(location_name.strip())

    if geocode_result.get("coordinates") and not geocode_result.get("error"):
        geocoded_coords = geocode_result["coordinates"]
        geocoded_lat = geocoded_coords["latitude"]
        geocoded_lon = geocoded_coords["longitude"]

        elevation_geocoded = get_elevation(geocoded_lat, geocoded_lon)

        coord_source = {
            "coordinates": geocoded_coords,
            "source": f"geocoded from geo_loc_name: {location_name}",
            "geocode_result": geocode_result,
            "map_url": f"https://www.google.com/maps/place/{geocoded_lat},{geocoded_lon}/@{geocoded_lat},{geocoded_lon},15z",
        }

        if elevation_geocoded is not None:
            coord_source["elevation_meters"] = elevation_geocoded

        return coord_source

    return None


def _create_coordinate_sources(
    asserted_lat: Optional[float],
    asserted_lon: Optional[float],
    location_name: Optional[str],
    full_biosample: Dict[str, Any],
    inferred: Dict[str, Any],
) -> Dict[str, Any]:
    """Create comprehensive coordinate sources information."""
    coord_sources: Dict[str, Any] = {}

    # Process asserted coordinates
    if asserted_lat is not None and asserted_lon is not None:
        coord_sources["from_asserted_coords"] = {
            "coordinates": {"latitude": asserted_lat, "longitude": asserted_lon},
            "source": "lat_lon field",
            "map_url": f"https://www.google.com/maps/place/{asserted_lat},{asserted_lon}/@{asserted_lat},{asserted_lon},15z",
        }

    # Process geocoded coordinates
    if location_name and location_name.strip():
        geocoded_source = _process_geocoded_coordinates(location_name)
        if geocoded_source:
            coord_sources["from_geo_loc_name"] = geocoded_source

    # Add combined analysis if we have both coordinate sources
    if "from_asserted_coords" in coord_sources and "from_geo_loc_name" in coord_sources:
        _add_combined_coordinate_analysis(coord_sources)

    # Add elevation comparison if available
    _add_elevation_comparison(coord_sources, full_biosample, inferred)

    return coord_sources


def _add_combined_coordinate_analysis(coord_sources: Dict[str, Any]) -> None:
    """Add combined map URL and distance calculations."""
    asserted_coords = coord_sources["from_asserted_coords"]["coordinates"]
    geocoded_coords = coord_sources["from_geo_loc_name"]["coordinates"]

    asserted_lat, asserted_lon = (
        asserted_coords["latitude"],
        asserted_coords["longitude"],
    )
    geocoded_lat, geocoded_lon = (
        geocoded_coords["latitude"],
        geocoded_coords["longitude"],
    )

    distance_meters = calculate_distance(
        asserted_lat, asserted_lon, geocoded_lat, geocoded_lon
    )
    distance_km = round(distance_meters / 1000, 2)

    combined_map_url = (
        f"https://www.google.com/maps/dir/{asserted_lat},{asserted_lon}/"
        f"{geocoded_lat},{geocoded_lon}/@{asserted_lat},{asserted_lon},10z"
    )

    coord_sources["combined_map_url"] = combined_map_url
    coord_sources["combined_map_description"] = (
        f"Asserted coords ({asserted_lat}, {asserted_lon}) vs "
        f"Geocoded coords ({geocoded_lat}, {geocoded_lon})"
    )
    coord_sources["coordinate_distance"] = {
        "meters": round(distance_meters, 1),
        "kilometers": distance_km,
        "description": f"Distance between asserted and geocoded coordinates: {distance_km} km",
    }


def _add_elevation_comparison(
    coord_sources: Dict[str, Any],
    full_biosample: Dict[str, Any],
    inferred: Dict[str, Any],
) -> None:
    """Add elevation comparison if data is available."""
    asserted_elevation = full_biosample.get("elev")
    if (
        asserted_elevation is not None
        and "geospatial_from_asserted_coords" in inferred
        and inferred["geospatial_from_asserted_coords"].get("elevation")
    ):
        geospatial_elevation = inferred["geospatial_from_asserted_coords"]["elevation"][
            "meters"
        ]
        elevation_diff = round(abs(asserted_elevation - geospatial_elevation), 1)

        coord_sources["elevation_comparison"] = {
            "asserted_elevation_m": asserted_elevation,
            "geospatial_elevation_m": geospatial_elevation,
            "difference_m": elevation_diff,
            "description": f"Elevation difference: {elevation_diff}m between asserted ({asserted_elevation}m) and geospatial lookup ({geospatial_elevation}m)",
        }


def _process_publication_analysis(
    biosample_id: str, email: str, inferred: Dict[str, Any]
) -> None:
    """Process publication analysis and add to inferred data."""
    publications = get_publication_analysis(biosample_id, email)
    if publications:
        # Extract study_info and all_dois to be siblings
        study_info = publications.pop("study_info", None)
        all_dois = publications.pop("_all_dois", None)

        if study_info:
            inferred["study_info"] = study_info
        if all_dois:
            inferred["all_dois"] = all_dois
        inferred["publication_analysis"] = publications


def analyze_biosample(
    biosample_id: str, email: str, search_radius: int = 1000
) -> Optional[Dict[str, Any]]:
    """Perform comprehensive analysis of a biosample."""
    try:
        # Get complete biosample data
        full_biosample = cached_fetch_nmdc_entity_by_id(biosample_id)
        if not full_biosample:
            return None

        # Extract coordinates and date information
        asserted_lat, asserted_lon, location_name, date, parsed_date = (
            _extract_biosample_coordinates_and_date(full_biosample)
        )

        inferred = {}

        # Analyze using asserted coordinates
        if asserted_lat is not None and asserted_lon is not None:
            asserted_analyses = _analyze_asserted_coordinates(
                asserted_lat, asserted_lon, date, parsed_date, search_radius
            )
            inferred.update(asserted_analyses)

        # Create coordinate sources summary
        coord_sources = _create_coordinate_sources(
            asserted_lat, asserted_lon, location_name, full_biosample, inferred
        )
        if coord_sources:
            inferred["coordinate_sources"] = coord_sources

        # Publication analysis
        _process_publication_analysis(biosample_id, email, inferred)

        return {"asserted": full_biosample, "inferred": inferred}

    except Exception:
        return None

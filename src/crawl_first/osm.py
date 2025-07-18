"""
OpenStreetMap utilities for crawl-first.

Handles OSM feature extraction and environmental analysis.
"""

from dataclasses import dataclass
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import requests

from .cache import cache_key, get_cache, save_cache
from .geospatial import calculate_distance

# Environmental feature types for OSM
OSM_ENVIRONMENTAL_TAGS = {
    "natural": {
        # Water features
        "water",
        "coastline",
        "bay",
        "cape",
        "peninsula",
        "strait",
        "fjord",
        "spring",
        "hot_spring",
        "geyser",
        "mineral_spring",
        # Vegetation
        "wood",
        "tree",
        "tree_row",
        "grassland",
        "scrub",
        "heath",
        "moor",
        "fell",
        "tundra",
        "shrubbery",
        # Wetlands
        "wetland",
        "marsh",
        "swamp",
        "bog",
        "fen",
        "wet_meadow",
        "reedbed",
        "saltmarsh",
        # Geological features
        "beach",
        "sand",
        "shingle",
        "bare_rock",
        "scree",
        "cliff",
        "rock",
        "stone",
        "boulder",
        "peak",
        "ridge",
        "valley",
        "saddle",
        "col",
        "spur",
        "arete",
        "cave_entrance",
        "cave",
        "sinkhole",
        "karst",
        # Volcanic/thermal
        "volcano",
        "crater",
        "lava_field",
        "fumarole",
        "mud",
        # Ice/snow
        "glacier",
        "snowfield",
        "ice_shelf",
        # Desert features
        "desert",
        "dune",
        "oasis",
    },
    "landuse": {
        # Agricultural
        "forest",
        "farmland",
        "meadow",
        "orchard",
        "vineyard",
        "agricultural",
        "grass",
        "pasture",
        "allotments",
        "plant_nursery",
        # Conservation
        "nature_reserve",
        "conservation",
        "greenfield",
        # Water management
        "wetland",
        "water",
        "reservoir",
        "salt_pond",
        "aquaculture",
        "basin",
        # Recreation
        "recreation_ground",
        "village_green",
        "cemetery",
        # Industrial/extractive
        "quarry",
        "landfill",
        "brownfield",
        "industrial",
        "commercial",
        # Waste management
        "dump",
        "waste_disposal",
        "scrap_yard",
        "salvage_yard",
        # Other
        "floodplain",
    },
    "waterway": {
        "river",
        "stream",
        "canal",
        "drain",
        "ditch",
        "brook",
        "creek",
        "rapids",
        "waterfall",
        "dam",
        "weir",
        "lock_gate",
        "dock",
        "tidal_channel",
        "intermittent",
    },
    "water": {
        "lake",
        "pond",
        "reservoir",
        "river",
        "stream",
        "canal",
        "lagoon",
        "bay",
        "wetland",
        "marsh",
        "swamp",
        "salt_water",
        "fresh_water",
        "oxbow",
        "reflecting_pool",
        "wastewater",
        "intermittent",
    },
    "wetland": {
        "marsh",
        "swamp",
        "bog",
        "fen",
        "wet_meadow",
        "reedbed",
        "saltmarsh",
        "tidalflat",
        "saltern",
        "mangrove",
    },
    "leisure": {
        "nature_reserve",
        "park",
        "garden",
        "common",
        "recreation_ground",
        "dog_park",
        "picnic_site",
        "beach_resort",
        "fishing",
    },
    "boundary": {
        "national_park",
        "protected_area",
        "nature_reserve",
        "marine_protected_area",
        "aboriginal_lands",
        "indigenous_territory",
    },
    "place": {"island", "islet", "archipelago", "atoll", "reef"},
    "geological": {"outcrop", "moraine", "ridge", "valley"},
    "man_made": {
        # Water infrastructure
        "pier",
        "breakwater",
        "groyne",
        "dyke",
        "embankment",
        "dam",
        "reservoir_covered",
        "water_tower",
        "water_well",
        "wastewater_plant",
        # Monitoring/research
        "monitoring_station",
        "weather_station",
        "research_station",
        # Waste management
        "waste_disposal",
        "wastewater_plant",
        "recycling",
    },
    "highway": {
        "track",
        "path",
        "footway",
        "bridleway",  # Can indicate human impact/access
    },
    "barrier": {"fence", "wall", "hedge"},  # Can indicate land use boundaries
    "building": {
        # Agricultural
        "barn",
        "farm",
        "farm_auxiliary",
        "greenhouse",
        "silo",
        "stable",
        "cowshed",
        "chicken_coop",
        "livestock",
        "slurry_tank",
        # Research/Educational
        "university",
        "college",
        "research",
        "laboratory",
        "observatory",
        # Industrial/Processing
        "industrial",
        "warehouse",
        "factory",
        "refinery",
        "power_plant",
        "water_treatment",
        "wastewater_treatment",
        "sewage_treatment",
        # Mining/Extractive
        "mine",
        "quarry_office",
        "mining",
        # Storage/Hazardous
        "fuel_storage",
        "storage_tank",
        "hazmat",
        "chemical_storage",
        # Utilities
        "pumping_station",
        "substation",
        "utility",
    },
    "amenity": {
        "research_station",
        "waste_disposal",
        "recycling",
        "waste_transfer_station",
        # Water/waste management
        "waste_basket",
        "toilets",
        "drinking_water",
        "water_point",
        "wastewater_plant",
        "fuel",
        "charging_station",
        # Research/education
        "university",
        "college",
        "research_institute",
        "library",
        # Agriculture related
        "animal_shelter",
        "veterinary",
        # Additional waste facilities
        "dump",
        "landfill",
    },
    "industrial": {
        "factory",
        "refinery",
        "mine",
        "quarry",
        "oil_well",
        "gas_well",
        "chemical",
        "petroleum",
        "wastewater_treatment",
        "water_treatment",
        "power",
        "solar_park",
        "wind_farm",
    },
    "power": {
        "plant",
        "generator",
        "substation",
        "transformer",
        "solar_panel",
        "wind_turbine",
        "geothermal",
    },
}


@dataclass
class OSMFeature:
    """
    Represents a single OpenStreetMap (OSM) feature.

    This class encapsulates the properties of an OSM feature, including its
    unique identifier, type, tags, geometry, and spatial information.

    Coordinate System:
        All coordinates use the WGS84 coordinate system (EPSG:4326), which is
        the standard used by OpenStreetMap and GPS systems. Coordinates are
        expressed in decimal degrees.

    Distance Calculations:
        Distances are calculated using the great circle distance formula
        (haversine formula) on the WGS84 ellipsoid.

    Attributes:
        feature_id (str): The unique identifier of the OSM feature. This is
            typically a numeric ID assigned by OpenStreetMap.
        feature_type (str): The type of the feature in "category:value" format
            (e.g., "natural:water", "landuse:forest"). This represents the
            primary classification of the feature.
        tags (Dict[str, str]): Key-value pairs representing OSM tags that
            describe the feature's properties. These include all OSM tags
            such as name, description, and specific attributes.
        geometry_type (str): The type of geometry from OSM data model
            (e.g., "node" for points, "way" for lines/polygons, "relation"
            for complex geometries).
        coordinates (Tuple[float, float]): The latitude and longitude of the
            feature's location in decimal degrees (WGS84). For non-point
            geometries, this represents the centroid or representative point.
            Format: (latitude, longitude) where latitude ranges from -90 to 90
            and longitude ranges from -180 to 180.
        distance_from_center (float): The great circle distance of the feature
            from a reference center point, measured in meters. Calculated using
            the haversine formula for accurate distance on Earth's surface.
        area (Optional[float]): The area of the feature in square meters, if
            applicable and available from OSM data. Only present for polygon
            features (ways and relations). Defaults to None for point features
            or when area data is not available.
    """

    feature_id: str
    feature_type: str
    tags: Dict[str, str]
    geometry_type: str
    coordinates: Tuple[float, float]
    distance_from_center: float
    area: Optional[float] = None


def build_overpass_query(lat: float, lon: float, radius: float = 1000) -> str:
    """Build Overpass API query for environmental features."""
    tag_queries = []
    for key, values in OSM_ENVIRONMENTAL_TAGS.items():
        for value in values:
            tag_queries.append(f'nwr["{key}"="{value}"](around:{radius},{lat},{lon});')

    return f"""
    [out:json][timeout:120];
    ({' '.join(tag_queries)});
    out body center qt;
    """


def get_feature_type(tags: Dict[str, str]) -> Optional[str]:
    """Determine primary feature type from OSM tags."""
    # Check in priority order - more specific environmental features first
    categories = [
        "natural",
        "water",
        "wetland",
        "waterway",
        "geological",
        "landuse",
        "leisure",
        "boundary",
        "place",
        "man_made",
        "building",
        "industrial",
        "power",
        "amenity",
        "highway",
        "barrier",
    ]

    for category in categories:
        if category in tags and tags[category] in OSM_ENVIRONMENTAL_TAGS.get(
            category, set()
        ):
            return f"{category}:{tags[category]}"
    return None


def extract_coordinates(element: Dict) -> Tuple[float, float]:
    """Extract coordinates from OSM element."""
    if "center" in element:
        return (element["center"]["lat"], element["center"]["lon"])
    elif "lat" in element and "lon" in element:
        return (element["lat"], element["lon"])
    else:
        raise ValueError("No coordinates found in element")


def query_osm_features(
    lat: float, lon: float, radius: float = 1000
) -> List[OSMFeature]:
    """Query OpenStreetMap features around a point using Overpass API."""
    key = cache_key({"lat": round(lat, 4), "lon": round(lon, 4), "radius": radius})
    cached = get_cache("osm_features", key)
    if cached:
        features = []
        for feature_data in cached.get("features", []):
            feature = OSMFeature(
                feature_id=feature_data["feature_id"],
                feature_type=feature_data["feature_type"],
                tags=feature_data["tags"],
                geometry_type=feature_data["geometry_type"],
                coordinates=tuple(feature_data["coordinates"]),
                distance_from_center=feature_data["distance_from_center"],
                area=feature_data.get("area"),
            )
            features.append(feature)
        return features

    query = build_overpass_query(lat, lon, radius)
    url = "https://overpass-api.de/api/interpreter"

    for attempt in range(3):
        try:
            response = requests.post(url, data={"data": query}, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "elements" not in data:
                return []

            features = []
            for element in data["elements"]:
                try:
                    if "tags" not in element:
                        continue

                    feature_type = get_feature_type(element["tags"])
                    if not feature_type:
                        continue

                    coordinates = extract_coordinates(element)
                    distance = calculate_distance(
                        lat, lon, coordinates[0], coordinates[1]
                    )

                    if distance > radius:
                        continue

                    area = None
                    if "area" in element:
                        try:
                            area = float(element["area"])
                        except (ValueError, TypeError):
                            pass

                    feature = OSMFeature(
                        feature_id=str(element["id"]),
                        feature_type=feature_type,
                        tags=element["tags"],
                        geometry_type=element["type"],
                        coordinates=coordinates,
                        distance_from_center=distance,
                        area=area,
                    )
                    features.append(feature)

                except Exception:
                    continue

            cache_data = {
                "features": [
                    {
                        "feature_id": f.feature_id,
                        "feature_type": f.feature_type,
                        "tags": f.tags,
                        "geometry_type": f.geometry_type,
                        "coordinates": f.coordinates,
                        "distance_from_center": f.distance_from_center,
                        "area": f.area,
                    }
                    for f in features
                ]
            }
            save_cache("osm_features", key, cache_data)
            return features

        except requests.exceptions.RequestException:
            if attempt < 2:
                sleep(5)
                continue
            else:
                return []
        except Exception:
            # Catch any other exceptions and return empty list
            return []

    # Fallback return (should never be reached)
    return []


def summarize_osm_features(
    features: List[OSMFeature], center_lat: float, center_lon: float
) -> Dict[str, Any]:
    """Summarize OSM features by category with environmental focus."""
    summary: Dict[str, Any] = {
        "metadata": {
            "total_features": len(features),
            "query_coordinates": [center_lat, center_lon],
            "feature_counts": {},
        },
        "environmental_features": {},
        "nearest_features": [],
    }

    # Type-safe accessors to help mypy
    nearest_features_list = summary["nearest_features"]
    assert isinstance(nearest_features_list, list)
    environmental_features_dict = summary["environmental_features"]
    assert isinstance(environmental_features_dict, dict)
    metadata_dict = summary["metadata"]
    assert isinstance(metadata_dict, dict)
    feature_counts_dict = metadata_dict["feature_counts"]
    assert isinstance(feature_counts_dict, dict)

    # Temporary lists for counting (not included in final output)
    water_features_temp = []
    natural_areas_temp = []
    protected_areas_temp = []

    sorted_features = sorted(features, key=lambda x: x.distance_from_center)

    # Get nearest 5 features regardless of type
    for f in sorted_features[:5]:
        feature_info = {
            "type": f.feature_type,
            "distance_m": round(f.distance_from_center, 1),
        }
        # Only add name if it exists and isn't "Unnamed"
        name = f.tags.get("name")
        if name and name != "Unnamed":
            feature_info["name"] = name
        nearest_features_list.append(feature_info)

    for feature in features:
        category = feature.feature_type.split(":")[0]
        feature_type = feature.feature_type.split(":")[1]

        if category not in feature_counts_dict:
            feature_counts_dict[category] = {}
        if feature_type not in feature_counts_dict[category]:
            feature_counts_dict[category][feature_type] = 0
        feature_counts_dict[category][feature_type] += 1

        # Only include features that have names (skip unnamed features)
        name = feature.tags.get("name")
        if name and name != "Unnamed":
            if category not in environmental_features_dict:
                environmental_features_dict[category] = []

            feature_info = {
                "type": feature_type,
                "distance_m": round(feature.distance_from_center, 1),
                "name": name,
            }

            # Add relevant tags (excluding name since we handle it separately)
            relevant_tags = {
                k: v
                for k, v in feature.tags.items()
                if k in ["description", "wikipedia"] and v
            }
            if relevant_tags:
                feature_info["tags"] = relevant_tags

            if feature.area:
                feature_info["area_m2"] = feature.area

            environmental_features_dict[category].append(feature_info)

        # Count ALL features for environmental summary (including unnamed ones)
        if category in ["water", "waterway"] or feature_type in [
            "water",
            "lake",
            "pond",
            "river",
            "stream",
        ]:
            water_features_temp.append(1)  # Just count, don't store data

        if category == "natural" or feature_type in [
            "forest",
            "wood",
            "grassland",
            "wetland",
        ]:
            natural_areas_temp.append(1)  # Just count, don't store data

        if feature_type in [
            "nature_reserve",
            "national_park",
            "protected_area",
            "conservation",
        ]:
            protected_areas_temp.append(1)  # Just count, don't store data

    # Sort and deduplicate - keep only closest instance of each named feature
    for category in environmental_features_dict:
        # Sort by distance first
        category_features = environmental_features_dict[category]
        assert isinstance(category_features, list)  # Help mypy
        category_features.sort(key=lambda x: x["distance_m"])

        # Deduplicate by name - keep only the closest (first) occurrence
        seen_names = set()
        unique_features = []
        for feature in category_features:
            name = feature["name"]
            if name not in seen_names:
                seen_names.add(name)
                unique_features.append(feature)

        environmental_features_dict[category] = unique_features

    return summary

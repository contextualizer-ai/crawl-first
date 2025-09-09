"""
Comprehensive OSM enrichment with 'ask for everything' approach.

Single broad Overpass query to get all features within radius, then split into
named features vs unnamed counts with detailed geometry and distance tracking.
"""

import math
import requests
from typing import Dict, Any, List, Optional, Tuple


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


def distance_point_to_linestring_km(point_lat: float, point_lon: float, linestring: List[Tuple[float, float]]) -> float:
    """Calculate minimum distance from point to linestring in kilometers."""
    if not linestring:
        return float('inf')
    
    min_distance = float('inf')
    
    for i in range(len(linestring) - 1):
        # Get line segment
        lat1, lon1 = linestring[i]
        lat2, lon2 = linestring[i + 1]
        
        # Distance to endpoints
        dist1 = haversine_km(point_lat, point_lon, lat1, lon1)
        dist2 = haversine_km(point_lat, point_lon, lat2, lon2)
        min_distance = min(min_distance, dist1, dist2)
        
        # Distance to line segment (simplified - not exact but reasonable approximation)
        # For exact calculation would need to project point onto line segment
        
    return min_distance


def distance_point_to_polygon_km(point_lat: float, point_lon: float, polygon: List[Tuple[float, float]]) -> float:
    """Calculate distance from point to polygon in kilometers (0 if within)."""
    if not polygon:
        return float('inf')
    
    # Simple point-in-polygon check using ray casting
    x, y = point_lon, point_lat
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    if inside:
        return 0.0
    
    # If outside, calculate distance to edge
    return distance_point_to_linestring_km(point_lat, point_lon, polygon)


def get_element_centroid(element: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Extract centroid coordinates from OSM element."""
    if element.get("type") == "node":
        return element.get("lat"), element.get("lon")
    elif "center" in element:
        return element["center"].get("lat"), element["center"].get("lon")
    elif element.get("type") == "way" and "geometry" in element:
        # Calculate centroid of way
        coords = [(node["lat"], node["lon"]) for node in element["geometry"]]
        if coords:
            avg_lat = sum(lat for lat, lon in coords) / len(coords)
            avg_lon = sum(lon for lat, lon in coords) / len(coords)
            return avg_lat, avg_lon
    elif element.get("type") == "relation" and "members" in element:
        # For relations, try to get centroid from member geometry
        all_coords = []
        for member in element["members"]:
            if "geometry" in member:
                for node in member["geometry"]:
                    if "lat" in node and "lon" in node:
                        all_coords.append((node["lat"], node["lon"]))
        
        if all_coords:
            avg_lat = sum(lat for lat, lon in all_coords) / len(all_coords)
            avg_lon = sum(lon for lat, lon in all_coords) / len(all_coords)
            return avg_lat, avg_lon
    return None, None


def calculate_feature_distance(sample_lat: float, sample_lon: float, element: Dict[str, Any]) -> float:
    """Calculate distance from sample point to OSM feature geometry."""
    elem_type = element.get("type")
    
    if elem_type == "node":
        lat = element.get("lat")
        lon = element.get("lon")
        if lat is not None and lon is not None:
            return haversine_km(sample_lat, sample_lon, lat, lon)
    
    elif elem_type == "way" and "geometry" in element:
        coords = [(node["lat"], node["lon"]) for node in element["geometry"]]
        if coords:
            tags = element.get("tags", {})
            
            # Check if this is a closed way (polygon)
            if (len(coords) > 3 and coords[0] == coords[-1]) or tags.get("area") == "yes":
                return distance_point_to_polygon_km(sample_lat, sample_lon, coords)
            else:
                return distance_point_to_linestring_km(sample_lat, sample_lon, coords)
    
    # Fallback to centroid distance
    centroid_lat, centroid_lon = get_element_centroid(element)
    if centroid_lat is not None and centroid_lon is not None:
        return haversine_km(sample_lat, sample_lon, centroid_lat, centroid_lon)
    
    return float('inf')


def is_named_feature(tags: Dict[str, str]) -> bool:
    """Check if feature has naming tags."""
    name_tags = ["name", "official_name", "alt_name", "wikidata", "wikipedia", "short_name"]
    return any(tag in tags for tag in name_tags)


def enrich_osm_comprehensive(lat: float, lon: float, radius_m: int = 1000, timeout: int = 180) -> Dict[str, Any]:
    """
    Comprehensive OSM enrichment with single broad query.
    
    Args:
        lat: Latitude
        lon: Longitude
        radius_m: Search radius in meters
        timeout: Query timeout in seconds
        
    Returns:
        Dict with named features and unnamed counts following the schema
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Single comprehensive query for all features
    query = f"""
    [out:json][timeout:{timeout}];
    (
      node(around:{radius_m},{lat},{lon});
      way(around:{radius_m},{lat},{lon});
      relation(around:{radius_m},{lat},{lon});
    );
    out body geom qt;
    """
    
    try:
        response = requests.post(overpass_url, data=query, timeout=timeout + 10)
        response.raise_for_status()
        data = response.json()
        
        elements = data.get("elements", [])
        
        named_features = []
        unnamed_counts = {}
        
        # Process each element
        for element in elements:
            tags = element.get("tags", {})
            if not tags:  # Skip elements without tags
                continue
                
            osm_type = element.get("type")
            osm_id = element.get("id")
            
            # Calculate distance to feature
            distance_km = calculate_feature_distance(lat, lon, element)
            
            # Get centroid for named features
            centroid_lat, centroid_lon = get_element_centroid(element)
            
            if is_named_feature(tags):
                # Named feature - store complete information
                feature_data = {
                    "osm_type": osm_type,
                    "osm_id": osm_id,
                    "name": tags.get("name") or tags.get("official_name") or tags.get("alt_name"),
                    "tags": tags,
                    "centroid": {"lat": centroid_lat, "lon": centroid_lon} if centroid_lat else None,
                    "distance_km": round(distance_km, 3) if distance_km != float('inf') else None
                }
                
                # Add special distance calculation for water/transport features
                main_category = None
                for key in ["natural", "waterway", "highway", "railway", "aeroway", "amenity"]:
                    if key in tags:
                        main_category = key
                        break
                
                if main_category in ["natural", "waterway", "highway", "railway"]:
                    feature_data["category"] = main_category
                    feature_data["subcategory"] = tags.get(main_category)
                
                named_features.append(feature_data)
                
            else:
                # Unnamed feature - add to counts
                for key, value in tags.items():
                    if key not in unnamed_counts:
                        unnamed_counts[key] = {"_total": 0}
                    
                    unnamed_counts[key]["_total"] += 1
                    
                    if value not in unnamed_counts[key]:
                        unnamed_counts[key][value] = {"node": 0, "way": 0, "relation": 0}
                    
                    unnamed_counts[key][value][osm_type] += 1
        
        # Sort named features by distance (handle None values)
        named_features.sort(key=lambda x: x.get("distance_km") if x.get("distance_km") is not None else float('inf'))
        
        return {
            "query": {
                "radius_m": radius_m,
                "timeout_s": timeout,
                "center": {"lat": lat, "lon": lon}
            },
            "named_features": named_features,
            "unnamed_counts": unnamed_counts,
            "summary": {
                "total_elements": len(elements),
                "named_features": len(named_features),
                "unnamed_categories": len(unnamed_counts),
                "total_unnamed": sum(cat.get("_total", 0) for cat in unnamed_counts.values())
            },
            "provenance": {
                "data_source": "OpenStreetMap Overpass API",
                "query_timestamp": response.headers.get("Date"),
                "query_url": overpass_url,
                "method": "comprehensive_single_query"
            }
        }
        
    except requests.Timeout:
        return {
            "query": {"radius_m": radius_m, "timeout_s": timeout},
            "error": f"Query timeout after {timeout}s",
            "success": False,
            "provenance": {"data_source": "OpenStreetMap Overpass API", "method": "comprehensive_single_query"}
        }
    except Exception as e:
        return {
            "query": {"radius_m": radius_m},
            "error": str(e),
            "success": False,
            "provenance": {"data_source": "OpenStreetMap Overpass API", "method": "comprehensive_single_query"}
        }


def get_nearby_feature_summary(osm_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics from OSM enrichment results.
    
    Args:
        osm_result: Result from enrich_osm_comprehensive
        
    Returns:
        Dict with distance summaries for key feature types
    """
    if "named_features" not in osm_result:
        return {"error": "Invalid OSM result"}
    
    # Categorize distances for key feature types
    distances = {
        "water_features": [],
        "transport_features": [],
        "natural_features": [],
        "amenities": [],
        "buildings": []
    }
    
    for feature in osm_result["named_features"]:
        dist = feature.get("distance_km")
        if dist is None:
            continue
            
        tags = feature.get("tags", {})
        
        # Categorize feature (water features take priority over general natural)
        if ("natural" in tags and tags["natural"] in ["water", "lake", "pond", "river", "stream"]) or "waterway" in tags:
            distances["water_features"].append(dist)
        elif "highway" in tags or "railway" in tags or "aeroway" in tags:
            distances["transport_features"].append(dist)
        elif "natural" in tags and tags["natural"] not in ["water", "lake", "pond", "river", "stream"]:
            # Non-water natural features (forest, grassland, geyser, etc.)
            distances["natural_features"].append(dist)
        elif "amenity" in tags:
            distances["amenities"].append(dist)
        elif "building" in tags:
            distances["buildings"].append(dist)
    
    # Calculate summary statistics
    summary = {}
    for category, dist_list in distances.items():
        if dist_list:
            summary[f"nearest_{category}_km"] = round(min(dist_list), 3)
            summary[f"avg_{category}_km"] = round(sum(dist_list) / len(dist_list), 3)
            summary[f"{category}_within_1km"] = len([d for d in dist_list if d <= 1.0])
        else:
            summary[f"nearest_{category}_km"] = None
            summary[f"{category}_within_1km"] = 0
    
    # Add information about unnamed natural features within query radius
    if "unnamed_counts" in osm_result:
        natural_counts = osm_result["unnamed_counts"].get("natural", {})
        total_natural_unnamed = 0
        natural_types = []
        
        for nat_type, counts in natural_counts.items():
            if nat_type != "_total" and nat_type not in ["water", "lake", "pond", "river", "stream"]:
                total = counts.get("node", 0) + counts.get("way", 0) + counts.get("relation", 0)
                if total > 0:
                    total_natural_unnamed += total
                    natural_types.append(f"{nat_type}({total})")
        
        if total_natural_unnamed > 0:
            summary["unnamed_natural_features_in_radius"] = total_natural_unnamed
            summary["unnamed_natural_types"] = natural_types[:5]  # Top 5 types
    
    return summary


def batch_enrich_osm(locations: List[Tuple[float, float]], radius_m: int = 1000) -> List[Dict[str, Any]]:
    """
    Process multiple locations with OSM enrichment.
    
    Args:
        locations: List of (lat, lon) tuples
        radius_m: Search radius in meters
        
    Returns:
        List of OSM enrichment results
    """
    results = []
    
    for lat, lon in locations:
        result = enrich_osm_comprehensive(lat, lon, radius_m)
        results.append(result)
    
    return results
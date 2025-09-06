"""
Unified geospatial enrichment interface.

Bridge between crawl-first and gold-and-nmdc unified enrichment capabilities.
Provides a clean API for comprehensive geospatial analysis.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add gold-and-nmdc scripts to path for import
repo_root = Path(__file__).parent.parent.parent
gold_nmdc_scripts = repo_root / "gold-and-nmdc" / "scripts"
sys.path.insert(0, str(gold_nmdc_scripts))

try:
    from geospatial_enrichment import enrich_location_unified
    from coordinate_utils import haversine_distance, validate_coordinates
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False


def enrich_coordinates_unified(
    lat: float,
    lon: float,
    date: Optional[str] = None,
    enable_crosswalks: bool = True,
    search_radius_km: float = 1.0
) -> Dict[str, Any]:
    """
    Comprehensive coordinate enrichment using unified pipeline.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees  
        date: Collection date (YYYY-MM-DD format), optional
        enable_crosswalks: Whether to enable ENVO ontology crosswalks
        search_radius_km: Search radius for nearby features in km
        
    Returns:
        Enriched coordinate data with all available environmental context
        
    Raises:
        ImportError: If unified enrichment is not available
        ValueError: If coordinates are invalid
    """
    if not UNIFIED_AVAILABLE:
        raise ImportError(
            "Unified enrichment not available. Ensure gold-and-nmdc scripts are accessible."
        )
    
    if not validate_coordinates(lat, lon):
        raise ValueError(f"Invalid coordinates: {lat}, {lon}")
    
    # Use unified enrichment from gold-and-nmdc
    result = enrich_location_unified(
        lat=lat,
        lon=lon,
        date=date,
        enable_crosswalks=enable_crosswalks,
        search_radius_km=search_radius_km
    )
    
    return result


def get_coordinate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two coordinate pairs.
    
    Args:
        lat1, lon1: First coordinate pair
        lat2, lon2: Second coordinate pair
        
    Returns:
        Distance in kilometers
        
    Raises:
        ImportError: If coordinate utilities are not available
    """
    if not UNIFIED_AVAILABLE:
        raise ImportError(
            "Coordinate utilities not available. Ensure gold-and-nmdc scripts are accessible."
        )
    
    return haversine_distance(lat1, lon1, lat2, lon2)


def is_unified_enrichment_available() -> bool:
    """Check if unified enrichment capabilities are available."""
    return UNIFIED_AVAILABLE


def get_unified_enrichment_info() -> Dict[str, Any]:
    """Get information about unified enrichment capabilities."""
    info = {
        "available": UNIFIED_AVAILABLE,
        "gold_nmdc_scripts_path": str(gold_nmdc_scripts),
        "capabilities": []
    }
    
    if UNIFIED_AVAILABLE:
        info["capabilities"] = [
            "Comprehensive geospatial enrichment",
            "Weather data with coordinate distance tracking", 
            "Soil classification and properties",
            "Ecoregion identification",
            "OSM feature extraction",
            "ENVO ontology crosswalks",
            "Coordinate distance calculation",
            "Data quality assessment"
        ]
    
    return info
"""
Local raster sampling system for land cover and soil properties.

Replaces WMS/WCS calls with local COG sampling using rasterio, with buffered
median strategies for soil properties and proper ENVO crosswalk integration.
"""

import math
import json
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import requests


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


def sample_raster(
    lat: float, 
    lon: float, 
    raster_path: Union[str, Path], 
    strategy: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Sample raster at point location with configurable strategy.
    
    Args:
        lat: Latitude
        lon: Longitude
        raster_path: Path to raster file (local or URL)
        strategy: Sampling strategy config
        
    Returns:
        Dict with sampled value, distance, resolution, and metadata
    """
    if strategy is None:
        strategy = {"method": "nearest", "window_px": 1}
    
    try:
        import rasterio
        from rasterio.windows import Window
        from rasterio.transform import rowcol
        
        # Open raster
        with rasterio.open(str(raster_path)) as src:
            # Convert lat/lon to pixel coordinates
            row, col = rowcol(src.transform, lon, lat)
            
            # Check if point is within raster bounds
            if row < 0 or row >= src.height or col < 0 or col >= src.width:
                return {
                    "value": None,
                    "error": "Point outside raster bounds",
                    "resolution": f"{abs(src.transform[0]):.6f}°",
                    "nodata": src.nodata,
                    "source": str(raster_path)
                }
            
            method = strategy.get("method", "nearest")
            window_px = strategy.get("window_px", 1)
            
            if method == "nearest":
                # Single pixel sampling
                value = src.read(1)[row, col]
                distance_m = 0.0
                
            elif method == "bilinear":
                # Bilinear interpolation (simplified - use rasterio's resampling for exact)
                window = Window(col - 0.5, row - 0.5, 1, 1)
                data = src.read(1, window=window, out_shape=(1, 1), resampling=rasterio.enums.Resampling.bilinear)
                value = data[0, 0]
                distance_m = 0.0
                
            elif method == "buffered_median":
                # Buffered median sampling
                half_window = window_px // 2
                window = Window(
                    max(0, col - half_window),
                    max(0, row - half_window), 
                    min(window_px, src.width - max(0, col - half_window)),
                    min(window_px, src.height - max(0, row - half_window))
                )
                
                data = src.read(1, window=window)
                
                # Calculate median of valid (non-nodata) pixels
                valid_data = data[data != src.nodata] if src.nodata is not None else data.flatten()
                if len(valid_data) > 0:
                    value = statistics.median(valid_data)
                    # Distance is approximately half the window size in meters
                    pixel_size_m = abs(src.transform[0]) * 111000  # Rough conversion to meters
                    distance_m = half_window * pixel_size_m
                else:
                    value = src.nodata
                    distance_m = 0.0
                    
            else:
                raise ValueError(f"Unknown sampling method: {method}")
            
            # Check for nodata
            if src.nodata is not None and value == src.nodata:
                value = None
            
            return {
                "value": float(value) if value is not None else None,
                "distance_m": distance_m,
                "resolution": f"{abs(src.transform[0]):.6f}°",
                "pixel_size_m": abs(src.transform[0]) * 111000,  # Rough conversion
                "method": method,
                "window_px": window_px,
                "nodata": src.nodata,
                "source": str(raster_path),
                "crs": str(src.crs),
                "success": True
            }
            
    except ImportError:
        return {
            "value": None,
            "error": "rasterio library not available",
            "source": str(raster_path),
            "success": False
        }
    except Exception as e:
        return {
            "value": None,
            "error": str(e),
            "source": str(raster_path),
            "success": False
        }


def load_envo_crosswalk(crosswalk_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load ENVO crosswalk mappings for land cover classification."""
    if crosswalk_path is None:
        # Default path in repository
        crosswalk_path = Path(__file__).parent.parent.parent / "mappings" / "envo_crosswalk.json"
    
    try:
        if crosswalk_path.exists():
            with open(crosswalk_path) as f:
                return json.load(f)
        else:
            return {}
    except Exception:
        return {}


def get_land_cover_local(lat: float, lon: float, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get land cover from local COG files with WMS fallback.
    
    Args:
        lat: Latitude
        lon: Longitude
        config: Configuration with dataset paths and preferences
        
    Returns:
        Dict with land cover classification and ENVO terms
    """
    if config is None:
        config = {
            "datasets": {
                "worldcover_2021": "/path/to/worldcover_2021.tif",
                "nlcd_2019": "/path/to/nlcd_2019.tif"
            },
            "prefer_local": True,
            "envo_crosswalk_path": None
        }
    
    results = {}
    envo_crosswalk = load_envo_crosswalk(config.get("envo_crosswalk_path"))
    
    # Try local datasets first
    for dataset_name, dataset_path in config.get("datasets", {}).items():
        if not dataset_path or not Path(dataset_path).exists():
            continue
            
        # Determine sampling strategy based on dataset
        if "soil" in dataset_name.lower():
            strategy = {"method": "buffered_median", "window_px": 3}
        else:
            strategy = {"method": "nearest", "window_px": 1}
        
        sample_result = sample_raster(lat, lon, dataset_path, strategy)
        
        if sample_result.get("success") and sample_result.get("value") is not None:
            value = int(sample_result["value"])
            
            # Look up class name and ENVO mapping
            class_name = None
            envo_terms = []
            
            if dataset_name in envo_crosswalk:
                class_info = envo_crosswalk[dataset_name].get(str(value), {})
                class_name = class_info.get("class_name") 
                envo_terms = class_info.get("envo_terms", [])
            
            results[dataset_name] = {
                "code": value,
                "class": class_name or f"Class_{value}",
                "envo": envo_terms,
                "resolution": sample_result.get("resolution"),
                "distance_m": sample_result.get("distance_m", 0),
                "method": sample_result.get("method"),
                "source": "local_cog"
            }
    
    # If no local results and fallback enabled, try WMS
    if not results and not config.get("prefer_local", True):
        # Import existing WMS functions as fallback
        try:
            from .alternative_geospatial import get_land_cover_esa_worldcover, get_land_cover_nlcd
            
            # Try ESA WorldCover via WMS
            wms_result = get_land_cover_esa_worldcover(lat, lon, 2021)
            if wms_result.get("success"):
                results["worldcover_2021_wms"] = {
                    "code": wms_result.get("land_cover_class"),
                    "class": wms_result.get("class_name"),
                    "envo": wms_result.get("envo_terms", []),
                    "source": "wms_fallback",
                    "distance_m": 0,
                    "resolution": "10m"
                }
                
        except ImportError:
            pass
    
    return {
        "land_cover": results,
        "success": len(results) > 0,
        "datasets_attempted": list(config.get("datasets", {}).keys()),
        "provenance": {
            "method": "local_cog_with_wms_fallback",
            "envo_crosswalk": config.get("envo_crosswalk_path") is not None
        }
    }


def get_soil_properties_local(lat: float, lon: float, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get soil properties from local SoilGrids COG files with buffered median sampling.
    
    Args:
        lat: Latitude
        lon: Longitude
        config: Configuration with SoilGrids dataset paths
        
    Returns:
        Dict with soil properties and USDA texture classification
    """
    if config is None:
        config = {
            "soilgrids_datasets": {
                "ph_0_5cm": "/path/to/phh2o_0-5cm_mean.tif",
                "soc_0_5cm": "/path/to/soc_0-5cm_mean.tif", 
                "sand_0_5cm": "/path/to/sand_0-5cm_mean.tif",
                "silt_0_5cm": "/path/to/silt_0-5cm_mean.tif",
                "clay_0_5cm": "/path/to/clay_0-5cm_mean.tif",
                "bdod_0_5cm": "/path/to/bdod_0-5cm_mean.tif",
                "nitrogen_0_5cm": "/path/to/nitrogen_0-5cm_mean.tif"
            },
            "scaling_factors": {
                "ph_0_5cm": 0.1,      # pH×10 → pH
                "soc_0_5cm": 0.1,     # dg/kg → g/kg  
                "sand_0_5cm": 0.1,    # g/kg → %
                "silt_0_5cm": 0.1,    # g/kg → %
                "clay_0_5cm": 0.1,    # g/kg → %
                "bdod_0_5cm": 0.01,   # cg/cm³ → g/cm³
                "nitrogen_0_5cm": 0.01 # cg/kg → g/kg
            }
        }
    
    properties = {}
    strategy = {"method": "buffered_median", "window_px": 3}  # 3x3 median for soil properties
    
    # Sample each soil property
    for prop_name, dataset_path in config.get("soilgrids_datasets", {}).items():
        if not dataset_path or not Path(dataset_path).exists():
            properties[prop_name] = {
                "value": None,
                "error": "Dataset not available locally",
                "unit": _get_soil_unit(prop_name)
            }
            continue
        
        sample_result = sample_raster(lat, lon, dataset_path, strategy)
        
        if sample_result.get("success") and sample_result.get("value") is not None:
            # Apply scaling factor
            raw_value = sample_result["value"]
            scaling_factor = config.get("scaling_factors", {}).get(prop_name, 1.0)
            scaled_value = raw_value * scaling_factor
            
            properties[prop_name] = {
                "value": scaled_value,
                "unit": _get_soil_unit(prop_name),
                "raw_value": raw_value,
                "scaling_factor": scaling_factor,
                "distance_m": sample_result.get("distance_m"),
                "method": sample_result.get("method"),
                "source": "soilgrids_local_cog"
            }
        else:
            properties[prop_name] = {
                "value": None,
                "unit": _get_soil_unit(prop_name),
                "error": sample_result.get("error", "No data"),
                "source": "soilgrids_local_cog"
            }
    
    # Calculate USDA texture classification if sand/silt/clay available
    texture_class = None
    if all(prop in properties and properties[prop].get("value") is not None 
           for prop in ["sand_0_5cm", "silt_0_5cm", "clay_0_5cm"]):
        
        sand_pct = properties["sand_0_5cm"]["value"]
        silt_pct = properties["silt_0_5cm"]["value"] 
        clay_pct = properties["clay_0_5cm"]["value"]
        
        texture_class = _usda_texture_classification(sand_pct, silt_pct, clay_pct)
        
        properties["texture_classification"] = {
            "usda_class": texture_class,
            "sand_pct": sand_pct,
            "silt_pct": silt_pct,
            "clay_pct": clay_pct,
            "method": "usda_texture_triangle",
            "source": "calculated_from_soilgrids"
        }
    
    successful_properties = len([p for p in properties.values() if p.get("value") is not None])
    
    return {
        "soil_properties": properties,
        "success": successful_properties > 0,
        "properties_retrieved": successful_properties,
        "total_attempted": len(config.get("soilgrids_datasets", {})),
        "data_quality": "complete" if successful_properties > 6 else "partial" if successful_properties > 0 else "failed",
        "provenance": {
            "method": "soilgrids_local_cog_buffered_median",
            "window_size": "3x3_pixels",
            "resolution": "250m"
        }
    }


def _get_soil_unit(property_name: str) -> str:
    """Get unit for soil property."""
    units = {
        "ph_0_5cm": "pH",
        "soc_0_5cm": "g/kg",
        "sand_0_5cm": "%",
        "silt_0_5cm": "%", 
        "clay_0_5cm": "%",
        "bdod_0_5cm": "g/cm³",
        "nitrogen_0_5cm": "g/kg",
        "ocd_0_5cm": "kg/dm³",
        "ocs_0_30cm": "kg/m²"
    }
    return units.get(property_name, "unknown")


def _usda_texture_classification(sand_pct: float, silt_pct: float, clay_pct: float) -> str:
    """Classify USDA soil texture from sand/silt/clay percentages."""
    # USDA texture triangle classification rules
    if clay_pct >= 40:
        return "Clay"
    elif clay_pct >= 27:
        if sand_pct >= 45:
            return "Sandy clay"
        elif sand_pct >= 20:
            return "Clay loam"
        else:
            return "Silty clay"
    elif clay_pct >= 20:
        if sand_pct >= 45:
            return "Sandy clay loam"
        elif silt_pct >= 28:
            return "Silty clay loam"
        else:
            return "Clay loam"
    elif silt_pct >= 80:
        return "Silt"
    elif silt_pct >= 50:
        if sand_pct >= 20:
            return "Silt loam"
        else:
            return "Silt"
    elif sand_pct >= 85:
        return "Sand"
    elif sand_pct >= 70:
        return "Loamy sand"
    elif sand_pct >= 50:
        if silt_pct >= 28:
            return "Silt loam"
        else:
            return "Sandy loam"
    else:
        return "Loam"


def batch_sample_rasters(
    locations: List[Tuple[float, float]], 
    raster_configs: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Efficiently sample multiple rasters at multiple locations.
    
    Args:
        locations: List of (lat, lon) tuples
        raster_configs: Dict mapping dataset names to config dicts
        
    Returns:
        List of results for each location
    """
    results = []
    
    for lat, lon in locations:
        location_result = {"lat": lat, "lon": lon, "datasets": {}}
        
        for dataset_name, config in raster_configs.items():
            if "land_cover" in dataset_name:
                result = get_land_cover_local(lat, lon, config)
            elif "soil" in dataset_name:
                result = get_soil_properties_local(lat, lon, config)
            else:
                # Generic raster sampling
                result = sample_raster(lat, lon, config.get("path"), config.get("strategy"))
            
            location_result["datasets"][dataset_name] = result
        
        results.append(location_result)
    
    return results
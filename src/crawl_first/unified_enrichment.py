"""
Unified enrichment system implementing the comprehensive schema.

Coordinates all enrichment modules to produce consistent, distance-aware
results with proper provenance tracking following the unified schema.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import click

from .cache import cache_key, get_cache, save_cache

def _clean_result_for_cache(result: Dict[str, Any]) -> Dict[str, Any]:
    """Clean result for JSON serialization by converting DataFrames to strings."""
    import copy
    cleaned_result = copy.deepcopy(result)
    
    # Clean weather data DataFrames
    if "weather" in cleaned_result and "daily" in cleaned_result["weather"]:
        daily_weather = cleaned_result["weather"]["daily"]
        if "all_providers" in daily_weather:
            for provider_name, provider_data in daily_weather["all_providers"].items():
                if "hourly_df" in provider_data and provider_data["hourly_df"] is not None:
                    # Convert DataFrame to string representation
                    cleaned_result["weather"]["daily"]["all_providers"][provider_name]["hourly_df"] = str(provider_data["hourly_df"])
    
    return cleaned_result

from .weather_daily import weather_daily
from .site_typing import classify_site, haversine_km
from .osm_enrichment import enrich_osm_comprehensive, get_nearby_feature_summary
from .raster_sampling import get_land_cover_local, get_soil_properties_local, sample_raster
from .alternative_geospatial import (
    get_usda_nrcs_sda_soil, 
    get_soil_type_soilgrids, 
    get_isric_soilgrids_wcs_properties,
    get_land_cover_esa_worldcover,
    get_land_cover_nlcd
)
from .geospatial_enrichment import get_elevation_usgs, get_local_ecoregion, get_reverse_geocoding_nominatim
from .coast_distance import distance_to_coast_m
import geopandas as gpd
from shapely.geometry import Point
import requests
import os
from pathlib import Path


def load_enrichment_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load enrichment configuration from YAML file with JSON fallback."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "enrichment.yml"
    
    # Try YAML first
    try:
        import yaml
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if config:
                    return config
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback to JSON config
    json_path = config_path.with_suffix('.json')
    try:
        if json_path.exists():
            with open(json_path) as f:
                config = json.load(f)
                if config:
                    return config
    except Exception:
        pass
    
    # Return built-in defaults if no config files available
    return _get_default_config()


def _get_default_config() -> Dict[str, Any]:
    """Get default configuration when config file is not available."""
    return {
        "datasets": {
            "elevation": {
                "copernicus_dem": "/data/copernicus_dem_30m.tif",
                "usgs_point_service": True
            },
            "land_cover": {
                "worldcover_2021": "/data/worldcover_2021.tif",
                "nlcd_2019": "/data/nlcd_2019.tif"
            },
            "soils": {
                "soilgrids_ph": "/data/soilgrids/phh2o_0-5cm_mean.tif",
                "soilgrids_soc": "/data/soilgrids/soc_0-5cm_mean.tif",
                "soilgrids_sand": "/data/soilgrids/sand_0-5cm_mean.tif",
                "soilgrids_silt": "/data/soilgrids/silt_0-5cm_mean.tif",
                "soilgrids_clay": "/data/soilgrids/clay_0-5cm_mean.tif",
                "soilgrids_bdod": "/data/soilgrids/bdod_0-5cm_mean.tif"
            },
            "coastlines": "/data/coastlines/gshhs_c.shp",
            "inland_water": "/data/hydrolakes/HydroLAKES_polys_v10.shp"
        },
        "providers": {
            "weather": {
                "primary": "open_meteo",
                "secondary": "meteostat",
                "fallback": "prism_us"
            },
            "elevation": {
                "primary": "local_dem",
                "fallback": "usgs_point"
            }
        },
        "osm": {
            "default_radius_m": 1000,
            "timeout_s": 180,
            "max_features": 100
        },
        "site_typing": {
            "D_open_km": 20,
            "D_inland_km": 0.2,
            "D_coast_km": 2
        },
        "crosswalks": {
            "envo_mapping": "/mappings/envo_crosswalk.json"
        }
    }


def enrich_biosample_unified(
    biosample_id: str,
    lat: float,
    lon: float,
    collection_date: Optional[str] = None,
    place: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    save_cache_enabled: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive biosample enrichment using unified schema.
    
    Args:
        biosample_id: Unique biosample identifier
        lat: Latitude
        lon: Longitude
        collection_date: Collection date in YYYY-MM-DD format
        place: Optional place name for forward geocoding
        config: Configuration dict
        use_cache: Whether to use cached results (default: True)
        save_cache_enabled: Whether to save results to cache (default: True)
        
    Returns:
        Dict following unified enrichment schema
    """
    if config is None:
        config = load_enrichment_config()
    
    # Generate cache key from input parameters
    cache_params = {
        "biosample_id": biosample_id,
        "lat": lat,
        "lon": lon,
        "collection_date": collection_date,
        "place": place,
        "config_version": config.get("version", "default")
    }
    cache_key_str = cache_key(cache_params)
    
    # Check cache first if enabled
    if use_cache:
        cached_result = get_cache("unified_enrichment", cache_key_str)
        if cached_result:
            return cached_result
    
    # Initialize unified result structure
    result = {
        "id": biosample_id,
        "point": {"lat": lat, "lon": lon},
        "collection_date": collection_date,
        "site_type": {},
        "elevation": {},
        "administrative": {},
        "forward_geocoding": {},
        "weather": {},
        "land_cover": {},
        "soils": {},
        "ecoregion": {},
        "osm": {},
        "nearby": {},
        "provenance": {
            "run_id": f"enrichment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "software": "crawl_first unified_enrichment v1.0",
            "config_version": config.get("version", "default")
        }
    }
    
    enrichment_errors = []
    
    # 1. Site typing (terrestrial/aquatic/marine classification)
    try:
        site_config = config.get("site_typing", {})
        site_result = classify_site(lat, lon, site_config)
        result["site_type"] = site_result
    except Exception as e:
        enrichment_errors.append(f"Site typing failed: {e}")
        result["site_type"] = {"error": str(e)}
    
    # 2. Elevation data (multi-provider: Local DEM + USGS + Google)
    try:
        elevation_sources = {}
        
        # Local DEM sampling
        dem_datasets = config.get("datasets", {}).get("elevation", {})
        for dem_name, dem_path in dem_datasets.items():
            if isinstance(dem_path, str) and Path(dem_path).exists():
                dem_result = sample_raster(lat, lon, dem_path, {"method": "bilinear"})
                if dem_result.get("success"):
                    elevation_sources[dem_name] = {
                        "elev_m": dem_result["value"],
                        "distance_km": dem_result["distance_m"] / 1000,
                        "resolution": dem_result["resolution"],
                        "method": dem_result["method"],
                        "source": "local_dem"
                    }
        
        # USGS point service (US only)
        if -170 <= lon <= -60 and 15 <= lat <= 75:  # US bounds
            try:
                usgs_result = get_elevation_usgs(lat, lon)
                if usgs_result.get("success"):
                    elevation_sources["usgs_point"] = {
                        "elev_m": round(usgs_result["elevation_meters"], 1),  # Round to 0.1m precision
                        "distance_km": 0.0,
                        "source": "usgs_point_service",
                        "units": usgs_result.get("units", "meters")
                    }
            except Exception:
                pass
        
        # Google Elevation API (global)
        try:
            google_result = get_elevation_google(lat, lon)
            if google_result.get("success"):
                elevation_sources["google_elevation"] = google_result
        except Exception:
            pass
        
        result["elevation"] = elevation_sources
        
    except Exception as e:
        enrichment_errors.append(f"Elevation enrichment failed: {e}")
        result["elevation"] = {"error": str(e)}
    
    # 3. Administrative boundaries (multi-provider: Nominatim + Google)
    try:
        admin_result = get_reverse_geocoding_multi(lat, lon)
        result["administrative"] = admin_result
    except Exception as e:
        enrichment_errors.append(f"Administrative boundary enrichment failed: {e}")
        result["administrative"] = {"error": str(e)}
    
    # 4. Forward geocoding (if place name provided)
    if place:
        try:
            forward_result = get_forward_geocoding_multi(place)
            result["forward_geocoding"] = forward_result
        except Exception as e:
            enrichment_errors.append(f"Forward geocoding failed: {e}")
            result["forward_geocoding"] = {"error": str(e), "query": place}
    
    # 5. Weather data (if collection date provided)
    if collection_date:
        try:
            weather_result = weather_daily(lat, lon, collection_date)
            result["weather"]["daily"] = weather_result
        except Exception as e:
            enrichment_errors.append(f"Weather enrichment failed: {e}")
            result["weather"] = {"error": str(e)}
    
    # 4. Land cover (multi-provider: ESA WorldCover + NLCD for US + Local files)
    try:
        land_cover_result = get_land_cover_multi(lat, lon, 2021)
        result["land_cover"] = land_cover_result
    except Exception as e:
        enrichment_errors.append(f"Land cover enrichment failed: {e}")
        result["land_cover"] = {"error": str(e)}
    
    # 5. Soil properties (try MCP soil service if local files unavailable)
    try:
        soil_config = {
            "soilgrids_datasets": {
                "ph_0_5cm": config.get("datasets", {}).get("soils", {}).get("soilgrids_ph"),
                "soc_0_5cm": config.get("datasets", {}).get("soils", {}).get("soilgrids_soc"),
                "sand_0_5cm": config.get("datasets", {}).get("soils", {}).get("soilgrids_sand"),
                "silt_0_5cm": config.get("datasets", {}).get("soils", {}).get("soilgrids_silt"),
                "clay_0_5cm": config.get("datasets", {}).get("soils", {}).get("soilgrids_clay"),
                "bdod_0_5cm": config.get("datasets", {}).get("soils", {}).get("soilgrids_bdod")
            }
        }
        
        # First try local files
        soil_result = get_soil_properties_local(lat, lon, soil_config)
        
        # If local files failed, try working APIs without complex mappings
        if soil_result.get("soil_properties", {}).get("ph_0_5cm", {}).get("error") == "Dataset not available locally":
            # Try premium soil APIs: USDA NRCS SDA for US, SoilGrids classification + WCS properties globally
            try:
                
                # For US locations, try USDA NRCS SDA first (much richer data)
                usda_result = get_usda_nrcs_sda_soil(lat, lon)
                soilgrids_result = get_soil_type_soilgrids(lat, lon)
                
                # Always get SoilGrids WCS properties (pH, SOC, texture) - works globally
                soilgrids_wcs_result = get_isric_soilgrids_wcs_properties(lat, lon)
                
                # Also use OSM to infer basic soil context from land use
                osm_data = enrich_osm_comprehensive(lat, lon, 500)  # Smaller radius for soil context
                
                # Extract land use context that might indicate soil conditions
                soil_context = []
                for feature in osm_data.get("named_features", []):
                    tags = feature.get("tags", {})
                    if "landuse" in tags:
                        soil_context.append(f"landuse:{tags['landuse']}")
                    if "natural" in tags and tags["natural"] in ["forest", "grassland", "wetland", "scrub"]:
                        soil_context.append(f"natural:{tags['natural']}")
                
                # Build comprehensive soil result with US-first priority
                soil_apis = {}
                
                if usda_result.get("success"):
                    soil_apis["usda_nrcs_sda"] = {
                        "mukey": usda_result.get("mukey"),
                        "component_name": usda_result.get("component_name"),
                        "component_percent": usda_result.get("component_percent"),
                        "taxonomic_class": usda_result.get("taxonomic_class"),
                        "full_taxonomy": usda_result.get("full_taxonomy"),
                        "tax_order": usda_result.get("tax_order"),
                        "tax_suborder": usda_result.get("tax_suborder"),
                        "tax_great_group": usda_result.get("tax_great_group"),
                        "tax_subgroup": usda_result.get("tax_subgroup"),
                        "drainage_class": usda_result.get("drainage_class"),
                        "hydrologic_group": usda_result.get("hydrologic_group"),
                        "mapunit_name": usda_result.get("mapunit_name"),
                        "data_source": "USDA NRCS SDA",
                        "method": "soil_data_access_api",
                        "success": True,
                        "coverage": "US only - high quality soil taxonomy"
                    }
                else:
                    soil_apis["usda_nrcs_sda"] = {
                        "error": usda_result.get("error", "Failed to retrieve USDA soil data"),
                        "data_source": "USDA NRCS SDA",
                        "success": False
                    }
                
                # Always include SoilGrids classification as global fallback/comparison
                if soilgrids_result.get("soil_type"):
                    soil_apis["soilgrids_classification"] = {
                        "fao_soil_type": soilgrids_result.get("soil_type"),
                        "confidence": soilgrids_result.get("confidence"),
                        "alternatives": soilgrids_result.get("alternatives", {}),
                        "data_source": "ISRIC SoilGrids v2.0 REST",
                        "method": "soilgrids_rest_api",
                        "success": True,
                        "coverage": "Global - 250m resolution"
                    }
                else:
                    soil_apis["soilgrids_classification"] = {
                        "error": "SoilGrids classification API failed",
                        "data_source": "ISRIC SoilGrids v2.0 REST",
                        "success": False
                    }
                
                # Include SoilGrids WCS properties (pH, SOC, texture, etc.)
                if soilgrids_wcs_result.get("success"):
                    soil_apis["soilgrids_properties"] = {
                        "soil_properties": soilgrids_wcs_result.get("soil_properties", {}),
                        "usda_texture_class": soilgrids_wcs_result.get("usda_texture_class"),
                        "data_source": "ISRIC SoilGrids WCS",
                        "method": "wcs_raster_sampling", 
                        "resolution": "250m",
                        "success": True,
                        "coverage": "Global - quantitative properties"
                    }
                else:
                    soil_apis["soilgrids_properties"] = {
                        "error": soilgrids_wcs_result.get("error", "SoilGrids WCS API failed"),
                        "data_source": "ISRIC SoilGrids WCS",
                        "success": False
                    }
                
                result["soils"] = {
                    **soil_apis,
                    "local_context": {
                        "land_use_indicators": soil_context[:5] if soil_context else ["no_indicators_found"],
                        "method": "osm_landuse_inference",
                        "radius_m": 500
                    },
                    "strategy": "US locations: USDA NRCS SDA primary, SoilGrids secondary. Global: SoilGrids primary."
                }
                
            except Exception as e:
                result["soils"] = {
                    "soilgrids": soil_result.get("soil_properties", {}),
                    "error": f"All soil data sources failed: {str(e)}"
                }
        else:
            result["soils"] = {"soilgrids": soil_result.get("soil_properties", {})}
            
    except Exception as e:
        enrichment_errors.append(f"Soil enrichment failed: {e}")
        result["soils"] = {"error": str(e)}
    
    # 6. Ecoregion (using existing TEOW system)
    try:
        ecoregion_result = get_local_ecoregion(lat, lon)
        if ecoregion_result.get("success"):
            result["ecoregion"]["teow_2017"] = {
                "eco_name": ecoregion_result.get("ecoregion_name"),
                "biome": ecoregion_result.get("biome_name"),
                "realm": ecoregion_result.get("realm"),
                "distance_km": 0.0,  # Point-in-polygon
                "source": "local_teow_shapefile"
            }
    except Exception as e:
        enrichment_errors.append(f"Ecoregion enrichment failed: {e}")
        result["ecoregion"] = {"error": str(e)}
    
    # 7. OSM enrichment
    try:
        osm_config = config.get("osm", {})
        radius_m = osm_config.get("default_radius_m", 1000)
        timeout_s = osm_config.get("timeout_s", 180)
        
        osm_result = enrich_osm_comprehensive(lat, lon, radius_m, timeout_s)
        result["osm"] = osm_result
        
        # Generate nearby feature summary
        if not osm_result.get("error"):
            nearby_summary = get_nearby_feature_summary(osm_result)
            result["nearby"] = nearby_summary
            
    except Exception as e:
        enrichment_errors.append(f"OSM enrichment failed: {e}")
        result["osm"] = {"error": str(e)}
        result["nearby"] = {"error": str(e)}
    
    # 8. Google Places enrichment
    try:
        places_result = get_places_google(lat, lon, 1000)  # 1km radius
        result["places"] = places_result
    except Exception as e:
        enrichment_errors.append(f"Google Places enrichment failed: {e}")
        result["places"] = {"error": str(e)}
    
    # 9. Air quality enrichment (multi-provider: Google + EPA + OpenWeatherMap)
    try:
        air_quality_result = get_air_quality_multi(lat, lon, collection_date)
        result["air_quality"] = air_quality_result
    except Exception as e:
        enrichment_errors.append(f"Air quality enrichment failed: {e}")
        result["air_quality"] = {"error": str(e)}
    
    # Add distance calculations to coastlines and major features
    try:
        result["nearby"]["distance_to_coast_km"] = _estimate_coast_distance(lat, lon)
        result["nearby"]["settlement_population_within_10km"] = _estimate_nearby_population(lat, lon)
    except Exception as e:
        enrichment_errors.append(f"Distance calculations failed: {e}")
    
    # Add error summary if there were any errors
    if enrichment_errors:
        result["enrichment_errors"] = enrichment_errors
        result["enrichment_success_rate"] = 1.0 - (len(enrichment_errors) / 9)  # 9 main enrichment types
    else:
        result["enrichment_success_rate"] = 1.0
    
    # Save to cache if enabled
    if save_cache_enabled:
        cleaned_result = _clean_result_for_cache(result)
        save_cache("unified_enrichment", cache_key_str, cleaned_result)
    
    return result


def _estimate_coast_distance(lat: float, lon: float) -> Optional[float]:
    """Calculate distance to coast using GPT-5's robust OSM-based solution."""
    try:
        dist_m = distance_to_coast_m(lat, lon)
        if dist_m is not None:
            return round(dist_m / 1000.0, 2)  # Convert to km
        return None
    except Exception:
        # Fallback to heuristic if coastline data fails
        return _fallback_coast_distance(lat, lon)


def _fallback_coast_distance(lat: float, lon: float) -> Optional[float]:
    """Fallback coast distance estimation using geographic heuristics."""
    continental_centers = [
        (39.0, -98.0),   # Center of USA
        (55.0, 100.0),   # Siberia  
        (-25.0, 135.0),  # Australian outback
        (0.0, 25.0),     # Central Africa
        (-15.0, -60.0),  # South America interior
    ]
    
    min_dist_to_center = min(
        haversine_km(lat, lon, clat, clon) for clat, clon in continental_centers
    )
    
    # Simple heuristic
    if min_dist_to_center < 500:
        return min_dist_to_center * 2  
    else:
        return max(0, abs(lon) * 50 - 100)


def _estimate_nearby_population(lat: float, lon: float) -> Optional[int]:
    """Rough estimate of nearby population (placeholder for actual population data)."""
    # This is a placeholder - in production would use actual population datasets
    major_cities = [
        (40.7128, -74.0060, 8000000),  # NYC
        (34.0522, -118.2437, 4000000), # LA
        (41.8781, -87.6298, 3000000),  # Chicago
        (29.7604, -95.3698, 2300000),  # Houston
        (51.5074, -0.1278, 9000000),   # London
        (48.8566, 2.3522, 2100000),    # Paris
    ]
    
    total_pop = 0
    for city_lat, city_lon, population in major_cities:
        distance = haversine_km(lat, lon, city_lat, city_lon)
        if distance <= 10:  # Within 10km
            total_pop += population
        elif distance <= 50:  # Population decay with distance
            total_pop += int(population * (50 - distance) / 50 * 0.1)
    
    return total_pop if total_pop > 0 else None


def batch_enrich_biosamples(
    biosamples: List[Dict[str, Any]], 
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Batch enrichment for multiple biosamples.
    
    Args:
        biosamples: List of dicts with id, lat, lon, collection_date
        config: Configuration dict
        
    Returns:
        List of enriched biosample results
    """
    if config is None:
        config = load_enrichment_config()
    
    results = []
    
    for biosample in biosamples:
        try:
            result = enrich_biosample_unified(
                biosample_id=biosample["id"],
                lat=biosample["lat"], 
                lon=biosample["lon"],
                collection_date=biosample.get("collection_date"),
                config=config
            )
            results.append(result)
        except Exception as e:
            # Create error result for failed enrichment
            error_result = {
                "id": biosample["id"],
                "point": {"lat": biosample["lat"], "lon": biosample["lon"]},
                "enrichment_error": str(e),
                "enrichment_success_rate": 0.0,
                "provenance": {
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "software": "crawl_first unified_enrichment v1.0"
                }
            }
            results.append(error_result)
    
    return results


# CLI Application
@click.group(help="Unified Geospatial Enrichment System")
def cli():
    """Unified Geospatial Enrichment System for biosample data."""
    pass


def get_forward_geocoding_nominatim(place_name: str) -> Dict[str, Any]:
    """Forward geocode using OSM Nominatim with comprehensive detail extraction."""
    import requests
    import time
    
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "addressdetails": 1,
        "extratags": 1,
        "namedetails": 1,
        "limit": 1,
        "dedupe": 1
    }
    
    headers = {"User-Agent": "BiosampleEnrichment/1.0 (research purposes)"}
    
    try:
        time.sleep(1.1)  # Respect rate limiting
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            result = data[0]
            address = result.get("address", {})
            
            return {
                "provider": "nominatim",
                "lat": float(result["lat"]),
                "lon": float(result["lon"]),
                "display_name": result.get("display_name"),
                "place_type": result.get("type"),
                "place_class": result.get("class"),
                "importance": result.get("importance"),
                "country": address.get("country"),
                "country_code": address.get("country_code"),
                "state": address.get("state"),
                "county": address.get("county"),
                "city": address.get("city") or address.get("town") or address.get("village"),
                "postcode": address.get("postcode"),
                "bounding_box": result.get("boundingbox"),
                "data_source": "OpenStreetMap Nominatim",
                "success": True,
                "query": place_name
            }
        else:
            return {
                "provider": "nominatim",
                "error": f"No results found for '{place_name}'",
                "success": False,
                "query": place_name
            }
            
    except Exception as e:
        return {
            "provider": "nominatim",
            "error": f"Nominatim failed: {str(e)}",
            "success": False,
            "query": place_name
        }


def get_forward_geocoding_google(place_name: str) -> Dict[str, Any]:
    """Forward geocode using Google Geocoding API."""
    import requests
    import os
    from pathlib import Path
    
    # Load local environment variables
    env_file = Path(__file__).parent.parent.parent / "local" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    
    api_key = os.environ.get("GOOGLE_ELEVATION_API_KEY")
    if not api_key:
        return {
            "provider": "google",
            "error": "GOOGLE_ELEVATION_API_KEY not set",
            "success": False,
            "query": place_name
        }
    
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": place_name,
        "key": api_key,
        "language": "en"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK" and data.get("results"):
            result = data["results"][0]
            location = result["geometry"]["location"]
            
            # Extract address components
            address_components = {comp["types"][0]: comp["long_name"] 
                                for comp in result.get("address_components", [])}
            
            return {
                "provider": "google",
                "lat": float(location["lat"]),
                "lon": float(location["lng"]),
                "display_name": result.get("formatted_address"),
                "place_type": result.get("types", [None])[0] if result.get("types") else None,
                "country": address_components.get("country"),
                "country_code": address_components.get("country"),
                "state": address_components.get("administrative_area_level_1"),
                "county": address_components.get("administrative_area_level_2"),
                "city": address_components.get("locality"),
                "postcode": address_components.get("postal_code"),
                "data_source": "Google Geocoding API",
                "success": True,
                "query": place_name
            }
        else:
            return {
                "provider": "google",
                "error": f"Google API status: {data.get('status', 'unknown')}",
                "success": False,
                "query": place_name
            }
            
    except Exception as e:
        return {
            "provider": "google",
            "error": f"Google geocoding failed: {str(e)}",
            "success": False,
            "query": place_name
        }


def get_reverse_geocoding_google(lat: float, lon: float) -> Dict[str, Any]:
    """Google reverse geocoding for administrative boundaries."""
    import requests
    import os
    from pathlib import Path
    
    # Load local environment variables
    env_file = Path(__file__).parent.parent.parent / "local" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    
    api_key = os.environ.get("GOOGLE_ELEVATION_API_KEY")
    if not api_key:
        return {
            "provider": "google",
            "success": False,
            "error": "GOOGLE_ELEVATION_API_KEY not set"
        }
    
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lon}",
        "key": api_key,
        "language": "en"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK" and data.get("results"):
            result_data = data["results"][0]
            address_components = result_data.get("address_components", [])
            geometry = result_data.get("geometry", {})
            location = geometry.get("location", {})
            
            # Parse address components
            component_types = {}
            for component in address_components:
                long_name = component.get("long_name")
                short_name = component.get("short_name")
                types = component.get("types", [])
                
                for type_name in types:
                    if type_name not in component_types:
                        component_types[type_name] = {
                            "long_name": long_name,
                            "short_name": short_name
                        }
            
            return {
                "provider": "google",
                "success": True,
                "country": component_types.get("country", {}).get("long_name"),
                "country_code": component_types.get("country", {}).get("short_name"),
                "state": component_types.get("administrative_area_level_1", {}).get("long_name"),
                "county": component_types.get("administrative_area_level_2", {}).get("long_name"),
                "city": component_types.get("locality", {}).get("long_name"),
                "postcode": component_types.get("postal_code", {}).get("long_name"),
                "display_name": result_data.get("formatted_address"),
                "place_type": result_data.get("types", [None])[0],
                "place_id": result_data.get("place_id"),
                "latitude": location.get("lat", lat),
                "longitude": location.get("lng", lon),
                "data_source": "Google Geocoding API"
            }
        else:
            return {
                "provider": "google",
                "success": False,
                "error": f"Google API status: {data.get('status', 'unknown')}"
            }
            
    except Exception as e:
        return {
            "provider": "google",
            "success": False,
            "error": str(e)
        }


def get_elevation_google(lat: float, lon: float) -> Dict[str, Any]:
    """Google Elevation API for additional elevation data source."""
    import requests
    import os
    from pathlib import Path
    
    # Load local environment variables
    env_file = Path(__file__).parent.parent.parent / "local" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    
    api_key = os.environ.get("GOOGLE_ELEVATION_API_KEY")
    if not api_key:
        return {
            "provider": "google_elevation",
            "success": False,
            "error": "GOOGLE_ELEVATION_API_KEY not set"
        }
    
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = {
        "locations": f"{lat},{lon}",
        "key": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK" and data.get("results"):
            result = data["results"][0]
            return {
                "provider": "google_elevation",
                "success": True,
                "elev_m": round(result.get("elevation", 0), 1),
                "distance_km": 0.0,  # Point-specific 
                "resolution": result.get("resolution", "unknown"),
                "source": "google_elevation_api",
                "data_source": "Google Elevation API",
                "location": {
                    "lat": result.get("location", {}).get("lat", lat),
                    "lon": result.get("location", {}).get("lng", lon)
                }
            }
        else:
            return {
                "provider": "google_elevation", 
                "success": False,
                "error": f"Google Elevation API status: {data.get('status', 'unknown')}"
            }
            
    except Exception as e:
        return {
            "provider": "google_elevation",
            "success": False,
            "error": str(e)
        }


def get_places_google(lat: float, lon: float, radius: int = 1000) -> Dict[str, Any]:
    """Google Places API for nearby place context and business data."""
    import requests
    import os
    from pathlib import Path
    
    # Load local environment variables
    env_file = Path(__file__).parent.parent.parent / "local" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    
    api_key = os.environ.get("GOOGLE_ELEVATION_API_KEY")  # Same key works for all Google Maps APIs
    if not api_key:
        return {
            "provider": "google_places",
            "success": False,
            "error": "GOOGLE_ELEVATION_API_KEY not set"
        }
    
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius": radius,
        "key": api_key,
        "language": "en"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK":
            places = data.get("results", [])
            
            # Process places for microbiome context
            place_types = {}
            notable_places = []
            
            for place in places[:20]:  # Limit to 20 most relevant
                place_info = {
                    "name": place.get("name"),
                    "types": place.get("types", []),
                    "rating": place.get("rating"),
                    "user_ratings_total": place.get("user_ratings_total"),
                    "vicinity": place.get("vicinity"),
                    "place_id": place.get("place_id")
                }
                
                # Distance calculation
                place_lat = place.get("geometry", {}).get("location", {}).get("lat")
                place_lon = place.get("geometry", {}).get("location", {}).get("lng")
                if place_lat and place_lon:
                    place_info["distance_km"] = round(haversine_km(lat, lon, place_lat, place_lon), 3)
                
                notable_places.append(place_info)
                
                # Count place types for environmental context
                for place_type in place.get("types", []):
                    place_types[place_type] = place_types.get(place_type, 0) + 1
            
            return {
                "provider": "google_places",
                "success": True,
                "total_places": len(places),
                "radius_m": radius,
                "place_types": place_types,
                "notable_places": notable_places,
                "data_source": "Google Places API",
                "environmental_context": {
                    "has_natural_features": any(t in place_types for t in ["park", "natural_feature", "establishment"]),
                    "has_industrial": any(t in place_types for t in ["gas_station", "car_repair", "storage"]),
                    "has_agriculture": any(t in place_types for t in ["farm", "food", "grocery_or_supermarket"]),
                    "human_activity_level": min(len(places) / 5, 5)  # 0-5 scale
                }
            }
        else:
            return {
                "provider": "google_places",
                "success": False,
                "error": f"Google Places API status: {data.get('status', 'unknown')}"
            }
            
    except Exception as e:
        return {
            "provider": "google_places",
            "success": False,
            "error": str(e)
        }


def get_air_quality_google(lat: float, lon: float, date: Optional[str] = None) -> Dict[str, Any]:
    """Google Air Quality API for comprehensive air pollution data."""
    import requests
    import os
    from pathlib import Path
    
    # Load local environment variables
    env_file = Path(__file__).parent.parent.parent / "local" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    
    api_key = os.environ.get("GOOGLE_ELEVATION_API_KEY")  # Same key for all Google APIs
    if not api_key:
        return {
            "provider": "google_air_quality",
            "success": False,
            "error": "GOOGLE_ELEVATION_API_KEY not set"
        }
    
    # Use historical endpoint if date provided, otherwise current conditions
    if date:
        url = "https://airquality.googleapis.com/v1/history:lookup"
        # Calculate next day for endTime (API expects date range, not single day)
        from datetime import datetime, timedelta
        start_date = datetime.strptime(date, "%Y-%m-%d")
        end_date = start_date + timedelta(days=1)
        
        request_body = {
            "location": {
                "latitude": lat,
                "longitude": lon
            },
            "period": {
                "startTime": start_date.strftime("%Y-%m-%dT00:00:00Z"),
                "endTime": end_date.strftime("%Y-%m-%dT00:00:00Z")
            },
            "pageSize": 24  # Request 24 hours of data
        }
    else:
        url = "https://airquality.googleapis.com/v1/currentConditions:lookup" 
        request_body = {
            "location": {
                "latitude": lat,
                "longitude": lon
            },
            "includeLocalAqi": True,
            "includeHealthSuggestions": True,
            "includeDominantPollutant": True,
            "includeAdditionalPollutantInfo": True
        }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(f"{url}?key={api_key}", 
                               json=request_body, 
                               headers=headers, 
                               timeout=30)
        
        # Debug: Log request details for troubleshooting
        if response.status_code != 200:
            error_details = {
                "status_code": response.status_code,
                "url": url,
                "request_body": request_body,
                "response_text": response.text[:500] if response.text else None
            }
            
            # Special handling for historical data limitations
            if date and "time period is not supported" in response.text:
                return {
                    "provider": "google_air_quality",
                    "success": False,
                    "error": f"Google Air Quality historical data not available for {date}. API may have limited historical coverage or require special permissions.",
                    "limitation": "Google Air Quality API historical data access appears limited. Consider alternative sources for historical air quality data.",
                    "debug_info": error_details
                }
            
            return {
                "provider": "google_air_quality",
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:200] if response.text else 'No response body'}",
                "debug_info": error_details
            }
        
        response.raise_for_status()
        data = response.json()
        
        # Handle both current conditions and historical data responses
        if "indexes" in data:
            # Current conditions response
            indexes = data.get("indexes", [])
            pollutants = data.get("pollutants", [])
            
            # Find universal AQI
            universal_aqi = None
            local_aqi = None
            for index in indexes:
                if index.get("code") == "uaqi":
                    universal_aqi = index
                elif index.get("code") == "usa_epa":
                    local_aqi = index
            
            # Process pollutant concentrations
            pollutant_data = {}
            for pollutant in pollutants:
                code = pollutant.get("code")
                concentration = pollutant.get("concentration", {})
                pollutant_data[code] = {
                    "value": concentration.get("value"),
                    "units": concentration.get("units"),
                    "full_name": pollutant.get("fullName"),
                    "additional_info": pollutant.get("additionalInfo", {})
                }
            
            return {
                "provider": "google_air_quality",
                "success": True,
                "data_source": "Google Air Quality API",
                "query_type": "historical" if date else "current",
                "query_date": date,
                "universal_aqi": {
                    "aqi": universal_aqi.get("aqi") if universal_aqi else None,
                    "category": universal_aqi.get("category") if universal_aqi else None,
                    "dominant_pollutant": universal_aqi.get("dominantPollutant") if universal_aqi else None
                },
                "local_aqi": {
                    "aqi": local_aqi.get("aqi") if local_aqi else None,
                    "category": local_aqi.get("category") if local_aqi else None
                } if local_aqi else None,
                "pollutants": pollutant_data,
                "health_recommendations": data.get("healthRecommendations", {}),
                "datetime_utc": data.get("dateTime")
            }
            
        elif "hoursInfo" in data:
            # Historical data response - aggregate daily values
            hours_info = data.get("hoursInfo", [])
            
            if not hours_info:
                return {
                    "provider": "google_air_quality",
                    "success": False,
                    "error": f"No historical air quality data available for {date}"
                }
            
            # Aggregate hourly data to daily statistics
            daily_pollutants = {}
            daily_aqi_values = []
            
            for hour_data in hours_info:
                # Collect AQI values
                for index in hour_data.get("indexes", []):
                    if index.get("code") == "uaqi":
                        aqi_val = index.get("aqi")
                        if aqi_val:
                            daily_aqi_values.append(aqi_val)
                
                # Collect pollutant concentrations
                for pollutant in hour_data.get("pollutants", []):
                    code = pollutant.get("code")
                    concentration = pollutant.get("concentration", {})
                    value = concentration.get("value")
                    
                    if value and code:
                        if code not in daily_pollutants:
                            daily_pollutants[code] = {
                                "values": [],
                                "units": concentration.get("units"),
                                "full_name": pollutant.get("fullName")
                            }
                        daily_pollutants[code]["values"].append(value)
            
            # Calculate daily statistics
            aqi_stats = {}
            if daily_aqi_values:
                aqi_stats = {
                    "min": min(daily_aqi_values),
                    "max": max(daily_aqi_values),
                    "avg": round(sum(daily_aqi_values) / len(daily_aqi_values), 1),
                    "hours_available": len(daily_aqi_values)
                }
            
            pollutant_stats = {}
            for code, poll_data in daily_pollutants.items():
                values = poll_data["values"]
                pollutant_stats[code] = {
                    "min": round(min(values), 3),
                    "max": round(max(values), 3), 
                    "avg": round(sum(values) / len(values), 3),
                    "units": poll_data["units"],
                    "full_name": poll_data["full_name"],
                    "hours_available": len(values)
                }
            
            return {
                "provider": "google_air_quality",
                "success": True,
                "data_source": "Google Air Quality API",
                "query_type": "historical",
                "query_date": date,
                "universal_aqi_daily": aqi_stats,
                "pollutants_daily": pollutant_stats,
                "total_hours": len(hours_info),
                "coverage": "complete" if len(hours_info) >= 20 else "partial"  # 20+ hours = good coverage
            }
        else:
            return {
                "provider": "google_air_quality",
                "success": False,
                "error": "Unexpected response format from Google Air Quality API"
            }
            
    except Exception as e:
        return {
            "provider": "google_air_quality",
            "success": False,
            "error": str(e)
        }


def get_air_quality_epa_airnow(lat: float, lon: float) -> Dict[str, Any]:
    """EPA AirNow API for official US government air quality data."""
    import requests
    import os
    from pathlib import Path
    
    # Load local environment variables for EPA API key
    env_file = Path(__file__).parent.parent.parent / "local" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    
    api_key = os.environ.get("EPA_AIRNOW_API_KEY")
    if not api_key:
        return {
            "provider": "epa_airnow",
            "success": False,
            "error": "EPA_AIRNOW_API_KEY not set (US only - free registration at airnowapi.org)"
        }
    
    # Check if location is within US bounds
    if not (-170 <= lon <= -60 and 15 <= lat <= 75):
        return {
            "provider": "epa_airnow",
            "success": False,
            "error": "EPA AirNow only covers US locations"
        }
    
    url = "https://www.airnowapi.org/aq/observation/latLong/current/"
    params = {
        "format": "application/json",
        "latitude": lat,
        "longitude": lon,
        "distance": 25,  # Search within 25 miles
        "API_KEY": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            # Process EPA AirNow data
            observations = {}
            for obs in data:
                parameter = obs.get("ParameterName", "").lower().replace(".", "")
                observations[parameter] = {
                    "aqi": obs.get("AQI"),
                    "category": obs.get("Category", {}).get("Name"),
                    "category_number": obs.get("Category", {}).get("Number"),
                    "site_name": obs.get("ReportingArea"),
                    "state_code": obs.get("StateCode"),
                    "date_observed": obs.get("DateObserved"),
                    "hour_observed": obs.get("HourObserved"),
                    "local_time_zone": obs.get("LocalTimeZone")
                }
            
            return {
                "provider": "epa_airnow",
                "success": True,
                "data_source": "EPA AirNow",
                "observations": observations,
                "coverage": "US only - official government data"
            }
        else:
            return {
                "provider": "epa_airnow",
                "success": False,
                "error": "No air quality stations found within 25 miles"
            }
            
    except Exception as e:
        return {
            "provider": "epa_airnow",
            "success": False,
            "error": str(e)
        }


def get_air_quality_openaq(lat: float, lon: float, date: Optional[str] = None) -> Dict[str, Any]:
    """OpenAQ API for global historical air quality data (no API key required)."""
    import requests
    from datetime import datetime, timedelta
    
    # Try v3 API first, then v2 if needed
    base_url = "https://api.openaq.org/v3/measurements"
    
    # If no date provided, get recent data
    if date:
        # Get data for the specific date
        start_date = datetime.strptime(date, "%Y-%m-%d")
        end_date = start_date + timedelta(days=1)
        date_from = start_date.strftime("%Y-%m-%d")
        date_to = end_date.strftime("%Y-%m-%d")
    else:
        # Get recent data (last 7 days)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        date_from = start_date.strftime("%Y-%m-%d")
        date_to = end_date.strftime("%Y-%m-%d")
    
    # Parameters for OpenAQ v3 API
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": 25000,  # 25km radius in meters
        "datetime_from": f"{date_from}T00:00:00Z",
        "datetime_to": f"{date_to}T00:00:00Z",
        "limit": 1000,
        "sort": "desc"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        if not results:
            return {
                "provider": "openaq",
                "success": False,
                "error": f"No air quality measurements found within 25km of coordinates for {date if date else 'recent period'}"
            }
        
        # Group measurements by parameter and calculate daily statistics
        pollutants = {}
        locations = set()
        
        for measurement in results:
            parameter = measurement.get("parameter")
            value = measurement.get("value")
            unit = measurement.get("unit")
            location = measurement.get("location")
            
            if parameter and value is not None:
                if parameter not in pollutants:
                    pollutants[parameter] = {
                        "values": [],
                        "unit": unit,
                        "measurements_count": 0
                    }
                
                pollutants[parameter]["values"].append(value)
                pollutants[parameter]["measurements_count"] += 1
                
                if location:
                    locations.add(location)
        
        # Calculate statistics for each pollutant
        pollutant_stats = {}
        for param, data in pollutants.items():
            values = data["values"]
            pollutant_stats[param] = {
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "avg": round(sum(values) / len(values), 3),
                "unit": data["unit"],
                "measurements_count": data["measurements_count"]
            }
        
        return {
            "provider": "openaq",
            "success": True,
            "data_source": "OpenAQ - Open Air Quality Data",
            "query_date": date,
            "query_period": f"{date_from} to {date_to}",
            "search_radius_km": 25,
            "total_measurements": len(results),
            "unique_locations": len(locations),
            "pollutants": pollutant_stats,
            "coverage": "Global - community and governmental monitoring stations",
            "api_url": f"{base_url}?{requests.compat.urlencode(params)}"
        }
        
    except Exception as e:
        return {
            "provider": "openaq",
            "success": False,
            "error": str(e)
        }


def get_air_quality_openweather(lat: float, lon: float) -> Dict[str, Any]:
    """OpenWeatherMap Air Quality API for global air pollution data."""
    import requests
    import os
    from pathlib import Path
    
    # Load local environment variables
    env_file = Path(__file__).parent.parent.parent / "local" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        return {
            "provider": "openweather_aq",
            "success": False,
            "error": "OPENWEATHER_API_KEY not set"
        }
    
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "list" in data and len(data["list"]) > 0:
            air_data = data["list"][0]
            main = air_data.get("main", {})
            components = air_data.get("components", {})
            
            return {
                "provider": "openweather_aq",
                "success": True,
                "data_source": "OpenWeatherMap Air Quality",
                "aqi": main.get("aqi"),
                "aqi_description": {
                    1: "Good",
                    2: "Fair", 
                    3: "Moderate",
                    4: "Poor",
                    5: "Very Poor"
                }.get(main.get("aqi"), "Unknown"),
                "pollutants": {
                    "co": components.get("co"),      # Carbon monoxide (μg/m³)
                    "no": components.get("no"),      # Nitrogen monoxide (μg/m³)
                    "no2": components.get("no2"),    # Nitrogen dioxide (μg/m³)
                    "o3": components.get("o3"),      # Ozone (μg/m³)
                    "so2": components.get("so2"),    # Sulphur dioxide (μg/m³)
                    "pm2_5": components.get("pm2_5"), # PM2.5 (μg/m³)
                    "pm10": components.get("pm10"),   # PM10 (μg/m³)
                    "nh3": components.get("nh3")      # Ammonia (μg/m³)
                },
                "units": "μg/m³",
                "datetime_utc": air_data.get("dt"),
                "coverage": "Global"
            }
        else:
            return {
                "provider": "openweather_aq",
                "success": False,
                "error": "No air quality data available"
            }
            
    except Exception as e:
        return {
            "provider": "openweather_aq",
            "success": False,
            "error": str(e)
        }


def get_air_quality_multi(lat: float, lon: float, date: Optional[str] = None) -> Dict[str, Any]:
    """
    Multi-provider air quality that queries ALL providers and saves all responses.
    
    Queries: Google Air Quality + EPA AirNow (US only) + OpenWeatherMap
    Returns comprehensive air pollution data for microbiome research.
    
    Args:
        lat: Latitude
        lon: Longitude  
        date: Optional collection date in YYYY-MM-DD format for historical data
    """
    providers = [
        ("google_air_quality", lambda lat, lon: get_air_quality_google(lat, lon, date)),
        ("openaq", lambda lat, lon: get_air_quality_openaq(lat, lon, date)),
        ("openweather_aq", get_air_quality_openweather)
    ]
    
    # Add EPA AirNow for US locations only
    if -170 <= lon <= -60 and 15 <= lat <= 75:  # US bounds
        providers.insert(1, ("epa_airnow", get_air_quality_epa_airnow))
    
    all_results = {}
    successful_results = []
    
    # Query ALL providers
    for provider_name, provider_func in providers:
        try:
            result = provider_func(lat, lon)
            all_results[provider_name] = result
            
            if result.get("success"):
                successful_results.append((provider_name, result))
                
        except Exception as e:
            all_results[provider_name] = {
                "provider": provider_name,
                "success": False,
                "error": str(e)
            }
    
    # Determine primary result (prefer EPA for US, Google for global)
    primary_result = None
    primary_provider = None
    
    if successful_results:
        # Prefer EPA AirNow for US locations (highest quality government data)
        for provider_name, result in successful_results:
            if provider_name == "epa_airnow" and -170 <= lon <= -60 and 15 <= lat <= 75:
                primary_result = result
                primary_provider = provider_name
                break
            elif provider_name == "google_air_quality":
                primary_result = result
                primary_provider = provider_name
        
        # If no preferred result, use first successful
        if not primary_result:
            primary_provider, primary_result = successful_results[0]
    
    # Build comprehensive response
    if primary_result:
        # Return primary result with all provider data included
        response = primary_result.copy()
        response["all_providers"] = all_results
        response["providers_queried"] = list(all_results.keys())
        response["successful_providers"] = [p for p, r in successful_results]
        response["primary_provider"] = primary_provider
        return response
    else:
        # All providers failed
        return {
            "error": "All air quality providers failed",
            "success": False,
            "all_providers": all_results,
            "providers_queried": list(all_results.keys()),
            "successful_providers": [],
            "provider_errors": {k: v.get("error", "Unknown error") for k, v in all_results.items()}
        }


def get_land_cover_multi(lat: float, lon: float, year: int = 2021) -> Dict[str, Any]:
    """
    Multi-provider land cover that queries ALL providers and saves all responses.
    
    Queries all land cover services: ESA WorldCover + NLCD (US only) + Local files
    Returns the best result plus all provider responses for comparison.
    """
    providers = [
        ("esa_worldcover", lambda lat, lon: get_land_cover_esa_worldcover(lat, lon, year)),
    ]
    
    # Add NLCD for US locations only  
    if -170 <= lon <= -60 and 15 <= lat <= 75:  # US bounds
        providers.append(("nlcd", lambda lat, lon: get_land_cover_nlcd(lat, lon, year)))
    
    all_results = {}
    successful_results = []
    
    # Query ALL providers
    for provider_name, provider_func in providers:
        try:
            result = provider_func(lat, lon)
            all_results[provider_name] = result
            
            if result.get("success"):
                # Standardize result format
                standardized_result = {
                    "provider": provider_name,
                    "success": True,
                    "land_cover_class": result.get("land_cover_class"),
                    "class_name": result.get("class_name"),
                    "envo_terms": result.get("envo_terms", []),
                    "confidence": result.get("confidence"),
                    "source": result.get("source", "api"),
                    "resolution": result.get("resolution"),
                    "year": year,
                    "data_source": result.get("data_source")
                }
                
                successful_results.append((provider_name, standardized_result))
                
        except Exception as e:
            all_results[provider_name] = {
                "provider": provider_name,
                "success": False,
                "error": str(e)
            }
    
    # Determine primary result (prefer NLCD for US, otherwise ESA WorldCover)
    primary_result = None
    primary_provider = None
    
    if successful_results:
        # Prefer NLCD for US locations, ESA WorldCover for global
        for provider_name, result in successful_results:
            if provider_name == "nlcd" and -170 <= lon <= -60 and 15 <= lat <= 75:
                primary_result = result
                primary_provider = provider_name
                break
            elif provider_name == "esa_worldcover":
                primary_result = result
                primary_provider = provider_name
        
        # If no preferred result, use first successful
        if not primary_result:
            primary_provider, primary_result = successful_results[0]
    
    # Build comprehensive response
    if primary_result:
        # Return primary result with all provider data included
        response = primary_result.copy()
        response["all_providers"] = all_results
        response["providers_queried"] = list(all_results.keys())
        response["successful_providers"] = [p for p, r in successful_results]
        response["primary_provider"] = primary_provider
        return response
    else:
        # All providers failed
        return {
            "error": "All land cover providers failed",
            "success": False,
            "all_providers": all_results,
            "providers_queried": list(all_results.keys()),
            "successful_providers": [],
            "provider_errors": {k: v.get("error", "Unknown error") for k, v in all_results.items()}
        }


def get_reverse_geocoding_multi(lat: float, lon: float) -> Dict[str, Any]:
    """
    Multi-provider reverse geocoding that queries ALL providers and saves all responses.
    
    Queries all reverse geocoding services: Nominatim + Google
    Returns the best result plus all provider responses for comparison.
    """
    providers = [
        ("nominatim", get_reverse_geocoding_nominatim),
        ("google", get_reverse_geocoding_google)
    ]
    
    all_results = {}
    successful_results = []
    
    # Query ALL providers
    for provider_name, geocode_func in providers:
        try:
            result = geocode_func(lat, lon)
            
            # Handle cached result format for Nominatim (data nested under "data" key)
            if "data" in result:
                result = result["data"]
            
            all_results[provider_name] = result
            
            if result.get("success"):
                # Standardize result format
                standardized_result = {
                    "provider": result.get("provider", provider_name),
                    "country": result.get("country"),
                    "country_code": result.get("country_code"), 
                    "state_province": result.get("state"),
                    "county_region": result.get("county"),
                    "city": result.get("city"),
                    "postcode": result.get("postcode"),
                    "display_name": result.get("display_name"),
                    "place_type": result.get("place_type"),
                    "coordinates": {
                        "lat": result.get("latitude", lat),
                        "lon": result.get("longitude", lon)
                    },
                    "distance_km": round(haversine_km(lat, lon, 
                                                  result.get("latitude", lat),
                                                  result.get("longitude", lon)), 3),
                    "data_source": result.get("data_source"),
                    "success": True
                }
                
                successful_results.append((provider_name, standardized_result))
                
        except Exception as e:
            all_results[provider_name] = {
                "provider": provider_name,
                "error": str(e),
                "success": False
            }
    
    # Determine primary result (prefer Google if successful, otherwise first successful)
    primary_result = None
    primary_provider = None
    
    if successful_results:
        # Prefer Google, then others in order  
        for provider_name, result in successful_results:
            if provider_name == "google":
                primary_result = result
                primary_provider = provider_name
                break
        
        # If no Google result, use first successful
        if not primary_result:
            primary_provider, primary_result = successful_results[0]
    
    # Build comprehensive response
    if primary_result:
        # Return primary result with all provider data included
        response = primary_result.copy()
        response["all_providers"] = all_results
        response["providers_queried"] = list(all_results.keys())
        response["successful_providers"] = [p for p, r in successful_results]
        response["primary_provider"] = primary_provider
        return response
    else:
        # All providers failed
        return {
            "error": "All reverse geocoding providers failed",
            "success": False,
            "all_providers": all_results,
            "providers_queried": list(all_results.keys()),
            "successful_providers": [],
            "provider_errors": {k: v.get("error", "Unknown error") for k, v in all_results.items()}
        }


def get_forward_geocoding_multi(place_name: str) -> Dict[str, Any]:
    """
    Multi-provider forward geocoding that queries ALL providers and saves all responses.
    
    Queries all geocoding services: Google + Nominatim
    Returns the best result plus all provider responses for comparison.
    """
    providers = [
        ("google", get_forward_geocoding_google),
        ("nominatim", get_forward_geocoding_nominatim)
    ]
    
    all_results = {}
    successful_results = []
    
    # Query ALL providers
    for provider_name, geocode_func in providers:
        try:
            result = geocode_func(place_name)
            all_results[provider_name] = result
            
            if result.get("success"):
                successful_results.append((provider_name, result))
                
        except Exception as e:
            all_results[provider_name] = {
                "provider": provider_name,
                "error": f"Provider {provider_name} failed: {str(e)}",
                "success": False,
                "query": place_name
            }
    
    # Determine primary result (prefer Google if successful, otherwise first successful)
    primary_result = None
    primary_provider = None
    
    if successful_results:
        # Prefer Google, then others in order
        for provider_name, result in successful_results:
            if provider_name == "google":
                primary_result = result
                primary_provider = provider_name
                break
        
        # If no Google result, use first successful
        if not primary_result:
            primary_provider, primary_result = successful_results[0]
    
    # Build comprehensive response
    if primary_result:
        # Return primary result with all provider data included
        response = primary_result.copy()
        response["all_providers"] = all_results
        response["providers_queried"] = list(all_results.keys())
        response["successful_providers"] = [p for p, r in successful_results]
        response["primary_provider"] = primary_provider
        return response
    else:
        # All providers failed
        return {
            "provider": "none",
            "error": "All geocoding providers failed",
            "success": False,
            "query": place_name,
            "all_providers": all_results,
            "providers_queried": list(all_results.keys()),
            "successful_providers": [],
            "provider_errors": {k: v.get("error", "Unknown error") for k, v in all_results.items()}
        }


@cli.command()
@click.option('--lat', required=True, type=float, help='Latitude coordinate')
@click.option('--lon', required=True, type=float, help='Longitude coordinate')
@click.option('--place', help='Place name to forward geocode (optional)')
@click.option('--id', 'biosample_id', default='test_sample', help='Biosample identifier')
@click.option('--date', help='Collection date (YYYY-MM-DD)')
@click.option('--output', type=click.Path(), help='Output JSON file path')
@click.option('--config', 'config_path', type=click.Path(exists=True), help='Custom config file path')
@click.option('--enable-crosswalks', is_flag=True, help='Enable ENVO crosswalk mappings')
@click.option('--use-cache/--no-use-cache', default=True, help='Use cached results (default: enabled)')
@click.option('--save-cache/--no-save-cache', default=True, help='Save results to cache (default: enabled)')
@click.option('--verbose', is_flag=True, help='Verbose output')
@click.option('--pretty', is_flag=True, help='Pretty-print JSON output')
def enrich(lat, lon, place, biosample_id, date, output, config_path, enable_crosswalks, use_cache, save_cache, verbose, pretty):
    """
    Run comprehensive geospatial enrichment for a single biosample.
    
    Example:
        enrich-geo enrich --lat 44.428 --lon -110.5885 --date 2021-08-20 --output results.json --verbose --pretty
    """
    if verbose:
        click.echo(f"🌍 Running unified geospatial enrichment...")
        click.echo(f"📍 Location: {lat:.4f}, {lon:.4f}")
        click.echo(f"🆔 Biosample ID: {biosample_id}")
        if date:
            click.echo(f"📅 Collection date: {date}")
    
    try:
        # Load configuration
        config = load_enrichment_config(Path(config_path)) if config_path else load_enrichment_config()
        
        if verbose:
            click.echo(f"⚙️  Configuration loaded: version {config.get('version', 'unknown')}")
        
        # Run enrichment
        result = enrich_biosample_unified(
            biosample_id=biosample_id,
            lat=lat,
            lon=lon,
            collection_date=date,
            place=place,
            config=config,
            use_cache=use_cache,
            save_cache_enabled=save_cache
        )
        
        # Output results
        if pretty:
            json_output = json.dumps(result, indent=2, default=str)
        else:
            json_output = json.dumps(result, default=str)
        
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(json_output)
            
            if verbose:
                click.echo(f"✅ Results saved to: {output}")
                click.echo(f"📊 Success rate: {result.get('enrichment_success_rate', 0):.1%}")
                
                if result.get("enrichment_errors"):
                    click.echo(f"⚠️  Errors: {len(result['enrichment_errors'])}")
        else:
            click.echo(json_output)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--input', 'input_file', required=True, type=click.Path(exists=True), help='Input JSON file with biosample data')
@click.option('--output', type=click.Path(), help='Output JSON file path')
@click.option('--config', 'config_path', type=click.Path(exists=True), help='Custom config file path')
@click.option('--max-samples', default=100, help='Maximum number of samples to process')
@click.option('--verbose', is_flag=True, help='Verbose output')
@click.option('--pretty', is_flag=True, help='Pretty-print JSON output')
def batch(input_file, output, config_path, max_samples, verbose, pretty):
    """
    Run batch enrichment for multiple biosamples from input file.
    
    Input file should contain JSON array with objects having: id, lat, lon, collection_date (optional)
    
    Example:
        enrich-geo batch --input biosamples.json --output enriched.json --max-samples 50 --verbose
    """
    if verbose:
        click.echo(f"🧬 Running batch geospatial enrichment...")
        click.echo(f"📁 Input file: {input_file}")
        click.echo(f"📊 Max samples: {max_samples}")
    
    try:
        # Load input data
        with open(input_file) as f:
            biosamples = json.load(f)
        
        if not isinstance(biosamples, list):
            raise ValueError("Input file must contain a JSON array of biosample objects")
        
        # Limit samples
        if len(biosamples) > max_samples:
            biosamples = biosamples[:max_samples]
            if verbose:
                click.echo(f"⚠️  Limited to first {max_samples} samples")
        
        # Load configuration
        config = load_enrichment_config(Path(config_path)) if config_path else load_enrichment_config()
        
        if verbose:
            click.echo(f"⚙️  Configuration loaded: version {config.get('version', 'unknown')}")
            click.echo(f"🚀 Processing {len(biosamples)} biosamples...")
        
        # Run batch enrichment
        results = batch_enrich_biosamples(biosamples, config)
        
        # Calculate summary statistics
        success_rates = [r.get('enrichment_success_rate', 0) for r in results]
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        if verbose:
            click.echo(f"✅ Batch enrichment completed")
            click.echo(f"📊 Average success rate: {avg_success_rate:.1%}")
            click.echo(f"📈 Success rate range: {min(success_rates):.1%} - {max(success_rates):.1%}")
        
        # Output results
        if pretty:
            json_output = json.dumps(results, indent=2, default=str)
        else:
            json_output = json.dumps(results, default=str)
        
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(json_output)
            
            if verbose:
                click.echo(f"💾 Results saved to: {output}")
        else:
            click.echo(json_output)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--create-sample', type=click.Path(), help='Create sample config file')
@click.option('--validate', type=click.Path(exists=True), help='Validate config file')
def config(show, create_sample, validate):
    """
    Configuration management for the enrichment system.
    
    Examples:
        enrich-geo config --show
        enrich-geo config --create-sample my_config.yml
        enrich-geo config --validate config/enrichment.yml
    """
    if show:
        config = load_enrichment_config()
        click.echo("📋 Current Configuration:")
        click.echo("=" * 50)
        click.echo(json.dumps(config, indent=2, default=str))
        
    elif create_sample:
        sample_config = _get_default_config()
        create_sample_path = Path(create_sample)
        
        if create_sample_path.suffix.lower() in ['.yml', '.yaml']:
            try:
                import yaml
                content = yaml.dump(sample_config, default_flow_style=False, sort_keys=False)
            except ImportError:
                click.echo("❌ PyYAML not available, creating JSON instead", err=True)
                create_sample_path = create_sample_path.with_suffix('.json')
                content = json.dumps(sample_config, indent=2)
        else:
            content = json.dumps(sample_config, indent=2)
        
        create_sample_path.parent.mkdir(parents=True, exist_ok=True)
        with open(create_sample_path, 'w') as f:
            f.write(content)
        
        click.echo(f"✅ Sample configuration created: {create_sample_path}")
        
    elif validate:
        try:
            config = load_enrichment_config(Path(validate))
            click.echo(f"✅ Configuration file is valid: {validate}")
            click.echo(f"📊 Version: {config.get('version', 'unknown')}")
            click.echo(f"📁 Datasets: {len(config.get('datasets', {}))}")
            click.echo(f"🔧 Providers: {len(config.get('providers', {}))}")
        except Exception as e:
            click.echo(f"❌ Configuration validation failed: {e}", err=True)
            raise click.Abort()
    else:
        click.echo("Please specify an action: --show, --create-sample, or --validate")


def main():
    """Entry point for the CLI application."""
    cli()


if __name__ == "__main__":
    main()
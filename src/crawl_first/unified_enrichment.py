#!/usr/bin/env python3
"""
Unified biosample enrichment entrypoint.

Combines all geospatial lookup and normalization capabilities from the gold-and-nmdc 
directory into a single comprehensive enrichment pipeline. Runs all available APIs
and normalization functions, producing a complete JSON output with raw API responses
and normalized linked data.

This script unifies the successful approach from the root project's Makefile with
the working alternative APIs developed in gold-and-nmdc.
"""

import json
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

import click

# Import all enrichment modules (handle both relative and direct imports)
try:
    # Try relative imports first (when run as module)
    from .geospatial_enrichment import enrich_location as main_enrich_location
    from .crosswalk_loader import CrosswalkManager
    from .crosswalk_loader import (
        normalize_usda_texture, 
        normalize_osm_natural,
        CROSSWALKS
    )
    from .generate_mappings import ENVOMappingGenerator
    CROSSWALKS_AVAILABLE = True
except ImportError:
    # Fall back to absolute imports (when run directly)
    try:
        from geospatial_enrichment import enrich_location as main_enrich_location
        from crosswalk_loader import CrosswalkManager
        from crosswalk_loader import (
            normalize_usda_texture, 
            normalize_osm_natural,
            CROSSWALKS
        )
        from generate_mappings import ENVOMappingGenerator
        CROSSWALKS_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Enrichment modules not available: {e}")
        main_enrich_location = None
        CrosswalkManager = None
        ENVOMappingGenerator = None
        CROSSWALKS_AVAILABLE = False


def get_project_version() -> str:
    """Get the project version from pyproject.toml."""
    try:
        # Look for pyproject.toml in the project root
        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            with open(pyproject_path, 'rb') as f:
                config = tomllib.load(f)
                return config.get("project", {}).get("version", "unknown")
        else:
            return "unknown"
    except Exception:
        return "unknown"


def get_iso_timestamp() -> str:
    """Get consistent ISO 8601 UTC timestamp with Z suffix."""
    return datetime.utcnow().isoformat() + "Z"


# API base URLs for tracking attempts
API_BASE_URLS = {
    # Primary enrichment APIs
    "elevation": "https://api.open-elevation.com",
    "elevation_usgs": "https://epqs.nationalmap.gov", 
    "weather": "https://archive-api.open-meteo.com",
    "nominatim": "https://nominatim.openstreetmap.org",
    "overpass": "https://overpass-api.de",
    "soil_nrcs": "https://sdmdataaccess.nrcs.usda.gov",
    "soilgrids_rest": "https://rest.isric.org", 
    "soilgrids_wcs": "https://maps.isric.org",
    "ecoregions": "https://ecoregions.appspot.com",
    
    # Alternative enrichment APIs
    "soilgrids_classification": "https://rest.isric.org",
    "worldcover": "https://services.terrascope.be",
    "nlcd": "https://www.mrlc.gov"
}


class UnifiedBiosampleEnricher:
    """Unified biosample enrichment using all available data sources and normalizations."""
    
    def __init__(self, enable_crosswalks: bool = True, verbose: bool = False):
        """
        Initialize unified enricher.
        
        Args:
            enable_crosswalks: Whether to use crosswalk normalization
            verbose: Enable verbose logging
        """
        self.enable_crosswalks = enable_crosswalks and CROSSWALKS_AVAILABLE
        self.verbose = verbose
        self.mapping_generator = ENVOMappingGenerator() if ENVOMappingGenerator else None
        
        # Check crosswalk mappings if available
        if self.enable_crosswalks:
            try:
                if CROSSWALKS and len(CROSSWALKS) > 0:
                    if self.verbose:
                        loaded_crosswalks = [k for k, v in CROSSWALKS.items() if v]
                        print(f"✓ Crosswalk mappings loaded: {', '.join(loaded_crosswalks)}")
                else:
                    if self.verbose:
                        print("⚠ No crosswalk mappings found")
                    self.enable_crosswalks = False
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Crosswalk loading failed: {e}")
                self.enable_crosswalks = False
    
    def enrich_coordinates(self, lat: float, lon: float, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive coordinate enrichment using all available data sources.
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)  
            date: Collection date in YYYY-MM-DD format (optional)
            
        Returns:
            Complete enrichment results with raw API responses and normalizations
        """
        if self.verbose:
            print(f"🌍 Enriching coordinates: {lat}, {lon}")
            if date:
                print(f"📅 Collection date: {date}")
        
        # Initialize comprehensive result structure
        enrichment_result = {
            "input_coordinates": {
                "latitude": lat,
                "longitude": lon,
                "collection_date": date
            },
            "enrichment_metadata": {
                "timestamp": get_iso_timestamp(),
                "unified_enricher_version": get_project_version(),
                "apis_attempted": [],
                "apis_successful": [],
                "crosswalks_enabled": self.enable_crosswalks
            },
            "primary_enrichment": {},
            "comprehensive_analysis": {},
            "normalization_results": {},
            "envo_mappings": {},
            "data_quality_assessment": {},
            "unified_summary": {}
        }
        
        # 1. Primary enrichment using main geospatial module
        if self.verbose:
            print("🔍 Running primary geospatial enrichment...")
        
        try:
            primary_result = main_enrich_location(lat, lon, date)
            enrichment_result["primary_enrichment"] = primary_result
            enrichment_result["enrichment_metadata"]["apis_attempted"].extend([
                API_BASE_URLS["elevation"],
                API_BASE_URLS["elevation_usgs"], 
                API_BASE_URLS["weather"],
                API_BASE_URLS["nominatim"],
                API_BASE_URLS["overpass"],
                API_BASE_URLS["soil_nrcs"],
                API_BASE_URLS["soilgrids_rest"],
                API_BASE_URLS["soilgrids_wcs"],
                API_BASE_URLS["ecoregions"]
            ])
            
            # Track successful APIs from primary enrichment by checking actual success flags
            self._track_primary_api_success(primary_result, enrichment_result["enrichment_metadata"]["apis_successful"])
            
            if self.verbose:
                success_count = primary_result.get("enrichment_summary", {}).get("total_enrichments", 0)
                print(f"  ✓ Primary enrichment: {success_count}/8 successful")
                
        except Exception as e:
            enrichment_result["primary_enrichment"] = {"error": str(e)}
            if self.verbose:
                print(f"  ✗ Primary enrichment failed: {e}")
        
        # 2. Alternative APIs as fallbacks and supplements
        if self.verbose:
            print("🔬 Running alternative APIs as fallbacks and supplements...")
        
        try:
            alt_result = get_comprehensive_site_analysis(lat, lon, date)
            
            # Integrate alternative data into primary structure instead of separate section
            self._integrate_alternative_data(alt_result, enrichment_result["primary_enrichment"])
            
            enrichment_result["enrichment_metadata"]["apis_attempted"].extend([
                API_BASE_URLS["soilgrids_classification"],
                API_BASE_URLS["worldcover"], 
                API_BASE_URLS["nlcd"]
            ])
            
            # Track successful alternative APIs
            self._track_alternative_api_success(alt_result, enrichment_result["enrichment_metadata"]["apis_successful"])
                
            if self.verbose:
                envo_terms = len(alt_result.get("envo_terms_summary", []))
                print(f"  ✓ Alternative APIs: {envo_terms} additional ENVO terms found")
                
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Alternative APIs failed: {e}")
        
        # 3. Apply crosswalk normalization if enabled
        if self.enable_crosswalks:
            if self.verbose:
                print("🔗 Applying crosswalk normalization...")
            
            enrichment_result["normalization_results"] = self._apply_crosswalk_normalization(
                enrichment_result["primary_enrichment"]
            )
        
        # 4. Generate comprehensive ENVO mappings
        if self.verbose:
            print("🏷️ Generating ENVO mappings...")
        
        enrichment_result["envo_mappings"] = self._generate_envo_mappings(
            enrichment_result["primary_enrichment"]
        )
        
        # 5. Data quality assessment
        if self.verbose:
            print("📊 Assessing data quality...")
        
        enrichment_result["data_quality_assessment"] = self._assess_data_quality(
            enrichment_result["primary_enrichment"]
        )
        
        # 6. Create unified summary
        enrichment_result["unified_summary"] = self._create_unified_summary(enrichment_result)
        
        if self.verbose:
            total_apis = len(enrichment_result["enrichment_metadata"]["apis_successful"])
            print(f"🎯 Enrichment complete: {total_apis} successful API calls")
        
        return enrichment_result
    
    def _track_primary_api_success(self, primary_result: Dict[str, Any], apis_successful: List[str]) -> None:
        """Track successful primary APIs by checking individual success flags."""
        # Helper function to extract data from cache wrapper
        def extract_cached_data(data_obj):
            if isinstance(data_obj, dict):
                if "data" in data_obj and "cached_at" in data_obj:
                    return data_obj["data"]
                return data_obj
            return {}
        
        # Check elevation APIs
        elevation_data = extract_cached_data(primary_result.get("elevation", {}))
        if elevation_data.get("success"):
            apis_successful.append(API_BASE_URLS["elevation"])
        
        # Check weather API
        weather_data = extract_cached_data(primary_result.get("weather", {}))
        if weather_data.get("success"):
            apis_successful.append(API_BASE_URLS["weather"])
        
        # Check location context (Nominatim)
        location_data = extract_cached_data(primary_result.get("location_context", {}))
        if location_data.get("success"):
            apis_successful.append(API_BASE_URLS["nominatim"])
        
        # Check nearby features (Overpass)
        nearby_data = extract_cached_data(primary_result.get("nearby_features", {}))
        if nearby_data.get("success"):
            apis_successful.append(API_BASE_URLS["overpass"])
        
        # Check soil classification
        soil_class_data = extract_cached_data(primary_result.get("soil_classification", {}))
        if soil_class_data.get("success"):
            # Could be either NRCS or SoilGrids - check data source
            data_source = soil_class_data.get("data_source", "")
            if "NRCS" in data_source or "USDA" in data_source:
                apis_successful.append(API_BASE_URLS["soil_nrcs"])
            elif "SoilGrids" in data_source:
                apis_successful.append(API_BASE_URLS["soilgrids_rest"])
        
        # Check soil properties (SoilGrids WCS)
        soil_props = primary_result.get("soil_properties", {})
        if soil_props.get("success"):
            apis_successful.append(API_BASE_URLS["soilgrids_wcs"])
        
        # Check ecoregion
        ecoregion_data = primary_result.get("ecoregion", {})
        if ecoregion_data.get("success"):
            apis_successful.append(API_BASE_URLS["ecoregions"])
    
    def _integrate_alternative_data(self, alt_result: Dict[str, Any], primary_data: Dict[str, Any]) -> None:
        """Integrate alternative API data into primary enrichment structure."""
        # Add alternative soil classification if primary is missing or failed
        alt_soil = alt_result.get("soil_analysis", {})
        if alt_soil.get("soil_type") and not primary_data.get("soil_classification", {}).get("success"):
            primary_data["alternative_soil_classification"] = {
                "soil_type": alt_soil.get("soil_type"),
                "confidence": alt_soil.get("confidence"),
                "data_source": "Alternative SoilGrids Classification API",
                "success": True
            }
        
        # Add land cover data if available
        alt_land_cover = alt_result.get("land_cover_analysis", {})
        current_cover = alt_land_cover.get("current", {})
        if current_cover.get("land_cover_class"):
            primary_data["land_cover"] = {
                "current": current_cover,
                "historical": alt_land_cover.get("historical", {}),
                "data_source": "Alternative Land Cover API",
                "success": True
            }
        
        # Add temporal analysis if available
        temporal = alt_result.get("temporal_analysis", {})
        if temporal:
            primary_data["temporal_analysis"] = temporal
        
        # Merge ENVO terms into existing enrichment summary
        alt_envo_terms = alt_result.get("envo_terms_summary", [])
        if alt_envo_terms:
            if "envo_terms" not in primary_data:
                primary_data["envo_terms"] = []
            primary_data["envo_terms"].extend(alt_envo_terms)
            
            # Update enrichment summary to include ENVO mapping success
            if "enrichment_summary" in primary_data:
                primary_data["enrichment_summary"]["alternative_data_integrated"] = True
                primary_data["enrichment_summary"]["envo_terms_found"] = len(primary_data["envo_terms"])
    
    def _track_alternative_api_success(self, alt_result: Dict[str, Any], apis_successful: List[str]) -> None:
        """Track successful alternative APIs by checking individual success flags."""
        # Check soil analysis (SoilGrids Classification)
        soil_analysis = alt_result.get("soil_analysis", {})
        if soil_analysis.get("soil_type"):
            apis_successful.append(API_BASE_URLS["soilgrids_classification"])
        
        # Check land cover analysis
        land_cover = alt_result.get("land_cover_analysis", {})
        current_cover = land_cover.get("current", {})
        if current_cover.get("land_cover_class"):
            # Check data source to determine which API was successful
            # For now, assume WorldCover as it's more commonly available
            apis_successful.append(API_BASE_URLS["worldcover"])
    
    def _apply_crosswalk_normalization(self, primary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply crosswalk-based normalization to soil and land cover data."""
        normalization_results = {
            "soil_texture_normalization": {},
            "natural_features_normalization": {},
            "crosswalk_metadata": {
                "applied_crosswalks": [],
                "normalization_timestamp": get_iso_timestamp()
            }
        }
        
        # Normalize soil texture classification
        soil_props = primary_data.get("soil_properties", {}).get("soil_properties", {})
        texture_class = soil_props.get("texture_classification", {})
        
        if texture_class.get("usda_texture_class") and normalize_usda_texture:
            try:
                normalized = normalize_usda_texture(
                    texture_class["usda_texture_class"],
                    texture_class.get("sand_percent"),
                    texture_class.get("clay_percent"),
                    texture_class.get("silt_percent")
                )
                normalization_results["soil_texture_normalization"] = normalized
                normalization_results["crosswalk_metadata"]["applied_crosswalks"].append("usda_texture_to_envo")
            except Exception as e:
                normalization_results["soil_texture_normalization"] = {"error": str(e)}
        
        # Normalize natural features from OSM (improved to handle cached data)
        nearby_features_data = primary_data.get("nearby_features", {})
        
        # Extract from cache wrapper if needed
        if "data" in nearby_features_data and "cached_at" in nearby_features_data:
            nearby_features_data = nearby_features_data["data"]
        
        features = nearby_features_data.get("features", [])
        normalized_features = []
        
        if features and normalize_osm_natural:
            for feature in features:
                natural_tag = feature.get("natural")
                if natural_tag:
                    try:
                        normalized = normalize_osm_natural(natural_tag)
                        if normalized and normalized.get("linked_data", {}).get("success"):
                            normalized_features.append({
                                "original_feature": {
                                    "osm_id": feature.get("id"),
                                    "natural": natural_tag,
                                    "name": feature.get("name"),
                                    "water": feature.get("water")
                                },
                                "normalized": normalized
                            })
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Natural feature normalization failed for {natural_tag}: {e}")
        
        if normalized_features:
            normalization_results["natural_features_normalization"] = {
                "features": normalized_features,
                "total_features_processed": len(features),
                "features_normalized": len(normalized_features)
            }
            normalization_results["crosswalk_metadata"]["applied_crosswalks"].append("osm_natural_to_envo")
        else:
            normalization_results["natural_features_normalization"] = {
                "features": [],
                "total_features_processed": len(features),
                "features_normalized": 0,
                "note": "No natural features found or normalization unavailable"
            }
        
        return normalization_results
    
    def _generate_envo_mappings(self, primary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive ENVO term mappings from all data sources."""
        envo_mappings = {
            "soil_envo_terms": [],
            "land_cover_envo_terms": [],
            "ecosystem_envo_terms": [],
            "combined_envo_terms": [],
            "mapping_metadata": {
                "generation_timestamp": get_iso_timestamp(),
                "sources_used": []
            }
        }
        
        # Extract ENVO terms that were integrated from alternative APIs
        integrated_envo_terms = primary_data.get("envo_terms", [])
        if integrated_envo_terms:
            envo_mappings["combined_envo_terms"].extend(integrated_envo_terms)
            envo_mappings["mapping_metadata"]["sources_used"].append("integrated_alternative_apis")
        
        # Extract ecosystem classification
        ecosystem_class = primary_data.get("ecosystem_classification", {})
        if ecosystem_class.get("ecosystem_path"):
            envo_mappings["ecosystem_envo_terms"].append({
                "ecosystem_path": ecosystem_class["ecosystem_path"],
                "confidence": ecosystem_class.get("confidence"),
                "reasoning": ecosystem_class.get("reasoning", [])
            })
            envo_mappings["mapping_metadata"]["sources_used"].append("ecosystem_classifier")
        
        # Remove duplicate ENVO terms
        unique_terms = []
        seen_ids = set()
        for term in envo_mappings["combined_envo_terms"]:
            term_id = term.get("id")
            if term_id and term_id not in seen_ids:
                unique_terms.append(term)
                seen_ids.add(term_id)
        
        envo_mappings["combined_envo_terms"] = unique_terms
        
        return envo_mappings
    
    def _assess_data_quality(self, primary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality and completeness of enrichment data."""
        quality_assessment = {
            "completeness_score": 0.0,
            "data_source_coverage": {},
            "geographic_coverage_assessment": {},
            "temporal_coverage_assessment": {},
            "quality_flags": [],
            "assessment_metadata": {
                "timestamp": get_iso_timestamp(),
                "version": get_project_version()
            }
        }
        
        # Assess primary data completeness
        primary_success = primary_data.get("enrichment_summary", {}).get("total_enrichments", 0)
        max_primary = 8  # Total possible primary enrichments
        primary_completeness = primary_success / max_primary
        
        # Assess integrated alternative data completeness
        alt_soil = 1 if primary_data.get("alternative_soil_classification", {}).get("success") else 0
        alt_land_cover = 1 if primary_data.get("land_cover", {}).get("success") else 0
        alt_completeness = (alt_soil + alt_land_cover) / 2
        
        # Overall completeness score (weighted average)
        quality_assessment["completeness_score"] = (primary_completeness * 0.7 + alt_completeness * 0.3)
        
        quality_assessment["data_source_coverage"] = {
            "primary_apis": f"{primary_success}/{max_primary}",
            "alternative_apis": f"{alt_soil + alt_land_cover}/2",
            "primary_completeness": primary_completeness,
            "alternative_completeness": alt_completeness
        }
        
        # Geographic coverage assessment - use coordinates from primary data
        coords = primary_data.get("original_coordinates", {})
        lat = coords.get("latitude", 0)
        lon = coords.get("longitude", 0)
        
        quality_assessment["geographic_coverage_assessment"] = {
            "coordinates_valid": -90 <= lat <= 90 and -180 <= lon <= 180,
            "us_coverage": -170 <= lon <= -60 and 15 <= lat <= 75,
            "global_coverage": True,  # Our APIs provide global coverage
            "high_resolution_available": True  # 250m SoilGrids, 10m WorldCover
        }
        
        # Temporal coverage assessment
        collection_date = primary_data.get("collection_date")
        if collection_date:
            try:
                date_obj = datetime.strptime(collection_date, "%Y-%m-%d")
                quality_assessment["temporal_coverage_assessment"] = {
                    "collection_date_provided": True,
                    "historical_weather_available": date_obj.year >= 1940,
                    "historical_land_cover_available": date_obj.year >= 2000,
                    "date_within_modern_era": date_obj.year >= 1990
                }
            except ValueError:
                quality_assessment["temporal_coverage_assessment"] = {
                    "collection_date_provided": False,
                    "date_format_valid": False
                }
        else:
            quality_assessment["temporal_coverage_assessment"] = {
                "collection_date_provided": False
            }
        
        # Generate quality flags
        if quality_assessment["completeness_score"] < 0.5:
            quality_assessment["quality_flags"].append("LOW_COMPLETENESS")
        
        if not quality_assessment["geographic_coverage_assessment"]["coordinates_valid"]:
            quality_assessment["quality_flags"].append("INVALID_COORDINATES")
        
        if not quality_assessment["temporal_coverage_assessment"].get("collection_date_provided", False):
            quality_assessment["quality_flags"].append("NO_COLLECTION_DATE")
        
        return quality_assessment
    
    def _create_unified_summary(self, enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a unified summary of all enrichment results."""
        summary = {
            "enrichment_success": True,
            "total_apis_called": len(enrichment_data["enrichment_metadata"]["apis_attempted"]),
            "successful_apis": len(enrichment_data["enrichment_metadata"]["apis_successful"]),
            "key_findings": {},
            "data_availability": {},
            "recommended_next_steps": [],
            "summary_metadata": {
                "generation_timestamp": get_iso_timestamp()
            }
        }
        
        # Helper function to extract data from cache wrapper
        def extract_cached_data(data_obj):
            if isinstance(data_obj, dict):
                if "data" in data_obj and "cached_at" in data_obj:
                    return data_obj["data"]  # Unwrap cached data
                return data_obj
            return {}
        
        # Extract key findings from primary enrichment
        primary = enrichment_data.get("primary_enrichment", {})
        
        # Elevation
        elevation_data = extract_cached_data(primary.get("elevation", {}))
        if elevation_data.get("success"):
            summary["key_findings"]["elevation_meters"] = elevation_data.get("elevation_meters")
        
        # Weather (comprehensive)
        weather_data = extract_cached_data(primary.get("weather", {}))
        if weather_data.get("success"):
            weather_details = weather_data.get("weather_data", {})
            
            # Core temperature data (only include non-null values)
            if weather_details.get("temperature_2m_mean") is not None:
                summary["key_findings"]["temperature_mean_celsius"] = weather_details.get("temperature_2m_mean")
            if weather_details.get("temperature_2m_max") is not None:
                summary["key_findings"]["temperature_max_celsius"] = weather_details.get("temperature_2m_max")
            if weather_details.get("temperature_2m_min") is not None:
                summary["key_findings"]["temperature_min_celsius"] = weather_details.get("temperature_2m_min")
            if weather_details.get("apparent_temperature_mean") is not None:
                summary["key_findings"]["apparent_temperature_mean_celsius"] = weather_details.get("apparent_temperature_mean")
            
            # Temperature analysis
            temp_max = weather_details.get("temperature_2m_max")
            temp_min = weather_details.get("temperature_2m_min")
            if temp_max is not None and temp_min is not None:
                summary["key_findings"]["temperature_range_celsius"] = round(temp_max - temp_min, 1)
            
            # Precipitation and moisture
            if weather_details.get("precipitation_sum") is not None:
                summary["key_findings"]["precipitation_total_mm"] = weather_details.get("precipitation_sum")
            if weather_details.get("rain_sum") is not None:
                summary["key_findings"]["rain_mm"] = weather_details.get("rain_sum")
            if weather_details.get("snowfall_sum") is not None:
                summary["key_findings"]["snowfall_mm"] = weather_details.get("snowfall_sum")
            if weather_details.get("relative_humidity_2m_mean") is not None:
                summary["key_findings"]["relative_humidity_percent"] = weather_details.get("relative_humidity_2m_mean")
            if weather_details.get("soil_moisture_0_1cm_mean") is not None:
                summary["key_findings"]["soil_moisture_percent"] = weather_details.get("soil_moisture_0_1cm_mean")
            
            # Wind conditions
            if weather_details.get("wind_speed_10m_max") is not None:
                summary["key_findings"]["wind_speed_max_kmh"] = weather_details.get("wind_speed_10m_max")
            if weather_details.get("wind_gusts_10m_max") is not None:
                summary["key_findings"]["wind_gusts_max_kmh"] = weather_details.get("wind_gusts_10m_max")
            if weather_details.get("wind_direction_10m_dominant") is not None:
                summary["key_findings"]["wind_direction_degrees"] = weather_details.get("wind_direction_10m_dominant")
            
            # Atmospheric conditions
            if weather_details.get("surface_pressure_mean") is not None:
                summary["key_findings"]["surface_pressure_hpa"] = weather_details.get("surface_pressure_mean")
            if weather_details.get("cloudcover_mean") is not None:
                summary["key_findings"]["cloud_cover_percent"] = weather_details.get("cloudcover_mean")
            if weather_details.get("dewpoint_2m_mean") is not None:
                summary["key_findings"]["dewpoint_celsius"] = weather_details.get("dewpoint_2m_mean")
            
            # Solar and daylight
            if weather_details.get("shortwave_radiation_sum") is not None:
                summary["key_findings"]["solar_radiation_mj_m2"] = weather_details.get("shortwave_radiation_sum")
            if weather_details.get("sunshine_duration") is not None:
                summary["key_findings"]["sunshine_duration_hours"] = round(weather_details.get("sunshine_duration") / 3600, 1)  # Convert seconds to hours
            if weather_details.get("daylight_duration") is not None:
                summary["key_findings"]["daylight_duration_hours"] = round(weather_details.get("daylight_duration") / 3600, 1)  # Convert seconds to hours
            if weather_details.get("et0_fao_evapotranspiration") is not None:
                summary["key_findings"]["evapotranspiration_mm"] = weather_details.get("et0_fao_evapotranspiration")
            
            # Soil conditions
            if weather_details.get("soil_temperature_0cm_mean") is not None:
                summary["key_findings"]["soil_temperature_celsius"] = weather_details.get("soil_temperature_0cm_mean")
                
            # Weather summary analysis
            temp_mean = weather_details.get("temperature_2m_mean")
            humidity = weather_details.get("relative_humidity_2m_mean")
            precipitation = weather_details.get("precipitation_sum", 0)
            
            weather_conditions = []
            if temp_mean is not None:
                if temp_mean < 0:
                    weather_conditions.append("freezing")
                elif temp_mean < 10:
                    weather_conditions.append("cold")
                elif temp_mean < 20:
                    weather_conditions.append("cool")
                elif temp_mean < 30:
                    weather_conditions.append("warm")
                else:
                    weather_conditions.append("hot")
            
            if humidity is not None:
                if humidity > 80:
                    weather_conditions.append("very humid")
                elif humidity > 60:
                    weather_conditions.append("humid")
                elif humidity < 30:
                    weather_conditions.append("dry")
            
            if precipitation > 10:
                weather_conditions.append("wet")
            elif precipitation > 0:
                weather_conditions.append("light precipitation")
            else:
                weather_conditions.append("dry")
                
            if weather_conditions:
                summary["key_findings"]["weather_conditions"] = weather_conditions
        
        # Location context
        location_data = extract_cached_data(primary.get("location_context", {}))
        if location_data.get("success"):
            summary["key_findings"]["country"] = location_data.get("country")
            summary["key_findings"]["administrative_region"] = location_data.get("state")
        
        # Soil classification
        soil_class = extract_cached_data(primary.get("soil_classification", {}))
        if soil_class.get("success"):
            classification = soil_class.get("soil_classification", {})
            if "primary_class" in classification:
                summary["key_findings"]["soil_type"] = classification["primary_class"]
            elif "component_name" in classification:
                summary["key_findings"]["soil_type"] = classification["component_name"]
        
        # Nearby features
        nearby_features = extract_cached_data(primary.get("nearby_features", {}))
        if nearby_features.get("success") and nearby_features.get("features"):
            feature_types = set()
            for feature in nearby_features["features"]:
                if feature.get("natural"):
                    feature_types.add(feature["natural"])
            if feature_types:
                summary["key_findings"]["natural_features"] = list(feature_types)
        
        # Ecoregion
        ecoregion_data = primary.get("ecoregion", {})
        if ecoregion_data.get("success"):
            summary["key_findings"]["ecoregion"] = ecoregion_data.get("ecoregion_name")
            summary["key_findings"]["biome"] = ecoregion_data.get("biome_name")
        
        # Ecosystem classification
        ecosystem_data = primary.get("ecosystem_classification", {})
        if ecosystem_data.get("success"):
            summary["key_findings"]["ecosystem_path"] = ecosystem_data.get("ecosystem_path")
        
        # Data availability assessment (using extracted data)
        summary["data_availability"] = {
            "elevation": bool(elevation_data.get("success")),
            "weather": bool(weather_data.get("success")),
            "location_context": bool(location_data.get("success")),
            "nearby_features": bool(nearby_features.get("success")),
            "soil_classification": bool(soil_class.get("success")),
            "soil_properties": bool(primary.get("soil_properties", {}).get("success")),
            "ecoregion_data": bool(ecoregion_data.get("success")),
            "land_cover_data": bool(primary.get("land_cover", {}).get("success")),
            "envo_mappings": len(enrichment_data.get("envo_mappings", {}).get("combined_envo_terms", [])) > 0
        }
        
        # Generate recommendations
        quality_score = enrichment_data.get("data_quality_assessment", {}).get("completeness_score", 0)
        
        if quality_score < 0.7:
            summary["recommended_next_steps"].append("Consider manual data verification for key missing fields")
        
        if not summary["data_availability"]["weather"] and enrichment_data.get("input_coordinates", {}).get("collection_date"):
            summary["recommended_next_steps"].append("Weather data unavailable - consider alternative weather APIs")
        
        if not summary["data_availability"]["ecoregion_data"]:
            # Check if it's a geopandas issue
            ecoregion_error = enrichment_data.get("primary_enrichment", {}).get("ecoregion", {}).get("error", "")
            if "geopandas" in ecoregion_error.lower():
                summary["recommended_next_steps"].append("Install geopandas for local ecoregion data: 'uv add geopandas'")
            else:
                summary["recommended_next_steps"].append("Ecoregion data unavailable - check TEOW 2017 setup")
        
        if not summary["data_availability"]["land_cover_data"]:
            summary["recommended_next_steps"].append("Land cover APIs may be rate-limited - retry or check API availability")
        
        if not summary["data_availability"]["envo_mappings"]:
            summary["recommended_next_steps"].append("Generate additional ENVO mappings using semantic matching")
        
        summary["enrichment_success"] = quality_score >= 0.5
        
        return summary


@click.command()
@click.option('--lat', type=float, required=True, help='Latitude coordinate (-90 to 90)')
@click.option('--lon', type=float, required=True, help='Longitude coordinate (-180 to 180)')
@click.option('--date', type=str, help='Collection date in YYYY-MM-DD format')
@click.option('--output', type=click.Path(path_type=Path), help='Output JSON file (default: stdout)')
@click.option('--enable-crosswalks/--disable-crosswalks', default=True, 
              help='Enable/disable crosswalk normalization')
@click.option('--verbose', is_flag=True, help='Show detailed progress information')
@click.option('--pretty', is_flag=True, help='Pretty-print JSON output')
def main(lat: float, lon: float, date: Optional[str], output: Optional[Path], 
         enable_crosswalks: bool, verbose: bool, pretty: bool):
    """
    Unified biosample enrichment using all available geospatial data sources.
    
    This tool combines all lookup and normalization capabilities from the 
    gold-and-nmdc directory, providing comprehensive enrichment with raw API
    responses and normalized linked data.
    
    Examples:
    
        # Basic enrichment
        python unified_enrichment.py --lat 37.7749 --lon -122.4194
        
        # With collection date and output file
        python unified_enrichment.py --lat 44.428 --lon -110.5885 \\
            --date 2021-08-20 --output yellowstone_enriched.json
        
        # Verbose output with crosswalk normalization
        python unified_enrichment.py --lat 40.7128 --lon -74.0060 \\
            --date 2020-06-15 --enable-crosswalks --verbose --pretty
    """
    
    # Validate coordinates
    if not (-90 <= lat <= 90):
        click.echo("Error: Latitude must be between -90 and 90", err=True)
        raise click.Abort()
    
    if not (-180 <= lon <= 180):
        click.echo("Error: Longitude must be between -180 and 180", err=True)
        raise click.Abort()
    
    # Validate date format if provided
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            click.echo("Error: Date must be in YYYY-MM-DD format", err=True)
            raise click.Abort()
    
    if verbose:
        click.echo("🚀 Starting unified biosample enrichment")
        click.echo(f"📍 Coordinates: {lat}, {lon}")
        if date:
            click.echo(f"📅 Collection date: {date}")
        click.echo(f"🔗 Crosswalks: {'enabled' if enable_crosswalks else 'disabled'}")
        click.echo()
    
    try:
        # Initialize enricher
        enricher = UnifiedBiosampleEnricher(
            enable_crosswalks=enable_crosswalks,
            verbose=verbose
        )
        
        # Run comprehensive enrichment
        start_time = time.time()
        result = enricher.enrich_coordinates(lat, lon, date)
        end_time = time.time()
        
        # Add timing information
        result["enrichment_metadata"]["processing_time_seconds"] = round(end_time - start_time, 2)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert numpy types before JSON serialization
        result = convert_numpy_types(result)
        
        # Format output
        json_kwargs = {"indent": 2} if pretty else {"separators": (',', ':')}
        json_output = json.dumps(result, **json_kwargs)
        
        # Save or print result
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'w') as f:
                f.write(json_output)
            if verbose:
                file_size = len(json_output.encode('utf-8'))
                click.echo(f"💾 Results saved to {output} ({file_size:,} bytes)")
        else:
            click.echo(json_output)
        
        if verbose:
            success_rate = result["unified_summary"]["successful_apis"] / result["unified_summary"]["total_apis_called"] * 100
            click.echo(f"✅ Enrichment complete in {result['enrichment_metadata']['processing_time_seconds']}s")
            click.echo(f"📊 Success rate: {success_rate:.1f}% ({result['unified_summary']['successful_apis']}/{result['unified_summary']['total_apis_called']} APIs)")
            
    except Exception as e:
        click.echo(f"❌ Enrichment failed: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
API Discovery and Interrogation Tools

This module contains functions to systematically interrogate geospatial APIs
to discover their capabilities, parameters, controlled vocabularies, and return formats.

Usage:
    python api_discovery.py --api=all
    python api_discovery.py --api=soilgrids
    python api_discovery.py --api=usda_sda
"""

import json
import requests
import time
from typing import Dict, List, Any, Optional
from pathlib import Path


def interrogate_usda_nrcs_sda() -> Dict[str, Any]:
    """
    Systematically discover USDA NRCS Soil Data Access capabilities.
    
    Returns:
        Dict with discovered endpoints, functions, and parameters
    """
    results = {
        "api_name": "USDA NRCS Soil Data Access (SDA)",
        "base_url": "https://sdmdataaccess.sc.egov.usda.gov",
        "discovery_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": {},
        "functions": {},
        "tables": {},
        "errors": []
    }
    
    # Test different endpoint variations
    endpoints_to_test = [
        "/Tabular/post.rest",
        "/Tabular/SDMTabularService.asmx",
        "/Spatial/SDMSpatialService.asmx",
        "/api/",
        "/help",
        "/documentation"
    ]
    
    for endpoint in endpoints_to_test:
        url = results["base_url"] + endpoint
        try:
            response = requests.get(url, timeout=10)
            results["endpoints"][endpoint] = {
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "content_length": len(response.content),
                "accessible": response.status_code < 400
            }
            
            # For accessible endpoints, try to extract more info
            if response.status_code < 400:
                content = response.text[:1000]  # First 1KB
                results["endpoints"][endpoint]["sample_content"] = content
                
        except Exception as e:
            results["endpoints"][endpoint] = {"error": str(e)}
    
    # Test function discovery queries
    function_queries = [
        "SELECT name FROM sys.objects WHERE type = 'FN'",
        "SELECT ROUTINE_NAME FROM INFORMATION_SCHEMA.ROUTINES WHERE ROUTINE_TYPE = 'FUNCTION'",
        "SELECT name FROM sysobjects WHERE type = 'FN' AND name LIKE '%mukey%'",
        "SELECT name FROM sysobjects WHERE type = 'FN' AND name LIKE '%SDA%'",
        "EXEC sp_helpdb",  # Get database info
        "SELECT name FROM sys.tables",  # Get table names
        "SELECT name FROM sys.procedures WHERE name LIKE '%SDA%'"  # Get stored procedures
    ]
    
    for query in function_queries:
        try:
            url = results["base_url"] + "/Tabular/post.rest"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "SERVICE": "query",
                "REQUEST": "query",
                "FORMAT": "JSON", 
                "QUERY": query
            }
            
            response = requests.post(url, headers=headers, data=data, timeout=30)
            
            results["functions"][f"query_{len(results['functions'])}"] = {
                "query": query,
                "status_code": response.status_code,
                "response": response.text[:500],  # First 500 chars
                "success": response.status_code == 200
            }
            
        except Exception as e:
            results["functions"][f"query_{len(results['functions'])}"] = {
                "query": query,
                "error": str(e)
            }
    
    # Test known function names from documentation
    known_functions = [
        "SDA_Get_Mukey_from_intersection_with_WktWgs84",
        "SDA_GetTabularSchema", 
        "SDA_Get_MupolygonGeoJSON",
        "muPolygonWktWgs84ByMukey",
        "mapUnitRasterDominant"
    ]
    
    for func_name in known_functions:
        # Test function existence
        test_query = f"SELECT * FROM INFORMATION_SCHEMA.ROUTINES WHERE ROUTINE_NAME = '{func_name}'"
        try:
            url = results["base_url"] + "/Tabular/post.rest"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "SERVICE": "query",
                "REQUEST": "query",
                "FORMAT": "JSON",
                "QUERY": test_query
            }
            
            response = requests.post(url, headers=headers, data=data, timeout=30)
            results["functions"][func_name] = {
                "existence_query": test_query,
                "status_code": response.status_code,
                "exists": "Table" in response.text and len(response.json().get("Table", [])) > 0,
                "response": response.text[:300]
            }
            
        except Exception as e:
            results["functions"][func_name] = {"error": str(e)}
    
    return results


def interrogate_isric_soilgrids() -> Dict[str, Any]:
    """
    Systematically discover ISRIC SoilGrids API capabilities.
    
    Returns:
        Dict with discovered endpoints and capabilities
    """
    results = {
        "api_name": "ISRIC SoilGrids",
        "discovery_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rest_api": {},
        "wcs_api": {},
        "webdav": {},
        "errors": []
    }
    
    # Test REST API endpoints
    rest_base = "https://rest.isric.org/soilgrids/v2.0"
    rest_endpoints = [
        "/",
        "/classification/query",
        "/properties/query",
        "/coverage/query",
        "/info"
    ]
    
    for endpoint in rest_endpoints:
        url = rest_base + endpoint
        try:
            # Test with sample parameters
            params = {"lon": -110.5885, "lat": 44.428} if "query" in endpoint else {}
            response = requests.get(url, params=params, timeout=20)
            
            results["rest_api"][endpoint] = {
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "accessible": response.status_code < 400
            }
            
            if response.status_code < 400:
                try:
                    json_data = response.json()
                    results["rest_api"][endpoint]["sample_response"] = json_data
                except:
                    results["rest_api"][endpoint]["text_response"] = response.text[:500]
                    
        except Exception as e:
            results["rest_api"][endpoint] = {"error": str(e)}
    
    # Test WCS capabilities
    wcs_base = "https://maps.isric.org/mapserv"
    wcs_properties = ["phh2o", "soc", "sand", "silt", "clay", "bdod", "nitrogen"]
    
    for prop in wcs_properties:
        try:
            # Test GetCapabilities
            params = {
                "map": f"/map/{prop}.map",
                "SERVICE": "WCS",
                "REQUEST": "GetCapabilities"
            }
            response = requests.get(wcs_base, params=params, timeout=20)
            
            results["wcs_api"][prop] = {
                "capabilities_status": response.status_code,
                "capabilities_accessible": response.status_code < 400,
                "map_file": f"/map/{prop}.map"
            }
            
            if response.status_code < 400:
                # Extract coverage IDs from capabilities
                content = response.text
                coverage_ids = []
                # Simple extraction of coverage IDs (would use XML parser in production)
                if f"{prop}_" in content:
                    lines = content.split('\n')
                    for line in lines:
                        if f"{prop}_" in line and "CoverageOfferingBrief" in line:
                            coverage_ids.append(line.strip())
                
                results["wcs_api"][prop]["sample_coverages"] = coverage_ids[:5]
            
        except Exception as e:
            results["wcs_api"][prop] = {"error": str(e)}
    
    # Test WebDAV file structure
    webdav_base = "https://files.isric.org/soilgrids/latest/data"
    
    for prop in wcs_properties[:3]:  # Test first 3 to avoid too many requests
        try:
            url = f"{webdav_base}/{prop}/"
            response = requests.get(url, timeout=20)
            
            results["webdav"][prop] = {
                "url": url,
                "status_code": response.status_code,
                "accessible": response.status_code < 400
            }
            
            if response.status_code < 400:
                # Extract file/directory listings (simple HTML parsing)
                content = response.text
                links = []
                if "href=" in content:
                    import re
                    href_matches = re.findall(r'href="([^"]*)"', content)
                    links = [link for link in href_matches if not link.startswith('http')]
                
                results["webdav"][prop]["sample_files"] = links[:10]
                
        except Exception as e:
            results["webdav"][prop] = {"error": str(e)}
    
    return results


def interrogate_open_meteo() -> Dict[str, Any]:
    """
    Systematically discover Open-Meteo API capabilities.
    
    Returns:
        Dict with discovered endpoints and parameters
    """
    results = {
        "api_name": "Open-Meteo",
        "discovery_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": {},
        "parameters": {},
        "errors": []
    }
    
    # Test different Open-Meteo endpoints
    endpoints = [
        "https://api.open-meteo.com/v1/forecast",
        "https://archive-api.open-meteo.com/v1/archive", 
        "https://climate-api.open-meteo.com/v1/climate",
        "https://marine-api.open-meteo.com/v1/marine",
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    ]
    
    for endpoint in endpoints:
        try:
            # Test basic accessibility
            response = requests.get(endpoint, timeout=10)
            
            results["endpoints"][endpoint] = {
                "status_code": response.status_code,
                "accessible": response.status_code < 400
            }
            
            if response.status_code < 400:
                try:
                    json_data = response.json()
                    results["endpoints"][endpoint]["response_structure"] = json_data
                except:
                    results["endpoints"][endpoint]["text_response"] = response.text[:300]
            
            # Test with sample parameters to discover capabilities
            if "archive" in endpoint:
                sample_params = {
                    "latitude": 44.428,
                    "longitude": -110.5885,
                    "start_date": "2021-08-20",
                    "end_date": "2021-08-20",
                    "hourly": "temperature_2m"  # Basic parameter
                }
                
                sample_response = requests.get(endpoint, params=sample_params, timeout=20)
                results["endpoints"][endpoint]["sample_query"] = {
                    "params": sample_params,
                    "status_code": sample_response.status_code,
                    "success": sample_response.status_code == 200
                }
                
                if sample_response.status_code == 200:
                    try:
                        sample_data = sample_response.json()
                        results["endpoints"][endpoint]["sample_response"] = sample_data
                    except:
                        pass
                        
        except Exception as e:
            results["endpoints"][endpoint] = {"error": str(e)}
    
    # Test parameter discovery for archive API (microbiome research focus)
    microbiome_params = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean", 
        "precipitation_sum",
        "soil_temperature_0_to_7cm_mean",
        "soil_moisture_0_to_7cm_mean",
        "vapour_pressure_deficit_mean",
        "et0_fao_evapotranspiration",
        "shortwave_radiation_sum",
        "sunshine_duration"
    ]
    
    for param in microbiome_params:
        try:
            test_params = {
                "latitude": 44.428,
                "longitude": -110.5885,
                "start_date": "2021-08-20",
                "end_date": "2021-08-20",
                "daily": param
            }
            
            response = requests.get(
                "https://archive-api.open-meteo.com/v1/archive", 
                params=test_params, 
                timeout=20
            )
            
            results["parameters"][param] = {
                "status_code": response.status_code,
                "available": response.status_code == 200,
                "test_params": test_params
            }
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "daily" in data and param in data["daily"]:
                        results["parameters"][param]["sample_values"] = data["daily"][param][:5]  # First 5 values
                except:
                    pass
                    
        except Exception as e:
            results["parameters"][param] = {"error": str(e)}
    
    return results


def interrogate_osm_overpass() -> Dict[str, Any]:
    """
    Systematically discover OpenStreetMap Overpass API capabilities.
    
    Returns:
        Dict with discovered endpoints and query types
    """
    results = {
        "api_name": "OpenStreetMap Overpass API",
        "discovery_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": {},
        "query_types": {},
        "tag_discovery": {},
        "errors": []
    }
    
    # Test different Overpass endpoints
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter",
        "https://overpass.osm.ch/api/interpreter"
    ]
    
    for endpoint in endpoints:
        try:
            # Test basic status
            status_response = requests.get(endpoint + "/status", timeout=10)
            
            results["endpoints"][endpoint] = {
                "status_endpoint": {
                    "status_code": status_response.status_code,
                    "accessible": status_response.status_code < 400
                }
            }
            
            if status_response.status_code < 400:
                results["endpoints"][endpoint]["status_response"] = status_response.text[:300]
            
            # Test basic query capability
            simple_query = "[out:json][timeout:5]; node(around:100,44.428,-110.5885)[name]; out;"
            
            query_response = requests.post(endpoint, data=simple_query, timeout=15)
            results["endpoints"][endpoint]["query_test"] = {
                "status_code": query_response.status_code,
                "query_works": query_response.status_code == 200,
                "sample_query": simple_query
            }
            
            if query_response.status_code == 200:
                try:
                    data = query_response.json()
                    results["endpoints"][endpoint]["sample_elements"] = len(data.get("elements", []))
                except:
                    pass
                    
        except Exception as e:
            results["endpoints"][endpoint] = {"error": str(e)}
    
    # Test different query patterns for tag discovery
    tag_categories = {
        "natural": ["water", "forest", "grassland", "wetland", "geyser", "hot_spring"],
        "landuse": ["forest", "farmland", "residential", "commercial", "industrial"],
        "waterway": ["stream", "river", "lake"],
        "amenity": ["restaurant", "hospital", "school"],
        "highway": ["primary", "secondary", "residential"]
    }
    
    for category, values in tag_categories.items():
        results["tag_discovery"][category] = {}
        
        for value in values[:3]:  # Test first 3 values to avoid too many requests
            try:
                query = f'[out:json][timeout:10]; node["{category}"="{value}"](around:1000,44.428,-110.5885); out count;'
                
                response = requests.post(
                    "https://overpass-api.de/api/interpreter", 
                    data=query, 
                    timeout=15
                )
                
                results["tag_discovery"][category][value] = {
                    "status_code": response.status_code,
                    "query": query,
                    "tag_exists": response.status_code == 200
                }
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        count = data.get("elements", [{}])[0].get("tags", {}).get("total", 0)
                        results["tag_discovery"][category][value]["element_count"] = count
                    except:
                        pass
                        
            except Exception as e:
                results["tag_discovery"][category][value] = {"error": str(e)}
            
            time.sleep(1)  # Rate limiting
    
    return results


def save_discovery_results(results: Dict[str, Any], output_dir: str = "data/outputs/api_discovery"):
    """Save API discovery results to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    api_name = results["api_name"].lower().replace(" ", "_")
    filename = f"{api_name}_discovery_{timestamp}.json"
    
    filepath = output_path / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Discovery results saved to: {filepath}")
    return filepath


def run_comprehensive_api_discovery():
    """Run discovery on all supported APIs."""
    print("🔍 Starting comprehensive API discovery...")
    
    apis = [
        ("USDA NRCS SDA", interrogate_usda_nrcs_sda),
        ("ISRIC SoilGrids", interrogate_isric_soilgrids), 
        ("Open-Meteo", interrogate_open_meteo),
        ("OSM Overpass", interrogate_osm_overpass)
    ]
    
    results = {}
    
    for api_name, interrogate_func in apis:
        print(f"\n📊 Discovering {api_name} capabilities...")
        try:
            api_results = interrogate_func()
            results[api_name] = api_results
            save_discovery_results(api_results)
            print(f"✅ {api_name} discovery complete")
        except Exception as e:
            print(f"❌ {api_name} discovery failed: {e}")
            results[api_name] = {"error": str(e)}
    
    # Save comprehensive results
    comprehensive_results = {
        "discovery_session": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_apis": len(apis),
            "successful_discoveries": len([r for r in results.values() if "error" not in r])
        },
        "api_results": results
    }
    
    save_discovery_results(comprehensive_results, "data/outputs/api_discovery")
    print(f"\n🎉 Comprehensive API discovery complete!")
    return comprehensive_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover geospatial API capabilities")
    parser.add_argument("--api", choices=["all", "usda", "soilgrids", "openmeteo", "osm"], 
                       default="all", help="Which API to interrogate")
    
    args = parser.parse_args()
    
    if args.api == "all":
        run_comprehensive_api_discovery()
    elif args.api == "usda":
        results = interrogate_usda_nrcs_sda()
        save_discovery_results(results)
    elif args.api == "soilgrids":
        results = interrogate_isric_soilgrids()
        save_discovery_results(results)
    elif args.api == "openmeteo":
        results = interrogate_open_meteo()
        save_discovery_results(results)
    elif args.api == "osm":
        results = interrogate_osm_overpass()
        save_discovery_results(results)
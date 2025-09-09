"""
WMS/WCS capabilities detection for proper axis order and parameter handling.

Provides automatic detection of WMS version, CRS axis order, and required
parameters to build correct requests for diverse WMS/WCS servers.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple
import requests
from urllib.parse import urlencode


def parse_wms_capabilities(capabilities_url: str) -> Dict[str, Any]:
    """
    Parse WMS GetCapabilities response to extract service metadata.
    
    Args:
        capabilities_url: URL for GetCapabilities request
        
    Returns:
        Dict with parsed capabilities metadata
    """
    try:
        response = requests.get(capabilities_url, timeout=30)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        # Extract namespace
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"
        
        capabilities = {
            "service_url": capabilities_url,
            "namespace": namespace,
            "version": root.get("version"),
            "layers": [],
            "supported_crs": [],
            "supported_formats": [],
            "axis_order": {},
            "contact_info": {}
        }
        
        # Extract service information
        service_elem = root.find(f"{namespace}Service")
        if service_elem is not None:
            name_elem = service_elem.find(f"{namespace}Name")
            if name_elem is not None:
                capabilities["service_name"] = name_elem.text
            
            title_elem = service_elem.find(f"{namespace}Title")
            if title_elem is not None:
                capabilities["service_title"] = title_elem.text
        
        # Extract capability information
        capability_elem = root.find(f"{namespace}Capability")
        if capability_elem is not None:
            
            # Extract request information
            request_elem = capability_elem.find(f"{namespace}Request")
            if request_elem is not None:
                getmap_elem = request_elem.find(f"{namespace}GetMap")
                if getmap_elem is not None:
                    # Extract supported formats
                    format_elems = getmap_elem.findall(f"{namespace}Format")
                    capabilities["supported_formats"] = [fmt.text for fmt in format_elems]
            
            # Extract layer information
            layer_elem = capability_elem.find(f"{namespace}Layer")
            if layer_elem is not None:
                capabilities["layers"] = _parse_layer_info(layer_elem, namespace)
                
                # Extract supported CRS from root layer
                crs_elems = layer_elem.findall(f"{namespace}CRS") or layer_elem.findall(f"{namespace}SRS")
                capabilities["supported_crs"] = [crs.text for crs in crs_elems]
        
        # Determine axis order for common CRS
        capabilities["axis_order"] = _determine_axis_order(capabilities["version"], capabilities["supported_crs"])
        
        return capabilities
        
    except Exception as e:
        return {
            "service_url": capabilities_url,
            "error": str(e),
            "success": False
        }


def _parse_layer_info(layer_elem: ET.Element, namespace: str) -> List[Dict[str, Any]]:
    """Parse layer information from capabilities XML."""
    layers = []
    
    # Parse current layer
    name_elem = layer_elem.find(f"{namespace}Name")
    title_elem = layer_elem.find(f"{namespace}Title")
    
    if name_elem is not None:
        layer_info = {
            "name": name_elem.text,
            "title": title_elem.text if title_elem is not None else name_elem.text,
            "queryable": layer_elem.get("queryable", "0") == "1",
            "crs": [],
            "styles": []
        }
        
        # Extract layer-specific CRS
        crs_elems = layer_elem.findall(f"{namespace}CRS") or layer_elem.findall(f"{namespace}SRS")
        layer_info["crs"] = [crs.text for crs in crs_elems]
        
        # Extract styles
        style_elems = layer_elem.findall(f"{namespace}Style")
        for style_elem in style_elems:
            style_name_elem = style_elem.find(f"{namespace}Name")
            style_title_elem = style_elem.find(f"{namespace}Title")
            if style_name_elem is not None:
                layer_info["styles"].append({
                    "name": style_name_elem.text,
                    "title": style_title_elem.text if style_title_elem is not None else style_name_elem.text
                })
        
        layers.append(layer_info)
    
    # Parse child layers recursively
    child_layers = layer_elem.findall(f"{namespace}Layer")
    for child_layer in child_layers:
        layers.extend(_parse_layer_info(child_layer, namespace))
    
    return layers


def _determine_axis_order(version: str, supported_crs: List[str]) -> Dict[str, str]:
    """Determine axis order for common CRS based on WMS version."""
    axis_order = {}
    
    # Default axis orders based on WMS version and CRS
    if version and version.startswith("1.3"):
        # WMS 1.3.0+ follows CRS definition strictly
        axis_order.update({
            "EPSG:4326": "lat,lon",  # Geographic CRS: latitude first
            "EPSG:3857": "lon,lat",  # Web Mercator: easting first
            "EPSG:4269": "lat,lon",  # NAD83: latitude first
            "EPSG:4267": "lat,lon",  # NAD27: latitude first
        })
    else:
        # WMS 1.1.1 and earlier: always lon,lat (easting,northing)
        axis_order.update({
            "EPSG:4326": "lon,lat",
            "EPSG:3857": "lon,lat", 
            "EPSG:4269": "lon,lat",
            "EPSG:4267": "lon,lat",
        })
    
    # Only return axis order for CRS that are actually supported
    return {crs: order for crs, order in axis_order.items() if crs in supported_crs}


def build_wms_request(
    base_url: str,
    layer_name: str, 
    lat: float,
    lon: float,
    capabilities: Dict[str, Any] = None,
    request_type: str = "GetFeatureInfo",
    **kwargs
) -> str:
    """
    Build proper WMS request URL based on capabilities.
    
    Args:
        base_url: Base WMS service URL
        layer_name: Layer name to query
        lat: Latitude
        lon: Longitude  
        capabilities: Parsed capabilities dict
        request_type: GetMap or GetFeatureInfo
        **kwargs: Additional parameters
        
    Returns:
        Complete WMS request URL
    """
    if capabilities is None:
        # Fetch capabilities if not provided
        caps_url = f"{base_url}?SERVICE=WMS&REQUEST=GetCapabilities"
        capabilities = parse_wms_capabilities(caps_url)
    
    # Default parameters
    params = {
        "SERVICE": "WMS",
        "REQUEST": request_type,
        "LAYERS": layer_name,
        "STYLES": kwargs.get("styles", ""),
        "FORMAT": kwargs.get("format", "image/png"),
        "WIDTH": kwargs.get("width", 100),
        "HEIGHT": kwargs.get("height", 100),
    }
    
    # Determine version and version-specific parameters
    version = capabilities.get("version", "1.3.0")
    params["VERSION"] = version
    
    crs = kwargs.get("crs", "EPSG:4326")
    
    if version.startswith("1.3"):
        # WMS 1.3.0+
        params["CRS"] = crs
        
        # Determine bbox order based on CRS
        axis_order = capabilities.get("axis_order", {}).get(crs, "lat,lon")
        buffer = kwargs.get("buffer", 0.001)
        
        if axis_order == "lat,lon":
            params["BBOX"] = f"{lat-buffer},{lon-buffer},{lat+buffer},{lon+buffer}"
        else:
            params["BBOX"] = f"{lon-buffer},{lat-buffer},{lon+buffer},{lat+buffer}"
        
        if request_type == "GetFeatureInfo":
            params["I"] = kwargs.get("i", 50)
            params["J"] = kwargs.get("j", 50)
            params["QUERY_LAYERS"] = layer_name
            params["INFO_FORMAT"] = kwargs.get("info_format", "application/json")
            params["FEATURE_COUNT"] = kwargs.get("feature_count", 1)
            
    else:
        # WMS 1.1.1 and earlier
        params["SRS"] = crs
        
        # Always lon,lat order for 1.1.1
        buffer = kwargs.get("buffer", 0.001)
        params["BBOX"] = f"{lon-buffer},{lat-buffer},{lon+buffer},{lat+buffer}"
        
        if request_type == "GetFeatureInfo":
            params["X"] = kwargs.get("x", 50)
            params["Y"] = kwargs.get("y", 50)
            params["QUERY_LAYERS"] = layer_name
            params["INFO_FORMAT"] = kwargs.get("info_format", "application/json")
            params["FEATURE_COUNT"] = kwargs.get("feature_count", 1)
    
    # Build URL
    return f"{base_url}?{urlencode(params)}"


def test_wms_endpoint(
    base_url: str,
    layer_name: str,
    lat: float = 40.0,
    lon: float = -100.0
) -> Dict[str, Any]:
    """
    Test WMS endpoint with both version strategies.
    
    Args:
        base_url: Base WMS service URL
        layer_name: Layer name to test
        lat: Test latitude
        lon: Test longitude
        
    Returns:
        Dict with test results for different versions
    """
    results = {
        "base_url": base_url,
        "layer_name": layer_name,
        "test_point": {"lat": lat, "lon": lon},
        "tests": {}
    }
    
    # Test WMS 1.3.0
    try:
        caps_130 = parse_wms_capabilities(f"{base_url}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities")
        if not caps_130.get("error"):
            url_130 = build_wms_request(base_url, layer_name, lat, lon, caps_130, 
                                       request_type="GetFeatureInfo", version="1.3.0")
            
            response = requests.get(url_130, timeout=30)
            results["tests"]["wms_1.3.0"] = {
                "capabilities_success": True,
                "request_url": url_130,
                "response_status": response.status_code,
                "response_success": response.status_code == 200,
                "axis_order": caps_130.get("axis_order", {}).get("EPSG:4326", "unknown")
            }
        else:
            results["tests"]["wms_1.3.0"] = {
                "capabilities_success": False,
                "error": caps_130.get("error")
            }
            
    except Exception as e:
        results["tests"]["wms_1.3.0"] = {
            "capabilities_success": False,
            "error": str(e)
        }
    
    # Test WMS 1.1.1
    try:
        caps_111 = parse_wms_capabilities(f"{base_url}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetCapabilities")
        if not caps_111.get("error"):
            url_111 = build_wms_request(base_url, layer_name, lat, lon, caps_111,
                                       request_type="GetFeatureInfo", version="1.1.1")
            
            response = requests.get(url_111, timeout=30)
            results["tests"]["wms_1.1.1"] = {
                "capabilities_success": True,
                "request_url": url_111,
                "response_status": response.status_code,
                "response_success": response.status_code == 200,
                "axis_order": "lon,lat"  # Always lon,lat for 1.1.1
            }
        else:
            results["tests"]["wms_1.1.1"] = {
                "capabilities_success": False,
                "error": caps_111.get("error")
            }
            
    except Exception as e:
        results["tests"]["wms_1.1.1"] = {
            "capabilities_success": False,
            "error": str(e)
        }
    
    # Determine recommended version
    successful_tests = [v for v, result in results["tests"].items() 
                       if result.get("response_success")]
    
    results["recommendation"] = {
        "working_versions": successful_tests,
        "preferred_version": successful_tests[0] if successful_tests else None,
        "notes": "Use capabilities detection for robust requests"
    }
    
    return results


def wms_capabilities_shim(base_url: str) -> Dict[str, Any]:
    """
    Simple capabilities detection shim for use in existing code.
    
    Args:
        base_url: Base WMS service URL
        
    Returns:
        Dict with essential capabilities for request building
    """
    try:
        # Try 1.3.0 first
        caps = parse_wms_capabilities(f"{base_url}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities")
        
        if caps.get("error"):
            # Fallback to 1.1.1
            caps = parse_wms_capabilities(f"{base_url}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetCapabilities")
        
        return {
            "version": caps.get("version", "1.3.0"),
            "axis_order": caps.get("axis_order", {}),
            "supported_formats": caps.get("supported_formats", ["image/png"]),
            "layers": [layer["name"] for layer in caps.get("layers", [])],
            "success": not caps.get("error"),
            "error": caps.get("error")
        }
        
    except Exception as e:
        return {
            "version": "1.3.0",  # Default assumption
            "axis_order": {"EPSG:4326": "lat,lon"},
            "success": False,
            "error": str(e)
        }
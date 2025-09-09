"""
MCP soil service integration for soil type classification.
"""

def get_soil_type_mcp(lat: float, lon: float) -> str:
    """
    Get FAO soil type using MCP landuse service.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        FAO soil classification string
    """
    try:
        # This would be called directly via the MCP system
        # For now, just return None to indicate service not available
        return None
    except Exception:
        return None
# Alternative Geospatial Data APIs

This directory contains replacement implementations for the ORNL landuse-mcp tools, providing soil type, land cover, and temporal analysis using free, open APIs.

## Overview

The implementation consists of:

1. **`alternative_geospatial.py`** - Main API client and data retrieval functions
2. **`generate_mappings.py`** - ENVO ontology mapping generator (AI + deterministic)
3. **`mappings/`** - External mapping files for different classification systems

## Features

### Data Sources

- **ISRIC SoilGrids**: Global soil classification (250m resolution)
- **ESA WorldCover**: Global land cover (10m resolution, 2020/2021)
- **USGS NLCD**: US land cover (30m resolution, multiple years)
- **Temporal Analysis**: Historical land cover change detection

### ENVO Ontology Integration

All results include ENVO (Environmental Ontology) term mappings for semantic interoperability.

### Caching & Error Handling

- 30-day file-based caching
- Retry logic with exponential backoff
- Graceful fallbacks between data sources

## Usage

### Basic Usage

```python
from alternative_geospatial import (
    get_soil_type_soilgrids,
    get_land_cover_esa_worldcover,
    get_land_cover_nlcd,
    get_comprehensive_site_analysis
)

# Get soil classification
lat, lon = 37.7749, -122.4194  # San Francisco
soil_data = get_soil_type_soilgrids(lat, lon)

# Get current land cover
land_cover = get_land_cover_esa_worldcover(lat, lon, year=2021)

# Get comprehensive analysis
analysis = get_comprehensive_site_analysis(lat, lon, "2020-06-15")
```

### API Requirements

| API | Authentication | Rate Limits | Coverage |
|-----|---------------|-------------|----------|
| ISRIC SoilGrids | None | Reasonable use | Global |
| ESA WorldCover | None | Unknown | Global |
| USGS NLCD | None | Reasonable use | Continental US |

### Response Format

All functions return dictionaries with consistent structure:

```python
{
    "soil_type": "Cambisols",           # Classification result
    "confidence": 0.85,                 # Confidence score (if available)
    "envo_terms": [{                    # ENVO ontology mappings
        "id": "ENVO:00002268",
        "label": "cambisol", 
        "term": "Cambisol",
        "confidence": "deterministic",
        "source": "FAO World Reference Base"
    }],
    "data_source": "ISRIC SoilGrids v2.0"
}
```

## ENVO Mappings

### Mapping Files

- **`mappings/soilgrids_fao_to_envo.json`** - FAO soil types → ENVO terms (deterministic)
- **`mappings/esa_worldcover_to_envo.json`** - ESA WorldCover classes → ENVO terms
- **`mappings/nlcd_to_envo.json`** - NLCD classes → ENVO terms

### Regenerating Mappings

```bash
# Generate all mappings with AI assistance
python generate_mappings.py

# Generate only deterministic mappings
python generate_mappings.py --no-ai

# Specify output directory
python generate_mappings.py --mappings-dir custom_mappings/
```

### AI-Assisted Mapping Generation

The mapping generator can use:

1. **Claude Code CLI** - For comprehensive ENVO term analysis
2. **OLS API** - As fallback for ontology searches

Example AI workflow:
```bash
# Generate ESA WorldCover mappings with AI
python generate_mappings.py
# Prompts Claude Code to analyze each land cover class
# Finds best ENVO term matches with explanations
# Saves results to JSON files
```

### Manual Mapping Curation

Mappings can be manually edited for accuracy:

```json
{
  "10": {
    "id": "ENVO:01000174",
    "label": "forest biome",
    "term": "Tree cover",
    "confidence": "manual_curated",
    "source": "ESA WorldCover 10m v200",
    "description": "All types of forest and woodland"
  }
}
```

## Implementation Details

### Soil Classification (ISRIC SoilGrids)

- **Endpoint**: `https://rest.soilgrids.org/soilgrids/v2.0/classification/query`
- **Input**: Latitude, longitude
- **Output**: FAO soil type with confidence scores
- **Resolution**: 250m
- **Coverage**: Global

### Land Cover (ESA WorldCover)

- **Endpoint**: `https://services.terrascope.be/wms/v2/worldcover`
- **Input**: Latitude, longitude, year (2020/2021)
- **Output**: Land cover class (11 categories)
- **Resolution**: 10m
- **Coverage**: Global

### Land Cover (USGS NLCD)

- **Endpoint**: `https://www.mrlc.gov/geoserver/mrlc_display/wms`
- **Input**: Latitude, longitude, year
- **Output**: Land cover class (20+ categories)
- **Resolution**: 30m  
- **Coverage**: Continental US, Alaska, Hawaii, Puerto Rico

### Temporal Analysis

- Compares historical vs current land cover
- Finds closest available year to target date
- Detects land cover changes over time
- Provides change descriptions

## Troubleshooting

### Common Issues

1. **"No mapping file found"**
   ```bash
   # Run mapping generator
   python generate_mappings.py --no-ai
   ```

2. **API timeouts/failures**
   - Check internet connection
   - APIs may have temporary outages
   - Results are cached for 30 days

3. **Coordinates outside bounds**
   - NLCD only works for US locations
   - Check latitude/longitude values

### Cache Location

Cached responses stored in: `cache/geospatial/`

Clear cache:
```bash
rm -rf cache/geospatial/
```

## Integration with Existing Code

### Replacing ORNL landuse-mcp calls

**Before:**
```python
from landuse_mcp.main import get_land_cover, get_landuse_dates, get_soil_type

soil = get_soil_type(lat=lat, lon=lon)
land_cover = get_land_cover(lat=lat, lon=lon, start_date=start, end_date=end)
dates = get_landuse_dates(lat=lat, lon=lon)
```

**After:**
```python
from alternative_geospatial import (
    get_soil_type_soilgrids, 
    get_land_cover_esa_worldcover,
    get_available_nlcd_years
)

soil = get_soil_type_soilgrids(lat, lon)
land_cover = get_land_cover_esa_worldcover(lat, lon, year=2021)  
available_years = get_available_nlcd_years()  # For US locations
```

### Updating Dependencies

Remove from `pyproject.toml`:
```toml
landuse-mcp>=0.1.9
```

Add:
```toml
requests>=2.32.0
```

## Contributing

### Adding New Data Sources

1. Add API client function to `alternative_geospatial.py`
2. Create ENVO mapping file in `mappings/`
3. Add mapping generation logic to `generate_mappings.py`
4. Update documentation

### Improving Mappings

1. Edit mapping files directly for corrections
2. Use AI generator for comprehensive reviews
3. Contribute improved mappings back to project

## Future Enhancements

- **Google Earth Engine** integration for advanced temporal analysis
- **Sentinel-2** land cover classification
- **OpenStreetMap** feature extraction for local context
- **Climate data** integration (precipitation, temperature trends)
- **Elevation** and **slope** analysis using DEMs
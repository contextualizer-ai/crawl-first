# Configuration System

This directory contains configuration files for the unified geospatial enrichment system.

## Configuration Files

### `enrichment.yml` (Primary)
The main configuration file in YAML format containing:

- **Dataset Paths**: Local COG/shapefile paths for elevation, land cover, soils, coastlines
- **Provider Settings**: Primary/fallback chains for weather, elevation, land cover, soils
- **Processing Parameters**: OSM settings, site typing thresholds, sampling strategies
- **Service Configuration**: WMS settings, caching, error handling, logging

### `enrichment.json` (Fallback)
JSON version of the configuration for environments without PyYAML.

## Key Configuration Sections

### Datasets
Configure paths to local geospatial datasets:
```yaml
datasets:
  elevation:
    copernicus_dem: "/data/elevation/copernicus_dem_30m.tif"
  land_cover:
    worldcover_2021: "/data/land_cover/esa_worldcover_2021_v200.tif"
  soils:
    soilgrids_ph: "/data/soils/soilgrids/phh2o_0-5cm_mean.tif"
```

### Providers
Set fallback chains for different data types:
```yaml
providers:
  weather:
    primary: "open_meteo"
    secondary: "meteostat" 
    fallback: "prism_us"
```

### Site Typing
Distance thresholds for site classification:
```yaml
site_typing:
  D_open_km: 20      # Open ocean threshold
  D_inland_km: 0.2   # Inland water threshold  
  D_coast_km: 2      # Coastal terrestrial threshold
```

### Sampling Strategies
Raster sampling methods by dataset type:
```yaml
sampling:
  soils:
    method: "buffered_median"
    window_px: 3  # 3x3 median for soil properties
```

## Usage

The configuration system automatically loads settings in this order:
1. `config/enrichment.yml` (if PyYAML available)
2. `config/enrichment.json` (fallback)
3. Built-in defaults (if no config files found)

Configuration can be overridden by passing a custom config dict to enrichment functions:

```python
from crawl_first.unified_enrichment import enrich_biosample_unified

# Use default config
result = enrich_biosample_unified("sample_001", 40.7128, -74.0060)

# Override specific settings
custom_config = {"osm": {"default_radius_m": 2000}}
result = enrich_biosample_unified("sample_001", 40.7128, -74.0060, config=custom_config)
```

## Dataset Setup

To use local datasets, ensure the following data files are available:

### Elevation
- Copernicus DEM 30m global coverage COG
- USGS Elevation Point Query Service (US locations)

### Land Cover  
- ESA WorldCover 2021 (10m global)
- NLCD 2019/2021 (30m US coverage)

### Soils
- SoilGrids 250m COG files for pH, SOC, sand, silt, clay, bulk density, nitrogen

### Vector Datasets
- GSHHS coastlines (Global Self-consistent Hierarchical High-resolution Geography)
- HydroLAKES global lake polygons
- WWF Terrestrial Ecoregions (TEOW 2017)

### Crosswalk Mappings
- ENVO ontology mappings for land cover classification

Paths in the configuration should be updated to match your local data storage structure.
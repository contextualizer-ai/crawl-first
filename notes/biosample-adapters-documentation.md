# Biosample Data Adapters

*Unified interface for extracting geospatial enrichment inputs from NMDC and GOLD biosample data*

## Overview

The biosample adapters provide a standardized way to extract key enrichment inputs (latitude, longitude, collection_date, textual_location) from both NMDC and GOLD biosample data across different storage formats.

## Key Features

### ✅ **Implemented - MongoDB Adapters**
- **MongoNMDCBiosampleFetcher**: Production-ready MongoDB fetcher for NMDC data
- **MongoGOLDBiosampleFetcher**: Production-ready MongoDB fetcher for GOLD data  
- **Unified Interface**: Single API for both databases

### 🚧 **Stubbed - File Adapters** 
- **FileBiosampleFetcher**: Stub for JSON/JSONL/TSV file processing
- Future implementation for offline/file-based workflows

## Core Components

### 1. BiosampleLocation Dataclass

Standardized container for extracted location data:

```python
@dataclass
class BiosampleLocation:
    # Required for API enrichment
    latitude: Optional[float]
    longitude: Optional[float] 
    collection_date: Optional[str]  # YYYY-MM-DD format
    textual_location: Optional[str]
    
    # Metadata
    sample_id: Optional[str]
    database_source: Optional[str]  # "NMDC" or "GOLD"
    
    # Quality indicators
    coordinate_precision: Optional[int]
    date_precision: Optional[str]
    location_completeness: Optional[float]  # 0.0-1.0
```

### 2. Format-Specific Adapters

**NMDCBiosampleAdapter**: Handles NMDC-specific field parsing
- `lat_lon` field (multiple formats: "42.36 -71.09", {"lat": 42.36, "lon": -71.09}, [42.36, -71.09])
- `collection_date` with ISO datetime parsing
- Location priority: `geo_loc_name` > `geographic_location` > `description`

**GOLDBiosampleAdapter**: Handles GOLD-specific field parsing
- Separate `latitude`/`longitude` fields
- `dateCollected` field parsing
- Location priority: `geoLocation` > `geographicLocation` > `description`

### 3. MongoDB Fetchers

Production-ready database interfaces:

```python
# NMDC MongoDB fetcher
nmdc_fetcher = MongoNMDCBiosampleFetcher(
    connection_string="mongodb://localhost:27017",
    database_name="nmdc",
    collection_name="biosamples"
)

# Fetch enrichable samples only
for location in nmdc_fetcher.fetch_enrichable_locations(limit=1000):
    if location.is_enrichable():
        print(f"Ready for API enrichment: {location.latitude}, {location.longitude}")
```

## Input Format Support

### NMDC Coordinate Formats
```python
# Space-separated string
{"lat_lon": "42.3601 -71.0928"}

# Comma-separated string  
{"lat_lon": "42.3601,-71.0928"}

# Dictionary format
{"lat_lon": {"latitude": 42.3601, "longitude": -71.0928}}

# Array format
{"lat_lon": [42.3601, -71.0928]}

# Separate fields
{"latitude": 42.3601, "longitude": -71.0928}
```

### Date Format Support
```python
# ISO datetime with timezone
{"collection_date": "2023-06-15T10:30:00Z"}

# Date only
{"collection_date": "2023-06-15"}

# Month precision
{"collection_date": "2023-06"}

# Year precision  
{"collection_date": "2023"}
```

## Quality Assessment

Each extracted location includes quality metrics:

- **coordinate_precision**: Decimal places in coordinates
- **date_precision**: "day", "month", or "year"
- **location_completeness**: 0.0-1.0 score based on field availability
- **is_enrichable()**: Boolean check for API readiness

## Usage Examples

### Basic Extraction
```python
from crawl_first.biosample_adapters import NMDCBiosampleAdapter

adapter = NMDCBiosampleAdapter()
sample = {
    "id": "nmdc:bsm-12-example",
    "lat_lon": "42.3601 -71.0928",
    "collection_date": "2023-06-15",
    "geo_loc_name": "Cambridge, MA"
}

location = adapter.extract_location(sample)
print(f"Coordinates: ({location.latitude}, {location.longitude})")
print(f"Date: {location.collection_date}")
print(f"Enrichable: {location.is_enrichable()}")
```

### MongoDB Fetching
```python
from crawl_first.biosample_adapters import MongoNMDCBiosampleFetcher

fetcher = MongoNMDCBiosampleFetcher("mongodb://localhost:27017")
fetcher.connect()

# Get statistics
print(f"Total samples: {fetcher.count_total_samples()}")
print(f"Enrichable: {fetcher.count_enrichable_samples()}")

# Fetch enrichable locations
for location in fetcher.fetch_enrichable_locations(limit=100):
    print(f"Sample {location.sample_id}: {location.textual_location}")
```

### Unified Interface
```python
from crawl_first.biosample_adapters import UnifiedBiosampleFetcher

unified = UnifiedBiosampleFetcher()
unified.configure_nmdc_mongo("mongodb://nmdc-server:27017")
unified.configure_gold_mongo("mongodb://gold-server:27017")

# Fetch from both databases
for location in unified.fetch_enrichable_locations(source="all", limit=1000):
    print(f"{location.database_source}: {location.sample_id}")

# Get enrichment statistics
stats = unified.get_enrichment_statistics()
print(f"Total enrichable samples: {stats['summary']['total_enrichable_samples']}")
```

## API Integration

Extracted locations are ready for geospatial API enrichment:

```python
location = adapter.extract_location(sample)

if location.is_enrichable():
    # Elevation API
    elevation = get_elevation_api(location.latitude, location.longitude)
    
    # Weather API
    weather = get_weather_api(location.latitude, location.longitude, location.collection_date)
    
    # Reverse Geocoding
    address = get_reverse_geocoding(location.latitude, location.longitude)
    
    # Forward Geocoding (validation)
    coords = get_forward_geocoding(location.textual_location)
```

## Database Schema Requirements

### NMDC Expected Fields
- `lat_lon` OR (`latitude` AND `longitude`)
- `collection_date` (optional)
- `geo_loc_name` OR `geographic_location` (optional)
- `id` OR `_id` for sample identification

### GOLD Expected Fields  
- `latitude` AND `longitude`
- `dateCollected` (optional)
- `geoLocation` OR `geographicLocation` (optional)
- `biosampleGoldId` OR `_id` for sample identification

## Dependencies

**Required:**
- `dataclasses` (Python 3.7+)
- `datetime`, `pathlib`, `typing` (standard library)

**Optional:**
- `pymongo` (for MongoDB operations)
- `pandas` (for future file format support)

## Testing

Comprehensive test suite included:

```bash
# Run adapter tests
uv run python -m pytest tests/test_biosample_adapters.py -v

# Run usage examples
uv run python -m crawl_first.biosample_adapter_usage
```

## Future Enhancements

### File Format Support (Planned)
- JSON/JSONL file parsing
- TSV/CSV file support  
- Mixed format detection
- Batch file processing

### Advanced Features (Planned)
- Coordinate validation and correction
- Location name standardization
- Duplicate sample detection
- Performance optimizations for large datasets

## Error Handling

Adapters handle common data issues gracefully:

- **Missing coordinates**: Returns `None` values, `is_enrichable()` returns `False`
- **Invalid date formats**: Returns `None`, continues processing
- **Type mismatches**: Attempts conversion, fallback to `None`
- **Database connection failures**: Raises clear error messages
- **Malformed data**: Logs warnings, continues with available data

This provides a robust foundation for biosample data extraction across different storage formats and data quality scenarios.
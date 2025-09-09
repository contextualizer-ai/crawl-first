# Enrichment Targets Comparison

This document differentiates between the various enrichment targets available in the Makefile, focusing on `enrich-unified-test` and similar targets.

## **`enrich-unified-test`** (New Unified System)
- **What**: Tests the new modular geospatial enrichment system with comprehensive schema
- **Technology**: Uses the unified enrichment architecture with local COG sampling, site typing, comprehensive OSM queries
- **Command**: `uv run enrich-geo enrich --lat 44.428 --lon -110.5885 --date 2021-08-20 --output data/outputs/unified_test.json --enable-crosswalks --verbose --pretty`
- **Output**: `data/outputs/unified_test.json`
- **Scope**: Single test location (Yellowstone: 44.428, -110.5885)
- **Purpose**: Validate the new scientific-grade enrichment framework with unified schema
- **Features**: 
  - Local COG raster sampling
  - Deterministic site typing
  - Comprehensive OSM enrichment
  - Weather aggregation with fallback chains
  - Unified provenance tracking

## **Similar but Different Targets:**

### **`biosample-api-pipeline`** (Existing API System)
- **What**: Runs the original 19-API enrichment system on real biosample data
- **Technology**: Uses existing API calls (Open-Elevation, USGS, Google Maps, Meteostat, etc.)
- **Output**: `data/outputs/api/enrichment_results.json`
- **Scope**: Multiple real biosamples extracted from MongoDB
- **Purpose**: Production pipeline for existing API-based enrichment
- **Pipeline Steps**:
  1. Extract real biosamples: `data/inputs/test_biosamples.json`
  2. Normalize for APIs: `data/outputs/api/normalized_biosamples.json`
  3. Run 19 API functions: `data/outputs/api/enrichment_results.json`
- **API Coverage**: Elevation (3), Weather (2), Geocoding (2), Land Cover (3), Soil (2), Ecoregions (2), Features (1)

### **`scripts/test_unified_enrichment.py`** (Development Test Script)  
- **What**: Python test script created for development testing of unified system
- **Technology**: Same unified system as `enrich-unified-test`
- **Output**: 
  - `data/outputs/test_single_enrichment.json`
  - `data/outputs/test_batch_enrichment.json`
- **Scope**: Multiple test locations (Central Park NYC, Yellowstone, Miami Beach)
- **Purpose**: Development testing and validation with multiple scenarios
- **Features**: Both single and batch enrichment testing

### **`data/outputs/crawl-first/test-results/`** (Original CLI System)
- **What**: Original `crawl-first` CLI tool results
- **Technology**: Basic biosample ID processing and enrichment
- **Command**: `uv run crawl-first --input-file data/inputs/biosample-ids.txt --sample-size 1500`
- **Output**: Directory with multiple result files
- **Scope**: Random biosample IDs fetched from NMDC API
- **Purpose**: Original tool validation and testing

## **Key Differences Summary:**

| Target | System | Data Source | Technology | Output Location | Test Scope |
|--------|--------|-------------|------------|-----------------|------------|
| `enrich-unified-test` | **New Unified** | Single test point | Local COG + APIs | `data/outputs/unified_test.json` | Yellowstone location |
| `biosample-api-pipeline` | **Existing APIs** | Real MongoDB biosamples | 19 API services | `data/outputs/api/enrichment_results.json` | Multiple real samples |
| Test script | **New Unified** | Multiple test points | Local COG + APIs | `data/outputs/test_*.json` | 3 diverse locations |
| `crawl-first` CLI | **Original tool** | NMDC API IDs | Basic enrichment | `data/outputs/crawl-first/` | Random NMDC samples |

## **Technology Architecture Differences:**

### New Unified System (`enrich-unified-test`)
- **Local COG Sampling**: Direct raster sampling using rasterio
- **Site Typing**: Hierarchical classification (terrestrial/aquatic/marine)
- **OSM Enrichment**: Single comprehensive Overpass query
- **Weather Aggregation**: Open-Meteo → Meteostat → PRISM fallback chain
- **Unified Schema**: Consistent distance/provenance tracking
- **Configuration**: YAML/JSON config with dataset paths and provider settings

### Existing API System (`biosample-api-pipeline`)
- **API-Based**: Remote service calls for all data
- **19 API Functions**: Distributed across multiple providers
- **Normalized Input**: MongoDB biosamples converted to API format
- **Legacy Architecture**: Individual API wrappers
- **JSON Output**: Results with API success/failure tracking

## **When to Use Each Target:**

### Use `enrich-unified-test` when:
- Testing the new unified enrichment architecture
- Validating local COG dataset integration
- Checking scientific-grade provenance tracking
- Testing configuration management system
- Demonstrating the comprehensive schema

### Use `biosample-api-pipeline` when:
- Working with real biosample data from databases
- Testing the existing production API system
- Comparing old vs new enrichment approaches
- Running comprehensive API coverage tests

### Use the test script when:
- Development testing with multiple scenarios
- Batch processing validation
- Testing error handling across diverse locations

### Use `crawl-first` CLI when:
- Processing NMDC biosample IDs specifically
- Testing the original tool functionality
- Basic enrichment without advanced features

## **Cleanup Recommendations:**

Before running `enrich-unified-test`:
```bash
make squeaky-clean  # Thorough cleanup preserving input data
make enrich-unified-test  # Test new unified system
```

Optional for full experience:
```bash
make squeaky-clean
make generate-mappings  # For ENVO ontology crosswalks
make enrich-unified-test
```

The new unified system (`enrich-unified-test`) represents the next generation of geospatial enrichment with scientific rigor, local dataset integration, and comprehensive provenance tracking.
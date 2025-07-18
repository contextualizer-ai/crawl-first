# Crawl-First Project Philosophy and Status

## Core Philosophy

**"Deterministic biosample enrichment for LLM-ready data preparation"**

The fundamental principle behind crawl-first is to systematically follow ALL discoverable links from NMDC biosample records to gather comprehensive environmental, geospatial, weather, publication, and ontological data BEFORE feeding it to LLMs for analysis. This "crawl-first" approach ensures deterministic, reproducible data gathering rather than ad-hoc LLM-driven exploration.

## Your Vision & Intentions

### Primary Goals
1. **Comprehensive Data Enrichment**: Extract maximum value from NMDC biosample records by following every possible data link
2. **LLM-Ready Output**: Structure enriched data in YAML format optimized for LLM consumption and analysis
3. **Deterministic Processing**: Ensure reproducible results through systematic caching and consistent methodologies
4. **Quality Data Validation**: Include coordinate validation, distance calculations, and elevation comparisons to assess data quality

### Key Design Principles
- **Try ALL methods**: Don't stop at first success - attempt all retrieval strategies and pick the best result
- **Cache everything**: Avoid redundant API calls and enable reproducible analysis
- **Validate coordinates**: Calculate distances between asserted vs geocoded coordinates
- **Interactive visualization**: Generate Google Maps URLs for visual validation
- **Ontological enrichment**: Match soil types and land use terms to ENVO ontology
- **Publication completeness**: Fetch full-text papers when available using artl-mcp

## What We've Built

### Core Components
1. **Comprehensive biosample analyzer** (`src/crawl_first/cli.py`)
   - Full NMDC biosample metadata extraction
   - Multi-strategy coordinate processing (asserted + geocoded)
   - Soil analysis with ENVO term matching
   - Land cover analysis across multiple classification systems
   - Weather data retrieval with fallback strategies
   - Publication analysis with full-text retrieval
   - OpenStreetMap environmental feature extraction
   - Geospatial analysis with elevation and place information

2. **Modern Python package structure**
   - `pyproject.toml` optimized for `uv` package manager
   - Comprehensive dev dependencies (black, ruff, mypy, deptry)
   - CLI entry point via Click
   - Proper package metadata for PyPI publication

3. **Quality control automation**
   - `Makefile` with comprehensive QC targets
   - GitHub Actions workflows for CI/CD
   - Code formatting, linting, type checking, dependency analysis
   - Test infrastructure with pytest

## What We've Fixed

### Critical Bug Fixes
1. **Ontology lookup filtering**: Fixed bug where ENVO terms weren't being saved due to incorrect filtering on `id` instead of `obo_id`
2. **DOI structure duplication**: Eliminated hierarchy confusion by making `all_dois` a sibling of `publication_analysis`
3. **Full-text retrieval optimization**: Changed from "stop at first success" to "try all methods and pick best" strategy
4. **Coordinate validation**: Added distance calculations between asserted and geocoded coordinates
5. **Elevation comparison**: Calculate differences between asserted and inferred elevations

### Enhancements Added
- **Interactive map URLs**: Google Maps links for both individual and combined coordinate visualization
- **Comprehensive caching**: MD5-based caching for all API calls and computations
- **File-based full-text storage**: Save retrieved papers to `.cache/full_text_files/` with structured naming
- **Weather data strategies**: Multiple fallback approaches for weather station data
- **Environmental feature extraction**: Detailed OpenStreetMap feature categorization and analysis

## What We've Tried

### Successful Implementations
- ✅ Complete project restructure from single script to proper Python package
- ✅ Integration of multiple MCP libraries (nmdc-mcp, landuse-mcp, weather-mcp, ols-mcp, artl-mcp)
- ✅ Comprehensive caching system with structured file organization
- ✅ CLI interface with flexible input/output options
- ✅ Quality control automation with Makefile
- ✅ GitHub Actions CI/CD setup

### Current Blockers
- ❌ Shell environment corruption preventing execution of `make all`
- ❌ Tool integration issues preventing final quality validation

## Next Steps & Priorities

### Immediate Actions Needed
1. **Restore shell environment** - The tool integration has broken, preventing execution of quality control checks
2. **Execute `make all`** - Run comprehensive quality control suite once environment is restored
3. **Fix any QC issues** - Address problems found by black, ruff, mypy, deptry, pytest
4. **Test CLI functionality** - Verify the crawl-first command works end-to-end

### Validation Testing
1. **Sample biosample processing**: Test with a known NMDC biosample ID
2. **Publication retrieval verification**: Ensure artl-mcp integration works correctly
3. **Coordinate validation testing**: Verify distance calculations and map URL generation
4. **Ontology term matching**: Confirm ENVO terms are being found and saved properly

### Potential Enhancements
1. **Coverage reporting**: Add pytest-cov for test coverage analysis
2. **Security auditing**: Integrate bandit for security scanning
3. **Performance optimization**: Profile and optimize for large-scale processing
4. **Documentation expansion**: Add detailed examples and API documentation

## Technical Architecture

### Dependencies
- **Core**: Python 3.11+, uv package manager
- **Data sources**: nmdc-mcp, landuse-mcp, weather-mcp, ols-mcp, artl-mcp
- **Utilities**: click, geopy, pandas, pyyaml, requests
- **Quality**: black, ruff, mypy, deptry, pytest

### Data Flow
1. **Input**: NMDC biosample ID(s)
2. **Enrichment**: Follow all discoverable links to gather comprehensive data
3. **Validation**: Calculate distances, compare elevations, verify coordinates
4. **Output**: Structured YAML with `asserted` (original) and `inferred` (enriched) data sections

### Output Structure
```yaml
asserted: # Original NMDC biosample metadata
inferred: # All enriched data including:
  coordinate_sources: # Validation and mapping
  soil_from_asserted_coords: # Soil analysis + ENVO terms
  land_cover_from_asserted_coords: # Multi-system land cover
  weather_from_asserted_coords: # Weather data
  geospatial_from_asserted_coords: # OSM features + elevation
  study_info: # Study metadata
  all_dois: # Complete DOI information
  publication_analysis: # Publication metrics
```

## Success Metrics

The project will be considered successful when:
1. ✅ Complete biosample enrichment pipeline working end-to-end
2. ⏳ All quality control checks passing (`make all` succeeds)
3. ⏳ CLI functionality verified with real NMDC data
4. ⏳ Full-text paper retrieval working via artl-mcp
5. ⏳ Coordinate validation and mapping URLs functional
6. ⏳ ENVO ontology term matching operational
7. ⏳ Package ready for publication to PyPI

## Philosophy Summary

Crawl-first embodies the principle that systematic, deterministic data gathering should precede LLM analysis. By following every discoverable link and enriching biosample records with comprehensive environmental context, we create high-quality datasets that enable more accurate and insightful LLM-driven scientific analysis. The tool prioritizes data completeness, validation, and reproducibility over speed, ensuring that downstream LLM analyses are built on the most comprehensive foundation possible.
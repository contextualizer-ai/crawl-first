# crawl-first

**Deterministic biosample enrichment for LLM-ready data preparation**

## Overview

`crawl-first` systematically follows all discoverable links from NMDC biosample records to gather comprehensive environmental, geospatial, weather, publication, and ontological data. This deterministic approach ensures complete data enrichment before downstream LLM analysis.

## Philosophy

Instead of letting LLMs make API calls or guess at missing data, `crawl-first` embodies the principle: **gather everything first, analyze second**. This ensures reproducible, comprehensive datasets for AI analysis.

## Features

- **Complete biosample enrichment**: Follows all linked data sources
- **Geospatial analysis**: Coordinates, elevation, land cover, soil types
- **Weather integration**: Historical weather data for sample collection dates  
- **Publication tracking**: DOI resolution, full-text retrieval when available
- **Ontology enrichment**: ENVO term matching for environmental descriptors
- **Quality validation**: Distance/elevation comparisons between data sources
- **Interactive maps**: Generated URLs for coordinate validation
- **Comprehensive caching**: Prevents redundant API calls

## Installation

```bash
pip install crawl-first
```

## Usage

### Single biosample
```bash
crawl-first --biosample-id nmdc:bsm-11-abc123 --email your-email@domain.com --output-file result.yaml
```

### Multiple biosamples
```bash
crawl-first --input-file biosample_ids.txt --email your-email@domain.com --output-dir results/
```

### Sample from large dataset
```bash
crawl-first --input-file all_biosamples.txt --sample-size 50 --email your-email@domain.com --output-dir sample_results/
```

## Output Structure

Each enriched biosample contains:
- **Asserted data**: Original NMDC biosample record
- **Inferred data**: All discovered linked information
  - Soil analysis with ENVO ontology terms
  - Land cover classification across multiple systems
  - Weather data from collection date
  - Publication metadata and full-text when available
  - Geospatial features within configurable radius
  - Coordinate validation and distance calculations

## Data Sources

- **NMDC API**: Biosample and study metadata
- **Land Use MCP**: Land cover classification systems
- **Weather MCP**: Historical meteorological data
- **OLS MCP**: Ontology term resolution
- **ARTL MCP**: Publication and full-text retrieval
- **OpenStreetMap**: Environmental feature mapping
- **Elevation APIs**: Topographic data validation

## Development

```bash
git clone https://github.com/contextualizer-ai/crawl-first.git
cd crawl-first
pip install -e ".[dev]"
```

## License

MIT License - see LICENSE file for details.
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

### Using uv (recommended)
```bash
uv add crawl-first
```

### Using pip
```bash
pip install crawl-first
```

## Usage

### Single biosample
```bash
uv run crawl-first --biosample-id nmdc:bsm-11-abc123 --email your-email@domain.com --output-file result.yaml
```

### Multiple biosamples
```bash
uv run crawl-first --input-file biosample_ids.txt --email your-email@domain.com --output-dir results/
```

### Sample from large dataset
```bash
uv run crawl-first --input-file all_biosamples.txt --sample-size 50 --email your-email@domain.com --output-dir sample_results/
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

# Install with uv
uv sync
uv pip install -e ".[dev]"

# Or create a virtual environment and install
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Code formatting and linting
uv run black .
uv run ruff check .
uv run mypy .
uv run deptry .
```

### MCP Configuration for Claude Integration

The repository includes Makefile targets that integrate with Claude Code for testing and automation. These targets require a properly configured `.mcp.json` file in your Claude configuration directory:

```json
{
  "mcpServers": {
    "weather-context-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["weather-context-mcp"]
    },
    "landuse-mcp": {
      "type": "stdio", 
      "command": "uvx",
      "args": ["landuse-mcp"]
    },
    "nmdc-mcp": {
      "type": "stdio",
      "command": "uvx", 
      "args": ["nmdc-mcp"]
    },
    "ols-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["ols-mcp"]
    },
    "artl-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["artl-mcp"]
    }
  }
}
```

**Note**: Makefile targets like `claude-weather-test.txt` and `random-ids-test.txt` will not work without proper MCP server configuration in Claude.

## License

MIT License - see LICENSE file for details.
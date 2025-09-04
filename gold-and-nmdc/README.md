# GOLD and NMDC Schema Comparison and Enrichment Analysis

Automated pipeline for comparing NMDC and GOLD biosample schemas using AI/LLMs, with a focus on identifying environmental data enrichment opportunities.

## Directory Structure

```
gold-and-nmdc/
├── scripts/                          # Python analysis scripts
│   ├── extract_nmdc_biosample_slots.py   # Extract NMDC schema using LinkML
│   ├── infer_schema.py                    # Infer MongoDB schemas using genson
│   └── infer_stats.py                     # Generate field coverage statistics
├── prompts/                           # AI analysis prompts
│   ├── schema-comparison-prompt.txt       # Schema comparison prompt
│   └── enrichment-analysis-prompt.txt    # Enrichment opportunity analysis
├── outputs/                           # Generated analysis results
│   ├── nmdc_biosample_slots.json          # NMDC schema slots
│   ├── gold_biosample_schema.json         # GOLD schema (inferred)
│   ├── *_biosample_stats.csv              # Field coverage statistics
│   ├── schema_comparison.json             # AI-powered schema mapping
│   └── enrichment_analysis.json          # Enrichment opportunities
├── Makefile                          # Automation pipeline
├── empty-mcp-config.json             # Claude Code MCP configuration
└── README.md                         # This file
```

## Quick Start

### Prerequisites

- Python with `uv` package manager
- MongoDB access to NMDC and GOLD databases
- Claude Code CLI tool

### Run Complete Analysis

```bash
cd gold-and-nmdc
make all
```

### Key Targets

- `make outputs/schema_comparison.json` - Schema comparison and field mapping
- `make outputs/enrichment_analysis.json` - Enrichment opportunity identification
- `make outputs/nmdc_biosample_stats.csv` - NMDC field coverage statistics
- `make outputs/gold_biosample_stats.csv` - GOLD field coverage statistics

## Features

### 1. Schema Extraction
- **NMDC**: Uses LinkML SchemaView with induction for complete class definitions
- **GOLD**: MongoDB schema inference using genson library with JSON Schema generation

### 2. Field Statistics
- Compass-like field coverage analysis
- Proper null-value handling (only counting non-null values)
- CSV and Markdown output formats

### 3. AI-Powered Schema Comparison
- Semantic field matching with similarity scores
- Relationship categorization (exact_match, near_match, semantic_match)
- Multi-field mapping support (e.g., lat_lon → latitude+longitude)

### 4. Enrichment Analysis
- **Weather Data**: Open-Meteo API for historical weather using coordinates + dates
- **Elevation**: Google Elevation API for missing elevation data
- **Land Cover**: NLCD/ESA WorldCover for terrestrial habitat context
- **Marine Biogeography**: Longhurst provinces via GeoTIFF lookup
- **Ontological Inference**: OpenStreetMap → ENVO term conversion workflows

## Key Technical Features

### LinkML Integration
Uses `.attributes` (not `.slots`) after induction for complete class definitions including inherited slots.

### MongoDB Schema Inference
Handles ObjectId serialization with proper JSON conversion and generates Draft 2020-12 compliant schemas.

### Coverage Calculation
Accurate field coverage percentages that never exceed 100% by only counting documents with non-null values.

### Claude Code Integration
Non-interactive AI analysis using prompt files and structured JSON output with proper MCP configuration.

## Configuration

Environment variables can be set to customize the pipeline:

```bash
export NMDC_SCHEMA_URL="https://raw.githubusercontent.com/microbiomedata/nmdc-schema/refs/heads/main/nmdc_schema/nmdc_materialized_patterns.yaml"
export GOLD_CSV_PATH="gold_metadata.flattened_biosamples.csv"
export MONGO_URI="mongodb://user:pass@localhost:27778/..."
export SAMPLE_SIZE="50000"
```

## Enrichment Strategies

The analysis identifies several types of enrichment opportunities:

1. **High-to-Low Coverage**: Use well-populated fields to enrich sparse ones
2. **Cross-Database**: Transfer unique data between NMDC and GOLD
3. **API Enrichment**: External services for missing environmental data
4. **Ontological Inference**: Geographic context → environmental classifications

## ENVO Ontology Mappings

### Overview

The `scripts/mappings/` directory contains curated mappings from geospatial classification systems to ENVO (Environmental Ontology) terms. These mappings enable semantic enrichment of biosample metadata by converting raw classification codes to standardized environmental terms.

### Mapping Files

- **`soilgrids_fao_to_envo.json`** - FAO soil types → ENVO soil terms (29 mappings)
- **`esa_worldcover_to_envo.json`** - ESA WorldCover land cover → ENVO biome terms (11 mappings)  
- **`nlcd_to_envo.json`** - US NLCD land cover → ENVO habitat terms (20 mappings)

### Mapping Structure

Each mapping file contains entries with this structure:

```json
{
  "classification_code": {
    "id": "ENVO:xxxxxxx",
    "label": "envo term label",
    "term": "Original classification term",
    "confidence": "deterministic|manual_curated|ai_assisted",
    "source": "Source classification system",
    "description": "Human-readable description"
  }
}
```

### Confidence Levels

- **`deterministic`** - Official mappings based on established standards
- **`manual_curated`** - Expert-reviewed mappings with high confidence
- **`ai_assisted`** - AI-generated mappings requiring human validation

### Lifecycle Management

#### Generation
```bash
# Generate all mappings (overwrites existing files)
uv run python scripts/generate_mappings.py

# Generate only deterministic mappings (no AI)
uv run python scripts/generate_mappings.py --no-ai
```

#### Usage
Mappings are loaded by `scripts/alternative_geospatial.py`:
```python
from scripts.alternative_geospatial import load_mapping_file

# Load mappings at runtime
esa_mappings = load_mapping_file("esa_worldcover_to_envo")
soil_mappings = load_mapping_file("soilgrids_fao_to_envo")
```

#### Maintenance
```bash
# Remove all mapping files
make clean-mappings

# Regenerate specific mappings as needed
uv run python scripts/generate_mappings.py
```

### Integration Points

**Current Usage:**
- `scripts/alternative_geospatial.py` - applies mappings to API responses
- Classification codes are automatically converted to ENVO terms with confidence scores

**Future Integration:**
- Can be extended to `scripts/geospatial_enrichment.py` for OpenStreetMap tag mapping
- Supports biosample metadata semantic enrichment workflows
- Enables AI-powered biosample validation with ontological context

### Best Practices

1. **Preserve existing mappings** unless specifically regenerating
2. **Validate AI-assisted mappings** before production use
3. **Use deterministic mappings** for stable, repeatable results
4. **Document mapping sources** for reproducibility
5. **Version control mapping changes** to track evolution

## Output Files

- **Schema Files**: Complete schema definitions and field mappings
- **Statistics**: Field coverage and population analysis
- **Comparison**: AI-powered semantic field matching
- **Enrichment**: Implementation workflows for data enhancement

## Usage Examples

```bash
# Schema comparison only
make outputs/schema_comparison.json

# Enrichment analysis only (with existing dependencies)
make -t outputs/nmdc_biosample_stats.csv outputs/gold_biosample_stats.csv outputs/schema_comparison.json outputs/enrichment_analysis.json

# Clean and rebuild everything
make clean && make all
```

## Dependencies

- Python packages: `linkml`, `genson`, `pandas`, `pymongo`, `tabulate`
- External tools: `curl`, `jq`, `claude` (Claude Code CLI)
- MongoDB access to NMDC and GOLD databases
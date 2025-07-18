# Quick Start Guide

## Prerequisites

- Python 3.11+ 
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install crawl-first
uv add crawl-first
```

## Basic Usage

```bash
# Process a single biosample
uv run crawl-first \
  --biosample-id nmdc:bsm-11-abc123 \
  --email your-email@domain.com \
  --output-file my_sample.yaml

# Process multiple biosamples
uv run crawl-first \
  --input-file biosample_list.txt \
  --email your-email@domain.com \
  --output-dir results/ \
  --verbose
```

## Development Setup

```bash
# Clone and setup
git clone https://github.com/contextualizer-ai/crawl-first.git
cd crawl-first

# Install dependencies
uv sync --dev

# Verify installation
uv run crawl-first --help

# Run tests
uv run pytest

# Code quality checks
uv run black .
uv run ruff check .
uv run mypy .
uv run deptry .
```

## Common Commands

```bash
# Format and check code
uv run black . && uv run ruff check . && uv run mypy .

# Run a small test sample
uv run crawl-first \
  --input-file test_ids.txt \
  --sample-size 5 \
  --email your-email@domain.com \
  --output-dir test_results/ \
  --verbose

# Process with custom search radius
uv run crawl-first \
  --biosample-id nmdc:bsm-11-abc123 \
  --email your-email@domain.com \
  --output-file result.yaml \
  --search-radius 2000  # 2km radius for geospatial features
```

## Output

Each processed biosample generates a comprehensive YAML file containing:
- **Asserted data**: Original NMDC biosample metadata
- **Inferred data**: All enriched information including:
  - Soil analysis with ENVO ontology terms
  - Land cover across multiple classification systems  
  - Weather data from collection date
  - Publication metadata and full-text when available
  - Geospatial features and interactive map URLs
  - Coordinate validation and distance calculations

Perfect for feeding into LLM analysis pipelines!
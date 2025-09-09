# Makefile Cleanup Analysis: Targets Using Archived Directories

## Overview
This document analyzes Makefile targets that reference directories we've moved to the `nmdc-with-ornl/` archival location or that may need cleanup.

## Moved Directories
- `data/outputs/claude/` → `nmdc-with-ornl/data/outputs/claude/`
- `data/outputs/crawl-first/` → `nmdc-with-ornl/data/outputs/crawl-first/`
- `data/outputs/tests/` → `nmdc-with-ornl/data/outputs/tests/`
- `data/outputs/validation*` → `nmdc-with-ornl/data/outputs/validation*`

## Deleted Directories
- `data/samples/` (was empty)
- `logs/` (only had .DS_Store)

---

## Makefile Targets Affected

### 1. Claude MCP Testing Targets
**Status**: Likely candidates for archival or deletion

```makefile
# Test Claude weather query
data/outputs/claude/weather-test.txt: | setup-dirs
	@if [ ! -f $@ ]; then \
		echo "🌤️  Testing Claude weather query..."; \
		time claude \
			--mcp-config .mcp.json \
			--dangerously-skip-permissions \
			--print "what was the weather like at the Statue of Liberty (latitude 40.6892, longitude -74.0445) on January 1st, 2025? what resources did you use to get my answer?" > $@; \
		echo "✅ Claude weather test saved to $@"; \
	else \
		echo "✅ Weather test file already exists: $@"; \
	fi

# Test Claude MCP server availability
data/outputs/claude/mcp-servers-test.txt: | setup-dirs
	@echo "🔧 Testing Claude MCP server availability..."
	claude \
		--dangerously-skip-permissions \
		--print "List all available MCP servers and tools you have access to. Be specific about what servers are loaded." > $@
	@echo "✅ Claude MCP servers test saved to $@"

# Test Claude landuse MCP
data/outputs/claude/landuse-mcp-test.txt: | setup-dirs
	@echo "🌱 Testing Claude landuse MCP..."
	claude \
		--dangerously-skip-permissions \
		--print "Use the landuse MCP to get land cover data for coordinates 40.7128, -74.0060 for date range 2020-01-01 to 2020-12-31. What MCP tools did you use?" > $@
	@echo "✅ Claude landuse MCP test saved to $@"

# MCP diagnostic tests - Claude interactions with MCP servers (includes weather test)
test-mcp: data/outputs/claude/weather-test.txt data/outputs/claude/mcp-servers-test.txt data/outputs/claude/landuse-mcp-test.txt
	@echo "🔧 MCP diagnostic tests complete"
```

### 2. Original Crawl-First Application Testing
**Status**: Likely candidates for archival - replaced by enhanced adapters

```makefile
# Process random biosample IDs
data/outputs/tests/random-ids-test.txt: data/inputs/biosample-ids.txt process_random_ids.sh
	@echo "🎲 Processing random biosample IDs..."
	./process_random_ids.sh --file data/inputs/biosample-ids.txt --count 5 > $@ || true
	@echo "✅ Random ID processing saved to $@"

# Run crawl-first on 10 random biosample IDs
data/outputs/crawl-first/test-results/: data/inputs/biosample-ids.txt
	@echo "🧬 Running crawl-first on 10 random biosample IDs..."
	mkdir -p $@
	uv run crawl-first \
		--input-file data/inputs/biosample-ids.txt \
		--sample-size 1500 \
		--email MAM@lbl.gov \
		--output-dir $@ \
		--verbose
	@echo "✅ Crawl-first results saved to $@"

# Full test suite - code quality + data processing + application testing (excludes slow Claude MCP tests)
full-test: all setup-dirs data/inputs/biosample-ids.txt data/outputs/tests/random-ids-test.txt data/outputs/crawl-first/test-results/ check-cli
	@echo "🎯 Full test suite complete - all code quality checks and application tests passed"
```

### 3. AI Validation Targets
**Status**: Experimental features - candidates for archival

```makefile
# Validate biosample enrichment data with AI - generates JSON output file (with streaming)
data/outputs/validation-results.json: archives/data/outputs/crawl-first/test-results/
	@echo "🔬 Running biosample validation with CBORG AI (streaming mode)..."
	@echo "📝 Output will be saved to: $@"
	@echo "💾 Individual results will be streamed to: data/outputs/validation-streaming/"
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "❌ Error: OPENAI_API_KEY not set"; exit 1; fi
	@mkdir -p $(dir $@)
	@mkdir -p data/outputs/validation-streaming/
	uv run python src/crawl_first/validation_agent.py \
		--results-dir $< \
		--max-samples 5 \
		--model "anthropic/claude-sonnet" \
		--base-url https://api.cborg.lbl.gov/v1 \
		--stream-dir data/outputs/validation-streaming/ \
		--output $@

# Custom validation with configurable parameters (with streaming)
data/outputs/validation-results-custom.json: archives/data/outputs/crawl-first/test-results/
	@echo "🔬 Running custom biosample validation (streaming mode)..."
	@echo "📝 Output will be saved to: $@"
	@echo "💾 Individual results will be streamed to: data/outputs/validation-streaming-custom/"
	@echo "💡 Usage: make $@ MODEL='model-name' MAX_SAMPLES=10 BASE_URL='https://api.example.com/v1'"
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "❌ Error: OPENAI_API_KEY not set"; exit 1; fi
	@mkdir -p $(dir $@)
	@mkdir -p data/outputs/validation-streaming-custom/
	uv run python src/crawl_first/validation_agent.py \
		--results-dir $< \
		--max-samples $(or $(MAX_SAMPLES),5) \
		--model "$(or $(MODEL),anthropic/claude-sonnet)" \
		--base-url $(or $(BASE_URL),https://api.cborg.lbl.gov/v1) \
		--stream-dir data/outputs/validation-streaming-custom/ \
		--output $@

# Resume interrupted validation
resume-validation: archives/data/outputs/crawl-first/test-results/
	@echo "🔄 Resuming validation from streaming directory..."
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "❌ Error: OPENAI_API_KEY not set"; exit 1; fi
	uv run python src/crawl_first/validation_agent.py \
		--results-dir $< \
		--max-samples $(or $(MAX_SAMPLES),50) \
		--model "$(or $(MODEL),anthropic/claude-sonnet)" \
		--base-url $(or $(BASE_URL),https://api.cborg.lbl.gov/v1) \
		--stream-dir data/outputs/validation-streaming/ \
		--resume \
		--output data/outputs/validation-results-resumed.json

# Convenience aliases for backward compatibility
validate-biosamples: data/outputs/validation-results.json
validate-biosamples-custom: data/outputs/validation-results-custom.json
```

---

## Recommendations

### Core Targets to Keep (Updated paths)
These are essential for current biosample adapter functionality:
- All `data/outputs/adapters/*` targets (already updated)
- All `data/outputs/schema/*` targets (current schema analysis)

### Candidates for Moving to nmdc-with-ornl.Makefile
1. **Claude MCP tests**: `test-mcp` and related targets
2. **Original crawl-first**: `data/outputs/crawl-first/test-results/` target
3. **Random ID processing**: `data/outputs/tests/random-ids-test.txt` target
4. **AI validation**: All validation targets
5. **Full test**: May need to be simplified to exclude archival dependencies

### Targets That Need Path Updates
If keeping any of the above, paths need to be updated from:
- `data/outputs/claude/` → `nmdc-with-ornl/data/outputs/claude/`
- `data/outputs/crawl-first/` → `nmdc-with-ornl/data/outputs/crawl-first/`
- `data/outputs/tests/` → `nmdc-with-ornl/data/outputs/tests/`
- `data/outputs/validation*` → `nmdc-with-ornl/data/outputs/validation*`

### Possible New Simplified Main Targets
```makefile
# Simplified core development
dev-core: format lint typecheck test
	@echo "🚀 Core development cycle complete"

# Core functionality test
test-core: test-adapters test-enhanced-adapters
	@echo "🧪 Core adapter tests complete"
```

---

## Source File Analysis: Core vs. Archival

Based on modification dates and current functionality priorities, here's the categorization of source files:

### **Core Current Functionality (Keep in Main Repo)**

#### **Enhanced Biosample Adapters** (Sep 8, 2024 - Most Recent)
- `biosample_adapters.py` - **Core enhanced adapters with ID normalization** 
- `extract_real_biosamples.py` - Real MongoDB data extraction
- `test_enhanced_adapters_with_data.py` - Real data testing

#### **Schema Analysis** (Sep 6-8, 2024)
- `infer_schema.py` - MongoDB schema inference from real data
- `infer_stats.py` - Field statistics generation (Compass-like)
- `extract_nmdc_biosample_slots.py` - NMDC schema extraction via LinkML

#### **Geospatial API & Enrichment** (Sep 6, 2024)
- `google_plus_api_functions.py` - **API functions organized by source** (Google, USGS, etc.)
- `unified_enrichment.py` - **Unified geospatial enrichment pipeline** (used by main Makefile targets)
- `unified_geospatial.py` - Unified geospatial processing
- `geospatial_enrichment.py` - Core geospatial enrichment using working APIs
- `coordinate_utils.py` - Distance calculations and coordinate transformations
- `enrich_location.py` - Location enrichment functions

#### **Mapping/Crosswalk Support** (Sep 6-8, 2024)
- `generate_mappings.py` - Ontology mappings generation (used by Makefile)
- `compile_crosswalks.py` - SSSOM crosswalks compilation (used by Makefile)  
- `crosswalk_loader.py` - Crosswalk loading utilities

---

### **Candidates for nmdc-with-ornl Archival**

#### **Original Crawl-First Application** (Jul-Aug 2024)
- `cli.py` - Original command-line interface (Aug 28)
- `biosample.py` - Original biosample class (Aug 28)
- `validation_agent.py` - AI validation with CBORG (Aug 28)

#### **Legacy Geospatial & Utilities** (Jul 18-20, 2024)
- `geospatial.py` - Original geospatial functions (Jul 19)
- `osm.py` - OpenStreetMap integration (Jul 18)
- `yaml_utils.py` - YAML processing utilities (Jul 18)
- `logging_utils.py` - Logging configuration (Jul 20)
- `analysis.py` - Legacy analysis functions (Jul 20)
- `cache.py` - Caching utilities (Jul 20)
- `direct_retrieval.py` - Direct data retrieval (Jul 20)

#### **Legacy Workflow Components** (Sep 6, 2024)
*These are older/alternative approaches that may be candidates for archival:*
- `alternative_geospatial.py` - Alternative geospatial approaches
- `batch_enrich_locations.py` - Batch location enrichment  
- `biosample_adapter_usage.py` - Original adapter demonstrations
- `adapter_demonstration.py` - Adapter usage examples
- `download_ecoregions.py` - Ecoregion data downloading
- `mongodb_connection_test.py` - MongoDB connection testing
- `test_gold_adapter.py` - Original GOLD adapter tests (superseded by enhanced version)
- `test_nmdc_adapter.py` - Original NMDC adapter tests (superseded by enhanced version)
- `validate_pipeline.py` - Pipeline validation

---

## Revised Summary

**Current Core (~14 files):** 
- Enhanced biosample adapters (3 files)
- Schema analysis (3 files) 
- **Geospatial API & enrichment (6 files) - Still actively used by Makefile targets**
- Mapping/crosswalk support (3 files)

**Archival Candidates (~18 files):** 
- Original application + legacy utilities (10 files, Jul-Aug)
- Alternative/older workflow components (8 files from Sep 6 batch)

**Key insight:** The unified enrichment pipeline (`unified_enrichment.py`, `geospatial_enrichment.py`, etc.) is still core functionality actively used by current Makefile targets like `enrich-unified-test`, `all-unified`, and `dev-unified`. Only the older alternative approaches and superseded adapter tests should be considered for archival.
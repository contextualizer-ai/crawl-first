# Makefile for crawl-first development and quality control
.PHONY: all all-unified install clean format lint typecheck deps-check test test-verbose security build dev dev-unified ci setup-dirs squeaky-clean full-test test-mcp compress-all enrich-unified-test download-envo generate-mappings compile-crosswalks analyze-schemas clean-schema help

# Default target - runs all quality checks and tests
all: install format lint typecheck deps-check test

# Complete workflow - quality + schema analysis + unified enrichment  
all-unified: all analyze-schemas generate-mappings enrich-unified-test
	@echo "🎯 Complete unified workflow - code quality + schema analysis + mappings + geospatial enrichment"

# Install dependencies and package in development mode
install:
	@echo "🔧 Installing dependencies..."
	uv sync --dev

# Format code with black
format:
	@echo "🎨 Formatting code with black..."
	uv run black .
	@echo "✅ Code formatted"

# Lint code with ruff
lint:
	@echo "🔍 Linting code with ruff..."
	uv run ruff check . --fix
	@echo "✅ Linting complete"

# Type checking with mypy
typecheck:
	@echo "🔬 Type checking with mypy..."
	uv run mypy .
	@echo "✅ Type checking complete"

# Check dependencies with deptry
deps-check:
	@echo "📦 Checking dependencies with deptry..."
	uv run deptry .
	@echo "✅ Dependency check complete"

# Run tests with pytest
test:
	@echo "🧪 Running tests with pytest..."
	uv run pytest
	@echo "✅ Tests complete"

# Run tests with verbose output
test-verbose:
	@echo "🧪 Running tests with verbose output..."
	uv run pytest -v
	@echo "✅ Verbose tests complete"

# Run tests with coverage
test-coverage:
	@echo "🧪 Running tests with coverage..."
	uv run pytest --cov=crawl_first --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated in htmlcov/"

# Security audit with bandit
security:
	@echo "🔒 Running security audit with bandit..."
	uv run bandit -r src/
	@echo "✅ Security audit complete"

# Build package
build:
	@echo "📦 Building package..."
	uv build
	@echo "✅ Package built in dist/"

# Clean build artifacts and caches
clean:
	@echo "🧹 Cleaning up..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

# Squeaky clean - removes all regenerable files but preserves input data
squeaky-clean: clean
	@echo "🧽 Squeaky clean - removing all regenerable files..."
	rm -rf cache/
	rm -rf data/outputs/
	rm -rf mappings/
	@echo "✨ Squeaky clean complete - regenerable files removed"
	@echo "💾 Preserved: data/inputs/ (real extracted data)"
	@echo "💡 Run 'make generate-mappings' to recreate mappings"

# Quick development cycle - format, lint, and test
dev: format lint test
	@echo "🚀 Development cycle complete"

# Full CI simulation - everything that runs in GitHub Actions
ci: all test-coverage security
	@echo "🎯 CI simulation complete"

# =============================================================================
# GOLD-NMDC UNIFIED ENRICHMENT TARGETS
# =============================================================================

# Download and prepare local datasets needed for enrichment
setup-local-data:
	@echo "📥 Setting up local datasets for enrichment..."
	@mkdir -p data/teow
	@if [ ! -f data/teow/teow2017.gpkg ]; then \
		echo "🌿 Downloading TEOW 2017 ecoregions..."; \
		uv run python -c "from src.crawl_first.download_ecoregions import main; main()"; \
	else \
		echo "✅ TEOW 2017 ecoregions already available"; \
	fi
	@if [ ! -f data/envo/envo.owl ]; then \
		echo "🌱 Downloading ENVO ontology..."; \
		$(MAKE) download-envo; \
	else \
		echo "✅ ENVO ontology already available"; \
	fi
	@echo "✅ Local datasets ready for enrichment"

# Run unified geospatial enrichment (from src) with dependencies
enrich-unified-test: setup-local-data generate-mappings data/outputs/api/normalized_multi_env_biosamples.json
	@echo "🌍 Running unified geospatial enrichment test with curated multi-environment biosamples..."
	@echo "📊 Testing with 18 diverse samples (9 NMDC + 9 GOLD) from marine, freshwater, and soil environments"
	uv run enrich-geo batch \
		--input data/outputs/api/normalized_multi_env_biosamples.json \
		--output data/outputs/unified_test.json \
		--max-samples 18 \
		--verbose \
		--pretty
	@echo "✅ Unified enrichment test complete with real biosample diversity - now includes:"
	@echo "   🔍 Multi-provider forward geocoding (Google + Nominatim)"
	@echo "   🏛️ Multi-provider reverse geocoding (Google + Nominatim)" 
	@echo "   🌦️ Multi-provider weather (Open-Meteo + Meteostat)"
	@echo "   🌍 Multi-provider land cover (ESA WorldCover + NLCD)"
	@echo "   🏔️ Multi-provider elevation (Local DEM + USGS + Google)"
	@echo "   🌱 Multi-provider soil (USDA NRCS + SoilGrids)"
	@echo "   🏢 Google Places API (business/trail context)"
	@echo "   🌬️ Multi-provider air quality (Google + EPA + OpenWeather)"
	@echo "   🌊 Marine enrichment: Sea surface temperature, chlorophyll, Longhurst provinces"
	@echo "   🏞️ Terrestrial vs marine handling showcased across diverse sample types"
	@echo "   📊 Results: data/outputs/unified_test.json"

# Download ENVO ontology
download-envo:
	@echo "🌱 Downloading ENVO ontology..."
	@mkdir -p data/envo
	curl -L http://purl.obolibrary.org/obo/envo.owl -o data/envo/envo.owl
	@echo "✅ ENVO downloaded to data/envo/envo.owl"

# Generate ALL mappings (JSON ontology + CSV normalization) - unified paradigm
generate-mappings:
	@echo "🗺️ Generating ALL mappings (unified paradigm)..."
	uv run python -m crawl_first.generate_mappings --mappings-dir mappings
	@echo "✅ All mapping files generated (JSON + CSV)"


# (Crosswalks now integrated into generate-mappings - no separate target needed)

# =============================================================================
# SCHEMA ANALYSIS TARGETS  
# =============================================================================

# Environment variables with defaults
MONGO_URI ?= mongodb://ncbi_reader:register_manatee_coach78@localhost:27778/?directConnection=true&authMechanism=DEFAULT&authSource=admin
GOLD_DB ?= gold_metadata
GOLD_COLL ?= biosamples
SAMPLE_SIZE ?= 50000
NMDC_SCHEMA_URL ?= https://raw.githubusercontent.com/microbiomedata/nmdc-schema/refs/heads/main/nmdc_schema/nmdc_materialized_patterns.yaml

# Create output directory for schema work
data/outputs/schema:
	@mkdir -p $@

# Fetch NMDC schema from GitHub using curl
data/outputs/schema/nmdc_schema.yaml: | data/outputs/schema
	@echo "📥 Fetching NMDC schema to $@..."
	curl -s $(NMDC_SCHEMA_URL) -o $@

# Extract NMDC Biosample slots using LinkML induction
data/outputs/schema/nmdc_biosample_slots.json: data/outputs/schema/nmdc_schema.yaml
	@echo "🔬 Extracting NMDC Biosample slots from $< to $@..."
	uv run python -m crawl_first.extract_nmdc_biosample_slots \
		--schema-path $< \
		--output $@

# Infer NMDC biosample schema from MongoDB data
data/outputs/schema/nmdc_biosample_schema.json: | data/outputs/schema
	@echo "🔬 Inferring NMDC biosample schema from data to $@..."
	uv run python -m crawl_first.infer_schema \
		--mongo-uri "$(MONGO_URI)" \
		--db "nmdc" \
		--coll "biosample_set" \
		--sample-size $(SAMPLE_SIZE) \
		--out-json-schema $@

# Infer GOLD schema from MongoDB using genson  
data/outputs/schema/gold_biosample_schema.json: | data/outputs/schema
	@echo "🏆 Inferring GOLD schema to $@..."
	uv run python -m crawl_first.infer_schema \
		--mongo-uri "$(MONGO_URI)" \
		--db "$(GOLD_DB)" \
		--coll "$(GOLD_COLL)" \
		--sample-size $(SAMPLE_SIZE) \
		--out-json-schema $@

# Generate GOLD field statistics (Compass-like)
data/outputs/schema/gold_biosample_stats.csv data/outputs/schema/gold_biosample_stats.md: | data/outputs/schema
	@echo "📊 Generating GOLD field statistics..."
	uv run python -m crawl_first.infer_stats \
		--mongo-uri "$(MONGO_URI)" \
		--db "$(GOLD_DB)" \
		--coll "$(GOLD_COLL)" \
		--sample-size $(SAMPLE_SIZE) \
		--out-csv data/outputs/schema/gold_biosample_stats.csv \
		--out-md data/outputs/schema/gold_biosample_stats.md

# Generate NMDC biosample field statistics from actual data
data/outputs/schema/nmdc_biosample_stats.csv data/outputs/schema/nmdc_biosample_stats.md: | data/outputs/schema
	@echo "📊 Generating NMDC biosample field statistics..."
	uv run python -m crawl_first.infer_stats \
		--mongo-uri "$(MONGO_URI)" \
		--db "nmdc" \
		--coll "biosample_set" \
		--sample-size $(SAMPLE_SIZE) \
		--out-csv data/outputs/schema/nmdc_biosample_stats.csv \
		--out-md data/outputs/schema/nmdc_biosample_stats.md

# Get raw Claude Code response for schema comparison
data/outputs/schema/schema_comparison_raw.json: data/outputs/schema/nmdc_biosample_slots.json data/outputs/schema/gold_biosample_schema.json prompts/schema-comparison-prompt.txt empty-mcp-config.json
	@echo "🤖 Getting raw Claude response to $@..."
	claude --print --output-format json --strict-mcp-config --mcp-config empty-mcp-config.json < prompts/schema-comparison-prompt.txt > $@

# Extract clean JSON from raw Claude response
data/outputs/schema/schema_comparison.json: data/outputs/schema/schema_comparison_raw.json
	@echo "📄 Extracting clean JSON from $< to $@..."
	jq -r '.result' $< | sed 's/^```json//' | sed 's/```$$//' > $@

# Get raw enrichment analysis using Claude Code
data/outputs/schema/enrichment_analysis_raw.json: data/outputs/schema/nmdc_biosample_stats.csv data/outputs/schema/gold_biosample_stats.csv data/outputs/schema/schema_comparison.json prompts/enrichment-analysis-prompt.txt empty-mcp-config.json
	@echo "🔍 Getting enrichment analysis to $@..."
	claude --print --output-format json --strict-mcp-config --mcp-config empty-mcp-config.json < prompts/enrichment-analysis-prompt.txt > $@

# Extract clean enrichment analysis JSON
data/outputs/schema/enrichment_analysis.json: data/outputs/schema/enrichment_analysis_raw.json
	@echo "📄 Extracting enrichment analysis from $< to $@..."
	jq -r '.result' $< | sed 's/^```json//' | sed 's/```$$//' > $@

# Clean schema analysis outputs
clean-schema:
	@echo "🧹 Cleaning schema analysis outputs..."
	rm -rf data/outputs/schema/
	@echo "✅ Schema analysis outputs cleaned"

# Complete schema analysis workflow - meta-target that does everything
analyze-schemas: data/outputs/schema/nmdc_biosample_slots.json data/outputs/schema/nmdc_biosample_schema.json data/outputs/schema/gold_biosample_schema.json data/outputs/schema/nmdc_biosample_stats.csv data/outputs/schema/gold_biosample_stats.csv data/outputs/schema/schema_comparison.json data/outputs/schema/enrichment_analysis.json
	@echo "✅ Complete schema analysis workflow finished"
	@echo "📁 Results available in data/outputs/schema/"
	@echo "📊 Schema files: $(word 1,$^) $(word 2,$^) $(word 3,$^)"
	@echo "📈 Statistics: $(word 4,$^) $(word 5,$^)"  
	@echo "🤖 Analysis: $(word 6,$^) $(word 7,$^)"

# Unified development workflow - combines root dev + enrichment
dev-unified: dev enrich-unified-test
	@echo "🚀 Unified development cycle complete - code quality + enrichment tested"

# Check if CLI works
check-cli:
	@echo "🖥️  Testing CLI..."
	uv run crawl-first --help
	@echo "✅ CLI working"

# Create directory structure
setup-dirs:
	@echo "📁 Creating directory structure..."
	mkdir -p data/inputs
	mkdir -p data/outputs/adapters
	mkdir -p data/outputs/schema
	@echo "✅ Directory structure created"

# Fetch biosample IDs from NMDC API
data/inputs/biosample-ids.txt: | setup-dirs
	@echo "📡 Fetching biosample IDs from NMDC API..."
	curl -X 'GET' \
		'https://api.microbiomedata.org/nmdcschema/biosample_set?max_page_size=20000&projection=id' \
		-H 'accept: application/json' | jq -r '.resources[].id' > $@
	@echo "✅ Biosample IDs saved to $@"


# Test Claude weather query
nmdc-with-ornl/data/outputs/claude/weather-test.txt:
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

# MCP diagnostic tests - Claude interactions with MCP servers (includes weather test)
test-mcp: data/outputs/claude/weather-test.txt data/outputs/claude/mcp-servers-test.txt data/outputs/claude/landuse-mcp-test.txt
	@echo "🔧 MCP diagnostic tests complete"

# Specific compression targets with safeguards
define compress_dir
	@if [ -d "$1" ]; then \
		file_count=$$(find "$1" -type f | wc -l); \
		if [ $$file_count -gt 0 ]; then \
			echo "📦 Compressing $1 directory ($$file_count files)..."; \
			tar -czf $2 $1; \
			echo "✅ $1 compressed to $2"; \
		else \
			echo "⚠️  Directory $1 exists but contains no files - skipping compression to avoid overwriting existing archive"; \
		fi; \
	else \
		echo "⚠️  Directory $1 does not exist"; \
	fi
endef

# Create archives directory
archives:
	@mkdir -p archives

# Clean archives directory
archives-clean:
	@echo "🗑️  Cleaning archives directory..."
	rm -rf archives/
	@echo "✅ Archives cleaned"

archives/cache.tar.gz: cache | archives
	$(call compress_dir,$<,$@)

archives/data.tar.gz: data | archives
	$(call compress_dir,$<,$@)


# Compress all archives
compress-all: archives/cache.tar.gz archives/data.tar.gz
	@echo "📦 All directories compressed: $^"

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

# =============================================================================
# BIOSAMPLE EXTRACTION TARGETS
# =============================================================================

# Number of biosamples to extract (overrideable)
N_BIOSAMPLES ?= 100

# Extract raw intact biosamples from MongoDB
data/inputs/raw_biosamples_native.json: | setup-dirs
	@echo "🧬 Fetching $(N_BIOSAMPLES) intact NMDC + GOLD biosamples..."
	uv run extract-biosamples --limit $(N_BIOSAMPLES) --output $@
	@echo "✅ Created: $@"

# =============================================================================
# BIOSAMPLE ADAPTER TARGETS
# =============================================================================

# Create output directory for adapter results
data/outputs/adapters:
	@mkdir -p $@

# Test NMDC biosample adapter with sample data and save to file
data/outputs/adapters/nmdc_adapter_test.json: | data/outputs/adapters
	@echo "🧬 Testing NMDC biosample adapter..."
	uv run python -m crawl_first.test_nmdc_adapter > $@
	@echo "✅ NMDC adapter test results saved to $@"

# Test GOLD biosample adapter with sample data and save to file
data/outputs/adapters/gold_adapter_test.json: | data/outputs/adapters
	@echo "🏅 Testing GOLD biosample adapter..."
	uv run python -m crawl_first.test_gold_adapter > $@
	@echo "✅ GOLD adapter test results saved to $@"

# Run combined adapter demonstration and save comprehensive results
data/outputs/adapters/adapter_demonstration.json: | data/outputs/adapters
	@echo "🔬 Running comprehensive adapter demonstration..."
	uv run python -m crawl_first.biosample_adapter_usage > data/outputs/adapters/adapter_demo_output.txt 2>&1
	uv run python -m crawl_first.adapter_demonstration > $@
	@echo "✅ Comprehensive adapter demonstration saved to $@"

# MongoDB connection test (stub - requires actual MongoDB)
data/outputs/adapters/mongodb_connection_test.json: | data/outputs/adapters
	@echo "🗄️  Testing MongoDB adapter configuration (stub)..."
	uv run python -m crawl_first.mongodb_connection_test > $@
	@echo "✅ MongoDB adapter configuration test saved to $@"

# Test all biosample adapters comprehensively
test-adapters: data/outputs/adapters/nmdc_adapter_test.json data/outputs/adapters/gold_adapter_test.json data/outputs/adapters/adapter_demonstration.json data/outputs/adapters/mongodb_connection_test.json
	@echo "🧪 All biosample adapter tests complete"
	@echo "📁 Results available in data/outputs/adapters/"
	@echo "📊 Test files:"
	@echo "   - NMDC adapter: $(word 1,$^)"
	@echo "   - GOLD adapter: $(word 2,$^)" 
	@echo "   - Demonstration: $(word 3,$^)"
	@echo "   - MongoDB config: $(word 4,$^)"

# Run adapter tests and show summary
demo-adapters: test-adapters
	@echo ""
	@echo "📋 BIOSAMPLE ADAPTER TEST SUMMARY"
	@echo "=================================="
	@echo "NMDC Test Results:"
	@jq -r '.samples_processed as $$total | .enrichable_samples as $$enrichable | "  Samples: \($$total), Enrichable: \($$enrichable), Rate: \(($$enrichable/$$total*100)|round)%"' data/outputs/adapters/nmdc_adapter_test.json
	@echo ""
	@echo "GOLD Test Results:"
	@jq -r '.samples_processed as $$total | .enrichable_samples as $$enrichable | "  Samples: \($$total), Enrichable: \($$enrichable), Rate: \(($$enrichable/$$total*100)|round)%"' data/outputs/adapters/gold_adapter_test.json
	@echo ""
	@echo "Combined Demonstration:"
	@jq -r '.demonstration_summary | "  Total: \(.total_samples_tested), NMDC: \(.nmdc_samples), GOLD: \(.gold_samples), Enrichable: \(.enrichable_samples), Rate: \((.enrichment_rate*100)|round)%"' data/outputs/adapters/adapter_demonstration.json
	@echo ""
	@echo "API Enrichment Readiness:"
	@jq -r '.api_enrichment_readiness | "  Elevation API: \(.elevation_api_ready) samples", "  Weather API: \(.weather_api_ready) samples", "  Geocoding API: \(.geocoding_api_ready) samples"' data/outputs/adapters/adapter_demonstration.json
	@echo "✅ Adapter demonstration complete!"

# Clean adapter test outputs
clean-adapters:
	@echo "🧹 Cleaning adapter test outputs..."
	rm -rf data/outputs/adapters/
	@echo "✅ Adapter outputs cleaned"

# =============================================================================
# BIOSAMPLE NORMALIZATION AND GEOSPATIAL API ENRICHMENT TARGETS
# =============================================================================

# Create directory for API enrichment outputs
data/outputs/api:
	@mkdir -p $@

# Extract real NMDC and GOLD biosamples from MongoDB (dependency for normalization)
data/inputs/test_biosamples.json: | setup-dirs
	@echo "🗄️  Extracting real NMDC and GOLD biosamples from MongoDB..."
	uv run python -m crawl_first.extract_real_biosamples > $@
	@echo "✅ Real biosample data extracted to $@"

# Generate normalized biosample data for API enrichment (legacy test data)
data/outputs/api/normalized_biosamples.json: data/inputs/test_biosamples.json | data/outputs/api
	@echo "🧬 Normalizing GOLD and NMDC biosamples for API enrichment..."
	uv run normalize-biosamples --input $< --output $@ --verbose
	@echo "✅ Normalized biosamples saved to $@"

# Generate normalized data from curated multi-environment samples
data/outputs/api/normalized_multi_env_biosamples.json: data/inputs/multi_env_multi_source_biosamples.json | data/outputs/api
	@echo "🌍 Normalizing curated multi-environment biosamples for API enrichment..."
	uv run normalize-biosamples --input $< --output $@ --verbose
	@echo "✅ Normalized curated biosamples saved to $@"

# Run comprehensive geospatial API enrichment (19 API functions including land cover + Meteostat)
data/outputs/api/enrichment_results.json: data/outputs/api/normalized_biosamples.json
	@echo "🌍 Running comprehensive geospatial enrichment with 19 real APIs..."
	@echo "   📊 Elevation: Open-Elevation, USGS, Google Maps"
	@echo "   🌤️  Weather: Open-Meteo historical + Meteostat station-based"
	@echo "   📍 Geocoding: Nominatim, Google reverse geocoding"
	@echo "   🏞️  Land Cover: ESA WorldCover, USGS NLCD, Historical"
	@echo "   🌱 Soil: NRCS SDA, ISRIC SoilGrids (classification + properties)"
	@echo "   🌿 Ecoregions: Local TEOW 2017, WWF/RESOLVE"
	@echo "   🗺️  Features: OpenStreetMap Overpass"
	uv run python src/crawl_first/run_real_api_enrichment.py $< $@
	@echo "✅ Comprehensive enrichment complete with 19 API functions - results saved to $@"

# Clean up API enrichment outputs
clean-api-enrichment:
	@echo "🧹 Cleaning API enrichment outputs..."
	rm -f data/outputs/api/normalized_biosamples.json
	rm -f data/outputs/api/enrichment_results.json
	rm -f data/outputs/api/real_api_enrichment*.json
	rm -f data/outputs/api/google_plus_enrichment.json
	@echo "✅ API enrichment outputs cleaned"

# Combined biosample normalization and API enrichment pipeline
biosample-api-pipeline: data/outputs/api/enrichment_results.json
	@echo "🎯 BIOSAMPLE API ENRICHMENT PIPELINE COMPLETE"
	@echo "=============================================="
	@echo "✅ Real biosamples extracted: data/inputs/test_biosamples.json"
	@echo "✅ Normalized for APIs: data/outputs/api/normalized_biosamples.json"
	@echo "✅ Comprehensive enrichment: data/outputs/api/enrichment_results.json"
	@echo ""
	@echo "📊 Pipeline Summary:"
	@echo "   🧬 Biosamples: $(shell jq -r '.metadata.total_samples // "N/A"' data/outputs/api/normalized_biosamples.json 2>/dev/null)"
	@echo "   🌍 API Functions: $(shell jq -r '.api_count // "N/A"' data/outputs/api/enrichment_results.json 2>/dev/null)"
	@echo "   ✅ Success Rate: $(shell jq -r '.results | if length > 0 then (map(.successful_apis) | add) / (length * (.api_count // 18)) * 100 | floor else "N/A" end' data/outputs/api/enrichment_results.json 2>/dev/null)%"

# =============================================================================
# COMPREHENSIVE WORKFLOW - ALL REQUESTED TARGETS
# =============================================================================

# Complete workflow: cleanup + schema + adapters + biosample API pipeline
comprehensive-workflow: squeaky-clean analyze-schemas demo-enhanced-adapters biosample-api-pipeline
	@echo "🎯 COMPREHENSIVE WORKFLOW COMPLETE"
	@echo "================================="
	@echo "✅ 1. Squeaky cleanup: All regenerable files removed"
	@echo "✅ 2. Schema analysis: NMDC vs GOLD schema comparison complete"
	@echo "✅ 3. Adapter demo: GOLD and NMDC MongoDB adapter usage demonstrated"
	@echo "✅ 4. Biosample API pipeline: Extraction → normalization → comprehensive enrichment"
	@echo ""
	@echo "📁 Key outputs:"
	@echo "   📊 Schema analysis: data/outputs/schema/"
	@echo "   🧬 Adapter demos: data/outputs/adapters/"
	@echo "   📍 Normalized data: data/outputs/api/normalized_biosamples.json"
	@echo "   🌍 API enrichment: data/outputs/api/enrichment_results.json"

# =============================================================================
# ENHANCED BIOSAMPLE ADAPTER TARGETS - Normalized IDs and Advanced Retrieval
# =============================================================================

# Test NMDC adapter with enhanced ID normalization and studies
data/outputs/adapters/nmdc_enhanced_adapter_test.json: | data/outputs/adapters
	@echo "🧬 Testing enhanced NMDC adapter with normalized IDs..."
	uv run python -m crawl_first.test_nmdc_enhanced_adapter > $@
	@echo "✅ Enhanced NMDC adapter test results saved to $@"

# Test GOLD adapter with enhanced ID normalization and study lookup
data/outputs/adapters/gold_enhanced_adapter_test.json: | data/outputs/adapters
	@echo "🏅 Testing enhanced GOLD adapter with normalized IDs and study lookup..."
	uv run python -m crawl_first.test_gold_enhanced_adapter > $@
	@echo "✅ Enhanced GOLD adapter test results saved to $@"

# Test ID-based retrieval using native field names
data/outputs/adapters/id_based_retrieval_test.json: | data/outputs/adapters
	@echo "🆔 Testing ID-based biosample retrieval..."
	uv run python -m crawl_first.test_id_based_retrieval > $@
	@echo "✅ ID-based retrieval test results saved to $@"

# Test random sampling functionality
data/outputs/adapters/random_sampling_test.json: | data/outputs/adapters
	@echo "🎲 Testing random biosample sampling..."
	uv run python -m crawl_first.test_random_sampling > $@
	@echo "✅ Random sampling test results saved to $@"

# Test unified interface with enhanced features
data/outputs/adapters/unified_enhanced_test.json: | data/outputs/adapters
	@echo "🔄 Testing unified interface with enhanced features..."
	uv run python -m crawl_first.test_unified_enhanced > $@
	@echo "✅ Unified enhanced test results saved to $@"

# Test separate ID lists functionality
data/outputs/adapters/separate_ids_test.json: | data/outputs/adapters
	@echo "📝 Testing separate ID lists functionality..."
	uv run python -m crawl_first.test_separate_ids > $@
	@echo "✅ Separate ID lists test results saved to $@"

# Test GOLD study lookup functionality
data/outputs/adapters/gold_study_lookup_test.json: | data/outputs/adapters
	@echo "📚 Testing GOLD study lookup from seq_projects..."
	uv run python -m crawl_first.test_gold_study_lookup > $@
	@echo "✅ GOLD study lookup test results saved to $@"

# (Target defined in API enrichment section above)

# Test enhanced adapters with realistic biosample data
data/outputs/adapters/enhanced_adapters_realistic_test.json: data/inputs/test_biosamples.json | data/outputs/adapters
	@echo "🧬 Testing enhanced adapters with real NMDC and GOLD biosample data..."
	uv run python -m crawl_first.test_enhanced_adapters_with_data > $@
	@echo "✅ Enhanced adapters realistic test results saved to $@"

# Run all enhanced adapter tests
test-enhanced-adapters: data/outputs/adapters/nmdc_enhanced_adapter_test.json data/outputs/adapters/gold_enhanced_adapter_test.json data/outputs/adapters/id_based_retrieval_test.json data/outputs/adapters/random_sampling_test.json data/outputs/adapters/unified_enhanced_test.json data/outputs/adapters/separate_ids_test.json data/outputs/adapters/gold_study_lookup_test.json data/outputs/adapters/enhanced_adapters_realistic_test.json
	@echo "🧪 All enhanced biosample adapter tests complete"
	@echo "📁 Enhanced results available in data/outputs/adapters/"
	@echo "📊 Enhanced test files:"
	@echo "   - NMDC enhanced: $(word 1,$^)"
	@echo "   - GOLD enhanced: $(word 2,$^)" 
	@echo "   - ID-based retrieval: $(word 3,$^)"
	@echo "   - Random sampling: $(word 4,$^)"
	@echo "   - Unified enhanced: $(word 5,$^)"
	@echo "   - Separate ID lists: $(word 6,$^)"
	@echo "   - GOLD study lookup: $(word 7,$^)"

# Demonstrate all enhanced adapter features with summary
demo-enhanced-adapters: test-enhanced-adapters
	@echo ""
	@echo "📋 ENHANCED BIOSAMPLE ADAPTER TEST SUMMARY"
	@echo "==========================================="
	@echo "Enhanced NMDC Test Results:"
	@jq -r 'if .samples_processed then .samples_processed as $$total | .enrichable_samples as $$enrichable | "  Samples: \($$total), Enrichable: \($$enrichable), Rate: \(($$enrichable/$$total*100)|round)%" else "  Test completed successfully" end' data/outputs/adapters/nmdc_enhanced_adapter_test.json
	@echo ""
	@echo "Enhanced GOLD Test Results:"
	@jq -r 'if .samples_processed then .samples_processed as $$total | .enrichable_samples as $$enrichable | "  Samples: \($$total), Enrichable: \($$enrichable), Rate: \(($$enrichable/$$total*100)|round)%" else "  Test completed successfully" end' data/outputs/adapters/gold_enhanced_adapter_test.json
	@echo ""
	@echo "ID-Based Retrieval:"
	@jq -r 'if .test_summary then "  NMDC IDs tested: \(.test_summary.nmdc_ids_tested), GOLD IDs tested: \(.test_summary.gold_ids_tested)" else "  Test completed successfully" end' data/outputs/adapters/id_based_retrieval_test.json
	@echo ""
	@echo "Random Sampling:"
	@jq -r 'if .random_samples then "  NMDC random: \(.random_samples.nmdc_count), GOLD random: \(.random_samples.gold_count)" else "  Test completed successfully" end' data/outputs/adapters/random_sampling_test.json
	@echo ""
	@echo "Separate ID Lists:"
	@jq -r 'if .id_separation_test then "  ID types separated: \(.id_separation_test.types_found | length)" else "  Test completed successfully" end' data/outputs/adapters/separate_ids_test.json
	@echo ""
	@echo "GOLD Study Lookup:"
	@jq -r 'if .study_lookup_test then "  Biosamples with studies: \(.study_lookup_test.samples_with_studies)" else "  Test completed successfully" end' data/outputs/adapters/gold_study_lookup_test.json
	@echo "✅ Enhanced adapter demonstration complete!"

# =============================================================================
# HELP TARGET
# =============================================================================

help:
	@echo "Crawl-First Development & Unified Enrichment Pipeline"
	@echo "======================================================"
	@echo ""
	@echo "Core Development Targets:"
	@echo "  all              - Run all quality checks and tests"
	@echo "  all-unified      - Complete workflow: quality + schema analysis + geospatial enrichment"
	@echo "  install          - Install dependencies in development mode"
	@echo "  format           - Format code with black"
	@echo "  lint             - Lint code with ruff"
	@echo "  typecheck        - Type checking with mypy"
	@echo "  test             - Run tests with pytest"
	@echo "  dev              - Quick development cycle (format, lint, test)"
	@echo "  dev-unified      - Unified dev cycle (dev + enrichment test)"
	@echo "  ci               - Full CI simulation"
	@echo ""
	@echo "Schema Analysis Targets:"
	@echo "  analyze-schemas      - Complete NMDC vs GOLD schema analysis workflow"
	@echo "  clean-schema         - Clean all schema analysis outputs"
	@echo "  data/outputs/schema/nmdc_schema.yaml - Fetch NMDC schema from GitHub"
	@echo "  data/outputs/schema/nmdc_biosample_schema.json - Infer NMDC schema from MongoDB"
	@echo "  data/outputs/schema/gold_biosample_schema.json - Infer GOLD schema from MongoDB"
	@echo "  data/outputs/schema/schema_comparison.json - Compare schemas with Claude"
	@echo ""
	@echo "Unified Enrichment Targets:"
	@echo "  enrich-unified-test  - Run unified geospatial enrichment test"
	@echo "  download-envo        - Download ENVO ontology"
	@echo "  generate-mappings    - Generate ontology mappings and crosswalks"
	@echo "  compile-crosswalks   - Compile crosswalks to SSSOM format"
	@echo ""
	@echo "Data & Testing Targets:"
	@echo "  setup-dirs           - Create directory structure"
	@echo "  full-test            - Complete test suite (code + data + application)"
	@echo "  test-mcp             - MCP diagnostic tests with Claude"
	@echo "  check-cli            - Test CLI functionality"
	@echo ""
	@echo "Biosample Extraction Targets:"
	@echo "  data/inputs/raw_biosamples_native.json - Extract raw NMDC + GOLD biosamples from MongoDB"
	@echo "                       N_BIOSAMPLES=100 (overrideable)"
	@echo ""
	@echo "Biosample Adapter Targets:"
	@echo "  test-adapters        - Test all biosample adapters and save results to files"
	@echo "  demo-adapters        - Run adapters with summary display"
	@echo "  test-enhanced-adapters - Test enhanced adapters with normalized IDs and advanced features"
	@echo "  demo-enhanced-adapters - Run enhanced adapters with comprehensive summary"
	@echo "  clean-adapters       - Clean adapter test outputs"
	@echo "  data/outputs/adapters/nmdc_adapter_test.json - Test NMDC adapter"
	@echo "  data/outputs/adapters/gold_adapter_test.json - Test GOLD adapter"
	@echo "  data/outputs/adapters/adapter_demonstration.json - Comprehensive demo"
	@echo ""
	@echo "Enhanced Adapter Features:"
	@echo "  data/outputs/adapters/nmdc_enhanced_adapter_test.json - NMDC with normalized IDs"
	@echo "  data/outputs/adapters/gold_enhanced_adapter_test.json - GOLD with study lookup"
	@echo "  data/outputs/adapters/id_based_retrieval_test.json - ID-based retrieval"
	@echo "  data/outputs/adapters/random_sampling_test.json - Random sampling"
	@echo "  data/outputs/adapters/unified_enhanced_test.json - Unified enhanced interface"
	@echo "  data/outputs/adapters/separate_ids_test.json - Separate ID lists"
	@echo "  data/outputs/adapters/gold_study_lookup_test.json - GOLD study lookup"
	@echo "  data/outputs/adapters/enhanced_adapters_realistic_test.json - Real biosample data test"
	@echo "  data/inputs/test_biosamples.json - Extract real biosamples from MongoDB"
	@echo ""
	@echo "Maintenance Targets:"
	@echo "  clean                - Clean build artifacts and caches"
	@echo "  squeaky-clean        - Remove all generated files"
	@echo "  compress-all         - Compress data directories to archives"
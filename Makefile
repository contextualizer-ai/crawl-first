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

# Squeaky clean - removes all generated files including data and test results
squeaky-clean: clean
	@echo "🧽 Squeaky clean - removing all generated files..."
	rm -rf cache/
	rm -rf data/
	rm -rf logs/
	@echo "✨ Squeaky clean complete - all generated files removed"

# Quick development cycle - format, lint, and test
dev: format lint test
	@echo "🚀 Development cycle complete"

# Full CI simulation - everything that runs in GitHub Actions
ci: all test-coverage security
	@echo "🎯 CI simulation complete"

# =============================================================================
# GOLD-NMDC UNIFIED ENRICHMENT TARGETS
# =============================================================================

# Run unified geospatial enrichment (from src)
enrich-unified-test:
	@echo "🌍 Running unified geospatial enrichment test..."
	uv run python -m crawl_first.unified_enrichment \
		--lat 44.428 \
		--lon -110.5885 \
		--date 2021-08-20 \
		--output data/outputs/unified_test.json \
		--enable-crosswalks \
		--verbose \
		--pretty
	@echo "✅ Unified enrichment test complete - results in data/outputs/unified_test.json"

# Download ENVO ontology
download-envo:
	@echo "🌱 Downloading ENVO ontology..."
	@mkdir -p data/envo
	curl -L http://purl.obolibrary.org/obo/envo.owl -o data/envo/envo.owl
	@echo "✅ ENVO downloaded to data/envo/envo.owl"

# Generate mappings and crosswalks
generate-mappings:
	@echo "🗺️ Generating ontology mappings and crosswalks..."
	uv run python -m crawl_first.generate_mappings
	@echo "✅ Mappings generated"

# Compile crosswalks to SSSOM format
compile-crosswalks:
	@echo "📋 Compiling crosswalks to SSSOM format..."
	uv run python -m crawl_first.compile_crosswalks
	@echo "✅ SSSOM crosswalks compiled"

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
	mkdir -p data/samples
	mkdir -p data/outputs/crawl-first
	mkdir -p data/outputs/tests
	mkdir -p data/outputs/claude
	@echo "✅ Directory structure created"

# Fetch biosample IDs from NMDC API
data/inputs/biosample-ids.txt: | setup-dirs
	@echo "📡 Fetching biosample IDs from NMDC API..."
	curl -X 'GET' \
		'https://api.microbiomedata.org/nmdcschema/biosample_set?max_page_size=20000&projection=id' \
		-H 'accept: application/json' | jq -r '.resources[].id' > $@
	@echo "✅ Biosample IDs saved to $@"

# Create random sample of 10 biosample IDs
data/samples/biosample-ids-10.txt: data/inputs/biosample-ids.txt
	@echo "🎲 Creating random sample of 10 biosample IDs..."
	shuf data/inputs/biosample-ids.txt | head -n 10 > $@
	@echo "✅ Random sample saved to $@"

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
full-test: all setup-dirs data/inputs/biosample-ids.txt data/samples/biosample-ids-10.txt data/outputs/tests/random-ids-test.txt data/outputs/crawl-first/test-results/ check-cli
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

archives/logs.tar.gz: logs | archives
	$(call compress_dir,$<,$@)

# Compress all archives
compress-all: archives/cache.tar.gz archives/data.tar.gz archives/logs.tar.gz
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
	@echo "Maintenance Targets:"
	@echo "  clean                - Clean build artifacts and caches"
	@echo "  squeaky-clean        - Remove all generated files"
	@echo "  compress-all         - Compress data directories to archives"
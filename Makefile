# Makefile for crawl-first development and quality control
.PHONY: all install clean format lint typecheck deps-check test test-verbose security build dev ci setup-dirs squeaky-clean full-test test-mcp compress-all

# Default target - runs all quality checks and tests
all: install format lint typecheck deps-check test

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
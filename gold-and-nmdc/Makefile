# NMDC vs GOLD schema comparison and analysis
.PHONY: all clean help

# Environment variables with defaults
NMDC_SCHEMA_URL ?= https://raw.githubusercontent.com/microbiomedata/nmdc-schema/refs/heads/main/nmdc_schema/nmdc_materialized_patterns.yaml
GOLD_CSV_PATH ?= gold_metadata.flattened_biosamples.csv
MONGO_URI ?= mongodb://ncbi_reader:register_manatee_coach78@localhost:27778/?directConnection=true&authMechanism=DEFAULT&authSource=admin
GOLD_DB ?= gold_metadata
GOLD_COLL ?= biosamples
SAMPLE_SIZE ?= 50000
OUTPUT_DIR ?= outputs

# Main target
all: $(OUTPUT_DIR)/nmdc_biosample_slots.json $(OUTPUT_DIR)/nmdc_biosample_schema.json $(OUTPUT_DIR)/gold_biosample_schema.json $(OUTPUT_DIR)/gold_biosample_stats.csv $(OUTPUT_DIR)/nmdc_biosample_stats.csv $(OUTPUT_DIR)/schema_comparison.json $(OUTPUT_DIR)/enrichment_analysis.json

# Geospatial enrichment workflow
all-geospatial: $(OUTPUT_DIR)/test_location.json $(OUTPUT_DIR)/enriched_locations.json $(OUTPUT_DIR)/natural_features.json $(OUTPUT_DIR)/envo_mappings.json

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $@

# Fetch NMDC schema from GitHub
$(OUTPUT_DIR)/nmdc_schema.yaml: $(OUTPUT_DIR)
	@echo "📥 Fetching NMDC schema to $@..."
	curl -s $(NMDC_SCHEMA_URL) -o $@

# Extract NMDC Biosample slots using LinkML induction
$(OUTPUT_DIR)/nmdc_biosample_slots.json: $(OUTPUT_DIR)/nmdc_schema.yaml
	@echo "🔬 Extracting NMDC Biosample slots to $@..."
	uv run python scripts/extract_nmdc_biosample_slots.py \
		--schema-path $(OUTPUT_DIR)/nmdc_schema.yaml \
		--output $@

# Infer NMDC biosample schema from actual MongoDB data
$(OUTPUT_DIR)/nmdc_biosample_schema.json: $(OUTPUT_DIR)
	@echo "🔬 Inferring NMDC biosample schema from data to $@..."
	uv run python scripts/infer_schema.py \
		--mongo-uri "$(MONGO_URI)" \
		--db "nmdc" \
		--coll "biosample_set" \
		--sample-size $(SAMPLE_SIZE) \
		--out-json-schema $@

# Infer GOLD schema from MongoDB using genson
$(OUTPUT_DIR)/gold_biosample_schema.json: $(OUTPUT_DIR)
	@echo "🏆 Inferring GOLD schema to $@..."
	uv run python scripts/infer_schema.py \
		--mongo-uri "$(MONGO_URI)" \
		--db "$(GOLD_DB)" \
		--coll "$(GOLD_COLL)" \
		--sample-size $(SAMPLE_SIZE) \
		--out-json-schema $@

# Generate GOLD field statistics (Compass-like)
$(OUTPUT_DIR)/gold_biosample_stats.csv $(OUTPUT_DIR)/gold_biosample_stats.md: $(OUTPUT_DIR)
	@echo "📊 Generating GOLD field statistics..."
	uv run python scripts/infer_stats.py \
		--mongo-uri "$(MONGO_URI)" \
		--db "$(GOLD_DB)" \
		--coll "$(GOLD_COLL)" \
		--sample-size $(SAMPLE_SIZE) \
		--out-csv $(OUTPUT_DIR)/gold_biosample_stats.csv \
		--out-md $(OUTPUT_DIR)/gold_biosample_stats.md

# Generate NMDC biosample field statistics from actual data
$(OUTPUT_DIR)/nmdc_biosample_stats.csv $(OUTPUT_DIR)/nmdc_biosample_stats.md: $(OUTPUT_DIR)
	@echo "📊 Generating NMDC biosample field statistics..."
	uv run python scripts/infer_stats.py \
		--mongo-uri "$(MONGO_URI)" \
		--db "nmdc" \
		--coll "biosample_set" \
		--sample-size $(SAMPLE_SIZE) \
		--out-csv $(OUTPUT_DIR)/nmdc_biosample_stats.csv \
		--out-md $(OUTPUT_DIR)/nmdc_biosample_stats.md

# Extract GOLD field names from CSV (fallback method)
$(OUTPUT_DIR)/gold_fields_csv.json: $(OUTPUT_DIR) $(GOLD_CSV_PATH)
	@echo "🏆 Extracting GOLD field names from CSV to $@..."
	head -1 $(GOLD_CSV_PATH) | tr ',' '\n' | grep -v '^_id$$' | sort | jq -R . | jq -s . > $@

# Get raw Claude Code response for schema comparison
$(OUTPUT_DIR)/schema_comparison_raw.json: $(OUTPUT_DIR)/nmdc_biosample_slots.json $(OUTPUT_DIR)/gold_biosample_schema.json prompts/schema-comparison-prompt.txt empty-mcp-config.json
	@echo "🤖 Getting raw Claude response to $@..."
	claude --print --output-format json --strict-mcp-config --mcp-config empty-mcp-config.json < prompts/schema-comparison-prompt.txt > $@

# Extract clean JSON from raw Claude response
$(OUTPUT_DIR)/schema_comparison.json: $(OUTPUT_DIR)/schema_comparison_raw.json
	@echo "📄 Extracting clean JSON to $@..."
	jq -r '.result' $< | sed 's/^```json//' | sed 's/```$$//' > $@

# Get raw enrichment analysis using Claude Code
$(OUTPUT_DIR)/enrichment_analysis_raw.json: $(OUTPUT_DIR)/nmdc_biosample_stats.csv $(OUTPUT_DIR)/gold_biosample_stats.csv $(OUTPUT_DIR)/schema_comparison.json prompts/enrichment-analysis-prompt.txt empty-mcp-config.json
	@echo "🔍 Getting enrichment analysis to $@..."
	claude --print --output-format json --strict-mcp-config --mcp-config empty-mcp-config.json < prompts/enrichment-analysis-prompt.txt > $@

# Extract clean enrichment analysis JSON
$(OUTPUT_DIR)/enrichment_analysis.json: $(OUTPUT_DIR)/enrichment_analysis_raw.json
	@echo "📄 Extracting enrichment analysis to $@..."
	jq -r '.result' $< | sed 's/^```json//' | sed 's/```$$//' > $@

# GEOSPATIAL ENRICHMENT TARGETS

# Test single location enrichment
$(OUTPUT_DIR)/test_location.json: scripts/enrich_location.py scripts/geospatial_enrichment.py $(OUTPUT_DIR)
	@echo "🌍 Testing single location enrichment to $@..."
	uv run python scripts/enrich_location.py \
		--lat 44.428 --lon -110.5885 --date 2021-08-20 \
		--output $@ --verbose

# Batch enrichment of test locations
$(OUTPUT_DIR)/enriched_locations.json: inputs/test_locations.json scripts/batch_enrich_locations.py scripts/geospatial_enrichment.py $(OUTPUT_DIR)
	@echo "🌍 Batch enriching locations from $< to $@..."
	uv run python scripts/batch_enrich_locations.py \
		--input $< --output $@ --verbose

# Extract unique natural features from enriched results
$(OUTPUT_DIR)/natural_features.json: $(OUTPUT_DIR)/enriched_locations.json
	@echo "🏷️ Extracting natural features from $< to $@..."
	jq '[.[] | .nearby_features.features[]? | select(.natural != null) | .natural] | unique | sort' $< > $@

# Generate ENVO mappings for natural features using OLS MCP
$(OUTPUT_DIR)/envo_mappings_raw.json: $(OUTPUT_DIR)/natural_features.json prompts/envo-mapping-prompt.txt empty-mcp-config.json
	@echo "🔬 Generating ENVO mappings from $< to $@..."
	claude --print --output-format json --strict-mcp-config --mcp-config empty-mcp-config.json < prompts/envo-mapping-prompt.txt > $@

# Extract clean ENVO mappings JSON
$(OUTPUT_DIR)/envo_mappings.json: $(OUTPUT_DIR)/envo_mappings_raw.json
	@echo "📄 Extracting clean ENVO mappings to $@..."
	jq -r '.result' $< | sed 's/^```json//' | sed 's/```$$//' > $@

# Clean all generated files
clean:
	rm -rf $(OUTPUT_DIR)

# Clean only geospatial enrichment files
clean-geospatial:
	rm -f $(OUTPUT_DIR)/test_location.json
	rm -f $(OUTPUT_DIR)/enriched_locations.json
	rm -f $(OUTPUT_DIR)/natural_features.json
	rm -f $(OUTPUT_DIR)/envo_mappings_raw.json
	rm -f $(OUTPUT_DIR)/envo_mappings.json

# Clean geospatial API cache
clean-cache:
	rm -rf scripts/cache/

# Clean generated mapping files
clean-mappings:
	rm -f scripts/mappings/*.json

# Show help
help:
	@echo "NMDC vs GOLD Schema Analysis Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all                                    - Run complete analysis pipeline"
	@echo "  $(OUTPUT_DIR)/nmdc_biosample_slots.json - Extract NMDC Biosample slots"
	@echo "  $(OUTPUT_DIR)/gold_fields.json         - Extract GOLD field names" 
	@echo "  $(OUTPUT_DIR)/schema_comparison.json   - AI-powered schema comparison"
	@echo "  clean                                  - Remove generated files"
	@echo ""
	@echo "Environment variables:"
	@echo "  NMDC_SCHEMA_URL   - URL to NMDC schema file (default: GitHub main branch)"
	@echo "  GOLD_CSV_PATH     - Path to GOLD CSV file (default: gold_metadata.flattened_biosamples.csv)"
	@echo "  OUTPUT_DIR        - Output directory (default: schema_analysis)"
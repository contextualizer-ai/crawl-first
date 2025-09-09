# NMDC Biosample Enrichment Project Status Report
**Date: 2025-09-04**

## What We've Accomplished

**Core Enrichment Pipeline:**
- Built `crawl-first` - a deterministic biosample enrichment tool that systematically follows discoverable links from NMDC records
- Integrated underlying libraries from MCP server projects (nmdc-mcp, landuse-mcp, weather-mcp, ols-mcp, artl-mcp) for comprehensive data gathering - using direct Python imports rather than MCP protocol
- Created structured YAML output with `asserted` (original) and `inferred` (enriched) sections
- Implemented caching system to prevent redundant API calls

**AI-Powered Validation Agent:**
- Transformed from rule-based to **AI-powered semantic analysis** using PydanticAI
- Dynamic NMDC schema validation (870 slots) with format compliance checking
- Study-level context analysis comparing samples within the same study
- Literature integration using cached full-text papers
- CBORG LLM integration for intelligent conflict detection

**Data Quality Improvements:**
- Coordinate validation with distance calculations between asserted vs geocoded locations
- Elevation comparison and plausibility checking
- ENVO ontology term matching for environmental descriptors
- Interactive map URL generation for visual validation

## Current Challenges

**API Availability Crisis:**
- ORNL APIs for elevation, soil type, and land use are no longer available
- This breaks core enrichment functionality for key biosample slots
- Need alternative data sources or API endpoints

**Other Issues:**
- **Low population rates**: Only 131 of 577 slots (~22.7%) currently populated in NMDC
- **Shell environment issues**: Tool integration problems preventing `make all` execution
- **Scale testing**: Need validation on larger sample sets to demonstrate robustness

**Immediate Actions Needed:**
- Identify replacement APIs for ORNL services
- Update MCP integrations to use alternative sources
- Validate that replacement data sources provide equivalent quality/coverage

## Future Directions & Expansion Plans

**Beyond NMDC Biosamples:**
- Expand input types beyond NMDC biosamples to other sample/specimen metadata formats
- Explore GOLD Biosamples or other datasets from the Bertron project (https://github.com/ber-data)
- Develop generalized enrichment framework that can work with diverse input schemas

### GOLD Biosamples Infrastructure

**API Access:**
- Public GOLD API: https://gold-ws.jgi.doe.gov/
- Custom NMDC team API for enhanced access
- Pattern: fetch objects by ID, fetch objects of type X related to object Y of type Z
- Web interface with limited bulk downloads
- Excel downloads available at: https://gold.jgi.doe.gov/downloads
- Challenging to get all records of a given type in native shape by specifying type alone
- Existing crawling clients fetch objects by iterating through IDs from Excel file at GOLD website

**Data Storage Options:**
- Large JSON dumps from crawling clients
- MongoDB loading as objects arrive from API
- External metadata awareness repo for flattened formats: https://github.com/microbiomedata/external-metadata-awareness/

**MongoDB Access (NERSC SPIN):**
```bash
# Authenticate with sshproxy
sshproxy https://docs.nersc.gov/connect/mfa/#sshproxy

# SSH tunnel to MongoDB
ssh -i ~/.ssh/nersc -L27778:mongo-ncbi-loadbalancer.mam.production.svc.spin.nersc.org:27017 -o ServerAliveInterval=60 {YOUR_NERSC_USERNAME}@dtn01.nersc.gov

# Read-only MongoDB connection
mongodb://ncbi_reader:register_manatee_coach78@localhost:27778/?directConnection=true&authMechanism=DEFAULT&authSource=admin
```

**Available Collections:**
- `gold_metadata.biosamples`: Native shape from API
- `gold_metadata.flattened_biosamples`: Minimally lossy shape amenable to CSV export (created with external_metadata_awareness repo)

**Beyond NMDC Schema:**
- Predict and generate metadata fields outside the NMDC Biosample schema (e.g., proximity to critical minerals)
- Explore novel derived fields that could enhance scientific analysis
- Develop AI-driven metadata prediction for fields not traditionally captured

## AI Demonstration Opportunities

**For "Contextualize AI" Funding:**

1. **Enhanced Validation Agent**: Run side-by-side comparisons showing AI dramatically improves data quality assessment over rule-based approaches

2. **Semantic Conflict Detection**: Demonstrate AI catching subtle inconsistencies humans would miss (e.g., "Lake Mendota sample with geo_loc_name 'USA: Wisconsin' should be 'USA: Wisconsin, Madison, Lake Mendota'")

3. **Literature-Informed Validation**: Show AI cross-referencing sample metadata with published methodology from full-text papers

4. **Study Pattern Recognition**: Demonstrate AI identifying design patterns and outliers across multi-sample studies

5. **Scalability Demo**: Process thousands of samples to show AI-powered enrichment can work at scale

The key AI value proposition is moving from brittle rule-based systems to intelligent, context-aware analysis that leverages the full scope of available scientific knowledge. This makes demonstrating the AI capabilities even more important since you'll need to show value despite infrastructure challenges.

## NMDC Biosample Slot Population Patterns

### Most Populated Slots (Well-Represented)
- **`env_*` triad**: 100% populated (environmental descriptors)
- **`collection_date`**: ~96% populated
- **`elev` (elevation)**: ~83% populated
- **`lat_lon` (coordinates)**: ~61% populated

### Moderately Populated Slots
- **`temp` (temperature)**: ~39.2% 
- **`soil_horizon`**: ~34.6%
- **`ph`**: ~34.3%
- **`water_content`**: ~32.7%
- **`habitat`**: ~23.6%

### Least Populated Slots (Under-represented)
- **`diss_oxygen` (dissolved oxygen)**: ~4.39%
- **`conduc` (electrical conductivity)**: ~2.46%
- **`nitrate`**: ~2.21%
- **`phosphate`**: ~2.08%
- **`soil_type`**: ~0.23%
- **`soil_texture_meth`**: ~0.23%
- **`salinity`**: ~0.14%
- **`sulfate`**: ~0.05%

### Never Populated Slots
- **`basin`**: 0% (watershed information)
- **`previous_land_use`**: 0% (historical land cover)

### Overall Statistics
- **Total slots in NMDC Biosample**: 577
- **Currently represented**: 131 (~22.7%)
- Many **built-environment slots** (door/window/room fields) are excluded as not applicable to environmental samples
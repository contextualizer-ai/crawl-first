# Validation Agent Improvements Summary

## Overview

This document summarizes the enhancements made to the biosample validation agent to improve data quality assessment using AI-powered analysis with PydanticAI and CBORG LLM endpoints.

## What We've Been Working On

### Initial Problem
The original validation agent had several limitations:
- **Hardcoded semantic rules** instead of leveraging AI for intelligent analysis
- **Empty validation fields** - most scores were `null` with minimal differentiation
- **No schema compliance checking** - couldn't validate field formats against NMDC standards
- **Isolated sample analysis** - no context from other samples in the same study
- **No literature integration** - missing research context for validation decisions

### Solution Approach
Transformed the validation agent from rule-based to **AI-powered semantic analysis** while maintaining mathematical precision for quantitative checks.

## Key Improvements Delivered

### 1. **NMDC Schema Validation** ✅
- **Dynamic schema loading**: Fetches 870 slot definitions from NMDC GitHub repo
- **Format compliance checking**: Validates field formats against official NMDC standards
- **Specific issue detection**: Catches problems like geo_loc_name format violations
- **Example improvement**: Now detects that "USA: Wisconsin" should be "USA: Wisconsin, Madison, Lake Mendota" per INSDC standard

### 2. **Study-Level Context Analysis** ✅  
- **Cross-sample comparison**: Analyzes samples within the same study
- **Design pattern recognition**: Identifies constant vs variable attributes across study design
- **Outlier detection**: Flags samples that don't match expected study patterns
- **Context-aware validation**: Uses study metadata to inform quality assessments

### 3. **Literature Integration** ✅
- **Cached literature usage**: Leverages full text files from `archives/cache/full_text_files/`
- **Research context**: Cross-references sample metadata with published methodology
- **DOI-based analysis**: Uses paper content to validate sampling approaches
- **Methodology alignment**: Checks if samples match published study design

### 4. **Enhanced Semantic Understanding** ✅
- **LLM-powered conflict detection**: Replaces hardcoded rules with intelligent analysis
- **Structured output**: Uses PydanticAI for consistent ValidationResult objects
- **CBORG integration**: Supports tool-calling capable models (Claude Sonnet)
- **Meaningful differentiation**: Actually distinguishes well-enriched vs problematic samples

## Technical Architecture

### Before (Rule-Based)
```python
# Hardcoded semantic rules
forest_indicators = ["woodland", "forest", "tree"]
urban_indicators = ["urban", "built-up", "impervious"]
if asserted_forest and inferred_urban:
    score -= 0.6  # Major conflict
```

### After (AI-Powered)
```python
# Dynamic schema + LLM analysis
schema_context = extract_relevant_slot_definitions(biosample_data, nmdc_schema)
study_context = get_study_context(biosample_data, results_dir)
literature_context = get_literature_context(biosample_data)

# LLM analyzes with full context
validation_result = await run_llm_validation(agent, biosample_data, contexts...)
```

## How to Observe These Improvements

### 1. **NMDC Schema Validation** - Compare Before/After

**OLD validation** (without schema):
```bash
# Run old version - would miss format issues
OPENAI_API_KEY="your-key" uv run python src/crawl_first/validation_agent.py --max-samples 1 --model "gpt-oss-120b"
```

**NEW validation** (with dynamic schema):
```bash
# You'll see this output difference:
🔄 Fetching NMDC schema...
✅ NMDC schema loaded (870 slots)

# And in the results, you'll now see:
"Schema non-compliance: geo_loc_name format doesn't follow INSDC standard"
```

**To verify**: Look for the Lake Mendota sample (`nmdc:bsm-11-ybjmrs89`) - the new version should specifically flag that "USA: Wisconsin" should be "USA: Wisconsin, Madison, Lake Mendota" per NMDC schema.

### 2. **Study-Level Context** - Check Multi-Sample Output

Run validation on multiple samples from the same study:
```bash
OPENAI_API_KEY="your-key" uv run python src/crawl_first/validation_agent.py --max-samples 5 --output study-context-test.json
```

**Look for study context sections** in the LLM prompts by adding debug output:
- You should see "Study Context (nmdc:sty-11-xxxx)" sections
- "Total samples in study: X" 
- Lists of other samples for comparison

### 3. **Literature Integration** - Verify Full Text Usage

**Check what literature is available**:
```bash
ls archives/cache/full_text_files/
```

**Run validation on samples with DOIs**:
```bash
OPENAI_API_KEY="your-key" uv run python src/crawl_first/validation_agent.py --max-samples 3 --output literature-test.json
```

**Look for**: "Literature Context" sections in the validation output mentioning specific DOIs and paper content.

### 4. **Enhanced Understanding** - Side-by-Side Comparison

**Run both versions and compare**:

```bash
# Enhanced version (current)
OPENAI_API_KEY="your-key" make data/outputs/validation-enhanced-$(date +%Y%m%d).json

# Compare with your existing results
diff data/outputs/validation-results-custom.json data/outputs/validation-enhanced-*.json
```

### 5. **Direct Observation Commands**

**See schema loading in real-time**:
```bash
OPENAI_API_KEY="your-key" uv run python src/crawl_first/validation_agent.py --max-samples 1 --model "anthropic/claude-sonnet" | head -10
```

**Check specific improvements for Lake Mendota**:
```bash
# Find the Lake Mendota sample and run validation on just that one
grep -r "Lake Mendota" archives/data/outputs/crawl-first/test-results/
OPENAI_API_KEY="your-key" uv run python src/crawl_first/validation_agent.py --results-dir archives/data/outputs/crawl-first/test-results/ --max-samples 1000 | grep -A 5 -B 5 "mendota\|Wisconsin"
```

**Verify geo_loc_name format detection**:
```bash
# Look for the specific schema violation message
OPENAI_API_KEY="your-key" uv run python src/crawl_first/validation_agent.py --max-samples 10 --output detailed-test.json
grep -i "geo_loc_name\|schema\|format" detailed-test.json
```

## Example Validation Results

### Before Enhancement
```json
{
  "biosample_id": "nmdc:bsm-11-ybjmrs89",
  "overall_score": 0.8,
  "coordinate_consistency": null,
  "elevation_plausibility": null,
  "env_triad_coherence": null,
  "issues": ["AI analysis: AgentRunResult(output='## 1..."],
  "recommendations": ["Review AI analysis output for detailed insights"]
}
```

### After Enhancement
```json
{
  "biosample_id": "nmdc:bsm-11-ybjmrs89",
  "overall_score": 0.75,
  "coordinate_consistency": 0.3,
  "elevation_plausibility": 1.0,
  "env_triad_coherence": 1.0,
  "issues": [
    "Major geographic coordinate discrepancy: 149.79 km distance between asserted coordinates and geocoded location",
    "geo_loc_name is too broad ('USA: Wisconsin') for a specific lake sample - should include lake name and city",
    "Schema non-compliance: geo_loc_name format doesn't follow INSDC standard"
  ],
  "recommendations": [
    "Update geo_loc_name to be more specific: 'USA: Wisconsin, Madison, Lake Mendota'",
    "Verify and correct coordinates to match Lake Mendota's actual location"
  ]
}
```

## Usage

### Standard Validation
```bash
OPENAI_API_KEY="your-key" make data/outputs/validation-results.json
```

### Custom Parameters
```bash
OPENAI_API_KEY="your-key" make data/outputs/validation-custom.json MODEL="anthropic/claude-sonnet" MAX_SAMPLES=10
```

### Direct Command
```bash
OPENAI_API_KEY="your-key" uv run python src/crawl_first/validation_agent.py \
  --max-samples 5 \
  --model "anthropic/claude-sonnet" \
  --output validation-test.json
```

## Impact for "Contextualize AI" Funding

This validation agent clearly demonstrates AI usage for the funding requirements:

1. **LLM-powered semantic analysis** - Uses large language models for intelligent conflict detection
2. **Dynamic context integration** - Combines schema, study, and literature contexts
3. **Structured AI outputs** - PydanticAI ensures reliable, validated results
4. **Research enhancement** - Improves data quality assessment beyond human capability

The agent now provides meaningful differentiation between well-enriched samples and problematic ones, using AI to understand complex semantic relationships that would be impossible to capture with simple rules.
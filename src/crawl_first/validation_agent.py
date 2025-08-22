"""
PydanticAI validation agent for biosample enrichment data quality assessment.

This agent validates consistency between asserted and inferred data fields,
identifies potential conflicts, and provides confidence scoring for enrichments.
"""

import asyncio
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
from datetime import datetime, timedelta
import click
from collections import Counter
import json

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel


class ValidationResult(BaseModel):
    """Result of validation analysis for a biosample."""
    
    biosample_id: str
    overall_score: float = Field(ge=0, le=1, description="Overall quality score")
    
    # Coordinate validation
    coordinate_consistency: Optional[float] = Field(
        None, description="Consistency between asserted coords and location text"
    )
    elevation_plausibility: Optional[float] = Field(
        None, description="Plausibility of elevation for given coordinates"
    )
    
    # Environmental triad validation  
    env_triad_coherence: Optional[float] = Field(
        None, description="Coherence of env_broad_scale -> env_local_scale -> env_medium"
    )
    land_cover_soil_consistency: Optional[float] = Field(
        None, description="Consistency between land cover and soil type"
    )
    ecosystem_alignment: Optional[float] = Field(
        None, description="Alignment between ecosystem fields and inferred data"
    )
    
    # Temporal validation
    date_season_consistency: Optional[float] = Field(
        None, description="Consistency of collection date with seasonal patterns"
    )
    
    # Data completeness
    enrichment_coverage: float = Field(
        description="Percentage of expected enrichment fields populated"
    )
    
    # Issues and recommendations
    issues: List[str] = Field(
        default_factory=list, description="List of validation issues found"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations for data improvement"
    )
    
    # Metadata
    validation_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class ValidationContext(RunContext):
    """Context for validation agent with geographic and taxonomic knowledge."""
    
    def __init__(self):
        # Known problematic coordinate patterns
        self.coord_anomalies = {
            "zero_zero": (0.0, 0.0),
            "null_island_vicinity": [(lat, lon) for lat in range(-1, 2) for lon in range(-1, 2)],
            "coordinate_swapped": []  # Will be populated based on geo_loc_name mismatches
        }
        
        # Elevation bounds by region (rough estimates)
        self.elevation_bounds = {
            "ocean": (-11000, 10),
            "coastal": (-100, 500), 
            "plains": (-500, 1000),
            "hills": (0, 2000),
            "mountains": (500, 9000),
            "default": (-500, 6000)
        }
        
        # Expected soil types by land cover
        self.soil_landcover_compatibility = {
            "Croplands": ["Cambisols", "Phaeozems", "Luvisols", "Fluvisols", "Vertisols"],
            "Grasslands": ["Phaeozems", "Chernozems", "Kastanozems", "Cambisols"],
            "Forest": ["Podzols", "Acrisols", "Ferralsols", "Cambisols", "Luvisols"],
            "Urban": ["Anthrosols", "Technosols"],
            "Water": ["Fluvisols", "Gleysols", "Histosols"],
            "Barren": ["Arenosols", "Leptosols", "Calcisols"]
        }


def create_validation_agent(model_name: str, base_url: Optional[str] = None, api_key: Optional[str] = None) -> Agent:
    """Create a validation agent with configurable model and endpoint."""
    
    if base_url and api_key:
        # Set environment variables for OpenAI client to use custom endpoint
        os.environ['OPENAI_API_KEY'] = api_key
        os.environ['OPENAI_BASE_URL'] = base_url
        
        # Use OpenAI model with custom endpoint
        model = f"openai:{model_name}"
    else:
        # Use default model string for standard providers
        model = model_name
    
    return Agent(
        model,
        output_type=ValidationResult,
        system_prompt="""You are an expert biosample data validation specialist for environmental microbiome metadata.

Your task is to analyze biosample data quality by comparing ASSERTED (original) vs INFERRED (enriched) data fields.

Focus on:
1. **Semantic conflicts**: Do asserted environment types conflict with inferred land cover data?
2. **Geographic consistency**: Do coordinates match location names and environmental context?
3. **Environmental coherence**: Do env_broad_scale → env_local_scale → env_medium make logical sense?
4. **Ecosystem alignment**: Do asserted ecosystem fields match inferred environmental data?
5. **Data quality issues**: Missing fields, contradictions, or low-confidence enrichments

Provide specific scores (0.0-1.0) and concrete issues with actionable recommendations.
Lower scores indicate more problems. Higher scores indicate good data quality."""
    )


# Global variables - will be set in main()
validation_agent = None
nmdc_schema = None


async def fetch_nmdc_schema() -> Dict[str, Any]:
    """Fetch and parse the NMDC schema from GitHub."""
    import aiohttp
    
    schema_url = "https://raw.githubusercontent.com/microbiomedata/nmdc-schema/refs/heads/main/nmdc_schema/nmdc_materialized_patterns.yaml"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(schema_url) as response:
                if response.status == 200:
                    schema_text = await response.text()
                    return yaml.safe_load(schema_text)
                else:
                    print(f"⚠️  Failed to fetch NMDC schema: HTTP {response.status}")
                    return {}
    except Exception as e:
        print(f"⚠️  Error fetching NMDC schema: {str(e)}")
        return {}


def extract_relevant_slot_definitions(biosample_data: Dict[str, Any], schema: Dict[str, Any]) -> str:
    """Extract slot definitions for fields present in the biosample data."""
    if not schema or 'slots' not in schema:
        return ""
    
    # Get all field names from asserted data
    asserted = biosample_data.get('asserted', {})
    field_names = set(asserted.keys())
    
    # Also add common validation fields
    validation_fields = {'geo_loc_name', 'env_broad_scale', 'env_local_scale', 'env_medium', 
                        'ecosystem', 'ecosystem_category', 'ecosystem_type', 'lat_lon', 'elev'}
    field_names.update(validation_fields)
    
    relevant_slots = []
    slots = schema.get('slots', {})
    
    for field_name in field_names:
        if field_name in slots:
            slot_def = slots[field_name]
            slot_info = f"**{field_name}**:\n"
            
            # Add description
            if 'description' in slot_def:
                slot_info += f"  Description: {slot_def['description']}\n"
            
            # Add expected value pattern
            if 'annotations' in slot_def and 'expected_value' in slot_def['annotations']:
                expected = slot_def['annotations']['expected_value'].get('value', '')
                slot_info += f"  Expected format: {expected}\n"
            
            # Add pattern if available
            if 'pattern' in slot_def:
                slot_info += f"  Pattern: {slot_def['pattern']}\n"
            
            # Add range/type info
            if 'range' in slot_def:
                slot_info += f"  Type: {slot_def['range']}\n"
                
            relevant_slots.append(slot_info)
    
    if relevant_slots:
        return "\n## NMDC Schema Definitions:\n" + "\n".join(relevant_slots) + "\n"
    else:
        return ""


def get_study_context(biosample_data: Dict[str, Any], results_dir: Path) -> str:
    """Get context from other samples in the same study."""
    study_info = biosample_data.get('inferred', {}).get('study_info', {})
    study_id = study_info.get('study_id')
    
    if not study_id:
        return ""
    
    # Find other samples from the same study
    study_samples = []
    for yaml_file in results_dir.glob("*.yaml"):
        if yaml_file.name.startswith("nmdc_"):
            try:
                with open(yaml_file, 'r') as f:
                    other_sample = yaml.safe_load(f)
                    other_study_id = other_sample.get('inferred', {}).get('study_info', {}).get('study_id')
                    if other_study_id == study_id:
                        # Extract key attributes for comparison
                        asserted = other_sample.get('asserted', {})
                        sample_summary = {
                            'id': asserted.get('id'),
                            'ecosystem': asserted.get('ecosystem'),
                            'ecosystem_category': asserted.get('ecosystem_category'), 
                            'env_broad_scale': asserted.get('env_broad_scale', {}).get('term', {}).get('name'),
                            'geo_loc_name': asserted.get('geo_loc_name', {}).get('has_raw_value')
                        }
                        study_samples.append(sample_summary)
            except Exception:
                continue
    
    if len(study_samples) > 1:
        study_context = f"\n## Study Context ({study_id}):\n"
        study_context += f"Study: {study_info.get('study_name', 'Unknown')}\n"
        study_context += f"Total samples in study: {len(study_samples)}\n"
        study_context += "Other samples for comparison:\n"
        for sample in study_samples[:5]:  # Limit to 5 samples
            study_context += f"  - {sample}\n"
        return study_context
    
    return ""


def get_literature_context(biosample_data: Dict[str, Any]) -> str:
    """Get context from cached literature files."""
    dois = biosample_data.get('inferred', {}).get('all_dois', [])
    if not dois:
        return ""
    
    # Check for cached full text files
    cache_dir = Path("archives/cache/full_text_files")
    if not cache_dir.exists():
        return ""
    
    literature_context = ""
    for doi_entry in dois[:2]:  # Limit to first 2 DOIs
        if isinstance(doi_entry, dict):
            doi_value = doi_entry.get('doi_value', '').replace('doi:', '')
            # Look for cached files matching this DOI
            doi_clean = doi_value.replace('/', '_').replace('.', '_')
            
            for cached_file in cache_dir.glob(f"*{doi_clean}*"):
                try:
                    if cached_file.suffix == '.txt':
                        with open(cached_file, 'r') as f:
                            content = f.read()[:1000]  # First 1000 chars
                        literature_context += f"\n## Literature Context ({doi_value}):\n{content}...\n"
                        break
                except Exception:
                    continue
    
    return literature_context

async def analyze_coordinate_consistency(
    ctx: ValidationContext,
    asserted_lat: float,
    asserted_lon: float, 
    geo_loc_name: Optional[str] = None,
    elevation: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyze consistency of coordinates with location name and elevation.
    
    Returns consistency scores and identified issues.
    """
    issues = []
    scores = {}
    
    # Check for obvious coordinate problems
    if (asserted_lat, asserted_lon) == ctx.coord_anomalies["zero_zero"]:
        issues.append("Coordinates are exactly (0,0) - likely default/missing values")
        scores["coordinate_validity"] = 0.0
    elif abs(asserted_lat) > 90 or abs(asserted_lon) > 180:
        issues.append("Coordinates outside valid geographic bounds")
        scores["coordinate_validity"] = 0.0
    else:
        scores["coordinate_validity"] = 1.0
    
    # Basic geographic consistency with location name
    if geo_loc_name:
        geo_consistency_score = 0.8  # Default assume reasonable
        
        # Simple heuristics for major geographic mismatches
        if "USA" in geo_loc_name and not (-180 <= asserted_lon <= -60 and 20 <= asserted_lat <= 72):
            issues.append(f"USA location claimed but coordinates {asserted_lat}, {asserted_lon} outside US bounds")
            geo_consistency_score = 0.2
        elif "Europe" in geo_loc_name and not (-15 <= asserted_lon <= 50 and 35 <= asserted_lat <= 75):
            issues.append(f"European location claimed but coordinates outside typical European bounds") 
            geo_consistency_score = 0.3
        elif "Australia" in geo_loc_name and not (110 <= asserted_lon <= 160 and -45 <= asserted_lat <= -10):
            issues.append(f"Australian location claimed but coordinates outside Australian bounds")
            geo_consistency_score = 0.2
            
        scores["geo_name_consistency"] = geo_consistency_score
    
    # Elevation plausibility
    if elevation is not None:
        if elevation < -500 and asserted_lat > 0:  # Northern hemisphere deep depression
            issues.append(f"Unusually low elevation {elevation}m for coordinates - check for data entry errors")
            scores["elevation_plausibility"] = 0.4
        elif elevation > 6000:  # Very high elevation
            issues.append(f"Very high elevation {elevation}m - verify if mountainous region")
            scores["elevation_plausibility"] = 0.6
        else:
            scores["elevation_plausibility"] = 0.9
    
    return {
        "scores": scores,
        "issues": issues,
        "overall_coordinate_score": sum(scores.values()) / len(scores) if scores else 0.5
    }


# Standalone tool function - will be called directly
async def validate_environmental_triad(
    ctx: ValidationContext,
    env_broad_scale: Optional[str] = None,
    env_local_scale: Optional[str] = None, 
    env_medium: Optional[str] = None,
    ecosystem: Optional[str] = None,
    ecosystem_category: Optional[str] = None,
    ecosystem_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate coherence of environmental context triads and ecosystem classifications.
    """
    issues = []
    scores = {}
    
    # Check for hierarchical consistency in env triad
    triad_coherence_score = 0.8  # Default assumption of reasonable coherence
    
    # Specific inconsistency patterns
    if (env_broad_scale and "terrestrial biome" in env_broad_scale.lower() and
        env_medium and any(term in env_medium.lower() for term in ["water", "marine", "aquatic"])):
        issues.append("Terrestrial broad scale conflicts with aquatic medium")
        triad_coherence_score = 0.2
        
    if (env_local_scale and "agricultural" in env_local_scale.lower() and
        env_medium and any(term in env_medium.lower() for term in ["forest", "woodland"])):
        issues.append("Agricultural local scale conflicts with forest/woodland medium")
        triad_coherence_score = 0.3
        
    scores["env_triad_coherence"] = triad_coherence_score
    
    # Ecosystem classification consistency
    ecosystem_score = 0.8
    if ecosystem and ecosystem_category:
        # Check some basic ecosystem consistency patterns
        if ecosystem == "Environmental" and ecosystem_category == "Terrestrial":
            ecosystem_score = 0.9  # Good consistency
        elif ecosystem == "Host-associated" and ecosystem_category == "Plants":
            ecosystem_score = 0.9  # Good consistency  
        elif ecosystem == "Environmental" and ecosystem_category not in ["Terrestrial", "Aquatic", "Atmospheric"]:
            issues.append(f"Environmental ecosystem with unexpected category: {ecosystem_category}")
            ecosystem_score = 0.4
            
    scores["ecosystem_consistency"] = ecosystem_score
    
    return {
        "scores": scores,
        "issues": issues, 
        "overall_triad_score": sum(scores.values()) / len(scores) if scores else 0.7
    }


# Standalone tool function - will be called directly
async def assess_soil_landcover_compatibility(
    ctx: ValidationContext,
    soil_type: Optional[str] = None,
    land_cover_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Assess compatibility between inferred soil type and land cover classifications.
    """
    issues = []
    compatibility_scores = []
    
    if not soil_type or not land_cover_data:
        return {"scores": {}, "issues": ["Missing soil type or land cover data"], "overall_score": 0.5}
    
    # Extract dominant land cover types from multiple classification systems
    dominant_covers = []
    if "land_cover" in land_cover_data:
        for system, entries in land_cover_data["land_cover"].items():
            if isinstance(entries, list) and entries:
                cover_type = entries[0].get("system_term", "")
                if cover_type:
                    dominant_covers.append(cover_type)
    
    # Check compatibility with each land cover classification
    for cover_type in dominant_covers:
        compatible_soils = ctx.soil_landcover_compatibility.get(cover_type, [])
        if soil_type in compatible_soils:
            compatibility_scores.append(0.9)
        elif any(soil in soil_type for soil in compatible_soils):  # Partial match
            compatibility_scores.append(0.7)
        else:
            issues.append(f"Soil type '{soil_type}' unusual for land cover '{cover_type}'")
            compatibility_scores.append(0.4)
    
    overall_score = sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.5
    
    return {
        "scores": {"soil_landcover_compatibility": overall_score},
        "issues": issues,
        "overall_score": overall_score
    }


# Standalone tool function - will be called directly
async def calculate_enrichment_coverage(
    inferred_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate coverage of expected enrichment fields and identify gaps.
    """
    # Expected enrichment categories from your existing data structure
    expected_sections = [
        "coordinate_sources",
        "soil_from_asserted_coords", 
        "land_cover_from_asserted_coords",
        "weather_from_asserted_coords",
        "geospatial_from_asserted_coords", 
        "study_info",
        "all_dois",
        "publication_analysis"
    ]
    
    present_sections = [section for section in expected_sections if section in inferred_data]
    coverage_score = len(present_sections) / len(expected_sections)
    
    missing_sections = [section for section in expected_sections if section not in inferred_data]
    
    # More detailed analysis of content within sections
    detailed_coverage = {}
    if "soil_from_asserted_coords" in inferred_data:
        soil_data = inferred_data["soil_from_asserted_coords"]
        detailed_coverage["soil_enrichment"] = 1.0 if "soil_type" in soil_data else 0.5
        
    if "weather_from_asserted_coords" in inferred_data:
        weather_data = inferred_data["weather_from_asserted_coords"]
        weather_fields = ["temperature", "humidity", "precipitation"] 
        present_weather = sum(1 for field in weather_fields if field in weather_data)
        detailed_coverage["weather_enrichment"] = present_weather / len(weather_fields)
    
    return {
        "coverage_score": coverage_score,
        "missing_sections": missing_sections,
        "detailed_coverage": detailed_coverage,
        "recommendations": [
            f"Consider enriching missing section: {section}" for section in missing_sections[:3]
        ]
    }


async def validate_biosample(biosample_data: Dict[str, Any]) -> ValidationResult:
    """
    AI-powered biosample validation using LLM for semantic analysis.
    """
    global validation_agent
    
    if validation_agent is None:
        raise RuntimeError("Validation agent not initialized. Call main() first.")
    
    biosample_id = biosample_data.get("asserted", {}).get("id", "unknown")
    asserted = biosample_data.get("asserted", {})
    inferred = biosample_data.get("inferred", {})
    
    # Only do mathematical/quantitative checks with rules
    elevation_score = calculate_elevation_difference(asserted, inferred)
    enrichment_coverage = calculate_data_coverage_score(inferred)
    
    # Use LLM for all semantic analysis
    try:
        # Get results directory from global context (we'll pass it through the validation function)
        results_dir = Path("archives/data/outputs/crawl-first/test-results/")  # Default path
        validation_result = await run_llm_validation(validation_agent, biosample_data, results_dir)
        
        # Combine rule-based scores with LLM analysis
        validation_result.elevation_plausibility = elevation_score
        validation_result.enrichment_coverage = enrichment_coverage
        validation_result.validation_timestamp = datetime.now().isoformat()
        
        return validation_result
        
    except Exception as e:
        # Fallback if LLM fails
        return ValidationResult(
            biosample_id=biosample_id,
            overall_score=0.5,
            elevation_plausibility=elevation_score,
            enrichment_coverage=enrichment_coverage,
            issues=[f"LLM validation failed: {str(e)}"],
            recommendations=["Check LLM configuration and try again"],
            validation_timestamp=datetime.now().isoformat()
        )


async def run_llm_validation(agent: Agent, biosample_data: Dict[str, Any], results_dir: Path) -> ValidationResult:
    """Use LLM to perform semantic validation of biosample data with dynamic context."""
    global nmdc_schema
    
    biosample_id = biosample_data.get("asserted", {}).get("id", "unknown")
    
    # Build dynamic context sections
    schema_context = extract_relevant_slot_definitions(biosample_data, nmdc_schema) if nmdc_schema else ""
    study_context = get_study_context(biosample_data, results_dir)
    literature_context = get_literature_context(biosample_data)
    
    # Create comprehensive prompt with dynamic context
    prompt = f"""
    Analyze this biosample data for quality and consistency issues:

    BIOSAMPLE ID: {biosample_id}

    ASSERTED DATA (original):
    {yaml.dump(biosample_data.get('asserted', {}), default_flow_style=False)}

    INFERRED DATA (enriched):
    {yaml.dump(biosample_data.get('inferred', {}), default_flow_style=False)}
    {schema_context}
    {study_context}
    {literature_context}

    Evaluate and provide scores (0.0-1.0) for:

    1. **coordinate_consistency**: Do coordinates match the location name and environmental context?
    2. **env_triad_coherence**: Do env_broad_scale → env_local_scale → env_medium form a logical hierarchy?
    3. **ecosystem_alignment**: Do asserted ecosystem fields align with inferred land cover and environmental data?
    4. **overall_score**: Overall data quality based on consistency and reliability

    Additional validation considerations:
    - **Schema compliance**: Check field formats against NMDC schema definitions (especially geo_loc_name format)
    - **Study design consistency**: Compare with other samples from the same study to identify expected vs. unexpected variations
    - **Literature alignment**: Consider if sample metadata aligns with published study methodology

    Identify specific issues and provide actionable recommendations for data improvement.
    Focus particularly on semantic conflicts between asserted and inferred fields.
    """
    
    # Run the LLM validation
    result = await agent.run(prompt)
    
    # Access the structured ValidationResult
    validation_result = result.output
    validation_result.biosample_id = biosample_id  # Ensure biosample_id is set
    
    return validation_result


def calculate_elevation_difference(asserted: Dict, inferred: Dict) -> Optional[float]:
    """Mathematical check: elevation difference between asserted and inferred."""
    asserted_elev = asserted.get("elev")
    
    coord_sources = inferred.get("coordinate_sources", {})
    elevation_comp = coord_sources.get("elevation_comparison", {})
    
    if not asserted_elev or not elevation_comp:
        return None
    
    difference = abs(elevation_comp.get("difference_m", 0))
    
    # Score based on elevation difference magnitude
    if difference <= 5:
        return 1.0
    elif difference <= 20:
        return 0.8
    elif difference <= 50:
        return 0.6
    else:
        return 0.2


def calculate_data_coverage_score(inferred: Dict) -> float:
    """Mathematical check: data coverage and completeness."""
    sections = ['soil_from_asserted_coords', 'land_cover_from_asserted_coords', 
                'weather_from_asserted_coords', 'geospatial_from_asserted_coords']
    
    available_sections = sum(1 for section in sections if inferred.get(section))
    return available_sections / len(sections)


async def validate_directory(results_dir: Path, max_samples: Optional[int] = None, 
                            output_dir: Optional[Path] = None, resume: bool = False) -> List[ValidationResult]:
    """
    Validate all YAML files in the test results directory with streaming output.
    
    Args:
        results_dir: Directory containing YAML biosample files
        max_samples: Maximum number of samples to validate
        output_dir: Directory to write individual validation results (streaming mode)
        resume: Skip samples that already have validation results
    """
    yaml_files = list(results_dir.glob("*.yaml"))
    
    if max_samples:
        yaml_files = yaml_files[:max_samples]
    
    # Filter out already processed files if resuming
    if resume and output_dir and output_dir.exists():
        existing_results = set()
        for result_file in output_dir.glob("validation_*.json"):
            # Extract biosample ID from filename: validation_nmdc:bsm-11-abc123.json
            biosample_id = result_file.stem.replace("validation_", "")
            existing_results.add(biosample_id)
        
        # Filter yaml_files to only include unprocessed samples
        original_count = len(yaml_files)
        yaml_files = [f for f in yaml_files if not any(biosample_id in str(f) for biosample_id in existing_results)]
        skipped_count = original_count - len(yaml_files)
        
        if skipped_count > 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 Resume mode: Skipping {skipped_count} already validated samples")
    
    total_files = len(yaml_files)
    validation_results = []
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Validating {total_files} biosample files...")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Streaming results to: {output_dir}")
    print("Progress: [" + " " * 50 + "] 0%", end="\r")
    
    for i, yaml_file in enumerate(yaml_files, 1):
        try:
            with open(yaml_file, 'r') as f:
                biosample_data = yaml.safe_load(f)
            
            result = await validate_biosample(biosample_data)
            validation_results.append(result)
            
            # Stream individual result to file immediately
            if output_dir:
                # Create safe filename from biosample ID
                safe_id = result.biosample_id.replace(":", "_").replace("/", "_")
                result_file = output_dir / f"validation_{safe_id}.json"
                
                with open(result_file, 'w') as f:
                    json.dump(result.model_dump(), f, indent=2)
            
            # Update progress bar
            progress_percent = int((i / total_files) * 100)
            progress_chars = int((i / total_files) * 50)
            progress_bar = "█" * progress_chars + " " * (50 - progress_chars)
            
            print(f"Progress: [{progress_bar}] {progress_percent}% ({i}/{total_files})", end="\r")
            
            # Show completed validation on new line
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✓ Validated {result.biosample_id} (score: {result.overall_score:.2f})")
            if output_dir:
                print(f"  💾 Saved to: validation_{safe_id}.json")
            
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✗ Error validating {yaml_file.name}: {str(e)}")
            continue
    
    # Final progress completion
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress: [{'█' * 50}] 100% ({total_files}/{total_files})")
    print("="*60)
    
    return validation_results


def collect_streaming_results(stream_dir: Path) -> List[ValidationResult]:
    """Collect all validation results from streaming directory."""
    results = []
    
    if not stream_dir.exists():
        return results
    
    for result_file in stream_dir.glob("validation_*.json"):
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                result = ValidationResult(**result_data)
                results.append(result)
        except Exception as e:
            print(f"⚠️  Error reading {result_file.name}: {str(e)}")
            continue
    
    # Sort by biosample_id for consistent ordering
    results.sort(key=lambda r: r.biosample_id)
    return results


@click.command()
@click.option('--results-dir', default='archives/data/outputs/crawl-first/test-results/', 
              help='Directory containing YAML biosample files to validate')
@click.option('--max-samples', default=10, type=int, 
              help='Maximum number of samples to validate')
@click.option('--model', default='anthropic/claude-sonnet', 
              help='Model name to use')
@click.option('--base-url', envvar='OPENAI_BASE_URL', default='https://api.cborg.lbl.gov/v1',
              help='Base URL for OpenAI-compatible API')
@click.option('--api-key', envvar='OPENAI_API_KEY', required=True,
              help='API key for the LLM service')
@click.option('--output', '-o', help='Output file for validation results (JSON)')
@click.option('--stream-dir', help='Directory to stream individual validation results (prevents data loss)')
@click.option('--resume', is_flag=True, help='Resume validation by skipping already processed samples')
def main(results_dir, max_samples, model, base_url, api_key, output, stream_dir, resume):
    """Validate biosample enrichment data using AI analysis."""
    
    async def async_main():
        # Initialize global variables
        global validation_agent, nmdc_schema
        
        # Create validation agent with specified configuration
        validation_agent = create_validation_agent(model, base_url, api_key)
        
        # Fetch NMDC schema for dynamic validation
        start_time = datetime.now()
        click.echo(f"[{start_time.strftime('%H:%M:%S')}] 🔄 Fetching NMDC schema...")
        nmdc_schema = await fetch_nmdc_schema()
        if nmdc_schema:
            schema_time = datetime.now()
            click.echo(f"[{schema_time.strftime('%H:%M:%S')}] ✅ NMDC schema loaded ({len(nmdc_schema.get('slots', {}))} slots)")
        else:
            click.echo(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  NMDC schema not available - continuing without schema validation")
        
        results_path = Path(results_dir)
        if not results_path.exists():
            click.echo(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Results directory not found: {results_path}")
            return 1
        
        click.echo(f"[{datetime.now().strftime('%H:%M:%S')}] 🔬 Using model: {model}")
        click.echo(f"[{datetime.now().strftime('%H:%M:%S')}] 🌐 Using endpoint: {base_url}")
        
        # Prepare streaming directory
        stream_path = None
        if stream_dir:
            stream_path = Path(stream_dir)
            click.echo(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Streaming mode: Individual results will be saved to {stream_path}")
            if resume:
                click.echo(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Resume mode enabled - will skip already validated samples")
        
        # Run validation
        try:
            results = await validate_directory(results_path, max_samples, stream_path, resume)
        except Exception as e:
            click.echo(f"❌ Validation failed: {str(e)}")
            return 1
        
        return results
    
    # Run the async main function
    try:
        results = asyncio.run(async_main())
        if isinstance(results, int):  # Error code
            return results
    except Exception as e:
        click.echo(f"❌ Validation failed: {str(e)}")
        return 1
    
    # If streaming was used but no consolidated output requested, collect results for summary
    if stream_path and not output and not results:
        results = collect_streaming_results(stream_path)
        click.echo(f"📊 Collected {len(results)} results from streaming directory for summary")
    
    # Print summary
    scores = [r.overall_score for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    click.echo(f"\n🎯 Validation Summary:")
    click.echo(f"   Average quality score: {avg_score:.2f}")
    click.echo(f"   Samples validated: {len(results)}")
    click.echo(f"   Issues found: {sum(len(r.issues) for r in results)}")
    
    # Show top issues
    all_issues = []
    for r in results:
        all_issues.extend(r.issues)
    
    issue_counts = Counter(all_issues)
    
    if issue_counts:
        click.echo(f"\n🔍 Most Common Issues:")
        for issue, count in issue_counts.most_common(3):
            click.echo(f"   • {issue} ({count}x)")
    
    # Save consolidated results if requested
    if output:
        # If results weren't collected yet (e.g., non-streaming mode), use what we have
        if not results and stream_path:
            results = collect_streaming_results(stream_path)
        
        results_data = [r.model_dump() for r in results]
        with open(output, 'w') as f:
            json.dump(results_data, f, indent=2)
        click.echo(f"💾 Consolidated results saved to: {output}")
    
    # Show streaming directory info if used
    if stream_path:
        individual_count = len(list(stream_path.glob("validation_*.json")))
        click.echo(f"💾 Individual results available in: {stream_path} ({individual_count} files)")
        click.echo(f"💡 To resume validation later: --stream-dir {stream_path} --resume")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Extract and normalize biosample data for API enrichment.

Outputs clean JSON containing normalized NMDC and GOLD biosample data
suitable for feeding into Google Plus API functions.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import click
from crawl_first.biosample_adapters import (
    NMDCBiosampleAdapter,
    GOLDBiosampleAdapter,
    BiosampleLocation
)


def load_biosample_data(input_file: Path):
    """Load biosample data from input file."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file) as f:
        return json.load(f)


def normalize_biosamples(data, include_non_enrichable=False):
    """Extract and normalize biosamples for API enrichment."""
    
    # Initialize adapters
    nmdc_adapter = NMDCBiosampleAdapter()
    gold_adapter = GOLDBiosampleAdapter()
    
    # Process NMDC samples
    nmdc_results = []
    nmdc_total = 0
    for sample in data.get("nmdc_samples", []):
        nmdc_total += 1
        location = nmdc_adapter.extract_location(sample)
        if location.is_enrichable() or include_non_enrichable:
            nmdc_results.append(location.to_dict())
    
    # Process GOLD samples
    gold_results = []
    gold_total = 0
    for sample in data.get("gold_samples", []):
        gold_total += 1
        location = gold_adapter.extract_location(sample)
        if location.is_enrichable() or include_non_enrichable:
            gold_results.append(location.to_dict())
    
    # Combine results
    all_results = nmdc_results + gold_results
    
    # Create normalized output
    normalized_output = {
        "metadata": {
            "normalization_timestamp": datetime.now().isoformat(),
            "total_input_samples": nmdc_total + gold_total,
            "total_output_samples": len(all_results),
            "nmdc_input": nmdc_total,
            "nmdc_output": len(nmdc_results),
            "gold_input": gold_total,
            "gold_output": len(gold_results),
            "enrichable_only": not include_non_enrichable,
            "source_metadata": data.get("metadata", {})
        },
        "results": all_results
    }
    
    return normalized_output


@click.command()
@click.option('--input', '-i', 'input_file', 
              type=click.Path(exists=True, path_type=Path),
              help='Input JSON file containing NMDC and GOLD biosample data')
@click.option('--output', '-o', 'output_file',
              type=click.Path(path_type=Path),
              help='Output JSON file (if not specified, prints to stdout)')
@click.option('--include-non-enrichable', is_flag=True,
              help='Include samples that are not enrichable (missing coordinates/dates)')
@click.option('--verbose', '-v', is_flag=True,
              help='Print verbose status messages to stderr')
def main(input_file, output_file, include_non_enrichable, verbose):
    """Normalize NMDC and GOLD biosample data for API enrichment.
    
    Takes raw biosample data and extracts location/date information needed
    for geospatial API enrichment using the biosample adapters.
    """
    try:
        # Use default input file if not specified
        if not input_file:
            input_file = Path(__file__).parent.parent.parent / "data" / "inputs" / "test_biosamples.json"
            if verbose:
                click.echo(f"Using default input file: {input_file}", err=True)
        
        if verbose:
            click.echo(f"Loading biosample data from: {input_file}", err=True)
        
        # Load input data
        data = load_biosample_data(input_file)
        
        if verbose:
            nmdc_count = len(data.get("nmdc_samples", []))
            gold_count = len(data.get("gold_samples", []))
            click.echo(f"Loaded {nmdc_count} NMDC + {gold_count} GOLD samples", err=True)
        
        # Normalize biosamples
        normalized_data = normalize_biosamples(data, include_non_enrichable)
        
        if verbose:
            meta = normalized_data["metadata"]
            click.echo(f"Normalized {meta['total_input_samples']} → {meta['total_output_samples']} samples", err=True)
            click.echo(f"NMDC: {meta['nmdc_input']} → {meta['nmdc_output']}", err=True)
            click.echo(f"GOLD: {meta['gold_input']} → {meta['gold_output']}", err=True)
        
        # Output results
        json_output = json.dumps(normalized_data, indent=2)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(json_output)
            if verbose:
                click.echo(f"✅ Normalized data saved to: {output_file}", err=True)
        else:
            print(json_output)
        
    except Exception as e:
        click.echo(f"❌ Error normalizing biosamples: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
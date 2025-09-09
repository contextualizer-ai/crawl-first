#!/usr/bin/env python3
"""
Extract and normalize biosample data for API enrichment.

Outputs clean JSON containing normalized NMDC and GOLD biosample data
suitable for feeding into Google Plus API functions.
"""

import json
import sys
from pathlib import Path
from crawl_first.biosample_adapters import (
    NMDCBiosampleAdapter,
    GOLDBiosampleAdapter,
    BiosampleLocation
)


def load_test_data():
    """Load test biosample data."""
    test_file = Path(__file__).parent.parent.parent / "data" / "inputs" / "test_biosamples.json"
    with open(test_file) as f:
        return json.load(f)


def normalize_biosamples():
    """Extract and normalize biosamples for API enrichment."""
    data = load_test_data()
    
    # Initialize adapters
    nmdc_adapter = NMDCBiosampleAdapter()
    gold_adapter = GOLDBiosampleAdapter()
    
    # Process NMDC samples
    nmdc_results = []
    for sample in data.get("nmdc_samples", []):
        location = nmdc_adapter.extract_location(sample)
        if location.is_enrichable():
            nmdc_results.append(location.to_dict())
    
    # Process GOLD samples
    gold_results = []
    for sample in data.get("gold_samples", []):
        location = gold_adapter.extract_location(sample)
        if location.is_enrichable():
            gold_results.append(location.to_dict())
    
    # Combine results
    all_results = nmdc_results + gold_results
    
    # Create normalized output
    normalized_output = {
        "metadata": {
            "normalization_timestamp": "2024-09-08T00:00:00Z",
            "total_samples": len(all_results),
            "nmdc_samples": len(nmdc_results),
            "gold_samples": len(gold_results),
            "enrichable_only": True
        },
        "results": all_results
    }
    
    return normalized_output


def main():
    """Main entry point - output clean JSON only."""
    try:
        normalized_data = normalize_biosamples()
        print(json.dumps(normalized_data, indent=2))
    except Exception as e:
        # Write error to stderr so stdout remains clean JSON
        print(f"Error normalizing biosamples: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test enhanced biosample adapters using realistic test data.

Demonstrates ID normalization, study extraction, and separate ID lists
using actual NMDC and GOLD biosample structures.
"""

import json
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


def test_nmdc_adapter():
    """Test NMDC adapter with realistic data."""
    print("🧬 Testing NMDC Adapter with Enhanced Features")
    print("=" * 50)
    
    data = load_test_data()
    adapter = NMDCBiosampleAdapter()
    
    results = []
    for sample in data["nmdc_samples"]:
        location = adapter.extract_location(sample)
        results.append(location.to_dict())
        
        print(f"\nSample ID: {location.sample_id}")
        print(f"NMDC ID: {location.nmdc_biosample_id}")
        print(f"GOLD ID: {location.gold_biosample_id}")
        print(f"Coordinates: ({location.latitude}, {location.longitude})")
        print(f"Collection Date: {location.collection_date}")
        print(f"Location: {location.textual_location}")
        print(f"NMDC Studies: {location.nmdc_studies}")
        print(f"Alternative IDs: {location.alternative_identifiers}")
        print(f"External DB IDs: {location.external_database_identifiers}")
        print(f"Biosample IDs: {location.biosample_identifiers}")
        print(f"Sample IDs: {location.sample_identifiers}")
        print(f"Enrichable: {location.is_enrichable()}")
    
    return {
        "adapter_type": "NMDC",
        "samples_processed": len(results),
        "enrichable_samples": sum(1 for r in results if r["is_enrichable"]),
        "samples": results
    }


def test_gold_adapter():
    """Test GOLD adapter with realistic data."""
    print("\n\n🏅 Testing GOLD Adapter with Enhanced Features")
    print("=" * 50)
    
    data = load_test_data()
    adapter = GOLDBiosampleAdapter()
    
    # Create a mock database for testing study lookup if sample seq_projects available
    mock_database = None
    if "sample_seq_projects" in data and data["sample_seq_projects"]:
        mock_database = {
            "seq_projects": MockSeqProjectsCollection(data["sample_seq_projects"])
        }
    
    results = []
    for sample in data["gold_samples"]:
        location = adapter.extract_location(sample, mock_database)
        results.append(location.to_dict())
        
        print(f"\nSample ID: {location.sample_id}")
        print(f"GOLD ID: {location.gold_biosample_id}")
        print(f"NMDC ID: {location.nmdc_biosample_id}")
        print(f"Coordinates: ({location.latitude}, {location.longitude})")
        print(f"Collection Date: {location.collection_date}")
        print(f"Location: {location.textual_location}")
        print(f"GOLD Studies: {location.gold_studies}")
        print(f"Alternative IDs: {location.alternative_identifiers}")
        print(f"External DB IDs: {location.external_database_identifiers}")
        print(f"Biosample IDs: {location.biosample_identifiers}")
        print(f"Sample IDs: {location.sample_identifiers}")
        print(f"Enrichable: {location.is_enrichable()}")
    
    return {
        "adapter_type": "GOLD",
        "samples_processed": len(results),
        "enrichable_samples": sum(1 for r in results if r["is_enrichable"]),
        "samples": results
    }


class MockSeqProjectsCollection:
    """Mock seq_projects collection for testing GOLD study lookup."""
    
    def __init__(self, seq_projects_data):
        self.data = seq_projects_data
    
    def find(self, query):
        """Mock find method that returns matching seq_projects."""
        biosample_gold_id = query.get("biosampleGoldId")
        if biosample_gold_id:
            return [proj for proj in self.data if proj["biosampleGoldId"] == biosample_gold_id]
        return []


def test_id_normalization():
    """Test ID normalization across both adapters."""
    print("\n\n🆔 Testing ID Normalization and Cross-References")
    print("=" * 50)
    
    data = load_test_data()
    nmdc_adapter = NMDCBiosampleAdapter()
    gold_adapter = GOLDBiosampleAdapter()
    
    # Test cross-references between NMDC and GOLD
    nmdc_sample = data["nmdc_samples"][0]  # Has gold_biosample_id
    gold_sample = data["gold_samples"][0]  # Has nmdc_biosample_id
    
    nmdc_location = nmdc_adapter.extract_location(nmdc_sample)
    gold_location = gold_adapter.extract_location(gold_sample)
    
    print(f"NMDC sample references GOLD ID: {nmdc_location.gold_biosample_id}")
    print(f"GOLD sample references NMDC ID: {gold_location.nmdc_biosample_id}")
    print(f"Cross-reference match: {nmdc_location.gold_biosample_id == gold_location.gold_biosample_id}")
    
    # Test separate ID lists
    print(f"\nNMDC ID separation:")
    print(f"  Alternative: {nmdc_location.alternative_identifiers}")
    print(f"  External DB: {nmdc_location.external_database_identifiers}")
    print(f"  Biosample: {nmdc_location.biosample_identifiers}")
    print(f"  Sample: {nmdc_location.sample_identifiers}")
    
    print(f"\nGOLD ID separation:")
    print(f"  Alternative: {gold_location.alternative_identifiers}")
    print(f"  External DB: {gold_location.external_database_identifiers}")
    print(f"  Biosample: {gold_location.biosample_identifiers}")
    print(f"  Sample: {gold_location.sample_identifiers}")


def main():
    """Run all enhanced adapter tests."""
    print("Enhanced Biosample Adapter Testing with Realistic Data")
    print("=" * 60)
    
    # Test adapters
    nmdc_results = test_nmdc_adapter()
    gold_results = test_gold_adapter()
    test_id_normalization()
    
    # Summary
    print("\n\n📊 Test Summary")
    print("=" * 30)
    print(f"NMDC: {nmdc_results['samples_processed']} samples, "
          f"{nmdc_results['enrichable_samples']} enrichable")
    print(f"GOLD: {gold_results['samples_processed']} samples, "
          f"{gold_results['enrichable_samples']} enrichable")
    
    # Output JSON for further analysis
    output = {
        "test_timestamp": "2024-09-08T00:00:00Z",
        "nmdc_results": nmdc_results,
        "gold_results": gold_results,
        "test_summary": {
            "total_samples": nmdc_results['samples_processed'] + gold_results['samples_processed'],
            "total_enrichable": nmdc_results['enrichable_samples'] + gold_results['enrichable_samples'],
            "features_tested": [
                "ID normalization",
                "Study extraction", 
                "Separate ID lists",
                "Cross-database references",
                "MongoDB study lookup simulation"
            ]
        }
    }
    
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
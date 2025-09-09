#!/usr/bin/env python3
"""
Test script for the unified enrichment system.

Demonstrates the comprehensive geospatial enrichment pipeline with a single
biosample example using the new modular architecture.
"""

import sys
from pathlib import Path

# Add src to path for imports  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawl_first.unified_enrichment import enrich_biosample_unified, load_enrichment_config
import json


def test_enrichment_config():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    config = load_enrichment_config()
    print(f"✓ Configuration loaded: version {config.get('version', 'unknown')}")
    print(f"  - Datasets configured: {len(config.get('datasets', {}))}")
    print(f"  - Providers configured: {len(config.get('providers', {}))}")
    print(f"  - Weather providers: {config.get('providers', {}).get('weather', {})}")
    print()


def test_single_biosample():
    """Test enrichment for a single biosample."""
    print("Testing single biosample enrichment...")
    
    # Test location: Central Park, NYC
    biosample_id = "test_sample_001"
    lat = 40.7829
    lon = -73.9654
    collection_date = "2023-08-15"
    
    print(f"Enriching biosample: {biosample_id}")
    print(f"Location: {lat:.4f}, {lon:.4f}")
    print(f"Collection date: {collection_date}")
    print()
    
    try:
        result = enrich_biosample_unified(
            biosample_id=biosample_id,
            lat=lat,
            lon=lon,
            collection_date=collection_date
        )
        
        print("✓ Enrichment completed successfully")
        print(f"  - Success rate: {result.get('enrichment_success_rate', 0):.1%}")
        
        if result.get("enrichment_errors"):
            print(f"  - Errors encountered: {len(result['enrichment_errors'])}")
            for error in result["enrichment_errors"][:3]:  # Show first 3 errors
                print(f"    • {error}")
        
        # Show key results
        print("\nKey Results:")
        
        # Site type
        site_type = result.get("site_type", {})
        if site_type.get("type"):
            print(f"  Site Type: {site_type['type']}")
        
        # Weather
        weather = result.get("weather", {}).get("daily", {})
        if weather.get("success"):
            temp_avg = weather.get("temperature_2m_mean")
            if temp_avg:
                print(f"  Weather: {temp_avg:.1f}°C avg, provider: {weather.get('provider')}")
        
        # OSM features
        osm = result.get("osm", {})
        if osm.get("named_features"):
            print(f"  OSM: {len(osm['named_features'])} named features, {osm.get('summary', {}).get('total_unnamed', 0)} unnamed")
        
        # Land cover
        land_cover = result.get("land_cover", {})
        if land_cover:
            datasets = list(land_cover.keys())
            print(f"  Land Cover: {len(datasets)} datasets available")
        
        # Soils
        soils = result.get("soils", {}).get("soilgrids", {})
        if soils:
            soil_props = [k for k, v in soils.items() if v.get("value") is not None]
            print(f"  Soils: {len(soil_props)} properties retrieved")
        
        print(f"\nProvenance: {result.get('provenance', {}).get('software', 'unknown')}")
        print(f"Created: {result.get('provenance', {}).get('created_at', 'unknown')}")
        
        return result
        
    except Exception as e:
        print(f"✗ Enrichment failed: {e}")
        return None


def test_batch_enrichment():
    """Test batch enrichment for multiple locations."""
    print("\n" + "="*60)
    print("Testing batch enrichment...")
    
    # Test locations
    biosamples = [
        {"id": "test_central_park", "lat": 40.7829, "lon": -73.9654, "collection_date": "2023-08-15"},
        {"id": "test_yellowstone", "lat": 44.4280, "lon": -110.5885, "collection_date": "2023-07-20"},
        {"id": "test_miami_beach", "lat": 25.7907, "lon": -80.1300, "collection_date": "2023-06-10"}
    ]
    
    from crawl_first.unified_enrichment import batch_enrich_biosamples
    
    try:
        results = batch_enrich_biosamples(biosamples)
        
        print(f"✓ Batch enrichment completed: {len(results)} samples")
        
        for i, result in enumerate(results):
            sample_id = result.get("id", f"sample_{i}")
            success_rate = result.get("enrichment_success_rate", 0)
            site_type = result.get("site_type", {}).get("type", "unknown")
            
            print(f"  {sample_id}: {success_rate:.1%} success, {site_type}")
        
        return results
        
    except Exception as e:
        print(f"✗ Batch enrichment failed: {e}")
        return []


def save_test_results(single_result, batch_results):
    """Save test results to files."""
    output_dir = Path("data") / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if single_result:
        single_path = output_dir / "test_single_enrichment.json"
        with open(single_path, 'w') as f:
            json.dump(single_result, f, indent=2, default=str)
        print(f"\nSingle enrichment result saved to: {single_path}")
    
    if batch_results:
        batch_path = output_dir / "test_batch_enrichment.json"
        with open(batch_path, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        print(f"Batch enrichment results saved to: {batch_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED GEOSPATIAL ENRICHMENT SYSTEM TEST")
    print("=" * 60)
    
    # Test configuration
    test_enrichment_config()
    
    # Test single biosample
    single_result = test_single_biosample()
    
    # Test batch enrichment
    batch_results = test_batch_enrichment()
    
    # Save results
    save_test_results(single_result, batch_results)
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
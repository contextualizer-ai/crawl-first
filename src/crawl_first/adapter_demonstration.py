#!/usr/bin/env python3
"""Generate comprehensive adapter demonstration results."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from crawl_first.biosample_adapters import GOLDBiosampleAdapter, NMDCBiosampleAdapter

# Generate comprehensive test results
nmdc_samples = [
    {
        "id": "nmdc:demo1",
        "lat_lon": "42.3601 -71.0928",
        "collection_date": "2023-06-15",
        "geo_loc_name": "MIT, Cambridge, MA",
    },
    {
        "id": "nmdc:demo2",
        "lat_lon": [40.7128, -74.0060],
        "collection_date": "2023-06-16",
        "geo_loc_name": "NYC, NY",
    },
    {
        "id": "nmdc:demo3",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "collection_date": "2023-06-17",
    },
]

gold_samples = [
    {
        "biosampleGoldId": "Gb_demo1",
        "latitude": 36.6002,
        "longitude": -121.8947,
        "dateCollected": "2023-07-20",
        "geoLocation": "Monterey Bay, CA",
    },
    {
        "biosampleGoldId": "Gb_demo2",
        "latitude": 59.9139,
        "longitude": 10.7522,
        "dateCollected": "2023-08-05",
        "geoLocation": "Oslo, Norway",
    },
]

nmdc_adapter = NMDCBiosampleAdapter()
gold_adapter = GOLDBiosampleAdapter()

nmdc_results = [nmdc_adapter.extract_location(s).to_dict() for s in nmdc_samples]
gold_results = [gold_adapter.extract_location(s).to_dict() for s in gold_samples]

all_results = nmdc_results + gold_results
enrichable_count = sum(1 for r in all_results if r["is_enrichable"])

# Calculate statistics
completeness_scores = [
    r["location_completeness"]
    for r in all_results
    if r["location_completeness"] is not None
]
avg_completeness = (
    sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
)

output = {
    "demonstration_summary": {
        "total_samples_tested": len(all_results),
        "nmdc_samples": len(nmdc_results),
        "gold_samples": len(gold_results),
        "enrichable_samples": enrichable_count,
        "enrichment_rate": enrichable_count / len(all_results) if all_results else 0,
        "average_completeness": round(avg_completeness, 3),
    },
    "adapter_capabilities": {
        "nmdc_coordinate_formats": [
            "space_separated",
            "array",
            "dict",
            "separate_fields",
        ],
        "gold_coordinate_formats": ["separate_numeric_fields"],
        "date_formats": ["iso_datetime", "date_only", "year_month", "year_only"],
        "location_text_sources": [
            "geo_loc_name",
            "geoLocation",
            "geographic_location",
            "description",
        ],
    },
    "api_enrichment_readiness": {
        "coordinates_available": enrichable_count,
        "elevation_api_ready": enrichable_count,
        "weather_api_ready": sum(
            1 for r in all_results if r["collection_date"] and r["is_enrichable"]
        ),
        "geocoding_api_ready": enrichable_count,
    },
    "nmdc_results": nmdc_results,
    "gold_results": gold_results,
}

print(json.dumps(output, indent=2))

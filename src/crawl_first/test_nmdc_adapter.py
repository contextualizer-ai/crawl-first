#!/usr/bin/env python3
"""Test NMDC biosample adapter with sample data."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from crawl_first.biosample_adapters import BiosampleLocation, NMDCBiosampleAdapter

# Sample NMDC data
samples = [
    {
        "id": "nmdc:bsm-12-test1",
        "lat_lon": "42.3601 -71.0928",
        "collection_date": "2023-06-15T10:30:00Z",
        "geo_loc_name": "MIT Campus, Cambridge, MA",
    },
    {
        "id": "nmdc:bsm-12-test2",
        "lat_lon": {"latitude": 40.7128, "longitude": -74.0060},
        "collection_date": "2023-06-16",
        "geo_loc_name": "Central Park, New York, NY",
    },
    {
        "id": "nmdc:bsm-12-test3",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "collection_date": "2023-06",
        "geographic_location": "Golden Gate Park, San Francisco, CA",
    },
]

adapter = NMDCBiosampleAdapter()
results = []

for sample in samples:
    location = adapter.extract_location(sample)
    results.append(location.to_dict())

output = {
    "adapter_type": "NMDC",
    "test_timestamp": BiosampleLocation().extraction_timestamp,
    "samples_processed": len(samples),
    "enrichable_samples": sum(1 for r in results if r["is_enrichable"]),
    "results": results,
}

print(json.dumps(output, indent=2))

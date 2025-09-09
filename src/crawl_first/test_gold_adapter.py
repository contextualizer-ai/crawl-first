#!/usr/bin/env python3
"""Test GOLD biosample adapter with sample data."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from crawl_first.biosample_adapters import BiosampleLocation, GOLDBiosampleAdapter

# Sample GOLD data
samples = [
    {
        "biosampleGoldId": "Gb0123456",
        "latitude": 36.6002,
        "longitude": -121.8947,
        "dateCollected": "2023-07-20T14:15:00Z",
        "geoLocation": "Monterey Bay, California, USA",
    },
    {
        "biosampleGoldId": "Gb0123457",
        "latitude": "59.9139",
        "longitude": "10.7522",
        "dateCollected": "2023-08-05",
        "geoLocation": "Oslo Fjord, Norway",
    },
    {
        "biosampleGoldId": "Gb0123458",
        "latitude": -3.4653,
        "longitude": -62.2159,
        "dateCollected": "2023-09-12",
        "geoLocation": "Amazon Rainforest, Brazil",
    },
]

adapter = GOLDBiosampleAdapter()
results = []

for sample in samples:
    location = adapter.extract_location(sample)
    results.append(location.to_dict())

output = {
    "adapter_type": "GOLD",
    "test_timestamp": BiosampleLocation().extraction_timestamp,
    "samples_processed": len(samples),
    "enrichable_samples": sum(1 for r in results if r["is_enrichable"]),
    "results": results,
}

print(json.dumps(output, indent=2))

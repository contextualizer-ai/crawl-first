#!/usr/bin/env python3
"""Test MongoDB adapter configuration (stub - no actual connection)."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from crawl_first.biosample_adapters import (
    MongoGOLDBiosampleFetcher,
    MongoNMDCBiosampleFetcher,
    UnifiedBiosampleFetcher,
)

# Test adapter configuration without actual connection
nmdc_fetcher = MongoNMDCBiosampleFetcher(
    connection_string="mongodb://localhost:27017",
    database_name="nmdc_test",
    collection_name="biosamples",
)

gold_fetcher = MongoGOLDBiosampleFetcher(
    connection_string="mongodb://localhost:27017",
    database_name="gold_test",
    collection_name="biosamples",
)

unified = UnifiedBiosampleFetcher()
unified.configure_nmdc_mongo("mongodb://nmdc-server:27017")
unified.configure_gold_mongo("mongodb://gold-server:27017")

output = {
    "mongodb_adapter_test": {
        "status": "configuration_only",
        "note": "Actual MongoDB connection requires database setup",
        "nmdc_fetcher": {
            "database": nmdc_fetcher.database_name,
            "collection": nmdc_fetcher.collection_name,
            "adapter_type": type(nmdc_fetcher.adapter).__name__,
        },
        "gold_fetcher": {
            "database": gold_fetcher.database_name,
            "collection": gold_fetcher.collection_name,
            "adapter_type": type(gold_fetcher.adapter).__name__,
        },
        "unified_interface": {
            "configured": True,
            "nmdc_ready": unified.nmdc_mongo is not None,
            "gold_ready": unified.gold_mongo is not None,
        },
    },
    "usage_examples": {
        "fetch_enrichable": 'unified.fetch_enrichable_locations(source="all", limit=1000)',
        "get_statistics": "unified.get_enrichment_statistics()",
        "nmdc_only": 'unified.fetch_enrichable_locations(source="nmdc", limit=500)',
        "gold_only": 'unified.fetch_enrichable_locations(source="gold", limit=500)',
    },
}

print(json.dumps(output, indent=2))

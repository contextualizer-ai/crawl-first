#!/usr/bin/env python3
"""
Example usage of biosample adapters for geospatial enrichment.

Demonstrates how to extract lat/lon/date/location from NMDC and GOLD
biosample data for API enrichment workflows.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from crawl_first.biosample_adapters import (
    GOLDBiosampleAdapter,
    MongoGOLDBiosampleFetcher,
    MongoNMDCBiosampleFetcher,
    NMDCBiosampleAdapter,
    UnifiedBiosampleFetcher,
)


def demonstrate_nmdc_adapter():
    """Demonstrate NMDC biosample data extraction."""
    print("🧬 NMDC Biosample Adapter Demo")
    print("=" * 40)

    # Sample NMDC biosample data with different coordinate formats
    nmdc_samples = [
        {
            "id": "nmdc:bsm-12-example1",
            "lat_lon": "42.3601 -71.0928",  # Space-separated
            "collection_date": "2023-06-15T10:30:00Z",
            "geo_loc_name": "Massachusetts Institute of Technology, Cambridge, MA",
        },
        {
            "id": "nmdc:bsm-12-example2",
            "lat_lon": {"latitude": 40.7128, "longitude": -74.0060},  # Dict format
            "collection_date": "2023-06-16",
            "geo_loc_name": "Central Park, New York, NY",
        },
        {
            "id": "nmdc:bsm-12-example3",
            "latitude": 37.7749,
            "longitude": -122.4194,  # Separate fields
            "collection_date": "2023-06",  # Month precision
            "geographic_location": "Golden Gate Park, San Francisco, CA",
        },
        {
            "id": "nmdc:bsm-12-example4",
            "lat_lon": [44.4280, -110.5885],  # Array format
            "collection_date": "2023",  # Year precision
            "geo_loc_name": "Yellowstone National Park, WY",
        },
    ]

    adapter = NMDCBiosampleAdapter()

    for i, sample in enumerate(nmdc_samples, 1):
        print(f"\n📍 Sample {i}: {sample['id']}")
        location = adapter.extract_location(sample)

        print(f"   Coordinates: ({location.latitude}, {location.longitude})")
        print(
            f"   Date: {location.collection_date} (precision: {location.date_precision})"
        )
        print(f"   Location: {location.textual_location}")
        print(f"   Enrichable: {'✅' if location.is_enrichable() else '❌'}")
        print(f"   Completeness: {location.location_completeness:.1%}")
        print(f"   Coord Precision: {location.coordinate_precision} decimal places")


def demonstrate_gold_adapter():
    """Demonstrate GOLD biosample data extraction."""
    print("\n\n🏅 GOLD Biosample Adapter Demo")
    print("=" * 40)

    # Sample GOLD biosample data
    gold_samples = [
        {
            "biosampleGoldId": "Gb0123456",
            "latitude": 36.6002,
            "longitude": -121.8947,
            "dateCollected": "2023-07-20T14:15:00Z",
            "geoLocation": "Monterey Bay, California, USA",
        },
        {
            "biosampleGoldId": "Gb0123457",
            "latitude": "59.9139",  # String coordinates
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
            "description": "Tropical forest soil sample",
        },
    ]

    adapter = GOLDBiosampleAdapter()

    for i, sample in enumerate(gold_samples, 1):
        print(f"\n📍 Sample {i}: {sample['biosampleGoldId']}")
        location = adapter.extract_location(sample)

        print(f"   Coordinates: ({location.latitude}, {location.longitude})")
        print(f"   Date: {location.collection_date}")
        print(f"   Location: {location.textual_location}")
        print(f"   Enrichable: {'✅' if location.is_enrichable() else '❌'}")
        print(f"   Completeness: {location.location_completeness:.1%}")


def demonstrate_mongodb_setup():
    """Demonstrate MongoDB fetcher setup (without actual connection)."""
    print("\n\n🗄️  MongoDB Fetcher Setup Demo")
    print("=" * 40)

    # Example MongoDB configuration (would require actual database)
    nmdc_fetcher = MongoNMDCBiosampleFetcher(
        connection_string="mongodb://localhost:27017",
        database_name="nmdc_production",
        collection_name="biosamples",
    )

    gold_fetcher = MongoGOLDBiosampleFetcher(
        connection_string="mongodb://localhost:27017",
        database_name="gold_production",
        collection_name="biosamples",
    )

    print("📊 NMDC MongoDB Fetcher configured:")
    print(f"   Database: {nmdc_fetcher.database_name}")
    print(f"   Collection: {nmdc_fetcher.collection_name}")
    print(f"   Adapter: {type(nmdc_fetcher.adapter).__name__}")

    print("\n📊 GOLD MongoDB Fetcher configured:")
    print(f"   Database: {gold_fetcher.database_name}")
    print(f"   Collection: {gold_fetcher.collection_name}")
    print(f"   Adapter: {type(gold_fetcher.adapter).__name__}")


def demonstrate_unified_interface():
    """Demonstrate unified fetcher interface."""
    print("\n\n🔗 Unified Fetcher Interface Demo")
    print("=" * 40)

    # Create unified fetcher
    unified = UnifiedBiosampleFetcher()

    # Configure both databases (example configuration)
    unified.configure_nmdc_mongo(
        connection_string="mongodb://nmdc-db:27017",
        database="nmdc_production",
        collection="biosamples",
    )

    unified.configure_gold_mongo(
        connection_string="mongodb://gold-db:27017",
        database="gold_production",
        collection="biosamples",
    )

    print("🔧 Unified fetcher configured for both NMDC and GOLD")
    print("   Ready to fetch enrichable locations from both databases")
    print("   Usage examples:")
    print("     - unified.fetch_enrichable_locations(source='nmdc', limit=100)")
    print("     - unified.fetch_enrichable_locations(source='gold', limit=50)")
    print("     - unified.fetch_enrichable_locations(source='all', limit=1000)")
    print("     - unified.get_enrichment_statistics()")


def demonstrate_api_enrichment_workflow():
    """Demonstrate complete workflow from biosample to API enrichment."""
    print("\n\n🌍 API Enrichment Workflow Demo")
    print("=" * 40)

    # Sample biosample with coordinates and date
    sample_data = {
        "id": "nmdc:bsm-12-workflow-demo",
        "lat_lon": "42.3601 -71.0928",
        "collection_date": "2023-06-15",
        "geo_loc_name": "MIT Campus, Cambridge, MA",
    }

    # Extract location data
    adapter = NMDCBiosampleAdapter()
    location = adapter.extract_location(sample_data)

    print("📊 Extracted Location Data:")
    print(f"   Sample ID: {location.sample_id}")
    print(f"   Coordinates: ({location.latitude}, {location.longitude})")
    print(f"   Collection Date: {location.collection_date}")
    print(f"   Location Name: {location.textual_location}")
    print(f"   Enrichable: {'✅' if location.is_enrichable() else '❌'}")

    if location.is_enrichable():
        print("\n🔗 Ready for API Enrichment:")
        print(f"   Elevation API: lat={location.latitude}, lon={location.longitude}")
        print(
            f"   Weather API: lat={location.latitude}, lon={location.longitude}, date={location.collection_date}"
        )
        print(
            f"   Reverse Geocoding: lat={location.latitude}, lon={location.longitude}"
        )
        print(f"   Forward Geocoding: query='{location.textual_location}'")

        # Convert to API-ready format
        api_params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "collection_date": location.collection_date,
            "location_query": location.textual_location,
        }

        print("\n📋 API Parameters Dictionary:")
        for key, value in api_params.items():
            print(f"   {key}: {value}")


def main():
    """Run all demonstration examples."""
    print("🧪 Biosample Adapters Demonstration")
    print("=" * 50)

    demonstrate_nmdc_adapter()
    demonstrate_gold_adapter()
    demonstrate_mongodb_setup()
    demonstrate_unified_interface()
    demonstrate_api_enrichment_workflow()

    print("\n\n✅ All demonstrations complete!")
    print("\n💡 Next Steps:")
    print("   1. Connect to actual MongoDB instances")
    print("   2. Configure connection strings in environment")
    print("   3. Use adapters with API enrichment pipeline")
    print("   4. Implement batch processing for large datasets")


if __name__ == "__main__":
    main()

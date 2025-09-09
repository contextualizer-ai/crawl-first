"""
Test biosample adapters for NMDC and GOLD data extraction.

Tests the adapter functionality with sample data to ensure proper extraction
of latitude, longitude, collection_date, and textual location names.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from crawl_first.biosample_adapters import (
    BiosampleLocation,
    GOLDBiosampleAdapter,
    MongoGOLDBiosampleFetcher,
    MongoNMDCBiosampleFetcher,
    NMDCBiosampleAdapter,
    UnifiedBiosampleFetcher,
)


class TestBiosampleLocation:
    """Test BiosampleLocation dataclass functionality."""

    def test_complete_location(self):
        """Test location with all required fields."""
        location = BiosampleLocation(
            latitude=42.3601,
            longitude=-71.0928,
            collection_date="2023-06-15",
            textual_location="Cambridge, Massachusetts, USA",
            sample_id="nmdc:bsm-12-test123",
            database_source="NMDC",
        )

        assert location.is_enrichable() == True
        assert location.location_completeness == 1.0
        assert location.database_source == "NMDC"

        # Test dictionary conversion
        data = location.to_dict()
        assert data["latitude"] == 42.3601
        assert data["is_enrichable"] == True

    def test_minimal_location(self):
        """Test location with only coordinates."""
        location = BiosampleLocation(latitude=42.3601, longitude=-71.0928)

        assert location.is_enrichable() == True
        assert location.location_completeness == 0.5  # 2/4 required fields

    def test_invalid_coordinates(self):
        """Test location with invalid coordinates."""
        location = BiosampleLocation(latitude=200.0, longitude=-71.0928)  # Invalid

        assert location.is_enrichable() == False

    def test_missing_coordinates(self):
        """Test location without coordinates."""
        location = BiosampleLocation(
            collection_date="2023-06-15", textual_location="Cambridge, MA"
        )

        assert location.is_enrichable() == False
        assert location.location_completeness == 0.5  # 2/4 required fields


class TestNMDCBiosampleAdapter:
    """Test NMDC biosample data adapter."""

    def setup_method(self):
        self.adapter = NMDCBiosampleAdapter()

    def test_complete_nmdc_sample(self):
        """Test NMDC sample with all fields."""
        nmdc_sample = {
            "id": "nmdc:bsm-12-test123",
            "lat_lon": "42.3601 -71.0928",
            "collection_date": "2023-06-15T10:30:00Z",
            "geo_loc_name": "Cambridge, Massachusetts, USA",
        }

        location = self.adapter.extract_location(nmdc_sample)

        assert location.latitude == 42.3601
        assert location.longitude == -71.0928
        assert location.collection_date == "2023-06-15"
        assert location.textual_location == "Cambridge, Massachusetts, USA"
        assert location.sample_id == "nmdc:bsm-12-test123"
        assert location.database_source == "NMDC"
        assert location.is_enrichable() == True

    def test_nmdc_lat_lon_formats(self):
        """Test different lat_lon format variations."""
        # Test comma-separated
        sample1 = {"lat_lon": "42.3601,-71.0928"}
        location1 = self.adapter.extract_location(sample1)
        assert location1.latitude == 42.3601
        assert location1.longitude == -71.0928

        # Test dictionary format
        sample2 = {"lat_lon": {"latitude": 42.3601, "longitude": -71.0928}}
        location2 = self.adapter.extract_location(sample2)
        assert location2.latitude == 42.3601
        assert location2.longitude == -71.0928

        # Test array format
        sample3 = {"lat_lon": [42.3601, -71.0928]}
        location3 = self.adapter.extract_location(sample3)
        assert location3.latitude == 42.3601
        assert location3.longitude == -71.0928

        # Test separate fields
        sample4 = {"latitude": 42.3601, "longitude": -71.0928}
        location4 = self.adapter.extract_location(sample4)
        assert location4.latitude == 42.3601
        assert location4.longitude == -71.0928

    def test_nmdc_date_formats(self):
        """Test different date format variations."""
        # Test ISO format with time
        sample1 = {"collection_date": "2023-06-15T10:30:00Z"}
        location1 = self.adapter.extract_location(sample1)
        assert location1.collection_date == "2023-06-15"

        # Test date only
        sample2 = {"collection_date": "2023-06-15"}
        location2 = self.adapter.extract_location(sample2)
        assert location2.collection_date == "2023-06-15"

        # Test year-month only
        sample3 = {"collection_date": "2023-06"}
        location3 = self.adapter.extract_location(sample3)
        assert location3.collection_date == "2023-06-01"

        # Test year only
        sample4 = {"collection_date": "2023"}
        location4 = self.adapter.extract_location(sample4)
        assert location4.collection_date == "2023-01-01"

    def test_nmdc_location_priority(self):
        """Test textual location field priority."""
        sample = {
            "geo_loc_name": "Primary Location",
            "geographic_location": "Secondary Location",
            "description": "Description Location",
        }

        location = self.adapter.extract_location(sample)
        assert location.textual_location == "Primary Location"

    def test_nmdc_precision_assessment(self):
        """Test coordinate and date precision assessment."""
        sample = {
            "lat_lon": "42.360123 -71.092847",  # 6 decimal places
            "collection_date": "2023-06-15",  # Day precision
        }

        location = self.adapter.extract_location(sample)
        assert location.coordinate_precision == 6
        assert location.date_precision == "day"

    def test_batch_extraction(self):
        """Test batch extraction from multiple samples."""
        samples = [
            {
                "id": "sample1",
                "lat_lon": "42.3601 -71.0928",
                "collection_date": "2023-06-15",
            },
            {
                "id": "sample2",
                "lat_lon": "40.7128 -74.0060",
                "collection_date": "2023-06-16",
            },
            {"id": "sample3", "description": "Sample without coordinates"},
        ]

        locations = self.adapter.extract_locations_batch(samples)

        assert len(locations) == 3
        assert locations[0].is_enrichable() == True
        assert locations[1].is_enrichable() == True
        assert locations[2].is_enrichable() == False


class TestGOLDBiosampleAdapter:
    """Test GOLD biosample data adapter."""

    def setup_method(self):
        self.adapter = GOLDBiosampleAdapter()

    def test_complete_gold_sample(self):
        """Test GOLD sample with all fields."""
        gold_sample = {
            "biosampleGoldId": "Gb0123456",
            "latitude": 42.3601,
            "longitude": -71.0928,
            "dateCollected": "2023-06-15",
            "geoLocation": "Cambridge, Massachusetts, USA",
        }

        location = self.adapter.extract_location(gold_sample)

        assert location.latitude == 42.3601
        assert location.longitude == -71.0928
        assert location.collection_date == "2023-06-15"
        assert location.textual_location == "Cambridge, Massachusetts, USA"
        assert location.sample_id == "Gb0123456"
        assert location.database_source == "GOLD"
        assert location.is_enrichable() == True

    def test_gold_coordinate_conversion(self):
        """Test coordinate conversion from different types."""
        # Test string coordinates
        sample1 = {"latitude": "42.3601", "longitude": "-71.0928"}
        location1 = self.adapter.extract_location(sample1)
        assert location1.latitude == 42.3601
        assert location1.longitude == -71.0928

        # Test numeric coordinates
        sample2 = {"latitude": 42.3601, "longitude": -71.0928}
        location2 = self.adapter.extract_location(sample2)
        assert location2.latitude == 42.3601
        assert location2.longitude == -71.0928

        # Test invalid coordinates
        sample3 = {"latitude": "invalid", "longitude": "-71.0928"}
        location3 = self.adapter.extract_location(sample3)
        assert location3.latitude is None
        assert location3.longitude is None

    def test_gold_date_formats(self):
        """Test different GOLD date format variations."""
        # Test ISO format
        sample1 = {"dateCollected": "2023-06-15T10:30:00Z"}
        location1 = self.adapter.extract_location(sample1)
        assert location1.collection_date == "2023-06-15"

        # Test date only
        sample2 = {"dateCollected": "2023-06-15"}
        location2 = self.adapter.extract_location(sample2)
        assert location2.collection_date == "2023-06-15"

    def test_gold_location_priority(self):
        """Test GOLD textual location field priority."""
        sample = {
            "geoLocation": "Primary Location",
            "geographicLocation": "Secondary Location",
            "description": "Description Location",
        }

        location = self.adapter.extract_location(sample)
        assert location.textual_location == "Primary Location"

    def test_batch_extraction(self):
        """Test batch extraction from multiple GOLD samples."""
        samples = [
            {"biosampleGoldId": "Gb01", "latitude": 42.3601, "longitude": -71.0928},
            {"biosampleGoldId": "Gb02", "latitude": 40.7128, "longitude": -74.0060},
            {"biosampleGoldId": "Gb03", "description": "Sample without coordinates"},
        ]

        locations = self.adapter.extract_locations_batch(samples)

        assert len(locations) == 3
        assert locations[0].is_enrichable() == True
        assert locations[1].is_enrichable() == True
        assert locations[2].is_enrichable() == False


class TestMongoFetchers:
    """Test MongoDB fetcher classes (without actual MongoDB connection)."""

    def test_nmdc_fetcher_initialization(self):
        """Test NMDC MongoDB fetcher initialization."""
        fetcher = MongoNMDCBiosampleFetcher(
            connection_string="mongodb://localhost:27017",
            database_name="test_nmdc",
            collection_name="test_biosamples",
        )

        assert fetcher.database_name == "test_nmdc"
        assert fetcher.collection_name == "test_biosamples"
        assert isinstance(fetcher.adapter, NMDCBiosampleAdapter)

    def test_gold_fetcher_initialization(self):
        """Test GOLD MongoDB fetcher initialization."""
        fetcher = MongoGOLDBiosampleFetcher(
            connection_string="mongodb://localhost:27017",
            database_name="test_gold",
            collection_name="test_biosamples",
        )

        assert fetcher.database_name == "test_gold"
        assert fetcher.collection_name == "test_biosamples"
        assert isinstance(fetcher.adapter, GOLDBiosampleAdapter)


class TestUnifiedInterface:
    """Test unified biosample fetcher interface."""

    def test_unified_fetcher_configuration(self):
        """Test unified fetcher configuration."""
        fetcher = UnifiedBiosampleFetcher()

        # Test NMDC configuration
        fetcher.configure_nmdc_mongo("mongodb://localhost:27017")
        assert fetcher.nmdc_mongo is not None
        assert isinstance(fetcher.nmdc_mongo, MongoNMDCBiosampleFetcher)

        # Test GOLD configuration
        fetcher.configure_gold_mongo("mongodb://localhost:27017")
        assert fetcher.gold_mongo is not None
        assert isinstance(fetcher.gold_mongo, MongoGOLDBiosampleFetcher)


# Integration test with sample data
def test_end_to_end_workflow():
    """Test complete workflow with sample data."""
    # Sample NMDC data
    nmdc_sample = {
        "id": "nmdc:bsm-12-test123",
        "lat_lon": "42.3601 -71.0928",
        "collection_date": "2023-06-15T10:30:00Z",
        "geo_loc_name": "Cambridge, Massachusetts, USA",
    }

    # Sample GOLD data
    gold_sample = {
        "biosampleGoldId": "Gb0123456",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "dateCollected": "2023-06-16",
        "geoLocation": "New York, New York, USA",
    }

    # Test NMDC extraction
    nmdc_adapter = NMDCBiosampleAdapter()
    nmdc_location = nmdc_adapter.extract_location(nmdc_sample)

    assert nmdc_location.is_enrichable() == True
    assert nmdc_location.database_source == "NMDC"
    assert nmdc_location.location_completeness == 1.0

    # Test GOLD extraction
    gold_adapter = GOLDBiosampleAdapter()
    gold_location = gold_adapter.extract_location(gold_sample)

    assert gold_location.is_enrichable() == True
    assert gold_location.database_source == "GOLD"
    assert gold_location.location_completeness == 1.0

    # Test serialization
    nmdc_dict = nmdc_location.to_dict()
    gold_dict = gold_location.to_dict()

    assert nmdc_dict["latitude"] == 42.3601
    assert gold_dict["latitude"] == 40.7128

    print("✅ End-to-end workflow test passed!")
    print(f"NMDC Location: {nmdc_location.textual_location}")
    print(f"GOLD Location: {gold_location.textual_location}")


if __name__ == "__main__":
    # Run the end-to-end test
    test_end_to_end_workflow()

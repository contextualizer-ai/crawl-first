"""
Biosample data adapters for extracting key enrichment inputs.

Provides unified interface for extracting latitude, longitude, collection_date,
and textual location names from NMDC and GOLD biosample data across different
storage formats (MongoDB, files).
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


@dataclass
class BiosampleLocation:
    """Standardized biosample location data for API enrichment."""

    # Required fields for API enrichment
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    collection_date: Optional[str] = None  # YYYY-MM-DD format

    # Location context
    textual_location: Optional[str] = None

    # Source metadata and IDs
    sample_id: Optional[str] = None  # Primary ID in native format
    database_source: Optional[str] = None  # "NMDC" or "GOLD"
    extraction_timestamp: Optional[str] = None
    
    # Normalized ID fields
    nmdc_biosample_id: Optional[str] = None  # Main NMDC ID
    gold_biosample_id: Optional[str] = None  # Main GOLD ID
    alternative_identifiers: Optional[List[str]] = None  # Alternative IDs
    external_database_identifiers: Optional[List[str]] = None  # External DB IDs
    biosample_identifiers: Optional[List[str]] = None  # Biosample-specific IDs
    sample_identifiers: Optional[List[str]] = None  # Sample IDs
    
    # Associated studies
    nmdc_studies: Optional[List[str]] = None  # NMDC associated_studies
    gold_studies: Optional[List[str]] = None  # GOLD seq_projects relationship

    # Quality indicators
    coordinate_precision: Optional[int] = None  # decimal places
    date_precision: Optional[str] = None  # "day", "month", "year"
    location_completeness: Optional[float] = None  # 0.0-1.0 score

    def __post_init__(self):
        """Set extraction timestamp and validate data."""
        if self.extraction_timestamp is None:
            self.extraction_timestamp = datetime.utcnow().isoformat() + "Z"

        # Calculate completeness score
        required_fields = [
            self.latitude,
            self.longitude,
            self.collection_date,
            self.textual_location,
        ]
        available_fields = sum(1 for field in required_fields if field is not None)
        self.location_completeness = available_fields / len(required_fields)

    def is_enrichable(self) -> bool:
        """Check if sample has minimum data for API enrichment."""
        return (
            self.latitude is not None
            and self.longitude is not None
            and -90 <= self.latitude <= 90
            and -180 <= self.longitude <= 180
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "collection_date": self.collection_date,
            "textual_location": self.textual_location,
            "sample_id": self.sample_id,
            "database_source": self.database_source,
            "extraction_timestamp": self.extraction_timestamp,
            "coordinate_precision": self.coordinate_precision,
            "date_precision": self.date_precision,
            "location_completeness": self.location_completeness,
            "is_enrichable": self.is_enrichable(),
            "nmdc_biosample_id": self.nmdc_biosample_id,
            "gold_biosample_id": self.gold_biosample_id,
            "alternative_identifiers": self.alternative_identifiers,
            "external_database_identifiers": self.external_database_identifiers,
            "biosample_identifiers": self.biosample_identifiers,
            "sample_identifiers": self.sample_identifiers,
            "nmdc_studies": self.nmdc_studies,
            "gold_studies": self.gold_studies,
        }


class BiosampleAdapter(ABC):
    """Abstract base class for biosample data adapters."""

    @abstractmethod
    def extract_location(self, biosample_data: Dict[str, Any]) -> BiosampleLocation:
        """Extract standardized location data from a single biosample."""
        pass

    @abstractmethod
    def extract_locations_batch(
        self, biosamples: List[Dict[str, Any]]
    ) -> List[BiosampleLocation]:
        """Extract location data from multiple biosamples."""
        pass


class NMDCBiosampleAdapter(BiosampleAdapter):
    """Adapter for NMDC Biosample data extraction."""

    def extract_location(self, biosample_data: Dict[str, Any]) -> BiosampleLocation:
        """Extract location data from NMDC biosample document."""

        # Extract coordinates from lat_lon field
        latitude, longitude = self._parse_nmdc_coordinates(biosample_data)

        # Extract collection date
        collection_date = self._parse_nmdc_date(biosample_data)

        # Extract textual location
        textual_location = self._parse_nmdc_location_text(biosample_data)

        # Get sample ID
        sample_id = biosample_data.get("id") or biosample_data.get("_id")

        # Extract normalized IDs
        nmdc_id, gold_id, id_collections = self._extract_nmdc_ids(biosample_data)

        # Extract associated studies
        nmdc_studies = self._extract_nmdc_studies(biosample_data)

        # Assess coordinate precision
        coord_precision = self._assess_coordinate_precision(latitude, longitude)

        # Assess date precision
        date_precision = self._assess_date_precision(collection_date)

        return BiosampleLocation(
            latitude=latitude,
            longitude=longitude,
            collection_date=collection_date,
            textual_location=textual_location,
            sample_id=str(sample_id) if sample_id else None,
            database_source="NMDC",
            coordinate_precision=coord_precision,
            date_precision=date_precision,
            nmdc_biosample_id=str(nmdc_id) if nmdc_id else None,
            gold_biosample_id=gold_id,
            alternative_identifiers=id_collections["alternative_identifiers"],
            external_database_identifiers=id_collections["external_database_identifiers"],
            biosample_identifiers=id_collections["biosample_identifiers"],
            sample_identifiers=id_collections["sample_identifiers"],
            nmdc_studies=nmdc_studies,
        )

    def extract_locations_batch(
        self, biosamples: List[Dict[str, Any]]
    ) -> List[BiosampleLocation]:
        """Extract location data from multiple NMDC biosamples."""
        return [self.extract_location(biosample) for biosample in biosamples]

    def _parse_nmdc_coordinates(
        self, data: Dict[str, Any]
    ) -> tuple[Optional[float], Optional[float]]:
        """Parse NMDC lat_lon field or separate lat/lon fields."""
        # Try lat_lon combined field first
        lat_lon = data.get("lat_lon")
        if lat_lon:
            # Handle different lat_lon formats
            if isinstance(lat_lon, str):
                # Format: "42.3601 -71.0928" or "42.3601,-71.0928"
                coords = re.split(r"[,\s]+", lat_lon.strip())
                if len(coords) >= 2:
                    try:
                        return float(coords[0]), float(coords[1])
                    except ValueError:
                        pass
            elif isinstance(lat_lon, dict):
                # Format: {"latitude": 42.3601, "longitude": -71.0928}
                lat = lat_lon.get("latitude")
                lon = lat_lon.get("longitude")
                if lat is not None and lon is not None:
                    return float(lat), float(lon)
            elif isinstance(lat_lon, list) and len(lat_lon) >= 2:
                # Format: [42.3601, -71.0928]
                try:
                    return float(lat_lon[0]), float(lat_lon[1])
                except (ValueError, IndexError):
                    pass

        # Try separate latitude/longitude fields
        lat = data.get("latitude")
        lon = data.get("longitude")
        if lat is not None and lon is not None:
            try:
                return float(lat), float(lon)
            except ValueError:
                pass

        return None, None

    def _parse_nmdc_date(self, data: Dict[str, Any]) -> Optional[str]:
        """Parse NMDC collection_date field."""
        collection_date = data.get("collection_date")
        if not collection_date:
            return None

        # Handle different date formats
        if isinstance(collection_date, str):
            # Try to parse and normalize to YYYY-MM-DD
            try:
                # Handle ISO format with time
                if "T" in collection_date:
                    dt = datetime.fromisoformat(collection_date.replace("Z", "+00:00"))
                    return dt.strftime("%Y-%m-%d")
                # Handle YYYY-MM-DD format
                elif len(collection_date) >= 10:
                    return collection_date[:10]
                # Handle YYYY-MM format
                elif len(collection_date) >= 7:
                    return collection_date + "-01"
                # Handle YYYY format
                elif len(collection_date) == 4:
                    return collection_date + "-01-01"
            except ValueError:
                pass
        elif isinstance(collection_date, dict):
            # Handle NMDC structured date format like {"has_raw_value": "2014-11-25", "type": "nmdc:TimestampValue"}
            raw_value = collection_date.get("has_raw_value")
            if raw_value and isinstance(raw_value, str):
                try:
                    # Handle ISO format with time
                    if "T" in raw_value:
                        dt = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
                        return dt.strftime("%Y-%m-%d")
                    # Handle YYYY-MM-DD format
                    elif len(raw_value) >= 10:
                        return raw_value[:10]
                    # Handle YYYY-MM format
                    elif len(raw_value) >= 7:
                        return raw_value + "-01"
                    # Handle YYYY format
                    elif len(raw_value) == 4:
                        return raw_value + "-01-01"
                except ValueError:
                    pass

        return None

    def _parse_nmdc_location_text(self, data: Dict[str, Any]) -> Optional[str]:
        """Parse NMDC textual location fields."""
        # Priority order for location text
        location_fields = [
            "geo_loc_name",
            "geographic_location", 
            "location",
            "sample_collection_site",
            "description",
        ]

        for field in location_fields:
            value = data.get(field)
            if value:
                if isinstance(value, str) and value.strip():
                    return value.strip()
                elif isinstance(value, dict):
                    # Handle NMDC structured format like {"has_raw_value": "USA: Columbia River, Washington", "type": "nmdc:TextValue"}
                    raw_value = value.get("has_raw_value")
                    if raw_value and isinstance(raw_value, str) and raw_value.strip():
                        return raw_value.strip()

        return None

    def _extract_nmdc_ids(self, data: Dict[str, Any]) -> tuple[Optional[str], Optional[str], Dict[str, Optional[List[str]]]]:
        """Extract and normalize ID fields from NMDC biosample."""
        nmdc_id = data.get("id")
        gold_id = None
        
        # Look for GOLD ID in various possible fields
        gold_fields = ["gold_biosample_id", "biosampleGoldId", "gold_id", "gold_biosample_identifiers"]
        for field in gold_fields:
            value = data.get(field)
            if value:
                if isinstance(value, list) and value:
                    # Extract from list like ["gold:Gb0115231"] - take first item and remove prefix
                    gold_val = str(value[0])
                    if gold_val.startswith("gold:"):
                        gold_id = gold_val[5:]  # Remove "gold:" prefix
                    else:
                        gold_id = gold_val
                    break
                elif isinstance(value, str):
                    if value.startswith("gold:"):
                        gold_id = value[5:]  # Remove "gold:" prefix
                    else:
                        gold_id = value
                    break
                else:
                    gold_id = str(value)
                    break
        
        # Extract separate ID lists by type
        id_collections = {
            "alternative_identifiers": [],
            "external_database_identifiers": [],
            "biosample_identifiers": [],
            "sample_identifiers": []
        }
        
        # Add MongoDB _id to alternative_identifiers if different from main ID
        mongodb_id = data.get("_id")
        if mongodb_id and str(mongodb_id) != str(nmdc_id):
            id_collections["alternative_identifiers"].append(str(mongodb_id))
        
        # Process each ID field type separately
        for field in id_collections.keys():
            value = data.get(field)
            if value:
                if isinstance(value, list):
                    id_collections[field].extend([str(v) for v in value if v])
                elif isinstance(value, str) and value.strip():
                    id_collections[field].append(value.strip())
                elif value:  # Other types (int, etc.)
                    id_collections[field].append(str(value))
        
        # Also categorize some known NMDC fields into appropriate ID lists
        known_id_fields = {
            "insdc_biosample_identifiers": "external_database_identifiers",
            "ncbi_biosample_identifiers": "external_database_identifiers",
            "jgi_portal_identifiers": "external_database_identifiers",
            "samp_name": "sample_identifiers",
            "name": "sample_identifiers"
        }
        
        for source_field, target_category in known_id_fields.items():
            value = data.get(source_field)
            if value:
                if isinstance(value, list):
                    id_collections[target_category].extend([str(v) for v in value if v])
                elif isinstance(value, str) and value.strip():
                    id_collections[target_category].append(value.strip())
                elif value:
                    id_collections[target_category].append(str(value))
        
        # Remove duplicates and filter out main IDs
        for field in id_collections:
            id_collections[field] = list(set(id_collections[field]))
            # Remove main IDs from secondary lists
            if nmdc_id and str(nmdc_id) in id_collections[field]:
                id_collections[field].remove(str(nmdc_id))
            if gold_id and gold_id in id_collections[field]:
                id_collections[field].remove(gold_id)
            # Convert empty lists to None
            if not id_collections[field]:
                id_collections[field] = None
            
        return nmdc_id, gold_id, id_collections

    def _extract_nmdc_studies(self, data: Dict[str, Any]) -> Optional[List[str]]:
        """Extract associated studies from NMDC biosample."""
        studies = data.get("associated_studies") or data.get("part_of")
        
        if not studies:
            return None
            
        if isinstance(studies, list):
            return [str(study) for study in studies if study]
        elif isinstance(studies, str):
            return [studies]
        else:
            return [str(studies)]

    def _assess_coordinate_precision(
        self, lat: Optional[float], lon: Optional[float]
    ) -> Optional[int]:
        """Assess coordinate precision based on decimal places."""
        if lat is None or lon is None:
            return None

        lat_str = str(lat)
        lon_str = str(lon)

        lat_decimals = len(lat_str.split(".")[-1]) if "." in lat_str else 0
        lon_decimals = len(lon_str.split(".")[-1]) if "." in lon_str else 0

        return min(lat_decimals, lon_decimals)

    def _assess_date_precision(self, date_str: Optional[str]) -> Optional[str]:
        """Assess date precision level."""
        if not date_str:
            return None

        if len(date_str) >= 10:  # YYYY-MM-DD
            return "day"
        elif len(date_str) >= 7:  # YYYY-MM
            return "month"
        elif len(date_str) >= 4:  # YYYY
            return "year"

        return None


class GOLDBiosampleAdapter(BiosampleAdapter):
    """Adapter for GOLD Biosample data extraction."""

    def extract_location(self, biosample_data: Dict[str, Any], database=None) -> BiosampleLocation:
        """Extract location data from GOLD biosample document."""

        # Extract coordinates
        latitude = biosample_data.get("latitude")
        longitude = biosample_data.get("longitude")

        # Convert to float if needed
        try:
            latitude = float(latitude) if latitude is not None else None
            longitude = float(longitude) if longitude is not None else None
        except (ValueError, TypeError):
            latitude, longitude = None, None

        # Extract collection date
        collection_date = self._parse_gold_date(biosample_data)

        # Extract textual location
        textual_location = self._parse_gold_location_text(biosample_data)

        # Get sample ID
        sample_id = (
            biosample_data.get("biosampleGoldId")
            or biosample_data.get("_id")
            or biosample_data.get("id")
        )

        # Extract normalized IDs
        gold_id, nmdc_id, id_collections = self._extract_gold_ids(biosample_data)

        # Extract associated studies
        gold_studies = self._extract_gold_studies(biosample_data, database)

        # Assess coordinate precision
        coord_precision = self._assess_coordinate_precision(latitude, longitude)

        # Assess date precision
        date_precision = self._assess_date_precision(collection_date)

        return BiosampleLocation(
            latitude=latitude,
            longitude=longitude,
            collection_date=collection_date,
            textual_location=textual_location,
            sample_id=str(sample_id) if sample_id else None,
            database_source="GOLD",
            coordinate_precision=coord_precision,
            date_precision=date_precision,
            nmdc_biosample_id=nmdc_id,
            gold_biosample_id=gold_id,
            alternative_identifiers=id_collections["alternative_identifiers"],
            external_database_identifiers=id_collections["external_database_identifiers"],
            biosample_identifiers=id_collections["biosample_identifiers"],
            sample_identifiers=id_collections["sample_identifiers"],
            gold_studies=gold_studies,
        )

    def extract_locations_batch(
        self, biosamples: List[Dict[str, Any]]
    ) -> List[BiosampleLocation]:
        """Extract location data from multiple GOLD biosamples."""
        return [self.extract_location(biosample) for biosample in biosamples]

    def _parse_gold_date(self, data: Dict[str, Any]) -> Optional[str]:
        """Parse GOLD dateCollected field."""
        date_collected = data.get("dateCollected")
        if not date_collected:
            return None

        # Handle different date formats
        if isinstance(date_collected, str):
            try:
                # Handle ISO format with time
                if "T" in date_collected:
                    dt = datetime.fromisoformat(date_collected.replace("Z", "+00:00"))
                    return dt.strftime("%Y-%m-%d")
                # Handle YYYY-MM-DD format
                elif len(date_collected) >= 10:
                    return date_collected[:10]
                # Handle YYYY-MM format
                elif len(date_collected) >= 7:
                    return date_collected + "-01"
                # Handle YYYY format
                elif len(date_collected) == 4:
                    return date_collected + "-01-01"
            except ValueError:
                pass

        return None

    def _parse_gold_location_text(self, data: Dict[str, Any]) -> Optional[str]:
        """Parse GOLD textual location fields."""
        # Priority order for location text
        location_fields = [
            "geoLocation",
            "geographicLocation",
            "sampleCollectionSite",
            "description",
            "habitat",
        ]

        for field in location_fields:
            value = data.get(field)
            if value and isinstance(value, str) and value.strip():
                return value.strip()

        return None

    def _assess_coordinate_precision(
        self, lat: Optional[float], lon: Optional[float]
    ) -> Optional[int]:
        """Assess coordinate precision based on decimal places."""
        if lat is None or lon is None:
            return None

        lat_str = str(lat)
        lon_str = str(lon)

        lat_decimals = len(lat_str.split(".")[-1]) if "." in lat_str else 0
        lon_decimals = len(lon_str.split(".")[-1]) if "." in lon_str else 0

        return min(lat_decimals, lon_decimals)

    def _assess_date_precision(self, date_str: Optional[str]) -> Optional[str]:
        """Assess date precision level."""
        if not date_str:
            return None

        if len(date_str) >= 10:  # YYYY-MM-DD
            return "day"
        elif len(date_str) >= 7:  # YYYY-MM
            return "month"
        elif len(date_str) >= 4:  # YYYY
            return "year"

        return None

    def _extract_gold_ids(self, data: Dict[str, Any]) -> tuple[Optional[str], Optional[str], Dict[str, Optional[List[str]]]]:
        """Extract and normalize ID fields from GOLD biosample."""
        gold_id = data.get("biosampleGoldId") or data.get("_id") or data.get("id")
        nmdc_id = None
        
        # Look for NMDC ID in various possible fields
        nmdc_fields = ["nmdc_biosample_id", "biosampleNmdcId", "nmdc_id"]
        for field in nmdc_fields:
            value = data.get(field)
            if value:
                nmdc_id = str(value)
                break
        
        # Extract separate ID lists by type
        id_collections = {
            "alternative_identifiers": [],
            "external_database_identifiers": [],
            "biosample_identifiers": [],
            "sample_identifiers": []
        }
        
        # Add MongoDB _id to alternative_identifiers if different from main ID
        mongodb_id = data.get("_id")
        if mongodb_id and str(mongodb_id) != str(gold_id):
            id_collections["alternative_identifiers"].append(str(mongodb_id))
        
        # Add projectGoldId to alternative_identifiers if different from biosampleGoldId
        project_gold_id = data.get("projectGoldId")
        if project_gold_id and str(project_gold_id) != str(gold_id):
            id_collections["alternative_identifiers"].append(str(project_gold_id))
        
        # Process each ID field type separately
        for field in id_collections.keys():
            value = data.get(field)
            if value:
                if isinstance(value, list):
                    id_collections[field].extend([str(v) for v in value if v])
                elif isinstance(value, str) and value.strip():
                    id_collections[field].append(value.strip())
                elif value:  # Other types (int, etc.)
                    id_collections[field].append(str(value))
        
        # Remove duplicates and filter out main IDs
        for field in id_collections:
            id_collections[field] = list(set(id_collections[field]))
            # Remove main IDs from secondary lists
            if gold_id and str(gold_id) in id_collections[field]:
                id_collections[field].remove(str(gold_id))
            if nmdc_id and nmdc_id in id_collections[field]:
                id_collections[field].remove(nmdc_id)
            # Convert empty lists to None
            if not id_collections[field]:
                id_collections[field] = None
            
        return str(gold_id) if gold_id else None, nmdc_id, id_collections

    def _extract_gold_studies(self, data: Dict[str, Any], database=None) -> Optional[List[str]]:
        """Extract associated studies from GOLD biosample by looking up seq_projects collection."""
        biosample_gold_id = data.get("biosampleGoldId")
        if not biosample_gold_id or not database:
            return None
        
        try:
            # Query seq_projects collection for matching biosampleGoldId
            seq_projects_collection = database["seq_projects"]
            cursor = seq_projects_collection.find({"biosampleGoldId": biosample_gold_id})
            
            study_gold_ids = []
            for project in cursor:
                study_gold_id = project.get("studyGoldId")
                if study_gold_id:
                    study_gold_ids.append(str(study_gold_id))
            
            # Remove duplicates and return
            unique_studies = list(set(study_gold_ids))
            return unique_studies if unique_studies else None
            
        except Exception:
            # If lookup fails, return None
            return None


# MongoDB Adapters (Main Implementation)


class MongoNMDCBiosampleFetcher:
    """MongoDB fetcher for NMDC biosample data."""

    def __init__(
        self,
        connection_string: str = None,
        database_name: str = "nmdc",
        collection_name: str = "biosamples",
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.adapter = NMDCBiosampleAdapter()
        self._client = None
        self._collection = None

    def connect(self):
        """Establish MongoDB connection."""
        try:
            import pymongo

            self._client = pymongo.MongoClient(self.connection_string)
            self._collection = self._client[self.database_name][self.collection_name]
            return True
        except ImportError:
            raise ImportError(
                "pymongo is required for MongoDB operations. Install with: pip install pymongo"
            )
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            return False

    def disconnect(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._collection = None

    def fetch_locations(
        self, query: Dict[str, Any] = None, limit: int = None
    ) -> Iterator[BiosampleLocation]:
        """Fetch and extract locations from NMDC biosamples."""
        if not self._collection:
            if not self.connect():
                raise RuntimeError("Failed to connect to MongoDB")

        cursor = self._collection.find(query or {})
        if limit:
            cursor = cursor.limit(limit)

        for document in cursor:
            yield self.adapter.extract_location(document)

    def fetch_enrichable_locations(
        self, limit: int = None
    ) -> Iterator[BiosampleLocation]:
        """Fetch only biosamples with coordinates suitable for enrichment."""
        # Query for documents with lat_lon or separate lat/lon fields
        query = {
            "$or": [
                {"lat_lon": {"$exists": True, "$ne": None}},
                {
                    "latitude": {"$exists": True, "$ne": None},
                    "longitude": {"$exists": True, "$ne": None},
                },
            ]
        }

        for location in self.fetch_locations(query, limit):
            if location.is_enrichable():
                yield location

    def count_total_samples(self) -> int:
        """Count total biosamples in collection."""
        if not self._collection:
            if not self.connect():
                return 0
        return self._collection.count_documents({})

    def count_enrichable_samples(self) -> int:
        """Count biosamples with coordinates."""
        if not self._collection:
            if not self.connect():
                return 0

        query = {
            "$or": [
                {"lat_lon": {"$exists": True, "$ne": None}},
                {
                    "latitude": {"$exists": True, "$ne": None},
                    "longitude": {"$exists": True, "$ne": None},
                },
            ]
        }
        return self._collection.count_documents(query)

    def fetch_locations_by_ids(
        self, ids: List[str], id_field: str = "id"
    ) -> Iterator[BiosampleLocation]:
        """Fetch biosamples by specific IDs using native field names."""
        if not self._collection:
            if not self.connect():
                raise RuntimeError("Failed to connect to MongoDB")

        # Query for documents with IDs in the specified field
        query = {id_field: {"$in": ids}}
        cursor = self._collection.find(query)

        for document in cursor:
            yield self.adapter.extract_location(document)

    def fetch_random_locations(self, n: int = 10) -> Iterator[BiosampleLocation]:
        """Fetch N random biosamples from collection."""
        if not self._collection:
            if not self.connect():
                raise RuntimeError("Failed to connect to MongoDB")

        # Use MongoDB aggregation pipeline for efficient random sampling
        pipeline = [{"$sample": {"size": n}}]
        cursor = self._collection.aggregate(pipeline)

        for document in cursor:
            yield self.adapter.extract_location(document)

    def fetch_random_enrichable_locations(self, n: int = 10) -> Iterator[BiosampleLocation]:
        """Fetch N random enrichable biosamples from collection."""
        if not self._collection:
            if not self.connect():
                raise RuntimeError("Failed to connect to MongoDB")

        # Query for enrichable samples first, then random sample
        match_query = {
            "$or": [
                {"lat_lon": {"$exists": True, "$ne": None}},
                {
                    "latitude": {"$exists": True, "$ne": None},
                    "longitude": {"$exists": True, "$ne": None},
                },
            ]
        }
        
        pipeline = [
            {"$match": match_query},
            {"$sample": {"size": n}}
        ]
        cursor = self._collection.aggregate(pipeline)

        for document in cursor:
            location = self.adapter.extract_location(document)
            if location.is_enrichable():
                yield location


class MongoGOLDBiosampleFetcher:
    """MongoDB fetcher for GOLD biosample data."""

    def __init__(
        self,
        connection_string: str = None,
        database_name: str = "gold",
        collection_name: str = "biosamples",
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.adapter = GOLDBiosampleAdapter()
        self._client = None
        self._collection = None

    def connect(self):
        """Establish MongoDB connection."""
        try:
            import pymongo

            self._client = pymongo.MongoClient(self.connection_string)
            self._collection = self._client[self.database_name][self.collection_name]
            return True
        except ImportError:
            raise ImportError(
                "pymongo is required for MongoDB operations. Install with: pip install pymongo"
            )
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            return False

    def disconnect(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._collection = None

    def fetch_locations(
        self, query: Dict[str, Any] = None, limit: int = None
    ) -> Iterator[BiosampleLocation]:
        """Fetch and extract locations from GOLD biosamples."""
        if not self._collection:
            if not self.connect():
                raise RuntimeError("Failed to connect to MongoDB")

        cursor = self._collection.find(query or {})
        if limit:
            cursor = cursor.limit(limit)

        for document in cursor:
            yield self.adapter.extract_location(document, self._client[self.database_name])

    def fetch_enrichable_locations(
        self, limit: int = None
    ) -> Iterator[BiosampleLocation]:
        """Fetch only biosamples with coordinates suitable for enrichment."""
        # Query for documents with latitude and longitude
        query = {
            "latitude": {"$exists": True, "$ne": None},
            "longitude": {"$exists": True, "$ne": None},
        }

        for location in self.fetch_locations(query, limit):
            if location.is_enrichable():
                yield location

    def count_total_samples(self) -> int:
        """Count total biosamples in collection."""
        if not self._collection:
            if not self.connect():
                return 0
        return self._collection.count_documents({})

    def count_enrichable_samples(self) -> int:
        """Count biosamples with coordinates."""
        if not self._collection:
            if not self.connect():
                return 0

        query = {
            "latitude": {"$exists": True, "$ne": None},
            "longitude": {"$exists": True, "$ne": None},
        }
        return self._collection.count_documents(query)

    def fetch_locations_by_ids(
        self, ids: List[str], id_field: str = "biosampleGoldId"
    ) -> Iterator[BiosampleLocation]:
        """Fetch biosamples by specific IDs using native field names."""
        if not self._collection:
            if not self.connect():
                raise RuntimeError("Failed to connect to MongoDB")

        # Query for documents with IDs in the specified field
        query = {id_field: {"$in": ids}}
        cursor = self._collection.find(query)

        for document in cursor:
            yield self.adapter.extract_location(document, self._client[self.database_name])

    def fetch_random_locations(self, n: int = 10) -> Iterator[BiosampleLocation]:
        """Fetch N random biosamples from collection."""
        if not self._collection:
            if not self.connect():
                raise RuntimeError("Failed to connect to MongoDB")

        # Use MongoDB aggregation pipeline for efficient random sampling
        pipeline = [{"$sample": {"size": n}}]
        cursor = self._collection.aggregate(pipeline)

        for document in cursor:
            yield self.adapter.extract_location(document, self._client[self.database_name])

    def fetch_random_enrichable_locations(self, n: int = 10) -> Iterator[BiosampleLocation]:
        """Fetch N random enrichable biosamples from collection."""
        if not self._collection:
            if not self.connect():
                raise RuntimeError("Failed to connect to MongoDB")

        # Query for enrichable samples first, then random sample
        match_query = {
            "latitude": {"$exists": True, "$ne": None},
            "longitude": {"$exists": True, "$ne": None},
        }
        
        pipeline = [
            {"$match": match_query},
            {"$sample": {"size": n}}
        ]
        cursor = self._collection.aggregate(pipeline)

        for document in cursor:
            location = self.adapter.extract_location(document, self._client[self.database_name])
            if location.is_enrichable():
                yield location


# File-based Adapter Stubs (Future Implementation)


class FileBiosampleFetcher:
    """File-based fetcher for biosample data (stub implementation)."""

    def __init__(self, file_path: Union[str, Path], format_type: str = "auto"):
        self.file_path = Path(file_path)
        self.format_type = format_type
        self.nmdc_adapter = NMDCBiosampleAdapter()
        self.gold_adapter = GOLDBiosampleAdapter()

    def detect_format(self) -> str:
        """Detect file format and biosample type."""
        # Stub: Would implement format detection logic
        # - JSON/JSONL detection
        # - NMDC vs GOLD schema detection
        # - Mixed format handling
        return "nmdc_json"  # Placeholder

    def fetch_locations(self, limit: int = None) -> Iterator[BiosampleLocation]:
        """Fetch locations from file (stub implementation)."""
        # Stub: Would implement file parsing logic
        # - Handle JSON, JSONL, TSV formats
        # - Auto-detect NMDC vs GOLD formats
        # - Handle mixed biosample types in same file
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Placeholder implementation
        with open(self.file_path) as f:
            try:
                data = json.load(f)
                # Stub: Would implement proper format detection and parsing
                if isinstance(data, list):
                    for item in data[:limit] if limit else data:
                        yield self.nmdc_adapter.extract_location(item)
                else:
                    yield self.nmdc_adapter.extract_location(data)
            except json.JSONDecodeError:
                # Stub: Would handle other formats (JSONL, TSV, etc.)
                pass


# Unified Interface


class UnifiedBiosampleFetcher:
    """Unified interface for fetching biosample locations from various sources."""

    def __init__(self):
        self.nmdc_mongo = None
        self.gold_mongo = None

    def configure_nmdc_mongo(
        self,
        connection_string: str,
        database: str = "nmdc",
        collection: str = "biosamples",
    ):
        """Configure NMDC MongoDB connection."""
        self.nmdc_mongo = MongoNMDCBiosampleFetcher(
            connection_string, database, collection
        )

    def configure_gold_mongo(
        self,
        connection_string: str,
        database: str = "gold",
        collection: str = "biosamples",
    ):
        """Configure GOLD MongoDB connection."""
        self.gold_mongo = MongoGOLDBiosampleFetcher(
            connection_string, database, collection
        )

    def fetch_enrichable_locations(
        self, source: str = "all", limit: int = None
    ) -> Iterator[BiosampleLocation]:
        """Fetch enrichable locations from configured sources."""
        count = 0

        if source in ("all", "nmdc") and self.nmdc_mongo:
            for location in self.nmdc_mongo.fetch_enrichable_locations(limit):
                if limit and count >= limit:
                    break
                yield location
                count += 1

        if source in ("all", "gold") and self.gold_mongo:
            remaining_limit = (limit - count) if limit else None
            for location in self.gold_mongo.fetch_enrichable_locations(remaining_limit):
                if limit and count >= limit:
                    break
                yield location
                count += 1

    def get_enrichment_statistics(self) -> Dict[str, Any]:
        """Get statistics about available enrichable samples."""
        stats = {}

        if self.nmdc_mongo:
            stats["nmdc"] = {
                "total_samples": self.nmdc_mongo.count_total_samples(),
                "enrichable_samples": self.nmdc_mongo.count_enrichable_samples(),
            }

        if self.gold_mongo:
            stats["gold"] = {
                "total_samples": self.gold_mongo.count_total_samples(),
                "enrichable_samples": self.gold_mongo.count_enrichable_samples(),
            }

        # Calculate totals
        total_samples = sum(db.get("total_samples", 0) for db in stats.values())
        total_enrichable = sum(db.get("enrichable_samples", 0) for db in stats.values())

        stats["summary"] = {
            "total_samples": total_samples,
            "total_enrichable_samples": total_enrichable,
            "enrichment_coverage": (
                total_enrichable / total_samples if total_samples > 0 else 0.0
            ),
        }

        return stats

    def fetch_locations_by_ids(
        self, ids: List[str], source: str = "all", id_field: str = None
    ) -> Iterator[BiosampleLocation]:
        """Fetch locations by IDs from configured sources."""
        if source in ("all", "nmdc") and self.nmdc_mongo:
            field = id_field or "id"
            for location in self.nmdc_mongo.fetch_locations_by_ids(ids, field):
                yield location

        if source in ("all", "gold") and self.gold_mongo:
            field = id_field or "biosampleGoldId"
            for location in self.gold_mongo.fetch_locations_by_ids(ids, field):
                yield location

    def fetch_random_locations(
        self, n: int = 10, source: str = "all"
    ) -> Iterator[BiosampleLocation]:
        """Fetch N random locations from configured sources."""
        count = 0
        
        if source in ("all", "nmdc") and self.nmdc_mongo:
            nmdc_n = n if source == "nmdc" else n // 2
            for location in self.nmdc_mongo.fetch_random_locations(nmdc_n):
                if count >= n:
                    break
                yield location
                count += 1

        if source in ("all", "gold") and self.gold_mongo:
            remaining_n = n - count if source == "all" else n
            for location in self.gold_mongo.fetch_random_locations(remaining_n):
                if count >= n:
                    break
                yield location
                count += 1

    def fetch_random_enrichable_locations(
        self, n: int = 10, source: str = "all"
    ) -> Iterator[BiosampleLocation]:
        """Fetch N random enrichable locations from configured sources."""
        count = 0
        
        if source in ("all", "nmdc") and self.nmdc_mongo:
            nmdc_n = n if source == "nmdc" else n // 2
            for location in self.nmdc_mongo.fetch_random_enrichable_locations(nmdc_n):
                if count >= n:
                    break
                yield location
                count += 1

        if source in ("all", "gold") and self.gold_mongo:
            remaining_n = n - count if source == "all" else n
            for location in self.gold_mongo.fetch_random_enrichable_locations(remaining_n):
                if count >= n:
                    break
                yield location
                count += 1

#!/usr/bin/env python3
"""
Extract real NMDC and GOLD biosamples from MongoDB for testing.

Uses the same MongoDB connection as the schema analysis targets.
"""

import json
import sys
from typing import Dict, List, Any

try:
    import pymongo
except ImportError:
    print("Error: pymongo not installed. Install with: uv add pymongo")
    sys.exit(1)


def connect_to_mongodb():
    """Connect to MongoDB using the same settings as Makefile."""
    # From Makefile: mongodb://ncbi_reader:register_manatee_coach78@localhost:27778/?directConnection=true&authMechanism=DEFAULT&authSource=admin
    mongo_uri = "mongodb://ncbi_reader:register_manatee_coach78@localhost:27778/?directConnection=true&authMechanism=DEFAULT&authSource=admin"
    
    try:
        client = pymongo.MongoClient(mongo_uri)
        # Test connection
        client.admin.command('ping')
        return client
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        print("Make sure MongoDB is running and accessible")
        return None


def extract_nmdc_biosamples(client, limit=3):
    """Extract real NMDC biosamples with coordinates."""
    print("🧬 Extracting NMDC biosamples...")
    
    db = client["nmdc"]
    collection = db["biosample_set"]
    
    # Query for biosamples with coordinates and good metadata
    query = {
        "$or": [
            {"lat_lon": {"$exists": True, "$ne": None}},
            {
                "latitude": {"$exists": True, "$ne": None},
                "longitude": {"$exists": True, "$ne": None}
            }
        ],
        "collection_date": {"$exists": True, "$ne": None},
        "geo_loc_name": {"$exists": True, "$ne": None}
    }
    
    # Get biosamples with good data
    cursor = collection.find(query).limit(limit * 3)  # Get extra in case some are bad
    
    biosamples = []
    for doc in cursor:
        # Clean up the document for JSON serialization
        doc.pop("_id", None)  # Remove ObjectId which isn't JSON serializable
        biosamples.append(doc)
        
        if len(biosamples) >= limit:
            break
    
    print(f"Found {len(biosamples)} NMDC biosamples")
    return biosamples


def extract_gold_biosamples(client, limit=3):
    """Extract real GOLD biosamples with coordinates."""
    print("🏅 Extracting GOLD biosamples...")
    
    db = client["gold_metadata"] 
    collection = db["biosamples"]
    
    # Query for biosamples with coordinates and good metadata
    query = {
        "latitude": {"$exists": True, "$ne": None, "$ne": ""},
        "longitude": {"$exists": True, "$ne": None, "$ne": ""},
        "dateCollected": {"$exists": True, "$ne": None},
        "geoLocation": {"$exists": True, "$ne": None, "$ne": ""}
    }
    
    # Get biosamples with good data
    cursor = collection.find(query).limit(limit * 3)  # Get extra in case some are bad
    
    biosamples = []
    for doc in cursor:
        # Clean up the document for JSON serialization
        doc.pop("_id", None)  # Remove ObjectId which isn't JSON serializable
        biosamples.append(doc)
        
        if len(biosamples) >= limit:
            break
    
    print(f"Found {len(biosamples)} GOLD biosamples")
    return biosamples


def extract_sample_seq_projects(client, gold_biosamples, limit=5):
    """Extract a few real seq_projects for the GOLD biosamples."""
    print("📚 Extracting sample seq_projects...")
    
    db = client["gold_metadata"]
    collection = db["seq_projects"]
    
    # Get biosampleGoldIds from our samples
    biosample_ids = [bs.get("biosampleGoldId") for bs in gold_biosamples if bs.get("biosampleGoldId")]
    
    if not biosample_ids:
        print("No biosampleGoldIds found in GOLD samples")
        return []
    
    # Query for seq_projects that match our biosamples
    query = {"biosampleGoldId": {"$in": biosample_ids}}
    cursor = collection.find(query).limit(limit)
    
    seq_projects = []
    for doc in cursor:
        # Clean up the document
        doc.pop("_id", None)
        # Keep only relevant fields
        cleaned_doc = {
            "biosampleGoldId": doc.get("biosampleGoldId"),
            "studyGoldId": doc.get("studyGoldId"),
            "projectGoldId": doc.get("projectGoldId"),
            "studyName": doc.get("studyName", ""),
            "projectName": doc.get("projectName", "")
        }
        seq_projects.append(cleaned_doc)
    
    print(f"Found {len(seq_projects)} seq_projects")
    return seq_projects


def main():
    """Extract real biosamples and save to test file."""
    print("Extracting Real NMDC and GOLD Biosamples from MongoDB")
    print("=" * 60)
    
    # Connect to MongoDB
    client = connect_to_mongodb()
    if not client:
        sys.exit(1)
    
    try:
        # Extract real biosamples
        nmdc_samples = extract_nmdc_biosamples(client, limit=3)
        gold_samples = extract_gold_biosamples(client, limit=3)
        seq_projects = extract_sample_seq_projects(client, gold_samples, limit=10)
        
        # Create the test data structure
        test_data = {
            "nmdc_samples": nmdc_samples,
            "gold_samples": gold_samples,
            "sample_seq_projects": seq_projects
        }
        
        # Output JSON
        print(json.dumps(test_data, indent=2, default=str))
        
        print(f"\n✅ Extraction complete!", file=sys.stderr)
        print(f"📊 Summary:", file=sys.stderr)
        print(f"  NMDC biosamples: {len(nmdc_samples)}", file=sys.stderr)
        print(f"  GOLD biosamples: {len(gold_samples)}", file=sys.stderr)
        print(f"  seq_projects: {len(seq_projects)}", file=sys.stderr)
        
    finally:
        client.close()


if __name__ == "__main__":
    main()
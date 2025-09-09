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


def extract_nmdc_biosamples(client, limit=100):
    """Extract real NMDC biosamples with coordinates and study associations."""
    print("🧬 Extracting NMDC biosamples...")
    
    db = client["nmdc"]
    biosample_collection = db["biosample_set"]
    study_collection = db["study_set"]
    
    # Query for intact biosamples (no filtering requirements)
    query = {}
    
    # Get biosamples with good data
    cursor = biosample_collection.find(query).limit(limit * 2)  # Get extra in case some are bad
    
    biosamples = []
    study_cache = {}
    
    for doc in cursor:
        # Clean up the document for JSON serialization
        doc.pop("_id", None)  # Remove ObjectId which isn't JSON serializable
        
        # Try to find associated study
        study_info = None
        if "associated_studies" in doc and doc["associated_studies"]:
            study_id = doc["associated_studies"][0]  # Take first study
            
            # Check cache first
            if study_id in study_cache:
                study_info = study_cache[study_id]
            else:
                # Query study collection
                study_doc = study_collection.find_one({"id": study_id})
                if study_doc:
                    study_doc.pop("_id", None)
                    study_info = study_doc
                    study_cache[study_id] = study_info
        
        # Add study info to biosample
        if study_info:
            doc["study_info"] = study_info
        
        biosamples.append(doc)
        
        if len(biosamples) >= limit:
            break
    
    print(f"Found {len(biosamples)} NMDC biosamples")
    print(f"With study associations: {len([b for b in biosamples if 'study_info' in b])}")
    return biosamples


def extract_gold_biosamples(client, limit=100):
    """Extract real GOLD biosamples with coordinates and study associations via seq_projects."""
    print("🏅 Extracting GOLD biosamples...")
    
    db = client["gold_metadata"] 
    biosample_collection = db["biosamples"]
    seq_projects_collection = db["seq_projects"]
    
    # Query for intact biosamples (no filtering requirements)
    query = {}
    
    # Get biosamples with good data
    cursor = biosample_collection.find(query).limit(limit * 2)  # Get extra in case some are bad
    
    biosamples = []
    study_cache = {}
    
    for doc in cursor:
        # Clean up the document for JSON serialization
        doc.pop("_id", None)  # Remove ObjectId which isn't JSON serializable
        
        # Find associated studies via seq_projects (following biosample_adapters.py pattern)
        study_ids = []
        biosample_id = doc.get("biosampleGoldId")
        
        if biosample_id:
            # Query seq_projects collection for matching biosampleGoldId
            seq_projects_cursor = seq_projects_collection.find({"biosampleGoldId": biosample_id})
            
            for project in seq_projects_cursor:
                study_gold_id = project.get("studyGoldId")
                if study_gold_id:
                    study_ids.append(str(study_gold_id))
            
            # Remove duplicates
            study_ids = list(set(study_ids))
        
        # Add study associations to biosample
        if study_ids:
            doc["associated_studies"] = study_ids
        
        biosamples.append(doc)
        
        if len(biosamples) >= limit:
            break
    
    print(f"Found {len(biosamples)} GOLD biosamples")
    print(f"With study associations: {len([b for b in biosamples if 'associated_studies' in b])}")
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract intact biosamples from MongoDB")
    parser.add_argument("--limit", type=int, default=100, help="Number of biosamples to extract from each database")
    parser.add_argument("--output", type=str, help="Output JSON file path (if not specified, prints to stdout)")
    args = parser.parse_args()
    
    print(f"Extracting {args.limit} Real NMDC and GOLD Biosamples from MongoDB", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    # Connect to MongoDB
    client = connect_to_mongodb()
    if not client:
        sys.exit(1)
    
    try:
        # Extract real biosamples
        nmdc_samples = extract_nmdc_biosamples(client, limit=args.limit)
        gold_samples = extract_gold_biosamples(client, limit=args.limit)
        seq_projects = extract_sample_seq_projects(client, gold_samples, limit=args.limit//2)
        
        # Create the test data structure
        test_data = {
            "nmdc_samples": nmdc_samples,
            "gold_samples": gold_samples,
            "sample_seq_projects": seq_projects
        }
        
        # Output JSON
        json_output = json.dumps(test_data, indent=2, default=str)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
            print(f"✅ JSON written to: {args.output}", file=sys.stderr)
        else:
            print(json_output)
        
        print(f"✅ Extraction complete!", file=sys.stderr)
        print(f"📊 Summary:", file=sys.stderr)
        print(f"  NMDC biosamples: {len(nmdc_samples)}", file=sys.stderr)
        print(f"  GOLD biosamples: {len(gold_samples)}", file=sys.stderr)
        print(f"  seq_projects: {len(seq_projects)}", file=sys.stderr)
        
    finally:
        client.close()


if __name__ == "__main__":
    main()
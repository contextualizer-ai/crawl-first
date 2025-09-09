#!/usr/bin/env python3
"""
Crosswalk loader for geospatial enrichment pipeline.

Loads CSV mapping tables at startup and provides normalization functions
for converting raw classifications to ENVO-linked data.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def load_crosswalk(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load a CSV crosswalk into a lookup dictionary."""
    by_code = {}

    if not csv_path.exists():
        print(f"Warning: Crosswalk file not found: {csv_path}")
        return by_code

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            by_code[row["source_code"]] = {
                "source_label": row["source_label"],
                "envo_curie": row.get("envo_curie", ""),
                "envo_label": row.get("envo_label", ""),
                "relation": row.get("relation", "skos:closeMatch"),
                "source_url": row.get("source_url", ""),
                "notes": row.get("notes", ""),
            }

    print(f"Loaded {len(by_code)} mappings from {csv_path}")
    return by_code


# Load crosswalks at module import
MAPPING_DIR = Path(__file__).parent.parent.parent / "mappings"
CROSSWALKS = {
    "USDA_TEXTURE_12": load_crosswalk(MAPPING_DIR / "usda_texture_12.csv"),
    "OSM:natural": load_crosswalk(MAPPING_DIR / "osm_natural.csv"),
    # "WRB": load_crosswalk(MAPPING_DIR / "wrb_groups.csv"),  # Future
}


def enrich_with_linked_data(
    raw_data: Dict[str, Any], source_system: str, source_code: str
) -> Dict[str, Any]:
    """
    Enrich raw classification data with linked data normalization.

    Args:
        raw_data: Raw classification data
        source_system: Source system identifier (e.g., "USDA_TEXTURE_12")
        source_code: Source code to look up (e.g., "Lo" for Loam)

    Returns:
        Dict with raw and normalized classifications
    """
    crosswalk_table = CROSSWALKS.get(source_system, {})
    mapping = crosswalk_table.get(source_code, {})

    result = {
        "raw_classification": {
            "source_system": source_system,
            "source_code": source_code,
            "source_label": mapping.get("source_label", raw_data.get("label", "")),
            "source_version": raw_data.get("version", ""),
            "source_url": mapping.get("source_url", ""),
        },
        "linked_data": {
            "normalized": {
                "curie": mapping.get("envo_curie", ""),
                "label": mapping.get("envo_label", ""),
                "mapping_relation": mapping.get("relation", "skos:closeMatch"),
            },
            "provenance": {
                "mapping_tool": "geospatial-enrichment-crosswalk-v1.0",
                "mapping_date": datetime.utcnow().isoformat(timespec="seconds"),
                "creator": "contextualizer-ai-pipeline",
                "notes": mapping.get("notes", ""),
            },
        },
    }

    # Add success flag based on whether we found a CURIE
    result["linked_data"]["success"] = bool(mapping.get("envo_curie"))

    return result


def normalize_usda_texture(
    texture_class: str,
    sand_pct: float = None,
    clay_pct: float = None,
    silt_pct: float = None,
) -> Dict[str, Any]:
    """
    Normalize USDA texture classification with ENVO mapping.

    Args:
        texture_class: USDA texture class name (e.g., "Loam")
        sand_pct: Sand percentage (optional)
        clay_pct: Clay percentage (optional)
        silt_pct: Silt percentage (optional)

    Returns:
        Enriched texture classification with ENVO normalization
    """
    # Map full names to codes for lookup
    texture_code_map = {
        "Clay": "Cl",
        "Silty clay": "SiCl",
        "Sandy clay": "SaCl",
        "Clay loam": "ClLo",
        "Silty clay loam": "SiClLo",
        "Sandy clay loam": "SaClLo",
        "Loam": "Lo",
        "Silt loam": "SiLo",
        "Sandy loam": "SaLo",
        "Silt": "Si",
        "Sand": "Sa",
        "Loamy sand": "LoSa",
    }

    texture_code = texture_code_map.get(texture_class, texture_class)

    raw_data = {
        "label": texture_class,
        "version": "USDA_12_class",
        "sand_percent": sand_pct,
        "clay_percent": clay_pct,
        "silt_percent": silt_pct,
    }

    return enrich_with_linked_data(raw_data, "USDA_TEXTURE_12", texture_code)


def normalize_osm_natural(feature_type: str) -> Dict[str, Any]:
    """
    Normalize OSM natural feature with ENVO mapping.

    Args:
        feature_type: OSM natural tag value (e.g., "grassland")

    Returns:
        Enriched feature classification with ENVO normalization
    """
    raw_data = {
        "label": feature_type,
        "version": "OSM",
    }

    return enrich_with_linked_data(raw_data, "OSM:natural", feature_type)


# Module-level diagnostics
if __name__ == "__main__":
    print("Crosswalk Loader Diagnostics")
    print("=" * 40)

    for system, crosswalk in CROSSWALKS.items():
        print(f"{system}: {len(crosswalk)} mappings")

    # Test texture normalization
    print("\nTest: USDA Texture Normalization")
    result = normalize_usda_texture("Loam", sand_pct=40, clay_pct=20, silt_pct=40)
    print(f"Raw: {result['raw_classification']['source_label']}")
    print(
        f"ENVO: {result['linked_data']['normalized']['curie']} - {result['linked_data']['normalized']['label']}"
    )
    print(f"Success: {result['linked_data']['success']}")

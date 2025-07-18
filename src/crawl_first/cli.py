#!/usr/bin/env python3
"""
crawl-first: Deterministic biosample enrichment for LLM-ready data preparation.

Systematically follows discoverable links from NMDC biosample records to gather
environmental, geospatial, weather, publication, and ontological data.
"""

import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, TextIO, Tuple, TypeVar, Union

import click
import requests
import yaml
from artl_mcp.tools import (
    doi_to_pmid,
    extract_paper_info,
    extract_pdf_text,
    get_abstract_from_pubmed_id,
    get_doi_metadata,
    get_doi_text,
    get_full_text_from_doi,
    get_pmcid_text,
    get_pmid_text,
    get_unpaywall_info,
)
from geopy.geocoders import Nominatim
from landuse_mcp.main import get_land_cover, get_landuse_dates, get_soil_type
from nmdc_mcp.api import (
    fetch_nmdc_entity_by_id,
    fetch_nmdc_entity_by_id_with_projection,
)
from nmdc_mcp.tools import get_study_doi_details, get_study_for_biosample
from ols_mcp.tools import search_all_ontologies
from weather_mcp.main import get_weather

T = TypeVar("T")


# Configure YAML to handle problematic strings properly and disable references
class NoRefsDumper(yaml.SafeDumper):
    """Custom YAML dumper that doesn't use references/anchors."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def str_presenter(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """Custom YAML string presenter that quotes strings with special characters."""
    if (
        "\n" in data
        or data.startswith("%")
        or any(
            char in data
            for char in [
                ":",
                "{",
                "}",
                "[",
                "]",
                ",",
                "#",
                "&",
                "*",
                "!",
                "|",
                ">",
                "'",
                '"',
                "`",
            ]
        )
    ):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


NoRefsDumper.add_representer(str, str_presenter)

# Global cache directory
CACHE_DIR = Path(".cache")
FULL_TEXT_DIR = Path(".cache/full_text_files")
LOG_DIR = Path(os.getenv("LOG_DIR", "crawl_first/logs"))


class LogCapture:
    """Capture stdout/stderr and redirect to logger."""

    def __init__(
        self,
        logger: logging.Logger,
        level: int = logging.INFO,
        prefix: str = "[STDOUT]",
    ):
        self.logger = logger
        self.level = level
        self.prefix = prefix

    def write(self, data: Union[str, bytes]) -> int:
        """Write data to logger."""
        # Handle str, bytes, and other types that might be passed to write()
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        elif isinstance(data, str):
            pass  # Already a string, no conversion needed
        else:
            # Convert other types (int, float, etc.) to string
            data = str(data)

        stripped = data.strip() if data else ""
        if stripped:
            # Remove any trailing newlines and log each line separately
            lines = data.rstrip("\n\r").split("\n")
            for line in lines:
                line_stripped = line.strip()
                if line_stripped:  # Only log non-empty lines
                    self.logger.log(self.level, f"{self.prefix} {line_stripped}")

        return len(data)

    def flush(self) -> None:
        """Flush - required for file-like interface."""
        pass

    def fileno(self) -> int:
        """Return file descriptor - not supported."""
        raise OSError("fileno not supported")

    def isatty(self) -> bool:
        """Return whether this is a tty."""
        return False


class OutputManager:
    """Manage stdout/stderr capture and restoration."""

    def __init__(self) -> None:
        self.original_stdout: Optional[TextIO] = None
        self.original_stderr: Optional[TextIO] = None

    def store_originals(self) -> None:
        """Store original stdout/stderr."""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def restore_originals(self) -> None:
        """Restore original stdout/stderr."""
        if self.original_stdout is not None:
            sys.stdout = self.original_stdout
        if self.original_stderr is not None:
            sys.stderr = self.original_stderr


class OutputCapture:
    """Context manager for capturing stdout/stderr output to logger."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.output_manager = OutputManager()
        self.active = False

    def __enter__(self) -> "OutputCapture":
        """Enter context - start capturing output."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context - restore original output."""
        self.stop()

    def start(self) -> None:
        """Start capturing output."""
        if not self.active:
            self.output_manager.store_originals()

            # Create log capture objects
            stdout_capture = LogCapture(self.logger, logging.INFO, "[STDOUT]")
            stderr_capture = LogCapture(self.logger, logging.WARNING, "[STDERR]")

            # Replace stdout/stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            self.active = True

    def stop(self) -> None:
        """Stop capturing output and restore originals."""
        if self.active:
            self.output_manager.restore_originals()
            self.active = False


def should_disable_output_capture() -> bool:
    """Check if output capture should be disabled.

    Can be overridden by setting CRAWL_FIRST_FORCE_OUTPUT_CAPTURE=true to enable
    output capture even in pytest environment, or CRAWL_FIRST_DISABLE_OUTPUT_CAPTURE=true
    to disable output capture in any environment.
    """
    # Allow environment variable override
    force_capture = os.getenv("CRAWL_FIRST_FORCE_OUTPUT_CAPTURE", "").lower() == "true"
    if force_capture:
        return False  # Pretend we're not in pytest to enable capture

    disable_capture = (
        os.getenv("CRAWL_FIRST_DISABLE_OUTPUT_CAPTURE", "").lower() == "true"
    )
    if disable_capture:
        return True  # Pretend we're in pytest to disable capture

    # Default behavior: detect pytest environment
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


def setup_logging(
    verbose: bool = False, capture_output: bool = True
) -> Tuple[logging.Logger, Optional[OutputCapture]]:
    """Setup logging configuration with file and console output.

    Returns:
        Tuple of (logger, output_capture) where output_capture is None if not enabled
    """
    # Create logs directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Configure root logger to capture all library logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear any existing handlers from root
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create our main logger
    logger = logging.getLogger("crawl_first")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters with UTC timezone
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    detailed_formatter.converter = time.gmtime  # Use UTC for log entries
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler - captures EVERYTHING (our logs + library logs + stdout)
    current_time = datetime.now(timezone.utc)
    log_file = LOG_DIR / f"crawl_first_{current_time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Add file handler to root logger to capture all library logs
    root_logger.addHandler(file_handler)

    # Console handler - only for our main app logs
    console_handler = logging.StreamHandler()
    if verbose:
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(detailed_formatter)
    else:
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)

    logger.addHandler(console_handler)

    # Configure common library loggers to be more verbose in files
    library_loggers = [
        "requests",
        "urllib3",
        "geopy",
        "artl_mcp",
        "nmdc_mcp",
        "ols_mcp",
        "weather_mcp",
        "landuse_mcp",
    ]

    for lib_name in library_loggers:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # Capture stdout/stderr if requested and not in testing
    output_capture = None
    if capture_output and not should_disable_output_capture():
        output_capture = OutputCapture(logger)
        output_capture.start()

    # Log the setup
    logger.info(f"Logging initialized. Log file: {log_file}")
    if capture_output:
        logger.info("Output capture enabled - all stdout/stderr will be logged")

    return logger, output_capture


# Environmental feature types for OSM
OSM_ENVIRONMENTAL_TAGS = {
    "natural": {
        # Water features
        "water",
        "coastline",
        "bay",
        "cape",
        "peninsula",
        "strait",
        "fjord",
        "spring",
        "hot_spring",
        "geyser",
        "mineral_spring",
        # Vegetation
        "wood",
        "tree",
        "tree_row",
        "grassland",
        "scrub",
        "heath",
        "moor",
        "fell",
        "tundra",
        "shrubbery",
        # Wetlands
        "wetland",
        "marsh",
        "swamp",
        "bog",
        "fen",
        "wet_meadow",
        "reedbed",
        "saltmarsh",
        # Geological features
        "beach",
        "sand",
        "shingle",
        "bare_rock",
        "scree",
        "cliff",
        "rock",
        "stone",
        "boulder",
        "peak",
        "ridge",
        "valley",
        "saddle",
        "col",
        "spur",
        "arete",
        "cave_entrance",
        "cave",
        "sinkhole",
        "karst",
        # Volcanic/thermal
        "volcano",
        "crater",
        "lava_field",
        "fumarole",
        "mud",
        # Ice/snow
        "glacier",
        "snowfield",
        "ice_shelf",
        # Desert features
        "desert",
        "dune",
        "oasis",
    },
    "landuse": {
        # Agricultural
        "forest",
        "farmland",
        "meadow",
        "orchard",
        "vineyard",
        "agricultural",
        "grass",
        "pasture",
        "allotments",
        "plant_nursery",
        # Conservation
        "nature_reserve",
        "conservation",
        "greenfield",
        # Water management
        "wetland",
        "water",
        "reservoir",
        "salt_pond",
        "aquaculture",
        "basin",
        # Recreation
        "recreation_ground",
        "village_green",
        "cemetery",
        # Industrial/extractive
        "quarry",
        "landfill",
        "brownfield",
        "industrial",
        "commercial",
        # Waste management
        "dump",
        "waste_disposal",
        "scrap_yard",
        "salvage_yard",
        # Other
        "floodplain",
    },
    "waterway": {
        "river",
        "stream",
        "canal",
        "drain",
        "ditch",
        "brook",
        "creek",
        "rapids",
        "waterfall",
        "dam",
        "weir",
        "lock_gate",
        "dock",
        "tidal_channel",
        "intermittent",
    },
    "water": {
        "lake",
        "pond",
        "reservoir",
        "river",
        "stream",
        "canal",
        "lagoon",
        "bay",
        "wetland",
        "marsh",
        "swamp",
        "salt_water",
        "fresh_water",
        "oxbow",
        "reflecting_pool",
        "wastewater",
        "intermittent",
    },
    "wetland": {
        "marsh",
        "swamp",
        "bog",
        "fen",
        "wet_meadow",
        "reedbed",
        "saltmarsh",
        "tidalflat",
        "saltern",
        "mangrove",
    },
    "leisure": {
        "nature_reserve",
        "park",
        "garden",
        "common",
        "recreation_ground",
        "dog_park",
        "picnic_site",
        "beach_resort",
        "fishing",
    },
    "boundary": {
        "national_park",
        "protected_area",
        "nature_reserve",
        "marine_protected_area",
        "aboriginal_lands",
        "indigenous_territory",
    },
    "place": {"island", "islet", "archipelago", "atoll", "reef"},
    "geological": {"outcrop", "moraine", "ridge", "valley"},
    "man_made": {
        # Water infrastructure
        "pier",
        "breakwater",
        "groyne",
        "dyke",
        "embankment",
        "dam",
        "reservoir_covered",
        "water_tower",
        "water_well",
        "wastewater_plant",
        # Monitoring/research
        "monitoring_station",
        "weather_station",
        "research_station",
        # Waste management
        "waste_disposal",
        "wastewater_plant",
        "recycling",
    },
    "highway": {
        "track",
        "path",
        "footway",
        "bridleway",  # Can indicate human impact/access
    },
    "barrier": {"fence", "wall", "hedge"},  # Can indicate land use boundaries
    "building": {
        # Agricultural
        "barn",
        "farm",
        "farm_auxiliary",
        "greenhouse",
        "silo",
        "stable",
        "cowshed",
        "chicken_coop",
        "livestock",
        "slurry_tank",
        # Research/Educational
        "university",
        "college",
        "research",
        "laboratory",
        "observatory",
        # Industrial/Processing
        "industrial",
        "warehouse",
        "factory",
        "refinery",
        "power_plant",
        "water_treatment",
        "wastewater_treatment",
        "sewage_treatment",
        # Mining/Extractive
        "mine",
        "quarry_office",
        "mining",
        # Storage/Hazardous
        "fuel_storage",
        "storage_tank",
        "hazmat",
        "chemical_storage",
        # Utilities
        "pumping_station",
        "substation",
        "utility",
    },
    "amenity": {
        "research_station",
        "waste_disposal",
        "recycling",
        "waste_transfer_station",
        # Water/waste management
        "waste_basket",
        "toilets",
        "drinking_water",
        "water_point",
        "wastewater_plant",
        "fuel",
        "charging_station",
        # Research/education
        "university",
        "college",
        "research_institute",
        "library",
        # Agriculture related
        "animal_shelter",
        "veterinary",
        # Additional waste facilities
        "dump",
        "landfill",
    },
    "industrial": {
        "factory",
        "refinery",
        "mine",
        "quarry",
        "oil_well",
        "gas_well",
        "chemical",
        "petroleum",
        "wastewater_treatment",
        "water_treatment",
        "power",
        "solar_park",
        "wind_farm",
    },
    "power": {
        "plant",
        "generator",
        "substation",
        "transformer",
        "solar_panel",
        "wind_turbine",
        "geothermal",
    },
}


@dataclass
class OSMFeature:
    """Represents a single OpenStreetMap feature."""

    feature_id: str
    feature_type: str
    tags: Dict[str, str]
    geometry_type: str
    coordinates: Tuple[float, float]
    distance_from_center: float
    area: Optional[float] = None


def cache_key(data: Dict[str, Any]) -> str:
    """Generate MD5 cache key from data."""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def parse_collection_date(date_string: str) -> Optional[str]:
    """
    Parse collection date from various formats into YYYY-MM-DD format.

    Handles formats like:
    - 2018-07 -> 2018-07-15 (middle of month)
    - 2016-08-23 00:00:00 -> 2016-08-23
    - 2016-08-23T10:30:00 -> 2016-08-23
    - 2016-08-23 -> 2016-08-23
    """
    if not date_string or not isinstance(date_string, str):
        return None

    date_string = date_string.strip()

    # Handle YYYY-MM format (assume middle of month)
    if re.match(r"^\d{4}-\d{2}$", date_string):
        return f"{date_string}-15"

    # Handle datetime with timestamp - extract just the date part
    if re.match(r"^\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}", date_string):
        return date_string.split()[0].split("T")[0]

    # Handle standard YYYY-MM-DD format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_string):
        return date_string

    # Handle ISO date format with timezone
    if "T" in date_string:
        return date_string.split("T")[0]

    # Try to parse other formats
    try:
        # Try parsing with various common formats
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"]:
            try:
                dt = datetime.strptime(date_string, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    except Exception:
        pass

    return None


def cached_search_all_ontologies(
    query: str, ontologies: str = "envo", max_results: int = 2, exact: bool = False
) -> List[Dict[str, Any]]:
    """Cached wrapper for OLS ontology search."""
    key = cache_key(
        {
            "query": query.lower(),
            "ontologies": ontologies,
            "max_results": max_results,
            "exact": exact,
        }
    )
    cached_results = get_cached_results("ols_search", key, "results", None)
    if cached_results is not None:
        return cached_results

    try:
        results = search_all_ontologies(
            query=query, ontologies=ontologies, max_results=max_results, exact=exact
        )
        save_cache("ols_search", key, {"results": results or []})
        return results or []
    except Exception:
        save_cache("ols_search", key, {"results": []})
        return []


def cached_fetch_nmdc_entity_by_id(entity_id: str) -> Optional[Dict[str, Any]]:
    """Cached wrapper for NMDC entity fetch."""
    key = cache_key({"entity_id": entity_id})
    found, cached_entity = get_cached_entity("nmdc_entity", key)
    if found:
        return cached_entity

    try:
        entity = fetch_nmdc_entity_by_id(entity_id)
        save_cache("nmdc_entity", key, {"entity": entity})
        # MCP functions return Any, ensure we return the expected type
        return entity if isinstance(entity, dict) or entity is None else None
    except Exception:
        save_cache("nmdc_entity", key, {"entity": None})
        return None


def cached_fetch_nmdc_entity_by_id_with_projection(
    entity_id: str, collection: str, projection: List[str]
) -> Optional[Dict[str, Any]]:
    """Cached wrapper for NMDC entity fetch with projection."""
    key = cache_key(
        {
            "entity_id": entity_id,
            "collection": collection,
            "projection": sorted(projection),
        }
    )
    found, cached_entity = get_cached_entity("nmdc_entity_projection", key)
    if found:
        return cached_entity

    try:
        entity = fetch_nmdc_entity_by_id_with_projection(
            entity_id=entity_id, collection=collection, projection=projection
        )
        save_cache("nmdc_entity_projection", key, {"entity": entity})
        # MCP functions return Any, ensure we return the expected type
        return entity if isinstance(entity, dict) or entity is None else None
    except Exception:
        save_cache("nmdc_entity_projection", key, {"entity": None})
        return None


def cached_get_study_for_biosample(biosample_id: str) -> Dict[str, Any]:
    """Cached wrapper for study lookup."""
    key = cache_key({"biosample_id": biosample_id})
    cached_result = get_cached_results("study_for_biosample", key, "result", None)
    if cached_result is not None:
        return cached_result

    try:
        result = get_study_for_biosample(biosample_id)
        save_cache("study_for_biosample", key, {"result": result})
        # MCP functions return Any, ensure we return the expected type
        return result if isinstance(result, dict) else {}
    except Exception:
        save_cache("study_for_biosample", key, {"result": {}})
        return {}


def cached_get_study_doi_details(study_id: str) -> Dict[str, Any]:
    """Cached wrapper for study DOI details."""
    key = cache_key({"study_id": study_id})
    cached_result = get_cached_results("study_doi_details", key, "result", None)
    if cached_result is not None:
        return cached_result

    try:
        result = get_study_doi_details(study_id)
        save_cache("study_doi_details", key, {"result": result})
        # MCP functions return Any, ensure we return the expected type
        return result if isinstance(result, dict) else {}
    except Exception:
        save_cache("study_doi_details", key, {"result": {}})
        return {}


def save_full_text_to_file(content: str, identifiers: Dict[str, str]) -> Optional[str]:
    """
    Save full text content to a file and return the file path.

    Args:
        content: Full text content to save
        identifiers: Dictionary with doi, pmid, pmcid for filename generation

    Returns:
        Relative file path or None if failed
    """
    try:
        # Create full text directory if it doesn't exist
        FULL_TEXT_DIR.mkdir(parents=True, exist_ok=True)

        # Generate filename from identifiers
        filename_parts = []
        if identifiers.get("doi"):
            # Clean DOI for filename
            clean_doi = (
                identifiers["doi"].replace("/", "_").replace(":", "_").replace(".", "_")
            )
            filename_parts.append(f"doi_{clean_doi}")
        if identifiers.get("pmid"):
            filename_parts.append(f"pmid_{identifiers['pmid']}")
        if identifiers.get("pmcid"):
            filename_parts.append(f"pmcid_{identifiers['pmcid']}")

        if not filename_parts:
            # Fallback to hash if no identifiers
            filename_parts = [cache_key(identifiers)]

        filename = (
            "_".join(filename_parts[:2]) + ".txt"
        )  # Limit to 2 parts to avoid overly long names
        filepath = FULL_TEXT_DIR / filename

        # Write content to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # Return relative path from current directory
        return str(filepath.relative_to(Path.cwd()))

    except Exception:
        return None


def cached_get_full_text(
    paper_metadata: Dict[str, Any], email: str
) -> Optional[Dict[str, Any]]:
    """
    Cached wrapper for full text fetching using artl-mcp with file storage.

    Args:
        paper_metadata: Paper metadata containing doi, pmid, pmcid, etc.
        email: Email address for API requests

    Returns:
        Dictionary with file info or None if failed
    """
    # Create cache key from available identifiers
    identifiers = {
        "doi": paper_metadata.get("doi", ""),
        "pmid": paper_metadata.get("pmid", ""),
        "pmcid": paper_metadata.get("pmcid", ""),
    }

    key = cache_key(identifiers)
    cached = get_cache("full_text", key)
    if cached:
        # Check if cached file still exists
        if cached.get("file_path") and Path(cached["file_path"]).exists():
            return cached

    # Try ALL methods and pick the best result
    attempts = {}

    try:
        pmcid = paper_metadata.get("pmcid")
        pmid = paper_metadata.get("pmid")
        doi = paper_metadata.get("doi")

        # Strategy 1: Try artl-mcp PMCID text if available
        if pmcid:
            try:
                text = get_pmcid_text(pmcid)
                if text and len(text.strip()) > 100:
                    attempts["artl_pmcid"] = {"text": text, "length": len(text.strip())}
            except Exception:
                pass

        # Strategy 2: Try artl-mcp PMID text if available
        if pmid:
            try:
                text = get_pmid_text(pmid)
                if text and len(text.strip()) > 100:
                    attempts["artl_pmid"] = {"text": text, "length": len(text.strip())}
            except Exception:
                pass

        # Strategy 3: Try artl-mcp DOI text (simple method)
        if doi:
            try:
                text = get_doi_text(doi)
                if text and len(text.strip()) > 100:
                    attempts["artl_doi_simple"] = {
                        "text": text,
                        "length": len(text.strip()),
                    }
            except Exception:
                pass

        # Strategy 4: Try artl-mcp DOI text (advanced method with email)
        if doi:
            try:
                text = get_full_text_from_doi(doi, email)
                if text and len(text.strip()) > 100:
                    attempts["artl_doi_advanced"] = {
                        "text": text,
                        "length": len(text.strip()),
                    }
            except Exception:
                pass

        # Strategy 5: Try Unpaywall + PDF extraction if DOI available
        if doi:
            try:
                unpaywall_info = get_unpaywall_info(doi, email)
                if unpaywall_info and unpaywall_info.get("is_oa"):
                    oa_locations = unpaywall_info.get("oa_locations", [])
                    for location in oa_locations:
                        pdf_url = location.get("url_for_pdf")
                        if pdf_url:
                            text = extract_pdf_text(pdf_url)
                            if text and len(text.strip()) > 100:
                                attempts["artl_unpaywall_pdf"] = {
                                    "text": text,
                                    "length": len(text.strip()),
                                }
                                break
            except Exception:
                pass

        # Pick the best result (longest text)
        if attempts:
            best_method = max(attempts.keys(), key=lambda k: attempts[k]["length"])
            full_text = attempts[best_method]["text"]
            retrieval_method = best_method
        else:
            full_text = None
            retrieval_method = None

        # Save content to file if retrieved
        if (
            full_text and len(full_text.strip()) > 100
        ):  # Only save if substantial content
            file_path = save_full_text_to_file(full_text, identifiers)

        # Prepare result with detailed attempt information
        result = {
            "method": retrieval_method,
            "identifiers": identifiers,
            "file_path": file_path,
            "content_length": len(full_text) if full_text else 0,
            "status": (
                "retrieved"
                if full_text and file_path
                else ("partial" if full_text else "not_available")
            ),
            "methods_attempted": list(attempts.keys()) if attempts else [],
            "method_results": (
                {k: {"length": v["length"]} for k, v in attempts.items()}
                if attempts
                else {}
            ),
        }

        # Cache result
        save_cache("full_text", key, result)

        return result

    except Exception:
        # Cache failure as well
        failure_result = {
            "method": None,
            "identifiers": identifiers,
            "file_path": None,
            "content_length": 0,
            "status": "error",
        }
        save_cache("full_text", key, failure_result)
        return failure_result


def find_closest_landuse_date(
    target_date: str, available_dates: List[str]
) -> Optional[str]:
    """
    Find the closest available landuse date to the target collection date.

    Args:
        target_date: Target date in YYYY-MM-DD format
        available_dates: List of available dates in YYYY-MM-DD format

    Returns:
        Closest available date or None
    """
    if not target_date or not available_dates:
        return None

    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")

        # Convert available dates to datetime objects with their original strings
        available_dt = []
        for date_str in available_dates:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                available_dt.append((dt, date_str))
            except ValueError:
                continue

        if not available_dt:
            return None

        # Find closest date
        closest = min(available_dt, key=lambda x: abs((x[0] - target_dt).days))
        return closest[1]

    except ValueError:
        # If target date parsing fails, return first available date
        return available_dates[0] if available_dates else None


def get_cache(cache_type: str, key: str) -> Optional[Dict[str, Any]]:
    """Get data from cache."""
    cache_file = CACHE_DIR / cache_type / f"{key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                # Ensure we return a dict or None
                return data if isinstance(data, dict) else None
        except Exception:
            pass
    return None


def save_cache(cache_type: str, key: str, data: Dict[str, Any]) -> None:
    """Save data to cache."""
    cache_dir = CACHE_DIR / cache_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def get_cached_results(cache_type: str, key: str, field: str, default: T) -> T:
    """Get a specific field from cache with proper typing."""
    cached = get_cache(cache_type, key)
    if cached and field in cached:
        value = cached[field]
        # Return the cached value if it's not None, otherwise return default
        return value if value is not None else default
    return default


def get_cached_entity(
    cache_type: str, key: str
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Get an entity from cache with proper typing. Returns (found_in_cache, entity)."""
    cached = get_cache(cache_type, key)
    if cached and "entity" in cached:
        return (True, cached["entity"])
    return (False, None)


def get_soil_analysis(lat: float, lon: float) -> Dict[str, Any]:
    """Get soil type and ontology matches."""
    key = cache_key({"lat": round(lat, 4), "lon": round(lon, 4)})
    cached = get_cache("soil", key)
    if cached:
        return cached

    try:
        soil_type = get_soil_type(lat=lat, lon=lon)
        ontology_matches = []
        if soil_type:
            try:
                matches = cached_search_all_ontologies(
                    query=soil_type, ontologies="envo", max_results=2, exact=False
                )
                if matches:
                    # Clean up matches - keep entries with valid obo_id
                    cleaned_matches = []
                    for match in matches[:2]:  # Only take top 2 most relevant
                        if match.get("obo_id"):  # Check for valid obo_id instead of id
                            cleaned_match = {
                                "iri": match.get("iri"),
                                "short_form": match.get("short_form"),
                                "obo_id": match.get("obo_id"),
                                "label": match.get("label"),
                                "description": match.get("description"),
                            }
                            cleaned_matches.append(cleaned_match)
                    ontology_matches = cleaned_matches
            except Exception:
                pass

        result = {"soil_type": soil_type, "ontology_matches": ontology_matches}
        save_cache("soil", key, result)
        return result
    except Exception:
        return {"soil_type": None, "ontology_matches": []}


def get_land_cover_analysis(lat: float, lon: float, date: str) -> Dict[str, Any]:
    """Get land cover data and ontology matches with intelligent date parsing."""
    # Parse the collection date
    parsed_date = parse_collection_date(date) if date else None

    if parsed_date:
        # Get available landuse dates for this location
        try:
            available_dates = get_landuse_dates(lat=lat, lon=lon)
            if available_dates and isinstance(available_dates, list):
                # Find closest available date
                closest_date = find_closest_landuse_date(parsed_date, available_dates)
                if closest_date:
                    year = closest_date.split("-")[0]
                    actual_start_date = closest_date
                    actual_end_date = closest_date
                else:
                    # Fallback to year from parsed date
                    year = parsed_date.split("-")[0]
                    actual_start_date = f"{year}-01-01"
                    actual_end_date = f"{year}-12-31"
            else:
                # Fallback to year from parsed date
                year = parsed_date.split("-")[0]
                actual_start_date = f"{year}-01-01"
                actual_end_date = f"{year}-12-31"
        except Exception:
            # Fallback to year from parsed date
            year = parsed_date.split("-")[0]
            actual_start_date = f"{year}-01-01"
            actual_end_date = f"{year}-12-31"
    else:
        # Default fallback
        year = "2001"
        actual_start_date = f"{year}-01-01"
        actual_end_date = f"{year}-12-31"

    key = cache_key(
        {"lat": round(lat, 4), "lon": round(lon, 4), "date": actual_start_date}
    )
    cached = get_cache("land_cover", key)
    if cached:
        return cached

    try:
        land_cover = get_land_cover(
            lat=lat, lon=lon, start_date=actual_start_date, end_date=actual_end_date
        )

        ontology_matches = {}
        if land_cover:
            unique_terms = set()
            for system, entries in land_cover.items():
                for entry in entries:
                    term = entry.get("envo_term")
                    if term and term != "ENVO Term unavailable":
                        unique_terms.add(term)

            for term in unique_terms:
                try:
                    matches = cached_search_all_ontologies(
                        query=term, ontologies="envo", max_results=2, exact=False
                    )
                    if matches:
                        # Clean up matches - keep entries with valid obo_id
                        cleaned_matches = []
                        for match in matches[:2]:  # Only take top 2 most relevant
                            if match.get(
                                "obo_id"
                            ):  # Check for valid obo_id instead of id
                                cleaned_match = {
                                    "iri": match.get("iri"),
                                    "short_form": match.get("short_form"),
                                    "obo_id": match.get("obo_id"),
                                    "label": match.get("label"),
                                    "description": match.get("description"),
                                }
                                cleaned_matches.append(cleaned_match)
                        if cleaned_matches:
                            ontology_matches[term] = cleaned_matches
                except Exception:
                    pass

        result = {"land_cover": land_cover, "ontology_matches": ontology_matches}
        save_cache("land_cover", key, result)
        return result
    except Exception:
        return {"land_cover": None, "ontology_matches": {}}


def get_nearby_day_date(date_str: str, day_offset: int) -> Optional[str]:
    """Get a date offset by specified days."""
    try:
        from datetime import timedelta

        dt = datetime.strptime(date_str, "%Y-%m-%d")
        new_dt = dt + timedelta(days=day_offset)
        return new_dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None


def get_weather_analysis(lat: float, lon: float, date: str) -> Optional[Dict[str, Any]]:
    """Get weather data with YAML-safe formatting and intelligent date parsing."""
    # Parse the collection date
    parsed_date = parse_collection_date(date) if date else None
    if not parsed_date:
        return None

    key = cache_key({"lat": round(lat, 3), "lon": round(lon, 3), "date": parsed_date})
    cached = get_cache("weather", key)
    if cached:
        # Handle both old format (direct data) and new format (with result field)
        if "result" in cached:
            result = cached["result"]
            return result if isinstance(result, dict) or result is None else None
        return cached if isinstance(cached, dict) else None

    # Try multiple strategies for weather data with progressively relaxed parameters
    strategies: List[Dict[str, Any]] = [
        # Strategy 1: Exact date, close radius, high coverage
        {"search_radius_km": 50, "coverage_threshold": 0.3, "date": parsed_date},
        # Strategy 2: Exact date, medium radius, lower coverage
        {"search_radius_km": 100, "coverage_threshold": 0.2, "date": parsed_date},
        # Strategy 3: Nearby dates (±1 day), close radius
        {
            "search_radius_km": 75,
            "coverage_threshold": 0.25,
            "date": get_nearby_day_date(parsed_date, -1),
        },
        {
            "search_radius_km": 75,
            "coverage_threshold": 0.25,
            "date": get_nearby_day_date(parsed_date, 1),
        },
        # Strategy 4: Last resort - wider radius with exact date
        {"search_radius_km": 200, "coverage_threshold": 0.1, "date": parsed_date},
    ]

    for i, strategy in enumerate(strategies):
        try:
            if not strategy["date"]:  # Skip if date calculation failed
                continue

            weather_data = get_weather(
                lat=lat,
                lon=lon,
                date=strategy["date"],
                search_radius_km=strategy["search_radius_km"],
                timeseries_type="daily",
                coverage_threshold=strategy["coverage_threshold"],
                measurement_units="scientific",
            )

            if weather_data:
                # Clean for YAML serialization
                cleaned = {
                    "coverage": weather_data.get("coverage"),
                    "strategy_used": i + 1,
                    "strategy_details": {
                        "search_radius_km": strategy["search_radius_km"],
                        "coverage_threshold": strategy["coverage_threshold"],
                        "date_used": strategy["date"],
                    },
                }

                if "station" in weather_data:
                    station = weather_data["station"]
                    cleaned["station"] = {
                        k: v
                        for k, v in station.items()
                        if k
                        in [
                            "name",
                            "country",
                            "region",
                            "wmo",
                            "icao",
                            "latitude",
                            "longitude",
                            "elevation",
                            "timezone",
                            "distance",
                        ]
                    }

                if "data" in weather_data:
                    measurements = {}
                    for measurement_type, values in weather_data["data"].items():
                        if isinstance(values, dict) and values:
                            for timestamp, value in values.items():
                                if value is not None and not (
                                    isinstance(value, float) and str(value) == "nan"
                                ):
                                    # Convert temperature from Kelvin to Celsius
                                    if measurement_type in [
                                        "tavg",
                                        "tmin",
                                        "tmax",
                                    ] and isinstance(value, (int, float)):
                                        value = round(
                                            value - 273.15, 1
                                        )  # Kelvin to Celsius
                                    measurements[measurement_type] = value
                                    break
                    cleaned["measurements"] = measurements

                save_cache("weather", key, {"result": cleaned})
                return cleaned

        except Exception:
            # Continue to next strategy on failure
            continue

    # If all strategies failed, cache and return None
    save_cache("weather", key, {"result": None})
    return None


def get_publication_analysis(biosample_id: str, email: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive study and DOI information."""
    key = cache_key({"biosample_id": biosample_id})
    cached = get_cache("publications", key)
    if cached:
        # Handle both old format (direct data) and new format (with result field)
        if "result" in cached:
            result = cached["result"]
            return result if isinstance(result, dict) or result is None else None
        return cached if isinstance(cached, dict) else None

    try:
        # Get study for biosample
        study_result = cached_get_study_for_biosample(biosample_id)
        if "error" in study_result or not study_result.get("study_id"):
            save_cache("publications", key, {"result": None})
            return None

        study_id = study_result["study_id"]

        # Get DOI details and comprehensive study data
        doi_details = cached_get_study_doi_details(study_id)
        study_data = cached_fetch_nmdc_entity_by_id_with_projection(
            entity_id=study_id,
            collection="study_set",
            projection=[
                "id",
                "name",
                "title",
                "description",
                "funding_sources",
                "websites",
                "associated_dois",
                "protocol_link",
                "objective",
                "notes",
                "homepage_website",
                "alternative_titles",
                "alternative_names",
                "alternative_descriptions",
            ],
        )

        if "error" in doi_details:
            save_cache("publications", key, {"result": None})
            return None

        # Build study info
        study_info = {
            "study_id": study_id,
            "study_name": doi_details.get("study_name", ""),
        }

        if study_data and isinstance(study_data, dict) and "error" not in study_data:
            study_info.update(
                {
                    "study_title": study_data.get("title", ""),
                    "study_description": study_data.get("description", ""),
                    "funding_sources": study_data.get("funding_sources", []),
                    "websites": study_data.get("websites", []),
                    "protocol_link": study_data.get("protocol_link", ""),
                    "objective": study_data.get("objective", ""),
                    "notes": study_data.get("notes", ""),
                    "homepage_website": study_data.get("homepage_website", ""),
                    "alternative_titles": study_data.get("alternative_titles", []),
                    "alternative_names": study_data.get("alternative_names", []),
                    "alternative_descriptions": study_data.get(
                        "alternative_descriptions", []
                    ),
                }
            )

        # Process DOIs
        all_dois = (
            study_data.get("associated_dois", [])
            if study_data
            else doi_details.get("associated_dois", [])
        )
        enhanced_dois = []
        publication_doi_count = 0

        for doi_entry in all_dois:
            doi_value = doi_entry.get("doi_value", "")
            doi_category = doi_entry.get("doi_category", "")

            if doi_value:
                enhanced_doi = {**doi_entry}

                # Add full URL for DOI
                clean_doi = doi_value.replace("doi:", "").strip()
                enhanced_doi["doi_url"] = f"https://doi.org/{clean_doi}"

                # Only get detailed metadata for publication DOIs
                if doi_category == "publication_doi":
                    publication_doi_count += 1
                    clean_doi = doi_value.replace("doi:", "").strip()

                    # Get PMID
                    pmid = None
                    try:
                        pmid = doi_to_pmid(clean_doi)
                    except Exception:
                        pass

                    # Get paper metadata
                    paper_metadata = {
                        "doi": clean_doi,
                        "pmid": pmid,
                        "title": "",
                        "authors": [],
                        "journal": "",
                        "citation_count": None,
                        "abstract": "",
                        "retrieval_errors": [],
                    }

                    try:
                        doi_metadata = get_doi_metadata(clean_doi)
                        if doi_metadata and doi_metadata.get("status") == "ok":
                            work_item = doi_metadata.get("message", {})
                            if work_item:
                                paper_info = extract_paper_info(work_item)
                                paper_metadata.update(paper_info)
                    except Exception as e:
                        paper_metadata["retrieval_errors"].append(
                            f"CrossRef error: {e}"
                        )

                    # Get abstract from PubMed if needed
                    if pmid and not paper_metadata.get("abstract"):
                        try:
                            abstract = get_abstract_from_pubmed_id(pmid)
                            if abstract and abstract.strip():
                                paper_metadata["abstract"] = abstract
                        except Exception as e:
                            paper_metadata["retrieval_errors"].append(
                                f"PubMed error: {e}"
                            )

                    # Try to get full text content
                    full_text_metadata = {
                        "doi": clean_doi,
                        "pmid": pmid,
                        "pmcid": None,  # Could be enhanced to fetch PMCID from PMID
                    }

                    full_text_result = cached_get_full_text(full_text_metadata, email)
                    if full_text_result:
                        # Add full text file information to paper metadata
                        paper_metadata["full_text"] = {
                            "status": full_text_result.get("status"),
                            "file_path": full_text_result.get("file_path"),
                            "content_length": full_text_result.get("content_length"),
                            "retrieval_method": full_text_result.get("method"),
                        }

                        # Include PDF URL if available but PyPDF2 not installed
                        if full_text_result.get("pdf_url"):
                            paper_metadata["full_text"]["pdf_url"] = (
                                full_text_result.get("pdf_url")
                            )
                    else:
                        paper_metadata["full_text"] = {
                            "status": "not_available",
                            "file_path": None,
                            "content_length": 0,
                            "retrieval_method": None,
                        }

                    enhanced_doi.update(
                        {
                            "converted_ids": {
                                "original_doi": doi_value,
                                "pmid": pmid,
                                "pmcid": None,
                            },
                            "paper_metadata": paper_metadata,
                        }
                    )

                enhanced_dois.append(enhanced_doi)

        result = {
            "biosample_id": biosample_id,
            "study_info": study_info,
            "total_dois": len(all_dois),
            "publication_doi_count": publication_doi_count,
            "award_doi_count": len(
                [doi for doi in enhanced_dois if doi.get("doi_category") == "award_doi"]
            ),
            "dataset_doi_count": len(
                [
                    doi
                    for doi in enhanced_dois
                    if doi.get("doi_category") == "dataset_doi"
                ]
            ),
        }

        # Store all_dois separately so it can be extracted as a sibling
        result["_all_dois"] = enhanced_dois

        save_cache("publications", key, {"result": result})
        return result

    except Exception:
        save_cache("publications", key, {"result": None})
        return None


def get_elevation(lat: float, lon: float) -> Optional[float]:
    """Get elevation for coordinates using multiple free APIs with fallbacks."""
    key = cache_key({"lat": round(lat, 6), "lon": round(lon, 6)})
    cached = get_cache("elevation", key)
    if cached:
        return cached.get("elevation")

    elevation = None

    # Try Open-Elevation API (free, no rate limits)
    try:
        url = "https://api.open-elevation.com/api/v1/lookup"
        params = {"locations": f"{lat},{lon}"}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            elevation = float(data["results"][0]["elevation"])
    except Exception:
        pass

    # Fallback to USGS Elevation Point Query Service (US only, but free)
    if elevation is None:
        try:
            url = "https://nationalmap.gov/epqs/pqs.php"
            params = {"x": str(lon), "y": str(lat), "units": "Meters", "output": "json"}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "USGS_Elevation_Point_Query_Service" in data:
                result = data["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]
                if result.get("Elevation") not in [None, -1000000]:
                    elevation = float(result["Elevation"])
        except Exception:
            pass

    save_cache("elevation", key, {"elevation": elevation})
    return elevation


def geocode_location_name(location_name: str) -> Dict[str, Any]:
    """Geocode location name to get coordinates using Nominatim."""
    key = cache_key({"location_name": location_name.lower().strip()})
    cached = get_cache("geocode", key)
    if cached:
        return cached

    geolocator = Nominatim(user_agent="biosample-analyzer/1.0")

    result: Dict[str, Any] = {
        "location_name": location_name,
        "coordinates": None,
        "display_name": None,
        "place_type": None,
        "address": {},
        "administrative": {},
        "error": None,
    }

    try:
        sleep(1.1)  # Rate limiting for Nominatim
        location = geolocator.geocode(location_name, language="en")

        if location:
            result["coordinates"] = {
                "latitude": location.latitude,
                "longitude": location.longitude,
            }
            result["display_name"] = location.address
            result["place_type"] = location.raw.get("type", "unknown")

            address = location.raw.get("address", {})
            address_fields = [
                "house_number",
                "road",
                "neighbourhood",
                "suburb",
                "village",
                "town",
                "city",
                "municipality",
                "county",
                "state",
                "country",
                "postcode",
                "country_code",
            ]

            for field in address_fields:
                if field in address:
                    result["address"][field] = address[field]
        else:
            result["error"] = "Location not found"

    except Exception as e:
        result["error"] = str(e)

    save_cache("geocode", key, result)
    return result


def reverse_geocode(lat: float, lon: float) -> Dict[str, Any]:
    """Reverse geocode coordinates to get place information using Nominatim."""
    key = cache_key({"lat": round(lat, 5), "lon": round(lon, 5)})
    cached = get_cache("reverse_geocode", key)
    if cached:
        return cached

    geolocator = Nominatim(user_agent="biosample-analyzer/1.0")

    result = {
        "coordinates": {"latitude": lat, "longitude": lon},
        "display_name": None,
        "place_type": None,
        "address": {},
        "administrative": {},
        "error": None,
    }

    try:
        sleep(1.1)  # Rate limiting for Nominatim
        location = geolocator.reverse(f"{lat}, {lon}", language="en", zoom=18)

        if location:
            result["display_name"] = location.address
            result["place_type"] = location.raw.get("type", "unknown")

            address = location.raw.get("address", {})
            address_fields = [
                "house_number",
                "road",
                "neighbourhood",
                "suburb",
                "village",
                "town",
                "city",
                "municipality",
                "county",
                "state",
                "country",
                "postcode",
                "country_code",
            ]

            # Safely access the address dict
            address_dict = result["address"]
            assert isinstance(address_dict, dict)  # Help mypy understand this is a dict

            for field in address_fields:
                if field in address:
                    address_dict[field] = address[field]

            # Safely access the administrative dict
            admin_dict = result["administrative"]
            assert isinstance(admin_dict, dict)  # Help mypy understand this is a dict

            admin_fields = ["state_district", "region", "province", "state_code"]
            for field in admin_fields:
                if field in address:
                    admin_dict[field] = address[field]

            if "amenity" in location.raw:
                result["amenity"] = location.raw["amenity"]
            if "natural" in location.raw:
                result["natural"] = location.raw["natural"]
            if "landuse" in location.raw:
                result["landuse"] = location.raw["landuse"]
        else:
            # Explicitly assign string to the error field
            error_msg: Any = "No location found"
            result["error"] = error_msg

    except Exception as e:
        # Explicitly assign string to the error field
        error_str: Any = str(e)
        result["error"] = error_str

    save_cache("reverse_geocode", key, result)
    return result


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points in meters."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c * 6371000  # Earth radius in meters


def build_overpass_query(lat: float, lon: float, radius: float = 1000) -> str:
    """Build Overpass API query for environmental features."""
    tag_queries = []
    for key, values in OSM_ENVIRONMENTAL_TAGS.items():
        for value in values:
            tag_queries.append(f'nwr["{key}"="{value}"](around:{radius},{lat},{lon});')

    return f"""
    [out:json][timeout:120];
    ({' '.join(tag_queries)});
    out body center qt;
    """


def get_feature_type(tags: Dict[str, str]) -> Optional[str]:
    """Determine primary feature type from OSM tags."""
    # Check in priority order - more specific environmental features first
    categories = [
        "natural",
        "water",
        "wetland",
        "waterway",
        "geological",
        "landuse",
        "leisure",
        "boundary",
        "place",
        "man_made",
        "building",
        "industrial",
        "power",
        "amenity",
        "highway",
        "barrier",
    ]

    for category in categories:
        if category in tags and tags[category] in OSM_ENVIRONMENTAL_TAGS.get(
            category, set()
        ):
            return f"{category}:{tags[category]}"
    return None


def extract_coordinates(element: Dict) -> Tuple[float, float]:
    """Extract coordinates from OSM element."""
    if "center" in element:
        return (element["center"]["lat"], element["center"]["lon"])
    elif "lat" in element and "lon" in element:
        return (element["lat"], element["lon"])
    else:
        raise ValueError("No coordinates found in element")


def query_osm_features(
    lat: float, lon: float, radius: float = 1000
) -> List[OSMFeature]:
    """Query OpenStreetMap features around a point using Overpass API."""
    key = cache_key({"lat": round(lat, 4), "lon": round(lon, 4), "radius": radius})
    cached = get_cache("osm_features", key)
    if cached:
        features = []
        for feature_data in cached.get("features", []):
            feature = OSMFeature(
                feature_id=feature_data["feature_id"],
                feature_type=feature_data["feature_type"],
                tags=feature_data["tags"],
                geometry_type=feature_data["geometry_type"],
                coordinates=tuple(feature_data["coordinates"]),
                distance_from_center=feature_data["distance_from_center"],
                area=feature_data.get("area"),
            )
            features.append(feature)
        return features

    query = build_overpass_query(lat, lon, radius)
    url = "https://overpass-api.de/api/interpreter"

    for attempt in range(3):
        try:
            response = requests.post(url, data={"data": query}, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "elements" not in data:
                return []

            features = []
            for element in data["elements"]:
                try:
                    if "tags" not in element:
                        continue

                    feature_type = get_feature_type(element["tags"])
                    if not feature_type:
                        continue

                    coordinates = extract_coordinates(element)
                    distance = calculate_distance(
                        lat, lon, coordinates[0], coordinates[1]
                    )

                    if distance > radius:
                        continue

                    area = None
                    if "area" in element:
                        try:
                            area = float(element["area"])
                        except (ValueError, TypeError):
                            pass

                    feature = OSMFeature(
                        feature_id=str(element["id"]),
                        feature_type=feature_type,
                        tags=element["tags"],
                        geometry_type=element["type"],
                        coordinates=coordinates,
                        distance_from_center=distance,
                        area=area,
                    )
                    features.append(feature)

                except Exception:
                    continue

            cache_data = {
                "features": [
                    {
                        "feature_id": f.feature_id,
                        "feature_type": f.feature_type,
                        "tags": f.tags,
                        "geometry_type": f.geometry_type,
                        "coordinates": f.coordinates,
                        "distance_from_center": f.distance_from_center,
                        "area": f.area,
                    }
                    for f in features
                ]
            }
            save_cache("osm_features", key, cache_data)
            return features

        except requests.exceptions.RequestException:
            if attempt < 2:
                sleep(5)
                continue
            else:
                return []
        except Exception:
            # Catch any other exceptions and return empty list
            return []

    # Fallback return (should never be reached)
    return []


def summarize_osm_features(
    features: List[OSMFeature], center_lat: float, center_lon: float
) -> Dict[str, Any]:
    """Summarize OSM features by category with environmental focus."""
    summary: Dict[str, Any] = {
        "metadata": {
            "total_features": len(features),
            "query_coordinates": [center_lat, center_lon],
            "feature_counts": {},
        },
        "environmental_features": {},
        "nearest_features": [],
    }

    # Type-safe accessors to help mypy
    nearest_features_list = summary["nearest_features"]
    assert isinstance(nearest_features_list, list)
    environmental_features_dict = summary["environmental_features"]
    assert isinstance(environmental_features_dict, dict)
    metadata_dict = summary["metadata"]
    assert isinstance(metadata_dict, dict)
    feature_counts_dict = metadata_dict["feature_counts"]
    assert isinstance(feature_counts_dict, dict)

    # Temporary lists for counting (not included in final output)
    water_features_temp = []
    natural_areas_temp = []
    protected_areas_temp = []

    sorted_features = sorted(features, key=lambda x: x.distance_from_center)

    # Get nearest 5 features regardless of type
    for f in sorted_features[:5]:
        feature_info = {
            "type": f.feature_type,
            "distance_m": round(f.distance_from_center, 1),
        }
        # Only add name if it exists and isn't "Unnamed"
        name = f.tags.get("name")
        if name and name != "Unnamed":
            feature_info["name"] = name
        nearest_features_list.append(feature_info)

    for feature in features:
        category = feature.feature_type.split(":")[0]
        feature_type = feature.feature_type.split(":")[1]

        if category not in feature_counts_dict:
            feature_counts_dict[category] = {}
        if feature_type not in feature_counts_dict[category]:
            feature_counts_dict[category][feature_type] = 0
        feature_counts_dict[category][feature_type] += 1

        # Only include features that have names (skip unnamed features)
        name = feature.tags.get("name")
        if name and name != "Unnamed":
            if category not in environmental_features_dict:
                environmental_features_dict[category] = []

            feature_info = {
                "type": feature_type,
                "distance_m": round(feature.distance_from_center, 1),
                "name": name,
            }

            # Add relevant tags (excluding name since we handle it separately)
            relevant_tags = {
                k: v
                for k, v in feature.tags.items()
                if k in ["description", "wikipedia"] and v
            }
            if relevant_tags:
                feature_info["tags"] = relevant_tags

            if feature.area:
                feature_info["area_m2"] = feature.area

            environmental_features_dict[category].append(feature_info)

        # Count ALL features for environmental summary (including unnamed ones)
        if category in ["water", "waterway"] or feature_type in [
            "water",
            "lake",
            "pond",
            "river",
            "stream",
        ]:
            water_features_temp.append(1)  # Just count, don't store data

        if category == "natural" or feature_type in [
            "forest",
            "wood",
            "grassland",
            "wetland",
        ]:
            natural_areas_temp.append(1)  # Just count, don't store data

        if feature_type in [
            "nature_reserve",
            "national_park",
            "protected_area",
            "conservation",
        ]:
            protected_areas_temp.append(1)  # Just count, don't store data

    # Sort and deduplicate - keep only closest instance of each named feature
    for category in environmental_features_dict:
        # Sort by distance first
        category_features = environmental_features_dict[category]
        assert isinstance(category_features, list)  # Help mypy
        category_features.sort(key=lambda x: x["distance_m"])

        # Deduplicate by name - keep only the closest (first) occurrence
        seen_names = set()
        unique_features = []
        for feature in category_features:
            name = feature["name"]
            if name not in seen_names:
                seen_names.add(name)
                unique_features.append(feature)

        environmental_features_dict[category] = unique_features

    return summary


def get_geospatial_analysis(
    lat: float, lon: float, radius: float = 1000
) -> Dict[str, Any]:
    """Get comprehensive geospatial analysis for a location."""
    analysis: Dict[str, Any] = {
        "coordinates": {"latitude": lat, "longitude": lon},
        "elevation": None,
        "place_info": {},
        "osm_features": {},
        "environmental_summary": {},
    }

    # Get elevation
    elevation = get_elevation(lat, lon)
    if elevation is not None:
        analysis["elevation"] = {
            "meters": elevation,
            "feet": round(elevation * 3.28084, 1),
        }

    # Get place information via reverse geocoding
    place_info = reverse_geocode(lat, lon)
    analysis["place_info"] = place_info

    # Get nearby OSM features
    osm_features = query_osm_features(lat, lon, radius)
    osm_summary = summarize_osm_features(osm_features, lat, lon)
    analysis["osm_features"] = osm_summary

    # Create environmental summary - counts come from feature_counts metadata
    feature_counts = osm_summary["metadata"]["feature_counts"]

    # Calculate counts from metadata (includes both named and unnamed features)
    water_count = 0
    natural_count = 0
    protected_count = 0

    for category, types in feature_counts.items():
        for feature_type, count in types.items():
            if category in ["water", "waterway"] or feature_type in [
                "water",
                "lake",
                "pond",
                "river",
                "stream",
            ]:
                water_count += count
            if category == "natural" or feature_type in [
                "forest",
                "wood",
                "grassland",
                "wetland",
            ]:
                natural_count += count
            if feature_type in [
                "nature_reserve",
                "national_park",
                "protected_area",
                "conservation",
            ]:
                protected_count += count

    # Create environmental summary with explicit typing
    env_summary: Dict[str, Any] = {
        "total_environmental_features": osm_summary["metadata"]["total_features"],
        "water_features_nearby": water_count,
        "natural_areas_nearby": natural_count,
        "protected_areas_nearby": protected_count,
        "dominant_land_types": [],
        "ecological_context": "",
    }
    analysis["environmental_summary"] = env_summary

    # Determine dominant land types
    if feature_counts:
        all_types = []
        for category, types in feature_counts.items():
            for type_name, count in types.items():
                all_types.append((f"{category}:{type_name}", count))

        dominant = sorted(all_types, key=lambda x: x[1], reverse=True)[:3]
        analysis["environmental_summary"]["dominant_land_types"] = [
            {"type": t[0], "count": t[1]} for t in dominant
        ]

    # Generate ecological context description
    eco_description = []
    if water_count > 0:
        eco_description.append(f"{water_count} water feature(s)")
    if natural_count > 0:
        eco_description.append(f"{natural_count} natural area(s)")
    if protected_count > 0:
        eco_description.append(f"{protected_count} protected area(s)")

    if eco_description:
        analysis["environmental_summary"][
            "ecological_context"
        ] = f"Location has {', '.join(eco_description)} within {radius}m radius"
    else:
        analysis["environmental_summary"][
            "ecological_context"
        ] = f"No major environmental features found within {radius}m radius"

    return analysis


def analyze_biosample(
    biosample_id: str, email: str, search_radius: int = 1000
) -> Optional[Dict[str, Any]]:
    """Perform comprehensive analysis of a biosample."""
    try:
        # Get complete biosample data
        full_biosample = cached_fetch_nmdc_entity_by_id(biosample_id)
        if not full_biosample:
            return None

        # Extract coordinates and date
        lat_lon = full_biosample.get("lat_lon", {})
        asserted_lat = lat_lon.get("latitude") if isinstance(lat_lon, dict) else None
        asserted_lon = lat_lon.get("longitude") if isinstance(lat_lon, dict) else None

        # Extract geo_loc_name for geocoding
        geo_loc_name = full_biosample.get("geo_loc_name", {})
        location_name = (
            geo_loc_name.get("has_raw_value")
            if isinstance(geo_loc_name, dict)
            else None
        )

        collection_date = full_biosample.get("collection_date", {})
        date = (
            collection_date.get("has_raw_value")
            if isinstance(collection_date, dict)
            else None
        )
        parsed_date = parse_collection_date(date) if date else None

        # Perform all analyses
        inferred = {}

        # Initialize coordinate sources
        coord_sources: Dict[str, Any] = {}

        # Process asserted coordinates
        if asserted_lat is not None and asserted_lon is not None:
            coord_sources["from_asserted_coords"] = {
                "coordinates": {"latitude": asserted_lat, "longitude": asserted_lon},
                "source": "lat_lon field",
                "map_url": f"https://www.google.com/maps/place/{asserted_lat},{asserted_lon}/@{asserted_lat},{asserted_lon},15z",
            }

            # Soil analysis from asserted coords
            inferred["soil_from_asserted_coords"] = get_soil_analysis(
                asserted_lat, asserted_lon
            )

            # Land cover analysis from asserted coords
            if date:
                inferred["land_cover_from_asserted_coords"] = get_land_cover_analysis(
                    asserted_lat, asserted_lon, date
                )

                # Available landuse dates (show matched date + neighbors)
                if parsed_date:
                    try:
                        dates_key = cache_key(
                            {
                                "lat": round(asserted_lat, 4),
                                "lon": round(asserted_lon, 4),
                            }
                        )
                        cached_dates = get_cache("landuse_dates", dates_key)
                        if not cached_dates:
                            cached_dates = get_landuse_dates(
                                lat=asserted_lat, lon=asserted_lon
                            )
                            save_cache("landuse_dates", dates_key, cached_dates)

                        if cached_dates and isinstance(cached_dates, list):
                            closest_date = find_closest_landuse_date(
                                parsed_date, cached_dates
                            )
                            if closest_date:
                                # Find index of closest date
                                try:
                                    idx = cached_dates.index(closest_date)
                                    # Get one before, matched, and one after
                                    neighbor_dates = []
                                    if idx > 0:
                                        neighbor_dates.append(cached_dates[idx - 1])
                                    neighbor_dates.append(closest_date)
                                    if idx < len(cached_dates) - 1:
                                        neighbor_dates.append(cached_dates[idx + 1])

                                    inferred["landuse_dates_from_asserted_coords"] = {
                                        "target_collection_date": parsed_date,
                                        "closest_available_date": closest_date,
                                        "neighboring_dates": neighbor_dates,
                                    }
                                except ValueError:
                                    pass
                    except Exception:
                        pass

                # Weather analysis from asserted coords
                weather = get_weather_analysis(asserted_lat, asserted_lon, date)
                if weather:
                    inferred["weather_from_asserted_coords"] = weather

            # Geospatial analysis from asserted coords
            geospatial_asserted = get_geospatial_analysis(
                asserted_lat, asserted_lon, radius=search_radius
            )
            inferred["geospatial_from_asserted_coords"] = geospatial_asserted

        # Process geocoded coordinates from location name
        if location_name and location_name.strip():
            geocode_result = geocode_location_name(location_name.strip())

            if geocode_result.get("coordinates") and not geocode_result.get("error"):
                geocoded_coords = geocode_result["coordinates"]
                geocoded_lat = geocoded_coords["latitude"]
                geocoded_lon = geocoded_coords["longitude"]

                # Get elevation from geocoded coordinates
                elevation_geocoded = get_elevation(geocoded_lat, geocoded_lon)

                coord_sources["from_geo_loc_name"] = {
                    "coordinates": geocoded_coords,
                    "source": f"geocoded from geo_loc_name: {location_name}",
                    "geocode_result": geocode_result,
                    "map_url": f"https://www.google.com/maps/place/{geocoded_lat},{geocoded_lon}/@{geocoded_lat},{geocoded_lon},15z",
                }

                # Add elevation to the geocoded coordinate source if available
                if elevation_geocoded is not None:
                    geo_loc_source = coord_sources["from_geo_loc_name"]
                    assert isinstance(geo_loc_source, dict)  # Help mypy
                    geo_loc_source["elevation_meters"] = elevation_geocoded

        # Add coordinate sources summary with combined map and distance calculations
        if coord_sources:
            # Create combined map URL and calculate distances if we have both coordinate sources
            if (
                "from_asserted_coords" in coord_sources
                and "from_geo_loc_name" in coord_sources
            ):
                asserted_coords = coord_sources["from_asserted_coords"]["coordinates"]
                geocoded_coords = coord_sources["from_geo_loc_name"]["coordinates"]

                # Type assertions to help mypy
                assert isinstance(asserted_coords, dict)
                assert isinstance(geocoded_coords, dict)

                # Create URL with both locations pinned
                asserted_lat, asserted_lon = (
                    asserted_coords["latitude"],
                    asserted_coords["longitude"],
                )
                geocoded_lat, geocoded_lon = (
                    geocoded_coords["latitude"],
                    geocoded_coords["longitude"],
                )

                # Calculate distance between asserted and geocoded coordinates
                distance_meters = calculate_distance(
                    asserted_lat, asserted_lon, geocoded_lat, geocoded_lon
                )
                distance_km = round(distance_meters / 1000, 2)

                # Use Google Maps directions/multiple destinations format to show both pins
                combined_map_url = (
                    f"https://www.google.com/maps/dir/{asserted_lat},{asserted_lon}/"
                    f"{geocoded_lat},{geocoded_lon}/@{asserted_lat},{asserted_lon},10z"
                )

                coord_sources["combined_map_url"] = combined_map_url
                coord_sources["combined_map_description"] = (
                    f"Asserted coords ({asserted_lat}, {asserted_lon}) vs "
                    f"Geocoded coords ({geocoded_lat}, {geocoded_lon})"
                )
                coord_sources["coordinate_distance"] = {
                    "meters": round(distance_meters, 1),
                    "kilometers": distance_km,
                    "description": f"Distance between asserted and geocoded coordinates: {distance_km} km",
                }

            # Calculate elevation difference if we have both asserted elevation and geospatial elevation
            asserted_elevation = full_biosample.get(
                "elev"
            )  # Check if biosample has asserted elevation
            if (
                asserted_elevation is not None
                and "geospatial_from_asserted_coords" in inferred
                and inferred["geospatial_from_asserted_coords"].get("elevation")
            ):

                geospatial_elevation = inferred["geospatial_from_asserted_coords"][
                    "elevation"
                ]["meters"]
                elevation_diff = round(
                    abs(asserted_elevation - geospatial_elevation), 1
                )

                coord_sources["elevation_comparison"] = {
                    "asserted_elevation_m": asserted_elevation,
                    "geospatial_elevation_m": geospatial_elevation,
                    "difference_m": elevation_diff,
                    "description": f"Elevation difference: {elevation_diff}m between asserted ({asserted_elevation}m) and geospatial lookup ({geospatial_elevation}m)",
                }

            inferred["coordinate_sources"] = coord_sources

        # Publication analysis (independent of coordinates)
        publications = get_publication_analysis(biosample_id, email)
        if publications:
            # Extract study_info and all_dois to be siblings, not nested under publication_analysis
            study_info = publications.pop("study_info", None)
            all_dois = publications.pop("_all_dois", None)

            if study_info:
                inferred["study_info"] = study_info
            if all_dois:
                inferred["all_dois"] = all_dois
            inferred["publication_analysis"] = publications

        return {"asserted": full_biosample, "inferred": inferred}

    except Exception:
        return None


def load_biosample_ids(file_path: str, sample_size: Optional[int] = None) -> List[str]:
    """Load biosample IDs from file with optional sampling."""
    try:
        with open(file_path, "r") as f:
            all_ids = [line.strip() for line in f if line.strip()]

        if sample_size is not None and sample_size < len(all_ids):
            return random.sample(all_ids, sample_size)
        return all_ids
    except Exception:
        return []


@click.command()
@click.option("--biosample-id", help="Single NMDC biosample ID")
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    help="File with biosample IDs (one per line)",
)
@click.option("--sample-size", type=int, help="Random sample size from input file")
@click.option(
    "--output-file",
    type=click.Path(),
    help="Output YAML file (single file for all results)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Output directory (individual YAML files per biosample)",
)
@click.option(
    "--email",
    required=True,
    help="Email address for API requests (required for full text fetching)",
)
@click.option(
    "--search-radius",
    type=int,
    default=1000,
    help="Search radius in meters for geospatial features (default: 1000)",
)
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.option(
    "--capture-output/--no-capture-output",
    default=True,
    help="Capture stdout/stderr to log file (default: True)",
)
def main(
    biosample_id: str,
    input_file: str,
    sample_size: int,
    output_file: str,
    output_dir: str,
    email: str,
    search_radius: int,
    verbose: bool,
    capture_output: bool,
) -> None:
    """crawl-first: Deterministic biosample enrichment for LLM-ready data preparation."""

    # Setup logging
    logger, output_capture = setup_logging(verbose, capture_output)

    # Validate input options
    if input_file and biosample_id:
        logger.error("Cannot specify both --biosample-id and --input-file")
        raise click.ClickException(
            "Cannot specify both --biosample-id and --input-file"
        )
    elif not input_file and not biosample_id:
        logger.error("Must specify either --biosample-id or --input-file")
        raise click.ClickException("Must specify either --biosample-id or --input-file")

    # Validate output options
    if output_file and output_dir:
        logger.error("Cannot specify both --output-file and --output-dir")
        raise click.ClickException("Cannot specify both --output-file and --output-dir")
    elif not output_file and not output_dir:
        logger.error("Must specify either --output-file or --output-dir")
        raise click.ClickException("Must specify either --output-file or --output-dir")

    # Get biosample IDs
    if input_file:
        logger.info(f"Loading biosample IDs from {input_file}")
        biosample_ids = load_biosample_ids(input_file, sample_size)
        if not biosample_ids:
            logger.error("No biosample IDs to process")
            raise click.ClickException("No biosample IDs to process")
        logger.info(f"Loaded {len(biosample_ids)} biosample IDs for processing")
    else:
        biosample_ids = [biosample_id]
        logger.info(f"Processing single biosample: {biosample_id}")

    # Process biosamples
    results = {}
    failed = []

    # Create output directory if using directory mode
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_path}")

    for i, bid in enumerate(biosample_ids):
        logger.info(f"Processing {i+1}/{len(biosample_ids)}: {bid}")

        result = analyze_biosample(bid, email, search_radius)
        if result:
            if output_dir:
                # Write individual YAML file immediately
                filename = bid.replace(":", "_") + ".yaml"
                file_path = output_path / filename
                try:
                    with open(file_path, "w") as f:
                        yaml.dump(
                            result,
                            f,
                            Dumper=NoRefsDumper,
                            default_flow_style=False,
                            sort_keys=False,
                            indent=2,
                        )
                    logger.debug(f"Saved: {file_path}")
                except Exception as e:
                    logger.error(f"Error saving {file_path}: {e}")
                    failed.append(bid)
                    continue
            else:
                # Collect for batch output (single file mode)
                results[bid] = result
        else:
            logger.warning(f"Failed to analyze biosample: {bid}")
            failed.append(bid)

    # Report summary
    if len(biosample_ids) > 1:
        if output_dir:
            successful_count = len(biosample_ids) - len(failed)
            logger.info(
                f"Processed {successful_count}/{len(biosample_ids)} biosamples successfully"
            )
            logger.info(f"Files saved to: {output_dir}")
        else:
            logger.info(
                f"Processed {len(results)}/{len(biosample_ids)} biosamples successfully"
            )

        if failed:
            logger.warning(f"Failed biosamples: {', '.join(failed)}")

    # Output results (only for single file mode since directory mode saves immediately)
    if output_file and results:
        output_data = list(results.values())[0] if len(results) == 1 else results
        with open(output_file, "w") as f:
            yaml.dump(
                output_data,
                f,
                Dumper=NoRefsDumper,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
            )
        logger.info(f"Data saved to: {output_file}")
    elif results and not output_file and not output_dir:
        # Simple display for single biosample
        if len(results) == 1:
            result = list(results.values())[0]
            asserted = result["asserted"]
            inferred = result["inferred"]

            logger.info(f"Biosample: {asserted.get('id')}")

            # Show coordinate sources
            coord_sources = inferred.get("coordinate_sources", {})
            if "from_asserted_coords" in coord_sources:
                coords = coord_sources["from_asserted_coords"]["coordinates"]
                logger.info(
                    f"Asserted coordinates: {coords['latitude']}, {coords['longitude']}"
                )

            if "from_geo_loc_name" in coord_sources:
                coords = coord_sources["from_geo_loc_name"]["coordinates"]
                source = coord_sources["from_geo_loc_name"]["source"]
                logger.info(
                    f"Geocoded coordinates: {coords['latitude']}, {coords['longitude']} ({source})"
                )

            # Show soil analysis
            for soil_key in ["soil_from_asserted_coords", "soil_from_geo_loc_name"]:
                if soil_key in inferred:
                    soil_type = inferred[soil_key].get("soil_type")
                    if soil_type:
                        source = (
                            "asserted coords"
                            if "asserted" in soil_key
                            else "geo_loc_name"
                        )
                        logger.info(f"Soil ({source}): {soil_type}")

            # Show publication info
            if "publication_analysis" in inferred:
                pub_data = inferred["publication_analysis"]
                logger.info(
                    f"Study DOIs: {pub_data.get('total_dois', 0)} total, {pub_data.get('publication_doi_count', 0)} publications"
                )

            # Show geospatial info from both sources
            for geo_key in [
                "geospatial_from_asserted_coords",
                "geospatial_from_geo_loc_name",
            ]:
                if geo_key in inferred:
                    geo_data = inferred[geo_key]
                    source = (
                        "asserted coords" if "asserted" in geo_key else "geo_loc_name"
                    )

                    elevation = geo_data.get("elevation")
                    if elevation:
                        logger.info(f"Elevation ({source}): {elevation.get('meters')}m")

                    place_info = geo_data.get("place_info", {})
                    if place_info.get("display_name"):
                        logger.info(
                            f"Location ({source}): {place_info['display_name']}"
                        )

                    env_summary = geo_data.get("environmental_summary", {})
                    total_features = env_summary.get("total_environmental_features", 0)
                    if total_features > 0:
                        logger.info(
                            f"Environmental features ({source}): {total_features} within {search_radius}m"
                        )

    if not results:
        logger.error("No data processed successfully")

    # Cleanup output capture
    if output_capture:
        output_capture.stop()


if __name__ == "__main__":
    main()

"""
Caching utilities for crawl-first.

Handles MD5-based caching for API calls and computations.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeVar

T = TypeVar("T")

# Global cache directory
CACHE_DIR = Path(".cache")
FULL_TEXT_DIR = Path(".cache/full_text_files")


def cache_key(data: Dict[str, Any]) -> str:
    """Generate MD5 cache key from data."""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


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

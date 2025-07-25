"""
Caching utilities for crawl-first.

Handles MD5-based caching for API calls and computations.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeVar

from .logging_utils import (
    log_cache_operation,
    log_enhanced_error,
    log_memory_usage,
    timed_operation,
)

T = TypeVar("T")

# Global cache directory - project root
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"
FULL_TEXT_DIR = CACHE_DIR / "full_text_files"


def cache_key(data: Dict[str, Any]) -> str:
    """Generate MD5 cache key from data."""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def generate_pdf_filename(identifiers: Dict[str, str]) -> str:
    """Generate a suggested PDF filename using the same convention as save_pdf_to_file.

    Args:
        identifiers: Dictionary with doi, pmid, pmcid for filename generation

    Returns:
        Suggested filename for PDF (e.g., "doi_10_1029_2022JG006889_pmid_12345.pdf")
    """
    # Generate filename from identifiers (same logic as save_pdf_to_file)
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

    # Ensure filename_parts is not empty and does not contain only empty strings
    if not filename_parts or all(not part for part in filename_parts):
        filename_parts = ["default"]

    filename = (
        "_".join(filename_parts[:2]) + ".pdf"
    )  # Limit to 2 parts to avoid overly long names

    return filename


def get_cache(cache_type: str, key: str) -> Optional[Dict[str, Any]]:
    """Get data from cache."""
    logger = logging.getLogger("crawl_first.cache")
    cache_file = CACHE_DIR / cache_type / f"{key}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                # Ensure we return a dict or None
                result = data if isinstance(data, dict) else None

            if result is not None:
                log_cache_operation(cache_type, "hit", key, logger)
                return result
            else:
                log_cache_operation(cache_type, "miss", key, logger)
                return None

        except Exception as e:
            # Safe key slicing to avoid IndexError
            key_display = key[:50] if len(key) > 50 else key
            log_enhanced_error(
                logger,
                e,
                f"reading cache {cache_type}",
                {"key": key_display, "file": str(cache_file)},
            )
            log_cache_operation(cache_type, "miss", key, logger)
            return None
    else:
        log_cache_operation(cache_type, "miss", key, logger)
        return None


@timed_operation("cache_save", include_args=False)
def save_cache(cache_type: str, key: str, data: Dict[str, Any]) -> None:
    """Save data to cache."""
    logger = logging.getLogger("crawl_first.cache")
    cache_dir = CACHE_DIR / cache_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"

    try:
        serialized_data = json.dumps(data, indent=2)
        with open(cache_file, "w") as f:
            f.write(serialized_data)

        # Log successful cache write with estimated file size
        # Using string length as approximation to avoid encoding overhead
        file_size_estimate = len(serialized_data)
        logger.debug(
            f"Cached {cache_type} data: ~{file_size_estimate} chars (key: {key[:50]}...)"
        )

    except Exception as e:
        # Safe key slicing to avoid IndexError
        key_display = key[:50] if len(key) > 50 else key
        log_enhanced_error(
            logger,
            e,
            f"writing cache {cache_type}",
            {"key": key_display, "file": str(cache_file), "data_size": len(str(data))},
        )


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


@timed_operation("pdf_file_save", include_args=False)
def save_pdf_to_file(content: bytes, identifiers: Dict[str, str]) -> Optional[str]:
    """
    Save PDF content to a file and return the file path.

    Args:
        content: PDF content as bytes to save
        identifiers: Dictionary with doi, pmid, pmcid for filename generation

    Returns:
        Relative file path or None if failed
    """
    logger = logging.getLogger("crawl_first.cache")

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

        # Ensure filename_parts is not empty or contains only empty strings
        if not any(filename_parts):
            filename_parts = ["default"]
        filename = (
            "_".join(filename_parts[:2]) + ".pdf"
        )  # Limit to 2 parts to avoid overly long names
        filepath = FULL_TEXT_DIR / filename

        # Write PDF content to file
        with open(filepath, "wb") as f:
            f.write(content)

        # Return relative path (handle absolute paths safely)
        try:
            relative_path = str(filepath.relative_to(Path.cwd()))
        except ValueError:
            # Fallback to absolute path if relative path fails
            relative_path = str(filepath)
        logger.debug(f"Saved PDF to file: {relative_path} ({len(content)} bytes)")
        return relative_path

    except Exception as e:
        logger.error(f"Failed to save PDF to file: {e}")
        return None


@timed_operation("full_text_file_save", include_args=False)
def save_full_text_to_file(content: str, identifiers: Dict[str, str]) -> Optional[str]:
    """
    Save full text content to a file and return the file path.

    Args:
        content: Full text content to save
        identifiers: Dictionary with doi, pmid, pmcid for filename generation

    Returns:
        Relative file path or None if failed
    """
    logger = logging.getLogger("crawl_first.cache")

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

        # Log successful file creation with size and identifiers
        file_size = filepath.stat().st_size
        content_size = len(content)
        logger.info(
            f"Saved full text to file: {file_size} bytes "
            f"(content: {content_size} chars, identifiers: {identifiers})"
        )
        log_memory_usage(logger, "full text file save")

        # Return relative path from current directory (handle absolute paths safely)
        try:
            return str(filepath.relative_to(Path.cwd()))
        except ValueError:
            # Fallback to absolute path if relative path fails
            return str(filepath)

    except Exception as e:
        log_enhanced_error(
            logger,
            e,
            "saving full text to file",
            {"identifiers": str(identifiers), "content_length": len(content)},
        )
        return None

"""
Analysis modules for crawl-first.

Handles various types of data analysis: soil, land cover, weather, publications, etc.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import yaml
from artl_mcp.tools import (
    doi_to_pmid,
    extract_paper_info,
    get_abstract_from_pubmed_id,
    get_doi_metadata,
    get_doi_text,
    get_full_text_from_doi,
    get_full_text_info,
    get_pmcid_text,
    get_pmid_text,
    get_unpaywall_info,
)
from landuse_mcp.main import get_land_cover, get_landuse_dates, get_soil_type
from nmdc_mcp.api import (
    fetch_nmdc_entity_by_id,
    fetch_nmdc_entity_by_id_with_projection,
)
from nmdc_mcp.tools import get_study_doi_details, get_study_for_biosample
from ols_mcp.tools import search_all_ontologies
from weather_mcp.main import get_weather

from .cache import (
    cache_key,
    get_cache,
    get_cached_entity,
    get_cached_results,
    save_cache,
    save_full_text_to_file,
)

# Constants for text processing
LINE_LENGTH_THRESHOLD = 80
YAML_WIDTH_LIMIT = 80
YAML_INDENT_SIZE = 2


class FullTextResult(NamedTuple):
    """Result from full text retrieval attempts."""

    text: Optional[str]
    method: Optional[str]
    pdf_url: Optional[str]


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


def get_weather_analysis(lat: float, lon: float, date: str) -> Optional[Dict[str, Any]]:
    """Get weather data with YAML-safe formatting using exact date and standard parameters."""
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

    # Single attempt with standard parameters - no tricks or fallbacks
    try:
        weather_data = get_weather(
            lat=lat,
            lon=lon,
            date=parsed_date,
            search_radius_km=150,  # Standard radius
            timeseries_type="daily",
            coverage_threshold=0.5,  # Standard coverage requirement
            measurement_units="scientific",
        )

        if weather_data:
            # Clean for YAML serialization
            cleaned = {
                "coverage": weather_data.get("coverage"),
                "search_radius_km": 150,
                "coverage_threshold": 0.5,
                "date_used": parsed_date,
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
        else:
            # No fallbacks - if it fails with standard parameters, cache and return None
            save_cache("weather", key, {"result": None})
            return None

    except Exception:
        # No retries with different parameters - if it fails, cache and return None
        save_cache("weather", key, {"result": None})
        return None


def _try_retrieve_text(
    retrieve_func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Generic function to retrieve and validate text.

    Args:
        retrieve_func: The function to call for text retrieval
        *args: Positional arguments to pass to retrieve_func
        **kwargs: Keyword arguments to pass to retrieve_func

    Returns:
        Dictionary with text and length if successful, None otherwise
    """
    try:
        text = retrieve_func(*args, **kwargs)
        if text and len(text.strip()) > 100:
            return {"text": text, "length": len(text.strip())}
    except Exception:
        pass
    return None


def _try_pmcid_text(pmcid: str) -> Optional[Dict[str, Any]]:
    """Try to retrieve full text using PMCID."""
    return _try_retrieve_text(get_pmcid_text, pmcid)


def _try_pmid_text(pmid: str) -> Optional[Dict[str, Any]]:
    """Try to retrieve full text using PMID."""
    return _try_retrieve_text(get_pmid_text, pmid)


def _try_doi_simple_text(doi: str) -> Optional[Dict[str, Any]]:
    """Try to retrieve full text using simple DOI method."""
    return _try_retrieve_text(get_doi_text, doi)


def _try_doi_advanced_text(doi: str, email: str) -> Optional[Dict[str, Any]]:
    """Try to retrieve full text using advanced DOI method."""
    return _try_retrieve_text(get_full_text_from_doi, doi, email)


def _try_unpaywall_pdf_url(doi: str, email: str) -> Optional[Dict[str, Any]]:
    """Try to retrieve PDF URL via Unpaywall for direct download."""
    try:
        unpaywall_info = get_unpaywall_info(doi, email)
        if unpaywall_info and unpaywall_info.get("is_oa"):
            oa_locations = unpaywall_info.get("oa_locations", [])
            for location in oa_locations:
                pdf_url = location.get("url_for_pdf")
                if pdf_url:
                    return {
                        "pdf_url": pdf_url,
                        "type": "pdf_download",
                        "source": "unpaywall",
                    }
    except Exception:
        pass
    return None


def _try_doi_fetcher_pdf_url(doi: str, email: str) -> Optional[Dict[str, Any]]:
    """Try to retrieve PDF URL via DOIFetcher for direct download."""
    try:
        full_text_info = get_full_text_info(doi, email)
        if full_text_info and full_text_info.get("pdf_url"):
            return {
                "pdf_url": full_text_info["pdf_url"],
                "type": "pdf_download",
                "source": "doi_fetcher",
            }
    except Exception:
        pass
    return None


def _structure_text_as_yaml(text: str, source: str) -> str:
    """Parse and structure extracted text into YAML format with section detection."""

    # Parse the text into structured data
    lines = text.strip().split("\n")
    structured_data: Dict[str, Any] = {
        "source": source,
        "content_type": "native_text",
        "sections": [],
    }

    current_section: Dict[str, Any] = {"title": "Main Content", "paragraphs": []}
    current_paragraph: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                current_section["paragraphs"].append(" ".join(current_paragraph))
                current_paragraph = []
        else:
            # Simple heuristic: lines that are short and end with certain patterns might be headers
            if len(line) < LINE_LENGTH_THRESHOLD and (
                line.isupper()
                or line.endswith(":")
                or line.startswith(
                    (
                        "Abstract",
                        "Introduction",
                        "Methods",
                        "Results",
                        "Discussion",
                        "Conclusion",
                    )
                )
            ):
                # Save current section if it has content
                if current_section["paragraphs"]:
                    structured_data["sections"].append(current_section)
                # Start new section
                current_section = {"title": line, "paragraphs": []}
                current_paragraph = []
            else:
                current_paragraph.append(line)

    # Add final paragraph and section
    if current_paragraph:
        current_section["paragraphs"].append(" ".join(current_paragraph))
    if current_section["paragraphs"]:
        structured_data["sections"].append(current_section)

    return yaml.dump(
        structured_data,
        default_flow_style=False,
        allow_unicode=True,
        width=YAML_WIDTH_LIMIT,
        indent=YAML_INDENT_SIZE,
    )


def _attempt_full_text_retrieval(
    paper_metadata: Dict[str, Any], email: str
) -> Dict[str, Dict[str, Any]]:
    """Attempt to retrieve full text using prioritized strategies: PDF URLs first, then native text."""
    attempts = {}
    pmcid = paper_metadata.get("pmcid")
    pmid = paper_metadata.get("pmid")
    doi = paper_metadata.get("doi")

    # FIRST PRIORITY: PDF URL Discovery
    if doi:
        # Try Unpaywall for PDF URLs
        result = _try_unpaywall_pdf_url(doi, email)
        if result:
            attempts["unpaywall_pdf_url"] = result

        # Try DOIFetcher for PDF URLs (may find different sources)
        result = _try_doi_fetcher_pdf_url(doi, email)
        if result:
            attempts["doi_fetcher_pdf_url"] = result

    # SECOND PRIORITY: Native Text (BioC XML from PubMed Central)
    if pmcid:
        result = _try_pmcid_text(pmcid)
        if result:
            # Convert to YAML format
            result["text"] = _structure_text_as_yaml(result["text"], "pmcid_bioc")
            result["format"] = "yaml"
            attempts["pmcid_native_text"] = result

    if pmid:
        result = _try_pmid_text(pmid)
        if result:
            # Convert to YAML format
            result["text"] = _structure_text_as_yaml(result["text"], "pmid_bioc")
            result["format"] = "yaml"
            attempts["pmid_native_text"] = result

    if doi:
        result = _try_doi_simple_text(doi)
        if result:
            # Convert to YAML format
            result["text"] = _structure_text_as_yaml(result["text"], "doi_bioc")
            result["format"] = "yaml"
            attempts["doi_native_text"] = result

    # AVOID: PDF extraction methods entirely
    # PDF extraction methods are avoided due to performance concerns and reliability issues.
    # Extracting text from PDFs can be computationally expensive and error-prone, especially
    # when dealing with complex layouts or scanned documents. Instead, we prioritize
    # structured data sources like BioC XML or metadata APIs to ensure more accurate and
    # efficient text retrieval.

    return attempts


def _select_best_result(
    attempts: Dict[str, Dict[str, Any]],
) -> FullTextResult:
    """Select the best result from attempts, preferring PDF downloads."""
    if not attempts:
        return FullTextResult(text=None, method=None, pdf_url=None)

    # Check for PDF download first (highest priority)
    for method, result in attempts.items():
        if result.get("type") == "pdf_download":
            return FullTextResult(
                text=None, method=method, pdf_url=result.get("pdf_url")
            )

    # Fall back to text retrieval - select the longest text
    text_attempts = {k: v for k, v in attempts.items() if v.get("text")}
    if text_attempts:
        best_method = max(
            text_attempts.keys(), key=lambda k: text_attempts[k]["length"]
        )
        return FullTextResult(
            text=text_attempts[best_method]["text"],
            method=best_method,
            pdf_url=None,
        )

    return FullTextResult(text=None, method=None, pdf_url=None)


def _build_full_text_result(
    identifiers: Dict[str, str],
    full_text: Optional[str],
    retrieval_method: Optional[str],
    file_path: Optional[str],
    pdf_url: Optional[str],
    attempts: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the full text result dictionary."""
    result = {
        "method": retrieval_method,
        "identifiers": identifiers,
        "file_path": file_path,
        "content_length": len(full_text) if full_text else 0,
        "methods_attempted": list(attempts.keys()) if attempts else [],
        "method_results": ({k: v for k, v in attempts.items()} if attempts else {}),
    }

    # Set status based on what we have
    if pdf_url:
        result["status"] = "pdf_available"
        result["pdf_url"] = pdf_url
    elif full_text and file_path:
        result["status"] = "retrieved"
    elif full_text:
        result["status"] = "partial"
    else:
        result["status"] = "not_available"

    return result


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

    try:
        # Attempt to retrieve full text using multiple strategies
        attempts = _attempt_full_text_retrieval(paper_metadata, email)

        # Select the best result (preferring PDF downloads)
        best_result = _select_best_result(attempts)
        full_text, retrieval_method, pdf_url = (
            best_result.text,
            best_result.method,
            best_result.pdf_url,
        )

        # Save content to file if retrieved
        file_path = None
        if full_text and len(full_text.strip()) > 100:
            file_path = save_full_text_to_file(full_text, identifiers)

        # Build result
        result = _build_full_text_result(
            identifiers, full_text, retrieval_method, file_path, pdf_url, attempts
        )

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


def _get_study_info(
    study_id: str, doi_details: Dict[str, Any], study_data: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build study information dictionary."""
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

    return study_info


def _get_paper_metadata(clean_doi: str, pmid: Optional[str]) -> Dict[str, Any]:
    """Get paper metadata from CrossRef and PubMed."""
    paper_metadata: Dict[str, Any] = {
        "doi": clean_doi,
        "pmid": pmid,
        "title": "",
        "authors": [],
        "journal": "",
        "citation_count": None,
        "abstract": "",
        "retrieval_errors": [],
    }

    # Get metadata from CrossRef
    try:
        doi_metadata = get_doi_metadata(clean_doi)
        if doi_metadata and doi_metadata.get("status") == "ok":
            work_item = doi_metadata.get("message", {})
            if work_item:
                paper_info = extract_paper_info(work_item)
                paper_metadata.update(paper_info)
    except Exception as e:
        retrieval_errors = paper_metadata["retrieval_errors"]
        assert isinstance(retrieval_errors, list)  # Help mypy understand this is a list
        retrieval_errors.append(f"CrossRef error: {e}")

    # Get abstract from PubMed if needed
    if pmid and not paper_metadata.get("abstract"):
        try:
            abstract = get_abstract_from_pubmed_id(pmid)
            if abstract and abstract.strip():
                paper_metadata["abstract"] = abstract
        except Exception as e:
            retrieval_errors = paper_metadata["retrieval_errors"]
            assert isinstance(
                retrieval_errors, list
            )  # Help mypy understand this is a list
            retrieval_errors.append(f"PubMed error: {e}")

    return paper_metadata


def _add_full_text_info(
    paper_metadata: Dict[str, Any], clean_doi: str, pmid: Optional[str], email: str
) -> None:
    """Add full text information to paper metadata."""
    full_text_metadata = {
        "doi": clean_doi,
        "pmid": pmid,
        "pmcid": None,
    }

    full_text_result = cached_get_full_text(full_text_metadata, email)
    if full_text_result:
        paper_metadata["full_text"] = {
            "status": full_text_result.get("status"),
            "file_path": full_text_result.get("file_path"),
            "content_length": full_text_result.get("content_length"),
            "retrieval_method": full_text_result.get("method"),
        }

        if full_text_result.get("pdf_url"):
            paper_metadata["full_text"]["pdf_url"] = full_text_result.get("pdf_url")
    else:
        paper_metadata["full_text"] = {
            "status": "not_available",
            "file_path": None,
            "content_length": 0,
            "retrieval_method": None,
        }


def _process_publication_doi(doi_entry: Dict[str, Any], email: str) -> Dict[str, Any]:
    """Process a publication DOI to get detailed metadata."""
    doi_value = doi_entry.get("doi_value", "")
    enhanced_doi = {**doi_entry}

    clean_doi = doi_value.replace("doi:", "").strip()
    enhanced_doi["doi_url"] = f"https://doi.org/{clean_doi}"

    # Get PMID
    pmid = None
    try:
        pmid = doi_to_pmid(clean_doi)
    except Exception:
        pass

    # Get paper metadata
    paper_metadata = _get_paper_metadata(clean_doi, pmid)

    # Add full text information
    _add_full_text_info(paper_metadata, clean_doi, pmid, email)

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

    return enhanced_doi


def _process_dois(
    all_dois: List[Dict[str, Any]], email: str
) -> tuple[List[Dict[str, Any]], int]:
    """Process all DOIs and return enhanced DOIs with publication count."""
    enhanced_dois = []
    publication_doi_count = 0

    for doi_entry in all_dois:
        doi_value = doi_entry.get("doi_value", "")
        doi_category = doi_entry.get("doi_category", "")

        if doi_value:
            if doi_category == "publication_doi":
                publication_doi_count += 1
                enhanced_doi = _process_publication_doi(doi_entry, email)
            else:
                # For non-publication DOIs, just add the URL
                enhanced_doi = {**doi_entry}
                clean_doi = doi_value.replace("doi:", "").strip()
                enhanced_doi["doi_url"] = f"https://doi.org/{clean_doi}"

            enhanced_dois.append(enhanced_doi)

    return enhanced_dois, publication_doi_count


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
        study_info = _get_study_info(study_id, doi_details, study_data)

        # Process DOIs
        all_dois = (
            study_data.get("associated_dois", [])
            if study_data
            else doi_details.get("associated_dois", [])
        )
        enhanced_dois, publication_doi_count = _process_dois(all_dois, email)

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

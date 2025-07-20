"""
Direct text retrieval functions for crawl-first.

Implements direct API calls to bypass artl-mcp issues while maximizing text retrieval.
Based on patterns from artl-mcp but with proper email handling and error resilience.
"""

import hashlib
import json
import logging
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# Import caching utilities
from .cache import cache_key, get_cache, save_cache

logger = logging.getLogger(__name__)

# Module-level constants
USER_AGENT = "crawl-first/1.0 (+https://github.com/contextualizer-ai/crawl-first)"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
CHUNK_SIZE = 8192

# Windows reserved filenames
WINDOWS_RESERVED_FILENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}

# Trusted domains for downloads
TRUSTED_DOMAINS = {
    "www.ebi.ac.uk",
    "europepmc.org",
    "ftp.ncbi.nlm.nih.gov",
    "www.ncbi.nlm.nih.gov",
    "eutils.ncbi.nlm.nih.gov",
    "api.unpaywall.org",
    "api.crossref.org",
}

# Default headers for HTTP requests
DEFAULT_HEADERS = {"User-Agent": USER_AGENT}

# API endpoints from artl-mcp and enhanced sources
BIOC_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmid}/ascii"
PUBMED_EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=xml"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
EUROPE_PMC_HTML_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextHTML"
)
EUROPE_PMC_XML_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
)
EUROPE_PMC_SUPP_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"
)
PMC_FTP_OA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/{dir1}/{dir2}/{pmcid}.tar.gz"
)
DOI_PATTERN = r"/(10\.\d{4,}/[\w\-.]+)"


def create_supplementary_files_directory() -> Path:
    """Create and return the supplementary files directory path."""
    supp_dir = Path("cache/supplementary_files")
    supp_dir.mkdir(parents=True, exist_ok=True)
    return supp_dir


def download_supplementary_file(file_info: Dict[str, Any], pmcid: str) -> Optional[str]:
    """Download a supplementary file and return the local file path."""
    logger.debug(
        f"Downloading supplementary file: {file_info.get('filename', 'unknown')} for {pmcid}"
    )

    download_url = file_info.get("download_url", "")
    filename = file_info.get("filename", "")

    if not download_url or not filename:
        logger.warning(
            f"Missing download URL or filename for supplementary file: {file_info}"
        )
        return None

    # Validate the download URL to prevent SSRF vulnerabilities
    try:
        parsed_url = urlparse(download_url)
        if parsed_url.netloc not in TRUSTED_DOMAINS:
            logger.warning(f"Untrusted domain in download URL: {parsed_url.netloc}")
            return None
    except Exception as e:
        logger.warning(f"Invalid download URL format: {download_url} - {e}")
        return None

    try:
        # Create supplementary files directory
        supp_dir = create_supplementary_files_directory()

        # Clean filename for filesystem safety using pathlib
        safe_filename = Path(filename).name  # Extract the name part of the filename
        safe_filename = "".join(
            c if c.isalnum() or c in "._-" else "_" for c in safe_filename
        )
        if not safe_filename or safe_filename in WINDOWS_RESERVED_FILENAMES:
            url_hash = hashlib.sha256(download_url.encode()).hexdigest()[:16]
            safe_filename = f"supplement_{pmcid}_{url_hash}"

        # Add PMCID prefix to avoid conflicts
        final_filename = f"{pmcid}_{safe_filename}"
        file_path = supp_dir / final_filename

        # Skip if file already exists
        if file_path.exists():
            logger.debug(f"Supplementary file already exists: {file_path}")
            return str(file_path.relative_to(Path.cwd()))

        # Download the file with User-Agent header
        headers = DEFAULT_HEADERS
        response = requests.get(download_url, timeout=30, stream=True, headers=headers)
        response.raise_for_status()

        # Save to file with size limit to prevent disk space exhaustion
        total_downloaded = 0
        try:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:  # Filter out keep-alive chunks
                        total_downloaded += len(chunk)
                        if total_downloaded > MAX_FILE_SIZE:
                            logger.warning(
                                f"File size exceeds the maximum limit of {MAX_FILE_SIZE} bytes. Aborting download."
                            )
                            file_path.unlink(
                                missing_ok=True
                            )  # Remove partially downloaded file
                            return None
                        f.write(chunk)
        except Exception as e:
            logger.error(
                f"An error occurred during file download: {e}. "
                f"Context: file_path={file_path}, download_url={download_url}, pmcid={pmcid}"
            )
            file_path.unlink(missing_ok=True)  # Ensure cleanup on unexpected errors
            raise

        file_size = file_path.stat().st_size
        logger.info(
            f"Downloaded supplementary file: {filename} ({file_size} bytes) to {file_path}"
        )

        # Return relative path
        return str(file_path.relative_to(Path.cwd()))

    except requests.RequestException as e:
        logger.warning(f"Network error downloading supplementary file {filename}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error downloading supplementary file {filename}: {e}")
        return None


def save_supplementary_files(
    file_list: List[Dict[str, Any]], pmcid: str
) -> List[Dict[str, Any]]:
    """Download and save supplementary files, return updated file list with local paths."""
    if not file_list:
        return []

    logger.info(
        f"Starting download of {len(file_list)} supplementary files for {pmcid}"
    )
    updated_files = []

    for file_info in file_list:
        updated_file = file_info.copy()

        # Download the file
        local_path = download_supplementary_file(file_info, pmcid)
        if local_path:
            updated_file["local_path"] = local_path
            updated_file["download_status"] = "success"
        else:
            updated_file["download_status"] = "failed"

        updated_files.append(updated_file)

    successful_downloads = len(
        [f for f in updated_files if f.get("download_status") == "success"]
    )
    logger.info(
        f"Downloaded {successful_downloads}/{len(file_list)} supplementary files for {pmcid}"
    )

    return updated_files


def safe_get(element: Any, attr: str, default: str = "") -> str:
    """Safely get an attribute from a BeautifulSoup element."""
    try:
        if hasattr(element, "get"):
            return element.get(attr, default) or default
        return default
    except Exception:
        return default


def extract_doi_from_url(url: str) -> Optional[str]:
    """Extract DOI from a given journal URL."""
    doi_match = re.search(DOI_PATTERN, url)
    return doi_match.group(1) if doi_match else None


def _fetch_field_from_doi(doi: str, field: str, timeout: int = 10) -> Optional[str]:
    """Fetch a specific field (e.g., 'pmid' or 'pmcid') from the NCBI ID Converter API."""
    try:
        api_url = (
            f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={doi}&format=json"
        )
        headers = DEFAULT_HEADERS
        response = requests.get(api_url, timeout=timeout, headers=headers)
        response.raise_for_status()
        data = response.json()

        records = data.get("records", [])
        field_value = records[0].get(field, None) if records else None
        return field_value
    except requests.RequestException as e:
        logger.warning(f"Network error fetching {field} for DOI {doi}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error for DOI {doi}: {e}")
        return None
    except (KeyError, IndexError) as e:
        logger.debug(f"Missing {field} field for DOI {doi}: {e}")
        return None


def doi_to_pmid(doi: str) -> Optional[str]:
    """Convert DOI to PMID using NCBI ID Converter API."""
    return _fetch_field_from_doi(doi, "pmid")


def doi_to_pmcid(doi: str) -> Optional[str]:
    """Convert DOI to PMCID using NCBI ID Converter API."""
    return _fetch_field_from_doi(doi, "pmcid")


def pmid_to_doi(pmid: str) -> Optional[str]:
    """Convert PMID to DOI using PubMed E-utilities."""
    try:
        # Ensure pmid is a string
        pmid = str(pmid)
        if ":" in pmid:
            pmid = pmid.split(":")[1]

        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
        headers = DEFAULT_HEADERS
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        data = response.json()

        article_info = data["result"][str(pmid)]
        for aid in article_info["articleids"]:
            if aid["idtype"] == "doi":
                return str(aid["value"])

        elocationid = article_info.get("elocationid", "")
        if elocationid.startswith("10."):  # DOI starts with "10."
            return str(elocationid)

        return None
    except requests.RequestException as e:
        logger.warning(f"Network error converting PMID {pmid} to DOI: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error for PMID {pmid}: {e}")
        return None
    except (KeyError, IndexError) as e:
        logger.debug(f"No DOI found for PMID {pmid}: {e}")
        return None


def get_pmid_from_pmcid(pmcid: str) -> Optional[str]:
    """Get PMID from PMC ID using Entrez E-utilities."""
    try:
        if ":" in pmcid:
            pmcid = pmcid.split(":")[1]

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {"db": "pmc", "id": pmcid.replace("PMC", ""), "retmode": "json"}
        headers = DEFAULT_HEADERS

        response = requests.get(url, params=params, timeout=10, headers=headers)
        response.raise_for_status()
        data = response.json()

        uid = data["result"]["uids"][0]
        article_ids = data["result"][uid]["articleids"]
        for item in article_ids:
            if item["idtype"] == "pmid":
                return str(item["value"])

        return None
    except requests.RequestException as e:
        logger.warning(f"Network error converting PMCID {pmcid} to PMID: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error for PMCID {pmcid}: {e}")
        return None
    except (KeyError, IndexError) as e:
        logger.debug(f"No PMID found for PMCID {pmcid}: {e}")
        return None


def get_unpaywall_info(doi: str, email: str) -> Optional[Dict[str, Any]]:
    """Get Unpaywall information with proper email handling."""
    try:
        base_url = f"https://api.unpaywall.org/v2/{doi}"
        headers = DEFAULT_HEADERS
        response = requests.get(
            f"{base_url}?email={email}", timeout=10, headers=headers
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.warning(f"Error fetching Unpaywall info for DOI {doi}: {e}")
        return None


def get_crossref_metadata(doi: str) -> Optional[Dict[str, Any]]:
    """Get metadata from CrossRef API."""
    try:
        base_url = "https://api.crossref.org/works/"
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        response = requests.get(f"{base_url}{doi}", headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "message" in data:
            message = data["message"]
            return message if isinstance(message, dict) else None
        return None
    except requests.RequestException as e:
        logger.warning(f"Network error fetching CrossRef metadata for DOI {doi}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error for CrossRef DOI {doi}: {e}")
        return None
    except (KeyError, IndexError) as e:
        logger.debug(f"Missing data in CrossRef response for DOI {doi}: {e}")
        return None


def get_bioc_xml_text(pmid: str) -> Optional[str]:
    """Get full text from PubMed Central BioC XML - raw format."""
    try:
        # Ensure pmid is a string
        pmid = str(pmid)
        if ":" in pmid:
            pmid = pmid.split(":")[1]

        headers = DEFAULT_HEADERS
        response = requests.get(BIOC_URL.format(pmid=pmid), timeout=10, headers=headers)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "xml")

        # Extract ONLY text from <text> tags within <passage>
        text_tags = soup.find_all("text")
        if not text_tags:
            logger.debug(f"No valid text tags found in BioC XML for PMID {pmid}")
            return None

        text_sections = []
        for text_tag in text_tags:
            if hasattr(text_tag, "get_text"):
                text_content = text_tag.get_text()
                if text_content and isinstance(text_content, str):
                    text_sections.append(text_content)

        full_text = "\n".join(text_sections).strip()
        return full_text if full_text else None
    except requests.RequestException as e:
        logger.warning(f"Network error fetching BioC XML for PMID {pmid}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error parsing BioC XML for PMID {pmid}: {e}")
        return None


def get_pubmed_abstract(pmid: str) -> Optional[str]:
    """Get title and abstract from PubMed - raw format."""
    try:
        # Ensure pmid is a string
        pmid = str(pmid)
        if ":" in pmid:
            pmid = pmid.split(":")[1]

        headers = DEFAULT_HEADERS
        response = requests.get(
            EFETCH_URL.format(pmid=pmid), timeout=10, headers=headers
        )

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "xml")

        # Extract title
        title_tag = soup.find("ArticleTitle")
        title = title_tag.get_text().strip() if title_tag else "No title available"

        # Extract abstract (may contain multiple sections)
        abstract_tags = soup.find_all("AbstractText")
        abstract = (
            "\n".join(tag.get_text().strip() for tag in abstract_tags)
            if abstract_tags
            else "No abstract available"
        )

        # Normalize whitespace but preserve structure
        title = re.sub(r"[^\S\n]", " ", title)
        title = re.sub(r" +", " ", title).strip()

        abstract = re.sub(r"[^\S\n]", " ", abstract)
        abstract = re.sub(r" +", " ", abstract).strip()

        return f"{title}\n\n{abstract}\n\nPMID:{pmid}"
    except requests.RequestException as e:
        logger.warning(f"Network error fetching PubMed abstract for PMID {pmid}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error parsing PubMed abstract for PMID {pmid}: {e}")
        return None


def download_pdf_from_url(pdf_url: str) -> Optional[bytes]:
    """Download PDF content from URL with basic validation."""
    try:
        headers = DEFAULT_HEADERS
        response = requests.get(pdf_url, timeout=30, stream=True, headers=headers)
        response.raise_for_status()

        # Check if content appears to be PDF (flexible validation)
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
            # Check the magic bytes for PDF
            first_bytes = response.raw.read(4)
            if not first_bytes.startswith(b"%PDF"):
                return None

        return response.content
    except requests.RequestException as e:
        logger.warning(f"Network error downloading PDF from {pdf_url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error downloading PDF from {pdf_url}: {e}")
        return None


def get_text_from_doi_direct(doi: str, email: str) -> Optional[Dict[str, Any]]:
    """
    Comprehensive text retrieval from DOI using multiple direct methods.
    Retrieves text and PDF content and returns it in a dictionary for the caller to handle.
    """
    result: Dict[str, Any] = {
        "doi": doi,
        "text": None,
        "source": None,
        "pdf_url": None,
        "pdf_content": None,
        "length": 0,
        "methods_tried": [],
    }

    # Try to get PMID and PMCID
    pmid = doi_to_pmid(doi)
    pmcid = doi_to_pmcid(doi)

    result["methods_tried"].append("doi_to_pmid")
    if pmid:
        result["pmid"] = pmid
    if pmcid:
        result["pmcid"] = pmcid

    # 1. Try BioC XML if we have PMID - this is full text, should not go in cache
    if pmid:
        result["methods_tried"].append("bioc_xml")
        text = get_bioc_xml_text(pmid)
        if text and len(text.strip()) > 100:
            # BioC XML is full text - mark for file storage, not cache
            result["text"] = text
            result["source"] = "bioc_xml_full_text"
            result["length"] = len(text.strip())
            result["is_full_text"] = True
            return result

    # 2. Try Unpaywall for PDF URLs and download them
    result["methods_tried"].append("unpaywall")
    unpaywall_info = get_unpaywall_info(doi, email)
    if unpaywall_info and unpaywall_info.get("is_oa"):
        oa_locations = unpaywall_info.get("oa_locations", [])
        best_oa = unpaywall_info.get("best_oa_location")
        if best_oa:
            oa_locations = [best_oa] + oa_locations

        for location in oa_locations:
            pdf_url = location.get("url_for_pdf")
            if pdf_url:
                result["pdf_url"] = pdf_url
                # Actually download the PDF
                pdf_content = download_pdf_from_url(pdf_url)
                if pdf_content:
                    result["pdf_content"] = pdf_content
                    result["source"] = "unpaywall_pdf"
                    result["length"] = len(pdf_content)
                    return result

    # 3. Try PubMed abstract as fallback
    if pmid:
        result["methods_tried"].append("pubmed_abstract")
        abstract = get_pubmed_abstract(pmid)
        if abstract and len(abstract.strip()) > 50:
            result["text"] = abstract
            result["source"] = "pubmed_abstract"
            result["length"] = len(abstract.strip())
            return result

    return result


def get_text_from_pmid_direct(pmid: str) -> Optional[Dict[str, Any]]:
    """
    Get text from PMID using direct methods.
    Returns raw text without YAML conversion.
    """
    result: Dict[str, Any] = {
        "pmid": pmid,
        "text": None,
        "source": None,
        "length": 0,
        "methods_tried": [],
    }

    # 1. Try BioC XML first - this is full text, should not go in cache
    result["methods_tried"].append("bioc_xml")
    text = get_bioc_xml_text(pmid)
    if text and len(text.strip()) > 100:
        result["text"] = text
        result["source"] = "bioc_xml_full_text"
        result["length"] = len(text.strip())
        result["is_full_text"] = True
        return result

    # 2. Fallback to abstract
    result["methods_tried"].append("pubmed_abstract")
    abstract = get_pubmed_abstract(pmid)
    if abstract and len(abstract.strip()) > 50:
        result["text"] = abstract
        result["source"] = "pubmed_abstract"
        result["length"] = len(abstract.strip())
        return result

    return result


def get_europe_pmc_full_text_html(pmcid: str) -> Optional[str]:
    """Get full text from Europe PMC as HTML, then convert to text."""
    try:
        # Ensure pmcid is properly formatted
        if ":" in pmcid:
            pmcid = pmcid.split(":")[1]
        if not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        url = EUROPE_PMC_HTML_URL.format(pmcid=pmcid)
        headers = DEFAULT_HEADERS
        response = requests.get(url, timeout=15, headers=headers)

        if response.status_code != 200:
            logger.debug(
                f"Europe PMC HTML API returned {response.status_code} for {pmcid}"
            )
            return None

        # Parse HTML and extract text
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it up
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text if len(text.strip()) > 500 else None

    except requests.RequestException as e:
        logger.warning(f"Network error fetching Europe PMC HTML for {pmcid}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error parsing Europe PMC HTML for {pmcid}: {e}")
        return None


def get_europe_pmc_full_text_xml(pmcid: str) -> Optional[str]:
    """Get full text from Europe PMC as JATS XML."""
    try:
        # Ensure pmcid is properly formatted
        if ":" in pmcid:
            pmcid = pmcid.split(":")[1]
        if not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        url = EUROPE_PMC_XML_URL.format(pmcid=pmcid)
        headers = DEFAULT_HEADERS
        response = requests.get(url, timeout=15, headers=headers)

        if response.status_code != 200:
            logger.debug(
                f"Europe PMC XML API returned {response.status_code} for {pmcid}"
            )
            return None

        # Parse JATS XML and extract text from body/article content
        soup = BeautifulSoup(response.text, "xml")

        # Extract text from main article sections, avoiding metadata
        text_elements = soup.find_all(["p", "sec", "title", "abstract", "body"])
        text_sections = []

        for element in text_elements:
            if element and hasattr(element, "get_text"):
                text_content = element.get_text().strip()
                if text_content and len(text_content) > 10:
                    text_sections.append(text_content)

        full_text = "\n\n".join(text_sections)
        return full_text if len(full_text.strip()) > 500 else None

    except requests.RequestException as e:
        logger.warning(f"Network error fetching Europe PMC XML for {pmcid}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error parsing Europe PMC XML for {pmcid}: {e}")
        return None


def get_pmc_efetch_xml(pmcid: str) -> Optional[str]:
    """Get full text from PMC using NCBI E-utilities EFetch with JATS XML."""
    try:
        # Ensure pmcid is properly formatted
        if ":" in pmcid:
            pmcid = pmcid.split(":")[1]
        if pmcid.startswith("PMC"):
            pmcid = pmcid[3:]  # Remove PMC prefix for E-utilities

        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&rettype=xml"
        headers = DEFAULT_HEADERS
        response = requests.get(url, timeout=15, headers=headers)

        if response.status_code != 200:
            logger.debug(f"PMC EFetch returned {response.status_code} for PMC{pmcid}")
            return None

        # Parse JATS/NLM XML
        soup = BeautifulSoup(response.text, "xml")

        # Extract text from article body sections
        text_elements = soup.find_all(["p", "sec", "title", "abstract", "body"])
        text_sections = []

        for element in text_elements:
            if element and hasattr(element, "get_text"):
                text_content = element.get_text().strip()
                if text_content and len(text_content) > 10:
                    text_sections.append(text_content)

        full_text = "\n\n".join(text_sections)
        return full_text if len(full_text.strip()) > 500 else None

    except requests.RequestException as e:
        logger.warning(f"Network error fetching PMC EFetch XML for PMC{pmcid}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error parsing PMC EFetch XML for PMC{pmcid}: {e}")
        return None


def get_europe_pmc_supplementary_files(pmcid: str) -> Optional[List[Dict[str, Any]]]:
    """Get supplementary files metadata from Europe PMC API with caching."""
    logger.debug(f"Starting Europe PMC supplementary files retrieval for: {pmcid}")

    # Ensure pmcid is properly formatted
    original_pmcid = pmcid
    if ":" in pmcid:
        pmcid = pmcid.split(":")[1]
    if not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    if original_pmcid != pmcid:
        logger.debug(f"Normalized PMCID from {original_pmcid} to {pmcid}")

    # Check cache first
    key = cache_key({"pmcid": pmcid, "type": "europe_pmc_supplements"})
    logger.debug(f"Cache key for Europe PMC supplements: {key}")
    cached = get_cache("europe_pmc_supplements", key)
    if cached and "files" in cached:
        logger.info(
            f"Cache HIT for Europe PMC supplementary files: {pmcid} ({len(cached['files'])} files)"
        )
        return cached["files"]  # type: ignore[no-any-return]

    logger.debug(f"Cache MISS for Europe PMC supplementary files: {pmcid}")

    try:
        url = EUROPE_PMC_SUPP_URL.format(pmcid=pmcid)
        logger.debug(f"Requesting Europe PMC supplements URL: {url}")
        headers = DEFAULT_HEADERS

        response = requests.get(url, timeout=10, headers=headers)
        logger.debug(
            f"Europe PMC supplements API response: {response.status_code} for {pmcid}"
        )

        if response.status_code != 200:
            logger.warning(
                f"Europe PMC supplementary files API returned {response.status_code} for {pmcid}"
            )
            # Cache negative result to avoid repeated failed requests
            save_cache("europe_pmc_supplements", key, {"files": []})
            return None

        data = response.json()
        logger.debug(f"Europe PMC supplements response data type: {type(data)}")

        # Parse supplementary files response
        if not isinstance(data, dict) or "supplementaryFiles" not in data:
            logger.warning(
                f"Invalid Europe PMC supplements response structure for {pmcid}"
            )
            save_cache("europe_pmc_supplements", key, {"files": []})
            return None

        supp_files = data["supplementaryFiles"]
        if not isinstance(supp_files, list):
            logger.warning(
                f"Europe PMC supplements 'supplementaryFiles' is not a list for {pmcid}"
            )
            save_cache("europe_pmc_supplements", key, {"files": []})
            return None

        logger.info(f"Found {len(supp_files)} supplementary files for {pmcid}")

        parsed_files = []
        for i, file_info in enumerate(supp_files):
            if isinstance(file_info, dict):
                parsed_file = {
                    "filename": file_info.get("fileName", ""),
                    "download_url": file_info.get("downloadUrl", ""),
                    "mime_type": file_info.get("mimeType", ""),
                    "caption": file_info.get("caption", ""),
                    "label": file_info.get("label", ""),
                    "size": file_info.get("size", 0),
                    "type": "supplementary",
                }
                parsed_files.append(parsed_file)
                logger.debug(
                    f"Parsed supplementary file {i+1}: {parsed_file['filename']} ({parsed_file['mime_type']})"
                )
            else:
                logger.warning(
                    f"Skipping non-dict supplementary file entry {i} for {pmcid}"
                )

        # Download the supplementary files
        result = parsed_files if parsed_files else []
        if result:
            logger.info(
                f"Downloading {len(result)} Europe PMC supplementary files for {pmcid}"
            )
            result = save_supplementary_files(result, pmcid)

        logger.info(f"Caching {len(result)} Europe PMC supplementary files for {pmcid}")
        save_cache("europe_pmc_supplements", key, {"files": result})

        logger.info(
            f"Successfully retrieved {len(result)} Europe PMC supplementary files for {pmcid}"
        )
        return result if result else None

    except requests.RequestException as e:
        logger.error(
            f"Network error fetching Europe PMC supplementary files for {pmcid}: {e}"
        )
        return None
    except json.JSONDecodeError as e:
        logger.error(
            f"JSON parsing error for Europe PMC supplementary files {pmcid}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error parsing Europe PMC supplementary files for {pmcid}: {e}"
        )
        return None


def get_pmc_oa_package(pmcid: str) -> Optional[Dict[str, Any]]:
    """
    Download and extract PMC Open Access package (.tar.gz) with full content.

    Returns complete article package including:
    - .nxml: Full-text JATS XML
    - .txt: ASCII text (if available)
    - .pdf: Author manuscript or final PDF
    - .jpg/.tif/.png: Figures
    - .supp: Supplementary files (zip, Excel, CSV, etc.)
    """
    logger.info(f"Starting PMC Open Access package retrieval for: {pmcid}")

    # Ensure pmcid is properly formatted
    original_pmcid = pmcid
    if ":" in pmcid:
        pmcid = pmcid.split(":")[1]
    if not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    if original_pmcid != pmcid:
        logger.debug(f"Normalized PMCID from {original_pmcid} to {pmcid}")

    # Check cache first (cache metadata, not full content due to size)
    key = cache_key({"pmcid": pmcid, "type": "pmc_oa_package"})
    logger.debug(f"Cache key for PMC OA package: {key}")
    cached = get_cache("pmc_oa_package", key)
    if cached and "result" in cached:
        logger.info(f"Cache HIT for PMC OA package: {pmcid}")
        return cached["result"]  # type: ignore[no-any-return]

    logger.debug(f"Cache MISS for PMC OA package: {pmcid}")

    try:

        # PMC OA FTP URL pattern - uses hierarchical directory structure
        pmc_num = pmcid[3:]  # Remove "PMC" prefix

        # Directory structure: first 2 digits, then next 2 digits (or remaining if < 4 total)
        if len(pmc_num) >= 4:
            dir1 = pmc_num[:2]
            dir2 = pmc_num[2:4]
        elif len(pmc_num) >= 2:
            dir1 = pmc_num[:2]
            dir2 = pmc_num[2:]
        else:
            dir1 = "00"
            dir2 = pmc_num

        ftp_url = PMC_FTP_OA_URL.format(dir1=dir1, dir2=dir2, pmcid=pmcid)
        logger.debug(
            f"Constructed PMC OA FTP URL: {ftp_url} (dir1={dir1}, dir2={dir2})"
        )

        # First check if package exists with HEAD request
        logger.debug(
            f"Checking PMC OA package availability with HEAD request for {pmcid}"
        )
        headers = DEFAULT_HEADERS
        head_response = requests.head(ftp_url, timeout=10, headers=headers)
        logger.debug(f"HEAD response status: {head_response.status_code} for {pmcid}")

        if head_response.status_code != 200:
            logger.warning(
                f"PMC OA package not available for {pmcid} (status: {head_response.status_code})"
            )
            # Cache negative result to avoid repeated failed requests
            save_cache("pmc_oa_package", key, {"result": None})
            return None

        logger.info(f"PMC OA package available, downloading from: {ftp_url}")
        headers = DEFAULT_HEADERS
        response = requests.get(ftp_url, timeout=60, stream=True, headers=headers)
        response.raise_for_status()

        content_length = response.headers.get("content-length")
        if content_length:
            logger.info(f"PMC OA package size: {int(content_length)} bytes for {pmcid}")
        else:
            logger.debug(f"Content-Length not available for PMC OA package {pmcid}")

        result: Dict[str, Any] = {
            "pmcid": pmcid,
            "package_url": ftp_url,
            "files": {},
            "file_count": 0,
            "total_size": 0,
            "text_content": None,
            "supplementary_files": [],
            "figures": [],
            "pdf_files": [],
            "source": "pmc_oa_package",
        }

        # Extract tar.gz in memory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save and extract tar.gz
            tar_path = temp_path / f"{pmcid}.tar.gz"
            with open(tar_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract contents
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(temp_path)

            # Process all extracted files
            for file_path in temp_path.rglob("*"):
                if file_path.is_file() and file_path.name != f"{pmcid}.tar.gz":
                    file_size = file_path.stat().st_size
                    result["total_size"] += file_size
                    result["file_count"] += 1

                    # Read file content based on type
                    file_ext = file_path.suffix.lower()
                    file_info = {
                        "filename": file_path.name,
                        "size": file_size,
                        "extension": file_ext,
                        "relative_path": str(file_path.relative_to(temp_path)),
                    }

                    if file_ext == ".nxml":
                        # Full-text JATS XML - extract text
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                xml_content = f.read()
                            soup = BeautifulSoup(xml_content, "xml")

                            # Parse supplementary material references from XML
                            supp_materials = soup.find_all("supplementary-material")
                            for supp in supp_materials:
                                href = safe_get(supp, "xlink:href") or safe_get(
                                    supp, "href"
                                )
                                if href:
                                    label_elem = (
                                        supp.find("label")
                                        if hasattr(supp, "find")
                                        else None
                                    )
                                    caption_elem = (
                                        supp.find("caption")
                                        if hasattr(supp, "find")
                                        else None
                                    )

                                    supp_info = {
                                        "filename": href,
                                        "label": (
                                            label_elem.get_text()
                                            if label_elem
                                            and hasattr(label_elem, "get_text")
                                            else ""
                                        ),
                                        "caption": (
                                            caption_elem.get_text()
                                            if caption_elem
                                            and hasattr(caption_elem, "get_text")
                                            else ""
                                        ),
                                        "mimetype": safe_get(supp, "mimetype", ""),
                                        "type": "supplementary_reference",
                                    }
                                    result["supplementary_files"].append(supp_info)

                            # Extract text from article body
                            text_elements = soup.find_all(
                                ["p", "sec", "title", "abstract", "body"]
                            )
                            text_sections = []
                            for element in text_elements:
                                if element and hasattr(element, "get_text"):
                                    text_content = element.get_text().strip()
                                    if text_content and len(text_content) > 10:
                                        text_sections.append(text_content)

                            extracted_text = "\n\n".join(text_sections)
                            if extracted_text and len(extracted_text.strip()) > 500:
                                result["text_content"] = extracted_text
                                file_info["extracted_text_length"] = len(extracted_text)

                        except Exception as e:
                            logger.warning(
                                f"Error parsing NXML file {file_path.name}: {e}"
                            )

                    elif file_ext == ".txt":
                        # ASCII text version - prefer this over extracted XML
                        try:
                            with open(
                                file_path, "r", encoding="utf-8", errors="replace"
                            ) as f:
                                txt_content = f.read().strip()
                            if txt_content and len(txt_content) > 500:
                                result["text_content"] = txt_content
                                file_info["text_length"] = len(txt_content)
                        except Exception as e:
                            logger.warning(
                                f"Error reading TXT file {file_path.name}: {e}"
                            )

                    elif file_ext == ".pdf":
                        # PDF file
                        file_info["type"] = "pdf"
                        result["pdf_files"].append(file_info)

                    elif file_ext in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".tif",
                        ".tiff",
                        ".gif",
                        ".svg",
                    ]:
                        # Figure files
                        file_info["type"] = "figure"
                        result["figures"].append(file_info)

                    elif (
                        file_ext
                        in [
                            ".zip",
                            ".xlsx",
                            ".xls",
                            ".csv",
                            ".tsv",
                            ".json",
                            ".xml",
                            ".supp",
                        ]
                        or "supp" in file_path.name.lower()
                    ):
                        # Supplementary files
                        file_info["type"] = "supplementary"
                        result["supplementary_files"].append(file_info)

                    result["files"][file_path.name] = file_info

        # Mark as full text if we got substantial content
        if result["text_content"] and len(result["text_content"]) > 500:
            result["is_full_text"] = True
            result["length"] = len(result["text_content"])

        logger.info(
            f"Successfully extracted PMC OA package for {pmcid}: {result['file_count']} files, {result['total_size']} bytes, {len(result['supplementary_files'])} supplementary files"
        )

        # Cache the result without full text content (only metadata)
        cache_result = result.copy()
        if "text_content" in cache_result:
            # Remove full text from cache, store only metadata
            del cache_result["text_content"]
            cache_result["text_content_length"] = len(result.get("text_content", ""))
            cache_result["has_full_text"] = bool(result.get("text_content"))
        save_cache("pmc_oa_package", key, {"result": cache_result})

        return result

    except requests.RequestException as e:
        logger.warning(f"Network error downloading PMC OA package for {pmcid}: {e}")
        # Cache negative result to avoid repeated failed downloads
        save_cache("pmc_oa_package", key, {"result": None})
        return None
    except tarfile.TarError as e:
        logger.warning(f"Error extracting PMC OA package for {pmcid}: {e}")
        save_cache("pmc_oa_package", key, {"result": None})
        return None
    except Exception as e:
        logger.warning(f"Error processing PMC OA package for {pmcid}: {e}")
        save_cache("pmc_oa_package", key, {"result": None})
        return None


def get_comprehensive_pmcid_package(pmcid: str) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive data for a PMCID including full text and supplementary files.
    Combines multiple methods: OA package, Europe PMC supplements, and text retrieval.
    """
    logger.info(f"Starting comprehensive PMCID package retrieval for: {pmcid}")

    # Ensure pmcid is properly formatted
    original_pmcid = pmcid
    if ":" in pmcid:
        pmcid = pmcid.split(":")[1]
    if not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    if original_pmcid != pmcid:
        logger.debug(f"Normalized PMCID from {original_pmcid} to {pmcid}")

    # Check cache first
    key = cache_key({"pmcid": pmcid, "type": "comprehensive_package"})
    logger.debug(f"Cache key for comprehensive PMCID package: {key}")
    cached = get_cache("comprehensive_pmcid", key)
    if cached and "result" in cached:
        logger.info(f"Cache HIT for comprehensive PMCID package: {pmcid}")
        return cached["result"]  # type: ignore[no-any-return]

    logger.debug(f"Cache MISS for comprehensive PMCID package: {pmcid}")

    result: Dict[str, Any] = {
        "pmcid": pmcid,
        "text": None,
        "source": None,
        "length": 0,
        "methods_tried": [],
        "supplementary_files": [],
        "figures": [],
        "pdf_files": [],
        "package_info": None,
    }

    # 1. Try PMC OA package first (most comprehensive)
    logger.debug(f"Step 1: Attempting PMC OA package retrieval for {pmcid}")
    result["methods_tried"].append("pmc_oa_package")
    oa_package = get_pmc_oa_package(pmcid)
    if oa_package:
        logger.info(
            f"PMC OA package found for {pmcid}: {oa_package.get('file_count', 0)} files, {len(oa_package.get('supplementary_files', []))} supplementary files"
        )
        result.update(
            {
                "text": oa_package.get("text_content"),
                "source": "pmc_oa_package_full_text",
                "length": oa_package.get("length", 0),
                "supplementary_files": oa_package.get("supplementary_files", []),
                "figures": oa_package.get("figures", []),
                "pdf_files": oa_package.get("pdf_files", []),
                "package_info": oa_package,
                "is_full_text": oa_package.get("is_full_text", False),
            }
        )

        # If we got full text from OA package, we're done
        if result["text"] and len(result["text"]) > 500:
            logger.info(
                f"Full text retrieved from PMC OA package for {pmcid}: {len(result['text'])} characters"
            )
            # Cache result without full text content (only metadata)
            cache_result = result.copy()
            if "text" in cache_result:
                # Remove full text from cache, store only metadata
                del cache_result["text"]
                cache_result["text_length"] = len(result.get("text", ""))
                cache_result["has_text"] = bool(result.get("text"))
            save_cache("comprehensive_pmcid", key, {"result": cache_result})
            return result
        else:
            logger.debug(
                f"PMC OA package for {pmcid} did not yield sufficient full text"
            )
    else:
        logger.debug(f"No PMC OA package available for {pmcid}")

    # 2. Try Europe PMC supplementary files API
    logger.debug(f"Step 2: Attempting Europe PMC supplementary files for {pmcid}")
    result["methods_tried"].append("europe_pmc_supplements")
    europe_supps = get_europe_pmc_supplementary_files(pmcid)
    if europe_supps:
        logger.info(
            f"Found {len(europe_supps)} Europe PMC supplementary files for {pmcid}"
        )
        # Files are already downloaded by get_europe_pmc_supplementary_files
        result["supplementary_files"].extend(europe_supps)
    else:
        logger.debug(f"No Europe PMC supplementary files found for {pmcid}")

    # 3. Fall back to standard PMCID text retrieval
    logger.debug(f"Step 3: Attempting standard PMCID text retrieval for {pmcid}")
    result["methods_tried"].append("standard_pmcid_text")
    text_result = get_text_from_pmcid_direct(pmcid)
    if text_result and text_result.get("text"):
        logger.info(
            f"Standard PMCID text retrieval successful for {pmcid}: {text_result['source']}, {len(text_result['text'])} characters"
        )
        result.update(
            {
                "text": text_result["text"],
                "source": text_result["source"],
                "length": text_result["length"],
                "is_full_text": text_result.get("is_full_text", False),
            }
        )
        result["methods_tried"].extend(text_result.get("methods_tried", []))
    else:
        logger.warning(f"Standard PMCID text retrieval failed for {pmcid}")

    # Cache the final result
    final_result = result if result["text"] or result["supplementary_files"] else None

    if final_result:
        logger.info(
            f"Comprehensive PMCID package completed for {pmcid}: text={'Yes' if final_result['text'] else 'No'}, {len(final_result['supplementary_files'])} supplementary files, {len(final_result['figures'])} figures, {len(final_result['pdf_files'])} PDFs"
        )
    else:
        logger.warning(
            f"Comprehensive PMCID package retrieval failed for {pmcid}: no text or supplementary files found"
        )

    # Cache result without full text content (only metadata)
    if final_result:
        cache_result = final_result.copy()
        if "text" in cache_result:
            # Remove full text from cache, store only metadata
            del cache_result["text"]
            cache_result["text_length"] = len(final_result.get("text", ""))
            cache_result["has_text"] = bool(final_result.get("text"))
        save_cache("comprehensive_pmcid", key, {"result": cache_result})
    else:
        save_cache("comprehensive_pmcid", key, {"result": None})
    return final_result


def get_text_from_pmcid_direct(pmcid: str) -> Optional[Dict[str, Any]]:
    """
    Get text from PMCID using multiple direct methods, prioritizing Europe PMC.
    Returns raw text without YAML conversion.
    """
    result: Dict[str, Any] = {
        "pmcid": pmcid,
        "text": None,
        "source": None,
        "length": 0,
        "methods_tried": [],
    }

    # 1. Try Europe PMC HTML first (best for clean full text)
    result["methods_tried"].append("europe_pmc_html")
    html_text = get_europe_pmc_full_text_html(pmcid)
    if html_text and len(html_text.strip()) > 500:
        result["text"] = html_text
        result["source"] = "europe_pmc_html_full_text"
        result["length"] = len(html_text.strip())
        result["is_full_text"] = True
        return result

    # 2. Try Europe PMC XML as fallback (JATS XML)
    result["methods_tried"].append("europe_pmc_xml")
    xml_text = get_europe_pmc_full_text_xml(pmcid)
    if xml_text and len(xml_text.strip()) > 500:
        result["text"] = xml_text
        result["source"] = "europe_pmc_xml_full_text"
        result["length"] = len(xml_text.strip())
        result["is_full_text"] = True
        return result

    # 3. Try PMC E-utilities EFetch XML (NLM/JATS XML)
    result["methods_tried"].append("pmc_efetch_xml")
    efetch_text = get_pmc_efetch_xml(pmcid)
    if efetch_text and len(efetch_text.strip()) > 500:
        result["text"] = efetch_text
        result["source"] = "pmc_efetch_xml_full_text"
        result["length"] = len(efetch_text.strip())
        result["is_full_text"] = True
        return result

    # 4. Convert PMCID to PMID and try PMID-based methods (fallback to abstract)
    result["methods_tried"].append("pmcid_to_pmid")
    pmid = get_pmid_from_pmcid(pmcid)
    if pmid:
        result["pmid"] = pmid
        # Use PMID-based retrieval (may return abstract)
        pmid_result = get_text_from_pmid_direct(pmid)
        if pmid_result and pmid_result.get("text"):
            result["text"] = pmid_result["text"]
            result["source"] = pmid_result["source"]
            result["length"] = pmid_result["length"]
            result["methods_tried"].extend(pmid_result["methods_tried"])
            if pmid_result.get("is_full_text"):
                result["is_full_text"] = True
            return result

    return result

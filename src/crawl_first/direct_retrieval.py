"""
Direct text retrieval functions for crawl-first.

Implements direct API calls to bypass artl-mcp issues while maximizing text retrieval.
Based on patterns from artl-mcp but with proper email handling and error resilience.
"""

import re
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup

# API endpoints from artl-mcp
BIOC_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmid}/ascii"
PUBMED_EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=xml"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
DOI_PATTERN = r"/(10\.\d{4,9}/[\w\-.]+)"


def extract_doi_from_url(url: str) -> Optional[str]:
    """Extract DOI from a given journal URL."""
    doi_match = re.search(DOI_PATTERN, url)
    return doi_match.group(1) if doi_match else None


def doi_to_pmid(doi: str) -> Optional[str]:
    """Convert DOI to PMID using NCBI ID Converter API."""
    try:
        api_url = (
            f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={doi}&format=json"
        )
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()

        records = data.get("records", [])
        pmid = records[0].get("pmid", None) if records else None
        return pmid
    except Exception:
        return None


def doi_to_pmcid(doi: str) -> Optional[str]:
    """Convert DOI to PMCID using NCBI ID Converter API."""
    try:
        api_url = (
            f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={doi}&format=json"
        )
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()

        records = data.get("records", [])
        pmcid = records[0].get("pmcid", None) if records else None
        return pmcid
    except Exception:
        return None


def pmid_to_doi(pmid: str) -> Optional[str]:
    """Convert PMID to DOI using PubMed E-utilities."""
    try:
        if ":" in pmid:
            pmid = pmid.split(":")[1]

        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
        response = requests.get(url, timeout=10)
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
    except Exception:
        return None


def get_pmid_from_pmcid(pmcid: str) -> Optional[str]:
    """Get PMID from PMC ID using Entrez E-utilities."""
    try:
        if ":" in pmcid:
            pmcid = pmcid.split(":")[1]

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {"db": "pmc", "id": pmcid.replace("PMC", ""), "retmode": "json"}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        uid = data["result"]["uids"][0]
        article_ids = data["result"][uid]["articleids"]
        for item in article_ids:
            if item["idtype"] == "pmid":
                return str(item["value"])

        return None
    except Exception:
        return None


def get_unpaywall_info(doi: str, email: str) -> Optional[Dict[str, Any]]:
    """Get Unpaywall information with proper email handling."""
    try:
        base_url = f"https://api.unpaywall.org/v2/{doi}"
        response = requests.get(f"{base_url}?email={email}", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def get_crossref_metadata(doi: str) -> Optional[Dict[str, Any]]:
    """Get metadata from CrossRef API."""
    try:
        base_url = "https://api.crossref.org/works/"
        headers = {
            "User-Agent": "crawl-first/1.0",
            "Accept": "application/json",
        }
        response = requests.get(f"{base_url}{doi}", headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "message" in data:
            message = data["message"]
            return message if isinstance(message, dict) else None
        return None
    except Exception:
        return None


def get_bioc_xml_text(pmid: str) -> Optional[str]:
    """Get full text from PubMed Central BioC XML - raw format."""
    try:
        if ":" in pmid:
            pmid = pmid.split(":")[1]

        response = requests.get(BIOC_URL.format(pmid=pmid), timeout=10)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "xml")

        # Extract ONLY text from <text> tags within <passage>
        text_sections = [text_tag.get_text() for text_tag in soup.find_all("text")]

        full_text = "\n".join(text_sections).strip()
        return full_text if full_text else None
    except Exception:
        return None


def get_pubmed_abstract(pmid: str) -> Optional[str]:
    """Get title and abstract from PubMed - raw format."""
    try:
        if ":" in pmid:
            pmid = pmid.split(":")[1]

        response = requests.get(EFETCH_URL.format(pmid=pmid), timeout=10)

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
    except Exception:
        return None


def download_pdf_from_url(pdf_url: str) -> Optional[bytes]:
    """Download PDF content from URL."""
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def get_text_from_doi_direct(doi: str, email: str) -> Optional[Dict[str, Any]]:
    """
    Comprehensive text retrieval from DOI using multiple direct methods.
    Downloads PDFs and saves text content as files.
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

    # 1. Try BioC XML if we have PMID
    if pmid:
        result["methods_tried"].append("bioc_xml")
        text = get_bioc_xml_text(pmid)
        if text and len(text.strip()) > 100:
            result["text"] = text
            result["source"] = "bioc_xml"
            result["length"] = len(text.strip())
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

    # 1. Try BioC XML first
    methods_tried = result["methods_tried"]
    assert isinstance(methods_tried, list)
    methods_tried.append("bioc_xml")
    text = get_bioc_xml_text(pmid)
    if text and len(text.strip()) > 100:
        result["text"] = text
        result["source"] = "bioc_xml"
        result["length"] = len(text.strip())
        return result

    # 2. Fallback to abstract
    methods_tried = result["methods_tried"]
    assert isinstance(methods_tried, list)
    methods_tried.append("pubmed_abstract")
    abstract = get_pubmed_abstract(pmid)
    if abstract and len(abstract.strip()) > 50:
        result["text"] = abstract
        result["source"] = "pubmed_abstract"
        result["length"] = len(abstract.strip())
        return result

    return result


def get_text_from_pmcid_direct(pmcid: str) -> Optional[Dict[str, Any]]:
    """
    Get text from PMCID using direct methods.
    Returns raw text without YAML conversion.
    """
    result: Dict[str, Any] = {
        "pmcid": pmcid,
        "text": None,
        "source": None,
        "length": 0,
        "methods_tried": [],
    }

    # Convert PMCID to PMID
    methods_tried = result["methods_tried"]
    assert isinstance(methods_tried, list)
    methods_tried.append("pmcid_to_pmid")
    pmid = get_pmid_from_pmcid(pmcid)
    if pmid:
        result["pmid"] = pmid
        # Use PMID-based retrieval
        pmid_result = get_text_from_pmid_direct(pmid)
        if pmid_result and pmid_result.get("text"):
            result["text"] = pmid_result["text"]
            result["source"] = pmid_result["source"]
            result["length"] = pmid_result["length"]
            methods_tried = result["methods_tried"]
            pmid_methods_tried = pmid_result["methods_tried"]
            assert isinstance(methods_tried, list)
            assert isinstance(pmid_methods_tried, list)
            methods_tried.extend(pmid_methods_tried)
            return result

    return result

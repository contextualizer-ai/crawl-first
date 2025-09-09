#!/usr/bin/env python3
"""
Download and setup local TEOW 2017 ecoregions for offline point-in-polygon lookup.

This replaces the unreliable WWF ArcGIS FeatureServer with a local dataset.
Based on official RESOLVE/TEOW 2017 data with CC-BY 4.0 license.
"""

import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import requests
from shapely.geometry import Point

# Official TEOW 2017 URLs (primary + fallbacks)
TEOW_URLS = [
    "https://storage.googleapis.com/teow2016/Ecoregions2017.zip",  # Official RESOLVE GCS
    # Additional fallbacks could be added here
]

DEST = "../data/teow/Ecoregions2017.zip"


def fetch_teow(urls=TEOW_URLS, dest=DEST, timeout=120) -> str:
    """
    Download TEOW 2017 shapefile with robust fallbacks.

    Args:
        urls: List of download URLs to try
        dest: Destination path for zip file
        timeout: Request timeout in seconds

    Returns:
        Path to downloaded zip file
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Check if already downloaded (>10MB indicates valid file)
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000_000:
        print(f"TEOW 2017 already downloaded: {dest}")
        return dest

    print("Downloading TEOW 2017 ecoregions...")
    last_err = None

    for i, url in enumerate(urls):
        try:
            print(f"Trying source {i+1}/{len(urls)}: {url}")

            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()

                # Download with progress
                total_size = int(r.headers.get("content-length", 0))
                downloaded = 0
                tmp = dest + ".tmp"

                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(
                                    f"\\rProgress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)",
                                    end="",
                                )

                print("\\nDownload complete!")
                os.replace(tmp, dest)  # Atomic rename
                return dest

        except Exception as e:
            print(f"\\nFailed: {e}")
            last_err = e

    raise RuntimeError(
        f"Failed to download TEOW from all sources. Last error: {last_err}"
    )


def teow_read(zip_path=DEST) -> Tuple[gpd.GeoDataFrame, Dict[str, str]]:
    """
    Read TEOW shapefile from zip with schema-tolerant column mapping.

    Args:
        zip_path: Path to TEOW zip file

    Returns:
        Tuple of (GeoDataFrame, column_mapping_dict)
    """
    # Find the .shp file inside the zip automatically
    with zipfile.ZipFile(zip_path) as z:
        shp_files = [n for n in z.namelist() if n.lower().endswith(".shp")]
        if not shp_files:
            raise RuntimeError("No .shp found in TEOW zip")
        shp_name = shp_files[0]

    # Use zip:// protocol for direct reading
    gdf = gpd.read_file(f"zip://{os.path.abspath(zip_path)}!{shp_name}")

    print(f"Loaded {len(gdf)} ecoregions")
    print(f"Available columns: {list(gdf.columns)}")

    # Schema-tolerant column mapping
    cols = {c.lower(): c for c in gdf.columns}

    def pick_column(*candidates):
        """Find first matching column from candidates."""
        for candidate in candidates:
            if candidate in cols:
                return cols[candidate]
        return None

    column_mapping = {
        "eco_name": pick_column(
            "eco_name", "ecoregion", "name", "terrestrial_ecoregions"
        ),
        "biome": pick_column("biome_name", "biome", "biome_num"),
        "realm": pick_column("realm", "realm_name"),
        "eco_id": pick_column("eco_id", "ecoregion_id", "id", "objectid"),
    }

    print(f"Column mapping: {column_mapping}")
    return gdf, column_mapping


def teow_lookup_point(
    lon: float, lat: float, teow_gdf: gpd.GeoDataFrame, teow_cols: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Fast point-in-polygon lookup for ecoregion data.

    Args:
        lon: Longitude
        lat: Latitude
        teow_gdf: TEOW GeoDataFrame with spatial index
        teow_cols: Column mapping dict

    Returns:
        Dict with ecoregion info or None if not found
    """
    pt = Point(lon, lat)

    # Use spatial index for fast candidate selection
    candidate_indices = list(teow_gdf.sindex.intersection(pt.bounds))
    if not candidate_indices:
        return None

    # Check candidates for actual containment
    candidates = teow_gdf.iloc[candidate_indices]
    hits = candidates[candidates.geometry.contains(pt)]

    if hits.empty:
        # Fallback: try touches() for boundary cases
        hits = candidates[candidates.geometry.touches(pt)]
        if hits.empty:
            return None

    # Return first hit
    row = hits.iloc[0]

    def get_column_value(key: str):
        col_name = teow_cols.get(key)
        return None if col_name is None else row[col_name]

    return {
        "ecoregion_name": get_column_value("eco_name"),
        "biome_name": get_column_value("biome"),
        "realm": get_column_value("realm"),
        "eco_id": get_column_value("eco_id"),
        "data_source": "TEOW 2017 (RESOLVE)",
        "license": "CC-BY 4.0",
        "source_url": "https://ecoregions.appspot.com/",
        "success": True,
    }


def setup_local_ecoregions() -> Optional[Tuple[gpd.GeoDataFrame, Dict[str, str]]]:
    """
    One-time setup: Download and prepare TEOW 2017 for fast lookups.

    Returns:
        Tuple of (GeoDataFrame, column_mapping) or None if failed
    """
    try:
        # Check if GeoPackage already exists for faster loading
        gpkg_path = Path(DEST).parent / "teow2017.gpkg"
        if gpkg_path.exists():
            try:
                print("Loading TEOW 2017 from GeoPackage...")
                teow_gdf = gpd.read_file(gpkg_path)

                # Rebuild column mapping for GeoPackage
                cols = {c.lower(): c for c in teow_gdf.columns}

                def pick_column(*candidates):
                    for candidate in candidates:
                        if candidate in cols:
                            return cols[candidate]
                    return None

                teow_cols = {
                    "eco_name": pick_column(
                        "eco_name", "ecoregion", "name", "terrestrial_ecoregions"
                    ),
                    "biome": pick_column("biome_name", "biome", "biome_num"),
                    "realm": pick_column("realm", "realm_name"),
                    "eco_id": pick_column("eco_id", "ecoregion_id", "id", "objectid"),
                }

                # Build spatial index
                _ = teow_gdf.sindex
                print(f"✅ Loaded {len(teow_gdf)} ecoregions from GeoPackage")
                return teow_gdf, teow_cols

            except Exception as e:
                print(f"⚠️ Failed to load GeoPackage: {e}")
                print("Falling back to ZIP download...")

        # Download if needed
        zip_path = fetch_teow()

        # Read and prepare
        teow_gdf, teow_cols = teow_read(zip_path)

        # Build spatial index
        _ = teow_gdf.sindex
        print("✅ Spatial index built for fast lookups")

        # Convert to GeoPackage for faster loading
        gpkg_path = Path(DEST).parent / "teow2017.gpkg"
        if not gpkg_path.exists():
            print("Converting to GeoPackage for faster future loading...")
            teow_gdf.to_file(gpkg_path, driver="GPKG")

            # Generate MD5 for verification
            import hashlib

            with open(gpkg_path, "rb") as f:
                md5_hash = hashlib.md5(f.read()).hexdigest()

            md5_path = gpkg_path.with_suffix(".gpkg.md5")
            with open(md5_path, "w") as f:
                f.write(f"{md5_hash}  {gpkg_path.name}\n")

            print(f"✅ GeoPackage saved: {gpkg_path}")
            print(f"✅ MD5 checksum: {md5_hash}")
        else:
            # Verify existing GeoPackage
            md5_path = gpkg_path.with_suffix(".gpkg.md5")
            if md5_path.exists():
                try:
                    import hashlib

                    with open(gpkg_path, "rb") as f:
                        current_md5 = hashlib.md5(f.read()).hexdigest()

                    with open(md5_path) as f:
                        stored_md5 = f.read().split()[0]

                    if current_md5 == stored_md5:
                        print(f"✅ GeoPackage verified: {gpkg_path}")
                    else:
                        print("⚠️ GeoPackage MD5 mismatch - will regenerate")
                        gpkg_path.unlink()
                        return setup_local_ecoregions()  # Retry
                except Exception as e:
                    print(f"⚠️ MD5 verification failed: {e}")
            else:
                print(f"✅ Using existing GeoPackage: {gpkg_path}")

        return teow_gdf, teow_cols

    except Exception as e:
        print(f"❌ Failed to setup TEOW 2017: {e}")
        return None


if __name__ == "__main__":
    # Test the setup
    result = setup_local_ecoregions()
    if result:
        teow_gdf, teow_cols = result

        # Test lookup with Yellowstone coordinates
        test_lon, test_lat = -110.5885, 44.4280
        info = teow_lookup_point(test_lon, test_lat, teow_gdf, teow_cols)
        print(f"\\nTest lookup ({test_lat}, {test_lon}): {info}")
    else:
        print("Setup failed!")

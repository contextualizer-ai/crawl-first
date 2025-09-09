"""
SoilGrids client (WCS-first, REST-optional)

- Robust WCS 2.0.1 -> 1.0.0 fallback
- 3x3 pixel sampling window around a lat/lon
- WRB "MostProbable" classification with numeric->label mapping
- Property fetch (e.g., phh2o, clay, sand, etc.) at depth & stat
- Reports the distance (meters) from the requested coordinate to the
  center of the pixel actually sampled

Tested against ISRIC SoilGrids WCS endpoints documented here:
  - https://www.isric.org/web-coverage-services-wcs
  - Service base: https://maps.isric.org/mapserv?map=/map/{service}.map

Notes
-----
* The SoilGrids REST endpoints (rest.soilgrids.org / rest.isric.org) are
  known to be flaky. This module does not rely on them. If you still want
  to try them, set USE_REST=True and ensure DNS resolves.
* Dependencies: requests, rasterio, numpy
* No GDAL command-line tools required; rasterio reads the GeoTIFF bytes in-memory.

Mark-friendly code style: inline comments, no emojis, no step numbers.
"""
from __future__ import annotations

import io
import math
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import rasterio
from rasterio.io import MemoryFile

# -----------------------------
# Constants and simple helpers
# -----------------------------
WCS_BASE = "https://maps.isric.org/mapserv?map=/map/{service}.map"
DEFAULT_TIMEOUT = 25  # seconds per HTTP request

# WRB numeric code -> label mapping (from ISRIC Data Hub metadata)
WRB_CODE_TO_LABEL: Dict[int, str] = {
    0: "Acrisols",
    1: "Albeluvisols",
    2: "Alisols",
    3: "Andosols",
    4: "Arenosols",
    5: "Calcisols",
    6: "Cambisols",
    7: "Chernozems",
    8: "Cryosols",
    9: "Durisols",
    10: "Ferralsols",
    11: "Fluvisols",
    12: "Gleysols",
    13: "Gypsisols",
    14: "Histosols",
    15: "Kastanozems",
    16: "Leptosols",
    17: "Lixisols",
    18: "Luvisols",
    19: "Nitisols",
    20: "Phaeozems",
    21: "Planosols",
    22: "Plinthosols",
    23: "Podzols",
    24: "Regosols",
    25: "Solonchaks",
    26: "Solonetz",
    27: "Stagnosols",
    28: "Umbrisols",
    29: "Vertisols",
}

# Property services exposed by SoilGrids (v2.0) via WCS
# See: https://soilgrids.readthedocs.io/ and ISRIC WCS docs
SOILGRIDS_SERVICES = {
    "bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o",
    "sand", "silt", "soc", "ocs", "ocd", "wrb"
}

# Units mapping for SoilGrids properties based on empirical analysis and soil science standards
# Raw values from WCS are typically scaled integers that need interpretation
# Documentation sources: SoilGrids technical specs (files.isric.org), soil science conventions
SOILGRIDS_UNITS = {
    "phh2o": "pH",             # pH in water (raw value 55 = pH 5.5, scaled by 10)
    "clay": "g/kg",            # Clay content (raw value 73 = 7.3%, values in g/kg)
    "sand": "g/kg",            # Sand content (raw value 486 = 48.6%, values in g/kg)  
    "silt": "g/kg",            # Silt content (raw value 376 = 37.6%, values in g/kg)
    "soc": "dg/kg",            # Soil organic carbon (raw value 366 = 36.6 dg/kg)
    "bdod": "cg/cm³",          # Bulk density (raw value 106 = 1.06 g/cm³, scaled by 100)
    "nitrogen": "cg/kg",       # Total nitrogen (raw value 257 = 2.57 g/kg, scaled by 100)
    "cec": "mmol(c)/kg",       # Cation exchange capacity
    "cfvo": "cm³/dm³",         # Coarse fragments volume
    "ocs": "t/ha",             # Organic carbon stock
    "ocd": "m"                 # Organic carbon density
}

# Scaling factors to convert raw WCS values to actual units
# Based on empirical analysis: pH 5.5 shows as 55, sand 48.6% shows as 486, etc.
SOILGRIDS_SCALING = {
    "phh2o": 0.1,      # pH scaled by 10
    "clay": 0.1,       # g/kg scaled by 10 (percentage conversion)
    "sand": 0.1,       # g/kg scaled by 10 
    "silt": 0.1,       # g/kg scaled by 10
    "soc": 0.1,        # dg/kg scaled by 10
    "bdod": 0.01,      # g/cm³ scaled by 100
    "nitrogen": 0.01,  # g/kg scaled by 100
}

# Standard depth intervals and typical stats (quantiles + mean)
STANDARD_DEPTHS = [
    "0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"
]
STANDARD_STATS = ["Q0.05", "Q0.5", "Q0.95", "mean"]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _deg_window_for_pixels(lat: float, pixels: int = 3, pixel_m: float = 250.0) -> Tuple[float, float]:
    """Approximate lat/lon degree window for a given pixel window.

    Assumes ~250 m native resolution; converts meters to degrees at latitude.
    Returns (dlat, dlon) as half-window in degrees (so that bbox = +/- dlat/dlon).
    """
    # meters per degree latitude ~ 110.574 km
    dlat = (pixels * pixel_m) / 110_574.0  # degrees
    # meters per degree longitude ~ 111.320 km * cos(latitude)
    m_per_deg_lon = 111_320.0 * max(0.0001, math.cos(math.radians(lat)))
    dlon = (pixels * pixel_m) / m_per_deg_lon
    return dlat, dlon


@dataclass
class WCSResult:
    array: np.ndarray           # 2D array
    transform: rasterio.Affine  # affine geotransform
    crs: str                    # CRS string (expected EPSG:4326)
    pixel_center_lat: float
    pixel_center_lon: float
    distance_m: float


class SoilGridsWCSClient:
    """Minimal WCS client tailored for SoilGrids on maps.isric.org.

    It tries a WCS 2.0.1 request first (SUBSET=lat/long) then falls back to WCS 1.0.0 (BBOX + WIDTH/HEIGHT).
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT, session: Optional[requests.Session] = None):
        self.timeout = timeout
        self.session = session or requests.Session()

    # -----------------
    # Core WCS methods
    # -----------------
    def _get_wcs_201(self, *, service: str, coverage_id: str, lat: float, lon: float,
                     pixels: int = 3) -> Optional[bytes]:
        dlat, dlon = _deg_window_for_pixels(lat, pixels=pixels)
        south, north = lat - dlat / 2, lat + dlat / 2
        west, east = lon - dlon / 2, lon + dlon / 2

        params = {
            "SERVICE": "WCS",
            "VERSION": "2.0.1",
            "REQUEST": "GetCoverage",
            "COVERAGEID": coverage_id,
            "FORMAT": "image/tiff",
            # Subset on geographic coords; ISRIC accepts axis labels `lat`/`long`
            "SUBSET": [f"lat({south},{north})", f"long({west},{east})"],
            "SUBSETTINGCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
            "OUTPUTCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
        }
        # requests can't encode duplicate keys with lists unless using tuples
        query = []
        for k, v in params.items():
            if isinstance(v, list):
                for item in v:
                    query.append((k, item))
            else:
                query.append((k, v))

        url = WCS_BASE.format(service=service)
        r = self.session.get(url, params=query, timeout=self.timeout)
        if r.ok and r.headers.get("Content-Type", "").lower().startswith(("image/tiff", "application/octet-stream")):
            return r.content
        return None

    def _get_wcs_100(self, *, service: str, coverage_id: str, lat: float, lon: float,
                     pixels: int = 3) -> Optional[bytes]:
        dlat, dlon = _deg_window_for_pixels(lat, pixels=pixels)
        south, north = lat - dlat / 2, lat + dlat / 2
        west, east = lon - dlon / 2, lon + dlon / 2

        params = {
            "SERVICE": "WCS",
            "VERSION": "1.0.0",
            "REQUEST": "GetCoverage",
            # In 1.0.0 it's COVERAGE (not COVERAGEID)
            "COVERAGE": coverage_id,
            "FORMAT": "GeoTIFF",
            "BBOX": f"{west},{south},{east},{north}",
            "CRS": "EPSG:4326",
            "RESPONSE_CRS": "EPSG:4326",
            # 3x3 pixel target grid
            "WIDTH": str(pixels),
            "HEIGHT": str(pixels),
        }
        url = WCS_BASE.format(service=service)
        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.ok and r.headers.get("Content-Type", "").lower().startswith(("image/tiff", "application/octet-stream")):
            return r.content
        return None

    def get_coverage_window(self, *, service: str, coverage_id: str, lat: float, lon: float,
                             pixels: int = 3) -> WCSResult:
        # Try WCS 2.0.1 first, then 1.0.0
        tiff = self._get_wcs_201(service=service, coverage_id=coverage_id, lat=lat, lon=lon, pixels=pixels)
        if tiff is None:
            tiff = self._get_wcs_100(service=service, coverage_id=coverage_id, lat=lat, lon=lon, pixels=pixels)
        if tiff is None:
            raise RuntimeError(f"Failed to fetch WCS coverage {service=}, {coverage_id=} for lat/lon {lat},{lon}")

        with MemoryFile(tiff) as mem:
            with mem.open() as ds:
                arr = ds.read(1)  # 2D array
                # Find the index (row, col) closest to requested point
                row, col = ds.index(lon, lat)
                row = int(np.clip(row, 0, arr.shape[0] - 1))
                col = int(np.clip(col, 0, arr.shape[1] - 1))
                # Compute center coordinate of that pixel
                x_center, y_center = ds.xy(row, col, offset="center")
                # ds.xy returns (x=lon, y=lat) for EPSG:4326
                px_lon, px_lat = float(x_center), float(y_center)
                dist_m = haversine_m(lat, lon, px_lat, px_lon)
                return WCSResult(
                    array=arr,
                    transform=ds.transform,
                    crs=str(ds.crs) if ds.crs else "EPSG:4326",
                    pixel_center_lat=px_lat,
                    pixel_center_lon=px_lon,
                    distance_m=dist_m,
                )

    # ---------------------------
    # Public convenience methods
    # ---------------------------
    def get_wrb_most_probable(self, lat: float, lon: float, pixels: int = 3) -> Dict[str, object]:
        service = "wrb"
        coverage_id = "MostProbable"  # WCS layer label used by ISRIC for WRB class mode
        res = self.get_coverage_window(service=service, coverage_id=coverage_id, lat=lat, lon=lon, pixels=pixels)
        # Select the nearest pixel value
        arr = res.array
        # nearest indices (removed buggy context manager usage)
        # rowcol returns a tuple, not a context manager
        # the pixel at the center of the 3x3 isn't guaranteed to be the nearest; take nearest from ds.index above
        # We stored only the chosen pixel center; find its array value by mapping lon/lat back to row/col
        # Easiest: compute row/col from transform and that lon/lat
        row, col = ~res.transform * (res.pixel_center_lon, res.pixel_center_lat)
        row_i, col_i = int(round(row)), int(round(col))
        row_i = max(0, min(arr.shape[0] - 1, row_i))
        col_i = max(0, min(arr.shape[1] - 1, col_i))
        code = int(arr[row_i, col_i])
        label = WRB_CODE_TO_LABEL.get(code, f"Unknown({code})")
        return {
            "wrb_code": code,
            "wrb_label": label,
            "pixel_center_lat": round(res.pixel_center_lat, 6),  # ~0.1m precision
            "pixel_center_lon": round(res.pixel_center_lon, 6),  # ~0.1m precision  
            "distance_m": round(res.distance_m, 1),  # Round to 0.1m precision
            "window_shape": arr.shape,
        }

    def get_property_value(self, *, lat: float, lon: float, property_id: str, depth: str = "0-5cm",
                           stat: str = "Q0.5", pixels: int = 3) -> Dict[str, object]:
        property_id = property_id.lower()
        if property_id not in SOILGRIDS_SERVICES:
            raise ValueError(f"{property_id!r} is not a recognized SoilGrids service ID")
        if depth not in STANDARD_DEPTHS:
            raise ValueError(f"Depth {depth!r} not in {STANDARD_DEPTHS}")
        if stat not in STANDARD_STATS:
            raise ValueError(f"stat {stat!r} not in {STANDARD_STATS}")

        # WRB has different coverages (MostProbable + per-class probabilities), not depth/quantile
        if property_id == "wrb":
            raise ValueError("Use get_wrb_most_probable() for WRB class. For per-class probabilities, fetch those coverages explicitly.")

        coverage_id = f"{property_id}_{depth}_{stat}"
        res = self.get_coverage_window(service=property_id, coverage_id=coverage_id, lat=lat, lon=lon, pixels=pixels)

        arr = res.array
        row, col = ~res.transform * (res.pixel_center_lon, res.pixel_center_lat)
        row_i, col_i = int(round(row)), int(round(col))
        row_i = max(0, min(arr.shape[0] - 1, row_i))
        col_i = max(0, min(arr.shape[1] - 1, col_i))
        raw_value = float(arr[row_i, col_i])
        
        # Apply scaling and add units
        scaling_factor = SOILGRIDS_SCALING.get(property_id, 1.0)
        scaled_value = raw_value * scaling_factor
        unit = SOILGRIDS_UNITS.get(property_id, "unknown")
        
        return {
            "property_id": property_id,
            "depth": depth,
            "stat": stat,
            "value": round(scaled_value, 2),
            "raw_value": raw_value,
            "unit": unit,
            "pixel_center_lat": round(res.pixel_center_lat, 6),  # ~0.1m precision
            "pixel_center_lon": round(res.pixel_center_lon, 6),  # ~0.1m precision  
            "distance_m": round(res.distance_m, 1),  # Round to 0.1m precision
            "window_shape": arr.shape,
        }


# --------------------
# Basic CLI for tests
# --------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Fetch SoilGrids values via WCS with robust fallbacks."
    )
    p.add_argument("lat", type=float)
    p.add_argument("lon", type=float)
    p.add_argument("kind", choices=["wrb", "property"], help="What to fetch")
    p.add_argument("--property", dest="prop", default="phh2o", help="Property id (e.g., phh2o, clay, sand)")
    p.add_argument("--depth", default="0-5cm", help=f"Depth interval (default: 0-5cm); options: {', '.join(STANDARD_DEPTHS)}")
    p.add_argument("--stat", default="Q0.5", help=f"Stat/quantile (default: Q0.5); options: {', '.join(STANDARD_STATS)}")
    p.add_argument("--pixels", type=int, default=3, help="Sampling window size in pixels for GetCoverage (default: 3)")
    args = p.parse_args()

    client = SoilGridsWCSClient()

    if args.kind == "wrb":
        out = client.get_wrb_most_probable(lat=args.lat, lon=args.lon, pixels=args.pixels)
    else:
        out = client.get_property_value(lat=args.lat, lon=args.lon, property_id=args.prop, depth=args.depth, stat=args.stat, pixels=args.pixels)

    print(textwrap.indent(textwrap.dedent(str(out)), prefix=""))

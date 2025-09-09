# coast_distance.py
from __future__ import annotations
import os, sys, io, zipfile, math, warnings
from typing import Optional, Iterable, Tuple, List

import geopandas as gpd
from shapely.geometry import Point, box
from shapely.strtree import STRtree
from shapely.ops import unary_union
from shapely import prepared

# --- Optional import: pyproj for exact meters ---
try:
    from pyproj import CRS, Transformer
    _HAVE_PYPROJ = True
except Exception:
    _HAVE_PYPROJ = False

# -------------------- CONFIG --------------------
OSM_COAST_ZIP = "https://osmdata.openstreetmap.de/download/coastlines-split-4326.zip"
OSM_LAND_ZIP  = "https://osmdata.openstreetmap.de/download/land-polygons-complete-4326.zip"
DEFAULT_CACHE = os.environ.get("COASTLINE_DATA_DIR", os.path.expanduser("~/.cache/osm_coastline"))
SEARCH_DEG_SEQUENCE = (2.0, 5.0, 10.0, 20.0, 60.0, 180.0)  # progressively expand search window

# ----------------- UTILITIES --------------------
def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    # Fast geodesic approximation (meters)
    R = 6371008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _download_if_needed(url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    local_zip = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(local_zip):
        import requests
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(local_zip, "wb") as f:
            f.write(resp.content)
    return local_zip

def _unzip_if_needed(zip_path: str, target_dir: str) -> str:
    # Extract once; return directory containing shapefile(s)
    base = os.path.splitext(os.path.basename(zip_path))[0]
    outdir = os.path.join(target_dir, base)
    if not os.path.isdir(outdir) or not os.listdir(outdir):
        os.makedirs(outdir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(outdir)
    return outdir

def _read_gdf_from_zip(url: str, cache_dir: str) -> gpd.GeoDataFrame:
    z = _download_if_needed(url, cache_dir)
    d = _unzip_if_needed(z, cache_dir)
    # Find the first .shp inside the extracted dir
    shp = None
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(".shp"):
                shp = os.path.join(root, f)
                break
        if shp: break
    if shp is None:
        raise RuntimeError(f"No .shp found inside {z}")
    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string().upper() not in ("EPSG:4326", "WGS84"):
        gdf = gdf.to_crs("EPSG:4326")
    return gdf

# ----------------- COAST INDEX ------------------
class CoastIndex:
    """
    Builds a spatial index from OSM coastlines (lines) or OSM land polygons (boundary).
    Use distance_to_coast_m(lat, lon) to compute distance in meters.

    If pyproj is available -> exact meters via local AEQD projection.
    Else -> vertex-based haversine approximation (conservative, usually within ~1-3% for <=100 km).
    """
    def __init__(self, cache_dir: str = DEFAULT_CACHE, prefer: str = "coastlines"):
        """
        prefer: 'coastlines' (lines) or 'land' (polygon boundaries)
        """
        self.cache_dir = cache_dir
        self.prefer = prefer
        self._gdf = None
        self._geoms: List = []
        self._tree: Optional[STRtree] = None
        self._prep_land_boundary = None  # for quick land/sea tests if using land polygons
        self._load_data()

    def _load_data(self):
        # Try preferred, then fallback
        urls = []
        if self.prefer == "coastlines":
            urls = [OSM_COAST_ZIP, OSM_LAND_ZIP]
        else:
            urls = [OSM_LAND_ZIP, OSM_COAST_ZIP]

        last_err = None
        for url in urls:
            try:
                gdf = _read_gdf_from_zip(url, self.cache_dir)
                # If using land polygons, compute boundaries for coastline distance
                if "land-polygons" in url:
                    gdf = gdf.explode(index_parts=False, ignore_index=True)
                    boundary = gdf.boundary
                    boundary = boundary[~boundary.is_empty]
                    self._gdf = gpd.GeoDataFrame(geometry=boundary, crs="EPSG:4326")
                    # Prepared union for quick land/sea tests if needed later
                    self._prep_land_boundary = prepared.prep(unary_union(gdf.geometry))
                else:
                    self._gdf = gdf[["geometry"]].copy()
                break
            except Exception as e:
                last_err = e
                continue
        if self._gdf is None:
            raise RuntimeError(f"Failed to load coastline data. Last error: {last_err}")

        self._geoms = list(self._gdf.geometry.values)
        self._tree = STRtree(self._geoms)

    @staticmethod
    def _local_aeqd(lon: float, lat: float) -> Transformer:
        if not _HAVE_PYPROJ:
            raise RuntimeError("pyproj is required for exact AEQD distances.")
        crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +units=m +ellps=WGS84")
        return Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def _candidates(self, lon: float, lat: float, search_deg: float) -> List:
        # bounding box query; then filter by intersection
        bbox = box(lon - search_deg, lat - search_deg, lon + search_deg, lat + search_deg)
        indices = self._tree.query(bbox)
        return [self._geoms[i] for i in indices if self._geoms[i].intersects(bbox)]

    def distance_to_coast_m(self, lat: float, lon: float) -> Optional[float]:
        # progressively widen search until we find something
        for win in SEARCH_DEG_SEQUENCE:
            cand = self._candidates(lon, lat, win)
            if cand:
                break
        else:
            return None  # nothing found (should not happen globally)

        # Exact meters via AEQD if pyproj is present
        if _HAVE_PYPROJ:
            to_aeqd = self._local_aeqd(lon, lat).transform
            pt = Point(*to_aeqd(lon, lat))
            mind = float("inf")
            for g in cand:
                g_m = gpd.GeoSeries([g], crs="EPSG:4326").to_crs(
                    CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=m +ellps=WGS84")
                ).iloc[0]
                d = pt.distance(g_m)
                if d < mind:
                    mind = d
            return float(mind)

        # Fallback: approximate by nearest vertex (haversine)
        warnings.warn("pyproj not found: using vertex-based geodesic approximation", RuntimeWarning)
        lonlat = (lon, lat)
        mind = float("inf")
        for g in cand:
            # iterate over coords (lines or polygon boundaries)
            try:
                coords = getattr(g, "coords", None)
                if coords is None:
                    for part in g.geoms:
                        for x, y in part.coords:
                            d = _haversine_m(lon, lat, x, y)
                            if d < mind:
                                mind = d
                else:
                    for x, y in g.coords:
                        d = _haversine_m(lon, lat, x, y)
                        if d < mind:
                            mind = d
            except Exception:
                # geometry may be empty or unexpected; skip
                continue
        return float(mind)

# -------------- Public convenience API --------------
_GLOBAL_IDX: Optional[CoastIndex] = None

def distance_to_coast_m(lat: float, lon: float,
                        cache_dir: str = DEFAULT_CACHE,
                        prefer: str = "coastlines") -> Optional[float]:
    """
    Drop-in function for your pipeline.
    - Downloads & caches OSM coastlines/land-polygons on first call (no fsspec).
    - Returns distance to nearest coastline in meters.
    - Uses pyproj (if available) for true meters; else, a geodesic approximation.
    """
    global _GLOBAL_IDX
    if _GLOBAL_IDX is None:
        _GLOBAL_IDX = CoastIndex(cache_dir=cache_dir, prefer=prefer)
    return _GLOBAL_IDX.distance_to_coast_m(lat, lon)
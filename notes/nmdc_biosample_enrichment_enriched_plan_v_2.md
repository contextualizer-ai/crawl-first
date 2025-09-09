# NMDC Biosample API Enrichment Strategy — v2 (Enriched)

> Purpose: turn under-populated NMDC Biosample records into analysis-grade datasets using deterministic, provenance-rich enrichments that are scalable to millions of samples and faithful to MIxS/NMDC semantics.

---

## 0) Snapshot & framing

- **Slots in NMDC Biosample**: 577
- **Currently represented**: 131 (~22.7%)
- **Heavily represented**: `env_*` triad (100%), `elev` (~83%), `lat_lon` (~61%), `collection_date` (~96%)
- **Under-represented but API-enrichable** (examples with current coverage): `soil_type` (~0.23%), `soil_texture_meth` (~0.23%), `salinity` (~0.14%), `conduc` (~2.46%), `diss_oxygen` (~4.39%), `nitrate` (~2.21%), `phosphate` (~2.08%), `sulfate` (~0.05%), `water_content` (~32.7%), `ph` (~34.3%), `temp` (~39.2%), `habitat` (~23.6%).

> Scope note: many *built-environment* slots (e.g., door/window/room fields) are not applicable to most environmental biosamples and are excluded from the priority plan except for clearly indoor datasets.

---

## 1) Prioritized enrichment backlog (actionable)

Ranked by **impact × feasibility** using asserted lat/lon, date, and ENVO triads.

| Priority | Slot | Current % | Target (90d) | Approach | Primary sources |
|---:|---|---:|---:|---|---|
| 1 | `soil_type` | ~0.23 | 40–60 | Raster lookup + taxonomy harmonization | ISRIC SoilGrids v2; USDA SSURGO (US); FAO DSMW |
| 2 | `soil_texture_meth` | ~0.23 | 30–50 | Derive textural class from SoilGrids *sand/silt/clay* layers; record "method"=derived | ISRIC SoilGrids |
| 3 | `salinity` (soil/terrain) | ~0.14 | 20–40 | Soil ECe / salinity indices; coast/ocean samples use WOA salinity climatology | SoilGrids (ECe); NOAA/WOA/CMEMS (marine) |
| 4 | `conduc` (electrical conductivity) | ~2.46 | 25–45 | Soil ECe → map to field EC with calibration; aquatic via conductivity–TDS heuristics | SoilGrids, WQP/USGS NWIS (US freshwaters) |
| 5 | `diss_oxygen` | ~4.39 | 25–40 | Temperature + altitude solubility model; refine with freshwater stations where available | WorldClim/ERA5 + USGS NWIS/HydroATLAS |
| 6 | `nitrate` / `nitrite` / `phosphate` / `sulfate` | 0.05–2.2 | 10–30 | Freshwater chemistry from WQP/NWIS/GLORICH by watershed match; marine via WOA climatology | EPA WQP, USGS NWIS, GLORICH, WOA |
| 7 | `water_content` | ~32.7 | 60–75 | Soil moisture from SMAP/ERA5-Land nearest-date match | NASA SMAP, ERA5-Land |
| 8 | `ph` | ~34.3 | 55–70 | Soil pH raster; aquatic pH from monitoring networks | SoilGrids pH(H2O), WQP/NWIS |
| 9 | `temp` | ~39.2 | 70–85 | Nearest-hour reanalysis; sea-surface temp for marine | ERA5, NOAA OISST |
| 10 | `habitat` | ~23.6 | 50–70 | Rule-based templating from land cover + triad + protected area type | Copernicus CGLS-LC100, WDPA |
| 11 | `soil_horizon` | ~34.6 | 55–70 | Infer from depth + pedon maps when available | SSURGO, GlobalSoilMap |
| 12 | `basin` (missing) | 0 | 70–90 | Watershed delineation / HydroBASINS ID | HydroBASINS, NHDPlusV2 (US) |
| 13 | `previous_land_use` (missing) | 0 | 20–40 | Historical land cover (ESA CCI) deltas | ESA CCI, LUH2 |
| 14 | `protected_area` (derived) | — | 70–90 | Point-in-polygon | WDPA |
| 15 | `ecosystem_services` (derived) | — | pilot | Model-based indices | InVEST/Co$tingNature (where licensing permits) |

> Targets assume global coverage; region-specific sources can exceed targets (e.g., US watersheds).

---

## 2) Algorithms & inference recipes (deterministic first)

### 2.1 Geospatial primitives
- **Snap**: round coordinates to 4–5 decimals (≈10–1 m) only for *tile caching* keys; never alter asserted values.
- **Tile cache**: WebMercator 1°/0.25°/0.05° hierarchy → drastically reduce raster/API calls.
- **Watershed join**: reverse-hydrology to find HydroBASINS basin ID; attach level 6–8 IDs.

### 2.2 Soil & geochem
- **Soil type**: fetch SoilGrids layers (*bdod, cec, clay, silt, sand, phh2o, ecec, soc, ec, wrb*) at point; derive USDA texture triangle class; map to ENVO soil classes when unambiguous.
- **pH**: use `phh2o` at 0–5/5–15/15–30 cm; choose depth band matching `depth`; record band.
- **EC/Salinity**: from SoilGrids ECe; report as ECe with unit; if needed, provide calibrated EC1:5; set `salinity_category` thresholded (FAO).

### 2.3 Hydrology & aquatic chemistry
- **Dissolved oxygen**: Weiss/O’Connor solubility from water temp + altitude/pressure; confidence downgraded if waterbody type unknown.
- **Nutrients (NO3−, NO2−, PO4, SO4)**: for inland → find nearest monitoring station in same basin within Δt=±30 days of `collection_date`; otherwise use long-term basin climatology (GLORICH/GEOChem). For marine → WOA monthly field at nearest grid cell.

### 2.4 Climate & microclimate
- **Near-surface temp**: ERA5 hourly at sample timestamp (or daily mean if time missing). Record temporal gap and aggregation window.
- **Water content**: SMAP L3 soil moisture daily; if vegetation/forest, apply canopy correction heuristic.
- **Season/phenology**: derive hemisphere-aware season; add NDVI percentile at date.

### 2.5 Land cover & human footprint
- **Land cover**: CGLS-LC100 or MODIS MCD12Q1 majority in 300 m radius; store class code + humanized label.
- **Protected area**: WDPA point-in-polygon; include IUCN category.
- **Human impact**: WorldPop density tile and OSM features count within 500 m.

### 2.6 Biodiversity context
- **GBIF/iNat**: 5 km buffered checklist for indicator taxa; optional (licensing-aware). Useful for feasibility/QA.

---

## 3) Data model & provenance (Pydantic)

```python
from pydantic import BaseModel, Field, conlist
from typing import Literal, Optional, List, Dict

class Evidence(BaseModel):
    source: str  # e.g., "SoilGrids v2.0"
    method: Literal[
        "point_lookup", "basin_join", "nearest_station",
        "reanalysis", "derived_model", "raster_majority"
    ]
    value: str | float | Dict
    unit: Optional[str] = None  # UCUM (e.g., 'mg/L')
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    spatial_support_m: Optional[int] = None  # buffer radius used
    version: Optional[str] = None
    confidence: float = Field(ge=0, le=1)

class EnrichmentField(BaseModel):
    asserted_value: Optional[str | float | Dict] = None
    api_enrichments: conlist(Evidence, min_items=1)
    validation_status: Literal["validated", "conflict", "unverified"]

class BiosampleEnrichment(BaseModel):
    soil_type: Optional[EnrichmentField]
    ph: Optional[EnrichmentField]
    temp: Optional[EnrichmentField]
    conductivity: Optional[EnrichmentField]
    dissolved_oxygen: Optional[EnrichmentField]
    nutrients: Dict[str, EnrichmentField]  # nitrate, nitrite, phosphate, sulfate
    land_cover: Optional[EnrichmentField]
    protected_area: Optional[EnrichmentField]
    basin: Optional[EnrichmentField]
```

### Units & UCUM
- Normalize with **pint** (UCUM registry) and emit UCUM codes in `unit`.
- Store raw strings separately to preserve provenance.

### Confidence tuning (per-method priors)
- `point_lookup` ≥0.9; `basin_join` 0.8–0.9; `nearest_station` scale by distance/time gap; `derived_model` 0.6–0.8.

---

## 4) Validation rules & conflict matrix

| Rule | Check | Action |
|---|---|---|
| Coordinate↔Elevation | SRTM/DEM vs asserted `elev` | Flag >200 m discrepancy → downgrade confidence, queue review |
| Triad coherence | `env_broad_scale`→`env_local_scale`→`env_medium` semantic path | If inconsistent with land cover/soil type, set `conflict` |
| Hydro context | Sample on land but `env_medium` indicates water (or vice versa) | Re-check geocoding; if persistent, mark as edge case |
| Date/Season | `collection_date` season vs climate normals extremes | Downgrade improbable values |
| Units | All numeric fields UCUM-valid | Reject non-normalizable units; keep raw in provenance |

---

## 5) Pipeline architecture (scalable)

- **Orchestrator**: plain asyncio + `httpx` with rate-limiters; or Prefect if desired later.
- **Batching**: spatial clustering (DBSCAN/H3) → group calls by tile/basin.
- **Caching**: diskcache/redis; keys by (source, tile/basin, date bucket, version).
- **Storage**: MongoDB `enrichments` collection with asserted/inferred sections and `enrichment_metadata` summary.
- **Idempotency**: deterministic cache keys; enrichment functions pure w.r.t inputs.

```python
class Enricher:
    def __init__(self, clients, cache): ...
    async def run(self, sample):
        # example fan-out
        soil, climate, hydro = await gather(
            self.soil(sample), self.climate(sample), self.hydro(sample)
        )
        return merge_results(soil, climate, hydro)
```

---

## 6) QA & observability

- **Coverage dashboard**: % populated per slot; deltas per run.
- **Accuracy auditing**: 1–5% stratified sample manual review; inter-rater agreement.
- **Golden cases**: MIxS 6.2 *valid/invalid* examples as unit tests; ensure parsers/validators capture errors deterministically.
- **Drift**: monitor upstream version changes (e.g., SoilGrids) and pin.

---

## 7) LinkML/NMDC integration (proposed slots)

```yaml
slots:
  basin:
    description: HydroBASINS or analogous watershed ID containing the sampling point.
    range: string
    exact_mappings: [hydrobasins:HYBAS_ID]

  protected_area:
    description: WDPA protected area intersecting the sampling point.
    range: ProtectedAreaValue

classes:
  ProtectedAreaValue:
    slots: [wdpa_id, name, iucn_category]
```

> For derived fields (e.g., modeled DO), consider dedicated `*_modeled` slots to avoid conflating with asserted measurements.

---

## 8) Milestones (near-term)

**M0 (2 weeks):**
- Implement SoilGrids point lookups (pH, texture, EC, soil type) with caching.
- Implement ERA5 temperature and SMAP soil moisture at timestamp/date.
- Implement WDPA and land cover joins.
- Emit Pydantic-provenance records and write to MongoDB.

**M1 (next 4–6 weeks):**
- Watershed delineation + WQP/NWIS joins for US freshwater chemistry.
- WOA joins for marine chemistry.
- Confidence calibration & conflict matrix enforcement.

**M2 (scale-up):**
- Batch clustering, tile prefetchers, dashboards, and regression-test suite.

---

## 9) Example: end-to-end enrichment JSON

```json
{
  "soil_type": {
    "asserted_value": null,
    "api_enrichments": [
      {
        "source": "SoilGrids v2.0",
        "method": "point_lookup",
        "value": {"wrb": "Leptosols", "texture": "sandy_loam"},
        "unit": null,
        "spatial_support_m": 250,
        "version": "2020-11",
        "confidence": 0.92
      }
    ],
    "validation_status": "validated"
  },
  "ph": { "api_enrichments": [{
      "source": "SoilGrids v2.0",
      "method": "point_lookup",
      "value": 6.4,
      "unit": "pH",
      "confidence": 0.9
  }], "validation_status": "validated" },
  "temp": { "api_enrichments": [{
      "source": "ERA5",
      "method": "reanalysis",
      "value": 18.7,
      "unit": "Cel",
      "time_start": "2024-06-15T12:00:00Z",
      "time_end": "2024-06-15T13:00:00Z",
      "confidence": 0.88
  }], "validation_status": "validated" }
}
```

---

## 10) Open items / decisions

- Model vs measured: introduce `*_modeled` vs `*_measured` slots or encode as provenance `method`.
- Global sources for freshwater chemistry outside US: GLORICH coverage + national portals.
- Licensing for ecosystem services layers; may be analysis-only (non-redistributable).

---

### Appendix A: Rate limits & batching defaults (suggested)

- SoilGrids: ≤ 10 qps; pre-tile popular areas.
- ERA5: local cache via CDS files where permitted; otherwise API with daily buckets.
- WDPA/CGLS: ship as local vector/raster and query locally.
- WQP/NWIS: per-agency rate limits; cache station time series by basin.

### Appendix B: Overpass sketch for human footprint

```
[out:json][timeout:50];
(
  node(around:500, {lat}, {lon})[highway];
  way(around:500, {lat}, {lon})[landuse=industrial];
);
out center;
```

---

**Done.** Ready to generate code stubs for SoilGrids/ERA5/WDPA/land-cover, wire up caching, and run a pilot on 10k samples.


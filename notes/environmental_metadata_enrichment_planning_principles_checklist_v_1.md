# Environmental Metadata Enrichment — Planning Principles & Checklist (v1)

_Last updated: 2025‑09‑08_

## Purpose

A concise, professional set of operating principles and a ready‑to‑use checklist for planning and reviewing work on the environmental metadata enrichment tool. These rules guide design, implementation, and evaluation across APIs, datasets, and inference steps.

---

## Core Principles (always apply)

- **Avoid circular validation.** Do **not** cross‑check results using another endpoint derived from the **same upstream dataset**.
- **Throughput matters.** Prefer designs that maximize end‑to‑end throughput (batching, sensible caching, concurrency where safe) without sacrificing reproducibility.
- **GeoTIFF prudence.** Use GeoTIFFs only when they are small enough to be practical **and** large enough to be informative; otherwise prefer server‑side subsetting (e.g., WCS, tiles) or vector alternatives.
- **Use latest data.** Use the latest API versions and dataset releases. For dated products, select the version **closest to the sample’s collection date**.
- **Progressive discovery.** Take a spidering/crawling approach to uncover additional APIs, endpoints, datasets, and layers; record discoveries and gaps.
- **Minimize hardcoding.** Keep hand‑rolled mappings to a minimum; when unavoidable, **document** them with rationale, source, and proof.
- **Environment is provisioned.** Assume required Python libraries and large dependencies are pre‑installed; avoid runtime “missing dependency” checks in the hot path.
- **Distances & confidence.** Every lookup/inference should return distance-to-source (e.g., station distance, coastline distance) and a confidence/quality indicator where available.
- **Prefer pre‑aggregated daily weather data.** When feasible, use authoritative daily aggregates (vs. raw streams that we must aggregate ourselves).
- **Iterate on working pieces.** Build iteratively from components known to work; reconnect dots with previously failed approaches to improve coverage.

---

## Data Sourcing & API Strategy

- **Primary scope (single script + single Makefile):** For a given `(latitude, longitude, date)` retrieve:
  1) **Elevation**  
  2) **Forward geocoding**  
  3) **Reverse geocoding**  
  4) **Weather** (daily resolution) including advanced fields (e.g., solar energy/irradiance, humidity, pressure, soil moisture/temperature)  
  5) **Soil characteristics**  
- **Ocean awareness.** When context suggests marine/near‑shore, expand with oceanographic datasets (e.g., bathymetry, distance to coast, currents, salinity, chlorophyll) as appropriate.
- **GOLD ecosystem paths (later step).** Plan for LLM‑assisted classification of sample context: terrestrial, inland freshwater, near‑shore, oceanic, plant‑associated, other host‑associated, other. Treat as its own inference module with audit trails.
- **Discovery log.** Maintain a running log (CSV/JSON) of APIs/endpoints tried, fields available, rate limits, coverage, and observed issues.

---

## Spatial & Temporal Policy

- **Station/source distance.** Always compute and report distance from input coordinates to the measurement location(s); include the selection window/nearest‑N policy in outputs.
- **Weather scale/window.** Choose a window that matches the product’s recommended spatial/temporal scale; document the policy (e.g., nearest station vs. grid cell centroid, max radius).
- **Temporal alignment.** For daily values, use the product’s native aggregation if available; otherwise document our aggregation method and time zone assumptions.

---

## OSM Feature Strategy

- **Named vs unnamed.** For **named** features: fetch full metadata (all tags). For **unnamed** features: return **counts by key/value** and totals.
- **No premature filtering.** Do not limit to a small set of keys/types; request as many fields as the API reasonably allows.
- **Search area.** Expose a configurable **radius/area** parameter for OSM queries; include it in outputs and logs.

---

## Outputs, Provenance & Units

- **Return everything categorical.** For endpoints that return categorical/enumerated values, collect **all** categories/fields rather than a curated subset.
- **Provenance & proof.** For any mapping, conversion factor, or decoding, store documentation in code and link to the authoritative source; where possible, capture a machine‑readable proof (e.g., versioned spec snippet, citation, or checksum).
- **Units policy.** Align with **NMDC Schema** unit expectations; prefer **UCUM** where applicable. Include both raw values and normalized unit annotations.
- **Soil taxonomy fidelity.** Avoid oversimplifying soil classification returns; preserve the full classification levels and any confidence scores.

---

## Mapping & Ontology Alignment

- **Defer crosswalks.** Do not attribute failures to “missing crosswalks” unless explicitly in scope. Perform mapping **later** in the pipeline (after data capture/evidence).
- **Destinations.** Primary destinations include ENVO (environmental context), plus project‑specific targets (e.g., NMDC Schema, GOLD). Keep this list explicit in plan docs.
- **Formats.** Keep both CSV **and** JSON forms of mappings for interoperability. Evaluation of mapping quality is a distinct, subsequent phase.

---

## Performance & Engineering Hygiene

- **Throughput tactics.** Batch requests where supported; cache responses with clear keys (dataset, version, params); minimize unnecessary large downloads.
- **GeoTIFF handling.** Prefer server‑side subsetting, tiling, or lower‑resolution products before downloading full rasters.
- **Code/targets.** Remove ad‑hoc test scripts/targets; avoid `python -m` targets; standardize on a single CLI entrypoint and a single Makefile.
- **TEOW & Meteostat.** TEOW ecoregions (local lookup) and Meteostat (via MCP or direct) are in‑scope; document versions and retrieval modes.

---

## Evaluation & QA (quantitative)

- **Coverage tracking.** Quantify how often required NMDC/GOLD slots are populated; report by source and by field.
- **Quality metrics.** Track station distances, confidence scores, data age vs. collection date, proportion of categorical fields captured.
- **Change detection.** When upgrading APIs/datasets, compare field availability and value deltas; record upgrades/downgrades.
- **Logs.** Keep machine‑readable logs for categories captured, errors, fallbacks used, and suggestions generated.

---

## Open Questions / To Decide

- **“AI‑enabled” definition.** What qualifies (model class, prompting style, uncertainty reporting, cost/latency budget)?
- **Suggestion provenance.** How are follow‑up suggestions generated, ranked, and attributed (rules vs. LLM vs. heuristics)?
- **Ocean data menu.** Which datasets are canonical for near‑shore vs. oceanic contexts?
- **Target vocabularies beyond ENVO.** Confirm the definitive list (e.g., NMDC Schema enums, GOLD types, Wikidata links) for later mapping.

---

## Planning Checklist (use for every new plan)

1. **Inputs fixed:** `(lat, lon, date)` and spatial **radius/window** chosen and recorded.
2. **APIs enumerated:** elevation, fwd/rev geocoding, daily weather (+ solar/soil), soil characteristics; marine add‑ons if applicable.
3. **Latest versions chosen:** API/data versions closest to collection date documented.
4. **No circular checks:** validation sources are independent of primary sources.
5. **Distances & confidence:** included for each result; window/nearest‑N policy recorded.
6. **OSM policy applied:** named → full metadata; unnamed → counts; radius logged.
7. **Units normalized:** UCUM/NMDC‑compatible, with raw + normalized values retained.
8. **Soil taxonomy preserved:** full depth + confidences captured (no flattening loss).
9. **Categoricals complete:** all enumerations captured; counts logged.
10. **Performance plan:** batching/caching/tiling strategy documented; GeoTIFF prudence applied.
11. **Provenance saved:** proofs/links for any mappings, conversions, and decodings.
12. **Evaluation hooks:** coverage metrics, error logs, and suggestion provenance enabled.

---

## Definition of Done (per run/module)

- Reproducible outputs with explicit versions and parameters.
- Distances, confidences, and provenance present for every inference.
- No unvetted hardcoded mappings; any required mappings documented with proof.
- Logs include categories captured, errors, fallbacks, and suggestion sources.
- Plan and results pass the Planning Checklist above.

# Marine Enrichment Priority Mapping

## Schema-Aligned Marine Parameters (High Priority)

Based on analysis of NMDC and GOLD biosample schemas, these marine parameters directly align with existing database slots:

### **Tier 1: Direct Schema Matches (Implement First)**

| Marine API Parameter | NMDC Schema Slot | GOLD Schema Slot | Marine API Source | Priority |
|---------------------|------------------|------------------|-------------------|----------|
| **Sea Surface Temperature** | `temp` | `sampleCollectionTemperature` | NOAA OISST v2.1 | **HIGHEST** |
| **Chlorophyll-a** | `chlorophyll` | - | ESA OC-CCI v6 | **HIGHEST** |
| **Salinity** | `salinity` | `salinity`, `salinityConcentration` | CMEMS Physics | **HIGHEST** |
| **Dissolved Oxygen** | `diss_oxygen` | `oxygenConcentration` | CMEMS BGC | **HIGH** |
| **pH** | `ph` | `ph` | CMEMS BGC | **HIGH** |
| **Water Column Depth** | `tot_depth_water_col`, `depth` | `depthInMeters` | GEBCO Bathymetry | **HIGHEST** |

### **Tier 2: Marine-Specific Extensions (Implement Second)**

| Marine Parameter | Target Schema Extension | Marine API Source | Justification |
|------------------|------------------------|-------------------|---------------|
| **Ocean Currents** | New: `ocean_surface_velocity_u/v` | OSCAR Currents | Critical for marine ecosystem characterization |
| **Significant Wave Height** | New: `wave_height` | NOAA WaveWatch III | Physical oceanographic context |
| **Ocean Biogeographic Province** | Extension of ecosystem classification | Longhurst/MEOW/PPOW | Already in GOLD (`longhurst` slot), extend to NMDC |
| **Marine Protected Area Status** | New: `marine_protection_status` | WDPA Marine | Conservation context |

### **Schema Slot Coverage Analysis**

#### **NMDC Biosample Slots (Marine-Relevant)**
- ✅ `chlorophyll` - **Direct match** for chlorophyll-a concentration
- ✅ `salinity` - **Direct match** for salinity measurements  
- ✅ `diss_oxygen` - **Direct match** for dissolved oxygen
- ✅ `ph` - **Direct match** for pH measurements
- ✅ `temp` - **Direct match** for sea surface temperature
- ✅ `depth` - **Direct match** for sampling depth
- ✅ `tot_depth_water_col` - **Direct match** for water column depth/bathymetry
- ✅ `elev` - Can store bathymetry (negative elevation for marine samples)

#### **GOLD Biosample Slots (Marine-Relevant)**
- ✅ `sampleCollectionTemperature` - **Direct match** for SST
- ✅ `salinity`, `salinityConcentration` - **Direct match** for salinity
- ✅ `oxygenConcentration` - **Direct match** for dissolved oxygen
- ✅ `ph` - **Direct match** for pH
- ✅ `depthInMeters`, `subsurfaceDepthInMeters` - **Direct match** for depth/bathymetry
- ✅ `longhurst` - **Already populated** with Longhurst marine biogeographic provinces
- ✅ `elevationInMeters` - Can store bathymetry (negative values)

## Implementation Priority Ranking

### **Phase 1: Core Marine Parameters (100% Schema Compatible)**
1. **NOAA OISST v2.1** → `temp` / `sampleCollectionTemperature` (SST)
2. **GEBCO Bathymetry** → `tot_depth_water_col` / `depthInMeters` (bathymetry)
3. **ESA OC-CCI v6** → `chlorophyll` (chlorophyll-a)

**Rationale**: These 3 parameters provide immediate value for ~1,500 marine samples with zero schema modifications required.

### **Phase 2: Chemical Parameters (High Schema Value)**
4. **CMEMS Physics** → `salinity` / `salinityConcentration` (salinity)
5. **CMEMS BGC** → `diss_oxygen` / `oxygenConcentration` (dissolved oxygen)
6. **CMEMS BGC** → `ph` / `ph` (pH)

**Rationale**: Direct enrichment of existing chemical measurement slots with historical marine data.

### **Phase 3: Advanced Marine Context**
7. **OSCAR Currents** → New slots for ocean circulation context
8. **Longhurst/MEOW Biogeography** → Extend existing `longhurst` coverage to NMDC
9. **WaveWatch III** → New slots for physical oceanographic conditions

## Marine API Catalog Alignment

| Priority | Marine API | Schema Slots Populated | Historical Coverage | Access Method |
|----------|------------|------------------------|-------------------|---------------|
| **1** | NOAA OISST v2.1 | `temp`, `sampleCollectionTemperature` | 1981-present | ERDDAP (no auth) |
| **1** | GEBCO Bathymetry | `tot_depth_water_col`, `depthInMeters`, `elev` | Static global | WMS/WCS (no auth) |
| **1** | ESA OC-CCI v6 | `chlorophyll` | 1997-present | ERDDAP via NEFSC |
| **2** | CMEMS Physics | `salinity`, `salinityConcentration` | 1993-present | Copernicus Marine Client |
| **2** | CMEMS BGC | `diss_oxygen`, `oxygenConcentration`, `ph` | 1993-present | Copernicus Marine Client |
| **3** | OSCAR Currents | New: `ocean_velocity_u/v` | 1992-present | OPeNDAP |

## Expected Impact

### **Marine Sample Enrichment Coverage**
- **Phase 1**: 3 core parameters × 1,500 marine samples = **4,500 new data points**
- **Phase 2**: 3 chemical parameters × 1,500 samples = **4,500 additional data points**  
- **Total**: **9,000+ marine environmental data points** from existing schema slots

### **Schema Population Rate Improvement**
- **Current NMDC**: ~22.7% slot population rate (131/577 slots)
- **Post Marine Enrichment**: Significant improvement in marine-specific slots
- **GOLD Enhancement**: Fill gaps in `sampleCollectionTemperature`, `salinityConcentration`, etc.

### **Cross-Database Value**
- Enable **marine-terrestrial comparative analysis** with consistent environmental metadata
- Support **ecosystem-aware sample selection** using biogeographic classifications
- Provide **temporal marine environmental context** for biosample analysis

## Technical Implementation Strategy

### **Realm-Aware Routing**
```python
if site_type == "marine":
    # Phase 1: Core marine parameters
    sst = fetch_oisst_temperature(lat, lon, date)          # → temp slot
    depth = fetch_gebco_bathymetry(lat, lon)               # → tot_depth_water_col slot  
    chl = fetch_oc_cci_chlorophyll(lat, lon, date)         # → chlorophyll slot
    
    # Phase 2: Chemical parameters (if date >= 1993)
    if date >= "1993-01-01":
        salinity = fetch_cmems_salinity(lat, lon, date)    # → salinity slot
        oxygen = fetch_cmems_oxygen(lat, lon, date)        # → diss_oxygen slot
        ph = fetch_cmems_ph(lat, lon, date)                # → ph slot
```

### **Validation Strategy**
- **SST**: NOAA OISST ↔ HadISST (independent processing)
- **Chlorophyll**: OC-CCI ↔ VIIRS (different sensors/algorithms)
- **Salinity/Oxygen**: CMEMS ↔ World Ocean Atlas (model vs. observation-based)

This schema-aligned approach ensures **maximum immediate value** by populating existing database slots with historically-covered marine environmental data, addressing the core requirement for 25-year historical coverage while maintaining full compatibility with current NMDC and GOLD biosample schemas.
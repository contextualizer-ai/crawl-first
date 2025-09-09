# Additional API Testing Opportunities

*Based on schema analysis from data/outputs/schema/ - Generated: January 6, 2025*

## 🔍 **Additional Input Types We Should Test**

### 🏛️ **Geographic/Administrative Context**
**Available Inputs from Schema**:
- `geo_loc_name` (99.8% coverage) - Location names like "Cambridge, MA"
- `isoCountry` (98.3% coverage) - ISO country codes
- `geoLocation` (99.4% coverage) - Geographic location descriptions

**Missing API Tests**:
- **Administrative Boundary APIs**: Test country/state/province lookup using ISO codes
- **Geographic Name Resolution**: Test location name standardization and validation
- **Administrative Hierarchy APIs**: Test postal code → admin boundary resolution

### 🌊 **Marine/Oceanic Context** 
**Available Inputs from Schema**:
- `longhurst` (0.4% coverage in GOLD) - Marine biogeographic provinces
- `salinity`/`salinityConcentration` - Marine salinity measurements
- Marine ecosystem classifications

**Missing API Tests**:
- **Marine Province APIs**: Test Longhurst Biogeographical Province lookup by coordinates
- **Ocean/Sea Name APIs**: Test marine water body identification
- **Bathymetry APIs**: Test ocean depth/seafloor elevation data
- **Marine Current APIs**: Test ocean current data by location

### 🏞️ **Terrestrial Land Use/Cover**
**Enrichment Opportunities from Analysis**:
- Land cover classification (0% current coverage)
- Vegetation/habitat types from coordinates
- Protected area status (national parks, reserves)

**Missing API Tests**:
- **Land Cover APIs**: Test NLCD, ESA WorldCover, or MODIS land cover classification
- **Protected Areas APIs**: Test national park/reserve identification
- **Vegetation Index APIs**: Test NDVI or vegetation density lookup

### 🧪 **Chemical/Environmental Parameters**
**Low Coverage Fields (1.5-37.7%)**:
- `temp`, `humidity`, `wind_speed`, `wind_direction`, `solar_irradiance`
- `ph`, `diss_oxygen`, `salinity`, `conductivity`
- Soil chemistry: `org_carb`, `nitro`, `calcium`, `magnesium`

**Missing API Tests**:
- **Soil Property APIs**: Test USDA SSURGO, SoilGrids for soil chemistry/texture
- **Water Quality APIs**: Test USGS water quality stations for aquatic samples
- **Air Quality APIs**: Test EPA/environmental air quality monitoring

### 🌍 **Global Reference Data**
**Available from Schema Analysis**:
- ENVO ontology terms (100% coverage but often generic)
- Ecosystem classifications with coordinates
- Temporal data (96.1% collection dates)

**Missing API Tests**:
- **Ecoregion APIs**: Test WWF Terrestrial Ecoregions, Bailey's Ecoregions
- **Climate Zone APIs**: Test Köppen climate classification
- **Biome APIs**: Test biome classification from coordinates
- **Time Zone APIs**: Test time zone identification from coordinates

## 📊 **Recommended Priority Order**

1. **Soil Property APIs** (USDA SSURGO, SoilGrids) - High impact for terrestrial samples
2. **Marine Province APIs** (Longhurst, IHO Sea Areas) - Fill marine data gaps
3. **Land Cover APIs** (NLCD, ESA WorldCover) - Universal terrestrial context
4. **Ecoregion APIs** (WWF, EPA) - Improve ecosystem classifications
5. **Administrative Boundary APIs** - Standardize location naming
6. **Water Quality APIs** (USGS) - For aquatic/freshwater samples

## 🔧 **Implementation Strategy**

These APIs would follow the same patterns as our existing tests:
- **Input**: Latitude/longitude coordinates (+ optional parameters)
- **Output**: Structured environmental/administrative data
- **Quality Assessment**: Completeness scoring and validation
- **Rate Limiting**: Respectful API usage
- **Error Handling**: Graceful degradation

## 📈 **Coverage Impact**

This would expand our test suite from **4 API categories** to **10+ API categories**, providing comprehensive geospatial enrichment coverage for biosample research workflows:

### Current Coverage
- ✅ Coordinates ↔ Addresses (Forward/Reverse Geocoding)
- ✅ Elevation lookup
- ✅ Historical weather data

### Potential New Coverage
- 🆕 Soil properties and chemistry
- 🆕 Marine biogeographic provinces
- 🆕 Land cover/land use classification
- 🆕 Ecoregion and biome classification
- 🆕 Administrative boundaries and time zones
- 🆕 Water quality monitoring data
- 🆕 Air quality and environmental monitoring

## 🎯 **Expected Enrichment Outcomes**

Based on enrichment analysis, implementing these APIs could:
- **Weather APIs**: Enrich ~12,400 samples with complete climate data
- **Soil APIs**: Provide detailed soil properties for ~7,000 terrestrial samples
- **Marine APIs**: Add biogeographic context to ~1,500 marine samples
- **Land Cover APIs**: Classify habitat for ~8,000 terrestrial samples
- **Ecoregion APIs**: Improve ecosystem classifications for all 13,006+ samples
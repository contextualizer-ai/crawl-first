# Biosample Enrichment Roadmap

## What You're Trying to Accomplish

**Core Goal**: Build a comprehensive biosample enrichment pipeline that can process biosamples from **multiple sources** (NMDC, GOLD, and others) and enrich them with complete environmental context for LLM-ready analysis.

**The Evolution**:
1. ✅ **Phase 1**: NMDC biosample analysis went well  
2. ❌ **Blocker**: ORNL geoloc-tools API went offline
3. 🔧 **Today's Work**: Consolidating all experimental geospatial work back into main package
4. 🎯 **Next Phase**: Expand to other biosample sources with robust testing

## Biosample Metadata Inference Capabilities

Based on your codebase, here's what you want to infer for **any biosample source**:

### **Input Requirements**
| Input Type | Required | Fallback Strategy |
|------------|----------|-------------------|
| **Coordinates** | Latitude/Longitude (decimal degrees) | Geocode from location names |
| **Collection Date** | ISO format or parseable date string | Use "latest available" for temporal data |
| **Optional Context** | Sample type, ecosystem hints, elevation | Use for smart API routing |

### **Environmental Metadata to Infer**

#### **🌍 Geospatial Context**
| Data Type | Sources | Returns |
|-----------|---------|---------|
| **Elevation** | Open Elevation API, USGS EPQS | Meters, feet, validation |
| **Administrative** | OSM Nominatim | Country, state, county, city |
| **Nearby Features** | OSM Overpass API | Natural features, water bodies, land use |
| **Place Classification** | OSM + coordinates | Urban/rural, protected areas |

#### **🏔️ Soil & Geology**  
| Data Type | Sources | Returns |
|-----------|---------|---------|
| **Soil Classification** | USDA NRCS SDA (US), ISRIC SoilGrids (Global) | USDA taxonomy, WRB classification |
| **Soil Properties** | ISRIC SoilGrids WCS | pH, SOC, texture (sand/silt/clay %), bulk density |
| **Soil Texture Class** | Calculated from properties | USDA 12-class texture triangle |
| **ENVO Normalization** | Crosswalk mappings | Soil type → ENVO ontology terms |

#### **🌦️ Weather & Climate**
| Data Type | Sources | Returns |
|-----------|---------|---------|
| **Historical Weather** | Open-Meteo Archive API | Temperature, precipitation, humidity, wind |
| **Weather Station Distance** | Coordinate tracking | Distance from sample to weather station |
| **Climate Context** | Weather patterns | Long-term climate characterization |

#### **🌱 Ecosystem & Land Cover**
| Data Type | Sources | Returns |
|-----------|---------|---------|
| **Ecoregion** | TEOW 2017/RESOLVE | Biome classification, realm |
| **Land Cover** | Multiple classification systems | NLCD, ESA WorldCover, land use types |
| **Ecosystem Classification** | Rule-based + LLM | terrestrial/freshwater/marine/host-associated |
| **ENVO Mappings** | Crosswalk system | Land cover → ENVO ontology terms |

#### **📊 Data Quality Assessment**
| Assessment Type | Method | Returns |
|----------------|--------|---------|
| **Coordinate Validation** | Distance calculations | Asserted vs. geocoded coordinate differences |
| **Elevation Verification** | Cross-source comparison | Elevation consistency checks |
| **Weather Station Quality** | Distance-based scoring | 5-point quality scale for weather data |
| **API Success Tracking** | Success/failure monitoring | Data completeness scoring |

### **Smart Biosample Source Detection**

Your system should be able to handle:

| Source Type | Detection Method | Special Processing |
|-------------|------------------|-------------------|
| **NMDC Biosamples** | ID pattern `nmdc:bsm-*` | Full NMDC metadata retrieval |
| **GOLD Biosamples** | Schema analysis, ecosystem paths | GOLD-specific field mapping |
| **Custom CSV/JSON** | Schema inference | Dynamic field mapping |
| **Other Standards** | Metadata patterns | Extensible source detection |

### **Resources We Can Use**

#### **Working APIs (Production Ready)**
- ✅ **USDA NRCS SDA**: US soil taxonomy (high quality)
- ✅ **ISRIC SoilGrids**: Global soil data (WRB + properties)  
- ✅ **Open Elevation**: Global elevation data
- ✅ **Open-Meteo**: Global historical weather with coordinate tracking
- ✅ **OSM Nominatim**: Global reverse geocoding
- ✅ **OSM Overpass**: Global environmental features
- ✅ **TEOW Ecoregions**: Global biome classification

#### **MCP Resources Available**
- **NMDC-MCP**: Biosample and functional annotation data
- **Weather-Context-MCP**: Enhanced weather analysis  
- **Landuse-MCP**: Land cover classification (when working)
- **OLS-MCP**: Ontology term resolution
- **ARTL-MCP**: Publication analysis

### **What They Return**

#### **Comprehensive JSON Output Structure**
```json
{
  "input_metadata": {}, 
  "enrichment_results": {
    "geospatial": {
      "elevation": {"meters": 2460, "source": "Open Elevation"},
      "location": {"country": "United States", "state": "Wyoming"},
      "nearby_features": {"natural": ["hot_spring", "grassland"]}
    },
    "soil": {
      "classification": {"usda": "Typic Haplocryepts", "wrb": "Cambisols"},
      "properties": {"ph": 5.5, "texture_class": "Loam"},
      "envo_terms": ["ENVO:00002258"]
    },
    "weather": {
      "temperature_c": {"mean": 12.3, "min": -2.1, "max": 23.7},
      "station_distance_km": 4.18,
      "quality_score": 4
    },
    "ecosystem": {
      "ecoregion": "South Central Rockies forests",
      "biome": "Temperate Conifer Forests", 
      "classification": "terrestrial"
    }
  },
  "data_quality": {
    "completeness_score": 0.85,
    "successful_apis": 8,
    "failed_apis": 2
  }
}
```

## Proposed Python Testing Strategy

Instead of complex Makefiles, let's build comprehensive **pytest-based testing**:

### **Test Categories**
1. **Unit Tests**: Individual API functions, coordinate validation, ENVO mappings
2. **Integration Tests**: Full enrichment pipeline with real coordinates
3. **Source Tests**: NMDC vs GOLD vs custom biosample processing  
4. **Performance Tests**: Throughput, caching, concurrent processing
5. **Quality Tests**: Data completeness, validation accuracy

### **Test Data Strategy**
- **Known Coordinates**: Yellowstone, San Francisco, oceanic samples
- **Biosample Fixtures**: Representative NMDC, GOLD, and custom samples
- **Expected Results**: Validated enrichment outputs for regression testing
- **Edge Cases**: Invalid coordinates, missing data, API failures

### **Next Steps**
The system can handle any biosample source and provide comprehensive environmental context with robust quality assessment. Ready to start building those Python tests!

## Current Status Summary

### ✅ **Accomplished Today**
- Consolidated all geospatial work from `gold-and-nmdc/` into main `src/crawl_first/` package
- Moved all non-recreatable assets (mappings, prompts, docs) to appropriate root locations
- Cleaned up recreatable files and removed the `gold-and-nmdc/` directory entirely
- Created unified Makefile with proper file-based targets using `$@` and `$<`
- Established comprehensive schema analysis workflow with MongoDB inference and Claude CLI integration
- Fixed import structure for package-based development

### 🎯 **Ready for Next Phase**
- Switch from Makefile-based workflow to comprehensive Python testing
- Expand biosample source support beyond NMDC to include GOLD and custom formats
- Build robust test suite covering all enrichment capabilities
- Implement production-ready biosample enrichment pipeline for multiple data sources
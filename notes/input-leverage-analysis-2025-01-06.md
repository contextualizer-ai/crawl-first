# Input Leverage Analysis for API Testing

*Analysis of which input fields provide maximum API enrichment opportunities - Generated: January 6, 2025*

## 🔑 **High-Coverage Input Fields Available**

### ✅ **Already Using**:
- `lat_lon` (100% coverage) 
- `collection_date` (96.1% coverage)

### 🤔 **Potential Additional Inputs**:
- `geo_loc_name` (99.8% coverage) - Geographic location names
- `isoCountry` (98.3% coverage) - ISO country codes  
- `elev` (84.7% coverage) - Elevation data
- `depth` (77.8% coverage) - Sample depth

## 💡 **Limited Leverage from Other Inputs**

**Geographic location names** (`geo_loc_name`) could provide some value:
- Input: "Cambridge, Massachusetts, USA" 
- APIs: Time zone lookup, administrative boundaries, postal codes
- **But**: Forward geocoding already converts this to lat/lon

**ISO country codes** (`isoCountry`) have narrow utility:
- Input: "USA", "CAN", "BRA"
- APIs: Country-specific datasets, administrative hierarchies
- **But**: Much less precise than coordinates

**Elevation** (`elev`) as input is very limited:
- Few APIs accept elevation as a lookup key
- Most useful as validation against elevation APIs

**Depth** for marine/soil samples:
- Input: Depth + coordinates could enhance soil layer APIs
- Marine depth could improve oceanographic lookups
- **But**: Minimal additional APIs beyond what coordinates provide

## 🎯 **Bottom Line**

**Coordinates + date provide ~95% of the leverage** for API enrichment. Other high-coverage fields offer minimal additional API opportunities because:

1. **Geographic names** → Forward geocoding already handles this
2. **Country codes** → Too coarse for most environmental APIs  
3. **Elevation/depth** → Very few APIs use these as primary lookup keys

The real opportunities lie in **expanding what we do with lat/lon/date**, not finding alternative input types. The schema shows that most other fields have <50% coverage, making them unsuitable as universal lookup keys.

## 📊 **Coverage Reality Check**

| Input Field | Coverage | API Leverage | Notes |
|-------------|----------|--------------|-------|
| `lat_lon` | 100% | **Highest** | Universal geospatial lookup key |
| `collection_date` | 96.1% | **High** | Historical weather, temporal context |
| `geo_loc_name` | 99.8% | **Low** | Redundant with forward geocoding |
| `isoCountry` | 98.3% | **Low** | Too coarse for environmental APIs |
| `elev` | 84.7% | **Minimal** | Few APIs use elevation as input |
| `depth` | 77.8% | **Minimal** | Limited depth-based APIs |

## 🚀 **Recommendation**

Focus API testing expansion on **maximizing lat/lon/date utilization** rather than seeking alternative input types. The current coordinate-centric approach provides the highest return on investment for biosample enrichment.
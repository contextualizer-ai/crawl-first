"""
Comprehensive weather API comparison for environmental microbiome research.

Tests multiple weather APIs to assess data richness, microbiome relevance,
and plant host biology parameters across different environmental contexts.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests


def load_local_env():
    """Load environment variables from local/.env file."""
    env_file = Path(__file__).parent.parent / "local" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


# Load local environment variables
load_local_env()


def call_open_meteo_comprehensive(lat: float, lon: float, date: str) -> Dict[str, Any]:
    """Call Open-Meteo with comprehensive environmental parameters using multiple API calls."""
    url = "https://archive-api.open-meteo.com/v1/archive"

    # PASS 1: Basic temperature and precipitation (known working parameters)
    pass1_params = [
        "temperature_2m_max",
        "temperature_2m_min",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
    ]

    # PASS 2: Wind, pressure, and humidity (known working parameters)
    pass2_params = [
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        "relative_humidity_2m_mean",
        "dewpoint_2m_mean",
        "surface_pressure_mean",
        "weather_code",
    ]

    # PASS 3: Solar radiation and plant biology
    pass3_params = [
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "sunshine_duration",
        "daylight_duration",
    ]

    # PASS 4: Advanced atmospheric parameters
    pass4_params = [
        "vapour_pressure_deficit_mean",
        "cloudcover_mean",
        "precipitation_hours",
    ]

    # PASS 5: Soil parameters (microbiome-specific)
    pass5_params = ["soil_temperature_0_to_7cm_mean", "soil_moisture_0_to_7cm_mean"]

    base_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "timezone": "UTC",
        "models": "best_match",
        "cell_selection": "nearest",
        "format": "json",
        "elevation": "nan",
        "temperature_unit": "celsius",
        "windspeed_unit": "kmh",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
    }

    combined_data = {}
    all_daily_data = {}
    metadata = {}
    total_successful_params = 0
    pass_results = []

    # Execute multiple passes
    parameter_sets = [
        ("Core Atmospheric", pass1_params),
        ("Solar & Plant Biology", pass2_params),
        ("Soil & Microbiome", pass3_params),
    ]

    for pass_name, param_set in parameter_sets:
        try:
            params = base_params.copy()
            params["daily"] = ",".join(param_set)

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract daily data from this pass
            pass_daily = data.get("daily", {})
            successful_params = len(
                [
                    k
                    for k, v in pass_daily.items()
                    if k != "time" and v and v[0] is not None
                ]
            )

            # Merge into combined dataset
            for key, values in pass_daily.items():
                if key != "time":
                    all_daily_data[key] = values

            # Store metadata from first successful pass
            if not metadata:
                metadata = {
                    "latitude": data.get("latitude", lat),
                    "longitude": data.get("longitude", lon),
                    "elevation": data.get("elevation"),
                    "timezone": data.get("timezone"),
                    "utc_offset_seconds": data.get("utc_offset_seconds"),
                    "generation_time_ms": data.get("generationtime_ms"),
                }

            total_successful_params += successful_params
            pass_results.append(
                {
                    "pass_name": pass_name,
                    "requested_params": len(param_set),
                    "successful_params": successful_params,
                    "success": True,
                }
            )

        except Exception as e:
            pass_results.append(
                {
                    "pass_name": pass_name,
                    "requested_params": len(param_set),
                    "successful_params": 0,
                    "success": False,
                    "error": str(e),
                }
            )

    # Check if at least one pass succeeded
    if total_successful_params == 0:
        return {
            "api": "Open-Meteo Multi-Pass",
            "success": False,
            "error": "All API passes failed",
            "data_source": "Open-Meteo Historical Weather API",
            "pass_results": pass_results,
        }

    # Calculate coordinate distance using metadata from successful pass
    returned_lat = metadata.get("latitude", lat)
    returned_lon = metadata.get("longitude", lon)

    def haversine_distance(lat1, lon1, lat2, lon2):
        import math

        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    coord_distance_km = haversine_distance(lat, lon, returned_lat, returned_lon)

    return {
        "api": "Open-Meteo Multi-Pass",
        "success": True,
        "data_source": "Open-Meteo Historical Weather API (Multi-Pass)",
        "total_parameters": total_successful_params,
        "daily_data": all_daily_data,
        "coordinate_distance_km": round(coord_distance_km, 3),
        "coordinate_adjustment": {
            "requested": {"lat": lat, "lon": lon},
            "actual": {"lat": returned_lat, "lon": returned_lon},
            "distance_km": round(coord_distance_km, 3),
            "data_quality": (
                "excellent"
                if coord_distance_km < 1
                else "good" if coord_distance_km < 5 else "moderate"
            ),
        },
        "elevation_meters": metadata.get("elevation"),
        "metadata": metadata,
        "multi_pass_results": pass_results,
        "successful_passes": len([p for p in pass_results if p["success"]]),
        "total_passes": len(pass_results),
        "schema_coverage": {
            "gold_elevation": metadata.get("elevation") is not None,
            "nmdc_temp": "mean_temperature_2m" in all_daily_data,
            "nmdc_humidity": "mean_relative_humidity_2m" in all_daily_data,
            "nmdc_wind": "wind_speed_10m_max" in all_daily_data,
            "nmdc_solar": "shortwave_radiation_sum" in all_daily_data,
            "gold_pressure": "mean_surface_pressure" in all_daily_data,
            "microbiome_soil": "soil_temperature_0_to_7cm_mean" in all_daily_data,
        },
    }


# Google Weather API removed - only provides 24 hours of historical data
# which is insufficient for environmental microbiome research needs


def call_open_meteo_historical_forecast(
    lat: float, lon: float, date: str
) -> Dict[str, Any]:
    """Call Open-Meteo Historical Forecast API for comprehensive microbiome-relevant parameters."""
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    # Comprehensive microbiome-relevant parameters from Historical Forecast API
    # These include the critical missing parameters: humidity, pressure, UV, detailed environmental data
    daily_params = [
        # Temperature (basic)
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "apparent_temperature_mean",
        # Humidity (MISSING from Historical Weather API)
        "relative_humidity_2m_mean",
        "relative_humidity_2m_max",
        "relative_humidity_2m_min",
        # Pressure (MISSING from Historical Weather API)
        "surface_pressure_mean",
        # Precipitation
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        # Wind
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        # Solar and UV (enhanced from Historical Weather API)
        "shortwave_radiation_sum",
        "uv_index_max",
        "sunshine_duration",
        "daylight_duration",
        # Atmospheric conditions (MISSING from Historical Weather API)
        "dewpoint_2m_mean",
        "vapour_pressure_deficit_max",
        "cloudcover_mean",
        "visibility_mean",
        # Plant biology
        "et0_fao_evapotranspiration",
        # Atmospheric physics
        "weather_code",
    ]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": ",".join(daily_params),
        "timezone": "UTC",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "daily" in data and data["daily"]:
            daily = data["daily"]

            # Extract daily data
            weather_data = {}
            for key, values in daily.items():
                if (
                    key != "time"
                    and values
                    and len(values) > 0
                    and values[0] is not None
                ):
                    weather_data[key] = values[0]

            # Get coordinate metadata (Historical Forecast API may also adjust coordinates)
            returned_lat = data.get("latitude", lat)
            returned_lon = data.get("longitude", lon)
            returned_elevation = data.get("elevation")

            # Calculate coordinate distance
            import math

            def haversine_distance(lat1, lon1, lat2, lon2):
                R = 6371
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = (
                    math.sin(dlat / 2) ** 2
                    + math.cos(math.radians(lat1))
                    * math.cos(math.radians(lat2))
                    * math.sin(dlon / 2) ** 2
                )
                c = 2 * math.asin(math.sqrt(a))
                return R * c

            coord_distance_km = haversine_distance(lat, lon, returned_lat, returned_lon)

            return {
                "api": "Open-Meteo Historical Forecast",
                "success": True,
                "data_source": "Open-Meteo Historical Forecast API (High-resolution archived forecasts)",
                "total_parameters": len(weather_data),
                "daily_data": weather_data,
                "coordinate_distance_km": round(coord_distance_km, 3),
                "coordinate_adjustment": {
                    "requested": {"lat": lat, "lon": lon},
                    "actual": {"lat": returned_lat, "lon": returned_lon},
                    "distance_km": round(coord_distance_km, 3),
                    "data_quality": (
                        "excellent"
                        if coord_distance_km < 1
                        else "good" if coord_distance_km < 5 else "moderate"
                    ),
                },
                "elevation_meters": returned_elevation,
                "date": date,
                "coverage_period": "~2021/2022 to present (archived forecast data)",
                "advantages": [
                    "Includes humidity, pressure, UV index - missing from Historical Weather API",
                    "High-resolution forecast model output with comprehensive variables",
                    "Better microbiome research parameter coverage",
                    "Includes atmospheric physics parameters (dewpoint, VPD, visibility)",
                ],
                "schema_coverage": {
                    "gold_elevation": returned_elevation is not None,
                    "nmdc_humidity": "relative_humidity_2m_mean" in weather_data,
                    "nmdc_pressure": "surface_pressure_mean" in weather_data,
                    "nmdc_uv": "uv_index_max" in weather_data,
                    "microbiome_dewpoint": "dewpoint_2m_mean" in weather_data,
                    "microbiome_vpd": "vapour_pressure_deficit_max" in weather_data,
                    "microbiome_visibility": "visibility_mean" in weather_data,
                },
            }
        else:
            return {
                "weather_data": {},
                "success": False,
                "data_source": "Open-Meteo Historical Forecast API",
            }

    except Exception as e:
        return {
            "weather_data": {},
            "error": str(e),
            "data_source": "Open-Meteo Historical Forecast API",
            "success": False,
        }


def call_meteostat_comprehensive(lat: float, lon: float, date: str) -> Dict[str, Any]:
    """Call Meteostat library directly for comprehensive station-based weather data with station distances."""
    try:
        # Import meteostat (should be available via weather-context-mcp dependency)
        import math
        from datetime import datetime, timedelta

        import pandas as pd
        from meteostat import Daily, Hourly, Point, Stations

        # Convert date string to datetime
        dt = datetime.strptime(date, "%Y-%m-%d")

        # Create Point object for location with enhanced radius for comprehensive station search
        location = Point(lat, lon, alt=None)

        # Get nearby weather stations with comprehensive search radius (expanded for maximum coverage)
        stations = Stations()
        stations = stations.nearby(
            lat, lon, radius=200000
        )  # 200km radius for maximum coverage
        stations = stations.fetch(
            limit=20
        )  # Get top 20 closest stations for better analysis

        # Haversine distance calculation function
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth's radius in kilometers
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(math.radians(lat1))
                * math.cos(math.radians(lat2))
                * math.sin(dlon / 2) ** 2
            )
            c = 2 * math.asin(math.sqrt(a))
            return R * c

        # Enhanced station information with distances
        station_details = []
        closest_station_distance = float("inf")
        closest_station_id = None

        if not stations.empty:
            for idx, station in stations.iterrows():
                distance = haversine_distance(
                    lat, lon, station.get("latitude", 0), station.get("longitude", 0)
                )
                station_info = {
                    "id": idx,
                    "name": station.get("name", "Unknown"),
                    "country": station.get("country", "Unknown"),
                    "region": station.get("region", "Unknown"),
                    "latitude": station.get("latitude"),
                    "longitude": station.get("longitude"),
                    "elevation": station.get("elevation"),
                    "distance_km": round(distance, 2),
                    "data_period": {
                        "start": (
                            str(station.get("hourly_start"))
                            if station.get("hourly_start")
                            else None
                        ),
                        "end": (
                            str(station.get("hourly_end"))
                            if station.get("hourly_end")
                            else None
                        ),
                    },
                }
                station_details.append(station_info)

                if distance < closest_station_distance:
                    closest_station_distance = distance
                    closest_station_id = idx

        # Get daily data with enhanced configuration for maximum coverage and quality
        daily_data = Daily(location, dt, dt)
        daily_data = daily_data.normalize()  # Apply quality controls and gap filling
        daily_data = daily_data.interpolate()  # Fill gaps with interpolated data
        daily_df = daily_data.fetch()

        # Get hourly data with enhanced configuration for maximum temporal resolution
        hourly_data = Hourly(location, dt, dt.replace(hour=23, minute=59))
        hourly_data = hourly_data.normalize()  # Apply quality controls
        hourly_data = hourly_data.interpolate()  # Fill gaps with interpolated data
        hourly_df = hourly_data.fetch()

        # Additional data coverage analysis
        coverage_daily = daily_data.coverage()
        coverage_hourly = hourly_data.coverage()

        # Extended date range for data availability assessment
        extended_start = dt - timedelta(days=7)
        extended_end = dt + timedelta(days=7)
        extended_daily = Daily(location, extended_start, extended_end)
        extended_daily_df = extended_daily.fetch()

        # Count available parameters
        daily_params = len(
            [col for col in daily_df.columns if not daily_df[col].isna().all()]
        )
        hourly_params = len(
            [col for col in hourly_df.columns if not hourly_df[col].isna().all()]
        )

        # Convert DataFrames to dict for JSON serialization
        daily_dict = daily_df.to_dict("records")[0] if not daily_df.empty else {}
        hourly_dict = hourly_df.to_dict("records") if not hourly_df.empty else []

        # Clean NaN values for JSON serialization
        def clean_nans(obj):
            if isinstance(obj, dict):
                return {k: (None if pd.isna(v) else v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nans(item) for item in obj]
            return obj

        daily_dict = clean_nans(daily_dict)
        hourly_dict = clean_nans(hourly_dict)

        # Enhanced data quality assessment
        data_completeness = {
            "daily_completeness": round(
                (daily_params / max(1, len(daily_df.columns))) * 100, 1
            ),
            "hourly_completeness": round(
                (hourly_params / max(1, len(hourly_df.columns))) * 100, 1
            ),
            "extended_period_days": len(extended_daily_df),
            "data_gaps": len(extended_daily_df)
            - len(
                [
                    i
                    for i in extended_daily_df.index
                    if not extended_daily_df.loc[i].isna().all()
                ]
            ),
            "daily_coverage": coverage_daily if "coverage_daily" in locals() else None,
            "hourly_coverage": (
                coverage_hourly if "coverage_hourly" in locals() else None
            ),
        }

        # Add detailed parameter breakdown for microbiome research
        available_params = {
            "temperature": ["tavg", "tmin", "tmax"] if not daily_df.empty else [],
            "precipitation": (
                ["prcp"] if not daily_df.empty and "prcp" in daily_df.columns else []
            ),
            "wind": ["wspd", "wdir", "wpgt"] if not daily_df.empty else [],
            "pressure": (
                ["pres"] if not daily_df.empty and "pres" in daily_df.columns else []
            ),
            "sunshine": (
                ["tsun"] if not daily_df.empty and "tsun" in daily_df.columns else []
            ),
            "snow": (
                ["snow"] if not daily_df.empty and "snow" in daily_df.columns else []
            ),
        }

        microbiome_relevant_count = sum(
            len(params) for params in available_params.values()
        )

        return {
            "api": "Meteostat",
            "success": True,
            "data_source": "Meteostat (Multi-source weather stations)",
            "daily_parameters": daily_params,
            "hourly_parameters": hourly_params,
            "total_parameters": daily_params + hourly_params,
            "microbiome_relevant_parameters": microbiome_relevant_count,
            "parameter_categories": available_params,
            "daily_data": daily_dict,
            "hourly_data": hourly_dict,
            "station_distance_km": (
                round(closest_station_distance, 3)
                if closest_station_distance != float("inf")
                else None
            ),
            "closest_station": {
                "id": closest_station_id,
                "distance_km": (
                    round(closest_station_distance, 3)
                    if closest_station_distance != float("inf")
                    else None
                ),
                "data_quality": (
                    "excellent"
                    if closest_station_distance < 5
                    else (
                        "good"
                        if closest_station_distance < 25
                        else "moderate" if closest_station_distance < 100 else "poor"
                    )
                ),
            },
            "stations_nearby": station_details[:5],  # Top 5 closest stations
            "total_stations_found": len(station_details),
            "location": {"latitude": lat, "longitude": lon},
            "date": date,
            "data_quality": {
                "source_type": "weather_stations",
                "station_based": True,
                "daily_records": len(daily_df),
                "hourly_records": len(hourly_df),
                "quality_controlled": True,
                "interpolated": True,
                "completeness": data_completeness,
                "spatial_representativeness": (
                    "high"
                    if closest_station_distance < 10
                    else "medium" if closest_station_distance < 50 else "low"
                ),
                "available_parameter_types": list(available_params.keys()),
            },
        }

    except ImportError:
        return {
            "api": "Meteostat",
            "success": False,
            "error": "Meteostat library not available - install via 'pip install meteostat'",
            "data_source": "Meteostat",
        }
    except Exception as e:
        return {
            "api": "Meteostat",
            "success": False,
            "error": str(e),
            "data_source": "Meteostat",
        }


def assess_microbiome_relevance(api_result: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the microbiome research relevance of API data."""
    if not api_result.get("success"):
        return {"relevance_score": 0, "missing_critical": "API failed"}

    # Critical parameters for environmental microbiome research
    critical_params = {
        "temperature": ["temp", "temperature", "temp_c"],
        "humidity": ["humidity", "relative_humidity", "rh"],
        "precipitation": ["precip", "precipitation", "rain"],
        "soil_temperature": ["soil_temp", "soil_temperature"],
        "soil_moisture": ["soil_moisture", "soil_water"],
        "solar_radiation": ["solar", "radiation", "shortwave"],
        "uv_index": ["uv", "uvindex"],
        "wind": ["wind", "windspeed"],
        "pressure": ["pressure", "surface_pressure"],
        "air_quality": ["aqi", "air_quality", "pm2_5", "pm10", "co", "no2", "o3"],
    }

    # Plant host biology parameters
    plant_params = {
        "evapotranspiration": ["et0", "evapotranspiration"],
        "growing_degree_days": ["gdd", "growing_degree"],
        "leaf_wetness": ["leaf_wetness", "wetness"],
        "daylight": ["daylight", "sunshine"],
        "frost": ["frost", "freeze"],
    }

    # Microbial ecology parameters
    microbial_params = {
        "dewpoint": ["dew", "dewpoint"],
        "cloud_cover": ["cloud", "cloudcover"],
        "visibility": ["visibility"],
        "atmospheric_chemistry": ["o3", "co", "no2", "so2"],
    }

    def find_param_in_data(param_names: List[str], data: Dict[str, Any]) -> bool:
        """Check if any parameter names exist in the data structure."""
        data_str = json.dumps(data).lower()
        return any(param in data_str for param in param_names)

    # Score each category
    critical_score = sum(
        1
        for param_list in critical_params.values()
        if find_param_in_data(param_list, api_result)
    )
    plant_score = sum(
        1
        for param_list in plant_params.values()
        if find_param_in_data(param_list, api_result)
    )
    microbial_score = sum(
        1
        for param_list in microbial_params.values()
        if find_param_in_data(param_list, api_result)
    )

    total_possible = len(critical_params) + len(plant_params) + len(microbial_params)
    overall_score = (critical_score + plant_score + microbial_score) / total_possible

    return {
        "relevance_score": round(overall_score, 3),
        "critical_environmental_params": f"{critical_score}/{len(critical_params)}",
        "plant_host_biology_params": f"{plant_score}/{len(plant_params)}",
        "microbial_ecology_params": f"{microbial_score}/{len(microbial_params)}",
        "total_microbiome_relevance": f"{critical_score + plant_score + microbial_score}/{total_possible}",
        "strengths": [],
        "gaps": [],
    }


class TestWeatherAPIMicrobiomeComparison:
    """Compare weather APIs for environmental microbiome research data richness."""

    def test_microbiome_research_locations(self):
        """Test weather APIs on locations relevant to microbiome research."""
        test_dates = [
            "2023-06-15",  # Summer date for plant activity
            "2023-01-15",  # Winter date for seasonal comparison
        ]

        microbiome_locations = [
            {
                "name": "Temperate Forest (Germany)",
                "lat": 50.7753,
                "lon": 6.0839,
                "ecosystem": "temperate deciduous forest",
                "microbiome_context": "soil and rhizosphere microbiomes",
            },
            {
                "name": "Agricultural Field (Iowa)",
                "lat": 42.0308,
                "lon": -93.6319,
                "ecosystem": "agricultural cropland",
                "microbiome_context": "crop rhizosphere and soil microbiomes",
            },
            {
                "name": "Coastal Marine (California)",
                "lat": 36.6002,
                "lon": -121.8947,
                "ecosystem": "marine coastal",
                "microbiome_context": "marine and coastal sediment microbiomes",
            },
            {
                "name": "Alpine Ecosystem (Colorado)",
                "lat": 40.0583,
                "lon": -105.6836,
                "ecosystem": "alpine tundra",
                "microbiome_context": "extreme environment microbiomes",
            },
        ]

        print("\n=== WEATHER API MICROBIOME RESEARCH COMPARISON ===")
        print(f"Test Dates: {', '.join(test_dates)}")
        print("=" * 80)

        api_summary = {}

        for test_date in test_dates:
            print(f"\n🗓️  TESTING DATE: {test_date}")
            print("=" * 60)

            for location in microbiome_locations:
                lat, lon = location["lat"], location["lon"]

                print(f"\n🌍 {location['name']} ({lat}, {lon})")
                print(f"   Ecosystem: {location['ecosystem']}")
                print(f"   Microbiome Context: {location['microbiome_context']}")
                print("-" * 60)

                # Test three APIs with different strengths for microbiome research
                apis = [
                    ("Open-Meteo Historical Weather", call_open_meteo_comprehensive),
                    (
                        "Open-Meteo Historical Forecast",
                        call_open_meteo_historical_forecast,
                    ),
                    ("Meteostat", call_meteostat_comprehensive),
                ]

                location_results = {}

                for api_name, api_func in apis:
                    print(f"\n  🔍 Testing {api_name}...")
                    result = api_func(lat, lon, test_date)

                    if result["success"]:
                        microbiome_assessment = assess_microbiome_relevance(result)
                        total_params = result.get("total_parameters", 0)

                        # Extract distance information
                        if "Open-Meteo Historical Weather" in api_name:
                            distance_km = result.get("coordinate_distance_km", 0)
                            coord_quality = result.get("coordinate_adjustment", {}).get(
                                "data_quality", "unknown"
                            )
                            successful_passes = result.get("successful_passes", 0)
                            total_passes = result.get("total_passes", 0)
                            distance_info = f"📍 Grid adjustment: {distance_km}km ({coord_quality}) | {successful_passes}/{total_passes} passes"
                        elif "Open-Meteo Historical Forecast" in api_name:
                            distance_km = result.get("coordinate_distance_km", 0)
                            coord_quality = result.get("coordinate_adjustment", {}).get(
                                "data_quality", "unknown"
                            )
                            coverage_period = result.get("coverage_period", "Unknown")
                            distance_info = f"📡 Forecast grid: {distance_km}km ({coord_quality}) | {coverage_period}"
                        elif api_name == "Meteostat":
                            distance_km = result.get("station_distance_km", 0)
                            station_quality = result.get("closest_station", {}).get(
                                "data_quality", "unknown"
                            )
                            stations_found = result.get("total_stations_found", 0)
                            distance_info = f"🏢 Closest station: {distance_km}km ({station_quality}) | {stations_found} stations found"
                        else:
                            distance_info = "📍 Distance: Unknown"

                        print(f"    ✅ Success: {total_params} parameters")
                        print(f"    {distance_info}")
                        print(
                            f"    🧬 Microbiome Relevance: {microbiome_assessment['relevance_score']}"
                        )
                        print(
                            f"    📊 Critical/Plant/Microbial: {microbiome_assessment['critical_environmental_params']} | {microbiome_assessment['plant_host_biology_params']} | {microbiome_assessment['microbial_ecology_params']}"
                        )

                        # Show multi-pass details for Open-Meteo
                        if api_name == "Open-Meteo" and "multi_pass_results" in result:
                            print("    📋 Pass Details:")
                            for pass_result in result["multi_pass_results"]:
                                status = "✅" if pass_result["success"] else "❌"
                                print(
                                    f"      {status} {pass_result['pass_name']}: {pass_result['successful_params']}/{pass_result['requested_params']} params"
                                )

                        location_results[api_name] = {
                            "success": True,
                            "total_parameters": total_params,
                            "microbiome_relevance": microbiome_assessment,
                            "distance_km": (
                                distance_km if "distance_km" in locals() else None
                            ),
                        }

                        # Track API performance across locations and dates
                        if api_name not in api_summary:
                            api_summary[api_name] = {
                                "successes": 0,
                                "total_params": [],
                                "relevance_scores": [],
                            }
                        api_summary[api_name]["successes"] += 1
                        api_summary[api_name]["total_params"].append(total_params)
                        api_summary[api_name]["relevance_scores"].append(
                            microbiome_assessment["relevance_score"]
                        )

                    else:
                        print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
                        location_results[api_name] = {
                            "success": False,
                            "error": result.get("error"),
                        }

        # Print comprehensive comparison summary
        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE API COMPARISON FOR MICROBIOME RESEARCH")
        print("=" * 80)

        total_tests = len(test_dates) * len(microbiome_locations)

        for api_name, summary in api_summary.items():
            if summary["successes"] > 0:
                avg_params = sum(summary["total_params"]) / len(summary["total_params"])
                avg_relevance = sum(summary["relevance_scores"]) / len(
                    summary["relevance_scores"]
                )

                print(f"\n📊 {api_name}:")
                print(
                    f"   Success Rate: {summary['successes']}/{total_tests} tests ({len(test_dates)} dates × {len(microbiome_locations)} locations)"
                )
                print(f"   Average Parameters: {avg_params:.1f}")
                print(f"   Average Microbiome Relevance: {avg_relevance:.3f}")
                print(
                    f"   Parameter Range: {min(summary['total_params'])}-{max(summary['total_params'])}"
                )
                print(
                    f"   Relevance Range: {min(summary['relevance_scores']):.3f}-{max(summary['relevance_scores']):.3f}"
                )

        print("\n🔬 MICROBIOME RESEARCH RECOMMENDATIONS:")
        print("1. Data Richness: Prioritize APIs with highest parameter counts")
        print(
            "2. Environmental Coverage: Ensure critical microbiome parameters are included"
        )
        print(
            "3. Plant Host Biology: Important for rhizosphere and phyllosphere studies"
        )
        print(
            "4. Spatial Resolution: Consider coordinate precision for microhabitat studies"
        )
        print(
            "5. Temporal Resolution: Hourly data captures microbial response dynamics"
        )

        # Ensure at least one API worked
        assert any(
            summary["successes"] > 0 for summary in api_summary.values()
        ), "No weather APIs succeeded"

    def test_api_data_structure_analysis(self):
        """Analyze the detailed data structures returned by each API for microbiome relevance."""
        test_location = {
            "lat": 40.7128,
            "lon": -74.0060,
            "name": "New York (Urban Microbiome)",
        }
        test_date = "2023-06-15"

        print("\n" + "=" * 80)
        print("🔬 DETAILED DATA STRUCTURE ANALYSIS FOR MICROBIOME RESEARCH")
        print("=" * 80)
        print(
            f"Location: {test_location['name']} ({test_location['lat']}, {test_location['lon']})"
        )
        print(f"Date: {test_date}")

        apis = [
            ("Open-Meteo Comprehensive", call_open_meteo_comprehensive),
            ("Meteostat", call_meteostat_comprehensive),
        ]

        for api_name, api_func in apis:
            print(f"\n{'='*20} {api_name} {'='*20}")
            result = api_func(test_location["lat"], test_location["lon"], test_date)

            if result["success"]:
                # Analyze data structure depth and breadth
                def analyze_data_structure(
                    data, prefix="", max_depth=3, current_depth=0
                ):
                    """Recursively analyze data structure."""
                    items = []
                    if current_depth >= max_depth:
                        return items

                    if isinstance(data, dict):
                        for key, value in data.items():
                            if (
                                isinstance(value, (dict, list))
                                and len(str(value)) > 100
                            ):
                                items.append(f"{prefix}{key}: [complex structure]")
                                if current_depth < max_depth - 1:
                                    items.extend(
                                        analyze_data_structure(
                                            value,
                                            f"{prefix}  ",
                                            max_depth,
                                            current_depth + 1,
                                        )
                                    )
                            else:
                                items.append(f"{prefix}{key}: {type(value).__name__}")
                    elif isinstance(data, list) and data:
                        items.append(
                            f"{prefix}[list of {len(data)} {type(data[0]).__name__} items]"
                        )
                        if isinstance(data[0], dict) and current_depth < max_depth - 1:
                            items.extend(
                                analyze_data_structure(
                                    data[0], f"{prefix}  ", max_depth, current_depth + 1
                                )
                            )

                    return items

                structure = analyze_data_structure(result)
                print("📋 Data Structure:")
                for item in structure[:30]:  # Limit output
                    print(f"  {item}")
                if len(structure) > 30:
                    print(f"  ... and {len(structure) - 30} more items")

                # Microbiome relevance assessment
                relevance = assess_microbiome_relevance(result)
                print("\n🧬 Microbiome Research Assessment:")
                print(f"  Overall Relevance Score: {relevance['relevance_score']}")
                print(
                    f"  Critical Environmental: {relevance['critical_environmental_params']}"
                )
                print(f"  Plant Host Biology: {relevance['plant_host_biology_params']}")
                print(f"  Microbial Ecology: {relevance['microbial_ecology_params']}")

            else:
                print(f"❌ API Failed: {result.get('error')}")

    def test_api_cost_benefit_analysis(self):
        """Analyze cost vs. benefit for each API for microbiome research budgets."""
        print("\n" + "=" * 80)
        print("💰 COST-BENEFIT ANALYSIS FOR MICROBIOME RESEARCH")
        print("=" * 80)

        api_profiles = {
            "Open-Meteo": {
                "cost": "Free",
                "limits": "10,000 requests/day",
                "microbiome_strengths": [
                    "Comprehensive soil parameters",
                    "High temporal resolution",
                    "Global coverage",
                    "Plant biology data",
                ],
                "microbiome_gaps": ["No air quality", "Limited atmospheric chemistry"],
                "research_fit": "Excellent for soil/plant microbiome studies",
            },
            "Meteostat": {
                "cost": "Free (library)",
                "limits": "No API limits (local processing)",
                "microbiome_strengths": [
                    "Station-based accuracy",
                    "Long historical records",
                    "Quality-controlled data",
                ],
                "microbiome_gaps": [
                    "Limited soil parameters",
                    "Depends on station density",
                ],
                "research_fit": "Excellent for high-quality historical climate context",
            },
        }

        for api_name, profile in api_profiles.items():
            print(f"\n📊 {api_name}")
            print(f"  💵 Cost: {profile['cost']}")
            print(f"  📏 Limits: {profile['limits']}")
            print(
                f"  ✅ Microbiome Strengths: {', '.join(profile['microbiome_strengths'])}"
            )
            print(f"  ⚠️  Microbiome Gaps: {', '.join(profile['microbiome_gaps'])}")
            print(f"  🎯 Research Fit: {profile['research_fit']}")

        print("\n🏆 RECOMMENDATIONS FOR MICROBIOME RESEARCH:")
        print("1. 🥇 PRIMARY: Open-Meteo (free, comprehensive soil/plant parameters)")
        print(
            "2. 🥈 SECONDARY: Meteostat (free, high-quality station data for climate context)"
        )
        print(
            "\n💡 STRATEGY: Use Open-Meteo for comprehensive environmental data + Meteostat for historical climate validation"
        )
        print(
            "\n❌ EXCLUDED: Google Weather API (only 24 hours historical data - insufficient for research)"
        )

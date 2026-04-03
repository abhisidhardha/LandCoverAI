from __future__ import annotations

import logging
import calendar
import threading
import time
from typing import Dict, Tuple

import numpy as np
import requests


_CACHE: Dict[Tuple[float, float], Dict] = {}
_CACHE_TS: Dict[Tuple[float, float], float] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60


def _cache_key(lat: float, lon: float) -> Tuple[float, float]:
    return round(float(lat), 4), round(float(lon), 4)


def _get_cached_value(key: Tuple[float, float]):
    with _CACHE_LOCK:
        value = _CACHE.get(key)
        timestamp = _CACHE_TS.get(key)
    if value is None or timestamp is None:
        return None
    if time.time() - timestamp > _CACHE_TTL_SECONDS:
        return None
    return dict(value)


def _store_cached_value(key: Tuple[float, float], value: Dict) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = dict(value)
        _CACHE_TS[key] = time.time()


def _mean_from_series(series) -> float | None:
    if not isinstance(series, dict) or not series:
        return None
    values = []
    for raw_value in series.values():
        try:
            numeric = float(raw_value)
        except Exception:
            continue
        if numeric == -999:
            continue
        values.append(numeric)
    if not values:
        return None
    return round(float(np.mean(values)), 1)


def _parse_month_key(key: str) -> Tuple[int, int] | None:
    text = str(key)
    if len(text) >= 6 and text[:6].isdigit():
        year = int(text[:4])
        month = int(text[4:6])
        if 1 <= month <= 12:
            return year, month
    return None


def _annual_rainfall_from_monthly(series: dict) -> float | None:
    if not isinstance(series, dict) or not series:
        return None

    yearly_totals: Dict[int, float] = {}
    for key, value in series.items():
        ym = _parse_month_key(str(key))
        if ym is None:
            continue
        year, month = ym
        try:
            mm_per_day = float(value)
        except Exception:
            continue
        if mm_per_day == -999:
            continue
        month_days = calendar.monthrange(year, month)[1]
        yearly_totals[year] = yearly_totals.get(year, 0.0) + (mm_per_day * month_days)

    if not yearly_totals:
        return None
    return round(float(np.mean(list(yearly_totals.values()))), 1)


def _fetch_nasa_features(lat: float, lon: float) -> Dict:
    out = {
        "rainfall_mm": None,
        "temp_avg": None,
        "temp_min": None,
        "temp_max": None,
    }

    annual_url = (
        "https://power.larc.nasa.gov/api/temporal/annual/point"
        f"?parameters=PRECTOTCORR,T2M,T2M_MAX,T2M_MIN"
        f"&community=AG&longitude={lon}&latitude={lat}&format=JSON&start=2020&end=2023"
    )

    monthly_url = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters=PRECTOTCORR,T2M,T2M_MAX,T2M_MIN"
        f"&community=AG&longitude={lon}&latitude={lat}&format=JSON&start=2020&end=2023"
    )

    params = {}
    try:
        response = requests.get(annual_url, timeout=15)
        if response.ok:
            params = response.json().get("properties", {}).get("parameter", {})
            out["rainfall_mm"] = _mean_from_series(params.get("PRECTOTCORR"))
            out["temp_avg"] = _mean_from_series(params.get("T2M"))
            out["temp_max"] = _mean_from_series(params.get("T2M_MAX"))
            out["temp_min"] = _mean_from_series(params.get("T2M_MIN"))
            return out
        logging.info("NASA annual endpoint unavailable (%s). Falling back to monthly aggregation.", response.status_code)
    except Exception as exc:
        logging.info("NASA annual lookup failed; falling back to monthly aggregation: %s", exc)

    response = requests.get(monthly_url, timeout=15)
    response.raise_for_status()
    params = response.json().get("properties", {}).get("parameter", {})

    out["rainfall_mm"] = _annual_rainfall_from_monthly(params.get("PRECTOTCORR"))
    out["temp_avg"] = _mean_from_series(params.get("T2M"))
    out["temp_max"] = _mean_from_series(params.get("T2M_MAX"))
    out["temp_min"] = _mean_from_series(params.get("T2M_MIN"))
    return out


def _map_soil_type(wrb_class: str | None) -> str:
    if not wrb_class:
        return "loamy"

    soil_map = {
        "Vertisols": "black",
        "Alfisols": "red",
        "Aridisols": "sandy",
        "Entisols": "sandy",
        "Inceptisols": "loamy",
        "Mollisols": "loamy",
        "Oxisols": "laterite",
        "Ultisols": "red",
        "clay": "clay",
        "sandy": "sandy",
        "loam": "loamy",
    }
    wrb_text = str(wrb_class).lower()
    for key, value in soil_map.items():
        if key.lower() in wrb_text:
            return value
    return "loamy"


def _derive_agro_zone(rainfall_mm: float | None, temp_avg: float | None) -> str | None:
    if rainfall_mm is None or temp_avg is None:
        return None
    if rainfall_mm > 2000:
        return "tropical_wet"
    if rainfall_mm > 1000 and temp_avg > 20:
        return "humid_subtropical"
    if rainfall_mm > 600:
        return "sub_humid"
    if rainfall_mm > 400:
        return "semi_arid"
    return "arid"


def get_location_features(lat: float, lon: float) -> Dict:
    """
    Fetch climate, soil, and elevation data for a location.
    All APIs used here are free and require no authentication.
    """
    key = _cache_key(lat, lon)
    cached = _get_cached_value(key)
    if cached is not None:
        return cached

    features = {
        "rainfall_mm": None,
        "temp_avg": None,
        "temp_min": None,
        "temp_max": None,
        "elevation_m": None,
        "soil_type": None,
        "agro_zone": None,
    }

    try:
        nasa = _fetch_nasa_features(lat, lon)
        features["rainfall_mm"] = nasa.get("rainfall_mm")
        features["temp_avg"] = nasa.get("temp_avg")
        features["temp_max"] = nasa.get("temp_max")
        features["temp_min"] = nasa.get("temp_min")
    except Exception as exc:
        logging.warning("NASA POWER climate lookup failed for %.4f, %.4f: %s", lat, lon, exc)

    try:
        elev_url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = requests.get(elev_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if results:
            elevation = results[0].get("elevation")
            if elevation is not None:
                features["elevation_m"] = int(round(float(elevation)))
    except Exception as exc:
        logging.warning("Open-Elevation lookup failed for %.4f, %.4f: %s", lat, lon, exc)

    try:
        soil_url = (
            "https://rest.isric.org/soilgrids/v2.0/classification/query"
            f"?lon={lon}&lat={lat}&number_classes=5"
        )
        response = requests.get(soil_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        wrb_class = (
            data.get("wrb_class_name")
            or data.get("wrb_class")
            or data.get("class_name")
            or data.get("properties", {}).get("wrb_class_name")
            or data.get("properties", {}).get("wrb_class")
        )
        features["soil_type"] = _map_soil_type(wrb_class)
    except Exception as exc:
        logging.warning("SoilGrids lookup failed for %.4f, %.4f: %s", lat, lon, exc)
        features["soil_type"] = "loamy"

    features["agro_zone"] = _derive_agro_zone(features.get("rainfall_mm"), features.get("temp_avg"))

    _store_cached_value(key, features)
    return dict(features)
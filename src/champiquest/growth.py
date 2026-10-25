"""Prédiction de pousse (partie 1) : météo d'un point → proba de fructification par espèce.

Utilise les 6 modèles LightGBM finaux (`data/models/growth_*.joblib`) et Open-Meteo
(forecast, `past_days=90`) pour couvrir « aujourd'hui » et les prochains jours (J+16 max).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

from champiquest.features.weather import compute_soil_features, compute_weather_features
from champiquest.models.evaluate import PRUNED_NUM_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/champiquest/growth.py → racine du repo
MODELS_DIR = REPO_ROOT / "data" / "models"
SOIL_POINTS = REPO_ROOT / "data" / "raw" / "soilgrids" / "soil_points.csv"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "growth"

# Les 6 espèces modélisées (mêmes clés que configs/species.yaml).
SPECIES = [
    "boletus_edulis",
    "macrolepiota_procera",
    "cantharellus_cibarius",
    "morchella",
    "imleria_badia",
    "boletus_reticulatus",
    "craterellus_cornucopioides",
    "hydnum_repandum",
]

GROWTH_VARIABLES = [
    "precipitation_sum",
    "temperature_2m_min",
    "temperature_2m_max",
    "temperature_2m_mean",
    "et0_fao_evapotranspiration",
    "soil_temperature_0_to_7cm_mean",
    "soil_moisture_0_to_7cm_mean",
]
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PAST_DAYS = 90
FORECAST_DAYS = 16

_models: dict = {}
_soil: pd.DataFrame | None = None


def _client() -> openmeteo_requests.Client:
    session = requests_cache.CachedSession(str(CACHE_DIR), expire_after=6 * 3600)  # cache 6 h
    return openmeteo_requests.Client(session=retry(session, retries=2, backoff_factor=1.0))


def _load_models() -> dict:
    global _models
    if not _models:
        _models = {sp: joblib.load(MODELS_DIR / f"growth_{sp}.joblib")["model"] for sp in SPECIES}
    return _models


def _load_soil() -> pd.DataFrame:
    global _soil
    if _soil is None:
        _soil = pd.read_csv(SOIL_POINTS)
    return _soil


def sand_nearest(lat: float, lon: float) -> float:
    """Valeur `sand_0-5cm` du point SoilGrids le plus proche (champ quasi statique)."""
    soil = _load_soil()
    d = (soil["lat"] - lat) ** 2 + (soil["lon"] - lon) ** 2
    return float(soil.loc[d.idxmin(), "sand_0-5cm"])


def fetch_weather(
    lat: float, lon: float, past_days: int = PAST_DAYS, forecast_days: int = FORECAST_DAYS
) -> pd.DataFrame:
    """Série quotidienne forecast (past_days → +forecast_days) au point donné."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": GROWTH_VARIABLES,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "Europe/Berlin",
    }
    resp = _client().weather_api(FORECAST_URL, params=params)[0]
    daily = resp.Daily()
    n = len(daily.Variables(0).ValuesAsNumpy())
    start = pd.Timestamp.now().normalize() - pd.Timedelta(days=past_days)
    dates = pd.date_range(start=start, periods=n, freq="D")
    data = {v: daily.Variables(i).ValuesAsNumpy() for i, v in enumerate(GROWTH_VARIABLES)}
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


def fetch_era5(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Série quotidienne ERA5 (réanalyse, précise, ~5 j de délai) pour une plage de dates."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": GROWTH_VARIABLES,
        "timezone": "Europe/Berlin",
    }
    resp = _client().weather_api(ARCHIVE_URL, params=params)[0]
    daily = resp.Daily()
    n = len(daily.Variables(0).ValuesAsNumpy())
    dates = pd.date_range(start=start_date, end=end_date, freq="D")[:n]
    data = {v: daily.Variables(i).ValuesAsNumpy() for i, v in enumerate(GROWTH_VARIABLES)}
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


def compute_features(daily: pd.DataFrame, date: str, lat: float, lon: float) -> pd.Series:
    """Calcule les 12 features `pruned` pour une date donnée."""
    obs = pd.Timestamp(date)
    feats = compute_weather_features(daily, obs)
    feats.update(compute_soil_features(daily, obs))
    feats["sand_0-5cm"] = sand_nearest(lat, lon)
    return pd.Series({f: feats[f] for f in PRUNED_NUM_FEATURES})


def predict_growth(lat: float, lon: float, date: str) -> dict:
    """Probabilités de fructification des 6 espèces pour (lat, lon, date)."""
    daily = fetch_weather(lat, lon)
    feats = compute_features(daily, date, lat, lon)
    X = pd.DataFrame([feats])[PRUNED_NUM_FEATURES]
    models = _load_models()
    return {sp: float(model.predict_proba(X)[0, 1]) for sp, model in models.items()}

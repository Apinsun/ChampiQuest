"""Store météo persistant (Parquet) : accumulation + lecture + déduplication.

Table plate : une ligne par (point, jour, source), colonnes :
`lat, lon, date, source, precipitation_sum, temperature_2m_min/max/mean,
et0_fao_evapotranspiration, soil_temperature_0_to_7cm_mean, soil_moisture_0_to_7cm_mean`.

Déduplication par (lat, lon, date) avec priorité `era5` > `forecast` : quand la
réanalyse (ERA5, plus précise, ~5 j de délai) devient disponible pour un jour déjà
couvert par le forecast, elle l'écrase.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/champiquest/weather_store.py → racine
STORE_PATH = REPO_ROOT / "data" / "weather" / "history.parquet"

VARIABLES = [
    "precipitation_sum",
    "temperature_2m_min",
    "temperature_2m_max",
    "temperature_2m_mean",
    "et0_fao_evapotranspiration",
    "soil_temperature_0_to_7cm_mean",
    "soil_moisture_0_to_7cm_mean",
]
BASE_COLS = ["lat", "lon", "date", "source"]
COLS = BASE_COLS + VARIABLES

# era5 (réanalyse, précise) écrase forecast (modèle de prévision) sur le même jour.
SOURCE_PRIORITY = {"era5": 0, "forecast": 1}


def load() -> pd.DataFrame | None:
    """Charge tout le store (ou None s'il n'existe pas encore)."""
    if not STORE_PATH.exists():
        return None
    df = pd.read_parquet(STORE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def append(df: pd.DataFrame) -> None:
    """Ajoute des lignes, déduplique (era5 prioritaire) et sauvegarde le Parquet."""
    df = df[COLS].copy()
    df["date"] = pd.to_datetime(df["date"])
    old = load()
    if old is not None:
        df = pd.concat([old, df], ignore_index=True)
    df["_prio"] = df["source"].map(SOURCE_PRIORITY)
    df = df.sort_values(["lat", "lon", "date", "_prio"])
    df = df.drop_duplicates(subset=["lat", "lon", "date"], keep="first")
    df = df.drop(columns=["_prio"])
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STORE_PATH, index=False)


def point_series(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Série quotidienne d'un point (index = date, colonnes = variables)."""
    sub = df[(df["lat"] == lat) & (df["lon"] == lon)]
    return sub.drop(columns=["lat", "lon", "source"]).set_index("date").sort_index()


def last_date(df: pd.DataFrame) -> pd.Timestamp:
    return df["date"].max()


def date_range(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return df["date"].min(), df["date"].max()


def drop_source(source: str) -> None:
    """Supprime les lignes d'une source donnée (ex. 'forecast' avant re-fetch).

    Sert à « flager » les données non-ERA5 : le forecast est transitoire, on le
    re-fetche à chaque run (il devient obsolète si plus d'un jour entre deux runs),
    tandis que l'ERA5 (réanalyse) est stable et ne se re-fetche pas.
    """
    df = load()
    if df is None:
        return
    df = df[df["source"] != source]
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STORE_PATH, index=False)

"""Téléchargement de l'historique météo Open-Meteo (archive) pour les points d'occurrence.

Usage :
    poetry run python -m champiquest.data.openmeteo             # télécharge tout
    poetry run python -m champiquest.data.openmeteo --dry-run   # affiche le plan seulement

Stratégie : on arrondit les coordonnées des occurrences à une grille de 0.25°
(résolution native d'ERA5 ~25 km), on prend les points uniques, et pour chacun on
télécharge la série quotidienne complète (2000 → aujourd'hui). Les agrégats
(fenêtres courtes/longues) seront calculés plus tard dans ``features/weather.py``.

Note sur les quotas : chaque requête longue compte comme plusieurs appels API
(~339 appels pour 26 ans × 5 variables). Le free tier limite à 600 appels/min,
5 000/h et 10 000/jour → on espace les requêtes et on gère les limites en retry.
"""

import argparse
import time
from datetime import date
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_GBIF = REPO_ROOT / "data" / "raw" / "gbif"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "openmeteo"
OUT_DIR = REPO_ROOT / "data" / "raw" / "openmeteo"

START_YEAR = 2000  # début de l'historique météo (on se limite aux données denses)
GRID_DEG = 0.25    # résolution native d'ERA5 (~25 km) : grille large = moins de points
DELAY_BETWEEN_POINTS = 100  # ~36 points/h (respecte la limite horaire de 5000 appels)
# Pass « sol » uniquement : les variables météo (precip/temp/et0) sont déjà dans les
# CSV existants. On ne télécharge que le sol (2 variables) pour limiter le coût API
# (formule Open-Meteo : jours/14 × variables/10 → ~136 appels/requête au lieu de ~475).
VARIABLES = [
    "soil_temperature_0_to_7cm_mean",
    "soil_moisture_0_to_7cm_mean",
]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def load_occurrences(raw_dir: Path = RAW_GBIF) -> pd.DataFrame:
    """Concatène les coordonnées et années de toutes les occurrences GBIF."""
    frames = [
        pd.read_csv(f, usecols=["decimalLatitude", "decimalLongitude", "year"])
        for f in sorted(raw_dir.glob("*.csv"))
        if f.stem != "all_fungi"  # fichier de co-fructification, pas une espèce cible
    ]
    return pd.concat(frames, ignore_index=True)


def unique_points(occ: pd.DataFrame, grid: float = GRID_DEG) -> pd.DataFrame:
    """Arrondit les coordonnées à la grille et retourne les points uniques + année min.

    L'année min de chaque point sert à ne télécharger que depuis sa première
    observation (plutôt que depuis 2000 pour tout le monde).
    """
    lat = (occ["decimalLatitude"] / grid).round() * grid
    lon = (occ["decimalLongitude"] / grid).round() * grid
    pts = pd.DataFrame({"lat": lat, "lon": lon, "year": occ["year"]})
    return pts.groupby(["lat", "lon"], as_index=False).agg(min_year=("year", "min")).reset_index(drop=True)


def build_client(cache_dir: Path = CACHE_DIR):
    """Client Open-Meteo avec cache persistant (re-téléchargement évité)."""
    session = requests_cache.CachedSession(str(cache_dir), expire_after=-1)
    retry_session = retry(session, retries=1, backoff_factor=2.0)
    return openmeteo_requests.Client(session=retry_session)


def fetch_point(client, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Télécharge la série quotidienne complète d'un point (archive)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": VARIABLES,
        "timezone": "Europe/Berlin",
    }
    response = client.weather_api(ARCHIVE_URL, params=params)[0]
    daily = response.Daily()

    n = len(daily.Variables(0).ValuesAsNumpy())
    dates = pd.date_range(start=start_date, end=end_date, freq="D")[:n]
    data = {var: daily.Variables(i).ValuesAsNumpy() for i, var in enumerate(VARIABLES)}
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


def fetch_point_with_retry(client, lat: float, lon: float, start_date: str, end_date: str):
    """Télécharge un point en gérant les limites de débit de l'API (attente + retry).

    - limite minute → attente 60 s ;
    - limite horaire → attente 3600 s ;
    - limite journalière → arrêt (RuntimeError) pour reprendre le lendemain.
    """
    while True:
        try:
            return fetch_point(client, lat, lon, start_date, end_date)
        except Exception as exc:
            msg = str(exc).lower()
            if "daily" in msg:
                raise RuntimeError(
                    "limite journalière atteinte — relancer demain (le script reprend où il en était)."
                ) from exc
            wait = 3600 if "hourly" in msg else 70
            print(f"    limite {'horaire' if wait == 3600 else 'minute'} atteinte, attente {wait}s…", flush=True)
            time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche uniquement les points qui seraient téléchargés, sans interroger l'API.",
    )
    args = parser.parse_args()

    occ = load_occurrences()
    points = unique_points(occ)
    end_date = date.today().isoformat()

    print(f"{len(occ)} occurrences → {len(points)} points uniques (grille {GRID_DEG}°)")
    print(f"Fin de plage : {end_date} (début = année min d'observation par point, min {START_YEAR})")

    if args.dry_run:
        print(f"[dry-run] {len(points)} requêtes archive seraient faites vers {OUT_DIR}/")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = build_client()

    for i, (_, row) in enumerate(points.iterrows(), start=1):
        lat, lon = row["lat"], row["lon"]
        start = f"{max(START_YEAR, int(row['min_year']))}-01-01"
        out = OUT_DIR / f"point_{lat:.2f}_{lon:.2f}.csv"
        if out.exists():
            # re-télécharge si les colonnes sol manquent (ancien fetch sans sol)
            existing_cols = pd.read_csv(out, index_col=0, nrows=0).columns
            if all(v in existing_cols for v in VARIABLES):
                continue
        try:
            df = fetch_point_with_retry(client, lat, lon, start, end_date)
            if out.exists():  # fusionne le sol dans le CSV météo existant
                existing = pd.read_csv(out, index_col=0, parse_dates=True)
                df = existing.join(df, how="left")
            df.to_csv(out)
            print(f"[{i}/{len(points)}] {lat:.2f}, {lon:.2f} OK ({len(df)} jours, depuis {start})", flush=True)
        except RuntimeError as exc:  # limite journalière → arrêt propre
            print(f"Arrêt : {exc}", flush=True)
            break
        except Exception as exc:  # autre échec → on passe au point suivant
            print(f"[{i}/{len(points)}] {lat:.2f}, {lon:.2f} : échec ({exc})", flush=True)
            continue
        time.sleep(DELAY_BETWEEN_POINTS)


if __name__ == "__main__":
    main()

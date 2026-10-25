"""Génère la carte de prédiction à partir du store météo (Parquet).

Pipeline (lancé manuellement, pas de cron pour l'instant) :
1. **refresh** : ERA5 (réanalyse précise, passé J-90 → J-5) + forecast (J-7 → J+3),
   ajoutés au store (le forecast, transitoire, est re-fetché à chaque run) ;
2. **generate** : pour chaque point, lit la série du store → 12 features pour
   J, J+1, J+2, J+3 → 6 modèles → season gate ;
3. sauvegarde `data/maps/<date>/probs.npz` (4 dates).

Usage :
    poetry run python -m champiquest.mapgen                # refresh + generate
    poetry run python -m champiquest.mapgen --skip-refresh  # generate seulement
    poetry run python -m champiquest.mapgen --limit 10      # test sur 10 points
"""

import argparse
import time
import warnings
from pathlib import Path

# pandas/NumPy 2.x : pd.Timedelta émet un DeprecationWarning bruité (harmless)
warnings.filterwarnings("ignore", message=".*generic.*unit for NumPy timedelta.*")

import joblib
import numpy as np
import pandas as pd

import champiquest.weather_store as ws
from champiquest.growth import MODELS_DIR, SPECIES, compute_features, fetch_era5, fetch_weather
from champiquest.models.constants import DATASET

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/champiquest/mapgen.py → racine
MAPS_DIR = REPO_ROOT / "data" / "maps"

BBOX = {"min_lat": 47.3, "max_lat": 49.3, "min_lon": 6.4, "max_lon": 8.7}
STEP = 0.055        # ~5 km
DELAY = 0.5         # s entre deux points (limite minute Open-Meteo)
ERA5_LAG = 5        # délai de publication de la réanalyse ERA5 (jours)
FORECAST_DAYS = 3   # horizon de prévision (J+3)
HISTORY_DAYS = 90   # fenêtre nécessaire aux features (precip_J90)
N_DATES = 4         # J, J+1, J+2, J+3


def build_grid() -> tuple[np.ndarray, np.ndarray]:
    lats = np.round(np.arange(BBOX["min_lat"], BBOX["max_lat"] + STEP, STEP), 4)
    lons = np.round(np.arange(BBOX["min_lon"], BBOX["max_lon"] + STEP, STEP), 4)
    return lats, lons


def load_season_windows() -> dict:
    """Fenêtre de fructification par espèce (p5−14 → p95+14), depuis le dataset."""
    df = pd.read_csv(DATASET, parse_dates=["date"])
    df = df[df["target"] == 1]
    windows = {}
    for sp in SPECIES:
        doy = df[df["taxon_key"] == sp]["date"].dt.dayofyear
        if len(doy) < 10:
            windows[sp] = (0, 366)
        else:
            windows[sp] = (int(doy.quantile(0.05)) - 14, int(doy.quantile(0.95)) + 14)
    return windows


def refresh_store(lats: np.ndarray, lons: np.ndarray, limit: int | None = None) -> None:
    """Fetch ERA5 (passé manquant) + forecast (récent) et met à jour le Parquet.

    **Resumable** : l'ERA5 est écrit par lots (50 points). Si le script s'arrête
    (ex. limite API journalière), le prochain run reprend là où il s'était arrêté
    (les points déjà ERA5 jusqu'à J-5 sont sautés).
    """
    today = pd.Timestamp.now().normalize()
    era5_end = today - pd.Timedelta(days=ERA5_LAG)
    ws.drop_source("forecast")  # le forecast est transitoire → re-fetch systématique

    store = ws.load()
    # dernière date ERA5 par point (pour le resume per-point)
    era5_last: dict = {}
    if store is not None:
        era5 = store[store["source"] == "era5"]
        for (lat, lon), grp in era5.groupby(["lat", "lon"]):
            era5_last[(round(float(lat), 4), round(float(lon), 4))] = grp["date"].max()

    cells = [(float(lat), float(lon)) for lat in lats for lon in lons]
    if limit:
        cells = cells[:limit]
    total = len(cells)

    rows = []
    for n, (lat, lon) in enumerate(cells, 1):
        key = (round(lat, 4), round(lon, 4))
        # 1. ERA5 : seulement si ce point n'a pas encore ERA5 jusqu'à J-5
        last = era5_last.get(key)
        start = today - pd.Timedelta(days=HISTORY_DAYS) if last is None else pd.Timestamp(last) + pd.Timedelta(days=1)
        if start <= era5_end:
            try:
                era = fetch_era5(lat, lon, start.date().isoformat(), era5_end.date().isoformat())
                era = era.reset_index()
                era["lat"] = lat
                era["lon"] = lon
                era["source"] = "era5"
                rows.append(era)
            except Exception:
                pass  # point manquant → NaN, repris au prochain run
        # 2. forecast : toujours (chevauche un peu l'ERA5 pour éviter les trous)
        try:
            fc = fetch_weather(lat, lon, past_days=ERA5_LAG + 2, forecast_days=FORECAST_DAYS)
            fc = fc.reset_index()
            fc["lat"] = lat
            fc["lon"] = lon
            fc["source"] = "forecast"
            rows.append(fc)
        except Exception:
            pass
        # écrit par lots → resumable
        if len(rows) >= 50:
            ws.append(pd.concat(rows, ignore_index=True))
            rows = []
        if n % 50 == 0:
            print(f"  refresh [{n}/{total}]", flush=True)
        time.sleep(DELAY)

    if rows:
        ws.append(pd.concat(rows, ignore_index=True))
    print("store rafraîchi")


def generate(dates: list[str], limit: int | None = None) -> None:
    """Lit le store → features → modèles → npz pour chaque date."""
    lats, lons = build_grid()
    store = ws.load()
    if store is None:
        raise SystemExit("store vide : lance d'abord le refresh (sans --skip-refresh).")

    models = {sp: joblib.load(MODELS_DIR / f"growth_{sp}.joblib")["model"] for sp in SPECIES}
    windows = load_season_windows()
    doys = {d: pd.Timestamp(d).dayofyear for d in dates}

    cells = [(i, j) for i in range(len(lats)) for j in range(len(lons))]
    if limit:
        cells = cells[:limit]

    date_probs = {d: {sp: np.full((len(lats), len(lons)), np.nan) for sp in SPECIES} for d in dates}

    for (i, j) in cells:
        lat, lon = float(lats[i]), float(lons[j])
        series = ws.point_series(store, lat, lon)
        if len(series) < HISTORY_DAYS:
            continue  # pas assez d'historique → NaN
        for d in dates:
            feats = compute_features(series, d, lat, lon)
            X = pd.DataFrame([feats])[feats.index.tolist()]
            doy = doys[d]
            for sp in SPECIES:
                p = float(models[sp].predict_proba(X)[0, 1])
                s, e = windows[sp]
                if not (s <= doy <= e):  # season gate
                    p = 0.0
                date_probs[d][sp][i, j] = p

    for d in dates:
        out_dir = MAPS_DIR / d
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "probs.npz", **date_probs[d], lats=lats, lons=lons)
        print(f"✔ {d} → {out_dir / 'probs.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="limite de points (test)")
    parser.add_argument("--skip-refresh", action="store_true", help="ne pas re-fetch la météo")
    parser.add_argument("--refresh-only", action="store_true", help="ne faire que le refresh (backfill), sans generate")
    args = parser.parse_args()

    lats, lons = build_grid()
    today = pd.Timestamp.now().normalize()
    dates = [(today + pd.Timedelta(days=i)).date().isoformat() for i in range(N_DATES)]
    print(f"dates cibles : {dates}")

    if not args.skip_refresh:
        refresh_store(lats, lons, limit=args.limit)
    if args.refresh_only:
        return
    generate(dates, limit=args.limit)


if __name__ == "__main__":
    main()

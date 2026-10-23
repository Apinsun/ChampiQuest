"""Téléchargement et traitement des données environnementales (raster + sol + CLC).

Commandes :
    dem        → tuiles DEM (altitude/pente/orientation) via AWS "Terrain Tiles" (sans compte)
    soil       → propriétés de sol via l'API SoilGrids (ISRIC, public, sans compte)
    clc        → occupation du sol Corine Land Cover (identify EEA, public)
    nasa_power → température de surface + humidité du sol (NASA POWER, daily, sans clé)

Usage :
    poetry run python -m champiquest.data.environnement dem            # télécharge le DEM
    poetry run python -m champiquest.data.environnement soil           # interroge SoilGrids
    poetry run python -m champiquest.data.environnement clc            # identifie CLC
    poetry run python -m champiquest.data.environnement nasa_power     # NASA POWER
"""

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from pyproj import Transformer

from champiquest.data.gbif import RAW_DIR as GBIF_RAW
from champiquest.data.openmeteo import GRID_DEG, load_occurrences, unique_points

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "species.yaml"
DEM_DIR = REPO_ROOT / "data" / "raw" / "dem"
SOIL_DIR = REPO_ROOT / "data" / "raw" / "soilgrids"

# --- DEM --------------------------------------------------------------------
# AWS Open Data "Terrain Tiles" : DEM mondial (EU-DEM 25 m pour l'Europe),
# servi en tuiles GeoTIFF 512×512 (web-mercator). Zoom 10 ≈ 76 m/px : bon compromis.
DEM_URL = "https://elevation-tiles-prod.s3.amazonaws.com/geotiff/{zoom}/{x}/{y}.tif"
DEM_ZOOM = 10

# --- SoilGrids --------------------------------------------------------------
SOIL_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_PROPERTIES = ["phh2o", "sand", "silt", "clay", "soc"]  # pH, texture, carbone organique
SOIL_DEPTH = "0-5cm"  # couche de surface (la plus pertinente pour le mycélium)


def load_bbox(path: Path = CONFIG_PATH) -> dict:
    """Charge la bbox depuis la config des espèces."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["bbox"]


# ---------------------------------------------------------------------------
# DEM
# ---------------------------------------------------------------------------

def deg2num(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convertit lat/lon en coordonnées de tuile (x, y) web-mercator."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tiles_for_bbox(bbox: dict, zoom: int = DEM_ZOOM) -> list[tuple[int, int]]:
    """Liste les tuiles (x, y) couvrant entièrement la bbox.

    En web-mercator, y croît vers le sud : la latitude nord (max_lat) donne le
    plus petit y (y_top), la latitude sud (min_lat) le plus grand (y_bottom).
    """
    x_min, y_top = deg2num(bbox["max_lat"], bbox["min_lon"], zoom)     # coin nord-ouest
    x_max, y_bottom = deg2num(bbox["min_lat"], bbox["max_lon"], zoom)  # coin sud-est
    return [
        (x, y)
        for x in range(x_min, x_max + 1)
        for y in range(y_top, y_bottom + 1)
    ]


def download_tile(zoom: int, x: int, y: int, out_dir: Path = DEM_DIR) -> Path:
    """Télécharge une tuile DEM (sautée si déjà présente)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{zoom}_{x}_{y}.tif"
    if out.exists():
        return out
    resp = requests.get(DEM_URL.format(zoom=zoom, x=x, y=y), timeout=60)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    return out


def cmd_dem(args) -> None:
    bbox = load_bbox()
    tiles = tiles_for_bbox(bbox, args.zoom)
    print(f"bbox {bbox['min_lat']}–{bbox['max_lat']} N, {bbox['min_lon']}–{bbox['max_lon']} E")
    print(f"zoom {args.zoom} → {len(tiles)} tuiles DEM")

    if args.dry_run:
        print("[dry-run] tuiles :", tiles)
        return

    for i, (x, y) in enumerate(tiles, start=1):
        out = download_tile(args.zoom, x, y)
        print(f"[{i}/{len(tiles)}] {args.zoom}/{x}/{y} → {out.name}", flush=True)


# ---------------------------------------------------------------------------
# SoilGrids
# ---------------------------------------------------------------------------

def query_soil(lat: float, lon: float, retries: int = 5) -> dict:
    """Interroge SoilGrids pour un point (propriétés de surface), avec retry.

    L'API SoilGrids est lente (20-60 s par point) et souvent saturée (503) :
    timeout généreux + retries avec backoff exponentiel (3, 10, 30, 60 s).
    """
    params = {
        "lon": lon,
        "lat": lat,
        "property": SOIL_PROPERTIES,
        "depth": SOIL_DEPTH,
        "value": "mean",
    }
    backoffs = [3, 10, 30, 60]
    for attempt in range(retries):
        try:
            resp = requests.get(SOIL_URL, params=params, timeout=90)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(backoffs[min(attempt, len(backoffs) - 1)])


def parse_soil(feature: dict) -> dict:
    """Extrait les valeurs d'une réponse SoilGrids, en appliquant le facteur d'échelle.

    SoilGrids renvoie des valeurs multipliées par un ``d_factor`` (ex. pH×10) ;
    on divise pour retrouver la vraie valeur (ex. 56 → 5.6).
    """
    out: dict = {}
    for layer in feature["properties"]["layers"]:
        name = layer["name"]
        factor = layer["unit_measure"]["d_factor"]
        for dep in layer["depths"]:
            val = dep["values"].get("mean")
            out[f"{name}_{dep['label']}"] = (val / factor) if val is not None else None
    return out


def cmd_soil(args) -> None:
    occ = load_occurrences()
    points = unique_points(occ)  # grille 0.25°, cohérente avec la météo
    out_path = SOIL_DIR / "soil_points.csv"
    print(f"{len(points)} points uniques (grille {GRID_DEG}°) → {SOIL_PROPERTIES} @ {SOIL_DEPTH}")

    if args.dry_run:
        print(f"[dry-run] {len(points)} requêtes SoilGrids seraient faites")
        return

    SOIL_DIR.mkdir(parents=True, exist_ok=True)
    # Repart des points déjà récupérés (reprise après interruption / élargissement de bbox)
    rows: list[dict] = []
    existing: set[str] = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        existing = {f"{la:.2f},{lo:.2f}" for la, lo in zip(prev["lat"], prev["lon"])}

    for i, (_, row) in enumerate(points.iterrows(), start=1):
        lat, lon = row["lat"], row["lon"]
        if f"{lat:.2f},{lon:.2f}" in existing:
            print(f"[{i}/{len(points)}] {lat:.2f}, {lon:.2f} déjà présent, skip", flush=True)
            continue
        try:
            parsed = parse_soil(query_soil(lat, lon))
            parsed.update({"lat": lat, "lon": lon})
            rows.append(parsed)
            print(f"[{i}/{len(points)}] {lat:.2f}, {lon:.2f} OK", flush=True)
        except Exception as exc:  # un point en échec ne bloque pas les autres
            print(f"[{i}/{len(points)}] {lat:.2f}, {lon:.2f} : échec ({exc})", flush=True)
        time.sleep(0.2)  # politesse vis-à-vis de l'API

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Sauvegardé : {out_path} ({len(df)} lignes × {len(df.columns)} colonnes)")


# ---------------------------------------------------------------------------
# CLC (Corine Land Cover 2018) — type de forêt / occupation du sol
# ---------------------------------------------------------------------------
# Le raster est servi publiquement par l'EEA (image.discomap.eea.europa.eu),
# en projection LAEA (EPSG:3035). On interroge chaque point via "identify".
CLCSRV_URL = (
    "https://image.discomap.eea.europa.eu/arcgis/rest/services/"
    "Corine/CLC2018_LAEA/MapServer/identify"
)
_to_laea = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


def identify_clc(lon: float, lat: float) -> dict:
    """Interroge le raster CLC 2018 (couche 1) et retourne {clc_code, clc_label}."""
    x, y = _to_laea.transform(lon, lat)
    params = {
        "f": "json",
        "geometryType": "esriGeometryPoint",
        "geometry": f"{x},{y}",
        "layers": "all:1",
        "returnGeometry": "false",
        "tolerance": "1",
        "mapExtent": f"{x - 1000},{y - 1000},{x + 1000},{y + 1000}",
        "imageDisplay": "100,100,96",
    }
    resp = requests.get(CLCSRV_URL, params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return {"clc_code": None, "clc_label": None}
    attrs = results[0].get("attributes", {})
    return {"clc_code": attrs.get("Raster.Code_18"), "clc_label": attrs.get("Raster.LABEL3")}


def cmd_clc(args) -> None:
    """Identifie la classe CLC de chaque occurrence (coordonnées exactes)."""
    occ = pd.concat(
        [pd.read_csv(f, usecols=["decimalLatitude", "decimalLongitude"]) for f in GBIF_RAW.glob("*.csv")],
        ignore_index=True,
    )
    pts = occ.drop_duplicates().reset_index(drop=True)
    out_path = REPO_ROOT / "data" / "raw" / "clc" / "clc_points.csv"
    print(f"{len(occ)} occurrences → {len(pts)} coordonnées uniques")

    if args.dry_run:
        print(f"[dry-run] {len(pts)} requêtes identify CLC seraient faites")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (_, r) in enumerate(pts.iterrows(), start=1):
        lon, lat = r["decimalLongitude"], r["decimalLatitude"]
        try:
            info = identify_clc(lon, lat)
            info.update({"lon": lon, "lat": lat})
            rows.append(info)
        except Exception as exc:  # un point en échec ne bloque pas les autres
            print(f"[{i}/{len(pts)}] {lon:.3f},{lat:.3f} : échec ({exc})", flush=True)
        if i % 250 == 0:
            print(f"[{i}/{len(pts)}]", flush=True)
        time.sleep(0.05)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Sauvegardé : {out_path} ({len(df)} lignes)")


# --- NASA POWER (température de surface + humidité du sol, daily) -----------
# NASA POWER (MERRA-2) fournit en daily et sans clé : TS (Earth Skin
# Temperature, °C) et GWETTOP (Surface Soil Wetness, 0-1). C'est le complément
# qui manquait : la température du sol (déclencheur printanier de la morille)
# et la vraie humidité du sol (remplace le proxy « bilan hydrique »).
NASA_DIR = REPO_ROOT / "data" / "raw" / "nasa_power"
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMS = ["TS", "GWETTOP"]
NASA_FILL = -999.0  # valeur de remplissage NASA POWER


def query_nasa_power(lat: float, lon: float, start: str, end: str, retries: int = 3) -> dict:
    """Interroge NASA POWER (daily) pour un point : TS + GWETTOP."""
    params = {
        "parameters": ",".join(NASA_PARAMS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "JSON",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(NASA_URL, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3)


def cmd_nasa_power(args) -> None:
    """Télécharge TS + GWETTOP (daily) pour chaque point de grille, en reprenant
    les points déjà téléchargés (skip si le CSV existe)."""
    occ = load_occurrences()
    points = unique_points(occ)
    start, end = "2000-01-01", "2026-12-31"
    print(f"{len(points)} points uniques → NASA POWER {NASA_PARAMS} ({start} → {end})")

    if args.dry_run:
        print(f"[dry-run] {len(points)} requêtes NASA POWER seraient faites")
        return

    NASA_DIR.mkdir(parents=True, exist_ok=True)
    for i, (_, row) in enumerate(points.iterrows(), start=1):
        lat, lon = row["lat"], row["lon"]
        out = NASA_DIR / f"point_{lat:.2f}_{lon:.2f}.csv"
        if out.exists():
            print(f"[{i}/{len(points)}] {lat:.2f}, {lon:.2f} déjà présent, skip", flush=True)
            continue
        try:
            data = query_nasa_power(lat, lon, start, end)
            props = data["properties"]["parameter"]
            ts, gwet = props["TS"], props["GWETTOP"]
            dates = sorted(ts.keys())
            df = pd.DataFrame({
                "date": pd.to_datetime(dates, format="%Y%m%d"),
                "ts": [ts[d] if ts[d] != NASA_FILL else float("nan") for d in dates],
                "gwet_top": [gwet[d] if gwet[d] != NASA_FILL else float("nan") for d in dates],
            })
            df.to_csv(out, index=False)
            print(f"[{i}/{len(points)}] {lat:.2f}, {lon:.2f} OK ({len(df)} jours)", flush=True)
        except Exception as exc:  # un point en échec ne bloque pas les autres
            print(f"[{i}/{len(points)}] {lat:.2f}, {lon:.2f} : échec ({exc})", flush=True)
        time.sleep(1.0)  # politesse / quotas NASA POWER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dem = sub.add_parser("dem", help="Télécharge les tuiles DEM de la zone d'étude")
    p_dem.add_argument("--dry-run", action="store_true", help="Liste les tuiles sans télécharger")
    p_dem.add_argument("--zoom", type=int, default=DEM_ZOOM, help="Niveau de zoom (défaut 10 ≈ 76 m/px)")

    p_soil = sub.add_parser("soil", help="Interroge SoilGrids pour les points d'étude")
    p_soil.add_argument("--dry-run", action="store_true", help="Affiche le plan sans interroger l'API")

    p_clc = sub.add_parser("clc", help="Identifie la classe CLC (Corine Land Cover) par point")
    p_clc.add_argument("--dry-run", action="store_true", help="Affiche le plan sans interroger l'API")

    p_nasa = sub.add_parser("nasa_power", help="Télécharge température de surface + humidité du sol (NASA POWER)")
    p_nasa.add_argument("--dry-run", action="store_true", help="Affiche le plan sans interroger l'API")

    args = parser.parse_args()
    if args.cmd == "dem":
        cmd_dem(args)
    elif args.cmd == "soil":
        cmd_soil(args)
    elif args.cmd == "clc":
        cmd_clc(args)
    elif args.cmd == "nasa_power":
        cmd_nasa_power(args)


if __name__ == "__main__":
    main()

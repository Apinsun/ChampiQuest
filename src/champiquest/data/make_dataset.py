"""Assemblage du dataset d'entraînement (v1, baseline).

Pour chaque observation (présence) : features météo (série du point de grille),
sol (point de grille), altitude (coordonnées exactes). Puis génération de
pseudo-absences 1:1 (même point, date aléatoire dans la saison de l'espèce).

Usage :
    poetry run python -m champiquest.data.make_dataset
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer

from champiquest.data.environnement import DEM_DIR, NASA_DIR, SOIL_DIR
from champiquest.data.gbif import RAW_DIR, load_config
from champiquest.data.openmeteo import OUT_DIR as WEATHER_DIR
from champiquest.features.weather import compute_soil_features, compute_weather_features

REPO_ROOT = Path(__file__).resolve().parents[3]
SEASON_PATH = REPO_ROOT / "configs" / "season_windows.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

GRID = 0.25  # grille des points météo/sol
DEM_ZOOM = 10
ABSENCE_BUFFER_DAYS = 14  # une absence ne doit pas tomber à moins de 14 j d'une observation réelle
MAX_ABSENCE_ATTEMPTS = 30  # essais de tirage avant d'abandonner une absence

# Variables de sol retenues (``silt`` exclu : collinéaire, sand+silt+clay = 100 %)
SOIL_COLS = ["phh2o_0-5cm", "sand_0-5cm", "clay_0-5cm", "soc_0-5cm"]
CLC_COLS = ["clc_code", "clc_label"]
CLC_PATH = REPO_ROOT / "data" / "raw" / "clc" / "clc_points.csv"

# Regroupement écologique : espèces proches (même saison/habitat) partageant un modèle.
# Seulement les espèces dont on a vérifié la corrélation de saison (cf. discussion).
GROUPS = {
    "boletus_edulis": ["boletus_edulis", "boletus_aereus"],  # cèpes d'automne (saison + altitude corrélées)
    "macrolepiota_procera": ["macrolepiota_procera", "chlorophyllum_rhacodes"],  # lépiotes (corrélées)
}

_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_tile_cache: dict = {}


def load_occurrences() -> pd.DataFrame:
    """Charge les occurrences GBIF, parse la date et ajoute le point de grille."""
    frames = []
    for f in sorted(RAW_DIR.glob("*.csv")):
        if f.stem == "all_fungi":  # fichier de co-fructification (toutes espèces), pas une espèce cible
            continue
        df = pd.read_csv(
            f,
            usecols=["decimalLatitude", "decimalLongitude", "year", "month", "day", "species"],
        )
        df["taxon_key"] = _group_of(f.stem)  # ex. boletus_edulis, morchella… (regroupement écologique)
        frames.append(df)
    occ = pd.concat(frames, ignore_index=True)

    # Date reconstruite depuis year/month/day (eventDate était peu fiable à parser)
    occ["date"] = pd.to_datetime(
        {"year": occ["year"], "month": occ["month"], "day": occ["day"]},
        errors="coerce",
    )
    occ = occ.dropna(subset=["date"])

    # Point de grille (0.25°) qui relie l'observation à la météo et au sol
    occ["glat"] = (occ["decimalLatitude"] / GRID).round() * GRID
    occ["glon"] = (occ["decimalLongitude"] / GRID).round() * GRID
    return occ


def load_weather() -> dict:
    """Charge les séries météo : {nom_fichier: DataFrame indexé par date}."""
    return {
        f.stem: pd.read_csv(f, index_col=0, parse_dates=True)
        for f in sorted(WEATHER_DIR.glob("*.csv"))
    }


def load_all_fungi() -> pd.DataFrame:
    """Charge toutes les occurrences fongiques (309k) avec cellule + date."""
    fungi = pd.read_csv(RAW_DIR / "all_fungi.csv")
    fungi["glat"] = (fungi["decimalLatitude"] / GRID).round() * GRID
    fungi["glon"] = (fungi["decimalLongitude"] / GRID).round() * GRID
    fungi["date"] = pd.to_datetime(
        {"year": fungi["year"], "month": fungi["month"], "day": fungi["day"]},
        errors="coerce",
    )
    return fungi.dropna(subset=["date"])


def load_cofruiting(fungi: pd.DataFrame, exclude_keys: set) -> dict:
    """{cellule: array trié de dates} des champignons hors taxons exclus.

    Sert à la preuve d'effort pour les pseudo-absences.
    """
    sub = fungi[~fungi["taxonKey"].isin(exclude_keys)]
    out: dict = {}
    for (glat, glon), grp in sub.groupby(["glat", "glon"]):
        out[_weather_key(float(glat), float(glon))] = np.sort(grp["date"].to_numpy(dtype="datetime64[ns]"))
    return out


def compute_proxy_scores(occ: pd.DataFrame, fungi: pd.DataFrame, target_keys: set) -> pd.DataFrame:
    """Score de co-occurrence de chaque espèce (hors cibles) avec chaque groupe.

    Renvoie un DataFrame (group, taxonKey, co_occ, saison, n) qui est sauvegardé
    pour pouvoir filtrer ensuite sans recalculer.
    """
    cell_fungi = {}
    for (glat, glon), grp in fungi.groupby(["glat", "glon"]):
        cell_fungi[(glat, glon)] = (
            grp["date"].to_numpy(dtype="datetime64[ns]"),
            grp["taxonKey"].to_numpy(),
        )

    rows = []
    for group, grp in occ.groupby("taxon_key"):
        doy = grp["date"].dt.dayofyear
        s, e = doy.quantile(0.05), doy.quantile(0.95)
        co_occ: dict = {}
        total = 0
        for _, o in grp.iterrows():
            key = (o["glat"], o["glon"])
            if key not in cell_fungi:
                continue
            dates, tks = cell_fungi[key]
            lo = np.datetime64(o["date"]) - np.timedelta64(7, "D")
            hi = np.datetime64(o["date"]) + np.timedelta64(7, "D")
            mask = (dates >= lo) & (dates <= hi)
            for tk in np.unique(tks[mask]):
                if tk not in target_keys:
                    co_occ[tk] = co_occ.get(tk, 0) + 1
            total += 1
        for tk, c in co_occ.items():
            fd = fungi[fungi["taxonKey"] == tk]["date"]
            ov = float(((fd.dt.dayofyear >= s) & (fd.dt.dayofyear <= e)).mean()) if len(fd) else 0.0
            rows.append({"group": group, "taxonKey": tk, "co_occ": c / total, "saison": ov, "n": len(fd)})
    return pd.DataFrame(rows)


def cofruct_count(cell_dates, obs_date: pd.Timestamp, window_days: int) -> float:
    """Nombre d'occurrences fongiques (autres espèces) dans [obs_date - window, obs_date]."""
    if cell_dates is None or len(cell_dates) == 0:
        return 0.0
    lo = np.datetime64(obs_date) - np.timedelta64(window_days, "D")
    hi = np.datetime64(obs_date)
    return float(
        np.searchsorted(cell_dates, hi, side="right") - np.searchsorted(cell_dates, lo, side="left")
    )


def load_soil() -> pd.DataFrame:
    """Charge le sol et impute les points manquants par le plus proche voisin."""
    soil = pd.read_csv(SOIL_DIR / "soil_points.csv")
    missing = soil[SOIL_COLS].isna().any(axis=1)
    if missing.any():
        valid = soil[~missing]
        for idx in soil[missing].index:
            row = soil.loc[idx]
            # distance euclidienne en degrés (suffisant pour un voisin à ~20 km)
            dist = valid.apply(
                lambda r: (r["lat"] - row["lat"]) ** 2 + (r["lon"] - row["lon"]) ** 2,
                axis=1,
            )
            nearest = valid.loc[dist.idxmin()]
            for col in SOIL_COLS:
                soil.loc[idx, col] = nearest[col]
    return soil


def load_clc() -> pd.DataFrame:
    """Charge les classes CLC (Corine Land Cover) identifiées par point."""
    return pd.read_csv(CLC_PATH)


def _group_of(stem: str) -> str:
    """Retourne la clé de groupe d'une espèce (elle-même si non regroupée)."""
    for group, members in GROUPS.items():
        if stem in members:
            return group
    return stem


def compute_group_seasons(occ: pd.DataFrame) -> dict:
    """Fenêtre de saison par groupe (percentiles 5/95 + buffer 14 j)."""
    seasons = {}
    for group, grp in occ.groupby("taxon_key"):
        doy = grp["date"].dt.dayofyear.dropna()
        seasons[group] = {
            "start_doy": int(doy.quantile(0.05)) - 14,
            "end_doy": int(doy.quantile(0.95)) + 14,
        }
    return seasons


def load_season_windows() -> dict:
    with open(SEASON_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _open_tile(x: int, y: int):
    """Ouvre (et met en cache) une tuile DEM."""
    key = (x, y)
    if key not in _tile_cache:
        _tile_cache[key] = rasterio.open(DEM_DIR / f"{DEM_ZOOM}_{x}_{y}.tif")
    return _tile_cache[key]


def _close_tiles() -> None:
    """Ferme les tuiles DEM ouvertes (évite une erreur à la sortie de l'interpréteur)."""
    for src in _tile_cache.values():
        src.close()
    _tile_cache.clear()


def elevation(lat: float, lon: float) -> float:
    """Altitude (m) aux coordonnées exactes, depuis les tuiles DEM."""
    n = 2 ** DEM_ZOOM
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    mx, my = _to_merc.transform(lon, lat)
    src = _open_tile(x, y)
    row, col = src.index(mx, my)
    return float(src.read(1)[row, col])


def _slope_aspect(lat: float, lon: float) -> tuple[float, float, float]:
    """Pente (°) et orientation (northness, eastness) aux coordonnées exactes.

    Calculées par gradient du DEM sur une fenêtre 3×3. ``northness = cos(aspect)``
    et ``eastness = sin(aspect)`` préservent la circularité de l'orientation
    (0° = 360°). northness=1 → versant nord (frais/humide), -1 → sud.
    """
    n = 2 ** DEM_ZOOM
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    mx, my = _to_merc.transform(lon, lat)
    src = _open_tile(x, y)
    row, col = src.index(mx, my)
    r0, r1 = max(0, row - 1), min(src.height, row + 2)
    c0, c1 = max(0, col - 1), min(src.width, col + 2)
    win = src.read(1, window=((r0, r1), (c0, c1))).astype(float)
    dy, dx = np.gradient(win)
    cy, cx = row - r0, col - c0  # pixel du point dans la fenêtre
    dzdx = dx[cy, cx] / abs(src.res[0])   # pente vers l'est
    dzdn = -dy[cy, cx] / abs(src.res[1])  # pente vers le nord
    pente = math.degrees(math.atan(math.sqrt(dzdx ** 2 + dzdn ** 2)))
    aspect = math.degrees(math.atan2(dzdx, dzdn)) % 360
    return pente, math.cos(math.radians(aspect)), math.sin(math.radians(aspect))


def _weather_key(glat: float, glon: float) -> str:
    return f"point_{glat:.2f}_{glon:.2f}"


def build_presences(occ: pd.DataFrame, weather: dict, cofruiting: dict) -> pd.DataFrame:
    """Calcule les features (météo + sol + co-fructification + altitude) pour chaque présence."""
    rows = []
    for _, o in occ.iterrows():
        key = _weather_key(o["glat"], o["glon"])
        if key not in weather:
            continue
        # Il faut au moins 30 jours de météo avant l'observation (sinon features vides)
        if o["date"] - pd.Timedelta(days=30) < weather[key].index.min():
            continue
        feats = compute_weather_features(weather[key], o["date"])
        feats.update(compute_soil_features(weather[key], o["date"]))  # sol Open-Meteo (0-7 cm)
        feats["altitude"] = elevation(o["decimalLatitude"], o["decimalLongitude"])
        pente, northness, eastness = _slope_aspect(o["decimalLatitude"], o["decimalLongitude"])
        feats["pente"] = pente
        feats["northness"] = northness
        feats["eastness"] = eastness
        feats.update(
            {
                "taxon_key": o["taxon_key"],
                "species": o["species"],
                "date": o["date"],
                "glat": o["glat"],
                "glon": o["glon"],
                "lat": o["decimalLatitude"],
                "lon": o["decimalLongitude"],
                "target": 1,
            }
        )
        rows.append(feats)
    return pd.DataFrame(rows)


def generate_absences(
    presences: pd.DataFrame,
    weather: dict,
    seasons: dict,
    rngs: dict,
    obs_dates: dict,
    year_pool: dict,
    cofruiting: dict,
    buffer_days: int = ABSENCE_BUFFER_DAYS,
    absences_per_presence: int = 1,
) -> pd.DataFrame:
    """Génère 1 pseudo-absence par présence.

    Même point de grille, date aléatoire dans la saison, en évitant les dates
    trop proches (± buffer_days) d'une observation réelle de la même espèce au
    même point (un carpophore persiste plusieurs jours → ce serait une fausse absence).

    L'année est tirée proportionnellement à l'effort de prospection de l'espèce
    (pool des années de présence) : une absence = « on a regardé, pas trouvé »,
    pas « personne n'a regardé » (target-group background temporel).
    """
    rows = []
    for _, p in presences.iterrows():
        rng = rngs[p["taxon_key"]]  # RNG propre à l'espèce (reproductible)
        win = seasons[p["taxon_key"]]
        key = _weather_key(p["glat"], p["glon"])
        dates = obs_dates.get(
            (p["taxon_key"], p["glat"], p["glon"]),
            np.array([], dtype="datetime64[ns]"),
        )

        effort = cofruiting.get(key)  # dates où d'autres champignons ont été vus (preuve d'effort)
        for _ in range(absences_per_presence):
            date = None
            for _ in range(MAX_ABSENCE_ATTEMPTS):
                if effort is not None and len(effort) > 0:
                    # Absence « à effort prouvé » : quelqu'un a vu d'autres champignons
                    # ce jour-là sans notre espèce → négatif bien plus propre.
                    candidate = pd.Timestamp(effort[int(rng.integers(0, len(effort)))])
                    if not (win["start_doy"] <= candidate.dayofyear <= win["end_doy"]):
                        continue
                else:
                    # Repli : cellule sans autre donnée fongique → tirage aléatoire en saison
                    year = int(rng.choice(year_pool[p["taxon_key"]]))
                    doy = int(rng.integers(win["start_doy"], win["end_doy"] + 1))
                    candidate = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
                # Il faut au moins 30 jours de météo avant l'absence
                if candidate - pd.Timedelta(days=30) < weather[key].index.min():
                    continue
                trop_proche = len(dates) and (
                    np.abs(dates - np.datetime64(candidate)) < np.timedelta64(buffer_days, "D")
                ).any()
                if trop_proche:
                    continue
                date = candidate
                break

            if date is None:  # rare : aucun tirage valide → on saute cette absence
                continue

            feats = compute_weather_features(weather[key], date)
            feats.update(compute_soil_features(weather[key], date))  # sol Open-Meteo (0-7 cm)
            feats["altitude"] = p["altitude"]  # même point → même altitude
            for col in ["pente", "northness", "eastness"]:  # même point → même relief
                feats[col] = p[col]
            for col in SOIL_COLS:  # même point → même sol
                feats[col] = p[col]
            for col in CLC_COLS:  # même point → même occupation du sol
                feats[col] = p[col]
            feats.update(
                {
                    "taxon_key": p["taxon_key"],
                    "species": p["species"],
                    "date": date,
                    "glat": p["glat"],
                    "glon": p["glon"],
                    "lat": p["glat"],  # l'absence est au point de grille (pas de coordonnée exacte)
                    "lon": p["glon"],
                    "target": 0,
                }
            )
            rows.append(feats)
    return pd.DataFrame(rows)


# Top-N de proxies par espèce (réglage final, validé par balayage top-1/3/5).
# 0 = pas d'augmentation (l'espèce préfère ses données propres).
PER_SPECIES_TOPK = {
    "boletus_edulis": 1,
    "macrolepiota_procera": 5,
    "cantharellus_cibarius": 0,
    "morchella": 0,
    "imleria_badia": 3,
    "boletus_reticulatus": 1,
    "craterellus_cornucopioides": 3,
    "hydnum_repandum": 5,
}

# Seeds figés des absences (arbitraires mais stables). Remplacent l'ancien
# `enumerate(sorted(...))` qui dépendait de l'ordre des espèces : ajouter une
# espèce décalait les seeds des autres. Valeurs 0-12 = anciennes positions triées
# (reproduit le dataset actuel) ; 13-15 = nouvelles espèces. Ne jamais changer.
RNG_SEEDS = {
    "boletus_edulis": 0,
    "boletus_pinophilus": 1,
    "boletus_reticulatus": 2,
    "cantharellus_amethysteus": 3,
    "cantharellus_cibarius": 4,
    "craterellus_cornucopioides": 5,
    "craterellus_tubaeformis": 6,
    "imleria_badia": 7,
    "macrolepiota_excoriata": 8,
    "macrolepiota_procera": 9,
    "morchella": 10,
    "xerocomellus_chrysenteron": 11,
    "xerocomus_subtomentosus": 12,
    "hydnum_repandum": 13,
    "boletus_aereus": 14,
    "chlorophyllum_rhacodes": 15,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer-days", type=int, default=ABSENCE_BUFFER_DAYS,
                        help="jours d'écart minimum entre une absence et une observation réelle")
    parser.add_argument("--absences-per-presence", type=int, default=1,
                        help="nombre de pseudo-absences générées par présence")
    parser.add_argument("--proxy-coocc", type=float, default=0.4,
                        help="seuil de co-occurrence (fraction) pour sélectionner un proxy")
    parser.add_argument("--proxy-saison", type=float, default=0.7,
                        help="seuil de recouvrement de saison pour sélectionner un proxy")
    parser.add_argument("--proxy-topk", type=int, default=1000,
                        help="nombre maximum de proxies retenus par espèce (défaut : sans limite)")
    parser.add_argument("--per-species", action="store_true",
                        help="utiliser le top-N par espèce (PER_SPECIES_TOPK) au lieu du top global")
    args = parser.parse_args()

    occ = load_occurrences()
    weather = load_weather()
    soil = load_soil()
    config = load_config()
    # Tous les taxonKeys réellement téléchargés (inclut les espèces filles d'un genre)
    target_keys = {s["gbif_key"] for s in config["species"]}
    for f in RAW_DIR.glob("*.csv"):
        if f.stem == "all_fungi":
            continue
        target_keys |= set(pd.read_csv(f, usecols=["taxonKey"])["taxonKey"].dropna().astype(int))
    fungi = load_all_fungi()
    scores = compute_proxy_scores(occ, fungi, target_keys)
    scores.to_csv(PROCESSED_DIR / "proxies_scores.csv", index=False)
    sel = scores[(scores["co_occ"] >= args.proxy_coocc) & (scores["saison"] >= args.proxy_saison)]
    if args.per_species:
        proxies = {}
        for group, n in PER_SPECIES_TOPK.items():
            gsel = sel[sel["group"] == group]
            proxies[group] = gsel.nlargest(n, "co_occ")["taxonKey"].tolist() if n > 0 else []
    else:
        proxies = {
            group: grp.nlargest(args.proxy_topk, "co_occ")["taxonKey"].tolist()
            for group, grp in sel.groupby("group")
        }
    proxy_keys = set().union(*proxies.values()) if proxies else set()
    cofruiting = load_cofruiting(fungi, target_keys | proxy_keys)
    # Augmentation : les observations des proxies comptent comme des présences
    for group, tks in proxies.items():
        if not tks:
            continue
        sub = fungi[fungi["taxonKey"].isin(tks)].copy()
        sub["taxon_key"] = group
        sub["species"] = "proxy"
        occ = pd.concat([occ, sub[["decimalLatitude", "decimalLongitude", "year", "month", "day", "species", "taxon_key", "date", "glat", "glon"]]], ignore_index=True)
    seasons = compute_group_seasons(occ)
    # RNG par espèce : seed figé (RNG_SEEDS) → absences reproductibles
    # quelle que soit la liste d'espèces (ordre / ajout).
    rngs = {g: np.random.default_rng(RNG_SEEDS[g]) for g in occ["taxon_key"].unique()}

    print(f"{len(occ)} observations (dont proxies), {len(proxy_keys)} taxons proxy, {len(weather)} séries météo")

    presences = build_presences(occ, weather, cofruiting)
    # Fusion du sol par point de grille (glat/glon) — silt exclu (collinéaire)
    soil_for_merge = soil[["lat", "lon"] + SOIL_COLS].rename(columns={"lat": "glat", "lon": "glon"})
    presences = presences.merge(soil_for_merge, on=["glat", "glon"], how="left")
    # Fusion CLC (occupation du sol) par coordonnées exactes
    presences = presences.merge(load_clc(), on=["lat", "lon"], how="left")

    # Dates d'observation par (espèce, point de grille) : servent à éviter les fausses absences
    obs_dates = {
        (taxon, glat, glon): grp["date"].to_numpy(dtype="datetime64[ns]")
        for (taxon, glat, glon), grp in presences.groupby(["taxon_key", "glat", "glon"])
    }
    # Années de présence par espèce : pool pour tirer l'année des absences
    # proportionnellement à l'effort de prospection (target-group background temporel)
    year_pool = {
        taxon: grp["date"].dt.year.to_numpy()
        for taxon, grp in presences.groupby("taxon_key")
    }
    absences = generate_absences(
        presences, weather, seasons, rngs, obs_dates, year_pool, cofruiting,
        buffer_days=args.buffer_days, absences_per_presence=args.absences_per_presence,
    )
    dataset = pd.concat([presences, absences], ignore_index=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "dataset.csv"
    dataset.to_csv(out, index=False)
    print(f"\nDataset écrit : {out}")
    print(f"  présences : {len(presences)} | absences : {len(absences)} | total : {len(dataset)}")
    print(f"  colonnes : {list(dataset.columns)}")
    print(f"  répartition target : {dataset['target'].value_counts().to_dict()}")
    print(f"  NaN restants : {int(dataset.isna().sum().sum())}")
    _close_tiles()


if __name__ == "__main__":
    main()
